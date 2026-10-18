"""The serving loop: many async clients, one engine thread (M6).

The engine is synchronous: ``step()`` runs one forward pass and blocks until it is done.
A web server is asynchronous: one event loop juggles many open connections. They meet
here:

    client tasks (event loop)                    engine thread
    ─────────────────────────                    ─────────────
    generate() ── ("add", req) ──► commands ──►  add_request
                                                 step()  step()  step() ...
    async for out in ... ◄── req.queue ◄──────── new tokens of each request, per step
    client gone ── ("abort", req) ──► commands ► abort: its blocks are freed at once

Only the engine thread touches the engine, so the scheduler, allocator and cache need no
locks. The thread drains the command queue between steps, which is exactly where
continuous batching lets requests join and leave, and it sleeps on the queue when there
is no work instead of spinning. Outputs go back with ``loop.call_soon_threadsafe``,
the one thread-safe way into an asyncio queue.

This layer only moves token ids; text (detokenize.py) and HTTP (api.py) sit on top.
"""

import asyncio
import contextlib
import queue
import threading
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

import torch

from tinyserve.engine import Engine
from tinyserve.sampling import SamplingParams
from tinyserve.sequence import SeqStatus


class QueueFullError(RuntimeError):
    """Too many requests in flight; the API answers 429 and the client retries later."""


class EngineDeadError(RuntimeError):
    """The engine thread crashed or was shut down; no request can be served."""


@dataclass(frozen=True)
class StreamOutput:
    token_ids: list[int]  # tokens generated since the previous output of this request
    finish_reason: str | None  # None while running, then "stop" (EOS) or "length"
    num_prompt_tokens: int
    num_output_tokens: int  # generated so far, including token_ids


@dataclass(eq=False)
class _Request:
    prompt: list[int]
    max_new_tokens: int
    eos_ids: frozenset[int]
    sampling: SamplingParams
    loop: asyncio.AbstractEventLoop
    out: "asyncio.Queue[StreamOutput | BaseException]" = field(default_factory=asyncio.Queue)
    seq_id: int | None = None  # set by the engine thread once added
    sent: int = 0  # output tokens already delivered
    done: bool = False  # engine thread: finished, aborted or rejected


class AsyncEngine:
    def __init__(self, engine: Engine, max_requests: int = 256) -> None:
        if max_requests <= 0:
            raise ValueError(f"max_requests must be positive, got {max_requests}")
        self.engine = engine
        self.max_requests = max_requests  # in flight: waiting + running
        self.num_requests = 0  # changed only on the event loop
        self._commands: queue.Queue[tuple[str, _Request] | None] = queue.Queue()
        self._live: dict[int, _Request] = {}  # engine thread only: seq_id -> request
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="tinyserve-engine", daemon=True)
        self._started = False

    # --- event loop side ----------------------------------------------------------------

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def shutdown(self) -> None:
        """Stop the engine thread after the current step; open requests get an error."""
        if self._started and self._thread.is_alive():
            self._commands.put(None)
            self._thread.join()

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    async def generate(
        self,
        prompt: list[int],
        max_new_tokens: int,
        sampling: SamplingParams | None = None,
        eos_ids: frozenset[int] = frozenset(),
    ) -> AsyncGenerator[StreamOutput, None]:
        """Stream one request's tokens, one ``StreamOutput`` per engine step.

        Raises ``QueueFullError`` when ``max_requests`` are already in flight,
        ``ValueError`` if the engine rejects the request (e.g. it can never fit), and
        ``EngineDeadError`` if the engine thread is not running. If the caller stops
        early (``aclose``, or its task is cancelled because the client disconnected),
        the request is aborted and its KV-cache blocks are freed.
        """
        if not self.is_alive:
            raise EngineDeadError("engine thread is not running") from self._error
        if self.num_requests >= self.max_requests:
            raise QueueFullError(
                f"{self.num_requests} requests in flight (max {self.max_requests})"
            )
        req = _Request(
            prompt,
            max_new_tokens,
            eos_ids,
            sampling or SamplingParams(),
            asyncio.get_running_loop(),
        )
        self.num_requests += 1
        finished = False
        try:
            self._commands.put(("add", req))
            while True:
                item = await req.out.get()
                if isinstance(item, BaseException):
                    finished = True  # the engine thread already dropped it
                    raise item
                if item.finish_reason is not None:
                    finished = True
                yield item
                if finished:
                    return
        finally:
            self.num_requests -= 1
            if not finished:
                self._commands.put(("abort", req))

    # --- engine thread ------------------------------------------------------------------

    @torch.inference_mode()  # thread-local: this thread needs its own
    def _run(self) -> None:
        try:
            while True:
                # Sleep on the queue when idle; otherwise take whatever arrived.
                cmds = [self._commands.get()] if not self.engine.has_work() else []
                while True:
                    try:
                        cmds.append(self._commands.get_nowait())
                    except queue.Empty:
                        break
                for cmd in cmds:
                    if cmd is None:
                        raise EngineDeadError("engine was shut down")
                    self._handle(*cmd)
                if self.engine.has_work():
                    self.engine.step()
                    self._deliver()
        except BaseException as e:  # every open request must hear about it
            self._error = e
            dead = e if isinstance(e, EngineDeadError) else EngineDeadError(f"engine failed: {e!r}")
            for req in self._live.values():
                self._send(req, dead)
            self._live.clear()
            while True:  # requests that arrived but were never added
                try:
                    cmd = self._commands.get_nowait()
                except queue.Empty:
                    break
                if cmd is not None and cmd[0] == "add":
                    self._send(cmd[1], dead)

    def _handle(self, kind: str, req: _Request) -> None:
        if kind == "add":
            try:
                req.seq_id = self.engine.add_request(
                    req.prompt, req.max_new_tokens, req.eos_ids, req.sampling
                )
            except ValueError as e:  # can never fit, bad prompt: this request only
                req.done = True
                self._send(req, e)
                return
            self._live[req.seq_id] = req
        elif kind == "abort" and not req.done and req.seq_id is not None:
            req.done = True
            self.engine.abort(req.seq_id)
            self.engine.release(req.seq_id)
            del self._live[req.seq_id]

    def _deliver(self) -> None:
        """After a step: send each live request the tokens it has not seen yet."""
        for seq_id, req in list(self._live.items()):
            seq = self.engine.seqs[seq_id]
            new = seq.output[req.sent :]
            finished = seq.status is SeqStatus.FINISHED
            if not new and not finished:
                continue  # waiting, or preempted this step
            req.sent = len(seq.output)
            reason = None
            if finished:
                reason = "stop" if seq.output[-1] in seq.eos_ids else "length"
                req.done = True
                self.engine.release(seq_id)
                del self._live[seq_id]
            self._send(req, StreamOutput(new, reason, len(seq.prompt), len(seq.output)))

    @staticmethod
    def _send(req: _Request, item: StreamOutput | BaseException) -> None:
        # RuntimeError: that client's event loop is closed, so nobody is listening.
        with contextlib.suppress(RuntimeError):
            req.loop.call_soon_threadsafe(req.out.put_nowait, item)
