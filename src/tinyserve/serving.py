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

With ``trace=True`` (M7) it also times each request across the two threads
(``RequestTrace``) and measures how late the event loop wakes up (``loop_lag_ms``): the
pieces of a time to first token that the engine alone cannot see.
"""

import asyncio
import contextlib
import queue
import threading
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import torch

from tinyserve.engine import Engine
from tinyserve.sampling import SamplingParams, TokenLogprobs
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
    logprobs: list[TokenLogprobs] | None = None  # one per token_ids, if the request asked


@dataclass
class RequestTrace:
    """When one request crossed each boundary, in ``time.perf_counter()`` seconds.

    ``perf_counter`` is a system-wide monotonic clock (CLOCK_MONOTONIC on Linux, QPC on
    Windows), so a client process on the same machine can compare its own times with these.

        received     the HTTP handler started (set by api.py; None outside HTTP)
        submitted    the event loop put the "add" command on the engine's queue
        added        the engine thread took it and added the request (between steps)
        first_token  the engine thread handed over the first output, after its step
        first_seen   the event loop picked that output up from the request's queue
    """

    tag: str = ""
    received: float | None = None
    submitted: float | None = None
    added: float | None = None
    added_at_step: int | None = None  # engine.num_steps when it was added
    first_token: float | None = None
    first_token_step: int | None = None  # engine.num_steps after the step that made it
    first_seen: float | None = None


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
    trace: RequestTrace | None = None


class AsyncEngine:
    def __init__(self, engine: Engine, max_requests: int = 256, trace: bool = False) -> None:
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
        self.trace = trace
        self.traces: list[RequestTrace] = []  # finished requests, oldest first (if trace)
        self.loop_lag_ms: list[float] = []  # how late each lag probe woke up (if trace)
        self._lag_task: asyncio.Task[None] | None = None

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

    def take_trace(self) -> dict[str, list[Any]]:
        """Event loop side, tracing only: hand over and forget what was recorded so far:
        finished requests, event-loop lag samples and the engine's steps."""
        steps = self.engine.step_records
        n = len(steps) if steps is not None else 0  # the engine thread may append more
        out: dict[str, list[Any]] = {
            "requests": self.traces,
            "loop_lag_ms": self.loop_lag_ms,
            "steps": steps[:n] if steps is not None else [],
        }
        self.traces, self.loop_lag_ms = [], []
        if steps is not None:
            del steps[:n]
        return out

    async def _probe_lag(self, period: float = 0.005) -> None:
        """Sleep ``period`` again and again and record how much later than asked the loop
        woke up: the time something else (another callback, or the engine thread holding
        the GIL) kept it from running."""
        while True:
            start = time.perf_counter()
            await asyncio.sleep(period)
            self.loop_lag_ms.append((time.perf_counter() - start - period) * 1e3)

    async def generate(
        self,
        prompt: list[int],
        max_new_tokens: int,
        sampling: SamplingParams | None = None,
        eos_ids: frozenset[int] = frozenset(),
        trace: RequestTrace | None = None,
    ) -> AsyncGenerator[StreamOutput, None]:
        """Stream one request's tokens, one ``StreamOutput`` per engine step.

        Raises ``QueueFullError`` when ``max_requests`` are already in flight,
        ``ValueError`` if the engine rejects the request (e.g. it can never fit), and
        ``EngineDeadError`` if the engine thread is not running. If the caller stops
        early (``aclose``, or its task is cancelled because the client disconnected),
        the request is aborted and its KV-cache blocks are freed. With tracing on, the
        request's times are kept in ``trace`` (a new one if not given) once it ends.
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
        if self.trace:
            if self._lag_task is None or self._lag_task.done():  # one probe per loop
                self._lag_task = req.loop.create_task(self._probe_lag())
            req.trace = trace or RequestTrace()
            req.trace.submitted = time.perf_counter()
        self.num_requests += 1
        finished = False
        try:
            self._commands.put(("add", req))
            while True:
                item = await req.out.get()
                if req.trace is not None and req.trace.first_seen is None:
                    req.trace.first_seen = time.perf_counter()
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
            if req.trace is not None:
                self.traces.append(req.trace)

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
            if req.trace is not None:
                req.trace.added = time.perf_counter()
                req.trace.added_at_step = self.engine.num_steps
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
            lps = self.engine.logprobs.get(seq_id)
            lps = None if lps is None else lps[req.sent :]
            req.sent = len(seq.output)
            reason = None
            if finished:
                reason = "stop" if seq.output[-1] in seq.eos_ids else "length"
                req.done = True
                self.engine.release(seq_id)
                del self._live[seq_id]
            if req.trace is not None and req.trace.first_token is None:
                req.trace.first_token = time.perf_counter()
                req.trace.first_token_step = self.engine.num_steps
            self._send(req, StreamOutput(new, reason, len(seq.prompt), len(seq.output), lps))

    @staticmethod
    def _send(req: _Request, item: StreamOutput | BaseException) -> None:
        # RuntimeError: that client's event loop is closed, so nobody is listening.
        with contextlib.suppress(RuntimeError):
            req.loop.call_soon_threadsafe(req.out.put_nowait, item)
