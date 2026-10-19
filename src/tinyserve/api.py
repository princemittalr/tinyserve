"""OpenAI-compatible HTTP API over the serving loop (M6).

    GET  /health                ok, or 503 once the engine thread has died
    GET  /v1/models             the one served model
    POST /v1/completions        prompt (text or token ids) -> text
    POST /v1/chat/completions   messages -> assistant message (Qwen3 chat template)

With ``"stream": true`` the answer is Server-Sent Events: one ``data: {json}`` line per
engine step that produced text, then ``data: [DONE]``, the format the ``openai`` client
library reads. Without it, the whole answer comes back as one JSON object.

Limits, each with the error a client gets back (OpenAI's error shape):
    request body over ``max_body_bytes``            413
    prompt over ``max_prompt_tokens``               400
    max_tokens over ``max_tokens_cap``              400
    n over ``max_n``, more than 4 stop strings     400
    a field out of range, or a feature not built   400 (echo, penalties)
    ``max_requests`` already in flight              429 (AsyncEngine)
    engine thread dead                              503

``stop`` (M9): up to 4 strings; the answer ends before the first one that appears, which
is not returned (``StopMatcher``), with finish_reason "stop", and the engine request is
aborted there. ``n`` (M9): n engine requests for the same prompt, one per choice, run
side by side; with a ``seed``, choice i samples with seed + i. Usage counts the prompt
once and every choice's tokens.

``logprobs`` (M9): for completions ``logprobs: k`` (0-5), for chat ``logprobs: true`` with
``top_logprobs: k`` (0-20), in OpenAI's two formats. Each generated token comes with its
log-probability under the model (before temperature and top-p) and the k most likely
tokens. A token's text is ``decode([id])`` alone, so a token holding part of a UTF-8
character shows as U+FFFD. With a stop string, the tokens are every token generated, up
to and including the one that completed the stop string.

A client that disconnects is aborted: for a stream, Starlette cancels the response
task; for a plain request, the handler checks for a disconnect after every step. Either
way ``AsyncEngine.generate`` closes and the request's KV-cache blocks are freed.
There is no authentication, which is why ``scripts/serve.py`` binds to 127.0.0.1.

    GET  /debug/trace           (M7, only when the AsyncEngine traces) per-request times,
                                event-loop lag and engine steps since the last call
"""

import asyncio
import dataclasses
import itertools
import json
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol, TypeVar

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tinyserve.detokenize import IncrementalDetokenizer, StopMatcher
from tinyserve.sampling import SamplingParams, TokenLogprobs
from tinyserve.serving import (
    AsyncEngine,
    EngineDeadError,
    QueueFullError,
    RequestTrace,
    StreamOutput,
)


class TextTokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


@dataclass(frozen=True)
class Limits:
    max_body_bytes: int = 1 << 20  # 1 MiB
    max_prompt_tokens: int = 4096
    max_tokens_cap: int = 1024  # the most a request may ask for
    default_max_tokens: int = 256  # when the request does not say
    max_n: int = 8  # choices per request (M9)


# --- request bodies -------------------------------------------------------------------


class StreamOptions(BaseModel):
    include_usage: bool = False


class _Common(BaseModel):
    # Unknown fields (user, metadata, ...) are ignored, as OpenAI-compatible servers do;
    # the ones that would change the answer are declared below and refused if used.
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    max_tokens: int | None = Field(None, ge=1)
    temperature: float = Field(1.0, ge=0, le=2)  # OpenAI's default is 1, not greedy
    top_p: float = Field(1.0, gt=0, le=1)
    seed: int | None = Field(None, ge=0, lt=2**63)
    stream: bool = False
    stream_options: StreamOptions | None = None
    n: int = Field(1, ge=1)
    stop: str | list[str] | None = None
    # Not implemented: accepted only at their "off" values.
    echo: Literal[False] = False
    presence_penalty: Literal[0] = 0
    frequency_penalty: Literal[0] = 0


class CompletionRequest(_Common):
    prompt: str | list[int] = Field(min_length=1)
    logprobs: int | None = Field(None, ge=0, le=5)  # the legacy API's limit


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(_Common):
    messages: list[ChatMessage] = Field(min_length=1)
    logprobs: bool = False
    top_logprobs: int | None = Field(None, ge=0, le=20)


def qwen3_chat_prompt(messages: list[ChatMessage]) -> str:
    """Qwen3's ChatML template with thinking off (HF ``enable_thinking=False``).

    Each turn is ``<|im_start|>{role}\\n{content}<|im_end|>\\n``; the prompt ends by
    opening the assistant's turn with an empty think block, so the model answers
    directly instead of writing out its reasoning first.
    """
    turns = "".join(f"<|im_start|>{m.role}\n{m.content}<|im_end|>\n" for m in messages)
    return turns + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


# --- errors ---------------------------------------------------------------------------


class APIError(Exception):
    def __init__(self, status: int, message: str, kind: str = "invalid_request_error") -> None:
        super().__init__(message)
        self.status, self.message, self.kind = status, message, kind

    def response(self) -> JSONResponse:
        body = {"error": {"message": self.message, "type": self.kind, "param": None, "code": None}}
        return JSONResponse(body, status_code=self.status)


def _validation_message(e: ValidationError) -> str:
    err = e.errors()[0]
    where = ".".join(str(p) for p in err["loc"]) or "body"
    return f"{where}: {err['msg']}"


_Req = TypeVar("_Req", bound=_Common)


# --- the app --------------------------------------------------------------------------


def create_app(
    engine: AsyncEngine,
    tokenizer: TextTokenizer,
    model_name: str,
    vocab_size: int,
    eos_ids: frozenset[int] = frozenset(),
    limits: Limits | None = None,
    chat_template: Callable[[list[ChatMessage]], str] = qwen3_chat_prompt,
) -> FastAPI:
    limits = limits or Limits()
    app = FastAPI(title="TinyServe", docs_url=None, redoc_url=None)

    @app.exception_handler(APIError)
    async def _api_error(request: Request, e: APIError) -> JSONResponse:
        return e.response()

    async def parse(request: Request, kind: type[_Req]) -> _Req:
        size = request.headers.get("content-length")
        if size is not None and size.isdigit() and int(size) > limits.max_body_bytes:
            raise APIError(413, f"request body over {limits.max_body_bytes} bytes")
        body = await request.body()
        if len(body) > limits.max_body_bytes:  # a chunked body has no content-length
            raise APIError(413, f"request body over {limits.max_body_bytes} bytes")
        try:
            req = kind.model_validate_json(body)
        except ValidationError as e:
            raise APIError(400, _validation_message(e)) from None
        if req.model is not None and req.model != model_name:
            raise APIError(404, f"model {req.model!r} not found; this server runs {model_name!r}")
        return req

    def stops(req: _Common) -> list[str]:
        out = [req.stop] if isinstance(req.stop, str) else list(req.stop or [])
        if len(out) > 4 or any(not s for s in out):
            raise APIError(400, "stop: at most 4 strings, none of them empty")
        return out

    def num_logprobs(req: _Common) -> int | None:
        """How many alternatives per token, or None for no logprobs at all."""
        if isinstance(req, ChatRequest):
            if req.top_logprobs is not None and not req.logprobs:
                raise APIError(400, "top_logprobs needs logprobs: true")
            return (req.top_logprobs or 0) if req.logprobs else None
        return req.logprobs if isinstance(req, CompletionRequest) else None

    def prepare(req: _Common, prompt_ids: list[int]) -> tuple[int, SamplingParams]:
        n = len(prompt_ids)
        if n > limits.max_prompt_tokens:
            raise APIError(400, f"prompt has {n} tokens, the limit is {limits.max_prompt_tokens}")
        if any(not 0 <= t < vocab_size for t in prompt_ids):
            # An id outside the embedding table would crash the forward pass for everyone.
            raise APIError(400, f"prompt token ids must be in [0, {vocab_size})")
        max_tokens = req.max_tokens or limits.default_max_tokens
        if max_tokens > limits.max_tokens_cap:
            raise APIError(400, f"max_tokens is {max_tokens}, the limit is {limits.max_tokens_cap}")
        if req.n > limits.max_n:
            raise APIError(400, f"n is {req.n}, the limit is {limits.max_n}")
        return max_tokens, SamplingParams(req.temperature, req.top_p, req.seed, num_logprobs(req))

    async def start(
        prompt_ids: list[int], max_tokens: int, params: SamplingParams, trace: RequestTrace
    ) -> AsyncGenerator[StreamOutput, None]:
        """Begin generating and wait for the first output, so that errors become HTTP
        status codes before any response (or stream) has started."""
        try:
            stream = engine.generate(prompt_ids, max_tokens, params, eos_ids, trace)
            first = await anext(stream)
        except QueueFullError as e:
            raise APIError(429, str(e), "server_overloaded") from None
        except EngineDeadError as e:
            raise APIError(503, str(e), "server_error") from None
        except ValueError as e:  # the engine refused it (it could never fit)
            raise APIError(400, str(e)) from None

        async def chain() -> AsyncGenerator[StreamOutput, None]:
            try:
                yield first
                async for out in stream:
                    yield out
            finally:
                await stream.aclose()  # abort if we stop early

        return chain()

    async def texts(
        index: int, outs: AsyncGenerator[StreamOutput, None], stop: list[str]
    ) -> AsyncGenerator[tuple[int, str, str | None, StreamOutput], None]:
        """Choice ``index``'s outputs as (index, new text, finish reason, output).

        Text is flushed at the end; with stop strings it ends before the first one, and
        leaving early closes ``outs``, which aborts the engine request."""
        detok, matcher = IncrementalDetokenizer(tokenizer.decode), StopMatcher(stop)
        try:
            async for out in outs:
                text = detok.push(out.token_ids)
                if out.finish_reason is not None:
                    text += detok.flush()
                text = matcher.push(text)
                if matcher.stopped:
                    yield index, text, "stop", out
                    return
                if out.finish_reason is not None:
                    text += matcher.flush()
                yield index, text, out.finish_reason, out
        finally:
            await outs.aclose()

    async def merged(
        gens: list[AsyncGenerator[tuple[int, str, str | None, StreamOutput], None]],
    ) -> AsyncGenerator[tuple[int, str, str | None, StreamOutput], None]:
        """The items of several choices' generators, in the order they arrive (n > 1)."""
        if len(gens) == 1:
            try:
                async for item in gens[0]:
                    yield item
            finally:
                await gens[0].aclose()
            return
        q: asyncio.Queue[Any] = asyncio.Queue()
        end = object()

        async def pump(g: AsyncGenerator[Any, None]) -> None:
            try:
                async for item in g:
                    await q.put(item)
                await q.put(end)
            except Exception as e:  # e.g. EngineDeadError: hand it to the consumer
                await q.put(e)

        tasks = [asyncio.create_task(pump(g)) for g in gens]
        try:
            left = len(gens)
            while left:
                item = await q.get()
                if item is end:
                    left -= 1
                elif isinstance(item, Exception):
                    raise item
                else:
                    yield item
        finally:  # done, failed or abandoned: stop every choice, which aborts its request
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for g in gens:
                await g.aclose()

    def entries(out: StreamOutput) -> list[tuple[str, float, list[tuple[str, float]]]]:
        """An output's tokens as (text, logprob, [(text, logprob) of the top k])."""
        lps: list[TokenLogprobs] = out.logprobs or []
        piece = tokenizer.decode
        return [
            (piece([t]), lp.logprob, [(piece([i]), v) for i, v in lp.top])
            for t, lp in zip(out.token_ids, lps, strict=True)
        ]

    def completion_logprobs(es: list[Any], offset: int) -> dict[str, Any]:
        """The legacy format; ``offset`` is where the first token starts in the answer."""
        starts = itertools.accumulate((len(t) for t, _, _ in es), initial=offset)
        return {
            "tokens": [t for t, _, _ in es],
            "token_logprobs": [v for _, v, _ in es],
            "top_logprobs": [dict(top) for _, _, top in es],
            "text_offset": list(starts)[:-1],
        }

    def chat_logprobs(es: list[Any]) -> dict[str, Any]:
        def one(t: str, v: float) -> dict[str, Any]:
            return {"token": t, "logprob": v, "bytes": list(t.encode())}

        return {"content": [{**one(t, v), "top_logprobs": [one(*x) for x in top]}
                            for t, v, top in es]}  # fmt: skip

    def usage(lasts: dict[int, StreamOutput]) -> dict[str, int]:
        prompt = next(iter(lasts.values())).num_prompt_tokens  # counted once, as OpenAI does
        completion = sum(out.num_output_tokens for out in lasts.values())
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

    async def respond(
        request: Request,
        req: _Common,
        prompt_ids: list[int],
        chat: bool,
        received: float,
    ) -> Response:
        max_tokens, params = prepare(req, prompt_ids)
        stop = stops(req)
        rid = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex}"
        gens = []
        try:
            for i in range(req.n):  # one engine request per choice
                seed = None if params.seed is None else params.seed + i
                p = dataclasses.replace(params, seed=seed)
                tag = rid if req.n == 1 else f"{rid}-{i}"
                outs = await start(prompt_ids, max_tokens, p, RequestTrace(tag, received))
                gens.append(texts(i, outs, stop))
        except APIError:
            for g in gens:
                await g.aclose()
            raise
        choices = merged(gens)
        head: dict[str, Any] = {
            "id": rid,
            "object": "chat.completion.chunk" if chat else "text_completion",
            "created": int(time.time()),
            "model": model_name,
        }

        want_lps = params.logprobs is not None
        offsets = dict.fromkeys(range(req.n), 0)  # completions: text so far, per choice

        def choice(
            i: int, text: str | None, reason: str | None, es: list[Any], first: bool = False
        ) -> dict[str, Any]:
            lps = None
            if want_lps and not first:
                lps = chat_logprobs(es) if chat else completion_logprobs(es, offsets[i])
                offsets[i] += sum(len(t) for t, _, _ in es)
            if not chat:
                return {"index": i, "text": text or "", "logprobs": lps, "finish_reason": reason}
            delta: dict[str, str] = {"role": "assistant"} if first else {}
            if text is not None:
                delta["content"] = text
            return {"index": i, "delta": delta, "logprobs": lps, "finish_reason": reason}

        if req.stream:
            include_usage = req.stream_options is not None and req.stream_options.include_usage

            async def events() -> AsyncGenerator[str, None]:
                def sse(obj: dict[str, Any]) -> str:
                    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

                lasts: dict[int, StreamOutput] = {}
                try:
                    if chat:  # OpenAI opens a chat stream with the role, per choice
                        for i in range(req.n):
                            c = choice(i, "", None, [], first=True)
                            yield sse({**head, "choices": [c]})
                    async for i, text, reason, out in choices:
                        lasts[i] = out
                        es = entries(out) if want_lps else []
                        if text or reason is not None or es:
                            c = choice(i, text or None, reason, es)  # chat's last delta: {}
                            yield sse({**head, "choices": [c]})
                except EngineDeadError as e:  # mid-stream: the status code is already sent
                    yield sse({"error": {"message": str(e), "type": "server_error"}})
                    return
                finally:
                    await choices.aclose()  # also on a disconnect: abort, free the blocks
                if include_usage and lasts:
                    yield sse({**head, "choices": [], "usage": usage(lasts)})
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        pieces: dict[int, list[str]] = {i: [] for i in range(req.n)}
        es_all: dict[int, list[Any]] = {i: [] for i in range(req.n)}
        reasons: dict[int, str] = {}
        outs_last: dict[int, StreamOutput] = {}
        try:
            async for i, text, reason, out in choices:
                pieces[i].append(text)
                outs_last[i] = out
                if want_lps:
                    es_all[i] += entries(out)
                if reason is not None:
                    reasons[i] = reason
                if await request.is_disconnected():
                    break  # nobody to answer: leaving the loop aborts every choice
        except EngineDeadError as e:
            raise APIError(503, str(e), "server_error") from None
        finally:
            await choices.aclose()
        if len(reasons) < req.n:
            return Response(status_code=499)  # client closed the request (nginx's code)
        if chat:
            chs = [
                {
                    "index": i,
                    "message": {"role": "assistant", "content": "".join(pieces[i])},
                    "logprobs": chat_logprobs(es_all[i]) if want_lps else None,
                    "finish_reason": reasons[i],
                }
                for i in range(req.n)
            ]
            body = {**head, "object": "chat.completion", "choices": chs}
        else:
            chs = [choice(i, "".join(pieces[i]), reasons[i], es_all[i]) for i in range(req.n)]
            body = {**head, "choices": chs}
        return JSONResponse({**body, "usage": usage(outs_last)})

    @app.get("/health")
    async def health() -> Response:
        if not engine.is_alive:
            return APIError(503, "engine thread is not running", "server_error").response()
        return JSONResponse({"status": "ok"})

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        card = {"id": model_name, "object": "model", "created": 0, "owned_by": "tinyserve"}
        return {"object": "list", "data": [card]}

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        received = time.perf_counter()
        req = await parse(request, CompletionRequest)
        prompt = req.prompt
        ids = tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        return await respond(request, req, ids, chat=False, received=received)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        received = time.perf_counter()
        req = await parse(request, ChatRequest)
        ids = tokenizer.encode(chat_template(req.messages))
        return await respond(request, req, ids, chat=True, received=received)

    @app.get("/debug/trace")
    async def debug_trace() -> Response:
        if not engine.trace:
            raise APIError(404, "tracing is off; start the server with --trace-requests")
        t = engine.take_trace()
        return JSONResponse({
            "requests": [asdict(r) for r in t["requests"]],
            "loop_lag_ms": t["loop_lag_ms"],
            "steps": [{**asdict(s), "total_ms": s.total_ms} for s in t["steps"]],
        })  # fmt: skip

    return app


def is_loopback(host: str) -> bool:
    """Only these hosts are reachable from this machine alone."""
    return host in {"127.0.0.1", "localhost", "::1"} or host.startswith("127.")
