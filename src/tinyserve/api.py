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
    a field out of range, or a feature not built   400 (n > 1, stop, logprobs, echo)
    ``max_requests`` already in flight              429 (AsyncEngine)
    engine thread dead                              503

A client that disconnects is aborted: for a stream, Starlette cancels the response
task; for a plain request, the handler checks for a disconnect after every step. Either
way ``AsyncEngine.generate`` closes and the request's KV-cache blocks are freed.
There is no authentication, which is why ``scripts/serve.py`` binds to 127.0.0.1.
"""

import json
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tinyserve.detokenize import IncrementalDetokenizer
from tinyserve.sampling import SamplingParams
from tinyserve.serving import AsyncEngine, EngineDeadError, QueueFullError, StreamOutput


class TextTokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


@dataclass(frozen=True)
class Limits:
    max_body_bytes: int = 1 << 20  # 1 MiB
    max_prompt_tokens: int = 4096
    max_tokens_cap: int = 1024  # the most a request may ask for
    default_max_tokens: int = 256  # when the request does not say


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
    # Not implemented: accepted only at their "off" values.
    n: Literal[1] = 1
    stop: None = None
    logprobs: Literal[False, None] = None
    echo: Literal[False] = False
    presence_penalty: Literal[0] = 0
    frequency_penalty: Literal[0] = 0


class CompletionRequest(_Common):
    prompt: str | list[int] = Field(min_length=1)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(_Common):
    messages: list[ChatMessage] = Field(min_length=1)


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
        return max_tokens, SamplingParams(req.temperature, req.top_p, req.seed)

    async def start(
        prompt_ids: list[int], max_tokens: int, params: SamplingParams
    ) -> AsyncGenerator[StreamOutput, None]:
        """Begin generating and wait for the first output, so that errors become HTTP
        status codes before any response (or stream) has started."""
        try:
            stream = engine.generate(prompt_ids, max_tokens, params, eos_ids)
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
        outs: AsyncIterator[StreamOutput],
    ) -> AsyncGenerator[tuple[str, StreamOutput], None]:
        """Each output with the new text it completes (flushed at the end)."""
        detok = IncrementalDetokenizer(tokenizer.decode)
        async for out in outs:
            text = detok.push(out.token_ids)
            if out.finish_reason is not None:
                text += detok.flush()
            yield text, out

    def usage(out: StreamOutput) -> dict[str, int]:
        return {
            "prompt_tokens": out.num_prompt_tokens,
            "completion_tokens": out.num_output_tokens,
            "total_tokens": out.num_prompt_tokens + out.num_output_tokens,
        }

    async def respond(
        request: Request,
        req: _Common,
        prompt_ids: list[int],
        chat: bool,
    ) -> Response:
        max_tokens, params = prepare(req, prompt_ids)
        outs = await start(prompt_ids, max_tokens, params)
        rid = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex}"
        head: dict[str, Any] = {
            "id": rid,
            "object": "chat.completion.chunk" if chat else "text_completion",
            "created": int(time.time()),
            "model": model_name,
        }

        def choice(text: str | None, reason: str | None, first: bool = False) -> dict[str, Any]:
            if not chat:
                return {"index": 0, "text": text or "", "logprobs": None, "finish_reason": reason}
            delta: dict[str, str] = {"role": "assistant"} if first else {}
            if text is not None:
                delta["content"] = text
            return {"index": 0, "delta": delta, "logprobs": None, "finish_reason": reason}

        if req.stream:
            include_usage = req.stream_options is not None and req.stream_options.include_usage

            async def events() -> AsyncGenerator[str, None]:
                def sse(obj: dict[str, Any]) -> str:
                    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

                last: StreamOutput | None = None
                try:
                    if chat:  # OpenAI opens a chat stream with the role
                        yield sse({**head, "choices": [choice("", None, first=True)]})
                    async for text, out in texts(outs):
                        last = out
                        if text or out.finish_reason is not None:
                            c = choice(text or None, out.finish_reason)  # chat's last delta: {}
                            yield sse({**head, "choices": [c]})
                except EngineDeadError as e:  # mid-stream: the status code is already sent
                    yield sse({"error": {"message": str(e), "type": "server_error"}})
                    return
                finally:
                    await outs.aclose()  # also on a disconnect: abort, free the blocks
                if include_usage and last is not None:
                    yield sse({**head, "choices": [], "usage": usage(last)})
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        pieces: list[str] = []
        last = None
        try:
            async for text, out in texts(outs):
                pieces.append(text)
                last = out
                if await request.is_disconnected():
                    break  # nobody to answer: leaving the loop aborts the request
        except EngineDeadError as e:
            raise APIError(503, str(e), "server_error") from None
        finally:
            await outs.aclose()
        if last is None or last.finish_reason is None:
            return Response(status_code=499)  # client closed the request (nginx's code)
        text = "".join(pieces)
        if chat:
            ch = {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "logprobs": None,
                "finish_reason": last.finish_reason,
            }
            body = {**head, "object": "chat.completion", "choices": [ch], "usage": usage(last)}
        else:
            ch = choice(text, last.finish_reason)
            body = {**head, "choices": [ch], "usage": usage(last)}
        return JSONResponse(body)

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
        req = await parse(request, CompletionRequest)
        prompt = req.prompt
        ids = tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        return await respond(request, req, ids, chat=False)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        req = await parse(request, ChatRequest)
        ids = tokenizer.encode(chat_template(req.messages))
        return await respond(request, req, ids, chat=True)

    return app


def is_loopback(host: str) -> bool:
    """Only these hosts are reachable from this machine alone."""
    return host in {"127.0.0.1", "localhost", "::1"} or host.startswith("127.")
