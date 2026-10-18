"""Spec for the OpenAI-compatible HTTP API (M6 step 3).

A real app over a real engine thread with the tiny random Qwen3 on CPU, driven through
FastAPI's TestClient. The tokenizer is one character per token (64 characters for the
tiny model's 64 ids), so expected text is easy to compute from M1's greedy tokens.
"""

import asyncio
import json
import string
import time
from collections.abc import Iterator, MutableMapping
from typing import Any

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.test_model import CFG, tiny_model
from tinyserve.api import ChatMessage, Limits, create_app, is_loopback, qwen3_chat_prompt
from tinyserve.engine import Engine
from tinyserve.kv_cache import KVCache
from tinyserve.model import Qwen3, greedy_generate
from tinyserve.sampling import SamplingParams
from tinyserve.serving import AsyncEngine

ALPHABET = string.ascii_lowercase + string.digits + " \n<|>_/" + string.ascii_uppercase[:21]
assert len(ALPHABET) == CFG.vocab_size == 64
NAME = "tiny"


class CharTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ALPHABET.find(c) if c in ALPHABET else ALPHABET.index(" ") for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(ALPHABET[i] for i in ids)


TOK = CharTokenizer()


def greedy_text(prompt: str, n: int, eos: frozenset[int] = frozenset()) -> str:
    return TOK.decode(greedy_generate(tiny_model(), TOK.encode(prompt), n, eos))


def make(
    num_blocks: int = 64, eos: frozenset[int] = frozenset(), model: Qwen3 | None = None, **kw: int
) -> tuple[TestClient, AsyncEngine]:
    cache = KVCache(CFG, num_blocks, 4, dtype=torch.float32)
    ae = AsyncEngine(Engine(model or tiny_model(), cache), max_requests=kw.pop("max_requests", 8))
    ae.start()
    limits = Limits(max_body_bytes=4096, max_prompt_tokens=200, max_tokens_cap=50,
                    default_max_tokens=10)  # fmt: skip
    return TestClient(create_app(ae, TOK, NAME, CFG.vocab_size, eos, limits)), ae


@pytest.fixture
def client() -> Iterator[TestClient]:
    c, ae = make()
    yield c
    ae.shutdown()


def sse(lines: Iterator[str]) -> list[Any]:
    """The JSON payloads of a Server-Sent Events stream; the last one must be [DONE]."""
    events = [line.removeprefix("data: ") for line in lines if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    return [json.loads(e) for e in events[:-1]]


def post_stream(c: TestClient, path: str, body: dict[str, Any]) -> list[Any]:
    with c.stream("POST", path, json={**body, "stream": True}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        return sse(r.iter_lines())


def error_of(r: Any, status: int) -> str:
    assert r.status_code == status, r.text
    err = r.json()["error"]
    assert set(err) == {"message", "type", "param", "code"}
    return str(err["message"])


# --- basics ---------------------------------------------------------------------------


def test_health_and_models(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    data = client.get("/v1/models").json()["data"]
    assert [m["id"] for m in data] == [NAME]


def test_completion_is_greedy_at_temperature_0(client: TestClient) -> None:
    r = client.post("/v1/completions", json={"prompt": "hello", "max_tokens": 12, "temperature": 0})
    body = r.json()
    assert r.status_code == 200 and body["object"] == "text_completion"
    assert body["model"] == NAME and body["id"].startswith("cmpl-")
    (choice,) = body["choices"]
    assert choice["text"] == greedy_text("hello", 12)
    assert choice["finish_reason"] == "length"
    assert body["usage"] == {"prompt_tokens": 5, "completion_tokens": 12, "total_tokens": 17}


def test_prompt_as_token_ids(client: TestClient) -> None:
    ids = TOK.encode("abc")
    r = client.post("/v1/completions", json={"prompt": ids, "max_tokens": 6, "temperature": 0})
    assert r.json()["choices"][0]["text"] == greedy_text("abc", 6)


def test_defaults_are_openais(client: TestClient) -> None:
    # No max_tokens: the server's default (10 here). No temperature: 1.0, i.e. sampling.
    r = client.post("/v1/completions", json={"prompt": "hi", "seed": 5})
    assert r.json()["usage"]["completion_tokens"] == 10
    e = Engine(tiny_model(), KVCache(CFG, 64, 4, dtype=torch.float32))
    offline = e.generate([TOK.encode("hi")], 10, sampling=SamplingParams(1.0, 1.0, 5))[0]
    assert r.json()["choices"][0]["text"] == TOK.decode(offline)


def test_eos_finishes_with_stop() -> None:
    eos = frozenset({TOK.encode(greedy_text("hello", 3)[2])[0]})
    c, ae = make(eos=eos)
    r = c.post("/v1/completions", json={"prompt": "hello", "max_tokens": 20, "temperature": 0})
    ae.shutdown()
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["text"] == greedy_text("hello", 20, eos)


class SkipTokenizer(CharTokenizer):
    """Decodes EOS to nothing, like a real tokenizer skipping special tokens."""

    def __init__(self, eos: int) -> None:
        self.eos = eos

    def decode(self, ids: list[int]) -> str:
        return super().decode([i for i in ids if i != self.eos])


def test_stream_ends_with_a_finish_chunk_even_without_text() -> None:
    ids = greedy_generate(tiny_model(), TOK.encode("hello"), 20)
    k = next(i for i in range(1, 20) if ids[i] not in ids[:i])  # first new token: EOS
    eos_id = ids[k]
    ae = AsyncEngine(Engine(tiny_model(), KVCache(CFG, 64, 4, dtype=torch.float32)))
    ae.start()
    c = TestClient(create_app(ae, SkipTokenizer(eos_id), NAME, CFG.vocab_size, frozenset({eos_id})))
    chunks = post_stream(
        c, "/v1/completions", {"prompt": "hello", "max_tokens": 20, "temperature": 0}
    )
    ae.shutdown()
    assert chunks[-1]["choices"][0] == {
        "index": 0, "text": "", "logprobs": None, "finish_reason": "stop"
    }  # fmt: skip
    assert "".join(ch["choices"][0]["text"] for ch in chunks) == TOK.decode(ids[:k])


class HalfCharTokenizer(CharTokenizer):
    """Every generated token is the first byte of a 2-byte character, so no piece is ever
    complete: all text is held back until the final flush."""

    def decode(self, ids: list[int]) -> str:
        return bytes([0xC3] * len(ids)).decode("utf-8", errors="replace")


def test_held_back_text_is_flushed_at_the_end() -> None:
    ae = AsyncEngine(Engine(tiny_model(), KVCache(CFG, 64, 4, dtype=torch.float32)))
    ae.start()
    c = TestClient(create_app(ae, HalfCharTokenizer(), NAME, CFG.vocab_size))
    r = c.post("/v1/completions", json={"prompt": "abc", "max_tokens": 4, "temperature": 0})
    chunks = post_stream(c, "/v1/completions", {"prompt": "abc", "max_tokens": 4, "temperature": 0})
    ae.shutdown()
    assert r.json()["choices"][0]["text"] == "\ufffd" * 4
    assert [ch["choices"][0]["text"] for ch in chunks] == ["\ufffd" * 4]


def test_streamed_completion_adds_up_to_the_plain_one(client: TestClient) -> None:
    body = {"prompt": "stream me", "max_tokens": 15, "temperature": 0}
    chunks = post_stream(
        client, "/v1/completions", {**body, "stream_options": {"include_usage": True}}
    )
    *texts, usage = chunks
    assert "".join(ch["choices"][0]["text"] for ch in texts) == greedy_text("stream me", 15)
    assert [ch["choices"][0]["finish_reason"] for ch in texts][-1] == "length"
    assert all(ch["choices"][0]["finish_reason"] is None for ch in texts[:-1])
    assert len({ch["id"] for ch in chunks}) == 1
    assert usage["choices"] == [] and usage["usage"]["completion_tokens"] == 15


def test_chat_uses_the_qwen3_template(client: TestClient) -> None:
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    prompt = qwen3_chat_prompt([ChatMessage.model_validate(m) for m in msgs])
    assert prompt == (
        "<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    r = client.post(
        "/v1/chat/completions", json={"messages": msgs, "max_tokens": 8, "temperature": 0}
    )
    body = r.json()
    assert body["object"] == "chat.completion" and body["id"].startswith("chatcmpl-")
    msg = body["choices"][0]["message"]
    assert msg == {"role": "assistant", "content": greedy_text(prompt, 8)}
    assert body["usage"]["prompt_tokens"] == len(prompt)


def test_streamed_chat_has_openai_deltas(client: TestClient) -> None:
    msgs = [{"role": "user", "content": "tell me"}]
    prompt = qwen3_chat_prompt([ChatMessage.model_validate(m) for m in msgs])
    chunks = post_stream(
        client, "/v1/chat/completions", {"messages": msgs, "max_tokens": 9, "temperature": 0}
    )
    assert all(ch["object"] == "chat.completion.chunk" for ch in chunks)
    deltas = [ch["choices"][0]["delta"] for ch in chunks]
    assert deltas[0] == {"role": "assistant", "content": ""}
    assert "".join(d.get("content", "") for d in deltas) == greedy_text(prompt, 9)
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


def test_seeded_sampling_is_reproducible_and_matches_the_engine(client: TestClient) -> None:
    body = {"prompt": "dice", "max_tokens": 12, "temperature": 1.5, "top_p": 0.9, "seed": 11}
    a = client.post("/v1/completions", json=body).json()["choices"][0]["text"]
    b = post_stream(client, "/v1/completions", body)
    assert "".join(ch["choices"][0]["text"] for ch in b) == a
    e = Engine(tiny_model(), KVCache(CFG, 64, 4, dtype=torch.float32))
    offline = e.generate([TOK.encode("dice")], 12, sampling=SamplingParams(1.5, 0.9, 11))[0]
    assert a == TOK.decode(offline)
    assert a != greedy_text("dice", 12)


# --- limits and errors ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "status", "message"),
    [
        ({"prompt": "x" * 201}, 400, "the limit is 200"),
        ({"prompt": "x", "max_tokens": 51}, 400, "the limit is 50"),
        ({"prompt": "x", "max_tokens": 0}, 400, "max_tokens"),
        ({"prompt": "x", "temperature": -1}, 400, "temperature"),
        ({"prompt": "x", "temperature": 2.5}, 400, "temperature"),
        ({"prompt": "x", "top_p": 0}, 400, "top_p"),
        ({"prompt": "x", "n": 2}, 400, "n"),
        ({"prompt": "x", "stop": ["\n"]}, 400, "stop"),
        ({"prompt": "x", "echo": True}, 400, "echo"),
        ({"prompt": "x", "presence_penalty": 0.5}, 400, "presence_penalty"),
        ({"prompt": ""}, 400, "prompt"),
        ({"prompt": [3, 64]}, 400, "token ids"),
        ({"prompt": [-1]}, 400, "token ids"),
        ({}, 400, "prompt"),
        ({"prompt": "x", "model": "gpt-4"}, 404, "not found"),
    ],
)
def test_bad_completion_requests(
    client: TestClient, body: dict[str, Any], status: int, message: str
) -> None:
    assert message in error_of(client.post("/v1/completions", json=body), status)


def test_bad_chat_requests(client: TestClient) -> None:
    assert "messages" in error_of(client.post("/v1/chat/completions", json={"messages": []}), 400)
    bad_role = {"messages": [{"role": "tool", "content": "x"}]}
    assert "role" in error_of(client.post("/v1/chat/completions", json=bad_role), 400)


def test_not_json_and_too_big(client: TestClient) -> None:
    r = client.post(
        "/v1/completions", content=b"{not json", headers={"content-type": "application/json"}
    )
    error_of(r, 400)
    big = {"prompt": "x", "padding": "y" * 5000}
    assert "4096 bytes" in error_of(client.post("/v1/completions", json=big), 413)
    raw = json.dumps(big).encode()
    chunked = client.post("/v1/completions", content=iter([raw[:3000], raw[3000:]]))
    assert "4096 bytes" in error_of(chunked, 413)  # no content-length to check up front


def test_request_that_never_fits_the_cache() -> None:
    c, ae = make(num_blocks=4)  # 16 tokens of KV cache
    r = c.post("/v1/completions", json={"prompt": "abcdefgh", "max_tokens": 20})
    assert "KV cache" in error_of(r, 400)
    assert c.post("/v1/completions", json={"prompt": "ab", "max_tokens": 5}).status_code == 200
    ae.shutdown()


class Exploding(Qwen3):
    def forward_batch(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise RuntimeError("simulated CUDA error")


def test_dead_engine_is_503() -> None:
    m = tiny_model()
    c, ae = make(model=Exploding(m.cfg, m.w))
    error_of(c.post("/v1/completions", json={"prompt": "a"}), 503)
    assert c.get("/health").status_code == 503
    error_of(c.post("/v1/completions", json={"prompt": "a"}), 503)
    assert not ae.is_alive


def test_is_loopback() -> None:
    assert all(is_loopback(h) for h in ("127.0.0.1", "localhost", "::1", "127.0.0.2"))
    assert not any(is_loopback(h) for h in ("0.0.0.0", "192.168.1.5", "::", "example.com"))


# --- client disconnects ---------------------------------------------------------------


class Call:
    """One HTTP request driven straight through the app's ASGI interface, so a test
    controls when the client hangs up (TestClient waits for whole responses)."""

    def __init__(self, app: FastAPI, path: str, body: dict[str, Any]) -> None:
        self.app, self.path, self.raw = app, path, json.dumps(body).encode()
        self.status: int | None = None
        self.chunks: list[bytes] = []
        self.got_chunk = asyncio.Event()
        self.hang_up = asyncio.Event()  # set it to make receive() report a disconnect

    async def run(self) -> None:
        pending: list[MutableMapping[str, Any]] = [
            {"type": "http.request", "body": self.raw, "more_body": False}
        ]

        async def receive() -> MutableMapping[str, Any]:
            if pending:
                return pending.pop()
            await self.hang_up.wait()
            return {"type": "http.disconnect"}

        async def send(message: MutableMapping[str, Any]) -> None:
            if message["type"] == "http.response.start":
                self.status = message["status"]
            elif message.get("body"):
                self.chunks.append(message["body"])
                self.got_chunk.set()

        path = self.path.encode()
        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "POST", "scheme": "http", "path": self.path, "raw_path": path,
            "query_string": b"", "root_path": "", "server": ("test", 80),
            "client": ("test", 1),
            "headers": [(b"content-type", b"application/json"),
                        (b"content-length", str(len(self.raw)).encode())],
        }  # fmt: skip
        await self.app(scope, receive, send)


def served(num_blocks: int = 256, max_requests: int = 8) -> tuple[FastAPI, Engine, AsyncEngine]:
    e = Engine(tiny_model(), KVCache(CFG, num_blocks, 4, dtype=torch.float32))
    ae = AsyncEngine(e, max_requests=max_requests)
    ae.start()
    return create_app(ae, TOK, NAME, CFG.vocab_size, limits=Limits(max_tokens_cap=900)), e, ae


def wait_until_released(e: Engine) -> None:
    deadline = time.monotonic() + 10
    while e.seqs and time.monotonic() < deadline:
        time.sleep(0.005)


def test_too_many_requests_is_429() -> None:
    app, e, ae = served(max_requests=1)

    async def main() -> None:
        first = Call(app, "/v1/completions", {"prompt": "a", "max_tokens": 900, "stream": True})
        task = asyncio.create_task(first.run())
        await first.got_chunk.wait()  # the first request is in flight
        second = Call(app, "/v1/completions", {"prompt": "b", "max_tokens": 2})
        await second.run()
        assert second.status == 429
        assert json.loads(b"".join(second.chunks))["error"]["type"] == "server_overloaded"
        first.hang_up.set()
        await task

    asyncio.run(main())
    wait_until_released(e)
    ae.shutdown()


@pytest.mark.parametrize("stream", [True, False])
def test_disconnect_aborts_the_request(stream: bool) -> None:
    app, e, ae = served()

    async def main() -> None:
        call = Call(app, "/v1/completions", {"prompt": "abc", "max_tokens": 900, "stream": stream})
        if stream:
            task = asyncio.create_task(call.run())
            while len(call.chunks) < 2:  # hang up after two events
                await asyncio.sleep(0.001)
            call.hang_up.set()
            await task
        else:
            call.hang_up.set()  # gone right after sending the request
            await call.run()
            assert call.status == 499

    asyncio.run(main())
    wait_until_released(e)
    ae.shutdown()
    assert not e.seqs
    assert e.num_steps < 500  # aborted, not run to its 900 tokens
    assert e.scheduler.allocator.num_free == 256
