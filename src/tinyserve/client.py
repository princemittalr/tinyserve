"""A small client for the HTTP API: start a server process, stream completions (M6).

Used by the benchmark and parity scripts and by the end-to-end test, so they all talk to
the server the way any outside client would: over a real socket, as SSE.
"""

import json
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx


@contextmanager
def running_server(args: list[str], port: int = 8000, timeout: float = 600) -> Iterator[str]:
    """Run ``python -m tinyserve.server --port PORT *args`` until the block ends.

    Yields the base URL once ``/health`` answers 200. Raises if the process exits or does
    not come up within ``timeout`` seconds (loading the weights can take a while).
    """
    url = f"http://127.0.0.1:{port}"
    cmd = [sys.executable, "-m", "tinyserve.server", "--port", str(port), *args]
    proc = subprocess.Popen(cmd)
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                if httpx.get(f"{url}/health").status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if proc.poll() is not None:
                raise RuntimeError(f"server exited with code {proc.returncode}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"server did not answer /health within {timeout:.0f} s")
            time.sleep(0.2)
        yield url
    finally:
        proc.terminate()
        proc.wait()


@dataclass
class StreamResult:
    text: str
    finish_reason: str | None
    completion_tokens: int
    sent: float  # perf_counter() when the request was sent
    chunk_times: list[float] = field(default_factory=list)  # when each choice chunk arrived

    @property
    def ttft(self) -> float:
        """Seconds to the first chunk: queueing + prefill + one HTTP round trip.

        A chunk normally carries one token. The server sends no chunk for a token whose
        text is held back (half a character) or empty (a special token), so then the
        first chunk comes one step later.
        """
        return self.chunk_times[0] - self.sent

    @property
    def tpot(self) -> float | None:
        """Mean seconds per output token after the first; None with a single token."""
        if self.completion_tokens < 2:
            return None
        return (self.chunk_times[-1] - self.chunk_times[0]) / (self.completion_tokens - 1)


async def stream_completion(
    client: httpx.AsyncClient, url: str, body: dict[str, Any]
) -> StreamResult:
    """POST a streaming ``/v1/completions`` request and time each chunk as it arrives."""
    payload = {**body, "stream": True, "stream_options": {"include_usage": True}}
    result = StreamResult("", None, 0, time.perf_counter())
    pieces: list[str] = []
    async with client.stream("POST", f"{url}/v1/completions", json=payload) as r:
        if r.status_code != 200:
            await r.aread()
            raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if "error" in event:
                raise RuntimeError(event["error"]["message"])
            if event.get("usage"):
                result.completion_tokens = event["usage"]["completion_tokens"]
            for choice in event["choices"]:  # empty for the usage chunk
                pieces.append(choice["text"])
                result.chunk_times.append(time.perf_counter())
                result.finish_reason = choice["finish_reason"] or result.finish_reason
    result.text = "".join(pieces)
    return result
