"""End to end over a real socket (M6 step 4): ``python -m tinyserve.server`` on a tiny
random checkpoint with a real byte-level BPE tokenizer, driven by ``tinyserve.client``,
the same code the T4 benchmark and parity scripts use."""

import asyncio
import json
import socket
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
import torch
from safetensors.torch import save_file

from tests.test_config import TINY
from tests.test_detokenize import byte_level_bpe
from tinyserve.client import StreamResult, running_server, stream_completion
from tinyserve.config import ModelConfig
from tinyserve.engine import Engine
from tinyserve.kv_cache import KVCache
from tinyserve.model import Qwen3
from tinyserve.profiling import ttft_segments
from tinyserve.weights import expected_shapes, load_weights


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("tiny_server")
    tok = byte_level_bpe()
    raw = {**TINY, "vocab_size": tok.get_vocab_size()}
    cfg = ModelConfig.from_dict(raw)
    g = torch.Generator().manual_seed(0)
    w = {
        n: (torch.randn(s, generator=g) * 0.5).contiguous() for n, s in expected_shapes(cfg).items()
    }
    save_file(w, str(d / "model.safetensors"))
    (d / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    tok.save(str(d / "tokenizer.json"))
    return d


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def offline_text(model_dir: Path, prompt: list[int], n: int) -> str:
    cfg = ModelConfig.from_file(model_dir / "config.json")
    model = Qwen3(cfg, load_weights(model_dir, cfg, dtype=torch.float32))
    out = Engine(model, KVCache(cfg, 64, 16, dtype=torch.float32)).generate([prompt], n)[0]
    return str(byte_level_bpe().decode(out))


def test_server_over_http(model_dir: Path) -> None:
    prompt = byte_level_bpe().encode("the capital of France is").ids
    args = ["--model-dir", str(model_dir), "--device", "cpu", "--kv-gib", "0.01",
            "--trace-requests"]  # fmt: skip
    with running_server(args, free_port(), timeout=120) as url:
        body = {"prompt": prompt, "max_tokens": 12, "temperature": 0}

        async def stream() -> StreamResult:
            async with httpx.AsyncClient(timeout=60) as client:
                return await stream_completion(client, url, body)

        streamed = asyncio.run(stream())
        plain = httpx.post(f"{url}/v1/completions", json=body, timeout=60).json()
        refused = httpx.post(f"{url}/v1/completions", json={**body, "max_tokens": 10**6})
        models = httpx.get(f"{url}/v1/models").json()
        trace = httpx.get(f"{url}/debug/trace").json()

    want = offline_text(model_dir, prompt, 12)
    assert streamed.text == plain["choices"][0]["text"] == want
    assert streamed.completion_tokens == 12 and streamed.finish_reason == "length"
    assert streamed.ttft > 0 and streamed.tpot is not None and streamed.tpot > 0
    assert refused.status_code == 400
    assert models["data"][0]["id"] == model_dir.name
    # The server's trace of the streamed request splits the client's TTFT exactly: the
    # client and the server process read the same perf_counter clock.
    by_id = {r["tag"]: r for r in trace["requests"]}
    assert streamed.id.startswith("cmpl-") and plain["id"] in by_id
    segs = ttft_segments(streamed.sent, streamed.chunk_times[0], by_id[streamed.id])
    assert all(v >= 0 for v in segs.values()), segs
    assert sum(segs.values()) == pytest.approx(streamed.ttft * 1e3)
    assert trace["steps"] and trace["loop_lag_ms"]


def test_server_with_the_m8_configuration(model_dir: Path) -> None:
    # Grouped kernel, CUDA graphs (eager on CPU) and batched prefill: same greedy text.
    pytest.importorskip("triton", reason="Triton is not installed (no Windows wheels)")
    prompts = [byte_level_bpe().encode(t).ids for t in ("the capital of", "one two three")]
    args = ["--model-dir", str(model_dir), "--device", "cpu", "--kv-gib", "0.01",
            "--attention", "triton_grouped", "--cuda-graphs", "--prefill-attention", "batched",
            "--max-num-seqs", "4", "--trace-requests"]  # fmt: skip
    with running_server(args, free_port(), timeout=120) as url:

        async def both() -> list[StreamResult]:
            async with httpx.AsyncClient(timeout=120) as client:
                bodies = [{"prompt": p, "max_tokens": 8, "temperature": 0} for p in prompts]
                return await asyncio.gather(*(stream_completion(client, url, b) for b in bodies))

        got = asyncio.run(both())
        trace = httpx.get(f"{url}/debug/trace").json()
    assert [r.text for r in got] == [offline_text(model_dir, p, 8) for p in prompts]
    assert any(s["graph"] for s in trace["steps"]), "no decode step took the graph path"


def test_server_refuses_cuda_graphs_without_a_kernel(model_dir: Path) -> None:
    cmd = [sys.executable, "-m", "tinyserve.server", "--model-dir", str(model_dir),
           "--cuda-graphs"]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    assert "--cuda-graphs needs" in r.stderr


def test_server_refuses_a_public_host(model_dir: Path) -> None:
    cmd = [sys.executable, "-m", "tinyserve.server", "--model-dir", str(model_dir),
           "--host", "0.0.0.0"]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    assert "--allow-remote" in r.stderr
