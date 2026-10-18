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
    args = ["--model-dir", str(model_dir), "--device", "cpu", "--kv-gib", "0.01"]
    with running_server(args, free_port(), timeout=120) as url:
        body = {"prompt": prompt, "max_tokens": 12, "temperature": 0}

        async def stream() -> StreamResult:
            async with httpx.AsyncClient(timeout=60) as client:
                return await stream_completion(client, url, body)

        streamed = asyncio.run(stream())
        plain = httpx.post(f"{url}/v1/completions", json=body, timeout=60).json()
        refused = httpx.post(f"{url}/v1/completions", json={**body, "max_tokens": 10**6})
        models = httpx.get(f"{url}/v1/models").json()

    want = offline_text(model_dir, prompt, 12)
    assert streamed.text == plain["choices"][0]["text"] == want
    assert streamed.completion_tokens == 12 and streamed.finish_reason == "length"
    assert streamed.ttft > 0 and streamed.tpot is not None and streamed.tpot > 0
    assert refused.status_code == 400
    assert models["data"][0]["id"] == model_dir.name


def test_server_refuses_a_public_host(model_dir: Path) -> None:
    cmd = [sys.executable, "-m", "tinyserve.server", "--model-dir", str(model_dir),
           "--host", "0.0.0.0"]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    assert "--allow-remote" in r.stderr
