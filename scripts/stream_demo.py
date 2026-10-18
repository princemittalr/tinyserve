"""Stream several requests at once through the async serving loop (M6 step 2 demo).

Usage: python scripts/stream_demo.py [--device cpu] [--max-new-tokens 12]

Three clients send their prompts at the same moment and print each piece of text as it
arrives; a fourth hangs up after 3 tokens, and its KV-cache blocks are freed at once.
The engine runs them all in one engine thread, batched step by step.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tinyserve.config import ModelConfig
from tinyserve.detokenize import IncrementalDetokenizer
from tinyserve.engine import Engine
from tinyserve.kv_cache import KVCache
from tinyserve.model import Qwen3
from tinyserve.sampling import SamplingParams
from tinyserve.serving import AsyncEngine
from tinyserve.weights import load_weights

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Three facts about the T4 GPU:",
    "Once upon a time",  # this client hangs up early
]


def eos_ids(model_dir: Path) -> frozenset[int]:
    path = model_dir / "generation_config.json"
    if not path.exists():
        return frozenset()
    eos = json.loads(path.read_text(encoding="utf-8")).get("eos_token_id", [])
    return frozenset(eos if isinstance(eos, list) else [eos])


async def client(
    i: int, ae: AsyncEngine, tok: Tokenizer, prompt: str, n: int, eos: frozenset[int], t0: float
) -> None:
    detok = IncrementalDetokenizer(tok.decode)
    params = SamplingParams(temperature=0.7, top_p=0.9, seed=i) if i == 1 else None
    stream = ae.generate(tok.encode(prompt).ids, n, params, eos)
    async for out in stream:
        text = detok.push(out.token_ids)
        if out.finish_reason is not None:
            text += detok.flush()
        print(f"{time.perf_counter() - t0:6.2f}s [{i}] {text!r}")
        if i == 3 and out.num_output_tokens == 3:
            await stream.aclose()  # hang up: the engine aborts the request
            print(f"{time.perf_counter() - t0:6.2f}s [{i}] hung up")
            return
        if out.finish_reason is not None:
            print(f"{time.perf_counter() - t0:6.2f}s [{i}] done: {out.finish_reason}")


async def run(ae: AsyncEngine, tok: Tokenizer, n: int, eos: frozenset[int]) -> None:
    t0 = time.perf_counter()
    await asyncio.gather(*(client(i, ae, tok, p, n, eos, t0) for i, p in enumerate(PROMPTS)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/Qwen3-0.6B"))
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = ModelConfig.from_file(args.model_dir / "config.json")
    dtype = getattr(torch, args.dtype)
    model = Qwen3(cfg, load_weights(args.model_dir, cfg, dtype, args.device))
    tok = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    cache = KVCache(cfg, 64, 16, dtype, args.device)
    engine = Engine(model, cache)
    ae = AsyncEngine(engine)
    ae.start()
    asyncio.run(run(ae, tok, args.max_new_tokens, eos_ids(args.model_dir)))
    ae.shutdown()
    free = engine.scheduler.allocator.num_free
    print(f"{engine.num_steps} engine steps for {len(PROMPTS)} requests; ", end="")
    print(f"KV blocks free afterwards: {free}/{cache.num_blocks}")


if __name__ == "__main__":
    main()
