"""Generation with TinyServe's own forward pass.

Usage: python scripts/generate.py --prompt "The capital of France is" --max-new-tokens 20
Add --no-cache to use M1's path, which re-runs the whole sequence every step.
Add --temperature 0.8 --top-p 0.95 --num-samples 4 [--seed 1] to sample (M6): the
engine runs the samples as one batch, each with its own seed (seed, seed + 1, ...).
Run scripts/inspect_weights.py once first to download the model.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tinyserve.block_allocator import BlockAllocator
from tinyserve.config import ModelConfig
from tinyserve.engine import Engine
from tinyserve.kv_cache import KVCache
from tinyserve.model import Qwen3, greedy_generate, greedy_generate_paged
from tinyserve.sampling import SamplingParams
from tinyserve.weights import load_weights


def eos_ids(model_dir: Path) -> frozenset[int]:
    path = model_dir / "generation_config.json"
    if not path.exists():
        return frozenset()
    eos = json.loads(path.read_text(encoding="utf-8")).get("eos_token_id", [])
    return frozenset(eos if isinstance(eos, list) else [eos])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/Qwen3-0.6B"))
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-cache", action="store_true", help="M1 path: no KV cache")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None, help="per-sample seeds start here")
    parser.add_argument("--num-samples", type=int, default=1)
    args = parser.parse_args()

    cfg = ModelConfig.from_file(args.model_dir / "config.json")
    weights = load_weights(args.model_dir, cfg, getattr(torch, args.dtype), args.device)
    model = Qwen3(cfg, weights)
    tok = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))

    prompt_ids = tok.encode(args.prompt).ids
    eos = eos_ids(args.model_dir)
    if args.temperature > 0 or args.num_samples > 1:
        sample_with_engine(args, model, cfg, tok, prompt_ids, eos)
        return
    if args.no_cache:
        mode = "no KV cache"
        start = time.perf_counter()
        new = greedy_generate(model, prompt_ids, args.max_new_tokens, eos)
    else:
        # Enough blocks for this one sequence: ceil(tokens / block_size).
        num_blocks = -(-(len(prompt_ids) + args.max_new_tokens) // args.block_size)
        cache = KVCache(cfg, num_blocks, args.block_size, getattr(torch, args.dtype), args.device)
        mode = f"paged KV cache, {num_blocks} blocks x {args.block_size} tokens"
        start = time.perf_counter()
        new = greedy_generate_paged(
            model, prompt_ids, args.max_new_tokens, cache, BlockAllocator(num_blocks), eos
        )
    secs = time.perf_counter() - start

    print(f"prompt ids: {prompt_ids}")
    print(f"new ids:    {new}")
    print(f"text: {args.prompt!r} -> {tok.decode(new)!r}")
    print(f"{len(new)} tokens in {secs:.1f}s on {args.device} ({args.dtype}), {mode}")


def sample_with_engine(
    args: argparse.Namespace,
    model: Qwen3,
    cfg: ModelConfig,
    tok: Tokenizer,
    prompt_ids: list[int],
    eos: frozenset[int],
) -> None:
    n = args.num_samples
    num_blocks = n * -(-(len(prompt_ids) + args.max_new_tokens) // args.block_size)
    cache = KVCache(cfg, num_blocks, args.block_size, getattr(torch, args.dtype), args.device)
    engine = Engine(model, cache, max_num_seqs=n)
    ids = []
    for i in range(n):
        seed = None if args.seed is None else args.seed + i
        params = SamplingParams(args.temperature, args.top_p, seed)
        ids.append(engine.add_request(prompt_ids, args.max_new_tokens, eos, params))
    start = time.perf_counter()
    while engine.has_work():
        engine.step()
    secs = time.perf_counter() - start
    for i in ids:
        print(f"[{i}] {args.prompt!r} -> {tok.decode(engine.seqs[i].output)!r}")
    total = sum(len(engine.seqs[i].output) for i in ids)
    print(
        f"{n} samples, {total} tokens in {secs:.1f}s on {args.device} ({args.dtype}), "
        f"temperature {args.temperature}, top_p {args.top_p}, seed {args.seed}, "
        f"{engine.num_steps} engine steps"
    )


if __name__ == "__main__":
    main()
