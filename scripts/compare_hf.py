"""Greedy tokens from TinyServe vs Hugging Face generate() (M1 acceptance; M2, M3 re-checks).

Both run the same checkpoint in float32 on CPU, from the same token ids. For each prompt
we compare the generated token lists exactly. On a mismatch we report where the two
diverge and HF's top-2 logit margin there: a tiny margin means a near-tie that float
rounding can flip, a large margin means a real bug.

Usage:
    python scripts/compare_hf.py --limit 5        # quick smoke run, paged KV cache (M2)
    python scripts/compare_hf.py                  # all prompts, paged KV cache
    python scripts/compare_hf.py --engine m1      # M1's no-cache path (slow)
    python scripts/compare_hf.py --engine batched # M3: all prompts at once through Engine
"""

import argparse
import datetime as dt
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

from tinyserve.block_allocator import BlockAllocator
from tinyserve.config import ModelConfig
from tinyserve.engine import Engine
from tinyserve.kv_cache import KVCache
from tinyserve.model import Qwen3, greedy_generate, greedy_generate_paged
from tinyserve.weights import load_weights


def read_prompts(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.replace("\\n", "\n") for ln in lines if ln.strip()]


def eos_ids(model_dir: Path) -> list[int]:
    path = model_dir / "generation_config.json"
    if not path.exists():
        return []
    eos = json.loads(path.read_text(encoding="utf-8")).get("eos_token_id", [])
    return eos if isinstance(eos, list) else [eos]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/Qwen3-0.6B"))
    parser.add_argument("--prompts", type=Path, default=Path("benchmarks/prompts/m1_greedy.txt"))
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--engine", choices=["paged", "m1", "batched"], default="paged")
    parser.add_argument("--max-num-seqs", type=int, default=32, help="batched engine only")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--out", type=Path, default=None, help="default: results/<m2|m1>_...")
    args = parser.parse_args()
    if args.out is None:
        milestone = {"m1": "m1", "paged": "m2", "batched": "m3"}[args.engine]
        args.out = Path(f"results/{milestone}_hf_compare.json")

    torch.set_num_threads(max(1, torch.get_num_threads()))
    cfg = ModelConfig.from_file(args.model_dir / "config.json")
    ours = Qwen3(cfg, load_weights(args.model_dir, cfg, dtype=torch.float32))
    hf: Any = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32)
    hf.eval()
    tok = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    eos = eos_ids(args.model_dir)

    prompts = read_prompts(args.prompts)[: args.limit]
    # One cache, reused by every prompt: big enough for the longest prompt + its new tokens.
    longest = max(len(tok.encode(p).ids) for p in prompts) + args.max_new_tokens
    num_blocks = -(-longest // args.block_size)
    cache = KVCache(cfg, num_blocks, args.block_size, dtype=torch.float32)
    allocator = BlockAllocator(num_blocks)

    ours_out: list[list[int]] = []
    batched_secs = None
    if args.engine == "batched":
        # Every prompt in one Engine run; the cache gets room for max_num_seqs of them.
        engine = Engine(
            ours,
            KVCache(cfg, num_blocks * args.max_num_seqs, args.block_size, dtype=torch.float32),
            max_num_seqs=args.max_num_seqs,
        )
        t0 = time.perf_counter()
        ours_out = engine.generate(
            [tok.encode(p).ids for p in prompts], args.max_new_tokens, frozenset(eos)
        )
        batched_secs = round(time.perf_counter() - t0, 1)
        preempted = engine.scheduler.num_preemptions
        print(f"batched engine: {len(prompts)} prompts in {batched_secs} s, "
              f"{engine.num_steps} steps, {preempted} preemptions")  # fmt: skip

    rows, max_prefill_diff, start = [], 0.0, time.perf_counter()
    for n, text in enumerate(prompts, 1):
        ids = tok.encode(text).ids
        x = torch.tensor(ids)
        prefill_diff = (ours.forward(x) - hf(x[None]).logits[0].float()).abs().max().item()
        max_prefill_diff = max(max_prefill_diff, prefill_diff)

        full = hf.generate(
            x[None],
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=eos or None,
            pad_token_id=eos[0] if eos else None,
        )
        expected = full[0, len(ids) :].tolist()
        if args.engine == "batched":
            got = ours_out[n - 1]
        elif args.engine == "paged":
            got = greedy_generate_paged(
                ours, ids, args.max_new_tokens, cache, allocator, frozenset(eos)
            )
        else:
            got = greedy_generate(ours, ids, args.max_new_tokens, frozenset(eos))

        row: dict[str, object] = {
            "prompt": text,
            "match": got == expected,
            "n_tokens": len(expected),
        }
        if got != expected:
            k = next(
                (i for i, (a, b) in enumerate(zip(got, expected, strict=False)) if a != b), None
            )
            k = min(len(got), len(expected)) if k is None else k
            top2 = hf(torch.tensor(ids + expected[:k])[None]).logits[0, -1].float().topk(2).values
            row |= {"diverge_at": k, "hf_top2_margin": (top2[0] - top2[1]).item(),
                    "ours": got, "hf": expected}  # fmt: skip
        rows.append(row)
        print(f"[{n}/{len(prompts)}] {'OK  ' if row['match'] else 'DIFF'} {text[:50]!r}")

    mismatches = [r for r in rows if not r["match"]]
    summary = {
        "date": dt.date.today().isoformat(),
        "hardware": f"{platform.system()} {platform.machine()} CPU, torch {torch.__version__}",
        "dtype": "float32",
        "engine": args.engine,
        "prompts": len(rows),
        "max_new_tokens": args.max_new_tokens,
        "tokens_compared": sum(int(str(r["n_tokens"])) for r in rows),
        "mismatches": len(mismatches),
        "max_prefill_logit_abs_diff": max_prefill_diff,
        "seconds": round(time.perf_counter() - start, 1),
    }
    if batched_secs is not None:
        summary |= {"max_num_seqs": args.max_num_seqs, "batched_engine_seconds": batched_secs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for r in mismatches:
        print(f"MISMATCH {r['prompt']!r}: diverges at token {r['diverge_at']}, "
              f"HF top-2 margin {r['hf_top2_margin']:.4f}")  # fmt: skip


if __name__ == "__main__":
    main()
