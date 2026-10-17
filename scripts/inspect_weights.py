"""M1 step 1 demo: download Qwen3-0.6B, load it strictly, and print a summary.

Usage: python scripts/inspect_weights.py [--model-dir checkpoints/Qwen3-0.6B]
"""

import argparse
from collections import Counter
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from tinyserve.config import ModelConfig
from tinyserve.weights import load_weights

REPO_ID = "Qwen/Qwen3-0.6B"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/Qwen3-0.6B"))
    args = parser.parse_args()

    # Downloads only config + weights (~1.5 GB) on first run; later runs reuse the files.
    snapshot_download(REPO_ID, local_dir=args.model_dir, allow_patterns=["*.json", "*.safetensors"])

    cfg = ModelConfig.from_file(args.model_dir / "config.json")
    print(cfg)
    w = load_weights(args.model_dir, cfg, dtype=torch.float16)

    total = sum(t.numel() for t in w.values())
    embed = w["model.embed_tokens.weight"].numel()
    print(f"tensors: {len(w)}")
    print(f"parameters: {total:,} total, {total - embed:,} excluding the embedding")
    print(f"fp16 size: {total * 2 / 2**30:.2f} GiB")
    kinds = Counter(name.split(".")[-2] for name in w)
    print("tensors per kind:", dict(sorted(kinds.items())))
    biggest = max(w, key=lambda n: w[n].abs().max().item())
    print(f"largest |weight|: {w[biggest].abs().max().item():.1f} in {biggest}")


if __name__ == "__main__":
    main()
