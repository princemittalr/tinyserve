"""M10 step 3: how fast the T4 runs a decode step's matrix multiplies, one by one and merged.

A decode step at batch M multiplies an [M, 1,024] activation by every weight matrix once:
per layer q, k, v, o, gate, up and down (7 GEMMs), then the output head. With M <= 64
that is memory-bound: the time should be the weight bytes / bandwidth. M9 run 2's
profile measured all of a C=32 step's GEMMs at 7.4 ms for 1.19 GB, ~160 GB/s of the
T4's 320. Part of that may be the small ones: k and v are 2 MB each, too small to keep
40 SMs streaming. vLLM merges q/k/v into one [4,096 x 1,024] matrix and gate/up into
one [6,144 x 1,024]: 4 GEMMs per layer instead of 7.

For each M this times, inside a CUDA graph (as the engine runs decode, so no launch
overhead), every layer's 7 GEMMs, each kind on its own, the merged version (qkv, o,
gate_up, down) and the output head, on the real weights (28 different layers, so the
T4's 4 MB L2 cannot hold one layer's weights for the next). It reports ms per step and
GB/s. If the merged GEMMs are clearly faster, merging the weights is the next step.

M11: ``--blas cublas,cublaslt`` repeats everything with each of PyTorch's BLAS back ends
(``torch.backends.cuda.preferred_blas_library``). cuBLASLt picks its kernels by its own
heuristics, so on these skinny shapes it may reach a bandwidth cuBLAS does not.

Usage (T4): python scripts/bench_gemm.py [--blas cublas,cublaslt]
"""

import argparse
import datetime as dt
import json
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F

from tinyserve.config import ModelConfig
from tinyserve.profiling import loop_ms
from tinyserve.weights import load_weights

KINDS = ("q", "k", "v", "o", "gate", "up", "down")
NAMES = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj",
         "o": "self_attn.o_proj", "gate": "mlp.gate_proj", "up": "mlp.up_proj",
         "down": "mlp.down_proj"}  # fmt: skip


def graphed(fn: Callable[[], object]) -> Callable[[], None]:
    """``fn`` recorded once as a CUDA graph; returns the replay. A warm-up run on a side
    stream first lets cuBLAS pick and load its kernels (as cuda_graphs.py does)."""
    side = torch.cuda.Stream()  # type: ignore[no-untyped-call]
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph.replay


def time_gemms(ws: list[torch.Tensor], inputs: dict[int, torch.Tensor], iters: int
               ) -> dict[str, float]:  # fmt: skip
    """``x @ w.T`` for every ``w`` in ``ws``, x the input of w's width, replayed as one
    graph: ms per replay, the weight bytes read and the bandwidth that is."""
    replay = graphed(lambda: [F.linear(inputs[x.shape[1]], x) for x in ws])
    ms = loop_ms(replay, torch.device("cuda"), iters)
    gb = sum(x.numel() * x.element_size() for x in ws) / 1e9
    return {"ms": ms, "gb": gb, "gbps": gb / ms * 1e3}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/Qwen3-0.6B"))
    parser.add_argument("--batches", default="1,2,4,8,32,64")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--blas", default="cublas", help="comma-separated: cublas, cublaslt")
    parser.add_argument("--out", type=Path, default=Path("results/m10_gemm_cuda.json"))
    args = parser.parse_args()
    dev, dtype = torch.device("cuda"), torch.float16
    cfg = ModelConfig.from_file(args.model_dir / "config.json")
    w = load_weights(args.model_dir, cfg, dtype, dev)
    layers = range(cfg.num_layers)
    mats = {k: [w[f"model.layers.{i}.{NAMES[k]}.weight"] for i in layers] for k in KINDS}
    merged = {
        "qkv": [torch.cat([mats["q"][i], mats["k"][i], mats["v"][i]]) for i in layers],
        "gate_up": [torch.cat([mats["gate"][i], mats["up"][i]]) for i in layers],
        "o": mats["o"],
        "down": mats["down"],
    }
    head = w["model.embed_tokens.weight"] if cfg.tie_word_embeddings else w["lm_head.weight"]

    rows = []
    for blas in args.blas.split(","):
        torch.backends.cuda.preferred_blas_library(blas)
        print(f"BLAS: {blas}")
        for m in (int(b) for b in args.batches.split(",")):
            inputs = {n: torch.randn(m, n, device=dev, dtype=dtype)
                      for n in {cfg.hidden_size, cfg.q_size, cfg.intermediate_size}}  # fmt: skip

            def timed(ws: list[torch.Tensor], inputs: dict[int, torch.Tensor] = inputs
                      ) -> dict[str, float]:  # fmt: skip
                return time_gemms(ws, inputs, args.iters)

            each = {k: timed(mats[k]) for k in KINDS}
            all7 = timed([x for i in layers for x in (mats[k][i] for k in KINDS)])
            all4 = timed([x for i in layers for x in (merged[k][i] for k in merged)])
            lm = timed([head])
            rows.append({"blas": blas, "m": m, "each": each, "separate": all7, "merged": all4,
                         "head": lm})  # fmt: skip
            print(f"M={m:<3} 7 GEMMs/layer {all7['ms']:.2f} ms ({all7['gbps']:.0f} GB/s) | merged "
                  f"4/layer {all4['ms']:.2f} ms ({all4['gbps']:.0f} GB/s), "
                  f"{1 - all4['ms'] / all7['ms']:.0%} less | head {lm['ms']:.2f} ms "
                  f"({lm['gbps']:.0f} GB/s)", flush=True)  # fmt: skip
            print("      each kind over 28 layers, ms (GB/s): " + ", ".join(
                f"{k} {e['ms']:.2f} ({e['gbps']:.0f})" for k, e in each.items()))  # fmt: skip
    report = {"date": dt.date.today().isoformat(), "hardware": torch.cuda.get_device_name(),
              "torch": torch.__version__, "rows": rows}  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
