# TinyServe on a T4: where the time goes (M7)

Every number below comes from a logged run on a Kaggle Tesla T4 (2026-10-18), fp16,
Qwen3-0.6B. Raw data is in `results/`; each table names its file. Anything not measured is
marked *inferred* or *projected*.

## 1. The roofline: the fastest a decode step could run

One decode step for B sequences reads every weight once (1.19 GB in fp16; 596,049,920
parameters) and every cached key and value once (112 KiB per token across the 28 layers).
The linear layers do 2 FLOPs per weight per sequence, so their arithmetic intensity is about
B FLOP/byte in fp16. Attention does 4 FLOPs per cached element (16 query heads share 8 k/v
heads), 2 FLOP/byte at any B. The T4's ridge point is 65 TFLOPS / 320 GB/s = 203 FLOP/byte,
so every benchmarked step (B <= 64, intensity 1-30) is memory-bound: bytes set the floor,
not FLOPs (`src/tinyserve/roofline.py`).

| C (sequences) | Context read per step | Floor (bytes / 320 GB/s) | Ceiling C / floor | TinyServe forward | Forward vs floor |
|---|---|---|---|---|---|
| 1 | 192 tokens | 3.8 ms | 264 tok/s | 33.0 ms | 11% |
| 8 | 1,536 | 4.3 ms | 1,871 tok/s | 33.3 ms | 13% |
| 32 | 6,143 (mean) | 5.9 ms | 5,399 tok/s | 34.2 ms | 17% |
| 64 | 12,266 (mean) | 8.1 ms | 7,880 tok/s | 36.5 ms | 22% |

The floor assumes the spec bandwidth; real kernels reach perhaps 70-85% of it, so the
ceiling is optimistic. The point is the shape: the floor grows with C (more k/v), but
TinyServe's forward is nearly flat at ~33 ms. Something other than memory traffic sets it.
Source: `results/m7_engine_profile_cuda.json`.

## 2. Where TinyServe's decode time goes

From the engine step trace and torch.profiler at C=32 (`scripts/profile_engine.py`):

- A step is 37.8 ms: forward 34.2 ms, `build_batch` 3.2 ms, schedule + sample + update 0.4 ms.
- Inside the forward, the GPU runs kernels for 16.7 ms: GEMMs 6.8 ms (176 GB/s, 55% of
  spec), the Triton decode kernel 5.6 ms, ~4.3 ms of small norm, RoPE and copy kernels.
  There are 1,851 kernel launches per step.
- *Inferred:* the other ~17 ms of the forward is the GPU waiting for Python to launch the next
  kernel: 1,851 launches in 34.2 ms is about 18 us of wall time each. The step is
  launch-bound. CUDA graphs remove most of this wait.
- The decode kernel reaches 89 GB/s counting each k/v element once, 179 GB/s as issued: it
  loads each k/v twice, once per query head of a GQA pair.
- `build_batch` builds a slot list for every cached token: 3.2 ms with 192-token contexts,
  13-15 ms with ~1,100-token ones (`results/m7_prefix_steps_cuda.json`).

The M4 Triton kernel already removed the first bottleneck (per-sequence attention, 94% of a
C=64 step in M3, 740 -> 40 ms). What is left is Python overhead around small kernels.

## 3. Where time to first token goes over HTTP

M6 measured a TTFT p99 of 5.6 s at 32 streams over HTTP vs 0.6 s for the engine alone, and
guessed GIL contention. Every request was traced through the server
(`scripts/profile_api.py`, `results/m7_api_profile_cuda.json`; the step 3 check with
chunk counts is in PROGRESS.md, Benchmark results):

- **Not the GIL.** The event loop's lag stays at p99 1.9-3.7 ms, max 11 ms. A 0.5 ms GIL
  switch interval instead of 5 ms changes nothing.
- **The p99 is held-back text, not a slow first token.** The slowest requests got exactly one
  SSE chunk, with empty text, at the very end. On random-token prompts with EOS ignored, their
  whole 128-token output decodes to nothing visible, and the API sends a chunk only when there
  is text. Measured to the first *token*, TTFT p99 is 731 ms at C=32 (to the first text:
  5,802 ms; 3 of 128 requests) and 1,426 ms at C=64 (7,214 ms; 5 of 256).
- **The median gap is waiting for the step in progress.** Over HTTP a request arrives in the
  middle of an engine step and waits for it to end: p50 107-161 ms at C=32, 224-285 ms at
  C=64. The offline benchmark adds requests exactly between steps, so it never waits.
- Serving costs the engine thread 2-3 ms per step, almost all in `build_batch` (inferred: the
  event loop's Python work takes GIL time from the most Python-heavy phase). Throughput over
  HTTP is 5.6-6.8% below the engine alone at C = 32 / 64.

With the prefix cache on (M5), the C=32 median TTFT got worse (346 -> 704 ms). The trace shows
why: cache hits arrive in waves of up to 25 prefills in one step (645-685 ms), and 77% of
such a step is prefill attention over the 1,024 cached tokens, run one sequence at a time in
PyTorch (~19 ms per hit).

## 4. Comparison with HF `generate()` and vLLM

Same workload for all: the same seeded 128-token random prompts, 128 greedy output tokens,
EOS ignored, closed loop at concurrency C (HF: static batches of C). Output tok/s:

| C | HF `generate()` | TinyServe engine | TinyServe over HTTP | vLLM 0.32 eager | vLLM 0.32 (CUDA graphs) |
|---|---|---|---|---|---|
| 1 | 25.2 | 30.7 (1.22x HF) | 30.0 | not run | 142.4 (5.65x HF) |
| 8 | 193.3 | 231.1 (1.20x) | 221.7 | not run | 682.0 (3.53x) |
| 32 | 415.6 | 766.4 (1.84x) | 723.8 | 814.3 | 1,103.2 (2.65x) |
| 64 | 451.3 | 1,205.9 (2.67x) | 1,124.0 | not run | 1,384.5 (3.07x) |

TinyServe reaches 22% / 34% / 69% / 87% of vLLM's throughput at C = 1 / 8 / 32 / 64.

What the comparison does and does not show:

- vLLM in eager mode (every kernel launched from Python, like TinyServe) is 6% faster than
  TinyServe at C=32 (814 vs 766 tok/s), within session-to-session noise. CUDA graphs make vLLM 1.35x faster than its own
  eager mode at C=32. This matches the profile: the gap to a mature engine is mostly launch
  overhead.
- vLLM at C=1 spends 7.0 ms per output token (including its share of prefill), 54% of the
  3.8 ms floor; TinyServe spends 32.6 ms. At C=1 there is almost no batching to hide launches.
- vLLM is also far from the floor at larger C: 29 ms per step at C=32 against a 5.9 ms floor.
  We did not profile vLLM, so where that time goes is not known.
- The M3 target was >= 3x HF at C=32. TinyServe is at 1.84x. vLLM itself is at 2.65x on this
  GPU and workload, so the target is above what vLLM 0.32 reaches here.
- Caveats: HF, TinyServe and vLLM ran in different Kaggle sessions, and vLLM on a newer torch
  (2.13.0+cu130 vs 2.11.0+cu128). TinyServe's own C=32 result varied 716-766 tok/s across
  three sessions (M4, M6, M7), so differences under ~7% are within session-to-session noise.
  Only throughput is compared: HF's static batches and vLLM's offline `generate` give no
  per-token times. Random-token prompts do not exercise the prefix cache. One workload only.

Sources: `results/m3_bench_hf_cuda.json`, `results/m6_engine_bench_cuda.json`,
`results/m6_api_bench_cuda.json`, `results/m7_vllm_bench_cuda.json`.

## 5. What would close the gap (projected, not measured)

1. **CUDA graphs for the decode step.** With the forward at its 16.7 ms of kernel time, a C=32
   step would be ~20 ms, about 1,600 tok/s. vLLM's own graphs-vs-eager gap (1.35x) is the
   measured analogue.
2. **Block tables instead of per-token slot lists in `build_batch`.** That makes it O(blocks)
   rather than O(context tokens): 3.2 ms at C=32 now, 13-15 ms at long contexts.
3. **A batched prefill attention kernel over cached blocks.** That would fix the prefix-cache
   median TTFT, and it shortens the steps that requests wait behind.
4. **GQA sharing in the decode kernel.** One k/v load per head group halves its traffic.

## How to reproduce

Kaggle T4, after the repo's notebook cell 1 (clone, `pip install --no-deps -e .`, weights):

```
!python scripts/profile_engine.py --attention triton
!python scripts/profile_engine.py --attention triton --workload prefix
!python scripts/profile_api.py --attention triton
```

vLLM replaces Kaggle's torch, so it runs last, in its own session. Kaggle's torchaudio is built
for CUDA 12.8 and breaks transformers' import under vLLM's CUDA 13 torch, so it is removed (the
benchmark uses no audio). This needs an NVIDIA driver >= 580 (Kaggle had 580.178.04):

```
!pip install -q vllm
!pip uninstall -y -q torchaudio
!python scripts/bench_vllm.py
```

GPU performance counters (Nsight Compute) are blocked in Kaggle's container
(CUPTI_ERROR_INSUFFICIENT_PRIVILEGES), and Nsight Systems is not installed, so every GPU
profile here comes from torch.profiler.
