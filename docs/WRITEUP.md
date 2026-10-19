# TinyServe on a T4: where the time goes (M7, M8, M9)

Every number below comes from a logged run on a Kaggle Tesla T4 (2026-10-18 and 2026-10-19),
fp16, Qwen3-0.6B. Sections 1-5 are the M7 analysis, before M8; section 6 is what M8 changed and section 7 what M9 changed. Raw data is in `results/`; each table names its file. Anything not measured is
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

## 5. What would close the gap (projected in M7, not measured then)

M8 built items 1, 3 and 4; section 6 has what they measured.

1. **CUDA graphs for the decode step.** With the forward at its 16.7 ms of kernel time, a C=32
   step would be ~20 ms, about 1,600 tok/s. vLLM's own graphs-vs-eager gap (1.35x) is the
   measured analogue.
2. **Block tables instead of per-token slot lists in `build_batch`.** That makes it O(blocks)
   rather than O(context tokens): 3.2 ms at C=32 now, 13-15 ms at long contexts.
3. **A batched prefill attention kernel over cached blocks.** That would fix the prefix-cache
   median TTFT, and it shortens the steps that requests wait behind.
4. **GQA sharing in the decode kernel.** One k/v load per head group halves its traffic.

## 6. What M8 changed

M8 added three things, each measured against the configuration before it in one session
(M3 workload, Kaggle T4, fp16; `scripts/benchmark.py`; output tok/s):

| C | M4 kernel, eager | + CUDA graphs | + batched prefill | + grouped kernel | final / HF | final / vLLM 0.32 |
|---|---|---|---|---|---|---|
| 1 | 30.4 | 87.4 | 87.5 | 84.3 | 3.35x | 59% |
| 8 | 231.2 | 545.8 | 548.8 | 571.0 | 2.95x | 84% |
| 32 | 763.0 | 1,142.3 | 1,206.5 | 1,343.8 | 3.23x | 122% |
| 64 | 1,213.9 | 1,419.8 | 1,493.8 | 1,753.7 | 3.89x | 127% |

The columns come from three sessions. The same-session pairs at C=32 are 763.0 / 1,142.3
(graphs, +50%), 1,148.0 / 1,206.5 (batched prefill, +5.1%) and 1,213.8 / 1,343.8 (grouped
kernel, +10.7%). The M3 target, >= 3x HF at C=32 (>= 1,247 tok/s), is met at 3.23x.
Correctness held at every step: fp32 greedy 0 mismatches / 1,920 tokens vs HF with all three
on, fp16 only near-tie flips (HF top-2 margin <= 0.0156), and on the GPU a graph replay equals
the eager decode pass bit for bit.

**CUDA graphs** (`src/tinyserve/cuda_graphs.py`). The decode step was rewritten to have fixed
shapes and no host syncs (`Qwen3.forward_decode` on padded `DecodeInputs`; padding rows write
into a KV-cache block no sequence owns), then captured once per batch-size bucket (1, 2, 4, 8,
then multiples of 8; 2.3-2.7 s and 222 MiB for 11 buckets). A C=32 decode step went from
34.2 ms of forward with 1,851 kernel launches to one graph replay; the GPU is busy 91% of the
step (M7: about half). The section 5 projection was ~1,600 tok/s at C=32; the measurement was
1,142. The projection ignored prefill: steps carrying a prefill (eager, ~225 ms each at C=32)
took 20% of the run's step time once decode steps were short.

**Batched prefill attention** (`src/tinyserve/attention.py`). One PyTorch SDPA call per layer
over every prefilling sequence's keys and values gathered from the cache, padded and masked,
in fp32 like the other paths, instead of ~17 small kernels per sequence per layer. A C=32
prefill step fell from 224.5 to 177.5 ms, not to the ~60 ms I projected; *inferred:* most of
what is left is the GEMMs over ~1,400 prompt tokens. On the prefix-cache workload it cut the
C=32 cache-on median TTFT from 644 to 505 ms.

**A grouped decode kernel** (`_paged_decode_grouped_kernel` in
`src/tinyserve/triton_attention.py`). One program per (sequence, kv head) loads each k/v block
once and runs the online softmax for both query heads of the GQA pair. In the C=32 profile it
takes 4.2 ms per step at 118 GB/s counting each k/v once (the M4 kernel: 6.2 ms in another
session, each k/v loaded twice). It is 2.7% slower at C=1; *inferred:* at batch 1 it runs 8
programs on the T4's 40 SMs, against 16 for the M4 kernel.

Where a C=32 decode step goes now (step trace and torch.profiler, grouped kernel): 20.0 ms in
all, 19.2 ms of it the forward; the roofline floor is 31% of that (M7: 17%). Of the 17.0 ms of
GPU time per profiled step, GEMMs take 7.1 ms (169 GB/s), the decode kernel 4.2 ms, and ~5.7 ms
is small norm, RoPE and copy kernels that the graph replays but does not fuse. Steps carrying
a prefill are 17% of the run's step time at C=32 and 23% at C=64.

On the comparison with vLLM: TinyServe is now ahead at C=32 and C=64 on this workload and
behind at C=1 and C=8. The caveats of section 4 apply in full: vLLM ran in another session,
on torch 2.13.0+cu130 against 2.11.0+cu128, with default settings and its own scheduler, and
TinyServe's own C=32 result has varied ~6% between sessions with unchanged code. The fair
reading is that a small engine with graphs, a GQA-aware decode kernel and batched prefill can
match a general-purpose engine's defaults on one model, GPU and synthetic workload; it says
nothing about other workloads (real prompts, EOS, long contexts), and vLLM's much lower C=1
time (7.0 vs 11.9 ms per token, the latter from 84.3 tok/s) shows there is still more to take
out of a small batch.

What is still open, measured: the eager prefill steps (~175 ms at C=32; chunked prefill would
cap them), the unfused small kernels (~5.7 ms of GPU time per C=32 step), the grouped kernel
at batch 1 (split-K over the context would give it more programs), and TTFT p99 over HTTP
(3,044 ms at C=32 with all three on; M6: 5,607 ms). *Inferred, not re-profiled:* that p99 is
still mostly the held-back empty text of section 3.

Over HTTP (one later session, `scripts/bench_api.py`): 85.2 / 566.6 / 1,332.0 / 1,724.2 tok/s
at C = 1 / 8 / 32 / 64, 3.20x HF at C=32 (M6: 723.8). Greedy text through the API (fp32, prefix
cache on, 120 prompts) is identical to HF `generate()`.

## 7. What M9 changed

M9 closed the items left open after M8. All numbers below are from one Kaggle T4 session
(fp16, the M3 workload, `scripts/benchmark.py`, output tok/s), except where noted.

| C | A: M8 | D: split-K, Triton prefill, fused | E: D + chunked prefill | F: split-K, batched prefill, fused | F / A | F / vLLM 0.32 |
|---|---|---|---|---|---|---|
| 1 | 81.2 | 132.9 | 131.5 | 129.0 | 1.59x | 91% |
| 8 | 562.6 | 793.4 | 721.8 | 775.0 | 1.38x | 114% |
| 32 | 1,325.2 | 1,600.8 | 1,624.9 | 1,580.3 | 1.19x | 143% |
| 64 | 1,722.5 | 1,931.1 | 1,972.3 | 1,969.4 | 1.14x | 142% |

F is now the server's default on a GPU. The vLLM column carries section 4's caveats in full
(another session, another torch, vLLM's defaults).

**Split-K for the decode kernel** (`_paged_decode_split_kernel`, `_merge_splits_kernel`). At
batch 1 the grouped kernel ran 8 programs on 40 SMs. Split-K cuts each sequence's blocks into
up to 16 ranges, one program each, and merges the partial softmaxes the way the online softmax
merges blocks. The number of ranges depends only on shapes, so CUDA graphs still replay one
grid per bucket, and from batch 10 up it is 1, the plain grouped kernel. With 2,048-token
prompts at C=1 it made decoding 1.39x faster (TPOT 20.9 -> 13.6 ms, another session).

**Fused kernels** (`src/tinyserve/triton_fused.py`). Section 6 left ~5.7 ms of small norm,
RoPE and copy kernels in a C=32 step. One kernel does RMSNorm per row, another the q/k head
norms, RoPE and the k/v cache write, following the PyTorch path's rounding step by step. A
C=32 decode step's GPU time fell from 17.3 to 13.8 ms (step 14.8 ms, GEMMs 7.4 ms). With
split-K, that is +59% at C=1 and +19% at C=32 over M8.

**The Triton prefill kernel** (`src/tinyserve/triton_prefill.py`) reads keys and values
straight from the cache blocks, without M8's padded gather. Its first version was 6.6x slower
than batched SDPA. Compiling for sm_75 without a GPU (`scripts/check_t4_compile.py`) showed
why: Triton emits no tensor-core instructions before sm_80, so `tl.dot` on the T4 is a loop of
FMAs with its tiles in registers, and that tile spilled 10.6 KB per thread to local memory.
The best tile measured (`scripts/bench_prefill_kernel.py`) is 15% faster than batched SDPA on
32 prefix hits (17.6 vs 20.6 ms per layer) and 25% slower on 8 fresh 512-token prompts (9.5
vs 7.6 ms). In the engine, D and F are within 1.3% on the M3 workload, and F is 3.6% faster on
the prefix workload, so batched prefill stays the default. *Inferred:* on a GPU where Triton
uses tensor cores (sm_80 and up) the kernel should win; that was not measured.

**Chunked prefill** caps a step at `--max-batch-tokens` (512 here) by splitting prompts across
steps. At C=32 it cut the median TTFT from 168 to 55 ms and the longest gap between two tokens
of a request from 173 to 61 ms, at the same throughput. At C=8 the one run showed a worse tail
(TTFT p99 430 vs 82 ms), not explained, so it is opt-in.

**Smaller items.** Prefill steps build their cache slots from block lists instead of
per-token Python lists (on a CPU: 9.0 ms of slot lists became a 1.4 ms `build_batch`). The
API gained `stop` strings, `n` choices and logprobs. CI runs Python 3.13 too.

Correctness held: fp32 greedy with every M9 option at once (and 64-token chunks) 0 mismatches
/ 1,920 tokens vs HF; fp16 6 near-tie divergences at HF top-2 margins <= 0.0625 (M8:
<= 0.0156; 0.0625 is one fp16 step at a logit of 64-128). Greedy text through the API with
every option and the prefix cache is identical to HF in fp32.

What is left, measured: decode at C=32 is ~15 ms per step against a roofline floor of ~6 ms
(GEMMs at ~160 GB/s of the T4's 320); vLLM is still faster at C=1 (142.4 vs 129.0 tok/s); and
TTFT p99 over HTTP at C=32 is 2,755 ms, *inferred* to be mostly section 3's held-back empty
text.

## How to reproduce

Kaggle T4, after the repo's notebook cell 1 (clone, `pip install --no-deps -e .`, weights):

```
!python scripts/profile_engine.py --attention triton
!python scripts/profile_engine.py --attention triton --workload prefix
!python scripts/profile_api.py --attention triton
```

M8 (same session as the commands above):

```
!python scripts/benchmark.py --attention triton --cuda-graphs --prefill-attention batched
!python scripts/benchmark.py --attention triton_grouped --cuda-graphs --prefill-attention batched
!python scripts/compare_hf.py --engine batched --device cuda --dtype float32 --attention triton_grouped --cuda-graphs --prefill-attention batched
!python scripts/profile_engine.py --device cuda --attention triton_grouped --cuda-graphs --prefill-attention batched
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

M9 (one session, after cell 1):

```
!python scripts/check_t4_compile.py
!python scripts/bench_prefill_kernel.py
!python scripts/benchmark.py --device cuda --cuda-graphs --attention triton_grouped --prefill-attention batched
!python scripts/benchmark.py --device cuda --cuda-graphs --attention triton_split --prefill-attention batched --fused-kernels
!python scripts/compare_hf.py --engine batched --device cuda --dtype float32 --cuda-graphs --attention triton_split --prefill-attention triton --fused-kernels --chunked-prefill --max-batch-tokens 64
```
