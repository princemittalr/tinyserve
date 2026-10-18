# TinyServe progress

## Milestones

| Milestone | Status |
|---|---|
| M0 Repo scaffold | done (2026-10-17) |
| M1 Forward pass, token-identical to HF | done (2026-10-18): 0 mismatches / 1,920 tokens |
| M2 Paged KV cache and allocator | in progress |
| M3 Continuous batching and benchmark harness | not started |
| M4 Triton paged-attention kernel | not started |
| M5 Radix prefix cache | not started |
| M6 Streaming API | not started |
| M7 Profiling and write-up | not started |

## Step log

- M0 step 1: pyproject, ruff, mypy (strict), pytest + Hypothesis, CI, README, PROGRESS.
- M0 step 2: pre-commit hook, `.gitattributes`, CI on 3.14, first push to GitHub.
- M0 step 3: CI `fail-fast: false`. Verified CI goes red on a failing test (canary PR #1, closed).
- M1 step 1: `ModelConfig` parsed from config.json, unsupported features rejected. Verified on the real Qwen3-0.6B config.json.
- M1 step 2: torch added; strict safetensors loader, synthetic-checkpoint tests, `scripts/inspect_weights.py`. Real checkpoint (2026-10-17, laptop): 310 tensors, 596,049,920 parameters (440,467,456 without the embedding), all finite in fp16; largest |weight| 192.0.
- M1 step 3: spec tests and skeletons for `rms_norm`, `rope_cos_sin`, `apply_rope` in `layers.py`. I attempted it, got stuck, and used the mentor's solution; I can explain each line.
- M1 step 4: SwiGLU MLP (`swiglu_mlp`), 5 spec tests; the three blanks were answered by the mentor.
- M1 step 5: reference causal grouped-query attention (`attention.py`), 8 spec tests incl. a match against PyTorch SDPA. Blank answers came from the mentor.
- M1 step 6: `Qwen3` forward pass (28 layers, no KV cache) and `greedy_generate`; tiny-random-model tests; `scripts/generate.py` demo. First real run (2026-10-18, laptop CPU, fp32): "The capital of France is" -> " Paris. The capital of Italy is Rome. ...", 20 tokens in 6.8 s.
- M1 step 7: HF parity tests on a tiny random Qwen3 (run in CI); `scripts/compare_hf.py` + 120 prompts for the real-model acceptance run. Config now also reads the newer `rope_parameters` layout.
- M1 closed: acceptance run passed (see Benchmark results).
- M2 step 1: `BlockAllocator` (free stack + used set) skeleton with 7 blanks for me to fill; 14 unit tests and a Hypothesis state machine (random allocate/free traffic; checks no double hand-out, double free rejected, no leaked blocks).

## Design decisions

- `src/` layout: tests import the installed package, so packaging bugs show up early.
- mypy `strict` from day one: cheaper than retrofitting types later.
- GPU tests use a `gpu` marker and auto-skip without CUDA: one suite for laptop, CI and Kaggle.
- pre-commit hooks are `local`: they reuse the venv's ruff/mypy, so local and CI versions never drift.
- `.gitattributes` forces LF: development is on Windows, CI is Linux; avoids whole-file CRLF diffs.
- CI tests 3.11, 3.12 and 3.14: Kaggle/Colab run 3.11-3.12, the laptop runs 3.14.
- CI matrix `fail-fast: false`: one red version should not hide the others' results.
- `xfail_strict = true`: a test marked xfail that starts passing fails the run, so stale xfail markers can't hide fixed or changed behaviour.
- Unsupported config features (bias, rope_scaling, non-SiLU) are rejected, not ignored: fail at load, not mid-generation.
- `head_dim` is read from config, never derived: Qwen3-0.6B uses 16 heads x 128 = 2048, not hidden_size 1024.
- No `transformers` at runtime: config.json is plain JSON, read with the standard library. huggingface_hub is only the download client.
- Weights default to fp16: Turing tensor cores do fp16, not bf16. bf16->fp16 can overflow, so the loader checks for inf/nan.
- Strict loader (missing, unexpected or misshapen tensor = error): a silent mismatch would surface much later as wrong tokens.
- RMSNorm and RoPE do their math in fp32 and cast back: fp16 squares overflow past 255, and RoPE angles reach 40960 rad.
- RoPE uses the rotate-half layout (pair i with i + D/2): it is what Qwen3 was trained with; interleaved pairs give wrong tokens.
- RoPE takes explicit positions per token: in a continuous batch every sequence is at a different position.
- Linear layers use `F.linear(x, w)` on the checkpoint's [out, in] weights: no transposed copies, so memory stays at 1x.
- Reference attention is plain PyTorch (einsum + mask + softmax in fp32): the paged cache (M2) and Triton kernel (M4) are tested against it.
- GQA maps query head h to kv head h // (Hq/Hkv) (`repeat_interleave`, not `repeat`): that is how Qwen3 groups heads.
- M1 has no KV cache: each step re-runs the whole sequence (O(n^2)). Simplest correct baseline; M2 replaces it.
- Qwen3 applies q_norm/k_norm per head before RoPE; the output head reuses the embedding matrix (tied).
- M1 acceptance is measured in fp32 on CPU: it isolates algorithm bugs from fp16 kernel rounding, which is a separate question for the GPU milestones.
- `transformers` is a dev-only dependency used as the reference in tests; tinyserve never imports it.
- CI installs CPU-only torch: no GPU there, and the default Linux wheel pulls ~3 GB of CUDA libraries.
- Block allocator is plain Python with no tensors: bookkeeping is separate from the KV memory, so it can be property-tested in milliseconds.
- Free blocks are a list used as a stack: allocate and free are O(1) per block, and a just-freed block is reused first.
- Used blocks are tracked in a set: the double-free check is O(1) instead of scanning the free list.
- allocate() and free() are all-or-nothing: on an error nothing changes, so the M3 scheduler can catch OutOfBlocksError and preempt safely.
- M2 starts without reference counts; the M5 prefix cache, where sequences share blocks, will add them.

## Environment

- Laptop: Windows, PowerShell, Python 3.14.8 (CPU only).
- GPU: Kaggle and Colab free T4.

## Benchmark results

| Date | Hardware | What | Result |
|---|---|---|---|
| 2026-10-18 | Windows AMD64 laptop CPU, torch 2.14.1+cpu, fp32 | M1 greedy parity vs HF `generate()`: 120 prompts x 16 tokens | 0 mismatches / 1,920 tokens; max prefill logit abs diff 1.23e-4; 1047.5 s total (no KV cache) |
| 2026-10-18 | same | `scripts/generate.py`, 5-token prompt, 20 new tokens | 6.8 s (no KV cache; correctness baseline, not a speed claim) |

Raw data: `results/m1_hf_compare.json`.

## Open questions

- Resolved 2026-10-18: the block allocator (core algorithm) is scaffolded as plain-Python blanks for me to fill, like M1.
- Greedy parity in fp16 on the T4: expect rare near-tie flips; decide the tolerance policy when GPU work starts.

- Resolved 2026-10-17: torch 2.14.1 on PyPI has a cp314 win_amd64 wheel.
- Dev tools use `>=` ranges, so CI may get a newer ruff than the laptop. Pin exact versions if that ever breaks CI.
- Resolved 2026-10-17: tensor names and shapes verified, `test_real_checkpoint.py` passes on the real files.
- Kaggle's preinstalled torch/CUDA version vs. the T4 (sm_75): check in the first GPU step.
