# TinyServe progress

## Milestones

| Milestone | Status |
|---|---|
| M0 Repo scaffold | done (2026-10-17) |
| M1 Forward pass, token-identical to HF | in progress (step 3: implementing RMSNorm and RoPE) |
| M2 Paged KV cache and allocator | not started |
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
- M1 step 3: spec tests and skeletons for `rms_norm`, `rope_cos_sin`, `apply_rope` in `layers.py`; implementation written by me.

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
- CI installs CPU-only torch: no GPU there, and the default Linux wheel pulls ~3 GB of CUDA libraries.

## Environment

- Laptop: Windows, PowerShell, Python 3.14.8 (CPU only).
- GPU: Kaggle and Colab free T4.

## Benchmark results

None yet.

## Open questions

- Resolved 2026-10-17: torch 2.14.1 on PyPI has a cp314 win_amd64 wheel.
- Dev tools use `>=` ranges, so CI may get a newer ruff than the laptop. Pin exact versions if that ever breaks CI.
- Resolved 2026-10-17: tensor names and shapes verified, `test_real_checkpoint.py` passes on the real files.
- Kaggle's preinstalled torch/CUDA version vs. the T4 (sm_75): check in the first GPU step.
