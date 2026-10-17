# TinyServe progress

## Milestones

| Milestone | Status |
|---|---|
| M0 Repo scaffold | done (2026-10-17) |
| M1 Forward pass, token-identical to HF | in progress (step 1 done) |
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
- M1 step 1: `ModelConfig` parsed from config.json, unsupported features rejected.

## Design decisions

- `src/` layout: tests import the installed package, so packaging bugs show up early.
- mypy `strict` from day one: cheaper than retrofitting types later.
- No runtime dependencies yet: torch arrives in M1, so CI stays fast until it is needed.
- GPU tests use a `gpu` marker and auto-skip without CUDA: one suite for laptop, CI and Kaggle.
- pre-commit hooks are `local`: they reuse the venv's ruff/mypy, so local and CI versions never drift.
- `.gitattributes` forces LF: development is on Windows, CI is Linux; avoids whole-file CRLF diffs.
- CI tests 3.11, 3.12 and 3.14: Kaggle/Colab run 3.11-3.12, the laptop runs 3.14.
- CI matrix `fail-fast: false`: one red version should not hide the others' results.
- `xfail_strict = true`: a test marked xfail that starts passing fails the run, so stale xfail markers can't hide fixed or changed behaviour.
- Unsupported config features (bias, rope_scaling, non-SiLU) are rejected, not ignored: fail at load, not mid-generation.
- `head_dim` is read from config, never derived: Qwen3-0.6B uses 16 heads x 128 = 2048, not hidden_size 1024.
- No `transformers` at runtime: config.json is plain JSON, read with the standard library.

## Environment

- Laptop: Windows, PowerShell, Python 3.14.8 (CPU only).
- GPU: Kaggle and Colab free T4.

## Benchmark results

None yet.

## Open questions

- Resolved 2026-10-17: torch 2.14.1 on PyPI has a cp314 win_amd64 wheel.
- Dev tools use `>=` ranges, so CI may get a newer ruff than the laptop. Pin exact versions if that ever breaks CI.
- Pin a CPU-only torch wheel in CI (M1) to keep install time down?
