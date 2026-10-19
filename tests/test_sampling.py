"""Spec for batched sampling (M6): greedy, temperature, top-p and per-request seeds.

The statistical tests draw many rows at once with a fixed seed, so they are
deterministic; their tolerances are about 6 standard deviations of the sampling error.
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.test_model import CFG, tiny_model
from tinyserve.engine import Engine
from tinyserve.kv_cache import KVCache
from tinyserve.model import greedy_generate
from tinyserve.sampling import SamplingParams, sample, token_logprobs, top_p_filter

N = 100_000  # rows per statistical test


def gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def frequencies(logits: torch.Tensor, params: SamplingParams, n: int = N) -> torch.Tensor:
    """Sample n times from one row of logits; return how often each token came out."""
    rows = logits.expand(n, -1).contiguous()
    tokens = sample(rows, [params] * n, [None] * n, gen())
    return torch.bincount(tokens, minlength=logits.shape[-1]).double() / n


@pytest.mark.parametrize(
    ("temperature", "top_p"),
    [(-0.1, 1.0), (float("nan"), 1.0), (float("inf"), 1.0), (1.0, 0.0), (1.0, 1.5), (1.0, -1.0)],
)
def test_bad_params_are_rejected(temperature: float, top_p: float) -> None:
    with pytest.raises(ValueError):
        SamplingParams(temperature=temperature, top_p=top_p)


def test_default_is_greedy() -> None:
    assert SamplingParams().greedy
    assert not SamplingParams(temperature=0.7).greedy


def test_all_greedy_is_argmax_and_draws_no_noise() -> None:
    logits = torch.randn(6, 50, generator=gen(1))
    g = gen()
    before = g.get_state()
    out = sample(logits, [SamplingParams()] * 6, [None] * 6, g)
    assert torch.equal(out, logits.argmax(-1))
    assert torch.equal(g.get_state(), before)  # greedy runs never touch the generator


def test_greedy_rows_stay_argmax_in_a_mixed_batch() -> None:
    logits = torch.randn(8, 50, generator=gen(2))
    params = [SamplingParams(temperature=1.0 if b % 2 else 0.0) for b in range(8)]
    out = sample(logits, params, [None] * 8, gen())
    assert torch.equal(out[::2], logits[::2].argmax(-1))


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_temperature_samples_from_softmax_of_scaled_logits(temperature: float) -> None:
    logits = torch.tensor([2.0, 1.0, 0.5, 0.0, -1.0])
    freq = frequencies(logits, SamplingParams(temperature=temperature))
    want = torch.softmax(logits.double() / temperature, -1)
    assert (freq - want).abs().max() < 0.01


def test_top_p_keeps_the_nucleus_and_renormalises() -> None:
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    logits = probs.log()
    # 0.5 < 0.8, 0.5 + 0.3 >= 0.8: the nucleus is the first two tokens.
    freq = frequencies(logits, SamplingParams(temperature=1.0, top_p=0.8))
    assert freq[2] == 0 and freq[3] == 0
    assert abs(freq[0] - 0.5 / 0.8) < 0.01
    # 0.85 needs the third token as well.
    freq = frequencies(logits, SamplingParams(temperature=1.0, top_p=0.85))
    assert freq[3] == 0 and freq[2] > 0
    assert abs(freq[2] - 0.15 / 0.95) < 0.01


def test_top_p_applies_per_request_in_a_mixed_batch() -> None:
    logits = torch.tensor([0.5, 0.3, 0.15, 0.05]).log().expand(2000, -1).contiguous()
    params = [SamplingParams(temperature=1.0, top_p=0.8 if b % 2 else 1.0) for b in range(2000)]
    out = sample(logits, params, [None] * 2000, gen())
    assert out[1::2].max() <= 1  # nucleus rows: only tokens 0 and 1
    assert out[::2].max() >= 2  # full rows still reach the tail


def test_tiny_top_p_or_tiny_temperature_is_argmax() -> None:
    logits = torch.randn(64, 100, generator=gen(3))
    for p in (SamplingParams(temperature=1.0, top_p=1e-6), SamplingParams(temperature=1e-4)):
        out = sample(logits, [p] * 64, [None] * 64, gen())
        assert torch.equal(out, logits.argmax(-1))


@settings(max_examples=200, deadline=None)
@given(
    weights=st.lists(st.integers(1, 10_000), min_size=1, max_size=30, unique=True),
    top_p=st.floats(0.01, 1.0),
)
def test_top_p_filter_keeps_the_smallest_most_likely_set(weights: list[int], top_p: float) -> None:
    probs = torch.tensor(weights, dtype=torch.float64)
    probs /= probs.sum()
    kept = top_p_filter(probs.unsqueeze(0), torch.tensor([top_p], dtype=torch.float64))[0]
    keep = kept > 0
    assert torch.equal(kept[keep], probs[keep])  # kept tokens keep their probability
    assert keep[probs.argmax()]  # never empty: the most likely token survives
    if (~keep).any():
        assert probs[keep].min() > probs[~keep].max()  # nothing kept is less likely
        assert probs[keep].sum() >= top_p - 1e-12  # the nucleus reaches top_p ...
    # ... and is minimal: without its least likely token it would fall short.
    assert probs[keep].sum() - probs[keep].min() < top_p + 1e-12


def test_top_p_filter_is_per_row() -> None:
    probs = torch.tensor([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]])
    out = top_p_filter(probs, torch.tensor([0.5, 1.0]))
    assert torch.equal(out[0], torch.tensor([0.5, 0.0, 0.0]))
    assert torch.equal(out[1], probs[1])


def test_seeded_row_ignores_its_neighbours() -> None:
    logits = torch.randn(4, 50, generator=gen(4))
    p = SamplingParams(temperature=1.0, seed=7)
    alone = [sample(logits[:1], [p], [gen(7)], gen(0)) for _ in range(2)]
    others = SamplingParams(temperature=1.0)
    crowd = sample(logits, [p, others, others, others], [gen(7), None, None, None], gen(99))
    assert alone[0][0] == alone[1][0] == crowd[0]


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError):
        sample(torch.zeros(2, 5), [SamplingParams()], [None, None], gen())


# --- through the engine ---------------------------------------------------------------

PROMPTS = [[1], [5, 9, 2], [3, 8, 11, 2, 7, 7, 1], [4, 4, 6, 1, 2]]
HOT = SamplingParams(temperature=1.5, top_p=0.95, seed=1234)


def engine(num_blocks: int = 64, max_num_seqs: int = 8, seed: int = 0) -> Engine:
    cache = KVCache(CFG, num_blocks, 4, dtype=torch.float32)
    return Engine(tiny_model(), cache, max_num_seqs=max_num_seqs, seed=seed)


def test_engine_default_is_still_m1_greedy() -> None:
    cold = SamplingParams(temperature=0.0, top_p=0.3, seed=5)  # top_p, seed ignored
    want = [greedy_generate(tiny_model(), p, 8) for p in PROMPTS]
    assert engine().generate(PROMPTS, 8) == want
    assert engine().generate(PROMPTS, 8, sampling=cold) == want


def test_engine_sampling_actually_samples() -> None:
    out = engine().generate(PROMPTS, 12, sampling=SamplingParams(temperature=1.5))
    want = [greedy_generate(tiny_model(), p, 12) for p in PROMPTS]
    assert out != want
    assert all(len(o) == 12 and all(0 <= t < CFG.vocab_size for t in o) for o in out)


def test_seeded_request_is_reproducible_across_batches_and_preemption() -> None:
    # The same seeded request: alone, among unseeded strangers, and in a cache so small
    # that sequences get preempted and recomputed. Its tokens must not change.
    alone = engine().generate([PROMPTS[2]], 10, sampling=HOT)[0]

    e = engine(max_num_seqs=4, seed=42)
    mine = e.add_request(PROMPTS[2], 10, sampling=HOT)
    for p in PROMPTS:
        e.add_request(p, 10, sampling=SamplingParams(temperature=1.5))
    while e.has_work():
        e.step()
    assert e.seqs[mine].output == alone

    tight = engine(num_blocks=7)
    ids = [tight.add_request(p, 10, sampling=HOT) for p in [PROMPTS[2], *PROMPTS]]
    while tight.has_work():
        tight.step()
    assert tight.scheduler.num_preemptions > 0
    assert tight.seqs[ids[0]].output == alone
    assert tight.scheduler.allocator.num_free == 7
    assert not tight.generators  # dropped as each request finished


def test_different_seeds_give_different_outputs() -> None:
    outs = {
        tuple(engine().generate([PROMPTS[1]], 12, sampling=SamplingParams(1.5, seed=s))[0])
        for s in range(5)
    }
    assert len(outs) > 1


# --- logprobs (M9) ----------------------------------------------------------------------


def test_logprobs_must_be_0_to_20() -> None:
    assert SamplingParams(logprobs=0).logprobs == 0 and SamplingParams(logprobs=20).logprobs == 20
    for k in (-1, 21):
        with pytest.raises(ValueError):
            SamplingParams(logprobs=k)


def test_token_logprobs_are_log_softmax_at_the_token_and_the_top_k() -> None:
    torch.manual_seed(0)
    logits = torch.randn(3, 10) * 4
    tokens = torch.tensor([7, 0, 3])
    got = token_logprobs(logits, tokens, [2, 0, 10])
    want = torch.log_softmax(logits, dim=-1)
    for b, lp in enumerate(got):
        assert lp.logprob == pytest.approx(want[b, tokens[b]].item(), abs=1e-6)
        order = want[b].argsort(descending=True).tolist()
        assert [i for i, _ in lp.top] == order[: [2, 0, 10][b]]
        for i, v in lp.top:
            assert v == pytest.approx(want[b, i].item(), abs=1e-6)


def reference_logprobs(prompt: list[int], output: list[int]) -> list[torch.Tensor]:
    """log_softmax of the M1 model's logits before each output token."""
    m = tiny_model()
    logits = m.forward(torch.tensor(prompt + output))[len(prompt) - 1 : -1]
    return list(torch.log_softmax(logits.float(), dim=-1))


@pytest.mark.parametrize(
    ("num_blocks", "chunked"),
    [(64, False), (64, True), (7, False)],  # 7: preemptions
)
def test_engine_logprobs_match_the_model(num_blocks: int, chunked: bool) -> None:
    e = Engine(tiny_model(), KVCache(CFG, num_blocks, 4, dtype=torch.float32),
               max_num_seqs=4, max_batch_tokens=8 if chunked else 2048,
               chunked_prefill=chunked)  # fmt: skip
    prompts = [[1, 2, 3, 4, 5, 6, 7, 8, 9], [9, 8, 7], [5, 5, 5, 5], [2, 4, 6, 8, 1]]
    hot = SamplingParams(1.5, 0.9, seed=3, logprobs=3)  # logprobs ignore temperature, top-p
    ids = [e.add_request(p, 8, sampling=hot if k % 2 else SamplingParams(logprobs=2))
           for k, p in enumerate(prompts)]  # fmt: skip
    plain = e.add_request([3, 1, 4], 8)  # no logprobs asked: none kept
    while e.has_work():
        e.step()
    if num_blocks == 7:
        assert e.scheduler.num_preemptions > 0
    assert plain not in e.logprobs
    for k, (i, p) in enumerate(zip(ids, prompts, strict=True)):
        out, lps = e.seqs[i].output, e.logprobs[i]
        assert len(lps) == len(out) == 8  # one per token, none twice after a preemption
        for t, lp, ref in zip(out, lps, reference_logprobs(p, out), strict=True):
            assert lp.logprob == pytest.approx(ref[t].item(), abs=1e-4)
            assert [j for j, _ in lp.top] == ref.argsort(descending=True)[
                : 3 if k % 2 else 2
            ].tolist()
            if k % 2 == 0:  # greedy: the token is the most likely one
                assert lp.top[0][0] == t and lp.top[0][1] == pytest.approx(lp.logprob)
    e.release(ids[0])
    assert ids[0] not in e.logprobs


def test_asking_for_logprobs_does_not_change_the_tokens() -> None:
    def run(params: SamplingParams) -> list[int]:
        e = Engine(tiny_model(), KVCache(CFG, 64, 4, dtype=torch.float32))
        return e.generate([[4, 2, 4, 2]], 10, sampling=params)[0]

    assert run(SamplingParams(1.2, 0.8, 7)) == run(SamplingParams(1.2, 0.8, 7, logprobs=5))
