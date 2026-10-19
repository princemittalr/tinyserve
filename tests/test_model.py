"""Forward-pass tests on a tiny random Qwen3. The real-model check is in step 7."""

import torch
import torch.nn.functional as F

from tests.test_config import TINY
from tinyserve.config import ModelConfig
from tinyserve.model import HEAD_MIN_ROWS, Qwen3, greedy_generate
from tinyserve.weights import expected_shapes

CFG = ModelConfig.from_dict(TINY)


def tiny_model(seed: int = 0) -> Qwen3:
    g = torch.Generator().manual_seed(seed)
    w = {n: torch.randn(s, generator=g) * 0.5 for n, s in expected_shapes(CFG).items()}
    return Qwen3(CFG, w)


def test_logits_shape_and_dtype() -> None:
    logits = tiny_model().forward(torch.tensor([1, 2, 3, 4]))
    assert logits.shape == (4, CFG.vocab_size)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_causal_appending_tokens_does_not_change_earlier_logits() -> None:
    m = tiny_model()
    short = m.forward(torch.tensor([5, 9, 2]))
    long = m.forward(torch.tensor([5, 9, 2, 7, 1]))
    torch.testing.assert_close(long[:3], short, atol=1e-5, rtol=1e-5)


def test_order_matters() -> None:
    # Without RoPE the last position would see the same set of tokens either way.
    m = tiny_model()
    a = m.forward(torch.tensor([3, 8, 11]))[-1]
    b = m.forward(torch.tensor([8, 3, 11]))[-1]
    assert not torch.allclose(a, b)


def test_tied_embeddings_reuse_embedding_matrix() -> None:
    m = tiny_model()
    assert m.lm_head is m.w["model.embed_tokens.weight"]


def test_output_head_pads_small_batches_and_returns_only_their_rows() -> None:
    # M10: fewer than HEAD_MIN_ROWS rows are padded for the matrix multiply; the logits
    # are those of the rows given, in order, and nothing else.
    m = tiny_model()
    x = torch.randn(HEAD_MIN_ROWS + 3, CFG.hidden_size, generator=torch.Generator().manual_seed(1))
    for n in (0, 1, 2, HEAD_MIN_ROWS - 1, HEAD_MIN_ROWS, HEAD_MIN_ROWS + 3):
        out = m.head(x[:n])
        assert out.shape == (n, CFG.vocab_size) and out.dtype == torch.float32
        torch.testing.assert_close(out, F.linear(x[:n], m.lm_head), atol=1e-5, rtol=1e-5)


def test_greedy_matches_argmax_step_by_step() -> None:
    m = tiny_model()
    prompt = [1, 2, 3]
    out = greedy_generate(m, prompt, max_new_tokens=5)
    ids = list(prompt)
    for tok in out:
        assert tok == int(m.forward(torch.tensor(ids))[-1].argmax())
        ids.append(tok)
    assert len(out) == 5


def test_greedy_stops_at_eos() -> None:
    m = tiny_model()
    first = greedy_generate(m, [1, 2, 3], max_new_tokens=1)[0]
    out = greedy_generate(m, [1, 2, 3], max_new_tokens=10, eos_ids=frozenset({first}))
    assert out == [first]


def test_deterministic() -> None:
    m = tiny_model()
    assert greedy_generate(m, [4, 4, 4], 6) == greedy_generate(m, [4, 4, 4], 6)
