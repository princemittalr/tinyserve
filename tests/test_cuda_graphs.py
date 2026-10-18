"""CUDA graphs for the decode step (M8): bucketing, the engine wiring, and the graphs.

On CPU, ``DecodeGraphs`` runs ``forward_decode`` eagerly on the same padded buffers, so
these tests check the bucketing, the padding and when the engine takes the graph path.
The ``gpu`` tests capture real graphs: a replay must equal the eager pass bit for bit,
capturing must not touch a sequence's blocks, and greedy output must not change.
"""

import pytest
import torch

pytest.importorskip("triton", reason="Triton is not installed (no Windows wheels)")

from tests.test_decode_inputs import random_cache
from tests.test_engine import PROMPTS, expected
from tests.test_kernel_forward import models, tables
from tests.test_model import CFG
from tinyserve.batch import DecodeInputs, fill_decode_inputs
from tinyserve.block_allocator import BlockAllocator
from tinyserve.cuda_graphs import DecodeGraphs, default_buckets
from tinyserve.engine import Engine
from tinyserve.kv_cache import KVCache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def cpu_graphs(cache: KVCache, **kw: object) -> DecodeGraphs:
    _, model = models()
    return DecodeGraphs(model, cache, capture=False, **kw)  # type: ignore[arg-type]


# ---------------------------------------------------------------- buckets


def test_default_buckets() -> None:
    assert default_buckets(64) == [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
    assert default_buckets(1) == [1]
    assert default_buckets(8) == [1, 2, 4, 8]
    assert default_buckets(12) == [1, 2, 4, 8, 12]
    with pytest.raises(ValueError, match=">= 1"):
        default_buckets(0)


def test_a_step_pads_to_the_smallest_bucket_that_fits() -> None:
    g = cpu_graphs(random_cache(), buckets=[8, 2, 4])
    assert g.buckets == [2, 4, 8] and g.max_batch == 8
    assert [g.bucket_for(n) for n in (1, 2, 3, 5, 8)] == [2, 2, 4, 8, 8]
    with pytest.raises(ValueError, match="largest bucket"):
        g.bucket_for(9)


def test_fits_checks_batch_size_and_block_table_width() -> None:
    cache = random_cache()
    g = cpu_graphs(cache, max_batch=2, max_blocks_per_seq=2)
    short, long, other = tables(BlockAllocator(cache.num_blocks), [3, 8, 0], [1, 1, 1])
    assert g.fits([short, other])
    assert not g.fits([long])  # 9 tokens = 3 blocks of 4
    assert not g.fits([short, other, short])
    assert not g.fits([])


def test_default_width_is_the_longest_sequence_the_cache_or_model_allows() -> None:
    assert cpu_graphs(random_cache(num_blocks=8)).max_blocks_per_seq == 8
    # The tiny config allows max_position_embeddings positions in blocks of 4.
    width = -(-CFG.max_position_embeddings // 4)
    assert cpu_graphs(random_cache(num_blocks=10_000)).max_blocks_per_seq == width


def test_needs_the_triton_model() -> None:
    torch_model, _ = models()
    with pytest.raises(ValueError, match="triton"):
        DecodeGraphs(torch_model, random_cache(), capture=False)


def test_prepare_and_replay_give_forward_decode_rows() -> None:
    _, model = models()
    cache = random_cache()
    g = DecodeGraphs(model, cache, buckets=[4], capture=False)
    ts = tables(BlockAllocator(cache.num_blocks), [5, 2, 7], [1, 1, 1])
    items = [(3, ts[0]), (4, ts[1]), (5, ts[2])]
    assert g.prepare(items) == 4
    got = g.replay(4, 3)
    inp = DecodeInputs.empty(4, g.max_blocks_per_seq, DEVICE)
    fill_decode_inputs(inp, items, cache.pad_block)
    torch.testing.assert_close(got, model.forward_decode(inp, cache)[:3], atol=0, rtol=0)


# ---------------------------------------------------------------- the engine


def graph_engine(
    num_blocks: int = 64, max_num_seqs: int = 8, capture: bool = False, **kw: object
) -> Engine:
    _, model = models()
    cache = KVCache(CFG, num_blocks, 4, dtype=torch.float32, device=DEVICE)
    graphs = DecodeGraphs(model, cache, capture=capture, **kw)  # type: ignore[arg-type]
    return Engine(model, cache, max_num_seqs=max_num_seqs, decode_graphs=graphs, trace_steps=True)


@pytest.mark.parametrize(
    ("num_blocks", "max_num_seqs"),
    [(64, 8), (64, 1), (7, 8)],  # roomy; one at a time; tight cache (preemptions)
)
def test_engine_with_graphs_matches_m1_greedy(num_blocks: int, max_num_seqs: int) -> None:
    e = graph_engine(num_blocks, max_num_seqs, max_batch=8)
    assert e.generate(PROMPTS, 10) == [expected(p, 10) for p in PROMPTS]
    assert e.step_records is not None
    graphed = [r.graph for r in e.step_records]
    assert any(graphed), "no step took the graph path"
    # Exactly the steps where every sequence decodes one token took it.
    assert graphed == [r.num_tokens == r.num_seqs for r in e.step_records]
    if num_blocks == 7:
        assert e.scheduler.num_preemptions > 0


def test_engine_runs_eagerly_what_does_not_fit_a_graph() -> None:
    # 5 sequences decode together, but the largest bucket is 2.
    e = graph_engine(max_num_seqs=8, buckets=[1, 2])
    assert e.generate(PROMPTS, 10) == [expected(p, 10) for p in PROMPTS]
    assert e.step_records is not None
    assert not any(r.graph for r in e.step_records if r.num_seqs > 2)
    # Two at a time, with 2-block (8-token) tables: sequences outgrow them while decoding.
    e = graph_engine(max_num_seqs=2, buckets=[1, 2], max_blocks_per_seq=2)
    assert e.generate(PROMPTS, 10) == [expected(p, 10) for p in PROMPTS]
    assert e.step_records is not None
    decode = [r for r in e.step_records if r.num_tokens == r.num_seqs]
    assert any(r.graph for r in decode) and any(not r.graph for r in decode)
    assert not any(r.graph for r in decode if r.context_tokens > 2 * 8)


def test_engine_with_graphs_and_prefix_cache_matches_m1_greedy() -> None:
    from tests.test_prefix_scheduler import QUESTIONS, SYSTEM

    _, model = models()
    cache = KVCache(CFG, 64, 4, dtype=torch.float32, device=DEVICE)
    graphs = DecodeGraphs(model, cache, max_batch=2, capture=False)
    e = Engine(model, cache, max_num_seqs=2, prefix_cache=True, decode_graphs=graphs)
    prompts = [SYSTEM + q for q in QUESTIONS] + [SYSTEM]
    assert e.generate(prompts, 6) == [expected(p, 6) for p in prompts]
    assert e.scheduler.num_cached_tokens > 0


def test_engine_rejects_graphs_of_another_cache() -> None:
    _, model = models()
    graphs = DecodeGraphs(model, random_cache(), capture=False)
    with pytest.raises(ValueError, match="model and cache"):
        Engine(model, random_cache(), decode_graphs=graphs)


# ---------------------------------------------------------------- real graphs (GPU)


@pytest.mark.gpu
def test_replay_equals_the_eager_pass_exactly() -> None:
    _, model = models()
    cache = random_cache()
    g = DecodeGraphs(model, cache, max_batch=8)
    assert sorted(g.graphs) == [1, 2, 4, 8]
    ts = tables(BlockAllocator(cache.num_blocks), [5, 0, 12, 3, 9], [1, 1, 1, 1, 1])
    for n in (1, 3, 5):
        items = [(7 + i, t) for i, t in enumerate(ts[:n])]
        bucket = g.prepare(items)
        got = g.replay(bucket, n).clone()
        want = model.forward_decode(g.inputs.head(bucket), cache)[:n]
        torch.testing.assert_close(got, want, atol=0, rtol=0)


@pytest.mark.gpu
def test_capture_touches_only_the_padding_block() -> None:
    _, model = models()
    cache = random_cache()
    before = cache.k[:, : cache.num_slots].clone()
    g = DecodeGraphs(model, cache, max_batch=4)
    assert g.capture_seconds > 0
    assert torch.equal(cache.k[:, : cache.num_slots], before)


@pytest.mark.gpu
@pytest.mark.parametrize(("num_blocks", "max_num_seqs"), [(64, 8), (7, 8)])
def test_engine_with_captured_graphs_matches_m1_greedy(num_blocks: int, max_num_seqs: int) -> None:
    e = graph_engine(num_blocks, max_num_seqs, capture=True, max_batch=8)
    assert e.decode_graphs is not None and e.decode_graphs.graphs
    assert e.generate(PROMPTS, 10) == [expected(p, 10) for p in PROMPTS]
    assert e.step_records is not None and any(r.graph for r in e.step_records)
