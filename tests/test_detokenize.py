"""Spec for incremental detokenisation (M6): the streamed pieces must add up to exactly
the text of the whole output, and never contain a half-decoded character."""

from hypothesis import given, settings
from hypothesis import strategies as st
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from tinyserve.detokenize import REPLACEMENT, IncrementalDetokenizer


def stream(detok: IncrementalDetokenizer, ids: list[int], sizes: list[int]) -> list[str]:
    """Push ``ids`` in groups of the given sizes (cycled), then flush."""
    pieces, i, k = [], 0, 0
    while i < len(ids):
        n = sizes[k % len(sizes)]
        pieces.append(detok.push(ids[i : i + n]))
        i, k = i + n, k + 1
    pieces.append(detok.flush())
    return pieces


@settings(max_examples=300, deadline=None)
@given(
    text=st.text(min_size=0, max_size=40),
    cuts=st.lists(st.integers(1, 5), min_size=1, max_size=10),
    sizes=st.lists(st.integers(1, 3), min_size=1, max_size=5),
)
def test_byte_tokens_stream_to_the_exact_text(text: str, cuts: list[int], sizes: list[int]) -> None:
    # A byte-level "tokenizer": each token is a run of 1-5 bytes, cut anywhere, so
    # tokens regularly end in the middle of a multi-byte UTF-8 character.
    data = text.encode()
    chunks, i, k = [], 0, 0
    while i < len(data):
        chunks.append(data[i : i + cuts[k % len(cuts)]])
        i, k = i + cuts[k % len(cuts)], k + 1

    def decode(ids: list[int]) -> str:
        return b"".join(chunks[j] for j in ids).decode("utf-8", errors="replace")

    pieces = stream(IncrementalDetokenizer(decode), list(range(len(chunks))), sizes)
    assert "".join(pieces) == text
    if REPLACEMENT not in text:
        assert all(REPLACEMENT not in p for p in pieces)


def byte_level_bpe() -> Tokenizer:
    """A small real byte-level BPE (the kind Qwen3 uses), trained on a few lines."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = ["hello world, the capital of France is Paris"] * 20
    alphabet = pre_tokenizers.ByteLevel.alphabet()  # all 256 bytes, so any text encodes
    trainer = trainers.BpeTrainer(vocab_size=300, initial_alphabet=alphabet, show_progress=False)
    tok.train_from_iterator(corpus, trainer)
    return tok


def test_real_byte_level_bpe_one_token_at_a_time() -> None:
    tok = byte_level_bpe()
    text = "héllo wörld 🙂 你好, the capital of France is Paris. naïve café 🇫🇷"
    ids = tok.encode(text).ids
    # The tokenizer never saw these characters, so they are split into single bytes.
    assert any(REPLACEMENT in tok.decode([i]) for i in ids)
    pieces = stream(IncrementalDetokenizer(tok.decode), ids, [1])
    assert "".join(pieces) == tok.decode(ids) == text
    assert all(REPLACEMENT not in p for p in pieces)


def test_flush_emits_an_unfinished_character() -> None:
    chunks = [b"a", "é".encode()[:1]]  # the stream stops mid-character

    def decode(ids: list[int]) -> str:
        return b"".join(chunks[j] for j in ids).decode("utf-8", errors="replace")

    d = IncrementalDetokenizer(decode)
    assert d.push([0]) == "a"
    assert d.push([1]) == ""  # held back
    assert d.flush() == REPLACEMENT
    assert d.flush() == ""


def test_window_stays_short() -> None:
    # Each push decodes only the last emitted piece plus the new tokens.
    seen: list[int] = []

    def decode(ids: list[int]) -> str:
        seen.append(len(ids))
        return "x" * len(ids)

    d = IncrementalDetokenizer(decode)
    for _ in range(100):
        d.push([0])
    assert max(seen) <= 2
