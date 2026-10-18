"""Incremental detokenisation: turn a stream of token ids into a stream of text (M6).

Decoding each new token on its own is wrong in two ways. A byte-level BPE token can hold
part of a UTF-8 character (an emoji is 4 bytes and may span 2 or 3 tokens), so the piece
decodes to the replacement character U+FFFD. And some decoders change a token's text
depending on what comes before it (leading spaces). So each step decodes a short window
that starts at a known character boundary, and emits only the text that is new:

    ids:  [ ... done ... | prefix_offset ... read_offset | new ids ]
    prefix = decode(ids[prefix_offset:read_offset])    text already sent
    full   = decode(ids[prefix_offset:])
    delta  = full[len(prefix):]                         unless full ends in U+FFFD

If ``full`` ends in U+FFFD the last character is incomplete: send nothing and wait for
the next token. ``read_offset`` only moves past complete text, so the window always
starts on a character boundary. Each push costs O(window), not O(output) (the same
scheme as Hugging Face TGI and vLLM).
"""

from collections.abc import Callable

REPLACEMENT = "�"


class IncrementalDetokenizer:
    def __init__(self, decode: Callable[[list[int]], str]) -> None:
        self.decode = decode
        self.ids: list[int] = []
        self.prefix_offset = 0
        self.read_offset = 0

    def push(self, new_ids: list[int]) -> str:
        """Add tokens; return the text they complete (possibly "")."""
        self.ids.extend(new_ids)
        prefix = self.decode(self.ids[self.prefix_offset : self.read_offset])
        full = self.decode(self.ids[self.prefix_offset :])
        if full.endswith(REPLACEMENT) or len(full) <= len(prefix):
            return ""  # an unfinished character (or nothing new): wait for more tokens
        self.prefix_offset, self.read_offset = self.read_offset, len(self.ids)
        return full[len(prefix) :]

    def flush(self) -> str:
        """At the end of the stream: whatever is still held back, even if incomplete."""
        prefix = self.decode(self.ids[self.prefix_offset : self.read_offset])
        full = self.decode(self.ids[self.prefix_offset :])
        self.prefix_offset = self.read_offset = len(self.ids)
        return full[len(prefix) :]
