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


class StopMatcher:
    """Stop strings over streamed text (M9): cut the answer before the first one.

        Text is pushed as it is decoded. ``push`` returns what is safe to send: everything
        except a tail that could still turn into a stop string (a suffix of the text that is a
        prefix of one), which is held back until the next push decides. When a stop string
        appears, ``push`` returns the text before it, drops the stop string and everything
        after it, and sets ``stopped``. So with stop ``"\\n\\n"`` and pushes ``"a\\n"``,
        ``"\\nb"``: the first returns ``"a"`` (the ``"\\n"`` may start a stop), the second
        ``""`` and stops. OpenAI's semantics: the stop string itself is not returned. If two
    stop strings overlap, the one that is complete first wins, as when checking after every
    token, but whatever pieces the text arrives in: stops ``"a"`` and ``"\\naa"`` cut
    ``"\\naa"`` to ``"\\n"``.
    """

    def __init__(self, stops: list[str]) -> None:
        if any(not s for s in stops):
            raise ValueError("stop strings must not be empty")
        self.stops = stops
        self.held = ""  # pushed but not returned yet
        self.stopped = False

    def push(self, text: str) -> str:
        if self.stopped:
            return ""
        buf = self.held + text
        found = [(i + len(s), i) for s in self.stops if (i := buf.find(s)) >= 0]
        if found:  # the stop string completed first (then the longest) wins
            self.held, self.stopped = "", True
            return buf[: min(found)[1]]
        keep = 0  # the longest tail of buf that some stop string starts with
        for k in range(min(len(buf), max((len(s) for s in self.stops), default=1) - 1), 0, -1):
            if any(s.startswith(buf[-k:]) for s in self.stops):
                keep = k
                break
        self.held = buf[len(buf) - keep :]
        return buf[: len(buf) - keep]

    def flush(self) -> str:
        """At the end of the answer: the held-back tail was not a stop string after all."""
        out, self.held = self.held, ""
        return "" if self.stopped else out
