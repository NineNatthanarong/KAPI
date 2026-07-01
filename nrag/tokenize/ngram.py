"""Character n-gram tokenizer (default trigrams).

Char n-grams give typo/morphology/multilingual robustness with no model. Used by the
SQLite/bm25s engines and in tests; the Tantivy engine uses its native
``Tokenizer.ngram`` but we keep a matching Python implementation for parity.
"""

from __future__ import annotations

import re
import unicodedata

_WS_RE = re.compile(r"\s+")


def _ascii_fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


class CharNgramTokenizer:
    def __init__(
        self,
        n: int = 3,
        *,
        lowercase: bool = True,
        ascii_fold: bool = True,
        collapse_ws: bool = True,
    ) -> None:
        self.n = n
        self.lowercase = lowercase
        self.ascii_fold = ascii_fold
        self.collapse_ws = collapse_ws

    def __call__(self, text: str) -> list[str]:
        if self.lowercase:
            text = text.lower()
        if self.ascii_fold:
            text = _ascii_fold(text)
        if self.collapse_ws:
            text = _WS_RE.sub(" ", text)
        n = self.n
        if len(text) < n:
            return [text] if text.strip() else []
        # sliding window over the whole (space-collapsed) string, matching Tantivy's
        # NgramTokenizer which spans token boundaries.
        return [text[i : i + n] for i in range(len(text) - n + 1)]
