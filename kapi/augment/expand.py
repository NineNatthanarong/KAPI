"""Query expansion — the one online LLM call that fixes BM25's vocabulary mismatch.

Two modes, auto-routed:
  * **query2doc** (Wang et al., EMNLP 2023): the LLM writes a short pseudo-document for
    the query; we concatenate it to the query. For sparse/BM25 retrieval the original
    query is repeated ~5x so its real terms aren't drowned by the longer pseudo-doc.
  * **cot_keywords** (Jagerman et al., 2023): the LLM lists related keywords via
    chain-of-thought; better for short keyword queries.

Both produce a single BM25-ready string. Per-query caching makes repeats free. Any LLM
error degrades gracefully to the raw query — retrieval never breaks.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Config

_QUERY2DOC_PROMPT = (
    "Write a short, factual passage (2-4 sentences) that would answer the following "
    "search query. Respond with only the passage.\n\nQuery: {q}\n\nPassage:"
)
_COT_PROMPT = (
    "A user typed this search query: {q}\n\n"
    "Think briefly about what a relevant document would contain, then on the FINAL line "
    "output the most useful search terms in the exact form:\nKeywords: term1, term2, term3, ..."
)


@dataclass(frozen=True)
class ExpandedQuery:
    raw: str
    assembled: str
    mode: str
    addition: str = ""


class QueryExpander:
    def __init__(self, llm, config: Config) -> None:
        self.llm = llm
        self.config = config
        self._cache: dict[tuple[str, str], ExpandedQuery] = {}

    def expand(self, query: str) -> ExpandedQuery:
        mode = self._resolve_mode(query)
        key = (mode, query)
        if key in self._cache:
            return self._cache[key]

        try:
            if mode == "cot_keywords":
                addition = self._cot_keywords(query)
            else:
                addition = self._query2doc(query)
        except Exception:
            addition = ""

        if addition:
            repeated = " ".join([query] * max(1, self.config.expand_query_repeat))
            assembled = f"{repeated} {addition}".strip()
            result = ExpandedQuery(query, assembled, mode, addition)
        else:
            result = ExpandedQuery(query, query, "none")
        self._cache[key] = result
        return result

    # ------------------------------------------------------------------ modes
    def _resolve_mode(self, query: str) -> str:
        mode = self.config.expand_mode
        if mode != "auto":
            return mode
        n_words = len(query.split())
        return "cot_keywords" if n_words <= self.config.expand_auto_short_query_words else "query2doc"

    def _query2doc(self, query: str) -> str:
        text = self.llm.complete(
            _QUERY2DOC_PROMPT.format(q=query),
            max_tokens=self.config.expand_max_tokens,
            temperature=0.0,
        )
        return " ".join((text or "").split())

    def _cot_keywords(self, query: str) -> str:
        text = self.llm.complete(
            _COT_PROMPT.format(q=query),
            max_tokens=self.config.expand_max_tokens,
            temperature=0.0,
        )
        return self._parse_keywords(text or "")

    @staticmethod
    def _parse_keywords(text: str) -> str:
        kw_line = ""
        for line in reversed(text.strip().splitlines()):
            low = line.lower()
            if "keyword" in low and ":" in line:
                kw_line = line.split(":", 1)[1]
                break
        if not kw_line:
            # no explicit Keywords: line — fall back to the last non-empty line
            lines = [ln for ln in text.strip().splitlines() if ln.strip()]
            kw_line = lines[-1] if lines else ""
        terms = [t.strip(" .;-") for t in kw_line.replace(";", ",").split(",")]
        return " ".join(t for t in terms if t)
