"""Grounded answer generation with inline citations.

Packs the top-k retrieved chunks into the prompt up to a token budget, instructs the
LLM to answer only from that context and cite sources as ``[n]`` markers that map back
to the packed chunks, then reports the citations actually used. With no LLM this module
is never reached (the facade returns hits-only results).
"""

from __future__ import annotations

import re
from typing import Iterator, List, Tuple

from .._types import Hit
from ..config import Config
from ..llm.base import estimate_tokens, stream_complete
from ..results import Answer, Citation

_SYSTEM = (
    "Answer the question using ONLY the provided context. Cite supporting sources inline "
    "with [n] markers that match the numbered context blocks. If the answer is not in the "
    "context, say you don't know. Be concise and accurate."
)
_USER = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer (cite sources as [n]):"
_MARKER_RE = re.compile(r"\[(\d+)\]")


class Generator:
    def __init__(self, llm, config: Config) -> None:
        self.llm = llm
        self.config = config

    def pack_context(self, hits: List[Hit]) -> Tuple[str, List[Citation]]:
        budget = self.config.context_token_budget
        used = 0
        blocks: List[str] = []
        cites: List[Citation] = []
        for i, h in enumerate(hits, start=1):
            text = h.text
            t = estimate_tokens(self.llm, text)
            if blocks and used + t > budget:
                break
            src = h.source or h.chunk_id
            blocks.append(f"[{i}] (source: {src})\n{text}")
            cites.append(Citation(marker=f"[{i}]", source=h.source,
                                  chunk_id=h.chunk_id, score=h.score))
            used += t
        return "\n\n".join(blocks), cites

    def _prompt(self, question: str, hits: List[Hit]) -> Tuple[str, List[Citation]]:
        context, cites = self.pack_context(hits)
        return _USER.format(context=context, question=question), cites

    def answer(self, question: str, hits: List[Hit]) -> Answer:
        if not hits:
            return Answer(text="I don't have any indexed context to answer that.",
                          citations=[], all_context=[])
        prompt, cites = self._prompt(question, hits)
        text = self.llm.complete(
            prompt, max_tokens=self.config.answer_max_tokens, temperature=0.0,
            system=_SYSTEM,
        ) or ""
        used = self._used_citations(text, cites)
        return Answer(text=text.strip(), citations=used, all_context=cites)

    def stream(self, question: str, hits: List[Hit]) -> Iterator[str]:
        if not hits:
            yield "I don't have any indexed context to answer that."
            return
        prompt, _cites = self._prompt(question, hits)
        yield from stream_complete(
            self.llm, prompt, max_tokens=self.config.answer_max_tokens,
            temperature=0.0, system=_SYSTEM,
        )

    @staticmethod
    def _used_citations(text: str, cites: List[Citation]) -> List[Citation]:
        present = {int(m) for m in _MARKER_RE.findall(text)}
        return [c for i, c in enumerate(cites, start=1) if i in present]
