"""Lightweight result/value types returned by the public API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ._types import Hit


class CostGuardError(RuntimeError):
    """Raised when an estimated index-time LLM cost exceeds the configured guard."""


@dataclass(frozen=True)
class Citation:
    marker: str          # e.g. "[1]"
    source: Optional[str]
    chunk_id: str
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    citations: List[Citation] = field(default_factory=list)
    all_context: List[Citation] = field(default_factory=list)


@dataclass(frozen=True)
class QueryResult:
    question: str
    hits: List[Hit]
    answer: Optional[str] = None
    citations: List[Citation] = field(default_factory=list)

    def __str__(self) -> str:  # convenient for print(rag.query(...))
        return self.answer if self.answer is not None else "\n\n".join(
            f"[{i}] {h.text[:200]}" for i, h in enumerate(self.hits, 1)
        )


@dataclass(frozen=True)
class AddReport:
    num_docs: int
    num_chunks: int
    added: int = 0
    changed: int = 0
    unchanged: int = 0
    deleted: int = 0
    contextualized: int = 0

    def __str__(self) -> str:
        return (f"AddReport(docs={self.num_docs}, chunks={self.num_chunks}, "
                f"added={self.added}, changed={self.changed}, unchanged={self.unchanged}, "
                f"deleted={self.deleted})")


@dataclass(frozen=True)
class CostEstimate:
    n_chunks: int
    input_tokens: int
    output_tokens: int
    est_usd: float
    model: str = "unknown"

    def __str__(self) -> str:
        return (f"CostEstimate(chunks={self.n_chunks}, in={self.input_tokens}, "
                f"out={self.output_tokens}, ~${self.est_usd:.4f} on {self.model})")
