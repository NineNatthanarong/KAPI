"""ARAG — fast, local, zero-setup RAG with no embedding model.

Public surface::

    from arag import ARAG
    from arag.llm import OpenAICompatLLM, CallableLLM

The :class:`ARAG` facade orchestrates lexical retrieval (BM25 over words + char
n-grams + title, fused) and optional LLM augmentation (contextual indexing + query
expansion + grounded generation). Everything works with no LLM (pure lexical) and
with no setup beyond ``pip install``.
"""

from __future__ import annotations

from ._types import (
    Chunk,
    Document,
    EngineConfig,
    FieldWeights,
    Hit,
    MetaFilter,
    attach_context,
    content_hash,
)
from .config import Config
from .results import AddReport, Answer, Citation, CostEstimate, CostGuardError, QueryResult

__version__ = "0.1.0"

__all__ = [
    "ARAG",
    "Config",
    "Document",
    "Chunk",
    "Hit",
    "FieldWeights",
    "MetaFilter",
    "EngineConfig",
    "QueryResult",
    "Answer",
    "Citation",
    "AddReport",
    "CostEstimate",
    "CostGuardError",
    "attach_context",
    "content_hash",
    "__version__",
]


def __getattr__(name: str):
    # Lazy import of the heavy facade so `import arag` stays cheap and so the
    # package imports cleanly even before optional engine deps are touched.
    if name == "ARAG":
        from .app import ARAG

        return ARAG
    raise AttributeError(f"module 'arag' has no attribute {name!r}")
