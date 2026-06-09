"""The swappable index-engine contract.

An ``IndexEngine`` is a persistent (ideally incremental) lexical index over chunks.
``chunk_id`` is the stable delete/update key; ``commit()`` is the durability + visibility
boundary. The retrieve layer branches on the capability flags: engines that score
multiple fields in one query (Tantivy) return already-fused hits; single-field engines
(SQLite/bm25s) return per-signal hits that the retrieve layer fuses with RRF.

Engines return lightweight ``Hit``s carrying ``chunk_id`` + ``score`` (+ which signal);
the retrieve layer hydrates them with full chunk text from the metadata store.
"""

from __future__ import annotations

from typing import Iterable, Optional, Protocol, runtime_checkable

from .._types import EngineConfig, FieldWeights, Hit, MetaFilter


@runtime_checkable
class IndexEngine(Protocol):
    @classmethod
    def open(
        cls,
        path: Optional[str] = None,
        *,
        create: bool = True,
        config: Optional[EngineConfig] = None,
    ) -> "IndexEngine": ...

    def close(self) -> None: ...

    # ---- mutation (staged until commit) ----
    def add(self, chunks: Iterable) -> int: ...
    def update(self, chunk_id: str, chunk) -> None: ...
    def delete(self, chunk_ids: Iterable[str]) -> int: ...
    def delete_doc(self, doc_id: str) -> int: ...
    def commit(self) -> None: ...

    # ---- read ----
    def search(
        self,
        query: str,
        *,
        k: int = 10,
        field_weights: FieldWeights = FieldWeights(),
        fuzzy: bool = False,
        filter: Optional[MetaFilter] = None,
    ) -> list[Hit]: ...

    # ---- capabilities / introspection ----
    @property
    def supports_incremental(self) -> bool: ...
    @property
    def supports_multifield(self) -> bool: ...
    @property
    def supports_prefilter(self) -> bool: ...
    def stats(self) -> dict: ...


def open_engine(
    name: str,
    path: Optional[str] = None,
    *,
    create: bool = True,
    config: Optional[EngineConfig] = None,
) -> IndexEngine:
    """Factory: resolve an engine by name. Keeps optional engines lazily imported."""
    name = (name or "tantivy").lower()
    if name == "tantivy":
        from .tantivy_engine import TantivyEngine

        return TantivyEngine.open(path, create=create, config=config)
    if name == "sqlite":
        from .sqlite_engine import SQLiteFTS5Engine

        return SQLiteFTS5Engine.open(path, create=create, config=config)
    if name == "bm25s":
        from .bm25s_engine import InMemoryBM25Engine

        return InMemoryBM25Engine.open(path, create=create, config=config)
    raise ValueError(f"unknown engine {name!r} (expected: tantivy | sqlite | bm25s)")
