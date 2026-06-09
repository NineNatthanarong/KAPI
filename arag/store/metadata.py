"""SQLite metadata store — the source of truth for chunk text and the doc manifest.

The engine holds only the inverted index; canonical chunk text + per-doc fingerprints
live here (one ``meta.sqlite`` next to the index). This keeps the engine index small and
divergence-free, and gives transactional change-detection for incremental re-ingest.
``sqlite3`` is stdlib, so this adds no dependency.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Iterable, List, Optional

from .._types import Chunk, Hit

_SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    doc_id       TEXT PRIMARY KEY,
    source       TEXT,
    content_hash TEXT,
    mtime        REAL,
    n_chunks     INTEGER,
    indexed_at   REAL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT,
    ordinal      INTEGER,
    raw_text     TEXT,
    indexed_text TEXT,
    title        TEXT,
    section      TEXT,
    start_off    INTEGER,
    end_off      INTEGER,
    content_hash TEXT,
    meta_json    TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


@dataclass(frozen=True)
class DocFingerprint:
    doc_id: str
    content_hash: str
    mtime: Optional[float] = None
    source: Optional[str] = None


@dataclass(frozen=True)
class IngestDiff:
    added: List[str]
    changed: List[str]
    unchanged: List[str]
    deleted: List[str]


class MetadataStore:
    def __init__(self, path: Optional[str] = None) -> None:
        if path is None:
            self.db_path = ":memory:"
        else:
            os.makedirs(path, exist_ok=True)
            self.db_path = os.path.join(path, "meta.sqlite")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------ chunks
    def upsert_chunks(self, chunks: Iterable[Chunk]) -> None:
        rows = [
            (
                c.chunk_id, c.doc_id, c.ordinal, c.raw_text, c.indexed_text,
                c.title, c.section, c.start_offset, c.end_offset, c.content_hash,
                json.dumps(c.metadata, ensure_ascii=False),
            )
            for c in chunks
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO chunks "
            "(chunk_id, doc_id, ordinal, raw_text, indexed_text, title, section, "
            " start_off, end_off, content_hash, meta_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )

    def _row_to_chunk(self, row: sqlite3.Row) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"], doc_id=row["doc_id"], ordinal=row["ordinal"],
            raw_text=row["raw_text"], indexed_text=row["indexed_text"] or row["raw_text"],
            title=row["title"] or "", section=row["section"] or "",
            start_offset=row["start_off"] or 0, end_offset=row["end_off"] or 0,
            content_hash=row["content_hash"] or "",
            metadata=json.loads(row["meta_json"]) if row["meta_json"] else {},
        )

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        row = self.conn.execute(
            "SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)
        ).fetchone()
        return self._row_to_chunk(row) if row else None

    def hydrate(self, hits: List[Hit]) -> List[Hit]:
        """Attach full ``Chunk`` objects to hits (drops hits whose chunk is gone)."""
        if not hits:
            return hits
        ids = [h.chunk_id for h in hits]
        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", ids
        ).fetchall()
        by_id = {r["chunk_id"]: self._row_to_chunk(r) for r in rows}
        out: List[Hit] = []
        for h in hits:
            chunk = by_id.get(h.chunk_id)
            if chunk is not None:
                h.chunk = chunk
                out.append(h)
        return out

    def delete_chunks(self, chunk_ids: Iterable[str]) -> None:
        self.conn.executemany(
            "DELETE FROM chunks WHERE chunk_id=?", [(c,) for c in chunk_ids]
        )

    def chunks_for_doc(self, doc_id: str) -> List[str]:
        rows = self.conn.execute(
            "SELECT chunk_id FROM chunks WHERE doc_id=? ORDER BY ordinal", (doc_id,)
        ).fetchall()
        return [r["chunk_id"] for r in rows]

    def all_chunks(self) -> List[Chunk]:
        rows = self.conn.execute("SELECT * FROM chunks ORDER BY doc_id, ordinal").fetchall()
        return [self._row_to_chunk(r) for r in rows]

    # ------------------------------------------------------------------ docs
    def record_doc(self, doc_id: str, content_hash: str, mtime: Optional[float],
                   source: Optional[str], n_chunks: int, indexed_at: float) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO docs "
            "(doc_id, source, content_hash, mtime, n_chunks, indexed_at) "
            "VALUES (?,?,?,?,?,?)",
            (doc_id, source, content_hash, mtime, n_chunks, indexed_at),
        )

    def doc_state(self, doc_id: str) -> Optional[tuple[str, Optional[float]]]:
        row = self.conn.execute(
            "SELECT content_hash, mtime FROM docs WHERE doc_id=?", (doc_id,)
        ).fetchone()
        return (row["content_hash"], row["mtime"]) if row else None

    def all_doc_ids(self) -> List[str]:
        return [r["doc_id"] for r in self.conn.execute("SELECT doc_id FROM docs").fetchall()]

    def delete_doc(self, doc_id: str) -> None:
        self.conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        self.conn.execute("DELETE FROM docs WHERE doc_id=?", (doc_id,))

    def diff(self, fingerprints: Iterable[DocFingerprint]) -> IngestDiff:
        """Classify a full incoming corpus vs what's indexed.

        ``mtime`` is a cheap gate; ``content_hash`` is authoritative. A doc counts as
        unchanged only if its stored hash matches.
        """
        fps = list(fingerprints)
        incoming = {fp.doc_id for fp in fps}
        existing = set(self.all_doc_ids())
        added, changed, unchanged = [], [], []
        for fp in fps:
            state = self.doc_state(fp.doc_id)
            if state is None:
                added.append(fp.doc_id)
            elif state[0] != fp.content_hash:
                changed.append(fp.doc_id)
            else:
                unchanged.append(fp.doc_id)
        deleted = sorted(existing - incoming)
        return IngestDiff(added=added, changed=changed, unchanged=unchanged, deleted=deleted)

    # ------------------------------------------------------------------ misc
    def commit(self) -> None:
        self.conn.commit()

    def stats(self) -> dict:
        nd = self.conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        nc = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"num_docs": nd, "num_chunks": nc, "path": self.db_path}

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()
