"""Recursive, structure-aware chunking.

Chunking is the single biggest quality lever in RAG and needs no embedding model. This
splitter is *span-based*: every chunk is an exact slice of the source text
(``doc.text[start:end] == chunk.raw_text``), so offsets are always reconstructable even
with overlap. It respects markdown heading structure (carrying a section path + title
for the title-boost signal) and prefers sentence/paragraph boundaries, falling back to
word boundaries only for over-long segments.

The emitted ``Chunk`` has ``indexed_text == raw_text``; the contextual-indexing step
(``augment/contextual.py``) is the single place that diverges them by prepending a blurb.
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from typing import Callable, List, Tuple

from .._types import Chunk, Document

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.M)
# cut after a blank line, a newline, or sentence-ending punctuation followed by space
_SEGMENT_RE = re.compile(r"\n{2,}|\n|(?<=[.!?;])\s+")

Span = Tuple[int, int]


@dataclass(frozen=True)
class ChunkConfig:
    target_tokens: int = 480       # ~400-512 band
    overlap_tokens: int = 64       # ~12-15% overlap
    min_tokens: int = 64           # merge a tiny trailing chunk back
    token_counter: str = "regex"   # "regex" (dep-free) | "tiktoken" (optional)
    respect_structure: bool = True


@dataclass(frozen=True)
class _Section:
    start: int
    end: int
    title: str
    path: str


def _make_counter(kind: str) -> Callable[[str], int]:
    if kind == "tiktoken":
        try:
            import tiktoken  # type: ignore

            enc = tiktoken.get_encoding("cl100k_base")
            return lambda s: len(enc.encode(s)) if s else 0
        except Exception:
            warnings.warn(
                "tiktoken not available; falling back to regex word-count token estimate.",
                RuntimeWarning,
                stacklevel=2,
            )
    return lambda s: len(_WORD_RE.findall(s))


def _sections(text: str, is_markdown: bool) -> List[_Section]:
    if not is_markdown:
        return [_Section(0, len(text), "", "")]
    headings = list(_HEADING_RE.finditer(text))
    if not headings:
        return [_Section(0, len(text), "", "")]
    sections: List[_Section] = []
    if text[: headings[0].start()].strip():
        sections.append(_Section(0, headings[0].start(), "", ""))
    stack: List[Tuple[int, str]] = []
    for i, m in enumerate(headings):
        level = len(m.group(1))
        title = m.group(2).strip()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        path = " > ".join(t for _, t in stack)
        sections.append(_Section(m.start(), end, title, path))
    return sections


def _segment_spans(text: str, start: int, end: int) -> List[Span]:
    """Cut [start,end) into contiguous segment spans at sentence/paragraph boundaries."""
    sub = text[start:end]
    spans: List[Span] = []
    last = 0
    for m in _SEGMENT_RE.finditer(sub):
        cut = m.end()
        if cut > last:
            spans.append((start + last, start + cut))
            last = cut
    if last < len(sub):
        spans.append((start + last, end))
    return spans


def _word_spans(text: str, start: int, end: int) -> List[Span]:
    return [(start + m.start(), start + m.end()) for m in _WORD_RE.finditer(text[start:end])]


class Chunker:
    def __init__(self, config: ChunkConfig | None = None) -> None:
        self.config = config or ChunkConfig()
        self._count = _make_counter(self.config.token_counter)

    # ------------------------------------------------------------------ public
    def chunk(self, doc: Document) -> List[Chunk]:
        cfg = self.config
        is_md = (
            cfg.respect_structure
            and doc.metadata.get("content_type") == "markdown"
        )
        doc_title = doc.metadata.get("title") or self._title_from_source(doc.source)

        chunks: List[Chunk] = []
        ordinal = 0
        for sec in _sections(doc.text, is_md):
            for (s, e) in self._chunk_section(doc.text, sec):
                raw = doc.text[s:e]
                if not raw.strip():
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=Chunk.make_id(doc.doc_id, ordinal),
                        doc_id=doc.doc_id,
                        ordinal=ordinal,
                        raw_text=raw,
                        indexed_text=raw,
                        title=sec.title or doc_title,
                        section=sec.path,
                        start_offset=s,
                        end_offset=e,
                        metadata=dict(doc.metadata),
                    )
                )
                ordinal += 1
        return chunks

    # ------------------------------------------------------------------ internals
    def _chunk_section(self, text: str, sec: _Section) -> List[Span]:
        cfg = self.config
        count = self._count
        # atomize into segments, sub-splitting over-long segments into words
        atoms: List[Span] = []
        for (s, e) in _segment_spans(text, sec.start, sec.end):
            if count(text[s:e]) > cfg.target_tokens:
                atoms.extend(_word_spans(text, s, e))
            else:
                atoms.append((s, e))
        atoms = [a for a in atoms if text[a[0]:a[1]].strip()]
        if not atoms:
            return []
        return self._pack(text, atoms)

    def _pack(self, text: str, atoms: List[Span]) -> List[Span]:
        cfg = self.config
        count = self._count
        tok = [count(text[s:e]) for (s, e) in atoms]

        chunks: List[Span] = []
        cur: List[int] = []          # indices into atoms
        cur_tok = 0
        i = 0
        n = len(atoms)
        while i < n:
            if cur and cur_tok + tok[i] > cfg.target_tokens:
                chunks.append((atoms[cur[0]][0], atoms[cur[-1]][1]))
                # carry trailing atoms (~overlap tokens) into the next chunk
                keep: List[int] = []
                ov = 0
                j = len(cur) - 1
                while j >= 0 and ov < cfg.overlap_tokens:
                    keep.insert(0, cur[j])
                    ov += tok[cur[j]]
                    j -= 1
                cur = keep
                cur_tok = sum(tok[k] for k in cur)
            cur.append(i)
            cur_tok += tok[i]
            i += 1
        if cur:
            chunks.append((atoms[cur[0]][0], atoms[cur[-1]][1]))

        # merge a tiny trailing chunk into its predecessor
        if len(chunks) >= 2 and count(text[chunks[-1][0]:chunks[-1][1]]) < cfg.min_tokens:
            chunks[-2] = (chunks[-2][0], chunks[-1][1])
            chunks.pop()
        return chunks

    @staticmethod
    def _title_from_source(source: str | None) -> str:
        if not source:
            return ""
        base = os.path.basename(source)
        stem = os.path.splitext(base)[0]
        return stem.replace("_", " ").replace("-", " ").strip()
