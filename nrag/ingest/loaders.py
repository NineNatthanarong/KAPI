"""Document loaders — turn files/dirs/globs/strings into a lazy stream of ``Document``.

The base install handles ``.txt``/``.md`` and ``.html`` (via the stdlib HTML parser).
PDF (text-layer only) and fast HTML are optional extras. Everything is generator-based
so large corpora stay flat in memory.
"""

from __future__ import annotations

import glob as _glob
import os
from html.parser import HTMLParser
from typing import Iterable, Iterator, Optional, Protocol

from .._types import Document

MARKDOWN_EXTS = {".md", ".markdown", ".mdown"}
TEXT_EXTS = {".txt", ".text", ".rst", ".log", ""}
HTML_EXTS = {".html", ".htm"}
PDF_EXTS = {".pdf"}


class Loader(Protocol):
    def can_load(self, path: str) -> bool: ...
    def load(self, path: str) -> Iterable[Document]: ...


def _doc(path: str, text: str, content_type: str) -> Document:
    try:
        mtime: Optional[float] = os.path.getmtime(path)
    except OSError:
        mtime = None
    return Document(
        doc_id=os.path.abspath(path),
        text=text,
        source=path,
        mtime=mtime,
        metadata={"content_type": content_type, "source": path},
    )


class TextLoader:
    def can_load(self, path: str) -> bool:
        return os.path.splitext(path)[1].lower() in TEXT_EXTS

    def load(self, path: str) -> Iterable[Document]:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            yield _doc(path, fh.read(), "text")


class MarkdownLoader:
    def can_load(self, path: str) -> bool:
        return os.path.splitext(path)[1].lower() in MARKDOWN_EXTS

    def load(self, path: str) -> Iterable[Document]:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            yield _doc(path, fh.read(), "markdown")


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            # surface headings as markdown so the chunker's structure pass can use them
            self._parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in ("p", "br", "div", "li", "tr"):
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


class HTMLLoader:
    """Stdlib HTML -> text (zero extra deps). Headings become markdown ``#`` markers."""

    def can_load(self, path: str) -> bool:
        return os.path.splitext(path)[1].lower() in HTML_EXTS

    def load(self, path: str) -> Iterable[Document]:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            ex = _TextExtractor()
            ex.feed(fh.read())
            yield _doc(path, ex.text(), "markdown")


class PDFTextLoader:
    """Text-layer PDF extraction via the optional ``pypdf`` extra (no OCR)."""

    def can_load(self, path: str) -> bool:
        return os.path.splitext(path)[1].lower() in PDF_EXTS

    def load(self, path: str) -> Iterable[Document]:
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "PDF support requires the 'pdf' extra: pip install nrag[pdf]"
            ) from exc
        reader = PdfReader(path)
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        yield _doc(path, text, "text")


DEFAULT_LOADERS: tuple[Loader, ...] = (
    MarkdownLoader(),
    HTMLLoader(),
    PDFTextLoader(),
    TextLoader(),  # last: it accepts the broadest set (incl. no-extension)
)


def _iter_files(paths: Iterable[str]) -> Iterator[str]:
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for name in sorted(files):
                    yield os.path.join(root, name)
        elif any(ch in p for ch in "*?[]"):
            yield from sorted(_glob.glob(p, recursive=True))
        elif os.path.isfile(p):
            yield p


def load_paths(
    source: str | Iterable[str],
    *,
    loaders: Optional[Iterable[Loader]] = None,
) -> Iterator[Document]:
    """Yield ``Document``s from a path, dir, glob, or iterable of those.

    Files with no matching loader are skipped silently (so mixing a few binaries into
    a docs folder doesn't crash ingestion).
    """
    paths = [source] if isinstance(source, str) else list(source)
    loaders = tuple(loaders) if loaders is not None else DEFAULT_LOADERS
    for path in _iter_files(paths):
        for loader in loaders:
            if loader.can_load(path):
                try:
                    yield from loader.load(path)
                except RuntimeError:
                    raise
                except Exception:
                    # a single unreadable file shouldn't abort a large ingest
                    pass
                break


def documents_from_texts(texts: Iterable[str], *, prefix: str = "doc") -> Iterator[Document]:
    """Build in-memory ``Document``s from raw strings (handy for tests/BEIR)."""
    for i, text in enumerate(texts):
        yield Document(doc_id=f"{prefix}-{i}", text=text, source=f"{prefix}-{i}",
                       metadata={"content_type": "text"})
