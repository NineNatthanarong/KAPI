from __future__ import annotations

import os

from kapi import Kapi


def _write(d, name, text):
    with open(os.path.join(d, name), "w", encoding="utf-8") as f:
        f.write(text)


def test_persist_and_reopen(tmp_path, corpus_docs):
    idx = str(tmp_path / "idx")
    rag = Kapi(path=idx)
    rag.add(corpus_docs)
    top1 = rag.search("programming language readable syntax", k=1)[0].chunk.doc_id
    rag.close()

    rag2 = Kapi.open(idx)
    assert rag2.store.stats()["num_chunks"] == len(corpus_docs)
    top2 = rag2.search("programming language readable syntax", k=1)[0].chunk.doc_id
    assert top1 == top2 == "python"
    rag2.close()


def test_incremental_sync(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    d = str(docs)
    _write(d, "python.md", "# Python\nPython is a programming language.")
    _write(d, "recursion.md", "# Recursion\nA function that calls itself.")
    _write(d, "fox.txt", "The quick brown fox jumps over the lazy dog.")

    idx = str(tmp_path / "idx")
    rag = Kapi(path=idx)
    rep = rag.add(d)
    assert rep.added == 3

    # edit recursion, add a new doc, delete fox
    _write(d, "recursion.md",
           "# Recursion\nA function calls itself; tail recursion is optimized.")
    _write(d, "sorting.md", "# Sorting\nQuicksort partitions around a pivot element.")
    os.remove(os.path.join(d, "fox.txt"))

    rep2 = rag.sync(d)
    assert rep2.added == 1 and rep2.changed == 1 and rep2.unchanged == 1 and rep2.deleted == 1

    assert rag.store.stats()["num_chunks"] == 3
    assert {h.chunk.doc_id for h in rag.search("pivot partition quicksort", k=1)}
    assert any("tail recursion" in h.chunk.raw_text
               for h in rag.search("tail recursion optimized", k=2))
    # deleted doc is unreachable by its unique terms
    assert not rag.search("pangram lazy dog jumps", k=3) or all(
        "fox" not in h.chunk.doc_id for h in rag.search("pangram lazy dog jumps", k=3))
    rag.close()


def test_force_reindex_no_duplicate_chunks(corpus_docs):
    rag = Kapi()
    rag.add(corpus_docs)
    n1 = rag.engine.stats()["num_chunks"]
    rag.add(corpus_docs, force=True)          # re-index everything
    n2 = rag.engine.stats()["num_chunks"]
    assert n1 == n2 == len(corpus_docs)        # no duplicates
    # still exactly one hit per unique doc
    hits = rag.search("python programming language", k=10)
    ids = [h.chunk_id for h in hits]
    assert len(ids) == len(set(ids))
    rag.close()


def test_unchanged_docs_skipped(tmp_path, corpus_docs):
    idx = str(tmp_path / "idx")
    rag = Kapi(path=idx)
    rag.add(corpus_docs)
    rep = rag.add(corpus_docs)  # nothing changed
    assert rep.added == 0 and rep.changed == 0 and rep.unchanged == len(corpus_docs)
    rag.close()
