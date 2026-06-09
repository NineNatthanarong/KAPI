from __future__ import annotations

from kapi._types import Chunk, content_hash
from kapi.store.metadata import DocFingerprint, MetadataStore


def _chunk(cid, doc_id="d1", text="hello world", **kw):
    return Chunk(chunk_id=cid, doc_id=doc_id, ordinal=0, raw_text=text,
                 indexed_text=text, **kw)


def test_upsert_and_hydrate_roundtrip():
    store = MetadataStore(None)
    c = _chunk("d1::0", title="T", section="A > B", metadata={"k": "v"})
    store.upsert_chunks([c])
    from kapi._types import Hit

    hits = store.hydrate([Hit(chunk_id="d1::0", score=1.0)])
    assert len(hits) == 1
    got = hits[0].chunk
    assert got.title == "T" and got.section == "A > B" and got.metadata["k"] == "v"


def test_hydrate_drops_missing():
    from kapi._types import Hit

    store = MetadataStore(None)
    hits = store.hydrate([Hit(chunk_id="ghost", score=1.0)])
    assert hits == []


def test_diff_classifies_added_changed_unchanged_deleted():
    store = MetadataStore(None)
    # seed two docs
    store.record_doc("a", content_hash("A"), 1.0, "a", 1, 0.0)
    store.record_doc("b", content_hash("B"), 1.0, "b", 1, 0.0)
    store.commit()

    fps = [
        DocFingerprint("a", content_hash("A")),       # unchanged
        DocFingerprint("b", content_hash("B-EDITED")),  # changed
        DocFingerprint("c", content_hash("C")),       # added
    ]  # "a","b" exist; "c" new; nothing maps to old set except a,b -> none deleted here
    diff = store.diff(fps)
    assert diff.added == ["c"]
    assert diff.changed == ["b"]
    assert diff.unchanged == ["a"]
    assert diff.deleted == []  # all existing docs present in incoming


def test_diff_detects_deletion():
    store = MetadataStore(None)
    store.record_doc("a", content_hash("A"), 1.0, "a", 1, 0.0)
    store.record_doc("b", content_hash("B"), 1.0, "b", 1, 0.0)
    store.commit()
    diff = store.diff([DocFingerprint("a", content_hash("A"))])
    assert diff.deleted == ["b"]
