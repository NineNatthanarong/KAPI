"""Rank fusion for combining multiple ranked lists.

RRF (Reciprocal Rank Fusion, Cormack et al. 2009) is the zero-tuning default: it fuses
by *rank*, sidestepping the score-normalization problem entirely. Weighted convex
combination (Bruch et al. 2023) is available when you have a few labeled queries to tune.
Used by the portable single-field engines (SQLite/bm25s); the Tantivy engine fuses
fields internally, so its results pass through unchanged.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from .._types import Hit


def rrf(rank_lists: Sequence[Sequence[Hit]], k: int = 60) -> List[Hit]:
    """Reciprocal Rank Fusion. score(d) = sum_i 1 / (k + rank_i(d))."""
    scores: Dict[str, float] = {}
    best: Dict[str, Hit] = {}
    for hits in rank_lists:
        for rank, h in enumerate(hits, start=1):
            r = h.rank or rank
            scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + 1.0 / (k + r)
            if h.chunk_id not in best:
                best[h.chunk_id] = h
    return _ranked(scores, best)


def _minmax(values: List[float]) -> Dict[int, float]:
    if not values:
        return {}
    lo, hi = min(values), max(values)
    span = hi - lo
    if span <= 0:
        return {i: 1.0 for i in range(len(values))}
    return {i: (v - lo) / span for i, v in enumerate(values)}


def convex(rank_lists: Sequence[Sequence[Hit]], weights: Sequence[float],
           normalize: str = "minmax") -> List[Hit]:
    """Weighted convex combination of min-max-normalized per-list scores."""
    scores: Dict[str, float] = {}
    best: Dict[str, Hit] = {}
    for li, hits in enumerate(rank_lists):
        w = weights[li] if li < len(weights) else 1.0
        norm = _minmax([h.score for h in hits]) if normalize == "minmax" else None
        for i, h in enumerate(hits):
            s = norm[i] if norm is not None else h.score
            scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + w * s
            if h.chunk_id not in best:
                best[h.chunk_id] = h
    return _ranked(scores, best)


def _ranked(scores: Dict[str, float], best: Dict[str, Hit]) -> List[Hit]:
    out: List[Hit] = []
    for rank, (cid, sc) in enumerate(
        sorted(scores.items(), key=lambda kv: kv[1], reverse=True), start=1
    ):
        h = best[cid]
        out.append(Hit(chunk_id=cid, score=sc, rank=rank, chunk=h.chunk, signal="fused"))
    return out


def fuse(rank_lists: Sequence[Sequence[Hit]], method: str = "rrf", **kw) -> List[Hit]:
    if method == "convex":
        weights = kw.get("weights") or [1.0] * len(rank_lists)
        return convex(rank_lists, weights, normalize=kw.get("normalize", "minmax"))
    return rrf(rank_lists, k=kw.get("k", 60))
