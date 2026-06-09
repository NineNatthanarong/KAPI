from __future__ import annotations

import math

from arag.eval.ir_metrics import evaluate_run, ndcg_at_k, recall_at_k, reciprocal_rank


def test_perfect_ranking():
    rel = {"a": 1, "b": 1}
    ranked = ["a", "b", "c"]
    assert recall_at_k(ranked, rel, 2) == 1.0
    assert ndcg_at_k(ranked, rel, 2) == 1.0
    assert reciprocal_rank(ranked, rel) == 1.0


def test_reciprocal_rank_second():
    assert reciprocal_rank(["x", "a"], {"a": 1}) == 0.5


def test_ndcg_known_value():
    # one relevant doc at rank 2 -> DCG = 1/log2(3); IDCG = 1/log2(2)=1
    rel = {"a": 1}
    val = ndcg_at_k(["x", "a", "y"], rel, 3)
    assert math.isclose(val, (1 / math.log2(3)), rel_tol=1e-9)


def test_evaluate_run_aggregates():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    run = {"q1": {"a": 3.0, "z": 1.0}, "q2": {"x": 2.0, "b": 1.0}}
    out = evaluate_run(qrels, run, ["recall@10", "mrr", "ndcg@10"])
    assert out["recall@10"] == 1.0
    assert math.isclose(out["mrr"], (1.0 + 0.5) / 2)
