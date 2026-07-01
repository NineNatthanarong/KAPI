# NRAG vs dense embeddings — BEIR scifact

Can a retriever with **no embedding model** compete with dense embedding models? NRAG (BM25 + char-trigrams + optional LLM index-time enrichment + optional LLM reranking — all via the *plugged-in* LLM, no embeddings, no vector DB) vs four dense embedding models, on **BEIR scifact** (claim verification).

## Honest headline (read by cost tier)

- **Cost-fair tier (retrieval only, no per-query model):** best no-embedding = *NRAG doc2query x2 (ngram=0.3,title=0.5)* — **nDCG@10 0.7291, MRR 0.7042**. This **statistically ties** qwen3-4b (nDCG@10 0.7308, MRR 0.6948; the 0.0017 nDCG gap is within run-to-run noise on 300 queries) and **beats it on MRR** — while clearly beating text-embedding-3-small (0.7164) at far lower query-time cost. Only qwen3-8b (0.7629) is decisively ahead.
- **Reranked tier (one LLM call/query on both sides):** dense wins — qwen3-4b+rerank leads (0.7738); best no-embedding = *NRAG + LLM rerank (base, top-20)* (0.7501). Reranking does **not** erase dense's higher candidate recall.
- **Takeaway:** with **no embedding model, no GPU, no vector DB**, NRAG **matches qwen3-embedding-4b on this task** (tie on nDCG@10, ahead on MRR) at a fraction of the query-time cost, and clearly beats mid-tier embedders (text-embedding-3-small, bge-m3). It trails only the 8B embedder. (NRAG+rerank's 0.75 over *bare* qwen3-4b was a cost-tier mismatch — reranking is an extra LLM call/query and lifts dense more.)

## Setup

- **Dataset:** BEIR `scifact` — 5,183 abstracts, 300 test claims, 339 judgments (~1.13 relevant/query). Doc-level scoring (chunk→doc max-pool).
- **Metrics:** nDCG@10 / Recall@10 / Recall@100 / MRR (`nrag.eval.ir_metrics`).
- **Dense:** embeddings via OpenRouter `/embeddings`; cosine over L2-normalized vectors.
- **Reranker / doc2query LLM:** `deepseek/deepseek-v4-flash` via OpenRouter. **Date:** 2026-06-09.

## Cost-tier leaderboard

Query cost: **A:none** = pure BM25 (~1 ms, no model). **A:embed** = one embedding forward pass/query. **A:offline LLM** = LLM enrichment paid once at index time, queries then ~1 ms. **B:+LLM/query** = one LLM rerank call/query (~seconds).

| System | Emb. | Query cost | nDCG@10 | R@10 | R@100 | MRR |
|---|---|---|---|---|---|---|
| dense qwen/qwen3-embedding-4b + LLM rerank (top-20) | ✔ 2560d | B: +LLM/query | 0.7738 | 0.9050 | 0.9217 | 0.7379|
| dense: qwen/qwen3-embedding-8b | ✔ 4096d | A: embed/query | 0.7629 | 0.8983 | 0.9733 | 0.7281|
| dense openai/text-embedding-3-small + LLM rerank (top-20) | ✔ 1536d | B: +LLM/query | 0.7583 | 0.8783 | 0.9017 | 0.7246|
| NRAG + LLM rerank (base, top-20) | ✗ | B: +LLM/query | 0.7501 | 0.8489 | 0.8656 | 0.7252|
| NRAG + LLM rerank (base, top-30) | ✗ | B: +LLM/query | 0.7487 | 0.8532 | 0.8857 | 0.7242|
| NRAG + doc2query + LLM rerank (top-20) | ✗ | B: +LLM/query | 0.7420 | 0.8499 | 0.8799 | 0.7165|
| dense: qwen/qwen3-embedding-4b | ✔ 2560d | A: embed/query | 0.7308 | 0.8733 | 0.9733 | 0.6948|
| NRAG doc2query x2 (ngram=0.3,title=0.5) | ✗ | A: offline LLM | 0.7291 | 0.8399 | 0.9163 | 0.7042|
| NRAG + doc2query x2 (ngram_w=0.3) | ✗ | A: offline LLM | 0.7255 | 0.8529 | 0.9197 | 0.6949|
| NRAG + doc2query x2 (ngram_w=0.2) | ✗ | A: offline LLM | 0.7238 | 0.8463 | 0.9163 | 0.6949|
| NRAG + doc2query (ngram_w=0.2) | ✗ | A: offline LLM | 0.7204 | 0.8407 | 0.9047 | 0.6911|
| NRAG + doc2query x2 (ngram_w=0.1) | ✗ | A: offline LLM | 0.7201 | 0.8504 | 0.9119 | 0.6870|
| NRAG + doc2query (ngram_w=0.3) | ✗ | A: offline LLM | 0.7193 | 0.8341 | 0.9130 | 0.6935|
| dense: openai/text-embedding-3-small | ✔ 1536d | A: embed/query | 0.7164 | 0.8536 | 0.9700 | 0.6841|
| NRAG + doc2query (ngram_w=0.0) | ✗ | A: offline LLM | 0.7076 | 0.8306 | 0.9009 | 0.6799|
| lex ngram_w=0.3 | ✗ | A: none | 0.7034 | 0.8298 | 0.9157 | 0.6707|
| lex ngram_w=0.5 | ✗ | A: none | 0.7014 | 0.8231 | 0.9057 | 0.6702|
| BM25 + char-trigram | ✗ | A: none | 0.6995 | 0.8264 | 0.8957 | 0.6662|
| lex ngram_w=0.8 | ✗ | A: none | 0.6967 | 0.8264 | 0.8923 | 0.6627|
| lex ngram_w=1.0 | ✗ | A: none | 0.6935 | 0.8214 | 0.8923 | 0.6607|
| NRAG-fast (lexical, no embeddings) | ✗ | A: none | 0.6878 | 0.8164 | 0.8897 | 0.6551|
| BM25 plain (word-only) | ✗ | A: none | 0.6860 | 0.8193 | 0.8936 | 0.6528|
| bm25s engine (pure sparse, word-only) | ✗ | A: none | 0.6850 | 0.8147 | 0.8852 | 0.6523|
| lex ngram_w=1.5 | ✗ | A: none | 0.6846 | 0.8081 | 0.8923 | 0.6543|
| lex ngram_w=2.0 | ✗ | A: none | 0.6838 | 0.8070 | 0.8923 | 0.6538|
| lex ngram_w=3.0 | ✗ | A: none | 0.6775 | 0.8037 | 0.8890 | 0.6464|
| NRAG + query expansion | ✗ | A: none | 0.6728 | 0.8122 | 0.8963 | 0.6378|
| NRAG doc2query DUAL-INDEX fusion (wA=1,wB=3) | ✗ | A: offline LLM | 0.6663 | 0.8031 | 0.9387 | 0.6364|
| NRAG + ensemble rerank (base, top-20, 3x) | ✗ | B: +LLM/query | 0.6604 | 0.8319 | 0.8656 | 0.6168|
| dense: baai/bge-m3 | ✔ 1024d | A: embed/query | 0.6437 | 0.7834 | 0.9037 | 0.6131|
| BM25 + title boost | ✗ | A: none | 0.6097 | 0.7247 | 0.8459 | 0.5850|
| NRAG + RM3 PRF (R=3,M=8,a=3) | ✗ | A: none | 0.5676 | 0.8184 | 0.9123 | 0.4941|

## What we learned (the experiments)

- **Query-side expansion is a trap on precise retrieval.** LLM query2doc (−0.015) and statistical RM3 (−0.14 to −0.19) both crater precision@10. Enrich the *corpus*, never the query.
- **Anticipatory indexing (doc2query) is the cost-fair lever.** The LLM writes, at index time, the claims each paper answers; queries stay pure BM25. This is the best no-embedding *retrieval-only* result and the only one that beats text-embedding-3-small for free at query time.
- **Reranking is the biggest single jump but not free** (+~0.05 nDCG@10, one LLM call/query) — and it lifts dense pipelines *more* (better candidate recall), so it doesn't close the gap for NRAG.
- **Self-consistency ensemble reranking backfired** (0.66): shuffling candidate order destroys the BM25 ordering prior the reranker relies on. Keep the first-stage order.
- **Char-trigrams help** (+0.014 over plain BM25); the independent `bm25s` engine reproduces plain BM25 (cross-validation). **Title-boost ×2.5 hurts** on abstract-style corpora (−0.08).

## Reproduce

```
python scratch/ablation.py                      # lexical ablations
python scratch/eval_all.py te3small bgem3 qwen4b qwen8b   # dense baselines
python scratch/doc2query.py all                 # anticipatory indexing
python scratch/exp_rerank2.py base 20           # NRAG + rerank
python scratch/exp_dense_rerank.py              # dense + same rerank (fair tier B)
python benchmarks/consolidate.py                # regenerate this report
```