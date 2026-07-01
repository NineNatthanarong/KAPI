# Kapi

**Fast, local, zero-setup RAG with no embedding model.** BM25 + char-trigrams + the LLM you plug in — no vectors, no GPU, no vector database. Query time is what BM25 already is: **~1 ms, $0, fully explainable.**

```bash
pip install kapi
```

```python
from kapi import Kapi

rag = Kapi(preset="fast")                      # pure lexical — no LLM, no setup
rag.add_texts(["Dijkstra finds shortest paths.", "Tomato soup needs basil."])
print(rag.search("shortest path", k=1)[0].text)
# Dijkstra finds shortest paths.
```

---

## Why no embeddings?

Everyone else puts the smart, expensive compute on the **hot path** — a model runs on *every query* (dense embeddings) or an LLM reasons on *every query* (query-time reasoning). Kapi's bet: move it to **index time**. An LLM runs **once per document, cached forever**, and *compiles* each document into an enriched, purely **lexical** representation. Queries then stay pure BM25.

- **$0 per query, no vector RAM.** Embedding stacks pay ~$0.02–0.13 / 1M tokens *at query time* and hold ~6.1 GB RAM per 1M docs resident. Kapi's query-time number is `$0` for compute and `$0` for vector storage.
- **Zero setup.** `pip install`, no model download, no Java, no GPU, no vector DB.
- **Explainable + deterministic.** Every score is a term match you can read.
- **Graceful.** With no LLM it degrades to a strong hybrid-lexical retriever. LLM features are pure add-ons.

It is not a toy: on **BEIR scifact** it **matches a 4B dense embedder** at a fraction of the query-time cost (see [Evaluation](#evaluation)).

---

## Install

```bash
pip install kapi                # core: tantivy engine + stemmer + http client. No models, ever.
pip install "kapi[openai]"      # openai SDK + tiktoken (exact token counts)
pip install "kapi[bm25s]"       # optional in-memory bm25s engine
pip install "kapi[pdf,html]"    # PDF text + fast HTML loaders
pip install "kapi[eval]"        # ranx / pytrec_eval / BEIR / RAGAS / datasets — for benchmarks
```

Python ≥ 3.10.

---

# How to use

## 1. Pure lexical (no LLM)

Everything works with no LLM at all — this is the `fast` preset.

```python
from kapi import Kapi

rag = Kapi(preset="fast", path="./idx")        # on-disk; omit path for in-memory
rag.add("docs/")                               # a dir, a file, a glob, texts, or Documents
for h in rag.search("how do refunds work?", k=5):
    print(f"{h.score:.3f}  {h.source}\n   {h.text[:100]}")
rag.close()
```

`add` accepts a directory, a file, a glob, a list of strings, or `Document` objects:

```python
from kapi import Document

rag.add("docs/")                                    # walk a directory
rag.add("report.pdf")                               # needs kapi[pdf]
rag.add_texts(["first passage", "second passage"])  # raw strings
rag.add([Document(doc_id="d1", text="...", source="d1.md", metadata={"team": "billing"})])
```

## 2. Plug in any LLM

The contract is one method, `complete`. Use the built-in OpenAI-compatible adapter, wrap a function, or pass nothing.

```python
from kapi import Kapi
from kapi.llm import OpenAICompatLLM, CallableLLM

# Any OpenAI-compatible endpoint: OpenAI, Ollama, vLLM, llama.cpp, LM Studio, Together, Groq...
llm = OpenAICompatLLM(base_url="http://localhost:11434/v1/", model="llama3.2", api_key="ollama")

# ...or wrap any callable (called with just the prompt string by default)
llm = CallableLLM(lambda prompt: my_model(prompt))

# ...or no LLM at all — pure lexical retrieval still works
rag = Kapi(preset="fast")
```

## 3. Presets

| Preset | LLM used | What runs | Use it for |
|---|---|---|---|
| `fast` | none | pure lexical (BM25 + trigrams + title) | sub-10 ms retrieval, no LLM |
| `quality` *(default)* | yes | + contextual indexing (offline) + query expansion + grounded answers | best general RAG |
| `compiled` | yes | + the index-time **compiler** (CSC) + the adaptive **router** | reasoning-heavy corpora, $0 queries |

Any field is overridable per instance: `Kapi(preset="compiled", consensus_k=5, engine="sqlite")`.

## 4. Compiled Retrieval (`preset="compiled"`)

> **Retrieval intelligence is a compile-time problem, not a serve-time problem.**

One cached offline pass per chunk emits an enrichment **bundle** — all plain text that lands in the lexical index (never in the cited text):

| Pillar | What the compiler adds | Prior art |
|---|---|---|
| **blurb** | a chunk-specific context sentence | Anthropic Contextual Retrieval, 2024 |
| **questions** | the queries this chunk answers | doc2query / docTTTTTquery |
| **propositions** | atomic, decontextualized facts (rare entities matchable) | Dense X, EMNLP 2024 |
| **reasoning** | second-order facts & multi-hop bridges *not lexically present* | the BRIGHT-winning signal, precomputed |

**CSC — Consensus Sparse Compilation** (the novel core): the compiler is sampled `k` times; a term's weight is its **agreement across samples** — a training-free, label-free learned-sparse weighter that doubles as a self-consistency filter (hallucinated terms appear once and are dropped; entailed terms recur and are promoted). **Literal anchoring** keeps every source-literal term (IDs, error codes, proper nouns) at a floor weight, so exact match is structurally protected. The result is a second sparse "leg" that fuses with plain BM25 — hybrid's two-error-profile win, **with no embedding model.**

```python
llm = OpenAICompatLLM(base_url="...", model="...", api_key="...")

rag = Kapi(llm=llm, preset="compiled", path="./idx")
rag.compile("docs/")                              # offline, cached by content-hash
print(rag.query("does this scale to a billion rows?").answer)   # ~1 ms lexical retrieval

rag = Kapi.open("./idx")                           # reopen with NO llm — the compiled index still serves
```

**Adaptive router** — the only query-time LLM use, and it's gated. The first lexical pass is ~1 ms and $0; a cheap confidence signal (no hits, low recall, or an ambiguous top-vs-2nd margin) decides whether to spend one LLM call escalating (query expansion + re-search). Short queries are treated as precise and never escalated, dodging the expansion *precision trap*. Inspect the decision:

```python
rag.search("how can I get my money back?", k=5)
print(rag.last_route)   # RouterDecision(escalate=..., reason='low_margin'|'no_hits'|'confident', ...)
```

## 5. Persistent & incremental

```python
rag = Kapi(llm=llm, path="./idx")
rag.add("docs/"); rag.close()

rag = Kapi.open("./idx", llm=llm)     # reopen later
rag.sync("docs/")                     # re-index only changed files; drop deleted ones
rag.remove("d1")                      # delete one document
```

## 6. Answers, citations, streaming

```python
res = rag.query("How do refunds work?", k=8)
print(res.answer)          # grounded answer (None if no LLM / generation off)
for c in res.citations:    # [1], [2], ... -> source + chunk_id + score
    print(c.marker, c.source, f"{c.score:.3f}")

for tok in rag.query_stream("How do refunds work?"):   # token stream
    print(tok, end="")
```

## 7. Command line

```bash
kapi compile ./docs --index ./idx --base-url http://localhost:11434/v1/ --model llama3.2
kapi query  "how do refunds work?" --index ./idx
kapi stats  --index ./idx
kapi tco    --queries-per-month 5000000 --months 36   # KAPI vs dense+vectorDB cost model
```

`compile` is `add` named for the mental model. With no `--base-url` it builds a pure-lexical index; retrieval is always embedding-free. LLM settings also read from `KAPI_LLM_BASE_URL` / `KAPI_LLM_MODEL` / `KAPI_LLM_API_KEY`.

## 8. Compile once, serve anywhere (air-gapped)

The expensive step (LLM compilation) runs **once**; the serving index is a plain lexical artifact — no LLM, no network, no vector DB. Bundle it and ship it to an on-prem / air-gapped box:

```bash
kapi export --index ./idx --out ship.kapi.tgz      # portable bundle (drops the LLM cache)
kapi import ship.kapi.tgz --index ./served         # unpack on the target machine
kapi query  "how do refunds work?" --index ./served   # $0, ~1 ms, no model
```

```python
rag.export_bundle("ship.kapi.tgz")
served = Kapi.import_bundle("ship.kapi.tgz", "./served")   # opens with no LLM
```

Or run the **hosted compilation service** — clients POST documents, get back a serving bundle (the smart compute stays server-side; no embedding model ever crosses the wire):

```bash
kapi serve --base-url http://localhost:11434/v1/ --model llama3.2   # POST /compile, GET /bundle/<job>
```

## 9. Drop into LangChain / LlamaIndex

```python
from kapi.integrations import to_langchain_retriever, to_llamaindex_retriever
lc = to_langchain_retriever(rag, k=5)      # a LangChain BaseRetriever
li = to_llamaindex_retriever(rag, k=5)     # a LlamaIndex BaseRetriever
```

---

# Evaluation

Kapi ships its own honest, **cost-tiered** evaluation harness (`kapi.eval`). The rule: never compare a $0-per-query lexical system against a system that pays a model per query without labelling the tier. Credibility is the moat.

## The metrics module (pure-Python, no deps)

`kapi.eval.ir_metrics` implements the standard IR metrics with zero dependencies. Supported metric strings: `ndcg@k`, `recall@k`, `precision@k`, `hit@k`, `mrr`, `map`.

```python
from kapi.eval import evaluate_run, ndcg_at_k

qrels = {"q1": {"docA": 1, "docC": 1}}                    # ground-truth relevance
run   = {"q1": {"docA": 9.1, "docB": 4.2, "docC": 2.0}}  # your system's doc -> score

print(evaluate_run(qrels, run, metrics=("ndcg@10", "recall@10", "mrr")))
# {'ndcg@10': ..., 'recall@10': ..., 'mrr': ...}
```

Evaluate Kapi on your own labelled queries in a few lines:

```python
rag = Kapi(preset="fast"); rag.add("corpus/")
run = {}
for qid, text in my_queries.items():
    scores = {}
    for h in rag.search(text, k=100):
        did = h.chunk_id.split("::", 1)[0]                # chunk -> parent doc
        scores[did] = max(scores.get(did, -1e9), h.score) # max-pool chunks
    run[qid] = scores
print(evaluate_run(my_qrels, run, ("ndcg@10", "recall@100", "mrr")))
```

## BEIR (breadth / parity)

`run_beir` builds a fresh index via your factory, indexes the corpus, runs every query, and scores it against the published BM25 anchor. Needs `pip install "kapi[eval]"`.

```python
from kapi import Kapi
from kapi.eval import run_beir

report = run_beir(lambda: Kapi(preset="fast"), dataset="scifact", split="test")
print(report)     # nDCG@10 / Recall@100 / MRR, next to the published BM25 nDCG@10
```

### The scifact leaderboard (real result)

No embedding model, no GPU, no vector DB — **Kapi matches a 4B dense embedder** and beats mid-tier embedders, at a fraction of the query-time cost. Query-cost tiers: **A:none** = pure BM25 (~1 ms). **A:embed** = one embedding pass/query. **A:offline LLM** = LLM enrichment paid *once* at index time. **B:+LLM/query** = one rerank call/query.

| System | Emb. | Query cost | nDCG@10 | MRR |
|---|---|---|--:|--:|
| dense qwen3-embedding-8b | ✔ 4096d | A: embed/query | 0.7629 | 0.7281 |
| **Kapi doc2query ×2** | ✗ | **A: offline LLM** | **0.7291** | **0.7042** |
| dense qwen3-embedding-4b | ✔ 2560d | A: embed/query | 0.7308 | 0.6948 |
| dense text-embedding-3-small | ✔ 1536d | A: embed/query | 0.7164 | 0.6841 |
| Kapi BM25 + char-trigram | ✗ | A: none | 0.6995 | 0.6662 |
| Kapi-fast (lexical) | ✗ | A: none | 0.6878 | 0.6551 |
| dense bge-m3 | ✔ 1024d | A: embed/query | 0.6437 | 0.6131 |

**Honest headline:** *Kapi doc2query×2* (offline-LLM tier) **statistically ties qwen3-embedding-4b on nDCG@10 and beats it on MRR**, while clearly beating text-embedding-3-small and bge-m3 — with `$0` at query time. Only the 8B embedder is decisively ahead. In the reranked tier (one LLM call/query on both sides) dense wins — reranking does not erase dense's higher candidate recall. Full 30-row table, setup, and ablations: [`benchmarks/scifact_results.md`](benchmarks/scifact_results.md).

**What the harness taught us (findings, not vibes):**
- **Query-side expansion is a trap on precise retrieval** — LLM query2doc (−0.015 nDCG) and RM3 (−0.14 to −0.19) both crater precision. *Enrich the corpus, never the query.* (This is exactly why the `compiled` router only expands weak/ambiguous queries.)
- **Anticipatory indexing (doc2query) is the cost-fair lever** — the LLM writes, at index time, the claims each doc answers; queries stay pure BM25. Best no-embedding retrieval-only result, free at query time.

## BRIGHT (the reasoning-intensive hero benchmark)

BRIGHT is built so surface similarity is *insufficient* — relevance needs multi-step reasoning. It is the board where off-the-shelf dense **collapses** (the #1 MTEB model scores 18.3) and embedding-free approaches sit at the top. This is where Compiled Retrieval is aimed.

```python
from kapi import Kapi
from kapi.eval import run_bright, run_bright_all

# one subset
report = run_bright(lambda: Kapi(llm=llm, preset="compiled"), subset="biology")
print(report)                       # nDCG@10 / Recall@100 / MRR + directional reference anchors

# all 12 subsets (heavy — downloads each)
results = run_bright_all(lambda: Kapi(llm=llm, preset="compiled"))
```

Directional reference anchors (nDCG@10, averaged over subsets — verify before quoting publicly):

| Reference point | nDCG@10 |
|---|--:|
| BM25 zero-shot | 14.3 |
| Off-the-shelf dense (SFR-Embedding-Mistral, #1 MTEB) | 18.3 |
| BM25 + GPT-4 CoT-rewritten queries | 27.0 |
| LATTICE (embedding-free SOTA, but pays an LLM per query) | 46.7 |

The target: **beat off-the-shelf dense at $0 query cost**, and approach query-time-reasoning systems while staying ~1 ms per query.

## The CSC benchmark script

`benchmarks/csc_eval.py` runs Compiled Retrieval live on scifact (loads via HF `datasets`, no torch/beir needed):

```bash
python benchmarks/csc_eval.py baseline                     # pure-lexical, free
OPENROUTER_API_KEY=... python benchmarks/csc_eval.py smoke  # compile a few docs, print bundle + term weights
OPENROUTER_API_KEY=... python benchmarks/csc_eval.py compiled --index ./idx_csc --k 3   # CSC, k consensus samples
```

## Cost evaluation (TCO)

Evaluation isn't only quality — it's the bill. `kapi tco` models Kapi (one-time compile, `$0` queries, `$0` vector RAM) against a dense + vector-DB stack (cheap index, but recurring per-query embedding + resident RAM):

```bash
kapi tco --docs 1000000 --queries-per-month 5000000 --months 36
```
```python
from kapi.tco import TCOInputs, compute_tco, format_report
print(format_report(TCOInputs(), compute_tco(TCOInputs())))
```

## Reproducing

```bash
pip install "kapi[eval]"
export KAPI_LLM_BASE_URL=... KAPI_LLM_MODEL=... KAPI_LLM_API_KEY=...   # any OpenAI-compatible endpoint
python -m pytest                                                        # 87 passing, 3 opt-in (eval/live) skipped
```

---

## How it works

```
        indexed_text (never cited)                      raw_text (cited)
  ┌─────────────────────────────────┐          ┌──────────────────────────┐
  │  Leg A: BM25 + trigrams + title │          │  Leg B: CSC consensus     │
  │  over the enriched text         │◄── RRF ──►│  sparse term weights      │
  └─────────────────────────────────┘  fusion  └──────────────────────────┘
                    ▲                                        ▲
     offline compiler (cached, cost-guarded)     adaptive router (query-time, gated)
```

- **Structure-aware chunking** with span-exact offsets; `indexed_text` (enriched, searched) is kept separate from `raw_text` (clean, cited).
- **Two sparse legs**, fused by RRF (zero-tuning) or convex combination — hybrid's complementarity with no dense leg.
- **Offline compiler** with a content-hash cache (re-indexing is free) and a cost guard.
- **Adaptive router** spends an LLM call only when the cheap path is unsure.

## Engines

Swap the lexical backend without changing anything else:

| Engine | Install | Notes |
|---|---|---|
| `tantivy` *(default)* | core | fast, persistent, multi-field scoring |
| `sqlite` | core | FTS5, zero extra deps, portable single file |
| `bm25s` | `kapi[bm25s]` | in-memory, pure-NumPy, fast batch |

```python
rag = Kapi(preset="fast", engine="sqlite", path="./idx")
```

## Design guarantees

- **No LLM required.** Pure-lexical retrieval always works; LLM features are add-ons, disabled by construction when no LLM is supplied.
- **All LLM cost is offline** (index-time, cached) or a single gated query-time call (the router).
- **Portable & explainable.** Deterministic scores; the serving index is a plain directory you can archive and ship.

## License

MIT.
