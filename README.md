# ARAG

**Fast, local, zero-setup RAG with no embedding model.** Plug in any LLM and go.

ARAG is a Python RAG library whose retriever is **lexical, not neural** — BM25 over words
**plus** char-trigrams **plus** title boosts, fused into one query. No embedding model to
download, no vector DB to run, no GPU. The LLM you plug in is used only where lexical
search is genuinely weak: to **contextualize chunks at index time** and **expand queries**
at search time. Everything works with no LLM too (pure lexical retrieval).

```bash
pip install arag        # core: tantivy + snowballstemmer + httpx. No models, ever.
```

```python
from arag import ARAG
from arag.llm import OpenAICompatLLM

# Local + low cost: Ollama via its OpenAI-compatible endpoint (no extra install needed)
llm = OpenAICompatLLM(base_url="http://localhost:11434/v1/", api_key="ollama", model="llama3.2")

rag = ARAG(llm=llm)                 # quality preset: contextual indexing + query expansion ON
rag.add("docs/")                    # ingest a dir / file / glob; chunks contextualized offline, then indexed
print(rag.query("How do refunds work?").answer)   # one online call -> grounded answer with [n] citations
```

## Why no embeddings?

Per **BEIR** (Thakur et al., NeurIPS 2021), BM25 is a famously strong *zero-shot* baseline
that dense retrievers frequently **fail to beat out-of-domain** — especially on
keyword/technical/argument queries. ARAG leans into that and fixes BM25's one real
weakness (vocabulary mismatch) with the LLM you already have, instead of a second model:

| BM25 weakness | ARAG's embedding-free fix |
|---|---|
| Lost context when chunked | **Contextual indexing** — LLM writes a 1–2 sentence blurb per chunk, prepended before indexing (offline, cached). ~−49% retrieval failures (Anthropic, 2024). |
| Vocabulary mismatch (synonyms) | **Query expansion** — query2doc / CoT keywords (one online call). Up to +15 nDCG@10 (Wang et al., 2023). |
| Typos / morphology | **Char-trigram signal** fused with the word signal. |
| Crude top-k ordering | **Multi-signal + RRF**, title boosts. |

## Plug in any LLM

```python
from arag.llm import OpenAICompatLLM, CallableLLM

# Any OpenAI-compatible endpoint: OpenAI, Ollama, vLLM, llama.cpp, LM Studio, Together, Groq...
OpenAICompatLLM(base_url="https://api.openai.com/v1", api_key=KEY, model="gpt-4o-mini")

# ...or wrap any function
rag = ARAG(llm=CallableLLM(lambda prompt: my_model(prompt)))

# ...or no LLM at all — pure lexical retrieval still works
rag = ARAG()
hits = rag.search("refund policy")          # ranked chunks, sub-10ms class
```

## Speed vs quality, your call

```python
ARAG(llm=llm)                      # preset="quality" (default): contextual + expansion ON
ARAG(llm=llm, preset="fast")       # pure-lexical retrieval; LLM only writes the final answer
ARAG()                             # no LLM: retrieval only
```
All LLM cost is **offline** (contextual indexing — one-time, cached, free with a local
model) or a **single online call** (query expansion). A cost guard refuses accidental
large paid-model runs; it's a no-op for local models.

## Persistent & incremental

```python
rag = ARAG(llm=llm, path="./index")   # on-disk index
rag.add("docs/")
rag.close()

rag = ARAG.open("./index", llm=llm)   # reopen later
rag.sync("docs/")                     # re-index only changed files; drop deleted ones
```

## Inspect results & citations

```python
res = rag.query("How do refunds work?", k=8)
res.answer                            # str | None (None if no LLM)
res.citations                         # [Citation(marker="[1]", source="docs/refunds.md", ...)]
for h in res.hits:                    # ranked retrieved chunks
    print(h.score, h.source, h.text[:120])

for token in rag.query_stream("..."): # stream the answer
    print(token, end="")
```

## Engines

ARAG ships a pluggable engine layer. The default needs no setup; alternatives are one
keyword away.

| Engine | `engine=` | Notes |
|---|---|---|
| **Tantivy** (default) | `"tantivy"` | Rust, Lucene-class speed, persistent + incremental, multi-signal in one query. pip wheel, no server. |
| **SQLite FTS5** | `"sqlite"` | Zero extra dependency (stdlib `sqlite3`). Word + trigram tables fused with RRF. |
| **bm25s** | `"bm25s"` | In-memory, fastest for fixed corpora. `pip install arag[bm25s]`. |

## Evaluate it (prove it's good enough)

```python
from arag.eval import evaluate_run                 # pure-Python nDCG@k / Recall@k / MRR
from arag.eval import run_beir                      # needs arag[eval]

report = run_beir(lambda: ARAG(engine="tantivy"), "scifact")
print(report)                                       # arag vs published BM25, side by side
```

## Install extras

```bash
pip install arag                # core (no models, local LLM path works)
pip install arag[openai]        # openai SDK + tiktoken (exact token counts)
pip install arag[bm25s]         # in-memory bm25s engine
pip install arag[eval]          # ranx / pytrec_eval / BEIR / RAGAS
pip install arag[pdf,html]      # PDF text + fast HTML loaders
```

## How it works

```
add(source) → load → chunk (recursive, structure-aware) → contextualize (offline LLM, cached)
            → index (BM25: word + trigram + title fields)
query(q)    → expand (1 LLM call) → multi-signal BM25 + RRF → top-k → generate (cited answer)
```

Requires Python ≥ 3.10. No embedding model. No server. No GPU.

## License

MIT
