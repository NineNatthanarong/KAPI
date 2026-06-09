"""Runnable Kapi demo — no external LLM server required.

Run:  python examples/demo.py

Shows three things end to end:
  1. Pure-lexical retrieval (no LLM) — including typo tolerance.
  2. The quality pipeline (contextual indexing + query expansion + grounded answer)
     using a tiny offline stand-in LLM, so it runs anywhere.
  3. Swapping to a real local LLM is a one-line change (see the bottom).
"""

from __future__ import annotations

from kapi import Kapi
from kapi._types import Document
from kapi.llm import CallableLLM

DOCS = [
    ("python", "Python", "Python is a high-level programming language with readable syntax."),
    ("recursion", "Recursion", "Recursion is when a function calls itself; a base case stops it."),
    ("refund", "Refund Policy",
     "Customers may be reimbursed to their original payment method after approval."),
    ("quicksort", "Quicksort", "Quicksort is a divide and conquer sort using a pivot element."),
]


def corpus():
    return [Document(doc_id=i, text=f"# {t}\n\n{b}", source=f"{i}.md",
                     metadata={"content_type": "markdown", "source": f"{i}.md"})
            for (i, t, b) in DOCS]


def offline_llm():
    """A deterministic stand-in so the demo needs no server."""
    def fn(prompt: str) -> str:
        if "situate this chunk" in prompt:
            return "This chunk is from a short reference document on the topic."
        if "Passage:" in prompt:
            return "Refund reimbursement money back returns to the original payment method."
        if "Keywords:" in prompt:
            return "Keywords: refund, money back, reimbursement, return"
        if prompt.startswith("Context:"):
            return "You can be reimbursed to your original payment method after approval [1]."
        return "ok"
    return CallableLLM(fn)


def main():
    print("=" * 70)
    print("1) PURE LEXICAL (no LLM)")
    print("=" * 70)
    rag = Kapi()
    rag.add(corpus())
    for q in ["readable programming language", "function that calls itself", "recurssion"]:
        hit = rag.search(q, k=1)[0]
        print(f"  {q!r:34} -> {hit.chunk.doc_id:10} (score {hit.score:.2f})")
    # 'money back' fails on pure lexical (vocabulary mismatch with 'reimbursed')
    mb = rag.search("money back", k=1)
    print(f"  {'money back':34} -> {mb[0].chunk.doc_id if mb else '(no match)'}")
    rag.close()

    print("\n" + "=" * 70)
    print("2) QUALITY PIPELINE (contextual indexing + expansion + answer)")
    print("=" * 70)
    rag = Kapi(llm=offline_llm())  # preset='quality'
    rep = rag.add(corpus())
    print(f"  indexed {rep.num_chunks} chunks, contextualized {rep.contextualized}")
    res = rag.query("money back", k=2)
    top = res.hits[0].chunk.doc_id if res.hits else "(none)"
    print(f"  'money back' now retrieves -> {top}   (vocabulary mismatch fixed)")
    print(f"  answer:    {res.answer}")
    print(f"  citations: {[(c.marker, c.source) for c in res.citations]}")
    rag.close()

    print("\n" + "=" * 70)
    print("3) USE A REAL LOCAL LLM (one line):")
    print("=" * 70)
    print("""  from kapi.llm import OpenAICompatLLM
  llm = OpenAICompatLLM(base_url="http://localhost:11434/v1/", api_key="ollama", model="llama3.2")
  rag = Kapi(llm=llm, path="./index")""")


if __name__ == "__main__":
    main()
