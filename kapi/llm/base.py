"""The LLM plug-in contract.

The interface a user must satisfy to "plug in any LLM" is deliberately tiny: one
required method, ``complete``. Everything else (batching for cheap offline contextual
indexing, streaming for generation, token counting for the cost guard) is optional and
discovered at runtime, with sensible fallbacks provided here so adapters stay small.
"""

from __future__ import annotations

import concurrent.futures
from typing import Iterator, Optional, Protocol, Sequence, runtime_checkable


@runtime_checkable
class LLM(Protocol):
    """Minimal protocol for a chat/completion model.

    Only ``complete`` is required. Optional members (checked via ``hasattr``):
      - ``batch(prompts, **opts) -> list[str]``  high-throughput offline path
      - ``stream(prompt, **opts) -> Iterator[str]``  token streaming for generation
      - ``count_tokens(text) -> int``  used by the cost guard / context budgeting
      - ``model_name: str``  used in cache keys and cost estimation
    """

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        stop: Optional[Sequence[str]] = None,
        system: Optional[str] = None,
    ) -> str: ...


# ---------------------------------------------------------------- capability probes
def has_batch(llm: object) -> bool:
    return callable(getattr(llm, "batch", None))


def has_stream(llm: object) -> bool:
    return callable(getattr(llm, "stream", None))


def model_name(llm: object) -> str:
    return getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"


# ---------------------------------------------------------------- shared helpers
def estimate_tokens(llm: object, text: str) -> int:
    """Token count via the LLM if it exposes ``count_tokens``, else a ~4-chars/token
    heuristic. Never raises; used for cost estimates and context budgeting."""
    fn = getattr(llm, "count_tokens", None)
    if callable(fn):
        try:
            return int(fn(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def batched_complete(
    llm: LLM,
    prompts: Sequence[str],
    *,
    concurrency: int = 8,
    **opts,
) -> list[str]:
    """Complete many prompts. Uses ``llm.batch`` if available, else a bounded
    thread pool over ``llm.complete`` (OpenAI-compatible servers have no batch
    primitive, so threading is how offline contextual indexing gets throughput)."""
    prompts = list(prompts)
    if not prompts:
        return []
    if has_batch(llm):
        return list(llm.batch(prompts, **opts))  # type: ignore[attr-defined]
    if concurrency <= 1 or len(prompts) == 1:
        return [llm.complete(p, **opts) for p in prompts]

    results: list[Optional[str]] = [None] * len(prompts)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(llm.complete, p, **opts): i for i, p in enumerate(prompts)}
        for fut in concurrent.futures.as_completed(futs):
            results[futs[fut]] = fut.result()
    return [r if r is not None else "" for r in results]


def stream_complete(llm: LLM, prompt: str, **opts) -> Iterator[str]:
    """Stream tokens if the LLM supports it, else yield the whole completion once."""
    if has_stream(llm):
        yield from llm.stream(prompt, **opts)  # type: ignore[attr-defined]
    else:
        yield llm.complete(prompt, **opts)
