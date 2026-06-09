"""Configuration and presets for Kapi.

A single frozen ``Config`` is the source of truth for all behavior. Presets are
factory classmethods; any field is overridable via kwargs to ``Kapi(...)``. The
default preset is ``quality`` (LLM enhancers ON) per the product brief; ``fast``
collapses to pure-lexical; ``for_no_llm`` is applied automatically when no LLM is
supplied so "no LLM still works" is guaranteed by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional

ExpandMode = Literal["query2doc", "cot_keywords", "auto"]
Preset = Literal["quality", "fast"]


@dataclass(frozen=True)
class Config:
    # ---- top-level ----
    preset: Preset = "quality"

    # ---- contextual indexing (offline, default ON in 'quality') ----
    contextual_enabled: bool = True
    contextual_model: Optional[str] = None          # None -> plugged-in LLM default
    contextual_max_tokens: int = 128                 # blurb budget (~50-100 tokens of content)
    contextual_concurrency: int = 8                  # bounded parallel offline calls
    contextual_window: Literal["full_doc", "section"] = "full_doc"
    contextual_window_token_limit: int = 6000        # above this, fall back to section window
    contextual_cost_guard_usd: float = 5.0           # 0 disables the guard
    contextual_cost_guard_mode: Literal["raise", "warn", "off"] = "raise"

    # ---- query expansion (online, single call, default ON in 'quality') ----
    expand_enabled: bool = True
    expand_mode: ExpandMode = "auto"
    expand_query_repeat: int = 5                      # query2doc ~5x repetition trick (sparse retrieval)
    expand_max_tokens: int = 256
    expand_auto_short_query_words: int = 5            # 'auto': <= N words -> cot_keywords

    # ---- retrieval ----
    k: int = 10                                       # final top-k returned
    retrieve_k: int = 50                              # candidates fetched before truncation
    fusion: Literal["rrf", "convex"] = "rrf"
    rrf_k: int = 60
    # field weights (mirrors FieldWeights defaults; kept here so presets can tune them)
    weight_body: float = 1.0
    weight_ngram: float = 0.6
    weight_title: float = 2.5
    enable_ngram: bool = True
    fuzzy: bool = False

    # ---- generation ----
    generate_enabled: bool = True                     # auto-False if no LLM
    answer_max_tokens: int = 512
    context_token_budget: int = 6000                  # cap on chunk tokens packed into the prompt
    citation_style: Literal["bracket", "none"] = "bracket"

    # ---- engine / io ----
    engine: Literal["tantivy", "sqlite", "bm25s"] = "tantivy"
    language: str = "english"
    path: Optional[str] = None                         # None -> in-memory

    # ---------------------------------------------------------------- presets
    @classmethod
    def quality(cls, **overrides) -> "Config":
        return replace(cls(preset="quality"), **overrides)

    @classmethod
    def fast(cls, **overrides) -> "Config":
        base = cls(preset="fast", contextual_enabled=False, expand_enabled=False)
        return replace(base, **overrides)

    @classmethod
    def from_preset(cls, preset: Preset = "quality", **overrides) -> "Config":
        return cls.fast(**overrides) if preset == "fast" else cls.quality(**overrides)

    def for_no_llm(self) -> "Config":
        """Disable every feature that needs an LLM (graceful degradation)."""
        return replace(self, contextual_enabled=False, expand_enabled=False,
                       generate_enabled=False)

    def with_overrides(self, **overrides) -> "Config":
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})
