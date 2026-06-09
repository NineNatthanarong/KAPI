"""LLM plug-in adapters. Plug in any LLM; the local Ollama path needs no extra deps."""

from __future__ import annotations

from .base import LLM, batched_complete, estimate_tokens, has_batch, has_stream, model_name
from .callable import CallableLLM
from .openai_compat import OpenAICompatLLM

__all__ = [
    "LLM",
    "CallableLLM",
    "OpenAICompatLLM",
    "batched_complete",
    "estimate_tokens",
    "has_batch",
    "has_stream",
    "model_name",
]
