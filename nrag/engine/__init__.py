"""Pluggable lexical index engines."""

from __future__ import annotations

from .base import IndexEngine, open_engine

__all__ = ["IndexEngine", "open_engine"]
