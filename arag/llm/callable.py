"""Wrap any plain Python function as an ``LLM``.

The simplest possible plug-in: ``CallableLLM(lambda prompt: my_model(prompt))``.
Advanced users can supply batch/stream/token-count callables too.
"""

from __future__ import annotations

from typing import Callable, Iterator, Optional, Sequence


class CallableLLM:
    """Adapt ``def f(prompt: str) -> str`` (and optional extras) to the LLM protocol.

    By default the wrapped function is called with just the prompt string, so the
    simplest signature plugs in with zero boilerplate. Set ``accepts_kwargs=True``
    to forward ``max_tokens``/``temperature``/``stop``/``system`` to your function.
    """

    def __init__(
        self,
        fn: Callable[..., str],
        *,
        batch_fn: Optional[Callable[..., Sequence[str]]] = None,
        stream_fn: Optional[Callable[..., Iterator[str]]] = None,
        token_counter: Optional[Callable[[str], int]] = None,
        model_name: str = "callable",
        accepts_kwargs: bool = False,
    ) -> None:
        self._fn = fn
        self._batch_fn = batch_fn
        self._stream_fn = stream_fn
        self._token_counter = token_counter
        self.model_name = model_name
        self._accepts_kwargs = accepts_kwargs

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        stop: Optional[Sequence[str]] = None,
        system: Optional[str] = None,
    ) -> str:
        if self._accepts_kwargs:
            return self._fn(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                system=system,
            )
        return self._fn(prompt)

    # Optional members are only exposed if the user supplied the corresponding fn,
    # so capability probes (has_batch/has_stream) report the truth.
    if False:  # pragma: no cover - documentation of the dynamic attributes below
        def batch(self, prompts, **opts): ...
        def stream(self, prompt, **opts): ...
        def count_tokens(self, text): ...

    def __getattr__(self, name: str):
        # Dynamically expose optional capabilities without advertising absent ones.
        if name == "batch" and self._batch_fn is not None:
            return lambda prompts, **opts: list(self._batch_fn(prompts))
        if name == "stream" and self._stream_fn is not None:
            return lambda prompt, **opts: self._stream_fn(prompt)
        if name == "count_tokens" and self._token_counter is not None:
            return self._token_counter
        raise AttributeError(name)
