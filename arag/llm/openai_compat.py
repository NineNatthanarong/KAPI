"""OpenAI-compatible LLM adapter.

Works against any ``/v1/chat/completions`` endpoint: OpenAI, Ollama, vLLM,
llama.cpp server, LM Studio, Together, Groq, etc. Implemented on ``httpx`` (a light
core dependency) so the headline *local* path — Ollama — needs no extra install. If
the official ``openai`` SDK is installed it is used transparently for parity on
retries/timeouts; otherwise the raw REST endpoint is called directly.

Concrete plug-in patterns::

    Ollama (local)   base_url="http://localhost:11434/v1/"  api_key="ollama"     model="llama3.2"
    llama.cpp server base_url="http://localhost:8080/v1"    api_key="sk-no-key"  model=<loaded>
    LM Studio        base_url="http://localhost:1234/v1"    api_key="lm-studio"  model=<shown>
    vLLM             base_url="http://localhost:8000/v1"    api_key="EMPTY"      model=<served id>
    OpenAI cloud     base_url="https://api.openai.com/v1"   api_key=$OPENAI_API_KEY  model="gpt-4o-mini"
"""

from __future__ import annotations

import json
import time
from typing import Iterator, Optional, Sequence

import httpx


class OpenAICompatLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "not-needed",
        *,
        timeout: float = 60.0,
        max_retries: int = 3,
        extra_headers: Optional[dict] = None,
        default_temperature: float = 0.0,
        use_sdk: Optional[bool] = None,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.model_name = model
        self.api_key = api_key or "not-needed"
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.extra_headers = extra_headers or {}
        self.default_temperature = default_temperature
        self._transport = transport  # injectable for testing
        self._sdk_client = self._maybe_make_sdk_client(use_sdk)
        self._client: Optional[httpx.Client] = None

    # ------------------------------------------------------------------ wiring
    def _maybe_make_sdk_client(self, use_sdk: Optional[bool]):
        if use_sdk is False:
            return None
        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            if use_sdk is True:
                raise
            return None
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            headers = {"Authorization": f"Bearer {self.api_key}",
                       "Content-Type": "application/json", **self.extra_headers}
            self._client = httpx.Client(timeout=self.timeout, headers=headers,
                                        transport=self._transport)
        return self._client

    def _messages(self, prompt: str, system: Optional[str]) -> list[dict]:
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    # ------------------------------------------------------------------ API
    def complete(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[Sequence[str]] = None,
        system: Optional[str] = None,
    ) -> str:
        temperature = self.default_temperature if temperature is None else temperature
        messages = self._messages(prompt, system)

        if self._sdk_client is not None:
            kwargs: dict = {"model": self.model, "messages": messages,
                            "temperature": temperature}
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            if stop:
                kwargs["stop"] = list(stop)
            resp = self._sdk_client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""

        payload: dict = {"model": self.model, "messages": messages,
                         "temperature": temperature}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if stop:
            payload["stop"] = list(stop)
        data = self._post("/chat/completions", payload)
        return data["choices"][0]["message"]["content"] or ""

    def stream(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[Sequence[str]] = None,
        system: Optional[str] = None,
    ) -> Iterator[str]:
        temperature = self.default_temperature if temperature is None else temperature
        payload: dict = {"model": self.model, "messages": self._messages(prompt, system),
                         "temperature": temperature, "stream": True}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if stop:
            payload["stop"] = list(stop)
        url = self.base_url + "/chat/completions"
        with self.client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[len("data:"):].strip()
                if chunk == "[DONE]":
                    break
                try:
                    delta = json.loads(chunk)["choices"][0]["delta"].get("content")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta

    def count_tokens(self, text: str) -> int:
        try:
            import tiktoken  # type: ignore
            try:
                enc = tiktoken.encoding_for_model(self.model)
            except Exception:
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return max(1, len(text) // 4)

    # ------------------------------------------------------------------ http
    def _post(self, path: str, payload: dict) -> dict:
        url = self.base_url + path
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.post(url, json=payload)
                if resp.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"server error {resp.status_code}", request=resp.request,
                        response=resp)
                resp.raise_for_status()
                return resp.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    time.sleep(min(0.5 * (attempt + 1), 5.0))
        raise RuntimeError(
            f"OpenAICompatLLM request to {url} failed after "
            f"{self.max_retries} attempts: {last_exc}")

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
