from __future__ import annotations

import json

import httpx

from arag.llm import OpenAICompatLLM


def _mock_llm(handler):
    return OpenAICompatLLM(base_url="http://localhost:11434/v1/", model="test-model",
                           api_key="ollama", use_sdk=False,
                           transport=httpx.MockTransport(handler))


def test_complete_builds_correct_payload():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "hello there"}}]})

    llm = _mock_llm(handler)
    out = llm.complete("hi", max_tokens=64, system="be nice", stop=["END"])
    assert out == "hello there"
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["auth"] == "Bearer ollama"
    body = captured["body"]
    assert body["model"] == "test-model"
    assert body["max_tokens"] == 64
    assert body["stop"] == ["END"]
    assert body["messages"][0] == {"role": "system", "content": "be nice"}
    assert body["messages"][1] == {"role": "user", "content": "hi"}


def test_retry_then_success():
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(500, json={"error": "transient"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    llm = _mock_llm(handler)
    assert llm.complete("hi") == "ok"
    assert state["n"] == 2  # retried once
