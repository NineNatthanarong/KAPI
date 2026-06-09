from __future__ import annotations

import builtins

import pytest

from arag.tokenize.ngram import CharNgramTokenizer
from arag.tokenize.text import WordTokenizer


def test_word_tokenizer_basic():
    tok = WordTokenizer("english")
    out = tok("The Running Dogs are JUMPING quickly!")
    assert "the" not in out  # stopword removed
    assert "run" in out and "dog" in out and "jump" in out  # stemmed


def test_word_tokenizer_stemmer_backend_present():
    tok = WordTokenizer("english")
    assert tok.stemmer_backend in {"pystemmer", "snowball"}


def test_word_tokenizer_identity_fallback(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name in ("Stemmer", "snowballstemmer"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.warns(RuntimeWarning):
        tok = WordTokenizer("english")
    assert tok.stemmer_backend == "none"
    assert tok("running") == ["running"]  # unstemmed


def test_ngram_tokenizer_trigrams():
    ng = CharNgramTokenizer(3)
    a, b = set(ng("recursion")), set(ng("recurssion"))
    assert len(a & b) >= 5  # heavy trigram overlap -> typo tolerance


def test_ngram_short_string():
    assert CharNgramTokenizer(3)("ab") == ["ab"]
    assert CharNgramTokenizer(3)("") == []
