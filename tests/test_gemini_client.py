"""Tests for gemini_client.py."""
import sys
from unittest.mock import MagicMock, patch

# Pre-mock heavy third-party packages so the module under test can be imported
for _pkg in [
    "llama_index",
    "llama_index.core",
    "llama_index.llms",
    "llama_index.llms.gemini",
]:
    sys.modules.setdefault(_pkg, MagicMock())

# Evict any stale mock of the module under test from a previous test file
sys.modules.pop("gemini_client", None)

import pytest

from gemini_client import initialize_gemini_llm


def test_initialize_gemini_llm_returns_llm_instance(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")

    mock_gemini_cls = MagicMock()
    mock_llm = MagicMock()
    mock_gemini_cls.return_value = mock_llm

    mock_settings = MagicMock()

    with patch("gemini_client.Gemini", mock_gemini_cls), \
         patch("gemini_client.Settings", mock_settings):
        result = initialize_gemini_llm()

    assert result is mock_llm
    mock_gemini_cls.assert_called_once_with(
        api_key="test-api-key",
        model="models/gemini-2.5-flash",
        temperature=0.1,
    )


def test_initialize_gemini_llm_sets_settings_llm(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")

    mock_gemini_cls = MagicMock()
    mock_llm = MagicMock()
    mock_gemini_cls.return_value = mock_llm

    mock_settings = MagicMock()

    with patch("gemini_client.Gemini", mock_gemini_cls), \
         patch("gemini_client.Settings", mock_settings):
        initialize_gemini_llm()

    assert mock_settings.llm is mock_llm


def test_initialize_gemini_llm_missing_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(KeyError):
        initialize_gemini_llm()


import gemini_client  # noqa: E402 — module already imported above, this is a reference alias
from gemini_client import (  # noqa: E402
    generate_text,
    get_llm,
    rerank_passages,
    reset_llm,
    summarize,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_mock(complete_return="mocked response"):
    llm = MagicMock()
    llm.model = "models/gemini-2.5-flash"
    llm.complete.return_value = complete_return
    return llm


# ---------------------------------------------------------------------------
# get_llm
# ---------------------------------------------------------------------------

def test_get_llm_returns_existing_singleton():
    mock_llm = _make_llm_mock()
    gemini_client._state["llm"] = mock_llm
    result = get_llm()
    assert result is mock_llm
    gemini_client._state["llm"] = None


def test_get_llm_initializes_when_none(monkeypatch):
    gemini_client._state["llm"] = None
    mock_llm = _make_llm_mock()

    def fake_init(**kwargs):
        gemini_client._state["llm"] = mock_llm
        return mock_llm

    monkeypatch.setattr(gemini_client, "initialize_gemini_llm", fake_init)
    result = get_llm()
    assert result is mock_llm


# ---------------------------------------------------------------------------
# reset_llm
# ---------------------------------------------------------------------------

def test_reset_llm_clears_singleton():
    gemini_client._state["llm"] = _make_llm_mock()
    reset_llm()
    assert gemini_client._state["llm"] is None


def test_reset_llm_idempotent():
    gemini_client._state["llm"] = None
    reset_llm()
    assert gemini_client._state["llm"] is None


# ---------------------------------------------------------------------------
# generate_text
# ---------------------------------------------------------------------------

def test_generate_text_returns_response(monkeypatch):
    mock_llm = _make_llm_mock("hello world")
    monkeypatch.setattr(gemini_client, "get_llm", lambda: mock_llm)
    result = generate_text("tell me something")
    assert result == "hello world"
    mock_llm.complete.assert_called_once_with("tell me something")


def test_generate_text_empty_prompt_raises():
    with pytest.raises(ValueError):
        generate_text("")


def test_generate_text_whitespace_prompt_raises():
    with pytest.raises(ValueError):
        generate_text("   ")


def test_generate_text_with_temperature_uses_tmp_instance(monkeypatch):
    mock_llm = _make_llm_mock()
    monkeypatch.setattr(gemini_client, "get_llm", lambda: mock_llm)

    tmp_llm = MagicMock()
    tmp_llm.complete.return_value = "temp response"

    mock_gemini_cls = MagicMock(return_value=tmp_llm)
    monkeypatch.setattr(gemini_client, "Gemini", mock_gemini_cls)
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    result = generate_text("prompt", temperature=0.5)
    assert result == "temp response"
    mock_gemini_cls.assert_called_once()
    tmp_llm.complete.assert_called_once_with("prompt")


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def test_summarize_calls_generate_text(monkeypatch):
    monkeypatch.setattr(gemini_client, "generate_text", lambda prompt: "summary result")
    result = summarize("Some long text here.")
    assert result == "summary result"


def test_summarize_empty_text_raises():
    with pytest.raises(ValueError):
        summarize("")


def test_summarize_whitespace_only_raises():
    with pytest.raises(ValueError):
        summarize("   ")


def test_summarize_prompt_includes_max_words(monkeypatch):
    captured = {}

    def fake_generate(prompt):
        captured["prompt"] = prompt
        return "ok"

    monkeypatch.setattr(gemini_client, "generate_text", fake_generate)
    summarize("some text", max_words=50)
    assert "50" in captured["prompt"]
    assert "some text" in captured["prompt"]


# ---------------------------------------------------------------------------
# rerank_passages
# ---------------------------------------------------------------------------

def test_rerank_passages_empty_returns_empty():
    assert rerank_passages("query", []) == []


def test_rerank_passages_reorders_by_llm(monkeypatch):
    monkeypatch.setattr(gemini_client, "generate_text", lambda prompt, temperature=None: "2,1,3")
    passages = ["a", "b", "c"]
    result = rerank_passages("q", passages)
    assert result == ["b", "a", "c"]


def test_rerank_passages_falls_back_on_bad_llm_output(monkeypatch):
    monkeypatch.setattr(gemini_client, "generate_text", lambda prompt, temperature=None: "not,valid,numbers")
    passages = ["x", "y"]
    result = rerank_passages("q", passages)
    assert result == passages


def test_rerank_passages_handles_out_of_range(monkeypatch):
    monkeypatch.setattr(gemini_client, "generate_text", lambda prompt, temperature=None: "99,1")
    passages = ["alpha", "beta"]
    result = rerank_passages("q", passages)
    assert "beta" in result
    assert "alpha" in result


def test_rerank_passages_appends_omitted(monkeypatch):
    monkeypatch.setattr(gemini_client, "generate_text", lambda prompt, temperature=None: "2")
    passages = ["first", "second"]
    result = rerank_passages("q", passages)
    assert result[0] == "second"
    assert result[1] == "first"
