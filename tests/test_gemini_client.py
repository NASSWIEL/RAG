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

import os
import pytest

from gemini_client import initialize_gemini_llm


def test_initialize_gemini_llm_returns_llm(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")

    mock_llm_instance = MagicMock()
    mock_gemini_cls = MagicMock(return_value=mock_llm_instance)
    mock_settings = MagicMock()

    with patch("gemini_client.Gemini", mock_gemini_cls), \
         patch("gemini_client.Settings", mock_settings):
        result = initialize_gemini_llm()

    assert result is mock_llm_instance
    mock_gemini_cls.assert_called_once_with(
        api_key="test-key-123",
        model="models/gemini-2.5-flash",
        temperature=0.1,
        max_tokens=1024,
    )


def test_initialize_gemini_llm_sets_settings_llm(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-abc")

    mock_llm_instance = MagicMock()
    mock_gemini_cls = MagicMock(return_value=mock_llm_instance)
    mock_settings = MagicMock()

    with patch("gemini_client.Gemini", mock_gemini_cls), \
         patch("gemini_client.Settings", mock_settings):
        initialize_gemini_llm()

    assert mock_settings.llm is mock_llm_instance


def test_initialize_gemini_llm_missing_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(KeyError):
        initialize_gemini_llm()


import gemini_client
from gemini_client import get_llm, generate_text, summarize, rerank_passages, reset_llm


# ---------------------------------------------------------------------------
# get_llm
# ---------------------------------------------------------------------------

def test_get_llm_returns_existing_instance():
    mock_llm = MagicMock()
    gemini_client._state["llm"] = mock_llm
    result = get_llm()
    assert result is mock_llm
    gemini_client._state["llm"] = None


def test_get_llm_initializes_when_none(monkeypatch):
    gemini_client._state["llm"] = None
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_llm = MagicMock()
    with patch("gemini_client.Gemini", return_value=mock_llm), \
         patch("gemini_client.Settings"):
        result = get_llm()
    assert result is mock_llm
    gemini_client._state["llm"] = None


# ---------------------------------------------------------------------------
# generate_text
# ---------------------------------------------------------------------------

def test_generate_text_raises_on_empty_prompt():
    with pytest.raises(ValueError, match="prompt must not be empty"):
        generate_text("")


def test_generate_text_raises_on_whitespace_prompt():
    with pytest.raises(ValueError, match="prompt must not be empty"):
        generate_text("   ")


def test_generate_text_returns_string(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "some response"
    gemini_client._state["llm"] = mock_llm
    result = generate_text("Hello world")
    assert isinstance(result, str)
    mock_llm.complete.assert_called_once_with("Hello world")
    gemini_client._state["llm"] = None


def test_generate_text_with_temperature_uses_tmp_instance(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_llm = MagicMock()
    mock_llm.model = "models/gemini-2.5-flash"
    gemini_client._state["llm"] = mock_llm
    tmp_mock = MagicMock()
    tmp_mock.complete.return_value = "temp response"
    with patch("gemini_client.Gemini", return_value=tmp_mock) as mock_gemini_cls:
        result = generate_text("hello", temperature=0.5)
    mock_gemini_cls.assert_called_once_with(
        api_key="test-key", model="models/gemini-2.5-flash", temperature=0.5
    )
    assert "temp response" in result
    gemini_client._state["llm"] = None


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def test_summarize_raises_on_empty_text():
    with pytest.raises(ValueError, match="text must not be empty"):
        summarize("")


def test_summarize_raises_on_whitespace_text():
    with pytest.raises(ValueError, match="text must not be empty"):
        summarize("  \n  ")


def test_summarize_calls_generate_text(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "A short summary."
    gemini_client._state["llm"] = mock_llm
    result = summarize("Long text about something important.", max_words=50)
    assert isinstance(result, str)
    call_args = mock_llm.complete.call_args[0][0]
    assert "Long text about something important." in call_args
    assert "50" in call_args
    gemini_client._state["llm"] = None


# ---------------------------------------------------------------------------
# rerank_passages
# ---------------------------------------------------------------------------

def test_rerank_passages_empty_input():
    assert rerank_passages("query", []) == []


def test_rerank_passages_reorders_correctly(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_llm = MagicMock()
    mock_llm.model = "models/gemini-2.5-flash"
    tmp_mock = MagicMock()
    tmp_mock.complete.return_value = "3,1,2"
    gemini_client._state["llm"] = mock_llm
    with patch("gemini_client.Gemini", return_value=tmp_mock):
        result = rerank_passages("What is Python?", ["passage A", "passage B", "passage C"])
    assert result == ["passage C", "passage A", "passage B"]
    gemini_client._state["llm"] = None


def test_rerank_passages_fallback_on_invalid_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_llm = MagicMock()
    mock_llm.model = "models/gemini-2.5-flash"
    tmp_mock = MagicMock()
    tmp_mock.complete.return_value = "not valid numbers!!"
    gemini_client._state["llm"] = mock_llm
    passages = ["alpha", "beta", "gamma"]
    with patch("gemini_client.Gemini", return_value=tmp_mock):
        result = rerank_passages("query", passages)
    assert result == passages
    gemini_client._state["llm"] = None


def test_rerank_passages_appends_omitted(monkeypatch):
    """Model returns only one index; missing passages should be appended at the end."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_llm = MagicMock()
    mock_llm.model = "models/gemini-2.5-flash"
    tmp_mock = MagicMock()
    tmp_mock.complete.return_value = "2"
    gemini_client._state["llm"] = mock_llm
    passages = ["first", "second", "third"]
    with patch("gemini_client.Gemini", return_value=tmp_mock):
        result = rerank_passages("query", passages)
    assert result[0] == "second"
    assert set(result) == set(passages)
    gemini_client._state["llm"] = None


# ---------------------------------------------------------------------------
# reset_llm
# ---------------------------------------------------------------------------

def test_reset_llm_clears_state():
    mock_llm = MagicMock()
    gemini_client._state["llm"] = mock_llm
    reset_llm()
    assert gemini_client._state["llm"] is None
