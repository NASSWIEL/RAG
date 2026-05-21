"""Tests for gemini_client.py."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unittest.mock import MagicMock, patch

# Pre-mock heavy third-party packages (and config which may not exist in test env)
_config_mock = MagicMock()
_config_mock.GOOGLE_API_KEY = "test-api-key"
sys.modules.setdefault("config", _config_mock)
for _pkg in [
    "dotenv",
    "llama_index",
    "llama_index.core",
    "llama_index.llms",
    "llama_index.llms.gemini",
]:
    sys.modules.setdefault(_pkg, MagicMock())

# Evict any stale mock of the module under test
sys.modules.pop("gemini_client", None)

import gemini_client  # noqa: E402
from gemini_client import initialize_gemini_llm  # noqa: E402


def test_initialize_gemini_llm_returns_llm():
    mock_llm = MagicMock()
    with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
        with patch("gemini_client.Gemini", return_value=mock_llm) as mock_gemini_cls:
            with patch("gemini_client.Settings") as mock_settings:
                result = initialize_gemini_llm()
                mock_gemini_cls.assert_called_once_with(
                    api_key="test-key",
                    model="models/gemini-2.5-flash",
                    temperature=0.1,
                )
                assert mock_settings.llm == mock_llm
                assert result is mock_llm


def test_initialize_gemini_llm_sets_settings_llm():
    mock_llm = MagicMock()
    with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}):
        with patch("gemini_client.Gemini", return_value=mock_llm):
            with patch("gemini_client.Settings") as mock_settings:
                initialize_gemini_llm()
                assert mock_settings.llm is mock_llm


def test_initialize_gemini_llm_missing_api_key_raises():
    env = {k: v for k, v in os.environ.items() if k != "GOOGLE_API_KEY"}
    with patch.dict("os.environ", env, clear=True):
        with pytest.raises((KeyError, Exception)):
            initialize_gemini_llm()


# ---------------------------------------------------------------------------
# reset_llm
# ---------------------------------------------------------------------------


def test_reset_llm_sets_state_to_none():
    gemini_client._state["llm"] = MagicMock()
    gemini_client.reset_llm()
    assert gemini_client._state["llm"] is None


# ---------------------------------------------------------------------------
# get_llm
# ---------------------------------------------------------------------------


def test_get_llm_returns_existing_instance():
    mock_llm = MagicMock()
    gemini_client._state["llm"] = mock_llm
    result = gemini_client.get_llm()
    assert result is mock_llm
    gemini_client._state["llm"] = None


def test_get_llm_initializes_when_none(monkeypatch):
    gemini_client._state["llm"] = None
    fake_llm = MagicMock()
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    with patch("gemini_client.Gemini", return_value=fake_llm):
        with patch("gemini_client.Settings"):
            result = gemini_client.get_llm()
    assert result is fake_llm
    gemini_client._state["llm"] = None


# ---------------------------------------------------------------------------
# generate_text
# ---------------------------------------------------------------------------


def test_generate_text_raises_on_empty_prompt():
    with pytest.raises(ValueError, match="prompt must not be empty"):
        gemini_client.generate_text("")


def test_generate_text_raises_on_whitespace_prompt():
    with pytest.raises(ValueError, match="prompt must not be empty"):
        gemini_client.generate_text("   ")


def test_generate_text_returns_string(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "hello world"
    gemini_client._state["llm"] = mock_llm
    result = gemini_client.generate_text("tell me something")
    assert isinstance(result, str)
    mock_llm.complete.assert_called_once()
    gemini_client._state["llm"] = None


def test_generate_text_with_temperature_uses_tmp_instance(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    mock_llm = MagicMock()
    mock_llm.model = "models/gemini-2.5-flash"
    gemini_client._state["llm"] = mock_llm
    tmp_mock = MagicMock()
    tmp_mock.complete.return_value = "result"
    with patch("gemini_client.Gemini", return_value=tmp_mock):
        result = gemini_client.generate_text("prompt", temperature=0.5)
    assert isinstance(result, str)
    tmp_mock.complete.assert_called_once_with("prompt")
    gemini_client._state["llm"] = None


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


def test_summarize_raises_on_empty_text():
    with pytest.raises(ValueError, match="text must not be empty"):
        gemini_client.summarize("")


def test_summarize_calls_generate_text_with_word_count(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "summary output"
    gemini_client._state["llm"] = mock_llm
    result = gemini_client.summarize("Some long text here.", max_words=50)
    assert isinstance(result, str)
    call_args = mock_llm.complete.call_args[0][0]
    assert "50" in call_args
    assert "Some long text here." in call_args
    gemini_client._state["llm"] = None


# ---------------------------------------------------------------------------
# rerank_passages
# ---------------------------------------------------------------------------


def test_rerank_passages_empty_list():
    result = gemini_client.rerank_passages("query", [])
    assert result == []


def test_rerank_passages_valid_reorder(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    passages = ["alpha", "beta", "gamma"]
    mock_llm = MagicMock()
    mock_llm.model = "models/gemini-2.5-flash"
    gemini_client._state["llm"] = mock_llm
    tmp_mock = MagicMock()
    tmp_mock.complete.return_value = "3,1,2"
    with patch("gemini_client.Gemini", return_value=tmp_mock):
        result = gemini_client.rerank_passages("what is gamma?", passages)
    assert result == ["gamma", "alpha", "beta"]
    gemini_client._state["llm"] = None


def test_rerank_passages_fallback_on_bad_response(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    passages = ["a", "b"]
    mock_llm = MagicMock()
    mock_llm.model = "models/gemini-2.5-flash"
    gemini_client._state["llm"] = mock_llm
    tmp_mock = MagicMock()
    tmp_mock.complete.return_value = "not valid numbers!!!"
    with patch("gemini_client.Gemini", return_value=tmp_mock):
        result = gemini_client.rerank_passages("query", passages)
    assert result == passages
    gemini_client._state["llm"] = None
