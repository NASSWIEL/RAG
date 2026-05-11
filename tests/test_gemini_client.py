"""Tests for gemini_client.py."""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Ensure the RAG root is on sys.path so gemini_client can be imported directly
_RAG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RAG_ROOT not in sys.path:
    sys.path.insert(0, _RAG_ROOT)

# Pre-mock heavy third-party packages so the module under test can be imported
for _pkg in [
    "llama_index",
    "llama_index.core",
    "llama_index.llms",
    "llama_index.llms.gemini",
]:
    sys.modules.setdefault(_pkg, MagicMock())

# Pre-mock config dependency
sys.modules.setdefault("config", MagicMock(GOOGLE_API_KEY="fake-api-key"))

# Evict stale mock of the module under test
sys.modules.pop("gemini_client", None)

from gemini_client import initialize_gemini_llm  # noqa: E402


def test_initialize_gemini_llm_returns_llm():
    mock_llm = MagicMock()
    with patch("gemini_client.Gemini", return_value=mock_llm) as mock_gemini_cls, \
         patch("gemini_client.Settings") as mock_settings:
        result = initialize_gemini_llm()
        mock_gemini_cls.assert_called_once_with(
            api_key="fake-api-key",
            model="models/gemini-2.5-flash",
            temperature=0.1,
            max_tokens=1024,
        )
        assert result is mock_llm


def test_initialize_gemini_llm_registers_with_settings():
    mock_llm = MagicMock()
    with patch("gemini_client.Gemini", return_value=mock_llm), \
         patch("gemini_client.Settings") as mock_settings:
        initialize_gemini_llm()
        assert mock_settings.llm is mock_llm


def test_initialize_gemini_llm_uses_correct_model():
    with patch("gemini_client.Gemini") as mock_gemini_cls, \
         patch("gemini_client.Settings"):
        initialize_gemini_llm()
        call_kwargs = mock_gemini_cls.call_args[1]
        assert call_kwargs["model"] == "models/gemini-2.5-flash"
        assert call_kwargs["temperature"] == 0.1


# ---------------------------------------------------------------------------
# Imports for new symbols
# ---------------------------------------------------------------------------
from gemini_client import get_llm, generate_text, summarize, rerank_passages, reset_llm  # noqa: E402
import gemini_client as _gc  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_llm(complete_return="mock response"):
    mock = MagicMock()
    mock.model = "models/gemini-2.5-flash"
    mock.complete.return_value = complete_return
    return mock


# ---------------------------------------------------------------------------
# reset_llm
# ---------------------------------------------------------------------------

def test_reset_llm_sets_instance_to_none():
    _gc._llm_state["instance"] = _make_mock_llm()
    reset_llm()
    assert _gc._llm_state["instance"] is None


# ---------------------------------------------------------------------------
# get_llm
# ---------------------------------------------------------------------------

def test_get_llm_returns_existing_instance():
    mock_llm = _make_mock_llm()
    _gc._llm_state["instance"] = mock_llm
    result = get_llm()
    assert result is mock_llm


def test_get_llm_initializes_when_none():
    _gc._llm_state["instance"] = None
    mock_llm = _make_mock_llm()
    with patch("gemini_client.initialize_gemini_llm", return_value=mock_llm) as mock_init:
        result = get_llm()
        mock_init.assert_called_once()
        assert result is mock_llm


# ---------------------------------------------------------------------------
# generate_text
# ---------------------------------------------------------------------------

def test_generate_text_raises_on_empty_prompt():
    with pytest.raises(ValueError, match="prompt must not be empty"):
        generate_text("")


def test_generate_text_raises_on_whitespace_prompt():
    with pytest.raises(ValueError, match="prompt must not be empty"):
        generate_text("   ")


def test_generate_text_returns_string():
    mock_llm = _make_mock_llm(complete_return="hello world")
    _gc._llm_state["instance"] = mock_llm
    result = generate_text("Tell me something")
    assert isinstance(result, str)
    assert result == "hello world"


def test_generate_text_with_temperature_uses_tmp_instance():
    mock_llm = _make_mock_llm()
    _gc._llm_state["instance"] = mock_llm
    tmp_llm = _make_mock_llm(complete_return="tmp response")
    with patch("gemini_client.Gemini", return_value=tmp_llm):
        result = generate_text("prompt", temperature=0.5)
    assert result == "tmp response"
    tmp_llm.complete.assert_called_once_with("prompt")
    mock_llm.complete.assert_not_called()


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def test_summarize_raises_on_empty_text():
    with pytest.raises(ValueError, match="text must not be empty"):
        summarize("")


def test_summarize_raises_on_whitespace_text():
    with pytest.raises(ValueError, match="text must not be empty"):
        summarize("  ")


def test_summarize_returns_string():
    mock_llm = _make_mock_llm(complete_return="short summary")
    _gc._llm_state["instance"] = mock_llm
    result = summarize("This is a long piece of text that needs summarizing.", max_words=50)
    assert isinstance(result, str)
    assert result == "short summary"


# ---------------------------------------------------------------------------
# rerank_passages
# ---------------------------------------------------------------------------

def test_rerank_passages_empty_list_returns_empty():
    result = rerank_passages("query", [])
    assert result == []


def test_rerank_passages_valid_reorder():
    passages = ["first", "second", "third"]
    mock_llm = _make_mock_llm(complete_return="3,1,2")
    _gc._llm_state["instance"] = mock_llm
    tmp_llm = _make_mock_llm(complete_return="3,1,2")
    with patch("gemini_client.Gemini", return_value=tmp_llm):
        result = rerank_passages("some query", passages)
    assert result == ["third", "first", "second"]


def test_rerank_passages_falls_back_on_bad_model_output():
    passages = ["a", "b", "c"]
    mock_llm = _make_mock_llm(complete_return="not valid output!!!")
    _gc._llm_state["instance"] = mock_llm
    result = rerank_passages("query", passages)
    assert result == passages


def test_rerank_passages_appends_omitted_passages():
    passages = ["alpha", "beta", "gamma"]
    mock_llm = _make_mock_llm(complete_return="2,1")
    _gc._llm_state["instance"] = mock_llm
    tmp_llm = _make_mock_llm(complete_return="2,1")
    with patch("gemini_client.Gemini", return_value=tmp_llm):
        result = rerank_passages("query", passages)
    assert result[0] == "beta"
    assert result[1] == "alpha"
    assert result[2] == "gamma"
