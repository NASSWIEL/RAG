"""Tests for C:/retrodoc/test-BT-AI/RAG/gemini_client.py."""
import sys
import os
from unittest.mock import MagicMock, patch

# Ensure the RAG root is on sys.path so gemini_client can be imported
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

# Evict any stale mock of the module under test from a previous test file
sys.modules.pop("gemini_client", None)

from gemini_client import initialize_gemini_llm  # noqa: E402


def test_initialize_gemini_llm_returns_llm_instance():
    """initialize_gemini_llm should return the Gemini LLM instance."""
    mock_llm = MagicMock()
    mock_gemini_cls = MagicMock(return_value=mock_llm)
    mock_settings = MagicMock()

    with patch("gemini_client.Gemini", mock_gemini_cls), \
         patch("gemini_client.Settings", mock_settings):
        result = initialize_gemini_llm()

    assert result is mock_llm


def test_initialize_gemini_llm_registers_with_settings():
    """initialize_gemini_llm should assign the llm to Settings.llm."""
    mock_llm = MagicMock()
    mock_gemini_cls = MagicMock(return_value=mock_llm)
    mock_settings = MagicMock()

    with patch("gemini_client.Gemini", mock_gemini_cls), \
         patch("gemini_client.Settings", mock_settings):
        initialize_gemini_llm()

    assert mock_settings.llm is mock_llm


def test_initialize_gemini_llm_uses_correct_model_params():
    """initialize_gemini_llm should construct Gemini with the expected params."""
    mock_gemini_cls = MagicMock(return_value=MagicMock())
    mock_settings = MagicMock()

    with patch("gemini_client.Gemini", mock_gemini_cls), \
         patch("gemini_client.Settings", mock_settings):
        initialize_gemini_llm()

    _, kwargs = mock_gemini_cls.call_args
    assert kwargs.get("model") == "models/gemini-2.5-flash"
    assert kwargs.get("temperature") == 0.1


# ---------------------------------------------------------------------------
# get_llm
# ---------------------------------------------------------------------------

import gemini_client as _gc  # noqa: E402


def test_get_llm_returns_existing_instance():
    """If _llm_instance[0] is already set, get_llm returns it without re-init."""
    mock_llm = MagicMock()
    original = _gc._llm_instance[0]
    try:
        _gc._llm_instance[0] = mock_llm
        result = _gc.get_llm()
        assert result is mock_llm
    finally:
        _gc._llm_instance[0] = original


def test_get_llm_initializes_when_none():
    """get_llm calls initialize_gemini_llm when the singleton is None."""
    original = _gc._llm_instance[0]
    try:
        _gc._llm_instance[0] = None
        mock_llm = MagicMock()
        with patch("gemini_client.initialize_gemini_llm", return_value=mock_llm) as mock_init:
            result = _gc.get_llm()
        mock_init.assert_called_once()
        assert result is mock_llm
    finally:
        _gc._llm_instance[0] = original


# ---------------------------------------------------------------------------
# generate_text
# ---------------------------------------------------------------------------

import pytest  # noqa: E402


def test_generate_text_raises_on_empty_prompt():
    """generate_text raises ValueError when prompt is empty."""
    with pytest.raises(ValueError, match="prompt must not be empty"):
        _gc.generate_text("")


def test_generate_text_raises_on_whitespace_prompt():
    """generate_text raises ValueError when prompt is whitespace-only."""
    with pytest.raises(ValueError, match="prompt must not be empty"):
        _gc.generate_text("   ")


def test_generate_text_returns_string():
    """generate_text calls llm.complete and returns str of the response."""
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "mocked response"
    with patch("gemini_client.get_llm", return_value=mock_llm):
        result = _gc.generate_text("hello world")
    assert isinstance(result, str)
    mock_llm.complete.assert_called_once_with("hello world")


def test_generate_text_with_temperature_uses_temp_instance():
    """generate_text with temperature override creates a new Gemini instance."""
    mock_llm = MagicMock()
    mock_llm.model = "models/gemini-2.5-flash"
    mock_tmp = MagicMock()
    mock_tmp.complete.return_value = "temp response"
    with patch("gemini_client.get_llm", return_value=mock_llm):
        with patch("gemini_client.Gemini", return_value=mock_tmp):
            result = _gc.generate_text("hello", temperature=0.5)
    assert isinstance(result, str)
    mock_tmp.complete.assert_called_once_with("hello")


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def test_summarize_raises_on_empty_text():
    """summarize raises ValueError when text is empty."""
    with pytest.raises(ValueError, match="text must not be empty"):
        _gc.summarize("")


def test_summarize_raises_on_whitespace_text():
    """summarize raises ValueError when text is whitespace-only."""
    with pytest.raises(ValueError, match="text must not be empty"):
        _gc.summarize("  ")


def test_summarize_calls_generate_text():
    """summarize delegates to generate_text with a constructed prompt."""
    with patch("gemini_client.generate_text", return_value="summary") as mock_gen:
        result = _gc.summarize("Some long text here.", max_words=50)
    assert result == "summary"
    mock_gen.assert_called_once()
    prompt_arg = mock_gen.call_args[0][0]
    assert "50" in prompt_arg
    assert "Some long text here." in prompt_arg


# ---------------------------------------------------------------------------
# rerank_passages
# ---------------------------------------------------------------------------

def test_rerank_passages_empty_list():
    """rerank_passages returns empty list when passages is empty."""
    result = _gc.rerank_passages("query", [])
    assert result == []


def test_rerank_passages_reorders_correctly():
    """rerank_passages reorders passages according to LLM output."""
    passages = ["alpha", "beta", "gamma"]
    with patch("gemini_client.generate_text", return_value="3,1,2"):
        result = _gc.rerank_passages("my query", passages)
    assert result == ["gamma", "alpha", "beta"]


def test_rerank_passages_falls_back_on_invalid_response():
    """rerank_passages returns original order when LLM output is unparseable."""
    passages = ["alpha", "beta"]
    with patch("gemini_client.generate_text", return_value="not a number list"):
        result = _gc.rerank_passages("query", passages)
    assert result == passages


def test_rerank_passages_handles_partial_indices():
    """rerank_passages appends omitted passages when model returns partial list."""
    passages = ["alpha", "beta", "gamma"]
    with patch("gemini_client.generate_text", return_value="2"):
        result = _gc.rerank_passages("query", passages)
    assert result[0] == "beta"
    assert set(result) == set(passages)


# ---------------------------------------------------------------------------
# reset_llm
# ---------------------------------------------------------------------------

def test_reset_llm_sets_singleton_to_none():
    """reset_llm resets _llm_instance[0] to None."""
    _gc._llm_instance[0] = MagicMock()
    _gc.reset_llm()
    assert _gc._llm_instance[0] is None


def test_reset_llm_idempotent():
    """reset_llm can be called multiple times without error."""
    _gc._llm_instance[0] = None
    _gc.reset_llm()
    assert _gc._llm_instance[0] is None
