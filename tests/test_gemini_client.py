"""Tests for C:/retrodoc/test-BT-AI/RAG/gemini_client.py."""
import os
import sys
from unittest.mock import MagicMock, patch

# Ensure the project root is on sys.path so gemini_client can be imported
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Pre-mock heavy third-party packages so the module under test can be imported
for _pkg in [
    "llama_index",
    "llama_index.core",
    "llama_index.llms",
    "llama_index.llms.gemini",
]:
    sys.modules.setdefault(_pkg, MagicMock())

# Pre-mock config module
sys.modules.setdefault("config", MagicMock(GOOGLE_API_KEY="fake-api-key"))

import gemini_client  # noqa: E402
from gemini_client import (  # noqa: E402
    generate_text,
    get_llm,
    initialize_gemini_llm,
    rerank_passages,
    reset_llm,
    summarize,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_llm(response_text="mocked response"):
    mock_llm = MagicMock()
    mock_llm.model = "models/gemini-2.5-flash"
    mock_llm.complete.return_value = MagicMock(__str__=lambda self: response_text)
    return mock_llm


def test_initialize_gemini_llm_returns_llm():
    """initialize_gemini_llm should return a Gemini LLM instance."""
    mock_llm = MagicMock()
    with patch("gemini_client.Gemini", autospec=False, return_value=mock_llm) as mock_gemini_cls, \
         patch("gemini_client.Settings") as mock_settings:
        result = initialize_gemini_llm()
        mock_gemini_cls.assert_called_once_with(
            api_key="fake-api-key",
            model="models/gemini-2.5-flash",
            temperature=0.1,
            max_tokens=1024,
        )
        assert result is mock_llm


def test_initialize_gemini_llm_assigns_to_settings():
    """initialize_gemini_llm should assign the LLM to Settings.llm."""
    mock_llm = MagicMock()
    with patch("gemini_client.Gemini", autospec=False, return_value=mock_llm), \
         patch("gemini_client.Settings") as mock_settings:
        initialize_gemini_llm()
        assert mock_settings.llm == mock_llm


def test_initialize_gemini_llm_uses_api_key():
    """initialize_gemini_llm should pass the configured GOOGLE_API_KEY to Gemini."""
    mock_llm = MagicMock()
    with patch("gemini_client.Gemini", autospec=False, return_value=mock_llm) as mock_gemini_cls, \
         patch("gemini_client.Settings"), \
         patch("gemini_client.GOOGLE_API_KEY", "test-key-123"):
        initialize_gemini_llm()
        call_kwargs = mock_gemini_cls.call_args.kwargs
        assert call_kwargs["api_key"] == "test-key-123"


# ---------------------------------------------------------------------------
# reset_llm
# ---------------------------------------------------------------------------

def test_reset_llm_clears_cache():
    gemini_client._llm_cache.clear()
    gemini_client._llm_cache.append(_make_mock_llm())
    assert len(gemini_client._llm_cache) == 1
    reset_llm()
    assert len(gemini_client._llm_cache) == 0


def test_reset_llm_idempotent_on_empty_cache():
    gemini_client._llm_cache.clear()
    reset_llm()
    assert len(gemini_client._llm_cache) == 0


# ---------------------------------------------------------------------------
# get_llm
# ---------------------------------------------------------------------------

def test_get_llm_returns_existing_instance():
    gemini_client._llm_cache.clear()
    mock_llm = _make_mock_llm()
    gemini_client._llm_cache.append(mock_llm)
    result = get_llm()
    assert result is mock_llm


def test_get_llm_initializes_when_cache_empty():
    gemini_client._llm_cache.clear()
    mock_llm = _make_mock_llm()
    with patch("gemini_client.initialize_gemini_llm", autospec=True, return_value=mock_llm) as mock_init:
        result = get_llm()
        mock_init.assert_called_once()
        assert result is mock_llm


# ---------------------------------------------------------------------------
# generate_text
# ---------------------------------------------------------------------------

def test_generate_text_raises_on_empty_prompt():
    import pytest
    with pytest.raises(ValueError, match="prompt must not be empty"):
        generate_text("")


def test_generate_text_raises_on_whitespace_prompt():
    import pytest
    with pytest.raises(ValueError, match="prompt must not be empty"):
        generate_text("   ")


def test_generate_text_returns_string():
    gemini_client._llm_cache.clear()
    mock_llm = _make_mock_llm("hello world")
    gemini_client._llm_cache.append(mock_llm)
    result = generate_text("What is 2+2?")
    assert isinstance(result, str)
    assert result == "hello world"
    mock_llm.complete.assert_called_once_with("What is 2+2?")


def test_generate_text_with_temperature_override():
    gemini_client._llm_cache.clear()
    mock_llm = _make_mock_llm("temp override response")
    gemini_client._llm_cache.append(mock_llm)

    mock_tmp = MagicMock()
    mock_tmp.complete.return_value = MagicMock(__str__=lambda self: "temp override response")

    with patch("gemini_client.Gemini", autospec=False, return_value=mock_tmp):
        result = generate_text("hello", temperature=0.5)
    assert result == "temp override response"
    mock_tmp.complete.assert_called_once_with("hello")


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def test_summarize_raises_on_empty_text():
    import pytest
    with pytest.raises(ValueError, match="text must not be empty"):
        summarize("")


def test_summarize_raises_on_whitespace_text():
    import pytest
    with pytest.raises(ValueError, match="text must not be empty"):
        summarize("   ")


def test_summarize_calls_generate_text():
    with patch("gemini_client.generate_text", autospec=True, return_value="summary") as mock_gen:
        result = summarize("Some long article text here.", max_words=50)
        assert result == "summary"
        mock_gen.assert_called_once()
        call_args = mock_gen.call_args[0][0]
        assert "50" in call_args
        assert "Some long article text here." in call_args


# ---------------------------------------------------------------------------
# rerank_passages
# ---------------------------------------------------------------------------

def test_rerank_passages_empty_list():
    result = rerank_passages("any query", [])
    assert result == []


def test_rerank_passages_valid_reorder():
    passages = ["passage A", "passage B", "passage C"]
    with patch("gemini_client.generate_text", autospec=True, return_value="3,1,2"):
        result = rerank_passages("my query", passages)
    assert result == ["passage C", "passage A", "passage B"]


def test_rerank_passages_falls_back_on_parse_failure():
    passages = ["passage A", "passage B"]
    with patch("gemini_client.generate_text", autospec=True, return_value="not valid at all"):
        result = rerank_passages("my query", passages)
    assert result == passages


def test_rerank_passages_handles_out_of_range_indices():
    passages = ["passage A", "passage B"]
    with patch("gemini_client.generate_text", autospec=True, return_value="1,99,2"):
        result = rerank_passages("my query", passages)
    assert result == ["passage A", "passage B"]


def test_rerank_passages_appends_omitted_passages():
    passages = ["passage A", "passage B", "passage C"]
    with patch("gemini_client.generate_text", autospec=True, return_value="2"):
        result = rerank_passages("my query", passages)
    assert result[0] == "passage B"
    assert set(result) == set(passages)
