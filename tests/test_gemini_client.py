"""Tests for gemini_client.py."""

import os
import sys
from unittest.mock import MagicMock, patch

# Ensure the repo root is on sys.path so gemini_client is importable
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Pre-mock heavy third-party packages so the module under test can be imported
for _pkg in [
    "llama_index",
    "llama_index.llms",
    "llama_index.llms.gemini",
    "llama_index.core",
]:
    sys.modules.setdefault(_pkg, MagicMock())

# Pre-mock config so it doesn't need a real .env / environment
_config_mock = MagicMock()
_config_mock.GOOGLE_API_KEY = "test-api-key"
sys.modules.setdefault("config", _config_mock)

# Evict any stale mock of the module under test from a previous test file
sys.modules.pop("gemini_client", None)

from gemini_client import initialize_gemini_llm, get_llm, generate_text, summarize, rerank_passages, reset_llm  # noqa: E402


def test_initialize_gemini_llm_returns_gemini_instance():
    mock_llm = MagicMock()
    mock_gemini_cls = MagicMock(return_value=mock_llm)

    with patch("gemini_client.Gemini", mock_gemini_cls):
        with patch("gemini_client.Settings"):
            result = initialize_gemini_llm()

    mock_gemini_cls.assert_called_once_with(
        api_key="test-api-key",
        model="models/gemini-2.5-flash",
        temperature=0.1,
        max_tokens=1024,
    )
    assert result is mock_llm


def test_initialize_gemini_llm_sets_settings_llm():
    mock_llm = MagicMock()
    mock_gemini_cls = MagicMock(return_value=mock_llm)

    with patch("gemini_client.Gemini", mock_gemini_cls):
        with patch("gemini_client.Settings") as mock_settings:
            initialize_gemini_llm()
            assert mock_settings.llm is mock_llm


def test_initialize_gemini_llm_uses_configured_api_key():
    mock_llm = MagicMock()
    mock_gemini_cls = MagicMock(return_value=mock_llm)

    with patch("gemini_client.Gemini", mock_gemini_cls):
        with patch("gemini_client.Settings"):
            with patch("gemini_client.GOOGLE_API_KEY", "custom-key-123"):
                initialize_gemini_llm()

    call_kwargs = mock_gemini_cls.call_args[1]
    assert call_kwargs["api_key"] == "custom-key-123"


# ---------------------------------------------------------------------------
# get_llm
# ---------------------------------------------------------------------------

def test_get_llm_initializes_when_none():
    reset_llm()
    mock_llm = MagicMock()
    mock_gemini_cls = MagicMock(return_value=mock_llm)
    with patch("gemini_client.Gemini", mock_gemini_cls):
        with patch("gemini_client.Settings"):
            result = get_llm()
    assert result is mock_llm


def test_get_llm_returns_existing_instance():
    reset_llm()
    mock_llm = MagicMock()
    mock_gemini_cls = MagicMock(return_value=mock_llm)
    with patch("gemini_client.Gemini", mock_gemini_cls):
        with patch("gemini_client.Settings"):
            first = get_llm()
            second = get_llm()
    assert first is second
    # Gemini constructor called exactly once across both calls
    assert mock_gemini_cls.call_count == 1


# ---------------------------------------------------------------------------
# generate_text
# ---------------------------------------------------------------------------

def test_generate_text_returns_string():
    reset_llm()
    mock_llm = MagicMock()
    mock_llm.model = "models/gemini-2.5-flash"
    mock_response = MagicMock()
    mock_response.__str__ = lambda self: "Hello world"
    mock_llm.complete.return_value = mock_response

    mock_gemini_cls = MagicMock(return_value=mock_llm)
    with patch("gemini_client.Gemini", mock_gemini_cls):
        with patch("gemini_client.Settings"):
            result = generate_text("Say hello")

    assert result == "Hello world"


def test_generate_text_raises_on_empty_prompt():
    import pytest
    with pytest.raises(ValueError, match="prompt must not be empty"):
        generate_text("")


def test_generate_text_raises_on_blank_prompt():
    import pytest
    with pytest.raises(ValueError, match="prompt must not be empty"):
        generate_text("   ")


def test_generate_text_with_temperature_override():
    reset_llm()
    mock_llm = MagicMock()
    mock_llm.model = "models/gemini-2.5-flash"
    mock_tmp = MagicMock()
    mock_response = MagicMock()
    mock_response.__str__ = lambda self: "Temp result"
    mock_tmp.complete.return_value = mock_response

    def gemini_factory(**kwargs):
        if kwargs.get("temperature") == 0.5:
            return mock_tmp
        return mock_llm

    with patch("gemini_client.Gemini", side_effect=gemini_factory):
        with patch("gemini_client.Settings"):
            result = generate_text("A prompt", temperature=0.5)

    assert result == "Temp result"
    mock_tmp.complete.assert_called_once_with("A prompt")


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def test_summarize_raises_on_empty_text():
    import pytest
    with pytest.raises(ValueError, match="text must not be empty"):
        summarize("")


def test_summarize_raises_on_blank_text():
    import pytest
    with pytest.raises(ValueError, match="text must not be empty"):
        summarize("  ")


def test_summarize_calls_generate_text_with_prompt():
    with patch("gemini_client.generate_text", return_value="Summary here") as mock_gt:
        result = summarize("Some long text to summarize", max_words=50)
    assert result == "Summary here"
    call_args = mock_gt.call_args[0][0]
    assert "50" in call_args
    assert "Some long text to summarize" in call_args


# ---------------------------------------------------------------------------
# rerank_passages
# ---------------------------------------------------------------------------

def test_rerank_passages_empty_list():
    result = rerank_passages("query", [])
    assert result == []


def test_rerank_passages_reorders_correctly():
    passages = ["passage A", "passage B", "passage C"]
    with patch("gemini_client.generate_text", return_value="3,1,2"):
        result = rerank_passages("my query", passages)
    assert result == ["passage C", "passage A", "passage B"]


def test_rerank_passages_falls_back_on_bad_output():
    passages = ["p1", "p2"]
    with patch("gemini_client.generate_text", return_value="not,valid,numbers"):
        result = rerank_passages("query", passages)
    assert result == passages


def test_rerank_passages_handles_out_of_range_indices():
    passages = ["p1", "p2"]
    with patch("gemini_client.generate_text", return_value="1,99,2"):
        result = rerank_passages("query", passages)
    # 99 is out-of-range, should be silently dropped
    assert set(result) == {"p1", "p2"}


# ---------------------------------------------------------------------------
# reset_llm
# ---------------------------------------------------------------------------

def test_reset_llm_clears_singleton():
    mock_llm = MagicMock()
    mock_gemini_cls = MagicMock(return_value=mock_llm)
    with patch("gemini_client.Gemini", mock_gemini_cls):
        with patch("gemini_client.Settings"):
            get_llm()  # populate singleton

    reset_llm()

    # After reset, get_llm must call Gemini constructor again
    mock_llm2 = MagicMock()
    mock_gemini_cls2 = MagicMock(return_value=mock_llm2)
    with patch("gemini_client.Gemini", mock_gemini_cls2):
        with patch("gemini_client.Settings"):
            result = get_llm()

    assert result is mock_llm2
    mock_gemini_cls2.assert_called_once()
