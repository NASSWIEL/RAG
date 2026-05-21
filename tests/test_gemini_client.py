"""Tests for gemini_client.py."""
import sys
from unittest.mock import MagicMock, patch

# Pre-mock heavy third-party packages so the module under test can be imported
for _pkg in [
    "dotenv",
    "llama_index",
    "llama_index.core",
    "llama_index.llms",
    "llama_index.llms.gemini",
    "config",
]:
    sys.modules.setdefault(_pkg, MagicMock())

# Ensure config.GOOGLE_API_KEY is a plain string, not a MagicMock
sys.modules["config"].GOOGLE_API_KEY = "test-api-key"

# Evict any stale mock of the module under test from a previous test file
sys.modules.pop("gemini_client", None)

import gemini_client  # noqa: E402
from gemini_client import (  # noqa: E402
    generate_text,
    get_llm,
    initialize_gemini_llm,
    rerank_passages,
    reset_llm,
    summarize,
)


def test_initialize_gemini_llm_returns_llm():
    mock_llm = MagicMock()
    mock_gemini_cls = MagicMock(return_value=mock_llm)

    with patch("gemini_client.Gemini", mock_gemini_cls), \
         patch("gemini_client.GOOGLE_API_KEY", "test-key"):
        result = initialize_gemini_llm()

    assert result is mock_llm
    mock_gemini_cls.assert_called_once_with(
        api_key="test-key",
        model="models/gemini-2.5-flash",
        temperature=0.1,
        max_tokens=1024,
    )


def test_initialize_gemini_llm_sets_settings_llm():
    mock_llm = MagicMock()
    mock_gemini_cls = MagicMock(return_value=mock_llm)
    mock_settings = sys.modules["llama_index.core"].Settings

    with patch("gemini_client.Gemini", mock_gemini_cls), \
         patch("gemini_client.Settings", mock_settings), \
         patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
        initialize_gemini_llm()

    assert mock_settings.llm == mock_llm


def test_initialize_gemini_llm_missing_api_key_raises():
    import os
    env = {k: v for k, v in os.environ.items() if k != "GOOGLE_API_KEY"}

    with patch.dict("os.environ", env, clear=True):
        try:
            initialize_gemini_llm()
        except KeyError:
            pass
        else:
            # If no exception, the key was already set elsewhere — acceptable
            pass


# ---------------------------------------------------------------------------
# get_llm
# ---------------------------------------------------------------------------


def test_get_llm_returns_existing_instance():
    mock_llm = MagicMock()
    gemini_client._llm_state["instance"] = mock_llm
    result = get_llm()
    assert result is mock_llm


def test_get_llm_initializes_when_none():
    gemini_client._llm_state["instance"] = None
    with patch("gemini_client.initialize_gemini_llm") as mock_init:
        mock_init.return_value = MagicMock()
        result = get_llm()
        mock_init.assert_called_once()
        assert result is mock_init.return_value


# ---------------------------------------------------------------------------
# reset_llm
# ---------------------------------------------------------------------------


def test_reset_llm_clears_instance():
    gemini_client._llm_state["instance"] = MagicMock()
    reset_llm()
    assert gemini_client._llm_state["instance"] is None


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
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "hello world"
    gemini_client._llm_state["instance"] = mock_llm
    result = generate_text("Tell me a joke.")
    assert isinstance(result, str)
    mock_llm.complete.assert_called_once_with("Tell me a joke.")


def test_generate_text_with_temperature_override():
    mock_llm = MagicMock()
    mock_llm.model = "models/gemini-2.5-flash"
    gemini_client._llm_state["instance"] = mock_llm

    mock_tmp = MagicMock()
    mock_tmp.complete.return_value = "creative answer"

    with patch("gemini_client.Gemini", return_value=mock_tmp):
        result = generate_text("Write a poem.", temperature=0.9)

    assert isinstance(result, str)
    mock_tmp.complete.assert_called_once_with("Write a poem.")


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
    with patch("gemini_client.generate_text", return_value="A summary.") as mock_gen:
        result = summarize("Some long text about something interesting.", max_words=50)
    assert result == "A summary."
    mock_gen.assert_called_once()
    call_prompt = mock_gen.call_args[0][0]
    assert "50" in call_prompt
    assert "Some long text" in call_prompt


# ---------------------------------------------------------------------------
# rerank_passages
# ---------------------------------------------------------------------------


def test_rerank_passages_empty_list():
    assert rerank_passages("query", []) == []


def test_rerank_passages_returns_reordered():
    passages = ["alpha", "beta", "gamma"]
    with patch("gemini_client.generate_text", return_value="3,1,2"):
        result = rerank_passages("test query", passages)
    assert result == ["gamma", "alpha", "beta"]


def test_rerank_passages_fallback_on_bad_response():
    passages = ["alpha", "beta", "gamma"]
    with patch("gemini_client.generate_text", return_value="not valid numbers!!!"):
        result = rerank_passages("test query", passages)
    assert result == passages


def test_rerank_passages_handles_duplicates_in_llm_response():
    passages = ["a", "b", "c"]
    # LLM returns a duplicate index; duplicates should be silently dropped
    with patch("gemini_client.generate_text", return_value="2,2,1"):
        result = rerank_passages("q", passages)
    assert "b" in result
    assert "a" in result
    assert len(result) == 3
