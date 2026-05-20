"""Tests for gemini_client.py."""
import sys
import pathlib
from unittest.mock import MagicMock, patch

# Ensure the RAG root (where gemini_client.py lives) is importable
_RAG_ROOT = str(pathlib.Path(__file__).parent.parent)
if _RAG_ROOT not in sys.path:
    sys.path.insert(0, _RAG_ROOT)

# Pre-mock heavy third-party packages so the module under test can be imported
for _pkg in [
    "llama_index",
    "llama_index.core",
    "llama_index.llms",
    "llama_index.llms.gemini",
    "config",
]:
    sys.modules.setdefault(_pkg, MagicMock())

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


def test_initialize_gemini_llm_returns_llm_instance():
    mock_llm = MagicMock()
    mock_gemini_cls = MagicMock(return_value=mock_llm)

    with patch("gemini_client.Gemini", mock_gemini_cls):
        result = initialize_gemini_llm()

    assert result is mock_llm


def test_initialize_gemini_llm_registers_with_settings():
    mock_llm = MagicMock()
    mock_gemini_cls = MagicMock(return_value=mock_llm)
    mock_settings = sys.modules["llama_index.core"].Settings

    with patch("gemini_client.Gemini", mock_gemini_cls):
        initialize_gemini_llm()

    assert mock_settings.llm is mock_llm


def test_initialize_gemini_llm_uses_correct_model():
    mock_llm = MagicMock()
    mock_gemini_cls = MagicMock(return_value=mock_llm)

    with patch("gemini_client.Gemini", mock_gemini_cls):
        initialize_gemini_llm()

    _, kwargs = mock_gemini_cls.call_args
    assert kwargs.get("model") == "models/gemini-2.5-flash"
    assert kwargs.get("temperature") == 0.1


# ---------------------------------------------------------------------------
# get_llm
# ---------------------------------------------------------------------------

def test_get_llm_initializes_when_none():
    """get_llm should call initialize_gemini_llm when singleton is None."""
    reset_llm()
    fake_llm = MagicMock()
    with patch("gemini_client.initialize_gemini_llm", return_value=fake_llm) as mock_init:
        result = get_llm()
    mock_init.assert_called_once()
    assert result is fake_llm
    reset_llm()


def test_get_llm_returns_cached_instance():
    """get_llm should return existing instance without re-initializing."""
    fake_llm = MagicMock()
    gemini_client._state["llm_instance"] = fake_llm
    with patch("gemini_client.initialize_gemini_llm") as mock_init:
        result = get_llm()
    mock_init.assert_not_called()
    assert result is fake_llm
    reset_llm()


# ---------------------------------------------------------------------------
# generate_text
# ---------------------------------------------------------------------------

def test_generate_text_returns_response():
    """generate_text should call llm.complete and return str."""
    fake_llm = MagicMock()
    fake_llm.complete.return_value = "some response"
    gemini_client._state["llm_instance"] = fake_llm

    result = generate_text("hello world")

    fake_llm.complete.assert_called_once_with("hello world")
    assert isinstance(result, str)
    reset_llm()


def test_generate_text_raises_on_empty_prompt():
    """generate_text should raise ValueError for empty prompt."""
    import pytest
    with pytest.raises(ValueError, match="prompt must not be empty"):
        generate_text("")


def test_generate_text_raises_on_whitespace_prompt():
    """generate_text should raise ValueError for whitespace-only prompt."""
    import pytest
    with pytest.raises(ValueError, match="prompt must not be empty"):
        generate_text("   ")


def test_generate_text_with_temperature_override():
    """generate_text should use a temporary Gemini instance when temperature is given."""
    fake_llm = MagicMock()
    fake_llm.model = "models/gemini-2.5-flash"
    gemini_client._state["llm_instance"] = fake_llm

    mock_tmp = MagicMock()
    mock_tmp.complete.return_value = "temp response"

    with patch("gemini_client.Gemini", return_value=mock_tmp):
        result = generate_text("test prompt", temperature=0.5)

    mock_tmp.complete.assert_called_once_with("test prompt")
    assert "temp response" in result
    reset_llm()


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def test_summarize_raises_on_empty_text():
    """summarize should raise ValueError for empty text."""
    import pytest
    with pytest.raises(ValueError, match="text must not be empty"):
        summarize("")


def test_summarize_calls_generate_text():
    """summarize should delegate to generate_text with a formatted prompt."""
    with patch("gemini_client.generate_text", return_value="summary") as mock_gen:
        result = summarize("some long text", max_words=50)
    mock_gen.assert_called_once()
    call_prompt = mock_gen.call_args[0][0]
    assert "50" in call_prompt
    assert "some long text" in call_prompt
    assert result == "summary"


def test_summarize_default_max_words():
    """summarize should default to 150 words in the prompt."""
    with patch("gemini_client.generate_text", return_value="ok") as mock_gen:
        summarize("text content")
    call_prompt = mock_gen.call_args[0][0]
    assert "150" in call_prompt


# ---------------------------------------------------------------------------
# rerank_passages
# ---------------------------------------------------------------------------

def test_rerank_passages_empty_returns_empty():
    """rerank_passages should return [] for empty input."""
    assert rerank_passages("query", []) == []


def test_rerank_passages_reorders_by_llm_response():
    """rerank_passages should reorder passages according to LLM output."""
    passages = ["first", "second", "third"]
    with patch("gemini_client.generate_text", return_value="3,1,2"):
        result = rerank_passages("my query", passages)
    assert result == ["third", "first", "second"]


def test_rerank_passages_fallback_on_bad_response():
    """rerank_passages should return original order on unparseable LLM output."""
    passages = ["a", "b", "c"]
    with patch("gemini_client.generate_text", return_value="not a valid list"):
        result = rerank_passages("query", passages)
    assert result == passages


def test_rerank_passages_handles_partial_llm_response():
    """rerank_passages should append omitted passages at the end."""
    passages = ["x", "y", "z"]
    with patch("gemini_client.generate_text", return_value="2"):
        result = rerank_passages("q", passages)
    assert result[0] == "y"
    assert set(result) == {"x", "y", "z"}


# ---------------------------------------------------------------------------
# reset_llm
# ---------------------------------------------------------------------------

def test_reset_llm_clears_singleton():
    """reset_llm should set llm_instance back to None."""
    gemini_client._state["llm_instance"] = MagicMock()
    reset_llm()
    assert gemini_client._state["llm_instance"] is None
