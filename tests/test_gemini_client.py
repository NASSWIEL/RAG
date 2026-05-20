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

from gemini_client import initialize_gemini_llm  # noqa: E402


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
