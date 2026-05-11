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
