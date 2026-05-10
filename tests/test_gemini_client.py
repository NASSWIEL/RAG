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

from gemini_client import initialize_gemini_llm  # noqa: E402


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
