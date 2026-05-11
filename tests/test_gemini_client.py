"""Tests for gemini_client.py."""
import sys
import os
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
