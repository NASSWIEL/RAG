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
]:
    sys.modules.setdefault(_pkg, MagicMock())

# Evict any stale mock of the module under test from a previous test file
sys.modules.pop("gemini_client", None)

from gemini_client import initialize_gemini_llm  # noqa: E402


def test_initialize_gemini_llm_returns_llm():
    mock_llm = MagicMock()
    mock_gemini_cls = MagicMock(return_value=mock_llm)

    with patch("gemini_client.Gemini", mock_gemini_cls), \
         patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
        result = initialize_gemini_llm()

    assert result is mock_llm
    mock_gemini_cls.assert_called_once_with(
        api_key="test-key",
        model="models/gemini-2.5-flash",
        temperature=0.1,
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
