"""Tests for gemini_client.py."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unittest.mock import MagicMock, patch

# Pre-mock heavy third-party packages
for _pkg in [
    "dotenv",
    "llama_index",
    "llama_index.core",
    "llama_index.llms",
    "llama_index.llms.gemini",
]:
    sys.modules.setdefault(_pkg, MagicMock())

# Evict any stale mock of the module under test
sys.modules.pop("gemini_client", None)

import gemini_client  # noqa: E402
from gemini_client import initialize_gemini_llm  # noqa: E402


def test_initialize_gemini_llm_returns_llm():
    mock_llm = MagicMock()
    with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
        with patch("gemini_client.Gemini", return_value=mock_llm) as mock_gemini_cls:
            with patch("gemini_client.Settings") as mock_settings:
                result = initialize_gemini_llm()
                mock_gemini_cls.assert_called_once_with(
                    api_key="test-key",
                    model="models/gemini-2.5-flash",
                    temperature=0.1,
                )
                assert mock_settings.llm == mock_llm
                assert result is mock_llm


def test_initialize_gemini_llm_sets_settings_llm():
    mock_llm = MagicMock()
    with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}):
        with patch("gemini_client.Gemini", return_value=mock_llm):
            with patch("gemini_client.Settings") as mock_settings:
                initialize_gemini_llm()
                assert mock_settings.llm is mock_llm


def test_initialize_gemini_llm_missing_api_key_raises():
    import os
    env = {k: v for k, v in os.environ.items() if k != "GOOGLE_API_KEY"}
    with patch.dict("os.environ", env, clear=True):
        import pytest
        with pytest.raises((KeyError, Exception)):
            initialize_gemini_llm()
