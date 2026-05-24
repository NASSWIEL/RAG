"""Tests for gemini_client.py."""
import sys
from unittest.mock import MagicMock, patch

# Pre-mock heavy third-party packages so the module under test can be imported
for _pkg in [
    "llama_index",
    "llama_index.core",
    "llama_index.llms",
    "llama_index.llms.gemini",
]:
    sys.modules.setdefault(_pkg, MagicMock())

# Evict any stale mock of the module under test from a previous test file
sys.modules.pop("gemini_client", None)

import pytest

from gemini_client import initialize_gemini_llm


def test_initialize_gemini_llm_returns_llm_instance(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")

    mock_gemini_cls = MagicMock()
    mock_llm = MagicMock()
    mock_gemini_cls.return_value = mock_llm

    mock_settings = MagicMock()

    with patch("gemini_client.Gemini", mock_gemini_cls), \
         patch("gemini_client.Settings", mock_settings):
        result = initialize_gemini_llm()

    assert result is mock_llm
    mock_gemini_cls.assert_called_once_with(
        api_key="test-api-key",
        model="models/gemini-2.5-flash",
        temperature=0.1,
    )


def test_initialize_gemini_llm_sets_settings_llm(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")

    mock_gemini_cls = MagicMock()
    mock_llm = MagicMock()
    mock_gemini_cls.return_value = mock_llm

    mock_settings = MagicMock()

    with patch("gemini_client.Gemini", mock_gemini_cls), \
         patch("gemini_client.Settings", mock_settings):
        initialize_gemini_llm()

    assert mock_settings.llm is mock_llm


def test_initialize_gemini_llm_missing_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(KeyError):
        initialize_gemini_llm()
