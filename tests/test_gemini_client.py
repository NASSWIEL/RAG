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

import os
import pytest

from gemini_client import initialize_gemini_llm


def test_initialize_gemini_llm_returns_llm(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")

    mock_llm_instance = MagicMock()
    mock_gemini_cls = MagicMock(return_value=mock_llm_instance)
    mock_settings = MagicMock()

    with patch("gemini_client.Gemini", mock_gemini_cls), \
         patch("gemini_client.Settings", mock_settings):
        result = initialize_gemini_llm()

    assert result is mock_llm_instance
    mock_gemini_cls.assert_called_once_with(
        api_key="test-key-123",
        model="models/gemini-2.5-flash",
        temperature=0.1,
    )


def test_initialize_gemini_llm_sets_settings_llm(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-abc")

    mock_llm_instance = MagicMock()
    mock_gemini_cls = MagicMock(return_value=mock_llm_instance)
    mock_settings = MagicMock()

    with patch("gemini_client.Gemini", mock_gemini_cls), \
         patch("gemini_client.Settings", mock_settings):
        initialize_gemini_llm()

    assert mock_settings.llm is mock_llm_instance


def test_initialize_gemini_llm_missing_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(KeyError):
        initialize_gemini_llm()
