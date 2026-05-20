"""Tests for text_processor.py."""
import os
import sys
from unittest.mock import MagicMock, patch

# Ensure the repo root is on sys.path so text_processor is importable
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Pre-mock heavy third-party packages so the module under test can be imported
for _pkg in [
    "llama_index",
    "llama_index.core",
    "llama_index.core.node_parser",
    "llama_index.embeddings",
    "llama_index.embeddings.huggingface",
]:
    sys.modules.setdefault(_pkg, MagicMock())

# Evict any stale mock of the module under test from a previous test file
sys.modules.pop("text_processor", None)

from text_processor import setup_advanced_text_processing, create_node_parser  # noqa: E402


def test_setup_advanced_text_processing_returns_embed_model():
    mock_embed_model = MagicMock()
    with patch("text_processor.HuggingFaceEmbedding", return_value=mock_embed_model) as mock_hf, \
         patch("text_processor.Settings") as mock_settings:
        result = setup_advanced_text_processing()

    mock_hf.assert_called_once_with(model_name="BAAI/bge-small-en-v1.5")
    assert result is mock_embed_model


def test_setup_advanced_text_processing_configures_settings():
    mock_embed_model = MagicMock()
    with patch("text_processor.HuggingFaceEmbedding", return_value=mock_embed_model), \
         patch("text_processor.Settings") as mock_settings:
        setup_advanced_text_processing()

    assert mock_settings.embed_model == mock_embed_model
    assert mock_settings.chunk_size == 512
    assert mock_settings.chunk_overlap == 50


def test_create_node_parser_returns_sentence_splitter():
    mock_parser = MagicMock()
    with patch("text_processor.SentenceSplitter", return_value=mock_parser) as mock_ss:
        result = create_node_parser()

    mock_ss.assert_called_once_with(chunk_size=512, chunk_overlap=50)
    assert result is mock_parser
