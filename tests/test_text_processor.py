"""Tests for C:/retrodoc/test-BT-AI/RAG/text_processor.py."""
import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock

# Ensure project root is on sys.path so 'text_processor' can be imported
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Pre-mock heavy third-party packages so the module under test can be imported
for _pkg in [
    "llama_index",
    "llama_index.core",
    "llama_index.core.node_parser",
    "llama_index.embeddings",
    "llama_index.embeddings.huggingface",
]:
    sys.modules.setdefault(_pkg, MagicMock())

# Ensure Settings is a MagicMock with assignable attributes
_mock_settings = MagicMock()
sys.modules["llama_index.core"].Settings = _mock_settings

# Remove any pre-mocked text_processor (e.g. from test_rag_engine.py) so the real module is used
sys.modules.pop("text_processor", None)

import text_processor
from text_processor import setup_advanced_text_processing, create_node_parser


def test_setup_advanced_text_processing_returns_embed_model():
    mock_embed = MagicMock()
    with patch("text_processor.HuggingFaceEmbedding", return_value=mock_embed) as mock_hf, \
         patch("text_processor.Settings") as mock_settings:
        result = setup_advanced_text_processing()
        mock_hf.assert_called_once_with(model_name="BAAI/bge-small-en-v1.5")
        assert result is mock_embed


def test_setup_advanced_text_processing_configures_settings():
    mock_embed = MagicMock()
    with patch("text_processor.HuggingFaceEmbedding", return_value=mock_embed), \
         patch("text_processor.Settings") as mock_settings:
        setup_advanced_text_processing()
        assert mock_settings.embed_model == mock_embed
        assert mock_settings.chunk_size == 512
        assert mock_settings.chunk_overlap == 50


def test_create_node_parser_returns_sentence_splitter():
    mock_parser = MagicMock()
    with patch("text_processor.SentenceSplitter", return_value=mock_parser) as mock_ss:
        result = create_node_parser()
        mock_ss.assert_called_once_with(chunk_size=512, chunk_overlap=50)
        assert result is mock_parser


def test_create_node_parser_returns_object():
    mock_parser = MagicMock()
    with patch("text_processor.SentenceSplitter", return_value=mock_parser):
        result = create_node_parser()
        assert result is not None
