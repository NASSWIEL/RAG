"""Tests for C:/retrodoc/test-BT-AI/RAG/text_processor.py."""
import sys
import os
from unittest.mock import MagicMock, patch

# Ensure the project root is on sys.path so that text_processor can be imported
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

# Evict any stale mock of the module under test from a previous test file
sys.modules.pop("text_processor", None)

from text_processor import setup_advanced_text_processing, create_node_parser  # noqa: E402


def test_setup_advanced_text_processing_returns_embed_model():
    mock_embed_model = MagicMock()
    mock_hf_class = sys.modules["llama_index.embeddings.huggingface"].HuggingFaceEmbedding
    mock_hf_class.return_value = mock_embed_model

    with patch("text_processor.HuggingFaceEmbedding", mock_hf_class):
        result = setup_advanced_text_processing()

    assert result is mock_embed_model
    mock_hf_class.assert_called_once_with(model_name="BAAI/bge-small-en-v1.5")


def test_setup_advanced_text_processing_configures_settings():
    mock_embed_model = MagicMock()
    mock_hf_class = MagicMock(return_value=mock_embed_model)
    mock_settings = sys.modules["llama_index.core"].Settings

    with patch("text_processor.HuggingFaceEmbedding", mock_hf_class):
        setup_advanced_text_processing()

    assert mock_settings.chunk_size == 512
    assert mock_settings.chunk_overlap == 50
    assert mock_settings.embed_model is mock_embed_model


def test_create_node_parser_returns_sentence_splitter():
    mock_parser_instance = MagicMock()
    mock_splitter_class = sys.modules["llama_index.core.node_parser"].SentenceSplitter
    mock_splitter_class.return_value = mock_parser_instance

    with patch("text_processor.SentenceSplitter", mock_splitter_class):
        result = create_node_parser()

    assert result is mock_parser_instance
    mock_splitter_class.assert_called_once_with(chunk_size=512, chunk_overlap=50)


def test_create_node_parser_chunk_parameters():
    mock_splitter_class = MagicMock()
    mock_instance = MagicMock()
    mock_splitter_class.return_value = mock_instance

    with patch("text_processor.SentenceSplitter", mock_splitter_class):
        result = create_node_parser()

    call_kwargs = mock_splitter_class.call_args
    assert call_kwargs is not None
    assert call_kwargs.kwargs.get("chunk_size") == 512 or (
        len(call_kwargs.args) > 0 and call_kwargs.args[0] == 512
    )
    assert call_kwargs.kwargs.get("chunk_overlap") == 50 or (
        len(call_kwargs.args) > 1 and call_kwargs.args[1] == 50
    )
    assert result is mock_instance
