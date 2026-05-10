"""Tests for text_processor.py."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path so we can import text_processor
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_create_node_parser_returns_sentence_splitter():
    mock_splitter = MagicMock()
    with patch("text_processor.SentenceSplitter", return_value=mock_splitter) as mock_cls:
        from text_processor import create_node_parser

        result = create_node_parser()
        mock_cls.assert_called_once_with(chunk_size=512, chunk_overlap=50)
        assert result is mock_splitter


def test_create_node_parser_chunk_settings():
    with patch("text_processor.SentenceSplitter") as mock_cls:
        mock_cls.return_value = MagicMock()
        from text_processor import create_node_parser

        create_node_parser()
        _, kwargs = mock_cls.call_args
        assert kwargs.get("chunk_size") == 512
        assert kwargs.get("chunk_overlap") == 50


def test_setup_advanced_text_processing_returns_embed_model():
    mock_embed = MagicMock()
    with patch("text_processor.HuggingFaceEmbedding", return_value=mock_embed), \
         patch("text_processor.Settings") as mock_settings:
        from text_processor import setup_advanced_text_processing

        result = setup_advanced_text_processing()
        assert result is mock_embed


def test_setup_advanced_text_processing_configures_settings():
    mock_embed = MagicMock()
    with patch("text_processor.HuggingFaceEmbedding", return_value=mock_embed) as mock_hf, \
         patch("text_processor.Settings") as mock_settings:
        from text_processor import setup_advanced_text_processing

        setup_advanced_text_processing()
        mock_hf.assert_called_once_with(model_name="BAAI/bge-small-en-v1.5")
        assert mock_settings.embed_model is mock_embed
        assert mock_settings.chunk_size == 512
        assert mock_settings.chunk_overlap == 50
