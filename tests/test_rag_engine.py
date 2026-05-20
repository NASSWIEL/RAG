"""Tests for rag_engine.py."""
import sys
from unittest.mock import MagicMock, patch

# Pre-mock heavy third-party packages so the module under test can be imported
for _pkg in [
    "llama_index",
    "llama_index.core",
    "llama_index.core.node_parser",
    "llama_index.readers",
    "llama_index.readers.file",
    "gemini_client",
    "pdf_loader",
    "text_processor",
]:
    sys.modules.setdefault(_pkg, MagicMock())

# Evict any stale mock of the module under test from a previous test file
sys.modules.pop("rag_engine", None)

from rag_engine import RAGEngine  # noqa: E402


def _make_engine_cached(tmp_path, mocker=None):
    """Helper: build a RAGEngine instance taking the cached-index branch."""
    with (
        patch("rag_engine.setup_advanced_text_processing", return_value=MagicMock()),
        patch("rag_engine.initialize_gemini_llm", return_value=MagicMock()),
        patch("rag_engine.create_node_parser", return_value=MagicMock()),
        patch("rag_engine.Settings"),
        patch("rag_engine.os.path.exists", return_value=True),
        patch("rag_engine.StorageContext") as mock_sc,
        patch("rag_engine.load_index_from_storage") as mock_load,
    ):
        mock_index = MagicMock()
        mock_load.return_value = mock_index
        engine = RAGEngine(
            pdf_url="http://example.com/doc.pdf",
            storage_dir=str(tmp_path / "storage"),
        )
        return engine, mock_load, mock_sc


def test_init_loads_cached_index(tmp_path):
    """When a cached index exists on disk, load_index_from_storage should be called."""
    with (
        patch("rag_engine.setup_advanced_text_processing", return_value=MagicMock()),
        patch("rag_engine.initialize_gemini_llm", return_value=MagicMock()),
        patch("rag_engine.create_node_parser", return_value=MagicMock()),
        patch("rag_engine.Settings"),
        patch("rag_engine.os.path.exists", return_value=True),
        patch("rag_engine.StorageContext"),
        patch("rag_engine.load_index_from_storage") as mock_load,
    ):
        mock_index = MagicMock()
        mock_load.return_value = mock_index

        engine = RAGEngine(
            pdf_url="http://example.com/doc.pdf",
            storage_dir=str(tmp_path / "storage"),
        )

        mock_load.assert_called_once()
        assert engine.query_engine is not None


def test_init_builds_new_index(tmp_path):
    """When no cached index exists, the PDF should be downloaded and a new index built."""
    with (
        patch("rag_engine.setup_advanced_text_processing", return_value=MagicMock()),
        patch("rag_engine.initialize_gemini_llm", return_value=MagicMock()),
        patch("rag_engine.create_node_parser", return_value=MagicMock()),
        patch("rag_engine.Settings"),
        patch("rag_engine.os.path.exists", return_value=False),
        patch("rag_engine.os.makedirs"),
        patch("rag_engine.load_pdf_from_url", return_value="/tmp/doc.pdf") as mock_url,
        patch("rag_engine.load_documents_from_pdf", return_value=[MagicMock()]) as mock_docs,
        patch("rag_engine.VectorStoreIndex") as mock_vsi,
    ):
        mock_index = MagicMock()
        mock_vsi.from_documents.return_value = mock_index

        engine = RAGEngine(
            pdf_url="http://example.com/newdoc.pdf",
            storage_dir=str(tmp_path / "storage"),
        )

        mock_url.assert_called_once_with("http://example.com/newdoc.pdf")
        mock_docs.assert_called_once()
        mock_vsi.from_documents.assert_called_once()
        assert engine.index is mock_index


def test_query_returns_string(tmp_path):
    """query() should return a string representation of the query engine response."""
    with (
        patch("rag_engine.setup_advanced_text_processing", return_value=MagicMock()),
        patch("rag_engine.initialize_gemini_llm", return_value=MagicMock()),
        patch("rag_engine.create_node_parser", return_value=MagicMock()),
        patch("rag_engine.Settings"),
        patch("rag_engine.os.path.exists", return_value=True),
        patch("rag_engine.StorageContext"),
        patch("rag_engine.load_index_from_storage") as mock_load,
    ):
        mock_index = MagicMock()
        mock_load.return_value = mock_index

        engine = RAGEngine(
            pdf_url="http://example.com/doc.pdf",
            storage_dir=str(tmp_path / "storage"),
        )

        engine.query_engine.query.return_value = "The answer is 42."
        result = engine.query("What is the answer?")

        assert result == "The answer is 42."
        assert isinstance(result, str)
