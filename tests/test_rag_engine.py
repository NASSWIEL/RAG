"""Tests for C:/retrodoc/test-BT-AI/RAG/rag_engine.py."""
import sys
import hashlib
import os
from unittest.mock import MagicMock, patch

# Pre-mock heavy third-party packages so the module under test can be imported
for _pkg in [
    "llama_index",
    "llama_index.core",
    "llama_index.core.node_parser",
    "llama_index.readers",
    "llama_index.readers.file",
]:
    sys.modules.setdefault(_pkg, MagicMock())

# Evict stale rag_engine only — do NOT replace gemini_client/pdf_loader/text_processor here:
# replacing them at module-collection time corrupts patch() targets in test_gemini_client.py
# and test_pdf_loader.py (their functions' __globals__ point to the real modules, not the mocks).
sys.modules.pop("rag_engine", None)

import pytest


def _make_rag_engine_class():
    """Import RAGEngine freshly with all dependencies mocked."""
    sys.modules.pop("rag_engine", None)

    # Mock local deps inside this function only; save originals so we can restore after import
    _orig_gemini = sys.modules.get("gemini_client")
    _orig_pdf = sys.modules.get("pdf_loader")
    _orig_text = sys.modules.get("text_processor")
    sys.modules["gemini_client"] = MagicMock()
    sys.modules["pdf_loader"] = MagicMock()
    sys.modules["text_processor"] = MagicMock()

    mock_settings = MagicMock()
    mock_storage_context = MagicMock()
    mock_vector_store_index = MagicMock()
    mock_load_index = MagicMock()

    llama_index_core = MagicMock()
    llama_index_core.Settings = mock_settings
    llama_index_core.StorageContext = mock_storage_context
    llama_index_core.VectorStoreIndex = mock_vector_store_index
    llama_index_core.load_index_from_storage = mock_load_index

    _orig_llama_core = sys.modules.get("llama_index.core")
    sys.modules["llama_index.core"] = llama_index_core
    sys.modules["llama_index"] = MagicMock()

    from rag_engine import RAGEngine

    # Restore everything so subsequent test files see the real modules and can patch them
    sys.modules["llama_index.core"] = _orig_llama_core
    sys.modules["gemini_client"] = _orig_gemini
    sys.modules["pdf_loader"] = _orig_pdf
    sys.modules["text_processor"] = _orig_text

    return (
        RAGEngine,
        mock_settings,
        mock_storage_context,
        mock_vector_store_index,
        mock_load_index,
    )


def test_rag_engine_init_no_cache(tmp_path):
    """RAGEngine initializes and builds a new index when no cached index exists."""
    (
        RAGEngine,
        mock_settings,
        mock_storage_context,
        mock_vector_store_index,
        mock_load_index,
    ) = _make_rag_engine_class()

    pdf_url = "http://example.com/test.pdf"
    storage_dir = str(tmp_path / "storage")

    mock_embed = MagicMock()
    mock_llm = MagicMock()
    mock_parser = MagicMock()
    mock_index = MagicMock()
    mock_query_engine = MagicMock()
    mock_index.as_query_engine.return_value = mock_query_engine
    mock_vector_store_index.from_documents.return_value = mock_index

    with patch("rag_engine.setup_advanced_text_processing", return_value=mock_embed), \
         patch("rag_engine.initialize_gemini_llm", return_value=mock_llm), \
         patch("rag_engine.create_node_parser", return_value=mock_parser), \
         patch("rag_engine.load_pdf_from_url", return_value="/tmp/test.pdf"), \
         patch("rag_engine.load_documents_from_pdf", return_value=["doc1"]), \
         patch("os.path.exists", return_value=False), \
         patch("os.makedirs"):

        engine = RAGEngine(pdf_url=pdf_url, storage_dir=storage_dir)

    assert engine.pdf_url == pdf_url
    assert engine.storage_dir == storage_dir
    assert engine.embed_model is mock_embed
    assert engine.llm is mock_llm
    assert engine.parser is mock_parser
    assert engine.query_engine is mock_query_engine
    mock_vector_store_index.from_documents.assert_called_once_with(["doc1"], show_progress=True)


def test_rag_engine_init_with_cache(tmp_path):
    """RAGEngine loads an existing index from cache when it exists."""
    (
        RAGEngine,
        mock_settings,
        mock_storage_context,
        mock_vector_store_index,
        mock_load_index,
    ) = _make_rag_engine_class()

    pdf_url = "http://example.com/cached.pdf"
    storage_dir = str(tmp_path / "storage")

    mock_embed = MagicMock()
    mock_llm = MagicMock()
    mock_parser = MagicMock()
    mock_index = MagicMock()
    mock_query_engine = MagicMock()
    mock_index.as_query_engine.return_value = mock_query_engine
    mock_load_index.return_value = mock_index

    with patch("rag_engine.setup_advanced_text_processing", return_value=mock_embed), \
         patch("rag_engine.initialize_gemini_llm", return_value=mock_llm), \
         patch("rag_engine.create_node_parser", return_value=mock_parser), \
         patch("os.path.exists", return_value=True):

        engine = RAGEngine(pdf_url=pdf_url, storage_dir=storage_dir)

    assert engine.index is mock_index
    mock_vector_store_index.from_documents.assert_not_called()
    mock_load_index.assert_called_once()


def test_rag_engine_query(tmp_path):
    """RAGEngine.query returns string response from query engine."""
    (
        RAGEngine,
        mock_settings,
        mock_storage_context,
        mock_vector_store_index,
        mock_load_index,
    ) = _make_rag_engine_class()

    pdf_url = "http://example.com/query.pdf"
    storage_dir = str(tmp_path / "storage")

    mock_embed = MagicMock()
    mock_llm = MagicMock()
    mock_parser = MagicMock()
    mock_index = MagicMock()
    mock_query_engine = MagicMock()
    mock_response = MagicMock()
    mock_response.__str__ = lambda self: "The answer is 42."
    mock_query_engine.query.return_value = mock_response
    mock_index.as_query_engine.return_value = mock_query_engine
    mock_load_index.return_value = mock_index

    with patch("rag_engine.setup_advanced_text_processing", return_value=mock_embed), \
         patch("rag_engine.initialize_gemini_llm", return_value=mock_llm), \
         patch("rag_engine.create_node_parser", return_value=mock_parser), \
         patch("os.path.exists", return_value=True):

        engine = RAGEngine(pdf_url=pdf_url, storage_dir=storage_dir)
        result = engine.query("What is the answer?")

    assert result == "The answer is 42."
    mock_query_engine.query.assert_called_once_with("What is the answer?")


def test_rag_engine_get_index_path():
    """_get_index_path returns a deterministic path based on pdf_url hash."""
    (
        RAGEngine,
        mock_settings,
        mock_storage_context,
        mock_vector_store_index,
        mock_load_index,
    ) = _make_rag_engine_class()

    pdf_url = "http://example.com/deterministic.pdf"
    storage_dir = "/some/storage"

    mock_embed = MagicMock()
    mock_llm = MagicMock()
    mock_parser = MagicMock()
    mock_index = MagicMock()
    mock_index.as_query_engine.return_value = MagicMock()
    mock_load_index.return_value = mock_index

    with patch("rag_engine.setup_advanced_text_processing", return_value=mock_embed), \
         patch("rag_engine.initialize_gemini_llm", return_value=mock_llm), \
         patch("rag_engine.create_node_parser", return_value=mock_parser), \
         patch("os.path.exists", return_value=True):

        engine = RAGEngine(pdf_url=pdf_url, storage_dir=storage_dir)

    expected_hash = hashlib.md5(pdf_url.encode(), usedforsecurity=False).hexdigest()
    expected_path = os.path.join(storage_dir, f"index_{expected_hash}")
    assert engine._get_index_path() == expected_path
