"""Tests for C:/retrodoc/test-BT-AI/RAG/rag_engine.py."""
import os as _os
import sys
from unittest.mock import MagicMock, patch, PropertyMock

# Ensure project root is on sys.path so rag_engine can be imported
_project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Remove any pre-mocked rag_engine (e.g. from test_main.py) so the real module is used
sys.modules.pop("rag_engine", None)

# Pre-mock heavy third-party packages so the module under test can be imported
for _pkg in [
    "llama_index",
    "llama_index.core",
    "llama_index.core.settings",
    "llama_index.core.storage",
    "llama_index.core.storage.storage_context",
    "llama_index.core.indices",
    "llama_index.core.indices.vector_store",
]:
    sys.modules.setdefault(_pkg, MagicMock())

# Mock the local dependencies as well
sys.modules.setdefault("gemini_client", MagicMock())
sys.modules.setdefault("pdf_loader", MagicMock())
sys.modules.setdefault("text_processor", MagicMock())

from unittest.mock import patch


def _make_rag_engine(pdf_url="http://example.com/doc.pdf", storage_dir="/tmp/rag_storage"):
    """Helper to construct a RAGEngine with all heavy dependencies mocked."""
    mock_embed = MagicMock()
    mock_llm = MagicMock()
    mock_parser = MagicMock()
    mock_index = MagicMock()
    mock_query_engine = MagicMock()
    mock_index.as_query_engine.return_value = mock_query_engine

    with patch("rag_engine.setup_advanced_text_processing", return_value=mock_embed, autospec=False), \
         patch("rag_engine.initialize_gemini_llm", return_value=mock_llm, autospec=False), \
         patch("rag_engine.create_node_parser", return_value=mock_parser, autospec=False), \
         patch("rag_engine.Settings", autospec=False), \
         patch("rag_engine.os.path.exists", return_value=False), \
         patch("rag_engine.load_pdf_from_url", return_value="/tmp/fake.pdf", autospec=False), \
         patch("rag_engine.load_documents_from_pdf", return_value=[MagicMock()], autospec=False), \
         patch("rag_engine.VectorStoreIndex") as mock_vsi, \
         patch("rag_engine.os.makedirs", autospec=False):
        mock_vsi.from_documents.return_value = mock_index
        # prevent actual persist
        mock_index.storage_context.persist = MagicMock()

        from rag_engine import RAGEngine
        engine = RAGEngine(pdf_url=pdf_url, storage_dir=storage_dir)
        # Attach mock_query_engine for later inspection
        engine._mock_query_engine = mock_query_engine
        return engine


def test_ragengine_init_creates_instance():
    """RAGEngine can be instantiated when dependencies are mocked."""
    engine = _make_rag_engine()
    from rag_engine import RAGEngine
    assert type(engine).__name__ == "RAGEngine"


def test_ragengine_init_loads_from_cache(tmp_path):
    """RAGEngine loads from cache when index path already exists."""
    mock_embed = MagicMock()
    mock_llm = MagicMock()
    mock_parser = MagicMock()
    mock_index = MagicMock()
    mock_query_engine = MagicMock()
    mock_index.as_query_engine.return_value = mock_query_engine
    mock_storage_context = MagicMock()

    with patch("rag_engine.setup_advanced_text_processing", return_value=mock_embed, autospec=False), \
         patch("rag_engine.initialize_gemini_llm", return_value=mock_llm, autospec=False), \
         patch("rag_engine.create_node_parser", return_value=mock_parser, autospec=False), \
         patch("rag_engine.Settings", autospec=False), \
         patch("rag_engine.os.path.exists", return_value=True), \
         patch("rag_engine.StorageContext") as mock_sc, \
         patch("rag_engine.load_index_from_storage", return_value=mock_index, autospec=False):
        mock_sc.from_defaults.return_value = mock_storage_context

        from rag_engine import RAGEngine
        engine = RAGEngine(pdf_url="http://example.com/cached.pdf", storage_dir=str(tmp_path))
        assert engine.index is mock_index


def test_ragengine_get_index_path_is_deterministic():
    """_get_index_path returns the same path for the same URL."""
    engine = _make_rag_engine(pdf_url="http://example.com/stable.pdf")
    path1 = engine._get_index_path()
    path2 = engine._get_index_path()
    assert path1 == path2


def test_ragengine_get_index_path_differs_per_url():
    """_get_index_path returns different paths for different URLs."""
    engine_a = _make_rag_engine(pdf_url="http://example.com/a.pdf")
    engine_b = _make_rag_engine(pdf_url="http://example.com/b.pdf")
    assert engine_a._get_index_path() != engine_b._get_index_path()


def test_query_returns_string():
    """query() returns a string response for a given question."""
    engine = _make_rag_engine()
    mock_response = MagicMock()
    mock_response.__str__ = lambda self: "The answer is 42."
    engine._mock_query_engine.query.return_value = mock_response
    engine.query_engine = engine._mock_query_engine

    result = engine.query("What is the answer?")
    assert isinstance(result, str)
    assert result == "The answer is 42."


def test_query_calls_engine_with_question():
    """query() passes the question to the underlying query engine."""
    engine = _make_rag_engine()
    mock_response = MagicMock()
    mock_response.__str__ = lambda self: "Some answer."
    engine._mock_query_engine.query.return_value = mock_response
    engine.query_engine = engine._mock_query_engine

    engine.query("What is RAG?")
    engine._mock_query_engine.query.assert_called_once_with("What is RAG?")


def test_query_empty_question():
    """query() handles an empty string question without error."""
    engine = _make_rag_engine()
    mock_response = MagicMock()
    mock_response.__str__ = lambda self: ""
    engine._mock_query_engine.query.return_value = mock_response
    engine.query_engine = engine._mock_query_engine

    result = engine.query("")
    assert isinstance(result, str)
