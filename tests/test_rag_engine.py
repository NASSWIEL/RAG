"""Tests for rag_engine.py."""
import hashlib
import os
import sys
import types
import unittest.mock as mock

import pytest

# ---------------------------------------------------------------------------
# Stub out heavy third-party imports so tests can import rag_engine without
# requiring llama_index, google-generativeai, etc. to be installed.
# ---------------------------------------------------------------------------

def _make_stub_module(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _ensure_stub(dotted_name, attr_chain=None):
    """Recursively create stub modules for a dotted path."""
    parts = dotted_name.split(".")
    for i in range(1, len(parts) + 1):
        sub = ".".join(parts[:i])
        if sub not in sys.modules:
            _make_stub_module(sub)
            if i > 1:
                parent = sys.modules[".".join(parts[: i - 1])]
                setattr(parent, parts[i - 1], sys.modules[sub])
    if attr_chain:
        for attr, val in attr_chain.items():
            setattr(sys.modules[dotted_name], attr, val)


# llama_index stubs
_ensure_stub("llama_index")
_ensure_stub("llama_index.core")
_ensure_stub("llama_index.core", {
    "Settings": mock.MagicMock(),
    "StorageContext": mock.MagicMock(),
    "VectorStoreIndex": mock.MagicMock(),
    "load_index_from_storage": mock.MagicMock(),
})

# Local dependency stubs
_gemini = _make_stub_module("gemini_client")
_gemini.initialize_gemini_llm = mock.MagicMock(return_value=mock.MagicMock())

_pdf_loader = _make_stub_module("pdf_loader")
_pdf_loader.load_documents_from_pdf = mock.MagicMock(return_value=[])
_pdf_loader.load_pdf_from_url = mock.MagicMock(return_value="/tmp/fake.pdf")

_text_proc = _make_stub_module("text_processor")
_text_proc.create_node_parser = mock.MagicMock(return_value=mock.MagicMock())
_text_proc.setup_advanced_text_processing = mock.MagicMock(return_value=mock.MagicMock())

# Now we can safely import rag_engine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_engine import RAGEngine  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: build a RAGEngine instance bypassing __init__
# ---------------------------------------------------------------------------

def _bare_engine(pdf_url="http://example.com/doc.pdf", storage_dir="/tmp/storage"):
    """Return a RAGEngine with __init__ skipped, attributes set manually."""
    engine = object.__new__(RAGEngine)
    engine.pdf_url = pdf_url
    engine.storage_dir = storage_dir
    engine.query_engine = mock.MagicMock()
    engine.index = mock.MagicMock()
    return engine


# ---------------------------------------------------------------------------
# Tests for _get_index_path
# ---------------------------------------------------------------------------

class TestGetIndexPath:
    def test_returns_expected_hash_path(self):
        url = "http://example.com/doc.pdf"
        engine = _bare_engine(pdf_url=url, storage_dir="/tmp/storage")
        result = engine._get_index_path()
        expected_hash = hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()
        expected = os.path.join("/tmp/storage", f"index_{expected_hash}")
        assert result == expected

    def test_different_urls_produce_different_paths(self):
        engine_a = _bare_engine(pdf_url="http://example.com/a.pdf")
        engine_b = _bare_engine(pdf_url="http://example.com/b.pdf")
        assert engine_a._get_index_path() != engine_b._get_index_path()

    def test_same_url_produces_same_path(self):
        url = "http://example.com/stable.pdf"
        engine1 = _bare_engine(pdf_url=url)
        engine2 = _bare_engine(pdf_url=url)
        assert engine1._get_index_path() == engine2._get_index_path()

    def test_storage_dir_is_respected(self):
        engine = _bare_engine(pdf_url="http://x.com/f.pdf", storage_dir="/custom/dir")
        result = engine._get_index_path()
        assert result.startswith("/custom/dir")


# ---------------------------------------------------------------------------
# Tests for query
# ---------------------------------------------------------------------------

class TestQuery:
    def test_query_returns_string(self):
        engine = _bare_engine()
        fake_response = mock.MagicMock()
        fake_response.__str__ = mock.MagicMock(return_value="The answer is 42.")
        engine.query_engine.query.return_value = fake_response

        result = engine.query("What is the answer?")
        assert isinstance(result, str)
        assert result == "The answer is 42."

    def test_query_calls_engine_with_question(self):
        engine = _bare_engine()
        engine.query_engine.query.return_value = mock.MagicMock(__str__=lambda self: "ok")
        engine.query("What is the capital?")
        engine.query_engine.query.assert_called_once_with("What is the capital?")

    def test_query_with_empty_string(self):
        engine = _bare_engine()
        engine.query_engine.query.return_value = mock.MagicMock(__str__=lambda self: "")
        result = engine.query("")
        assert result == ""


# ---------------------------------------------------------------------------
# Tests for RAGEngine.__init__ (cache-hit path — no real network/LLM)
# ---------------------------------------------------------------------------

class TestRAGEngineInit:
    def test_init_loads_from_cache_when_index_exists(self, tmp_path):
        pdf_url = "http://example.com/cached.pdf"
        url_hash = hashlib.md5(pdf_url.encode(), usedforsecurity=False).hexdigest()
        index_dir = tmp_path / f"index_{url_hash}"
        index_dir.mkdir()

        fake_index = mock.MagicMock()
        fake_index.as_query_engine.return_value = mock.MagicMock()

        with (
            mock.patch("text_processor.setup_advanced_text_processing", return_value=mock.MagicMock()),
            mock.patch("gemini_client.initialize_gemini_llm", return_value=mock.MagicMock()),
            mock.patch("text_processor.create_node_parser", return_value=mock.MagicMock()),
            mock.patch("rag_engine.RAGEngine._load_index", return_value=fake_index),
        ):
            engine = RAGEngine(pdf_url=pdf_url, storage_dir=str(tmp_path))

        assert engine.pdf_url == pdf_url
        assert engine.storage_dir == str(tmp_path)
        assert engine.index is fake_index

    def test_init_builds_index_when_no_cache(self, tmp_path):
        pdf_url = "http://example.com/new.pdf"

        fake_index = mock.MagicMock()
        fake_index.as_query_engine.return_value = mock.MagicMock()

        with (
            mock.patch("text_processor.setup_advanced_text_processing", return_value=mock.MagicMock()),
            mock.patch("gemini_client.initialize_gemini_llm", return_value=mock.MagicMock()),
            mock.patch("text_processor.create_node_parser", return_value=mock.MagicMock()),
            mock.patch("pdf_loader.load_pdf_from_url", return_value="/tmp/fake.pdf"),
            mock.patch("pdf_loader.load_documents_from_pdf", return_value=[mock.MagicMock()]),
            mock.patch("rag_engine.VectorStoreIndex") as mock_vsi,
            mock.patch("rag_engine.RAGEngine._save_index"),
        ):
            mock_vsi.from_documents.return_value = fake_index
            engine = RAGEngine(pdf_url=pdf_url, storage_dir=str(tmp_path))

        mock_vsi.from_documents.assert_called_once()
        assert engine.index is fake_index
