"""Tests for main.py."""
import sys
from unittest.mock import MagicMock, patch, call

# Pre-mock heavy third-party packages so the module under test can be imported
for _pkg in [
    "llama_index",
    "llama_index.core",
    "llama_index.core.settings",
    "llama_index.core.indices",
    "llama_index.core.storage",
    "llama_index.core.storage.storage_context",
    "gemini_client",
    "pdf_loader",
    "text_processor",
    "rag_engine",
]:
    sys.modules.setdefault(_pkg, MagicMock())

# Evict any stale mock of the module under test from a previous test file
sys.modules.pop("main", None)

from main import main  # noqa: E402


def test_main_exits_on_q():
    """main() should break out of the loop when user types 'q'."""
    mock_rag = MagicMock()
    mock_rag.query.return_value = "Some answer"

    with patch("main.RAGEngine", return_value=mock_rag) as mock_engine_cls, \
         patch("builtins.input", side_effect=["q"]), \
         patch("builtins.print"):
        main()

    mock_engine_cls.assert_called_once_with("https://arxiv.org/pdf/2005.11401.pdf")
    mock_rag.query.assert_not_called()


def test_main_queries_then_exits():
    """main() should call rag.query() for each non-quit input, then exit on 'q'."""
    mock_rag = MagicMock()
    mock_rag.query.return_value = "answer text"

    with patch("main.RAGEngine", return_value=mock_rag), \
         patch("builtins.input", side_effect=["What is RAG?", "q"]), \
         patch("builtins.print"):
        main()

    mock_rag.query.assert_called_once_with("What is RAG?")


def test_main_multiple_queries_before_quit():
    """main() should process multiple questions before quitting."""
    mock_rag = MagicMock()
    mock_rag.query.return_value = "some answer"

    questions = ["question one", "question two", "q"]

    with patch("main.RAGEngine", return_value=mock_rag), \
         patch("builtins.input", side_effect=questions), \
         patch("builtins.print"):
        main()

    assert mock_rag.query.call_count == 2
    mock_rag.query.assert_any_call("question one")
    mock_rag.query.assert_any_call("question two")
