"""Tests for C:/retrodoc/test-BT-AI/RAG/main.py."""
import os
import sys
from unittest.mock import MagicMock, patch

# Ensure the project root is on sys.path so 'main' can be imported
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Pre-mock heavy third-party packages so the module under test can be imported
for _pkg in [
    "llama_index",
    "llama_index.core",
    "llama_index.core.settings",
    "llama_index.core.storage",
    "llama_index.core.storage.storage_context",
    "llama_index.core.indices",
    "llama_index.core.indices.vector_store",
    "gemini_client",
    "pdf_loader",
    "text_processor",
]:
    sys.modules.setdefault(_pkg, MagicMock())

# Evict any stale mock of the modules under test from a previous test file
sys.modules.pop("rag_engine", None)
sys.modules.pop("main", None)

# Mock rag_engine so main.py can import RAGEngine without a real rag_engine
_mock_rag_engine_module = MagicMock()
sys.modules["rag_engine"] = _mock_rag_engine_module

import main  # noqa: E402


def test_main_exits_on_q():
    """Test that main() breaks out of the loop when user types 'q'."""
    mock_rag = MagicMock()
    mock_rag.query.return_value = "some answer"

    with patch("main.RAGEngine", return_value=mock_rag) as mock_engine_cls, \
         patch("builtins.input", side_effect=["q"]), \
         patch("builtins.print"):
        main.main()

    mock_engine_cls.assert_called_once_with("https://arxiv.org/pdf/2005.11401.pdf")
    mock_rag.query.assert_not_called()


def test_main_queries_before_quitting():
    """Test that main() calls rag.query for each question before 'q'."""
    mock_rag = MagicMock()
    mock_rag.query.return_value = "some answer"

    with patch("main.RAGEngine", return_value=mock_rag), \
         patch("builtins.input", side_effect=["what is RAG?", "q"]), \
         patch("builtins.print"):
        main.main()

    mock_rag.query.assert_called_once_with("what is RAG?")


def test_main_prints_answer():
    """Test that main() prints the answer returned by rag.query."""
    mock_rag = MagicMock()
    mock_rag.query.return_value = "RAG stands for Retrieval-Augmented Generation."

    printed = []

    with patch("main.RAGEngine", return_value=mock_rag), \
         patch("builtins.input", side_effect=["what is RAG?", "q"]), \
         patch("builtins.print", side_effect=lambda *a, **kw: printed.append(a)):
        main.main()

    # At least one print call should include the answer text
    answer_prints = [a for a in printed if any("RAG stands for" in str(x) for x in a)]
    assert len(answer_prints) >= 1
