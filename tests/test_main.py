"""Tests for C:/retrodoc/test-BT-AI/RAG/main.py."""
import os
import sys
from unittest.mock import MagicMock, patch, call

# Ensure the project root is on sys.path so main can be imported
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Pre-mock heavy third-party packages so the module under test can be imported
for _pkg in [
    "llama_index",
    "llama_index.core",
    "llama_index.core.node_parser",
    "llama_index.readers",
    "llama_index.readers.file",
    "rag_engine",
    "gemini_client",
    "pdf_loader",
    "text_processor",
]:
    sys.modules.setdefault(_pkg, MagicMock())

import main


def test_main_exits_on_q(monkeypatch):
    """main() should exit the loop when the user types 'q'."""
    mock_rag_instance = MagicMock()
    mock_rag_instance.query.return_value = "Some answer"

    mock_rag_class = MagicMock(return_value=mock_rag_instance)

    inputs = iter(["q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    with patch("main.RAGEngine", mock_rag_class):
        main.main()

    mock_rag_class.assert_called_once_with("https://arxiv.org/pdf/2005.11401.pdf")
    mock_rag_instance.query.assert_not_called()


def test_main_queries_before_quit(monkeypatch):
    """main() should call rag.query() for each question before 'q' exits."""
    mock_rag_instance = MagicMock()
    mock_rag_instance.query.return_value = "An answer"

    mock_rag_class = MagicMock(return_value=mock_rag_instance)

    inputs = iter(["What is RAG?", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    with patch("main.RAGEngine", mock_rag_class):
        main.main()

    mock_rag_instance.query.assert_called_once_with("What is RAG?")


def test_main_multiple_questions_before_quit(monkeypatch, capsys):
    """main() should process multiple questions and print each answer."""
    mock_rag_instance = MagicMock()
    mock_rag_instance.query.side_effect = ["Answer 1", "Answer 2"]

    mock_rag_class = MagicMock(return_value=mock_rag_instance)

    inputs = iter(["first question", "second question", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    with patch("main.RAGEngine", mock_rag_class):
        main.main()

    assert mock_rag_instance.query.call_count == 2
    mock_rag_instance.query.assert_any_call("first question")
    mock_rag_instance.query.assert_any_call("second question")

    captured = capsys.readouterr()
    assert "Answer 1" in captured.out
    assert "Answer 2" in captured.out
