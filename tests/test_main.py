"""Tests for main.py."""
import sys
from unittest.mock import MagicMock, patch

# Pre-mock heavy third-party packages so the module under test can be imported
for _pkg in [
    "llama_index",
    "llama_index.core",
    "llama_index.core.node_parser",
    "llama_index.readers",
    "llama_index.readers.file",
    "google",
    "google.generativeai",
    "llama_index.llms",
    "llama_index.llms.gemini",
    "llama_index.embeddings",
    "llama_index.embeddings.huggingface",
]:
    sys.modules.setdefault(_pkg, MagicMock())

# Evict any stale mock of the module under test from a previous test file
sys.modules.pop("main", None)
sys.modules.pop("rag_engine", None)


def test_main_runs_and_exits_on_q(monkeypatch):
    """main() should exit the loop cleanly when user types 'q'."""
    mock_rag = MagicMock()
    mock_rag.query.return_value = "Some answer"

    mock_rag_engine_cls = MagicMock(return_value=mock_rag)

    monkeypatch.setattr("builtins.input", lambda _: "q")

    with patch.dict(sys.modules, {"rag_engine": MagicMock(RAGEngine=mock_rag_engine_cls)}):
        # Re-import main after patching so it picks up the mocked rag_engine
        sys.modules.pop("main", None)
        import main as main_module
        main_module.RAGEngine = mock_rag_engine_cls
        main_module.main()

    # RAGEngine should have been instantiated once with the PDF URL
    mock_rag_engine_cls.assert_called_once_with("https://arxiv.org/pdf/2005.11401.pdf")
    # query should NOT have been called because user immediately typed 'q'
    mock_rag.query.assert_not_called()


def test_main_calls_query_before_exit(monkeypatch):
    """main() should call rag.query() for non-exit inputs, then exit on 'q'."""
    mock_rag = MagicMock()
    mock_rag.query.return_value = "The answer is 42"

    mock_rag_engine_cls = MagicMock(return_value=mock_rag)

    responses = iter(["what is RAG?", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    sys.modules.pop("main", None)
    import main as main_module
    main_module.RAGEngine = mock_rag_engine_cls
    main_module.main()

    mock_rag.query.assert_called_once_with("what is RAG?")
