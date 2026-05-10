"""Tests for C:/retrodoc/test-BT-AI/RAG/main.py."""
import sys
import types
from unittest.mock import MagicMock, patch


def _make_rag_engine_module():
    """Create a minimal fake rag_engine module so main.py can be imported."""
    mod = types.ModuleType("rag_engine")
    mock_class = MagicMock()
    mod.RAGEngine = mock_class
    return mod


def _import_main():
    """Import the main module, injecting a fake rag_engine dependency."""
    # Inject fake rag_engine before importing main
    if "rag_engine" not in sys.modules:
        sys.modules["rag_engine"] = _make_rag_engine_module()
    # Re-import main fresh each time
    if "main" in sys.modules:
        del sys.modules["main"]
    import importlib
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        "main",
        os.path.join(os.path.dirname(__file__), "..", "main.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_exits_on_q():
    """main() should exit the loop when user types 'q'."""
    fake_rag_instance = MagicMock()
    fake_rag_class = MagicMock(return_value=fake_rag_instance)

    fake_rag_mod = types.ModuleType("rag_engine")
    fake_rag_mod.RAGEngine = fake_rag_class
    sys.modules["rag_engine"] = fake_rag_mod

    main_mod = _import_main()

    with patch("builtins.input", return_value="q"), \
         patch("builtins.print"):
        main_mod.main()

    # RAGEngine should be instantiated once with the PDF URL
    fake_rag_class.assert_called_once_with("https://arxiv.org/pdf/2005.11401.pdf")
    # query should NOT be called because input was immediately 'q'
    fake_rag_instance.query.assert_not_called()


def test_main_calls_query_then_exits():
    """main() should call rag.query() for each non-exit question, then stop on 'q'."""
    fake_rag_instance = MagicMock()
    fake_rag_instance.query.return_value = "42"

    fake_rag_class = MagicMock(return_value=fake_rag_instance)
    fake_rag_mod = types.ModuleType("rag_engine")
    fake_rag_mod.RAGEngine = fake_rag_class
    sys.modules["rag_engine"] = fake_rag_mod

    main_mod = _import_main()

    inputs = iter(["What is RAG?", "How does it work?", "q"])
    with patch("builtins.input", side_effect=inputs), \
         patch("builtins.print"):
        main_mod.main()

    assert fake_rag_instance.query.call_count == 2
    fake_rag_instance.query.assert_any_call("What is RAG?")
    fake_rag_instance.query.assert_any_call("How does it work?")


def test_main_prints_answer(capsys):
    """main() should print the answer returned by rag.query()."""
    fake_rag_instance = MagicMock()
    fake_rag_instance.query.return_value = "The answer is 42."

    fake_rag_class = MagicMock(return_value=fake_rag_instance)
    fake_rag_mod = types.ModuleType("rag_engine")
    fake_rag_mod.RAGEngine = fake_rag_class
    sys.modules["rag_engine"] = fake_rag_mod

    main_mod = _import_main()

    inputs = iter(["What is the answer?", "q"])
    with patch("builtins.input", side_effect=inputs):
        main_mod.main()

    captured = capsys.readouterr()
    assert "The answer is 42." in captured.out
