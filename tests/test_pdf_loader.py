"""Tests for pdf_loader.py."""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

# ---------------------------------------------------------------------------
# Stub out third-party packages that are not installed in the test environment
# so that importing pdf_loader does not fail at collection time.
# ---------------------------------------------------------------------------

def _ensure_stub(module_name: str) -> None:
    """Insert a MagicMock stub for *module_name* if it is not already importable."""
    parts = module_name.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        if name not in sys.modules:
            sys.modules[name] = MagicMock()


for _mod in [
    "requests",
    "llama_index",
    "llama_index.readers",
    "llama_index.readers.file",
]:
    _ensure_stub(_mod)

# Now it is safe to import the module under test.
sys.path.insert(0, str(Path(__file__).parent.parent))
from pdf_loader import load_pdf_from_url, load_documents_from_pdf  # noqa: E402


# ---------------------------------------------------------------------------
# load_pdf_from_url
# ---------------------------------------------------------------------------


class TestLoadPdfFromUrl:
    """Tests for load_pdf_from_url."""

    def test_returns_temp_file_path(self):
        """Golden path: a successful download returns a local file path string."""
        fake_content = b"%PDF-1.4 fake pdf content"
        mock_response = MagicMock()
        mock_response.content = fake_content

        with patch("requests.get", return_value=mock_response), \
             patch("builtins.open", mock_open()):
            result = load_pdf_from_url("http://example.com/sample.pdf")

        assert isinstance(result, str)
        assert result.endswith("temp_rag_document.pdf")

    def test_calls_requests_get_with_correct_args(self):
        """requests.get must be called with the supplied URL and timeout=30."""
        mock_response = MagicMock()
        mock_response.content = b"data"

        with patch("requests.get", return_value=mock_response) as mock_get, \
             patch("builtins.open", mock_open()):
            load_pdf_from_url("http://example.com/doc.pdf")

        mock_get.assert_called_once_with("http://example.com/doc.pdf", timeout=30)

    def test_writes_response_content_to_file(self):
        """The binary content from the response must be written to the temp file."""
        fake_content = b"%PDF-1.4 binary data"
        mock_response = MagicMock()
        mock_response.content = fake_content

        m = mock_open()
        with patch("requests.get", return_value=mock_response), \
             patch("builtins.open", m):
            load_pdf_from_url("http://example.com/doc.pdf")

        m().write.assert_called_once_with(fake_content)

    def test_network_error_propagates(self):
        """If requests.get raises a ConnectionError, it propagates to the caller."""
        import requests as req

        with patch("requests.get", side_effect=Exception("no network")):
            with pytest.raises(Exception, match="no network"):
                load_pdf_from_url("http://unreachable.invalid/doc.pdf")


# ---------------------------------------------------------------------------
# load_documents_from_pdf
# ---------------------------------------------------------------------------


class TestLoadDocumentsFromPdf:
    """Tests for load_documents_from_pdf."""

    def test_returns_documents_list(self, tmp_path):
        """Golden path: PDFReader.load_data result is returned as-is."""
        fake_doc = MagicMock()
        fake_doc.text = "Hello, world!"
        mock_loader_instance = MagicMock()
        mock_loader_instance.load_data.return_value = [fake_doc]

        pdf_file = tmp_path / "sample.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        with patch("pdf_loader.PDFReader", return_value=mock_loader_instance):
            result = load_documents_from_pdf(str(pdf_file))

        assert result == [fake_doc]

    def test_passes_path_object_to_loader(self, tmp_path):
        """load_documents_from_pdf must convert the str argument to a Path."""
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        mock_loader_instance = MagicMock()
        mock_loader_instance.load_data.return_value = []

        with patch("pdf_loader.PDFReader", return_value=mock_loader_instance):
            load_documents_from_pdf(str(pdf_file))

        call_kwargs = mock_loader_instance.load_data.call_args
        passed = call_kwargs.kwargs.get("file") or (call_kwargs.args[0] if call_kwargs.args else None)
        assert isinstance(passed, Path)

    def test_empty_pdf_returns_empty_list(self, tmp_path):
        """If PDFReader returns no documents the function returns an empty list."""
        pdf_file = tmp_path / "empty.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        mock_loader_instance = MagicMock()
        mock_loader_instance.load_data.return_value = []

        with patch("pdf_loader.PDFReader", return_value=mock_loader_instance):
            result = load_documents_from_pdf(str(pdf_file))

        assert result == []
