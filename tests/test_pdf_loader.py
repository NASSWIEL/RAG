"""Tests for pdf_loader.py."""
import os
import sys
from unittest.mock import MagicMock, patch

# Ensure the repo root is on sys.path so pdf_loader is importable
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Pre-mock heavy third-party packages so the module under test can be imported
for _pkg in [
    "requests",
    "llama_index",
    "llama_index.readers",
    "llama_index.readers.file",
]:
    sys.modules.setdefault(_pkg, MagicMock())

# Evict any stale mock of the module under test from a previous test file
sys.modules.pop("pdf_loader", None)

from pdf_loader import load_pdf_from_url, load_documents_from_pdf  # noqa: E402


def test_load_pdf_from_url_returns_path():
    mock_response = MagicMock()
    mock_response.content = b"%PDF-1.4 fake content"

    with patch("pdf_loader.requests.get", return_value=mock_response) as mock_get:
        result = load_pdf_from_url("http://example.com/doc.pdf")

    mock_get.assert_called_once_with("http://example.com/doc.pdf", timeout=30)
    assert result.endswith("temp_rag_document.pdf")


def test_load_pdf_from_url_writes_content(tmp_path):
    mock_response = MagicMock()
    mock_response.content = b"%PDF-1.4 fake content"

    import tempfile
    with patch("pdf_loader.requests.get", return_value=mock_response):
        with patch("pdf_loader.tempfile.gettempdir", return_value=str(tmp_path)):
            result = load_pdf_from_url("http://example.com/doc.pdf")

    written = (tmp_path / "temp_rag_document.pdf").read_bytes()
    assert written == b"%PDF-1.4 fake content"
    assert result == str(tmp_path / "temp_rag_document.pdf").replace("\\", "/") or result.endswith("temp_rag_document.pdf")


def test_load_documents_from_pdf_calls_loader(tmp_path):
    fake_pdf = tmp_path / "sample.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    mock_docs = [MagicMock(), MagicMock()]
    mock_loader_instance = MagicMock()
    mock_loader_instance.load_data.return_value = mock_docs

    with patch("pdf_loader.PDFReader", return_value=mock_loader_instance):
        result = load_documents_from_pdf(str(fake_pdf))

    mock_loader_instance.load_data.assert_called_once()
    assert result == mock_docs


def test_load_documents_from_pdf_returns_list(tmp_path):
    fake_pdf = tmp_path / "sample.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    mock_loader_instance = MagicMock()
    mock_loader_instance.load_data.return_value = []

    with patch("pdf_loader.PDFReader", return_value=mock_loader_instance):
        result = load_documents_from_pdf(str(fake_pdf))

    assert isinstance(result, list)
    assert result == []
