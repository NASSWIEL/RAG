"""Tests for C:/retrodoc/test-BT-AI/RAG/pdf_loader.py."""
import os
import sys
from unittest.mock import MagicMock, patch

# Ensure the project root is on sys.path so pdf_loader can be imported
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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


def test_load_pdf_from_url_returns_temp_path():
    mock_response = MagicMock()
    mock_response.content = b"%PDF-1.4 fake content"

    with patch("pdf_loader.requests.get", return_value=mock_response) as mock_get:
        result = load_pdf_from_url("http://example.com/sample.pdf")

    mock_get.assert_called_once_with("http://example.com/sample.pdf", timeout=30)
    assert isinstance(result, str)
    assert result.endswith("temp_rag_document.pdf")


def test_load_pdf_from_url_writes_content(tmp_path):
    mock_response = MagicMock()
    mock_response.content = b"%PDF-1.4 test bytes"

    with patch("pdf_loader.requests.get", return_value=mock_response), \
         patch("pdf_loader.tempfile.gettempdir", return_value=str(tmp_path)), \
         patch("pdf_loader.os.path.join", return_value=str(tmp_path / "temp_rag_document.pdf")):
        result = load_pdf_from_url("http://example.com/doc.pdf")

    assert result == str(tmp_path / "temp_rag_document.pdf")
    written = (tmp_path / "temp_rag_document.pdf").read_bytes()
    assert written == b"%PDF-1.4 test bytes"


def test_load_documents_from_pdf_calls_reader(tmp_path):
    fake_pdf = tmp_path / "sample.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 dummy")

    mock_doc = MagicMock()
    mock_loader_instance = MagicMock()
    mock_loader_instance.load_data.return_value = [mock_doc]

    with patch("pdf_loader.PDFReader", return_value=mock_loader_instance) as mock_reader_cls:
        result = load_documents_from_pdf(str(fake_pdf))

    mock_reader_cls.assert_called_once()
    mock_loader_instance.load_data.assert_called_once()
    assert result == [mock_doc]


def test_load_documents_from_pdf_returns_list(tmp_path):
    fake_pdf = tmp_path / "another.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 content")

    mock_loader_instance = MagicMock()
    mock_loader_instance.load_data.return_value = []

    with patch("pdf_loader.PDFReader", return_value=mock_loader_instance):
        result = load_documents_from_pdf(str(fake_pdf))

    assert isinstance(result, list)
    assert result == []
