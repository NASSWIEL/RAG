"""Tests for pdf_loader.py."""
import sys
import tempfile
import os
from unittest.mock import MagicMock, patch

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

from pdf_loader import load_pdf_from_url, load_documents_from_pdf


def test_load_pdf_from_url_returns_temp_path():
    mock_response = MagicMock()
    mock_response.content = b"%PDF-1.4 fake content"

    with patch("pdf_loader.requests.get", return_value=mock_response) as mock_get:
        result = load_pdf_from_url("http://example.com/doc.pdf")

    mock_get.assert_called_once_with("http://example.com/doc.pdf", timeout=30)
    assert result.endswith("temp_rag_document.pdf")
    assert os.path.isabs(result)


def test_load_pdf_from_url_writes_content(tmp_path):
    mock_response = MagicMock()
    mock_response.content = b"%PDF-1.4 fake content"

    with patch("pdf_loader.requests.get", return_value=mock_response), \
         patch("pdf_loader.tempfile.gettempdir", return_value=str(tmp_path)):
        result = load_pdf_from_url("http://example.com/sample.pdf")

    assert os.path.exists(result)
    with open(result, "rb") as f:
        assert f.read() == b"%PDF-1.4 fake content"


def test_load_documents_from_pdf_calls_loader():
    mock_loader = MagicMock()
    mock_docs = [MagicMock(), MagicMock()]
    mock_loader.load_data.return_value = mock_docs

    with patch("pdf_loader.PDFReader", return_value=mock_loader):
        result = load_documents_from_pdf("/some/path/doc.pdf")

    mock_loader.load_data.assert_called_once()
    assert result == mock_docs


def test_load_documents_from_pdf_returns_list():
    mock_loader = MagicMock()
    mock_loader.load_data.return_value = []

    with patch("pdf_loader.PDFReader", return_value=mock_loader):
        result = load_documents_from_pdf("/some/path/empty.pdf")

    assert result == []
