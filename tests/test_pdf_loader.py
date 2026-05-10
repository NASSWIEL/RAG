"""Tests for C:/retrodoc/test-BT-AI/RAG/pdf_loader.py."""
import sys
import os as _os

# Ensure project root is on sys.path so pdf_loader can be imported
_project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from unittest.mock import MagicMock

# Pre-mock heavy third-party packages so the module under test can be imported
for _pkg in [
    "requests",
    "llama_index",
    "llama_index.readers",
    "llama_index.readers.file",
]:
    sys.modules.setdefault(_pkg, MagicMock())

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Remove any pre-mocked pdf_loader (e.g. from test_main.py) so the real module is used
sys.modules.pop("pdf_loader", None)

from pdf_loader import load_pdf_from_url, load_documents_from_pdf


def test_load_pdf_from_url_returns_temp_path():
    mock_response = MagicMock()
    mock_response.content = b"%PDF-1.4 fake content"

    with patch("pdf_loader.requests.get", return_value=mock_response) as mock_get:
        result = load_pdf_from_url("http://example.com/doc.pdf")

    mock_get.assert_called_once_with("http://example.com/doc.pdf", timeout=30)
    expected_path = os.path.join(tempfile.gettempdir(), "temp_rag_document.pdf")
    assert result == expected_path


def test_load_pdf_from_url_writes_content_to_file(tmp_path):
    mock_response = MagicMock()
    mock_response.content = b"%PDF-1.4 fake binary content"

    fake_temp = str(tmp_path / "temp_rag_document.pdf")

    with patch("pdf_loader.requests.get", return_value=mock_response):
        with patch("pdf_loader.os.path.join", return_value=fake_temp):
            result = load_pdf_from_url("http://example.com/sample.pdf")

    assert result == fake_temp
    with open(fake_temp, "rb") as f:
        assert f.read() == b"%PDF-1.4 fake binary content"


def test_load_documents_from_pdf_returns_documents():
    mock_loader = MagicMock()
    mock_documents = [MagicMock(), MagicMock()]
    mock_loader.load_data.return_value = mock_documents

    with patch("pdf_loader.PDFReader", return_value=mock_loader) as mock_reader_cls:
        result = load_documents_from_pdf("/some/path/doc.pdf")

    mock_reader_cls.assert_called_once_with()
    mock_loader.load_data.assert_called_once_with(file=Path("/some/path/doc.pdf"))
    assert result == mock_documents


def test_load_documents_from_pdf_passes_path_as_pathlib():
    mock_loader = MagicMock()
    mock_loader.load_data.return_value = []

    with patch("pdf_loader.PDFReader", return_value=mock_loader):
        load_documents_from_pdf("/another/path/report.pdf")

    call_kwargs = mock_loader.load_data.call_args
    assert call_kwargs.kwargs["file"] == Path("/another/path/report.pdf")
