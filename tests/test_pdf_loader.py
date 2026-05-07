"""Tests for pdf_loader.py."""
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

# Stub out uninstalled third-party dependencies before importing pdf_loader
_requests_mock = MagicMock()
sys.modules.setdefault("requests", _requests_mock)

_llama_index_mock = MagicMock()
_llama_readers_mock = MagicMock()
_llama_readers_file_mock = MagicMock()
sys.modules.setdefault("llama_index", _llama_index_mock)
sys.modules.setdefault("llama_index.readers", _llama_readers_mock)
sys.modules.setdefault("llama_index.readers.file", _llama_readers_file_mock)

from pdf_loader import load_pdf_from_url, load_documents_from_pdf  # noqa: E402


def test_load_pdf_from_url_returns_path():
    mock_response = MagicMock()
    mock_response.content = b"%PDF-1.4 fake pdf content"

    with patch("pdf_loader.requests.get", return_value=mock_response) as mock_get:
        result = load_pdf_from_url("http://example.com/sample.pdf")

    mock_get.assert_called_once_with("http://example.com/sample.pdf")
    assert isinstance(result, str)
    assert result.endswith("temp_rag_document.pdf")


def test_load_pdf_from_url_writes_content(tmp_path):
    fake_content = b"%PDF-1.4 binary data here"
    mock_response = MagicMock()
    mock_response.content = fake_content
    temp_file = str(tmp_path / "temp_rag_document.pdf")

    with patch("pdf_loader.requests.get", return_value=mock_response):
        with patch("pdf_loader.os.path.join", return_value=temp_file):
            result = load_pdf_from_url("http://example.com/sample.pdf")

    assert result == temp_file
    with open(temp_file, "rb") as f:
        assert f.read() == fake_content


def test_load_documents_from_pdf_returns_documents(tmp_path):
    fake_doc = MagicMock()
    fake_doc.text = "Hello from PDF"
    mock_loader_instance = MagicMock()
    mock_loader_instance.load_data.return_value = [fake_doc]

    with patch("pdf_loader.PDFReader", return_value=mock_loader_instance):
        pdf_path = str(tmp_path / "sample.pdf")
        Path(pdf_path).write_bytes(b"%PDF-1.4")
        result = load_documents_from_pdf(pdf_path)

    mock_loader_instance.load_data.assert_called_once_with(file=Path(pdf_path))
    assert result == [fake_doc]


def test_load_documents_from_pdf_empty_result(tmp_path):
    mock_loader_instance = MagicMock()
    mock_loader_instance.load_data.return_value = []

    with patch("pdf_loader.PDFReader", return_value=mock_loader_instance):
        pdf_path = str(tmp_path / "empty.pdf")
        Path(pdf_path).write_bytes(b"%PDF-1.4")
        result = load_documents_from_pdf(pdf_path)

    assert result == []
