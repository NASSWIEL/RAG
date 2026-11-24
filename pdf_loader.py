"""Utilities for loading PDF documents from URLs or local paths."""

import tempfile
from pathlib import Path

import requests
from llama_index.readers.file import PDFReader


def load_pdf_from_url(url):
    """Download a PDF from a URL and save it to a temporary file.

    Args:
        url: The URL of the PDF to download.

    Returns:
        str: The local file path of the downloaded PDF.
    """
    print(f"Downloading PDF from {url}...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(response.content)

    return f.name

def load_documents_from_pdf(pdf_path):
    """Load documents from a local PDF file using PDFReader.

    Args:
        pdf_path: The local file path of the PDF to load.

    Returns:
        list: A list of documents extracted from the PDF.
    """
    loader = PDFReader()
    documents = loader.load_data(file=Path(pdf_path))
    return documents
