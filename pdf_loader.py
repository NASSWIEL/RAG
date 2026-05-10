"""Utilities for loading PDF documents from URLs and local file paths."""

import os
import tempfile
from pathlib import Path

import requests
from llama_index.readers.file import PDFReader


def load_pdf_from_url(url):
    """Download a PDF from the given URL and save it to a temporary file.

    Args:
        url: The URL of the PDF to download.

    Returns:
        str: The local file path of the downloaded PDF.
    """
    print(f"Downloading PDF from {url}...")
    response = requests.get(url, timeout=30)

    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "temp_rag_document.pdf")

    with open(temp_path, "wb") as f:
        f.write(response.content)

    return temp_path


def load_documents_from_pdf(pdf_path):
    """Load and parse documents from a local PDF file.

    Args:
        pdf_path: The local file path to the PDF file.

    Returns:
        list: A list of documents extracted from the PDF.
    """
    loader = PDFReader()
    documents = loader.load_data(file=Path(pdf_path))
    return documents
