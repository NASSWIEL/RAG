"""Utilities for loading PDF documents from URLs or local paths."""

import contextlib
import os
import tempfile
from pathlib import Path

import requests
from llama_index.readers.file import PDFReader


def load_pdf_from_url(url):
    """Download a PDF from a URL and save it to a temporary file.

    The caller owns the returned file and must delete it when done.

    Args:
        url: The URL of the PDF to download.

    Returns:
        str: The local file path of the downloaded PDF.

    Raises:
        requests.HTTPError: If the server answers with a 4xx/5xx status.
        ValueError: If the response body is not a PDF (e.g. an HTML page served with 200).
    """
    print(f"Downloading PDF from {url}...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    # PDF readers accept a short preamble (BOM, whitespace) before the header.
    if b"%PDF" not in response.content[:1024]:
        content_type = response.headers.get("Content-Type", "unknown")
        raise ValueError(f"URL did not return a PDF (Content-Type: {content_type}): {url}")

    f = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        with f:
            f.write(response.content)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(f.name)
        raise

    return f.name

def load_documents_from_pdf(pdf_path, file_name=None):
    """Load documents from a local PDF file using PDFReader.

    Args:
        pdf_path: The local file path of the PDF to load.
        file_name: Name stored in each document's metadata instead of the local file name.

    Returns:
        list: A list of documents extracted from the PDF.
    """
    loader = PDFReader()
    extra_info = {"file_name": file_name} if file_name else None
    documents = loader.load_data(file=Path(pdf_path), extra_info=extra_info)
    return documents
