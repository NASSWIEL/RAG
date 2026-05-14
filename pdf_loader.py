import hashlib
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from llama_index.readers.file import PDFReader

DOWNLOAD_TIMEOUT = 60
CHUNK_BYTES = 1 << 15
PDF_MAGIC = b"%PDF-"


def _is_local_path(source: str) -> bool:
    return os.path.exists(source) or urlparse(source).scheme in ("", "file")


def _cache_path_for_url(url: str) -> str:
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return os.path.join(tempfile.gettempdir(), f"rag_pdf_{url_hash}.pdf")


def _validate_pdf(path: str) -> None:
    with open(path, "rb") as fh:
        head = fh.read(len(PDF_MAGIC))
    if head != PDF_MAGIC:
        raise ValueError(f"Downloaded file is not a valid PDF: {path}")


def load_pdf_from_url(url: str) -> str:
    """Download a PDF (or return a local path) and return the file path.

    - Streams the download to avoid loading large PDFs fully in memory.
    - Caches per-URL so re-runs reuse the same file.
    - Validates the PDF magic header before returning.
    """
    if _is_local_path(url):
        path = url[7:] if url.startswith("file://") else url
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        _validate_pdf(path)
        return path

    target = _cache_path_for_url(url)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        try:
            _validate_pdf(target)
            print(f"Reusing cached PDF: {target}")
            return target
        except ValueError:
            os.remove(target)

    print(f"Downloading PDF from {url}...")
    tmp = target + ".part"
    try:
        with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as resp:
            resp.raise_for_status()
            ctype = resp.headers.get("Content-Type", "").lower()
            if ctype and "pdf" not in ctype and "octet-stream" not in ctype:
                print(f"Warning: unexpected Content-Type '{ctype}', continuing.")
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=CHUNK_BYTES):
                    if chunk:
                        fh.write(chunk)
        _validate_pdf(tmp)
        os.replace(tmp, target)
        return target
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def load_documents_from_pdf(pdf_path: str):
    """Extract documents from a PDF, preserving page-level metadata."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(pdf_path)

    loader = PDFReader(return_full_document=False)
    documents = loader.load_data(file=path)

    source_name = path.name
    for doc in documents:
        doc.metadata = doc.metadata or {}
        doc.metadata.setdefault("source", source_name)
        doc.metadata.setdefault("file_path", str(path))

    return documents
