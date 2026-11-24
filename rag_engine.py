"""RAG engine for building and querying a vector index from a PDF document."""

import contextlib
import hashlib
import os
import shutil
import tempfile
import time
from urllib.parse import unquote, urlparse

from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage

from gemini_client import initialize_gemini_llm
from pdf_loader import load_documents_from_pdf, load_pdf_from_url
from text_processor import create_node_parser, setup_advanced_text_processing

# Written last inside a saved index; its absence means the folder is incomplete.
_COMPLETE_MARKER = "_complete"
_REPLACE_ATTEMPTS = 5
_REPLACE_RETRY_DELAY_S = 0.5
# Older than any realistic persist, so a temp dir of a run still in progress is never removed.
_STALE_TMP_AGE_S = 3600


class RAGEngine:
    """Retrieval-Augmented Generation engine that indexes a PDF and answers queries."""

    def __init__(self, pdf_url, storage_dir="./storage"):
        """Initialize the RAG engine by loading or building a vector index for the given PDF."""
        self.storage_dir = storage_dir
        self.pdf_url = pdf_url

        print("Setting up advanced text processing with state-of-the-art embeddings...")
        self.embed_model = setup_advanced_text_processing()

        print("Initializing Gemini LLM...")
        self.llm = initialize_gemini_llm()

        print("Creating node parser...")
        self.parser = create_node_parser()
        Settings.node_parser = self.parser

        index_path = self._get_index_path()

        if self._is_complete(index_path):
            print("Found existing embeddings for this PDF! Loading from cache...")
            print(f"Storage location: {index_path}")
            self.index = self._load_index(index_path)
            print("Index loaded successfully! Skipping embedding generation.")
        else:
            print("No cached embeddings found. Processing PDF for the first time...")
            print("Loading PDF from URL...")
            pdf_path = load_pdf_from_url(pdf_url)

            print("Extracting documents from PDF...")
            try:
                source_name = unquote(os.path.basename(urlparse(pdf_url).path))
                if not source_name.lower().endswith(".pdf"):
                    source_name = "document.pdf"
                documents = load_documents_from_pdf(pdf_path, file_name=source_name)
            finally:
                # A cleanup failure (e.g. antivirus lock on Windows) must not mask the parse result.
                with contextlib.suppress(OSError):
                    os.remove(pdf_path)

            print("Building vector index with semantic search...")
            self.index = VectorStoreIndex.from_documents(documents, show_progress=True)

            print("Saving embeddings to disk for future use...")
            self._save_index(index_path)
            print(f"Embeddings saved to: {index_path}")

        print("Creating query engine with advanced retrieval...")
        self.query_engine = self.index.as_query_engine(similarity_top_k=3, response_mode="compact")

        print("RAG Engine ready with state-of-the-art processing!")

    def _get_index_path(self):
        url_hash = hashlib.md5(self.pdf_url.encode(), usedforsecurity=False).hexdigest()
        return os.path.join(self.storage_dir, f"index_{url_hash}")

    @staticmethod
    def _is_complete(index_path):
        return os.path.exists(os.path.join(index_path, _COMPLETE_MARKER))

    def _save_index(self, index_path):
        # Persist into a per-run temp dir, then rename it into place, so concurrent runs never
        # share a folder and a failed persist never leaves an index_path that looks valid.
        os.makedirs(self.storage_dir, exist_ok=True)
        self._remove_stale_tmp_dirs(index_path)
        tmp_path = tempfile.mkdtemp(
            prefix=os.path.basename(index_path) + ".", suffix=".tmp", dir=self.storage_dir
        )
        try:
            self.index.storage_context.persist(persist_dir=tmp_path)
            with open(os.path.join(tmp_path, _COMPLETE_MARKER), "w", encoding="utf-8"):
                pass
            self._replace_dir(tmp_path, index_path)
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    def _remove_stale_tmp_dirs(self, index_path):
        prefix = os.path.basename(index_path) + "."
        now = time.time()
        for name in os.listdir(self.storage_dir):
            path = os.path.join(self.storage_dir, name)
            if not (name.startswith(prefix) and name.endswith(".tmp") and os.path.isdir(path)):
                continue
            with contextlib.suppress(OSError):
                if now - os.path.getmtime(path) > _STALE_TMP_AGE_S:
                    shutil.rmtree(path, ignore_errors=True)

    def _replace_dir(self, tmp_path, index_path):
        # Retries cover Windows antivirus/indexer handles that briefly block a directory
        # rename or delete, and another run removing the same incomplete folder concurrently.
        for attempt in range(_REPLACE_ATTEMPTS):
            if os.path.exists(index_path) and not self._is_complete(index_path):
                shutil.rmtree(index_path, ignore_errors=True)
            try:
                os.replace(tmp_path, index_path)
                return
            except OSError:
                if self._is_complete(index_path):
                    print(
                        f"Another run already saved a complete index at {index_path}; "
                        "keeping it and discarding this one."
                    )
                    return
                if attempt == _REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(_REPLACE_RETRY_DELAY_S)

    def _load_index(self, index_path):
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        return load_index_from_storage(storage_context)

    def query(self, question):
        """Query the index with a natural language question and return the answer.

        Args:
            question: The natural language question to answer.

        Returns:
            str: The generated answer from the RAG pipeline.
        """
        print(f"\nQuery: {question}")
        print("Performing semantic retrieval and generating answer...")

        response = self.query_engine.query(question)

        return str(response)
