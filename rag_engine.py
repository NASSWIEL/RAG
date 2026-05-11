"""RAG engine for semantic search and question answering over PDF documents."""

import hashlib
import os

from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage

from gemini_client import initialize_gemini_llm
from pdf_loader import load_documents_from_pdf, load_pdf_from_url
from text_processor import create_node_parser, setup_advanced_text_processing


class RAGEngine:
    """Retrieval-Augmented Generation engine that indexes a PDF and answers questions over it."""

    def __init__(self, pdf_url, storage_dir="./storage"):
        """Initialize the RAG engine by loading or building a vector index from the given PDF URL.

        Args:
            pdf_url: URL of the PDF document to index.
            storage_dir: Directory used to persist and load cached embeddings.
        """
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

        if os.path.exists(index_path):
            print("Found existing embeddings for this PDF! Loading from cache...")
            print(f"Storage location: {index_path}")
            self.index = self._load_index(index_path)
            print("Index loaded successfully! Skipping embedding generation.")
        else:
            print("No cached embeddings found. Processing PDF for the first time...")
            print("Loading PDF from URL...")
            pdf_path = load_pdf_from_url(pdf_url)

            print("Extracting documents from PDF...")
            documents = load_documents_from_pdf(pdf_path)

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

    def _save_index(self, index_path):
        os.makedirs(index_path, exist_ok=True)
        self.index.storage_context.persist(persist_dir=index_path)

    def _load_index(self, index_path):
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        return load_index_from_storage(storage_context)

    def query(self, question):
        """Query the RAG engine with a question and return the generated answer.

        Args:
            question: The question to ask against the indexed PDF content.

        Returns:
            str: The generated answer from the language model.
        """
        print(f"\nQuery: {question}")
        print("Performing semantic retrieval and generating answer...")

        response = self.query_engine.query(question)

        return str(response)
