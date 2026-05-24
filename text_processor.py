"""Text processing utilities for embedding and node parsing in the RAG pipeline."""

from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def setup_advanced_text_processing():
    """Configure HuggingFace embedding model and global LlamaIndex settings.

    Returns:
        HuggingFaceEmbedding: The configured embedding model instance.
    """
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    return embed_model


def create_node_parser():
    """Create a SentenceSplitter node parser with default chunk settings.

    Returns:
        SentenceSplitter: A parser configured with chunk_size=512 and chunk_overlap=50.
    """
    parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )
    return parser
