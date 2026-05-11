"""Text processing utilities for configuring embeddings and node parsing with LlamaIndex."""

from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def setup_advanced_text_processing():
    """Configure LlamaIndex global settings with a HuggingFace embedding model and chunk parameters.

    Returns:
        HuggingFaceEmbedding: The embedding model instance configured with BAAI/bge-small-en-v1.5.
    """
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    return embed_model


def create_node_parser():
    """Create and return a SentenceSplitter node parser with default chunk size and overlap.

    Returns:
        SentenceSplitter: A node parser configured with chunk_size=512 and chunk_overlap=50.
    """
    parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )
    return parser
