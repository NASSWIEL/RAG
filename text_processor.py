"""Text processing utilities for embedding model setup and sentence-level node parsing."""

from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def setup_advanced_text_processing():
    """Configure global LlamaIndex settings with a HuggingFace embedding model and chunk parameters.
    """
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    return embed_model


def create_node_parser():
    """Create and return a SentenceSplitter node parser with predefined chunk size and overlap."""
    parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )
    return parser
