"""Text processing utilities for embedding setup and node parsing."""

from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def setup_advanced_text_processing():
    """Configure LlamaIndex settings with a HuggingFace embedding model and chunking parameters.

    Returns:
        HuggingFaceEmbedding: The embedding model instance registered in Settings.
    """
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    return embed_model


def create_node_parser():
    """Create a SentenceSplitter node parser with default chunk size and overlap.

    Returns:
        SentenceSplitter: Configured parser instance.
    """
    parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )
    return parser
