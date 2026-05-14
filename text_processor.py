from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 200


def setup_advanced_text_processing(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.embed_model = embed_model
    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_overlap
    return embed_model


def create_node_parser(
    strategy: str = "semantic",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    """Build a node parser.

    strategy:
      - "semantic": split where embedding similarity drops (best retrieval
        quality, falls back to sentence splitting if the dependency is
        unavailable).
      - "sentence": fixed-size sentence splitter with overlap.
    """
    if strategy == "semantic":
        try:
            from llama_index.core.node_parser import SemanticSplitterNodeParser

            return SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=Settings.embed_model,
            )
        except Exception as exc:
            print(f"Semantic splitter unavailable ({exc}); using sentence splitter.")

    return SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
        secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?",
    )
