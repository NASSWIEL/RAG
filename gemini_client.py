"""Gemini LLM client initialization for the RAG pipeline."""

from config import GOOGLE_API_KEY
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini


def initialize_gemini_llm():
    """Initialize and configure the Gemini LLM, assign it to LlamaIndex settings, and return it."""
    llm = Gemini(
        api_key=GOOGLE_API_KEY,
        model="models/gemini-2.5-flash",
        temperature=0.1,
    )

    Settings.llm = llm

    return llm
