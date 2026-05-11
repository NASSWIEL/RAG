"""Gemini LLM client initialization for use with LlamaIndex."""

from config import GOOGLE_API_KEY
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini


def initialize_gemini_llm():
    """Initialize and configure a Gemini LLM instance and register it with LlamaIndex Settings.

    Returns:
        Gemini: The configured Gemini LLM instance.
    """
    llm = Gemini(
        api_key=GOOGLE_API_KEY,
        model="models/gemini-2.5-flash",
        temperature=0.1,
    )

    Settings.llm = llm

    return llm
