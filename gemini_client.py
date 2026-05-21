"""Gemini LLM client initialization for LlamaIndex Settings."""

import os

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini

load_dotenv()


def initialize_gemini_llm():
    """Initialize a Gemini LLM instance and register it in LlamaIndex Settings."""
    llm = Gemini(
        api_key=os.environ["GOOGLE_API_KEY"],
        model="models/gemini-2.5-flash",
        temperature=0.1,
    )

    Settings.llm = llm

    return llm
