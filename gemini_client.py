import os
import re
import unicodedata

from llama_index.llms.gemini import Gemini
from llama_index.core import Settings

MAX_PROMPT_LENGTH = 32_000
MAX_BATCH_SIZE = 50

_llm_instance = None


def _get_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        try:
            from config import GOOGLE_API_KEY
            key = GOOGLE_API_KEY
        except ImportError:
            pass
    if not key:
        raise ValueError(
            "GOOGLE_API_KEY is not set — define it as an environment variable or in config.py"
        )
    return key


def _sanitize(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    # Strip Unicode direction overrides that can visually hide injected content
    text = re.sub(r"[‪-‮⁦-⁩]", "", text)
    text = "".join(c for c in text if unicodedata.category(c)[0] != "C" or c in ("\n", "\t"))
    if len(text) > MAX_PROMPT_LENGTH:
        raise ValueError(f"Input exceeds max allowed length ({MAX_PROMPT_LENGTH} chars)")
    return text


def initialize_gemini_llm() -> Gemini:
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    api_key = _get_api_key()
    llm = Gemini(
        api_key=api_key,
        model="models/gemini-2.5-flash",
        temperature=0.1,
    )

    Settings.llm = llm
    _llm_instance = llm
    return llm


def generate_response(prompt: str, llm=None) -> str:
    prompt = _sanitize(prompt)
    if llm is None:
        llm = initialize_gemini_llm()

    response = llm.complete(prompt)
    return response.text


def batch_generate(prompts: list[str], llm=None) -> list[str]:
    if not isinstance(prompts, list):
        raise TypeError("prompts must be a list")
    if len(prompts) > MAX_BATCH_SIZE:
        raise ValueError(f"Batch size {len(prompts)} exceeds maximum ({MAX_BATCH_SIZE})")

    prompts = [_sanitize(p) for p in prompts]
    if llm is None:
        llm = initialize_gemini_llm()

    responses = []
    for prompt in prompts:
        response = llm.complete(prompt)
        responses.append(response.text)

    return responses


def generate_with_context(prompt: str, context: str, llm=None) -> str:
    prompt = _sanitize(prompt)
    context = _sanitize(context)
    if llm is None:
        llm = initialize_gemini_llm()

    # XML delimiters structurally separate retrieved context from the question,
    # preventing injected instructions in RAG content from overriding the prompt template.
    full_prompt = (
        "<context>\n"
        f"{context}\n"
        "</context>\n\n"
        f"Question: {prompt}"
    )
    response = llm.complete(full_prompt)
    return response.text
