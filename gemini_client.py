"""Gemini LLM client initialization for the RAG pipeline."""

import os

from config import GOOGLE_API_KEY
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini

_DEFAULT_MODEL = "models/gemini-2.5-flash"
_DEFAULT_TEMPERATURE = 0.1
_DEFAULT_MAX_TOKENS = 1024

# Singleton container: list of 0 or 1 element avoids the global-statement lint warning.
_llm_cache: list[Gemini] = []


def initialize_gemini_llm(
    model: str = _DEFAULT_MODEL,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> Gemini:
    """Initialize and configure the Gemini LLM, assign it to LlamaIndex settings, and return it.

    Args:
        model: Gemini model identifier.
        temperature: Sampling temperature (0 = deterministic, 1 = creative).
        max_tokens: Maximum tokens in the generated response.

    Returns:
        Gemini: The configured LLM instance.
    """
    llm = Gemini(
        api_key=GOOGLE_API_KEY,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    Settings.llm = llm
    _llm_cache.clear()
    _llm_cache.append(llm)
    return llm


def get_llm() -> Gemini:
    """Return the current Gemini LLM instance, initializing with defaults if not yet done.

    Returns:
        Gemini: The active LLM instance.
    """
    if not _llm_cache:
        return initialize_gemini_llm()
    return _llm_cache[0]


def generate_text(prompt: str, temperature: float | None = None) -> str:
    """Generate a plain-text response for a single prompt without RAG context.

    Useful for tasks like summarization, reformatting, or standalone generation
    that do not require vector retrieval.

    Args:
        prompt: The text prompt to send to Gemini.
        temperature: Optional override for sampling temperature.

    Returns:
        str: The model's response text.

    Raises:
        ValueError: If prompt is empty.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")

    llm = get_llm()
    if temperature is not None:
        # Temporary instance at the requested temperature; does not replace the singleton.
        api_key = os.environ.get("GOOGLE_API_KEY", GOOGLE_API_KEY)
        tmp = Gemini(api_key=api_key, model=llm.model, temperature=temperature)
        response = tmp.complete(prompt)
    else:
        response = llm.complete(prompt)
    return str(response)


def summarize(text: str, max_words: int = 150) -> str:
    """Summarize a block of text using the Gemini LLM.

    Args:
        text: The text to summarize.
        max_words: Approximate target length for the summary in words.

    Returns:
        str: A concise summary of the input text.

    Raises:
        ValueError: If text is empty.
    """
    if not text or not text.strip():
        raise ValueError("text must not be empty")

    prompt = (
        f"Summarize the following text in approximately {max_words} words. "
        "Be concise and preserve key facts:\n\n"
        f"{text}"
    )
    return generate_text(prompt)


def rerank_passages(query: str, passages: list[str]) -> list[str]:
    """Ask Gemini to rerank retrieved passages by relevance to the query.

    Uses the LLM as a zero-shot reranker: passages are presented in a numbered
    list and the model returns the preferred ordering.

    Args:
        query: The user question.
        passages: List of text passages to rerank (order is input order).

    Returns:
        list[str]: Passages reordered from most to least relevant.
                   Falls back to the original order on parse failure.
    """
    if not passages:
        return []

    numbered = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(passages))
    prompt = (
        f"Query: {query}\n\n"
        f"Passages:\n{numbered}\n\n"
        "Return ONLY a comma-separated list of the passage numbers ordered from most "
        "to least relevant to the query (e.g. '3,1,2'). No explanation."
    )
    raw = generate_text(prompt, temperature=0.0).strip()
    try:
        indices = [int(x.strip()) - 1 for x in raw.split(",")]
        # Guard against duplicates or out-of-range values.
        seen: set[int] = set()
        reranked: list[str] = []
        for idx in indices:
            if 0 <= idx < len(passages) and idx not in seen:
                reranked.append(passages[idx])
                seen.add(idx)
        # Append any passages the model omitted.
        for i, p in enumerate(passages):
            if i not in seen:
                reranked.append(p)
        return reranked
    except (ValueError, IndexError):
        return passages


def reset_llm() -> None:
    """Clear the module-level LLM singleton.

    Primarily useful in tests to avoid state leaking between test cases.
    """
    _llm_cache.clear()
