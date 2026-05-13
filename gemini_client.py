from llama_index.llms.gemini import Gemini
from llama_index.core import Settings
from config import GOOGLE_API_KEY


def initialize_gemini_llm():
    llm = Gemini(
        api_key=GOOGLE_API_KEY,
        model="models/gemini-2.5-flash",
        temperature=0.1,
    )

    Settings.llm = llm

    return llm


def generate_response(prompt: str, llm=None) -> str:
    if llm is None:
        llm = initialize_gemini_llm()

    response = llm.complete(prompt)
    return response.text


def batch_generate(prompts: list[str], llm=None) -> list[str]:
    if llm is None:
        llm = initialize_gemini_llm()

    responses = []
    for prompt in prompts:
        response = llm.complete(prompt)
        responses.append(response.text)

    return responses


def generate_with_context(prompt: str, context: str, llm=None) -> str:
    if llm is None:
        llm = initialize_gemini_llm()

    full_prompt = f"Context: {context}\n\nQuestion: {prompt}"
    response = llm.complete(full_prompt)
    return response.text
