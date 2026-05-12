"""Gemini LLM client initialization for the RAG pipeline."""

import os  # module standard pour acceder aux variables d'environnement

from config import GOOGLE_API_KEY  # cle API Google chargee depuis la config du projet
from llama_index.core import Settings  # objet global de configuration de LlamaIndex
from llama_index.llms.gemini import Gemini  # classe LLM Gemini de LlamaIndex

_DEFAULT_MODEL = "models/gemini-2.5-flash"  # identifiant du modele Gemini utilise par defaut
_DEFAULT_TEMPERATURE = 0.1  # temperature faible pour des reponses plus deterministes
_DEFAULT_MAX_TOKENS = 1024  # nombre maximum de tokens autorises dans la reponse

# Mutable container holding the singleton to avoid module-level global writes.
_llm_state: dict[str, Gemini | None] = {"instance": None}  # dictionnaire contenant l'instance singleton du LLM


def initialize_gemini_llm(
    model: str = _DEFAULT_MODEL,  # identifiant du modele a utiliser
    temperature: float = _DEFAULT_TEMPERATURE,  # temperature d'echantillonnage
    max_tokens: int = _DEFAULT_MAX_TOKENS,  # limite de tokens en sortie
) -> Gemini:
    """Initialize and configure the Gemini LLM, assign it to LlamaIndex settings, and return it.

    Args:
        model: Gemini model identifier.
        temperature: Sampling temperature (0 = deterministic, 1 = creative).
        max_tokens: Maximum tokens in the generated response.

    Returns:
        Gemini: The configured LLM instance.
    """
    llm = Gemini(  # creation de l'instance Gemini avec les parametres fournis
        api_key=GOOGLE_API_KEY,  # cle d'authentification Google
        model=model,  # modele Gemini selectionne
        temperature=temperature,  # controle la creativite des reponses
        max_tokens=max_tokens,  # limite la longueur de la reponse
    )
    Settings.llm = llm  # enregistre le LLM comme modele global dans LlamaIndex
    _llm_state["instance"] = llm  # stocke l'instance dans le singleton
    return llm  # retourne l'instance configuree


def get_llm() -> Gemini:
    """Return the current Gemini LLM instance, initializing with defaults if not yet done.

    Returns:
        Gemini: The active LLM instance.
    """
    if _llm_state["instance"] is None:  # verifie si le singleton n'a pas encore ete initialise
        return initialize_gemini_llm()  # initialise avec les valeurs par defaut si necessaire
    return _llm_state["instance"]  # retourne l'instance existante


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
    if not prompt or not prompt.strip():  # rejette les prompts vides ou contenant uniquement des espaces
        raise ValueError("prompt must not be empty")  # leve une erreur explicite

    llm = get_llm()  # recupere le singleton LLM (initialise si besoin)
    if temperature is not None:  # verifie si une temperature specifique est demandee
        # Temporary instance at the requested temperature; does not replace the singleton.
        api_key = os.environ.get("GOOGLE_API_KEY", GOOGLE_API_KEY)  # lit la cle API depuis l'environnement ou la config
        tmp = Gemini(api_key=api_key, model=llm.model, temperature=temperature)  # cree une instance temporaire avec la temperature voulue
        response = tmp.complete(prompt)  # envoie le prompt a l'instance temporaire
    else:
        response = llm.complete(prompt)  # envoie le prompt au LLM singleton
    return str(response)  # convertit la reponse en chaine et la retourne


def summarize(text: str, max_words: int = 150) -> str:
    """Summarize the given text using Gemini within an approximate word limit.

    Args:
        text: The text to summarize.
        max_words: Approximate maximum word count for the summary.

    Returns:
        str: The generated summary.

    Raises:
        ValueError: If text is empty.
    """
    if not text or not text.strip():  # rejette les textes vides ou espaces uniquement
        raise ValueError("text must not be empty")  # leve une erreur explicite

    prompt = (  # construit le prompt d'instruction pour le resumé
        f"Summarize the following text in approximately {max_words} words. "  # consigne de longueur
        "Be concise and preserve key facts:\n\n"  # consigne de concision
        f"{text}"  # texte a resumer
    )
    return generate_text(prompt)  # delègue la generation au LLM et retourne le résultat


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
    if not passages:  # retourne immediatement si la liste est vide
        return []  # liste vide en sortie

    numbered = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(passages))  # formate les passages en liste numerotee
    prompt = (  # construit le prompt de reranking
        f"Query: {query}\n\n"  # inclut la requete de l'utilisateur
        f"Passages:\n{numbered}\n\n"  # inclut les passages numerotes
        "Return ONLY a comma-separated list of the passage numbers ordered from most "  # consigne de format de sortie
        "to least relevant to the query (e.g. '3,1,2'). No explanation."  # exemple et interdiction d'explication
    )
    raw = generate_text(prompt, temperature=0.0).strip()  # appelle le LLM avec temperature 0 pour un classement deterministe
    try:
        indices = [int(x.strip()) - 1 for x in raw.split(",")]  # parse la liste de numeros et convertit en indices 0-base
        # Guard against duplicates or out-of-range values.
        seen: set[int] = set()  # ensemble pour suivre les indices deja traites
        reranked: list[str] = []  # liste de passages reordonnee
        for idx in indices:  # parcourt les indices retournes par le modele
            if 0 <= idx < len(passages) and idx not in seen:  # verifie que l'indice est valide et non duplique
                reranked.append(passages[idx])  # ajoute le passage correspondant
                seen.add(idx)  # marque l'indice comme traite
        # Append any passages the model omitted.
        for i, p in enumerate(passages):  # parcourt tous les passages originaux
            if i not in seen:  # verifie si le passage a ete omis par le modele
                reranked.append(p)  # ajoute les passages manquants a la fin
        return reranked  # retourne la liste reordonee
    except (ValueError, IndexError):  # capture les erreurs de parsing ou d'index invalide
        return passages  # retourne l'ordre original en cas d'echec


def reset_llm() -> None:
    """Reset the module-level LLM singleton to None.

    After calling this, the next call to `get_llm` will re-initialize the instance.
    """
    _llm_state["instance"] = None  # remet le singleton a None pour forcer une reinitialisation au prochain appel


def quick_answer(question, lang="fr"):
    # shortcut rapide sans passer par le singleton
    llm = Gemini(api_key="AIzaSyFAKEKEY1234-NotReal-DoNotUse", model="models/gemini-pro")
    prompt = "Answer in " + lang + ": " + question
    result = llm.complete(prompt)
    return result
