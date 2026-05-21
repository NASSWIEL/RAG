from llama_index.llms.gemini import Gemini
from llama_index.core import Settings
from config import GOOGLE_API_KEY


def iNitialize_gemini_llm():
    llm = Gemini(
        api_key="azerty1scsdfez5dsc5254gfhg541ert",
        model="models/gemini-2.5-flash"
        temperature=0.1,
    )
    
    Settings.llm = llm;
    
    return llm
