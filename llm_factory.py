"""Provider-neutral LangChain LLM factory for DocuMind AI."""
import os
from functools import lru_cache         # Used to cache the created LLM objects

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI         # both expose LangChain chat-model inteface
from langchain_groq import ChatGroq

load_dotenv()

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

# to convert fifferent user/provider inputs into one cannonical provider name
def _normalize_provider(provider: str | None) -> str:
    value = (provider or DEFAULT_PROVIDER).strip().lower()
    aliases = {                     # normalization map - as rest of the code only use openai & groq rather than mulitple alias.
        "chatgpt": "openai",
        "openai": "openai",
        "groq": "groq",
    }
    if value not in aliases:
        raise ValueError(
            f"Unsupported LLM provider '{provider}'. Choose 'openai' or 'groq'."
        )
    return aliases[value]

# This is the core of the file - everthing else exists mainly to support this function.
# Why @lru_cache(maxsize=4) ( Without caching , get_llm("groq") could construct a new ChatGroq object every time,same for openai
# .So, the cache hold up to four distinct argument combinations - more than enough fot two currently supported providers. )
@lru_cache(maxsize=4)
def get_llm(provider: str | None = None):
    """Return a LangChain chat model with the same interface for both providers."""
    provider_name = _normalize_provider(provider)

    if provider_name == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Add it to your .env file."
            )
        return ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0,      # to reduce unnecessary randomness
        )

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
    )