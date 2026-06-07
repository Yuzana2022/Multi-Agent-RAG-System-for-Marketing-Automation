import logging

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .env_utils import OPENAI_API_KEY, OPENAI_BASE_URL

logger = logging.getLogger(__name__)


def _embed_model(model: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=model,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )


def _make_llm(model: str, temperature: float) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )


def _make_llm_with_structure(schema, model: str, temperature: float):
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    ).with_structured_output(schema)
