import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    MIN_CHUNK_WORD_COUNT,
    RETRY_ATTEMPTS,
    RETRY_WAIT_MAX,
    RETRY_WAIT_MIN,
    WEB_LOADER_MAX_WORKERS,
)
from .llm_model import _make_llm
from .search_agent import AllSearchResults

logger = logging.getLogger(__name__)


class SearchDocResult(BaseModel):
    title: str = Field(..., description="The title of the search result")
    content: str = Field(..., description="The main body content re-edited from the crawled webpage by LLM")


class AllSearchDocResults(BaseModel):
    results: List[SearchDocResult] = Field(default_factory=list)


# LLM and prompt chain are built once at module load — not inside the per-URL loop.
_strict_llm = _make_llm(DEFAULT_LLM_MODEL, DEFAULT_LLM_TEMPERATURE)
_CLEAN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a Senior Copy Editor specializing in proofreading the text content."),
    (
        "human",
        "The following is raw text extracted from a website. "
        "Please filter out noise such as navigation bars, button text, and dates "
        "to preserve only the primary body content.\n\nRaw Content: {raw_content}",
    ),
])
_clean_chain = _CLEAN_PROMPT | _strict_llm


@retry(
    wait=wait_exponential(min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _load_and_clean(link: str, title: str) -> SearchDocResult:
    """Load a single URL, strip noise via LLM, and return a SearchDocResult."""
    loader = WebBaseLoader(link)
    docs = loader.load()
    raw = docs[0].page_content

    clean_chunks = [
        chunk.strip()
        for chunk in raw.split("\n")
        if chunk.strip() and len(chunk.strip().split()) >= MIN_CHUNK_WORD_COUNT
    ]
    final_content = "\n\n".join(clean_chunks)

    cleaned = _clean_chain.invoke({"raw_content": final_content})
    return SearchDocResult(title=title, content=cleaned.content)


def text_loader(search_results: AllSearchResults) -> AllSearchDocResults:
    """Load and clean web content from all search result URLs in parallel."""
    output_results = AllSearchDocResults()

    with ThreadPoolExecutor(max_workers=WEB_LOADER_MAX_WORKERS) as executor:
        future_to_result = {
            executor.submit(_load_and_clean, r.link, r.title): r
            for r in search_results.results
        }
        for future in as_completed(future_to_result):
            r = future_to_result[future]
            try:
                output_results.results.append(future.result())
            except Exception:
                logger.warning("Skipping URL %s after retries exhausted", r.link, exc_info=True)

    logger.info(
        "Loaded %d/%d web documents",
        len(output_results.results),
        len(search_results.results),
    )
    return output_results
