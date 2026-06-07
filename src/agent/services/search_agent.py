import logging
from typing import List, Optional

from pydantic import BaseModel, Field
from serpapi import GoogleSearch
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import (
    RETRY_ATTEMPTS,
    RETRY_WAIT_MAX,
    RETRY_WAIT_MIN,
    # SERPAPI_GL,
    SERPAPI_HL,
    # SERPAPI_LOCATION,
    SERPAPI_NUM_RESULTS,
    SERPAPI_TBM,
)
from .env_utils import SERPAPI_API_KEY

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    query: str = Field(..., description="The search query")
    link: str = Field(..., description="The URL of the search result")
    title: str = Field(..., description="The title of the search result")
    source: str = Field(..., description="The source of the search result")
    date: Optional[str] = Field(None, description="The publication date")
    snippet: str = Field(..., description="A brief summary")


class AllSearchResults(BaseModel):
    results: List[SearchResult] = Field(default_factory=list)


@retry(
    wait=wait_exponential(min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _do_search(query: str) -> dict:
    search = GoogleSearch({
        "q": query,
        # "location": SERPAPI_LOCATION,
        # "gl": SERPAPI_GL,
        "hl": SERPAPI_HL,
        "tbm": SERPAPI_TBM,
        "num": SERPAPI_NUM_RESULTS,
        "api_key": SERPAPI_API_KEY,
    })
    return search.get_dict()


def google_search(query: str) -> AllSearchResults:
    all_results = AllSearchResults()

    res = _do_search(query)
    news_items = res.get("news_results", [])

    if not news_items:
        logger.warning("No news results for query: %s", query)
        return all_results

    for item in news_items:
        result_obj = SearchResult(
            query=query,
            link=item.get("link", ""),
            title=item.get("title", ""),
            source=item.get("source", ""),
            date=item.get("date", ""),
            snippet=item.get("snippet", ""),
        )
        all_results.results.append(result_obj)
        logger.debug("Found: %s | %s | %s", result_obj.title, result_obj.source, result_obj.link)

    logger.info("Search returned %d results for query: %s", len(all_results.results), query)
    return all_results
