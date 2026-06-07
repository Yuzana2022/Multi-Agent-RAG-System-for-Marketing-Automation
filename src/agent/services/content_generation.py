import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .config import CONTENT_LLM_TEMPERATURE, DEFAULT_LLM_MODEL
from .insights_extract import AllStrategicInsights
from .llm_model import _make_llm_with_structure

logger = logging.getLogger(__name__)


class MarketingContent(BaseModel):
    insight_id: str = Field(..., description="Matches the ID from the source insight")
    content_format: str = Field(..., description="e.g., Blog, Articles, Poster, Video Script, Study Case")
    headline: str = Field(..., description="Catchy headline for the content")
    body_text: str = Field(..., description="The full copy of the marketing content")
    call_to_action: str = Field(..., description="Clear CTA for the reader")
    target_audience: str = Field(..., description="Primary persona this content targets")
    distribution_channel: List[str] = Field(..., description="Recommended channels, e.g., ['LinkedIn', 'Twitter']")
    sources: List[str] = Field(..., description="Reference IDs or links used")


class AllMarketingContents(BaseModel):
    contents: List[MarketingContent]


CONTENT_PROMPT = ChatPromptTemplate.from_template(
    """
    You are an expert Marketing Strategist.
    Generate concrete, ready-to-publish marketing content based on the following strategic insights.

    For each insight:
    1. Select the most effective content format for the target audience.
    2. Write professional, high-converting copy including a headline and body text.
    3. Ensure a compelling Call-to-Action (CTA).

    Strategic Insights to process:
    {insights}
    """
)

# Lazy-initialized chain — created on first call to avoid eager API key validation
# at import time and to simplify test mocking.
_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        llm = _make_llm_with_structure(AllMarketingContents, DEFAULT_LLM_MODEL, CONTENT_LLM_TEMPERATURE)
        _chain = CONTENT_PROMPT | llm
    return _chain


def marketing_content_agent(insights: AllStrategicInsights) -> AllMarketingContents:
    formatted_insights = "\n".join(
        f"ID: {i.insight_id} | Insight: {i.key_insight_content} | Relevance: {i.strategic_relevance}"
        for i in insights.insights
    )

    res = _get_chain().invoke({"insights": formatted_insights})

    if isinstance(res, AllMarketingContents):
        return res
    return AllMarketingContents(**res)
