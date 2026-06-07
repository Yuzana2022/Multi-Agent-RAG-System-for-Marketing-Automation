import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .config import DEFAULT_LLM_MODEL, INSIGHTS_LLM_TEMPERATURE
from .llm_model import _make_llm_with_structure
from .rag_agent import RagResult

logger = logging.getLogger(__name__)


class StrategicInsight(BaseModel):
    insight_id: str = Field(..., description="Unique identifier for the insight")
    key_insight_content: str = Field(..., description="The main content of the key insight")
    strategic_relevance: str = Field(
        ..., description="Explanation of why this insight is strategically relevant to the business"
    )


class AllStrategicInsights(BaseModel):
    insights: List[StrategicInsight]


INSIGHTS_PROMPT = ChatPromptTemplate.from_template(
    """
    Merge, deduplicate and synthesize the following insights
    into FIVE most important and strategic insights.

    Insights:
    {insights}

    Output JSON strictly matching:
    {schema}
    """,
    partial_variables={"schema": AllStrategicInsights.model_json_schema()},
)

# Lazy-initialized chain — created on first call to avoid eager API key validation
# at import time and to simplify test mocking.
_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        llm = _make_llm_with_structure(AllStrategicInsights, DEFAULT_LLM_MODEL, INSIGHTS_LLM_TEMPERATURE)
        _chain = INSIGHTS_PROMPT | llm
    return _chain


def insights_agent(raw_insights: RagResult) -> AllStrategicInsights:
    raw_insights = RagResult(**raw_insights.model_dump())
    res = _get_chain().invoke({"insights": "\n".join(raw_insights.content)})
    if isinstance(res, AllStrategicInsights):
        return res
    return AllStrategicInsights(**res)
