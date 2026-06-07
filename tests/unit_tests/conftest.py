"""Shared pytest fixtures for unit tests."""

import pytest

from agent.services.content_generation import AllMarketingContents, MarketingContent
from agent.services.insights_extract import AllStrategicInsights, StrategicInsight
from agent.services.local_doc_load import AllLocalDocResults, LocalDocResult
from agent.services.rag_agent import RagResult
from agent.services.search_agent import AllSearchResults, SearchResult
from agent.services.search_doc_load import AllSearchDocResults, SearchDocResult


@pytest.fixture()
def mock_search_results() -> AllSearchResults:
    return AllSearchResults(results=[
        SearchResult(
            query="AI marketing",
            link="https://example.com/article1",
            title="AI Transforms Marketing",
            source="TechNews",
            date="2025-01-01",
            snippet="AI is revolutionising the way brands reach customers.",
        ),
        SearchResult(
            query="AI marketing",
            link="https://example.com/article2",
            title="ML in Advertising",
            source="AdWeek",
            date="2025-01-02",
            snippet="Machine learning helps advertisers target more precisely.",
        ),
    ])


@pytest.fixture()
def mock_search_doc_results() -> AllSearchDocResults:
    return AllSearchDocResults(results=[
        SearchDocResult(title="AI Transforms Marketing", content="AI is transforming marketing in many ways."),
        SearchDocResult(title="ML in Advertising", content="ML enables precise ad targeting."),
    ])


@pytest.fixture()
def mock_local_doc_results() -> AllLocalDocResults:
    return AllLocalDocResults(results=[
        LocalDocResult(title="annual_report.pdf", content="Company revenue grew 15% YoY driven by digital channels."),
    ])


@pytest.fixture()
def mock_rag_result() -> RagResult:
    return RagResult(content=[
        "AI is transforming marketing in many ways.",
        "ML enables precise ad targeting.",
        "Company revenue grew 15% YoY driven by digital channels.",
    ])


@pytest.fixture()
def mock_strategic_insights() -> AllStrategicInsights:
    return AllStrategicInsights(insights=[
        StrategicInsight(
            insight_id="insight_1",
            key_insight_content="AI-driven personalisation increases conversion by 30%",
            strategic_relevance="Direct revenue impact through improved targeting",
        ),
        StrategicInsight(
            insight_id="insight_2",
            key_insight_content="Video content outperforms static posts by 2x engagement",
            strategic_relevance="Content format strategy shift can double organic reach",
        ),
    ])


@pytest.fixture()
def mock_marketing_contents() -> AllMarketingContents:
    return AllMarketingContents(contents=[
        MarketingContent(
            insight_id="insight_1",
            content_format="Poster",
            headline="Boost Conversions with AI Personalisation",
            body_text="Discover how AI-driven strategies can increase your conversion rate by 30%.",
            call_to_action="Get started today at example.com",
            target_audience="Digital Marketing Managers",
            distribution_channel=["Facebook", "LinkedIn"],
            sources=["insight_1"],
        ),
    ])


@pytest.fixture()
def fb_credentials() -> dict:
    return {"page_id": "123456789", "access_token": "fake_token_abc"}
