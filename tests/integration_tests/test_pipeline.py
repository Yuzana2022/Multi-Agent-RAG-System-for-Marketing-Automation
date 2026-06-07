"""Integration tests for the full marketing pipeline.

These tests mock all external I/O (APIs, LLMs, DB) and verify the complete
graph wiring — state flows, conditional routing, and skip flags.

Run with: make integration_tests
"""

from unittest.mock import MagicMock, patch

import pytest

from agent.graph import run_marketing_pipeline
from agent.services.auto_analysis_report import AnalyticsReport
from agent.services.auto_publish import DistributorOutput, SinglePostResult
from agent.services.content_generation import AllMarketingContents, MarketingContent
from agent.services.insights_extract import AllStrategicInsights, StrategicInsight
from agent.services.local_doc_load import AllLocalDocResults
from agent.services.rag_agent import RagResult
from agent.services.search_agent import AllSearchResults, SearchResult
from agent.services.search_doc_load import AllSearchDocResults, SearchDocResult


# ── Shared mock data ──────────────────────────────────────────────────────────

def _fake_search(query):
    return AllSearchResults(results=[
        SearchResult(
            query=query, link="https://example.com",
            title="AI News", source="TechNews", snippet="AI matters.",
        )
    ])


def _fake_text_loader(results):
    return AllSearchDocResults(results=[
        SearchDocResult(title="AI News", content="AI is changing marketing strategies.")
    ])


def _fake_reduce_agent(web, local, db_path, query, force_rebuild=False):
    return RagResult(content=["AI is changing marketing strategies."])


def _fake_insights_agent(rag):
    return AllStrategicInsights(insights=[
        StrategicInsight(
            insight_id="i1",
            key_insight_content="AI adoption is accelerating",
            strategic_relevance="High business impact",
        )
    ])


def _fake_content_agent(insights):
    return AllMarketingContents(contents=[
        MarketingContent(
            insight_id="i1",
            content_format="Poster",
            headline="AI is the Future",
            body_text="Embrace AI to stay competitive.",
            call_to_action="Learn more",
            target_audience="CMOs",
            distribution_channel=["Facebook"],
            sources=["i1"],
        )
    ])


# ── Tests ─────────────────────────────────────────────────────────────────────

@patch("agent.graph.google_search", side_effect=_fake_search)
@patch("agent.graph.text_loader", side_effect=_fake_text_loader)
@patch("agent.graph.reduce_agent", side_effect=_fake_reduce_agent)
@patch("agent.graph.insights_agent", side_effect=_fake_insights_agent)
@patch("agent.graph.marketing_content_agent", side_effect=_fake_content_agent)
def test_full_pipeline_skip_publishing(
    mock_content, mock_insights, mock_rag, mock_loader, mock_search
):
    """Pipeline runs end-to-end with all external calls mocked; publishing skipped."""
    result = run_marketing_pipeline(
        query="AI marketing trends",
        skip_publishing=True,
        skip_analytics=True,
    )
    
    assert result["errors"] == []
    assert result["search_results"] is not None
    assert result["web_documents"] is not None
    assert result["rag_results"] is not None
    assert result["strategic_insights"] is not None
    assert result["marketing_contents"] is not None
    assert result["publish_results"] is None


@patch("agent.graph.google_search", side_effect=_fake_search)
@patch("agent.graph.text_loader", side_effect=_fake_text_loader)
@patch("agent.graph.reduce_agent", side_effect=_fake_reduce_agent)
@patch("agent.graph.insights_agent", side_effect=_fake_insights_agent)
@patch("agent.graph.marketing_content_agent", side_effect=_fake_content_agent)
def test_human_approval_approved_runs_publishing(
    mock_content, mock_insights, mock_rag, mock_loader, mock_search
):
    """With human approval callback that approves, publishing node is reached."""
    with patch("agent.graph.distributor_agent") as mock_publish:
        mock_publish.return_value = DistributorOutput(results=[
            SinglePostResult(content_id="i1", post_id="post_123", status="success", response={"id": "post_123"})
        ])

        approval_callback = MagicMock(return_value=("approved", "Looks good"))

        result = run_marketing_pipeline(
            query="AI marketing trends",
            require_human_approval=True,
            approval_callback=approval_callback,
            skip_analytics=True,
            facebook_page_id="test_page_123",           # 添加假数据，mock_publish.assert_called_once()则会被调用1次
            facebook_access_token="test_token_abc",     # 添加假数据，mock_publish.assert_called_once()则会被调用1次
        )

    assert result["human_approval"] == "approved"
    assert result["human_feedback"] == "Looks good"
    mock_publish.assert_called_once()  # 现在应该通过了


@patch("agent.graph.google_search", side_effect=_fake_search)
@patch("agent.graph.text_loader", side_effect=_fake_text_loader)
@patch("agent.graph.reduce_agent", side_effect=_fake_reduce_agent)
@patch("agent.graph.insights_agent", side_effect=_fake_insights_agent)
@patch("agent.graph.marketing_content_agent", side_effect=_fake_content_agent)
def test_human_approval_rejected_skips_publishing(
    mock_content, mock_insights, mock_rag, mock_loader, mock_search
):
    """With human approval callback that rejects, publishing node is skipped."""
    with patch("agent.graph.distributor_agent") as mock_publish:
        approval_callback = MagicMock(return_value=("rejected", "Not ready"))

        result = run_marketing_pipeline(
            query="AI marketing trends",
            require_human_approval=True,
            approval_callback=approval_callback,
            skip_analytics=True,
        )

    assert result["human_approval"] == "rejected"
    assert result["publish_results"] is None
    mock_publish.assert_not_called()


@patch("agent.graph.google_search", side_effect=Exception("SerpAPI down"))
def test_pipeline_recovers_from_search_failure(mock_search):
    """Pipeline records the error but continues to the end without raising."""
    result = run_marketing_pipeline(
        query="AI marketing",
        skip_publishing=True,
        skip_analytics=True,
    )

    assert any("Search node error" in e for e in result["errors"])
    # Pipeline should not raise — it accumulates errors and returns
    assert "errors" in result
