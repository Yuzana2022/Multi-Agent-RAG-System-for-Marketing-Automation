"""Unit tests for content_generation.py."""

from unittest.mock import MagicMock, patch

from agent.services.content_generation import (
    AllMarketingContents,
    MarketingContent,
    _get_chain,
    marketing_content_agent,
)
from agent.services.insights_extract import AllStrategicInsights, StrategicInsight


def _make_insights():
    return AllStrategicInsights(insights=[
        StrategicInsight(
            insight_id="i1",
            key_insight_content="AI personalisation increases conversions",
            strategic_relevance="Direct revenue impact",
        ),
    ])


def _make_contents():
    return AllMarketingContents(contents=[
        MarketingContent(
            insight_id="i1",
            content_format="Poster",
            headline="Boost Sales with AI",
            body_text="AI-driven personalisation can increase your conversion rate by 30%.",
            call_to_action="Start your free trial",
            target_audience="Marketing Managers",
            distribution_channel=["Facebook"],
            sources=["i1"],
        )
    ])


@patch("agent.services.content_generation._make_llm_with_structure")
def test_chain_is_initialized_lazily(mock_make_llm):
    """LLM must not be instantiated at import — only on first call."""
    import agent.services.content_generation as mod

    original = mod._chain
    mod._chain = None
    mock_make_llm.return_value = MagicMock()

    try:
        _get_chain()
        assert mock_make_llm.call_count == 1

        _get_chain()
        assert mock_make_llm.call_count == 1
    finally:
        mod._chain = original


@patch("agent.services.content_generation._chain")
def test_marketing_content_agent_returns_contents(mock_chain):
    """marketing_content_agent returns AllMarketingContents from chain."""
    expected = _make_contents()
    mock_chain.invoke.return_value = expected

    result = marketing_content_agent(_make_insights())

    assert isinstance(result, AllMarketingContents)
    assert len(result.contents) == 1
    mock_chain.invoke.assert_called_once()


@patch("agent.services.content_generation._chain")
def test_marketing_content_agent_handles_dict_response(mock_chain):
    """marketing_content_agent wraps a dict response in AllMarketingContents."""
    mock_chain.invoke.return_value = {
        "contents": [
            {
                "insight_id": "i1",
                "content_format": "Blog",
                "headline": "Test",
                "body_text": "Body",
                "call_to_action": "CTA",
                "target_audience": "Everyone",
                "distribution_channel": ["LinkedIn"],
                "sources": [],
            }
        ]
    }

    result = marketing_content_agent(_make_insights())

    assert isinstance(result, AllMarketingContents)
    assert result.contents[0].headline == "Test"


@patch("agent.services.content_generation._chain")
def test_formatted_insights_string_structure(mock_chain):
    """Verify formatted insights string passed to chain contains required fields."""
    mock_chain.invoke.return_value = _make_contents()

    marketing_content_agent(_make_insights())

    call_args = mock_chain.invoke.call_args[0][0]
    formatted = call_args["insights"]
    assert "ID:" in formatted
    assert "Insight:" in formatted
    assert "Relevance:" in formatted
    assert "i1" in formatted
