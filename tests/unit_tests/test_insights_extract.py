"""Unit tests for insights_extract.py."""

from unittest.mock import MagicMock, patch

import pytest

from agent.services.insights_extract import (
    AllStrategicInsights,
    StrategicInsight,
    _get_chain,
    insights_agent,
)
from agent.services.rag_agent import RagResult


def _make_insights(*ids):
    return AllStrategicInsights(insights=[
        StrategicInsight(
            insight_id=i,
            key_insight_content=f"Content {i}",
            strategic_relevance=f"Relevance {i}",
        )
        for i in ids
    ])


def _make_rag_result():
    return RagResult(content=["AI insight text", "ML insight text"])


@patch("agent.services.insights_extract._make_llm_with_structure")
def test_chain_is_initialized_lazily(mock_make_llm):
    """LLM must not be instantiated at module import — only on first call."""
    import agent.services.insights_extract as mod

    # Reset the cached chain to simulate fresh import
    original = mod._chain
    mod._chain = None
    mock_make_llm.return_value = MagicMock()

    try:
        _get_chain()
        assert mock_make_llm.call_count == 1

        # Second call should NOT create another LLM
        _get_chain()
        assert mock_make_llm.call_count == 1
    finally:
        mod._chain = original  # restore


@patch("agent.services.insights_extract._chain")
def test_insights_agent_returns_structured_output(mock_chain):
    """insights_agent returns AllStrategicInsights when chain returns one."""
    expected = _make_insights("i1", "i2")
    mock_chain.invoke.return_value = expected

    result = insights_agent(_make_rag_result())

    assert isinstance(result, AllStrategicInsights)
    assert len(result.insights) == 2
    mock_chain.invoke.assert_called_once()


@patch("agent.services.insights_extract._chain")
def test_insights_agent_handles_dict_response(mock_chain):
    """insights_agent wraps a raw dict response in AllStrategicInsights."""
    dict_response = {
        "insights": [
            {
                "insight_id": "i1",
                "key_insight_content": "AI matters",
                "strategic_relevance": "High impact",
            }
        ]
    }
    mock_chain.invoke.return_value = dict_response

    result = insights_agent(_make_rag_result())

    assert isinstance(result, AllStrategicInsights)
    assert result.insights[0].insight_id == "i1"


@patch("agent.services.insights_extract._chain")
def test_insights_agent_passes_content_to_chain(mock_chain):
    """insights_agent joins RAG content and passes it to the chain."""
    mock_chain.invoke.return_value = _make_insights("i1")
    rag = RagResult(content=["line one", "line two"])

    insights_agent(rag)

    invocation_args = mock_chain.invoke.call_args[0][0]
    assert "line one" in invocation_args["insights"]
    assert "line two" in invocation_args["insights"]
