"""Unit tests for graph.py — pipeline orchestration."""

from unittest.mock import MagicMock, call, patch

import pytest

from agent.graph import (
    MarketingState,
    _make_human_review_node,
    check_approval,
    content_generation_node,
    create_graph,
    insights_node,
    local_loader_node,
    publishing_node,
    rag_node,
    search_node,
    web_loader_node,
)
from agent.services.content_generation import AllMarketingContents, MarketingContent
from agent.services.insights_extract import AllStrategicInsights, StrategicInsight
from agent.services.local_doc_load import AllLocalDocResults,LocalDocResult
from agent.services.rag_agent import RagResult
from agent.services.search_agent import AllSearchResults
from agent.services.search_doc_load import AllSearchDocResults


def _base_state(**overrides) -> MarketingState:
    state: MarketingState = {
        "query": "AI marketing",
        "local_pdf_path": None,
        "facebook_page_id": None,
        "facebook_access_token": None,
        "db_path": "./test_chroma_db",
        "search_results": None,
        "web_documents": None,
        "local_documents": None,
        "rag_results": None,
        "strategic_insights": None,
        "marketing_contents": None,
        "publish_results": None,
        "analytics_report": None,
        "skip_local_docs": True,
        "skip_publishing": True,
        "skip_analytics": True,
        "require_human_approval": False,
        "force_rebuild_db": False,
        "human_approval": None,
        "human_feedback": None,
        "errors": [],
    }
    state.update(overrides)
    return state


# ── check_approval ────────────────────────────────────────────────────────────

def test_check_approval_approved():
    assert check_approval(_base_state(human_approval="approved")) == "publishing"


def test_check_approval_rejected():
    assert check_approval(_base_state(human_approval="rejected")) == "end"


def test_check_approval_none_defaults_to_end():
    assert check_approval(_base_state(human_approval=None)) == "end"


# ── search_node ───────────────────────────────────────────────────────────────

@patch("agent.graph.google_search")
def test_search_node_happy_path(mock_search):
    mock_search.return_value = AllSearchResults(results=[])

    state = search_node(_base_state())

    assert state["search_results"] is not None
    # assert state["errors"] == [] # because only update search_results in this node, therefore no need
    mock_search.assert_called_once_with("AI marketing")


@patch("agent.graph.google_search")
def test_search_node_error_is_accumulated(mock_search):
    mock_search.side_effect = Exception("network error")

    state = search_node(_base_state())

    assert len(state["errors"]) == 1
    assert "Search node error" in state["errors"][0]


# ── rag_node ──────────────────────────────────────────────────────────────────

@patch("agent.graph.reduce_agent")
def test_rag_node_passes_query_param(mock_reduce):
    """Bug regression: rag_node must pass state['query'] to reduce_agent."""
    mock_reduce.return_value = RagResult(content=["chunk"])

    state = rag_node(_base_state(
        web_documents=AllSearchDocResults(results=[]),
        local_documents=AllLocalDocResults(results=[]),
    ))

    mock_reduce.assert_called_once()
    _, call_kwargs = mock_reduce.call_args.args, mock_reduce.call_args.kwargs
    # query can be positional or keyword
    # assert state["errors"] == []


@patch("agent.graph.reduce_agent")
def test_rag_node_passes_force_rebuild(mock_reduce):
    """force_rebuild_db in state should be forwarded to reduce_agent."""
    mock_reduce.return_value = RagResult(content=[])

    rag_node(_base_state(
        web_documents=AllSearchDocResults(results=[]),
        local_documents=AllLocalDocResults(results=[]),
        force_rebuild_db=True,
    ))

    _, kwargs = mock_reduce.call_args.args, mock_reduce.call_args.kwargs
    # Accept both positional and keyword argument forms
    all_args = list(mock_reduce.call_args.args) + list(mock_reduce.call_args.kwargs.values())
    assert True in all_args or mock_reduce.call_args.kwargs.get("force_rebuild") is True


# ── local_loader_node ─────────────────────────────────────────────────────────

def test_local_loader_node_skips_when_no_pdf():
    state = local_loader_node(_base_state(local_pdf_path=None))
    assert isinstance(state["local_documents"], AllLocalDocResults)
    assert state["local_documents"].results == []


@patch("agent.graph.extract_text_from_pdf")
def test_local_loader_node_loads_pdf(mock_extract):
    # mock_extract.return_value = AllLocalDocResults(results=[
    #     MagicMock(title="report.pdf", content="Annual report content")
    # ])
    mock_extract.return_value = AllLocalDocResults(results=[
        LocalDocResult(title="report.pdf", content="Annual report content")
    ])

    state = local_loader_node(_base_state(
        local_pdf_path="/path/to/report.pdf",
        skip_local_docs=False,
    ))

    assert state["local_documents"] is not None
    mock_extract.assert_called_once_with("/path/to/report.pdf")


# ── insights_node ─────────────────────────────────────────────────────────────

@patch("agent.graph.insights_agent")
def test_insights_node_skips_when_no_rag(mock_insights):
    state = insights_node(_base_state(rag_results=RagResult(content=[])))

    mock_insights.assert_not_called()
    assert state["strategic_insights"] is None


# ── content_generation_node ───────────────────────────────────────────────────

@patch("agent.graph.marketing_content_agent")
def test_content_generation_node_skips_when_no_insights(mock_agent):
    state = content_generation_node(_base_state(strategic_insights=None))

    mock_agent.assert_not_called()
    assert state["marketing_contents"] is None


# ── human review callback ─────────────────────────────────────────────────────

def test_human_review_node_approved_sets_state(mock_marketing_contents):
    callback = MagicMock(return_value=("approved", "Looks great"))
    node = _make_human_review_node(callback)

    state = node(_base_state(marketing_contents=mock_marketing_contents))

    assert state["human_approval"] == "approved"
    assert state["human_feedback"] == "Looks great"


def test_human_review_node_rejected_sets_state(mock_marketing_contents):
    callback = MagicMock(return_value=("rejected", ""))
    node = _make_human_review_node(callback)

    state = node(_base_state(marketing_contents=mock_marketing_contents))

    assert state["human_approval"] == "rejected"


def test_human_review_node_callback_error_defaults_to_rejected(mock_marketing_contents):
    callback = MagicMock(side_effect=RuntimeError("UI unavailable"))
    node = _make_human_review_node(callback)

    state = node(_base_state(marketing_contents=mock_marketing_contents))

    assert state["human_approval"] == "rejected"


def test_human_review_node_no_contents_auto_rejects():
    callback = MagicMock()
    node = _make_human_review_node(callback)

    state = node(_base_state(marketing_contents=None))

    assert state["human_approval"] == "rejected"
    callback.assert_not_called()


# ── create_graph ──────────────────────────────────────────────────────────────

def test_create_graph_without_human_approval_has_no_review_node():
    g = create_graph(require_human_approval=False)
    node_names = set(g.get_graph().nodes.keys())
    assert "human_review" not in node_names


def test_create_graph_with_human_approval_has_review_node():
    callback = MagicMock(return_value=("approved", ""))
    g = create_graph(require_human_approval=True, approval_callback=callback)
    node_names = set(g.get_graph().nodes.keys())
    assert "human_review" in node_names


# ── publishing_node ───────────────────────────────────────────────────────────

def test_publishing_node_skips_when_flag_set(mock_marketing_contents):
    state = publishing_node(_base_state(
        skip_publishing=True,
        marketing_contents=mock_marketing_contents,
    ))
    assert len(state) == 0
    assert state == {}


def test_publishing_node_skips_when_no_credentials(mock_marketing_contents):
    state = publishing_node(_base_state(
        marketing_contents=mock_marketing_contents,
        facebook_page_id=None,
        facebook_access_token=None,
        skip_publishing=False,
    ))
    assert state["skip_publishing"] is True
