"""Unit tests for rag_agent.py."""

import os
from unittest.mock import MagicMock, call, patch

import pytest

from agent.services.local_doc_load import AllLocalDocResults, LocalDocResult
from agent.services.rag_agent import RagResult, map_agent, reduce_agent, search_with_threshold
from agent.services.search_doc_load import AllSearchDocResults, SearchDocResult


def _web_docs(*titles):
    return AllSearchDocResults(results=[
        SearchDocResult(title=t, content=f"Content for {t}") for t in titles
    ])


def _local_docs(*titles):
    return AllLocalDocResults(results=[
        LocalDocResult(title=t, content=f"PDF content for {t}") for t in titles
    ])


@patch("agent.services.rag_agent.Chroma.from_documents")
@patch("agent.services.rag_agent._embed_model")
def test_map_agent_calls_chroma(mock_embed, mock_chroma_from_docs):
    """map_agent embeds documents and stores them in Chroma."""
    mock_embed.return_value = MagicMock()
    mock_chroma_from_docs.return_value = MagicMock()

    map_agent(_web_docs("Article A"), "/tmp/test_db")

    mock_chroma_from_docs.assert_called_once()
    call_kwargs = mock_chroma_from_docs.call_args[1]
    assert call_kwargs["persist_directory"] == "/tmp/test_db"


def test_reduce_agent_empty_inputs_returns_empty():
    """No documents → immediate empty RagResult, no Chroma calls."""
    result = reduce_agent(
        AllSearchDocResults(results=[]),
        AllLocalDocResults(results=[]),
        "/tmp/test_db",
        query="test",
    )

    assert isinstance(result, RagResult)
    assert result.content == []


@patch("agent.services.rag_agent.Chroma")
@patch("agent.services.rag_agent.map_agent")
def test_reduce_agent_explicit_query_param(mock_map, mock_chroma_cls):
    """Bug regression: reduce_agent accepts query as param (no .query attribute access)."""
    mock_chroma_instance = MagicMock()
    mock_chroma_cls.return_value = mock_chroma_instance
    mock_chroma_instance.similarity_search_with_relevance_scores.return_value = [
        (MagicMock(page_content="relevant content"), 0.9),
    ]

    # This must NOT raise AttributeError (old bug: accessed .query on SearchDocResult)
    result = reduce_agent(
        _web_docs("Article A"),
        AllLocalDocResults(results=[]),
        "/tmp/test_db",
        query="AI marketing trends",
    )

    assert "relevant content" in result.content
    mock_chroma_instance.similarity_search_with_relevance_scores.assert_called_once_with(
        "AI marketing trends", k=pytest.approx(50, abs=1)
    )


@patch("agent.services.rag_agent.shutil.rmtree")
@patch("agent.services.rag_agent.os.path.exists", return_value=True)
@patch("agent.services.rag_agent.Chroma")
@patch("agent.services.rag_agent.map_agent")
def test_reduce_agent_force_rebuild_deletes_db(mock_map, mock_chroma_cls, mock_exists, mock_rmtree):
    """force_rebuild=True must delete the existing DB directory."""
    mock_chroma_cls.return_value.similarity_search_with_relevance_scores.return_value = []

    reduce_agent(_web_docs("A"), _local_docs(), "/tmp/db", query="q", force_rebuild=True)

    mock_rmtree.assert_called_once_with("/tmp/db")


@patch("agent.services.rag_agent.shutil.rmtree")
@patch("agent.services.rag_agent.os.path.exists", return_value=True)
@patch("agent.services.rag_agent.Chroma")
@patch("agent.services.rag_agent.map_agent")
def test_reduce_agent_no_force_rebuild_preserves_db(mock_map, mock_chroma_cls, mock_exists, mock_rmtree):
    """force_rebuild=False (default) must NOT delete the DB."""
    mock_chroma_cls.return_value.similarity_search_with_relevance_scores.return_value = []

    reduce_agent(_web_docs("A"), _local_docs(), "/tmp/db", query="q", force_rebuild=False)

    mock_rmtree.assert_not_called()


def test_search_with_threshold_filters_low_scores():
    """Only documents at or above the threshold score are returned."""
    doc_high = MagicMock(page_content="high score doc")
    doc_low = MagicMock(page_content="low score doc")

    vectorstore = MagicMock()
    vectorstore.similarity_search_with_relevance_scores.return_value = [
        (doc_high, 0.8),
        (doc_low, 0.3),
    ]

    results = search_with_threshold(vectorstore, "query", threshold=0.5)

    assert len(results) == 1
    assert results[0].page_content == "high score doc"
