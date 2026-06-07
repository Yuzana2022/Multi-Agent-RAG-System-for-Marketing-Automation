"""Unit tests for search_doc_load.py."""

from unittest.mock import MagicMock, patch

import pytest

from agent.services.search_agent import AllSearchResults, SearchResult
from agent.services.search_doc_load import AllSearchDocResults, SearchDocResult, text_loader


def _make_search_results(*links):
    return AllSearchResults(results=[
        SearchResult(
            query="test",
            link=link,
            title=f"Title for {link}",
            source="TestSource",
            snippet="snippet",
        )
        for link in links
    ])


def _mock_loader_doc(content: str):
    doc = MagicMock()
    doc.page_content = content
    return doc


@patch("agent.services.search_doc_load._clean_chain")
@patch("agent.services.search_doc_load.WebBaseLoader")
def test_text_loader_happy_path(mock_loader_cls, mock_chain):
    """All URLs load and clean successfully."""
    mock_loader_cls.return_value.load.return_value = [
        _mock_loader_doc("word " * 20)  # >10 words per chunk
    ]
    cleaned = MagicMock()
    cleaned.content = "Cleaned article content."
    mock_chain.invoke.return_value = cleaned

    results = text_loader(_make_search_results("https://a.com", "https://b.com"))

    assert isinstance(results, AllSearchDocResults)
    assert len(results.results) == 2
    assert all(r.content == "Cleaned article content." for r in results.results)


@patch("agent.services.search_doc_load._clean_chain")
@patch("agent.services.search_doc_load.WebBaseLoader")
def test_text_loader_skips_failed_url(mock_loader_cls, mock_chain):
    """A URL that keeps failing is skipped; others succeed."""
    # First call raises, second succeeds
    mock_loader_cls.return_value.load.side_effect = [
        Exception("connection timeout"),
        [_mock_loader_doc("word " * 20)],
    ]
    cleaned = MagicMock()
    cleaned.content = "Good content."
    mock_chain.invoke.return_value = cleaned

    results = text_loader(_make_search_results("https://bad.com", "https://good.com"))

    # Only the successful URL should appear in results
    assert len(results.results) == 1
    assert results.results[0].content == "Good content."


@patch("agent.services.search_doc_load._clean_chain")
@patch("agent.services.search_doc_load.WebBaseLoader")
def test_text_loader_empty_input(mock_loader_cls, mock_chain):
    """Empty search results produce empty web doc results."""
    results = text_loader(AllSearchResults(results=[]))

    assert isinstance(results, AllSearchDocResults)
    assert results.results == []
    mock_loader_cls.assert_not_called()


@patch("agent.services.search_doc_load._clean_chain")
@patch("agent.services.search_doc_load.WebBaseLoader")
def test_text_loader_filters_short_chunks(mock_loader_cls, mock_chain):
    """Lines with fewer than MIN_CHUNK_WORD_COUNT words are stripped."""
    # Raw content with a nav-bar-like short line and a real paragraph
    raw = "nav\n\nThis is a real paragraph with more than ten words in it and content here.\n\nshort"
    mock_loader_cls.return_value.load.return_value = [_mock_loader_doc(raw)]

    cleaned = MagicMock()
    cleaned.content = "Real paragraph only."
    mock_chain.invoke.return_value = cleaned

    results = text_loader(_make_search_results("https://example.com"))

    # Verify LLM was invoked (short lines were filtered before LLM call)
    mock_chain.invoke.assert_called_once()
    call_raw = mock_chain.invoke.call_args[0][0]["raw_content"]
    assert "nav" not in call_raw
    assert "short" not in call_raw
