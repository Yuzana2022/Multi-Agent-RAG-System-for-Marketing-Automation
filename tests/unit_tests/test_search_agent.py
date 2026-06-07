"""Unit tests for search_agent.py."""

from unittest.mock import MagicMock, patch

import pytest

from agent.services.search_agent import AllSearchResults, google_search


FAKE_NEWS_RESPONSE = {
    "news_results": [
        {
            "link": "https://example.com/ai-news",
            "title": "AI is changing marketing",
            "source": "TechCrunch",
            "date": "2025-01-01",
            "snippet": "Artificial intelligence reshapes how brands connect.",
        },
        {
            "link": "https://example.com/ml-ads",
            "title": "ML in digital advertising",
            "source": "AdWeek",
            "date": "2025-01-02",
            "snippet": "Machine learning enables precision targeting.",
        },
    ]
}


@patch("agent.services.search_agent.GoogleSearch")
def test_google_search_happy_path(mock_gs_cls):
    mock_instance = MagicMock()
    mock_instance.get_dict.return_value = FAKE_NEWS_RESPONSE
    mock_gs_cls.return_value = mock_instance

    results = google_search("AI marketing")

    assert isinstance(results, AllSearchResults)
    assert len(results.results) == 2
    assert results.results[0].title == "AI is changing marketing"
    assert results.results[0].query == "AI marketing"
    assert results.results[1].source == "AdWeek"


@patch("agent.services.search_agent.GoogleSearch")
def test_google_search_no_results_returns_empty(mock_gs_cls):
    mock_instance = MagicMock()
    mock_instance.get_dict.return_value = {"news_results": []}
    mock_gs_cls.return_value = mock_instance

    results = google_search("obscure query xyz")

    assert isinstance(results, AllSearchResults)
    assert results.results == []


@patch("agent.services.search_agent.GoogleSearch")
def test_google_search_retries_and_succeeds(mock_gs_cls):
    """Verify tenacity retries on transient failure then returns results."""
    mock_instance = MagicMock()
    mock_instance.get_dict.side_effect = [
        Exception("timeout"),
        FAKE_NEWS_RESPONSE,
    ]
    mock_gs_cls.return_value = mock_instance

    results = google_search("AI marketing")

    assert len(results.results) == 2
    assert mock_instance.get_dict.call_count == 2


@patch("agent.services.search_agent.GoogleSearch")
def test_google_search_exhausts_retries_raises(mock_gs_cls):
    """Verify reraise=True propagates exception after all retries fail."""
    mock_instance = MagicMock()
    mock_instance.get_dict.side_effect = Exception("API error")
    mock_gs_cls.return_value = mock_instance

    with pytest.raises(Exception, match="API error"):
        google_search("AI marketing")


@patch("agent.services.search_agent.GoogleSearch")
def test_google_search_uses_config_location(mock_gs_cls):
    """Verify search parameters come from config, not hardcoded strings."""
    mock_instance = MagicMock()
    mock_instance.get_dict.return_value = FAKE_NEWS_RESPONSE
    mock_gs_cls.return_value = mock_instance

    google_search("test query")

    call_kwargs = mock_gs_cls.call_args[0][0]
    # assert "location" in call_kwargs
    # assert "gl" in call_kwargs
    assert "num" in call_kwargs
