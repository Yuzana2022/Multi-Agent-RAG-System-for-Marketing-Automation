"""Unit tests for auto_publish.py."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from agent.services.auto_publish import (
    BasePublisher,
    DistributorOutput,
    FacebookPostRequest,
    FacebookPublisher,
    SinglePostResult,
    distributor_agent,
)
from agent.services.content_generation import AllMarketingContents, MarketingContent


def _make_content(content_format="Poster", channels=None):
    return MarketingContent(
        insight_id="i1",
        content_format=content_format,
        headline="Test Headline",
        body_text="Test body text for the marketing post.",
        call_to_action="Click here",
        target_audience="Digital Marketers",
        distribution_channel=channels or ["Facebook"],
        sources=["i1"],
    )


def _make_request(contents):
    return FacebookPostRequest(
        marketing_data=AllMarketingContents(contents=contents),
        page_id="page123",
        access_token="token_abc",
    )


# ── BasePublisher ──────────────────────────────────────────────────────────────

def test_base_publisher_is_abstract():
    with pytest.raises(TypeError):
        BasePublisher()  # type: ignore[abstract]


# ── FacebookPublisher.can_publish ─────────────────────────────────────────────

@pytest.mark.parametrize("fmt,channels,expected", [
    ("Poster", ["LinkedIn"], True),          # matches format
    ("Blog", ["Facebook"], True),            # matches channel
    ("Blog", ["LinkedIn", "Twitter"], False), # no match
    ("Article", ["YouTube"], False),          # no match
])
def test_facebook_publisher_can_publish_routing(fmt, channels, expected):
    pub = FacebookPublisher()
    item = _make_content(content_format=fmt, channels=channels)
    assert pub.can_publish(item) is expected


# ── FacebookPublisher.publish ──────────────────────────────────────────────────

@patch("agent.services.auto_publish.requests.post")
def test_facebook_publisher_publish_success(mock_post):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"id": "post_999"}
    mock_resp.raise_for_status.return_value = None
    mock_post.return_value = mock_resp

    pub = FacebookPublisher()
    result = pub.publish(_make_content(), page_id="p1", access_token="tok")

    assert result.status == "success"
    assert result.post_id == "post_999"
    assert result.content_id == "i1"


@patch("agent.services.auto_publish.requests.post")
def test_facebook_publisher_publish_http_error_returns_failed(mock_post):
    mock_post.side_effect = requests.RequestException("500 Server Error")

    pub = FacebookPublisher()
    result = pub.publish(_make_content(), page_id="p1", access_token="tok")

    assert result.status == "failed"
    assert "error" in result.response


@patch("agent.services.auto_publish.requests.post")
def test_facebook_publisher_retries_on_request_exception(mock_post):
    """Verify tenacity retries transient network errors before succeeding."""
    success_resp = MagicMock()
    success_resp.json.return_value = {"id": "post_42"}
    success_resp.raise_for_status.return_value = None

    mock_post.side_effect = [
        requests.RequestException("timeout"),
        success_resp,
    ]

    pub = FacebookPublisher()
    result = pub.publish(_make_content(), page_id="p1", access_token="tok")

    assert result.status == "success"
    assert mock_post.call_count == 2


# ── distributor_agent ─────────────────────────────────────────────────────────

@patch("agent.services.auto_publish.requests.post")
def test_distributor_agent_filters_by_can_publish(mock_post):
    """Only Facebook/Poster content items are published."""
    success_resp = MagicMock()
    success_resp.json.return_value = {"id": "post_1"}
    success_resp.raise_for_status.return_value = None
    mock_post.return_value = success_resp

    contents = [
        _make_content(content_format="Poster", channels=["Facebook"]),  # should publish
        _make_content(content_format="Blog", channels=["LinkedIn"]),     # should skip
        _make_content(content_format="Article", channels=["Twitter"]),   # should skip
    ]
    request = _make_request(contents)

    result = distributor_agent(request)

    assert isinstance(result, DistributorOutput)
    assert len(result.results) == 1
    assert result.results[0].status == "success"
    assert mock_post.call_count == 1


def test_distributor_agent_accepts_custom_publisher():
    """Custom publisher is used instead of FacebookPublisher."""

    class AlwaysPublish(BasePublisher):
        def can_publish(self, item):
            return True

        def publish(self, item, page_id, access_token):
            return SinglePostResult(
                content_id=item.insight_id, post_id="custom_post", status="success", response={}
            )

    contents = [_make_content(), _make_content(content_format="Blog")]
    request = _make_request(contents)

    result = distributor_agent(request, publisher=AlwaysPublish())

    assert len(result.results) == 2
    assert all(r.status == "success" for r in result.results)
