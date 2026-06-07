"""Unit tests for auto_analysis_report.py."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from agent.services.auto_analysis_report import AnalyticsReport, PostPerformance, analytics_agent

def _mock_insight_response(impressions=1000, clicks=50, reach=800, engagement=200):
    return {
        "data": [
            {"name": "post_impressions_unique", "values": [{"value": reach}]},
            {"name": "post_impressions", "values": [{"value": impressions}]},
            {"name": "post_clicks_by_type", "values": [{"value": {"link click": clicks}}]},
            {"name": "post_engagements", "values": [{"value": engagement}]},
        ]
    }


@patch("agent.services.auto_analysis_report.requests.get")
def test_analytics_agent_happy_path(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_insight_response(impressions=1000, clicks=100)
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    report = analytics_agent(["post_1", "post_2"], "fake_token")

    assert isinstance(report, AnalyticsReport)
    assert len(report.summary_report) == 2
    assert report.total_avg_ctr == pytest.approx(0.1, abs=0.001)
    assert report.top_performing_post_id in ("post_1", "post_2")


@patch("agent.services.auto_analysis_report.requests.get")
def test_analytics_agent_empty_post_ids(mock_get):
    report = analytics_agent([], "fake_token")

    assert report.summary_report == []
    assert report.total_avg_ctr == 0.0
    assert report.top_performing_post_id is None
    mock_get.assert_not_called()


@patch("agent.services.auto_analysis_report.requests.get")
def test_analytics_agent_one_post_fails_others_included(mock_get):
    """Post that always fails is skipped; others are included."""
    success_resp = MagicMock()
    success_resp.json.return_value = _mock_insight_response(impressions=500, clicks=25)
    success_resp.raise_for_status.return_value = None

    mock_get.side_effect = [
        requests.RequestException("403 Forbidden"),  # post_1 fails all retries
        success_resp,                                  # post_2 succeeds
    ]

    report = analytics_agent(["post_1", "post_2"], "token")

    assert len(report.summary_report) == 1


@patch("agent.services.auto_analysis_report.requests.get")
def test_analytics_agent_all_fail_returns_empty(mock_get):
    mock_get.side_effect = requests.RequestException("network error")

    report = analytics_agent(["p1", "p2"], "token")

    assert report.summary_report == []
    assert report.total_avg_ctr == 0.0
    assert report.top_performing_post_id is None


@patch("agent.services.auto_analysis_report.requests.get")
def test_ctr_calculation_is_correct(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_insight_response(impressions=200, clicks=10)
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    report = analytics_agent(["post_1"], "token")

    assert report.summary_report[0].ctr == pytest.approx(0.05, abs=0.001)


@patch("agent.services.auto_analysis_report.requests.get")
def test_zero_impressions_does_not_raise(mock_get):
    """Zero impressions should yield ctr=0.0, not a ZeroDivisionError."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_insight_response(impressions=0, clicks=0)
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    report = analytics_agent(["post_1"], "token")

    assert report.summary_report[0].ctr == 0.0
