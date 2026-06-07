import logging
from typing import List, Optional

import requests
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import (
    FB_API_TIMEOUT_GET,
    FB_GRAPH_API_VERSION,
    FB_METRICS,
    RETRY_ATTEMPTS,
    RETRY_WAIT_MAX,
    RETRY_WAIT_MIN,
)

logger = logging.getLogger(__name__)


class PostPerformance(BaseModel):
    post_id: str
    reach: int = Field(default=0, description="Unique users who saw the post")
    impressions: int = Field(default=0, description="Total times the post was viewed")
    clicks: int = Field(default=0, description="Total link clicks")
    engagement: int = Field(default=0, description="Likes, comments, and shares")
    ctr: float = Field(default=0.0, description="Click-through rate (clicks/impressions)")


class AnalyticsReport(BaseModel):
    summary_report: List[PostPerformance]
    total_avg_ctr: float
    top_performing_post_id: Optional[str] = None


@retry(
    wait=wait_exponential(min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    retry=retry_if_exception_type(requests.RequestException),
    reraise=True,
)
def _fetch_post_insights(pid: str, access_token: str) -> list:
    url = f"https://graph.facebook.com/{FB_GRAPH_API_VERSION}/{pid}/insights"
    params = {"metric": FB_METRICS, "access_token": access_token}
    resp = requests.get(url, params=params, timeout=FB_API_TIMEOUT_GET)
    resp.raise_for_status()
    return resp.json().get("data", [])


def analytics_agent(post_ids: List[str], access_token: str) -> AnalyticsReport:
    report_data: List[PostPerformance] = []

    for pid in post_ids:
        try:
            data = _fetch_post_insights(pid, access_token)
            stats = {item["name"]: item["values"][0]["value"] for item in data}

            clicks_raw = stats.get("post_clicks_by_type", {})
            if isinstance(clicks_raw, dict):
                clicks = sum(clicks_raw.values())
            else:
                clicks = int(clicks_raw)

            impressions = stats.get("post_impressions", 0)
            ctr = (clicks / impressions) if impressions > 0 else 0.0

            report_data.append(PostPerformance(
                post_id=pid,
                reach=stats.get("post_impressions_unique", 0),
                impressions=impressions,
                clicks=clicks,
                engagement=stats.get("post_engagements", 0),
                ctr=round(ctr, 4),
            ))
        except Exception:
            logger.warning("Could not fetch analytics for post %s", pid, exc_info=True)

    if not report_data:
        return AnalyticsReport(summary_report=[], total_avg_ctr=0.0, top_performing_post_id=None)

    avg_ctr = sum(p.ctr for p in report_data) / len(report_data)
    top_post = max(report_data, key=lambda x: x.clicks).post_id

    logger.info("Analytics: %d posts analysed, avg CTR=%.4f, top post=%s", len(report_data), avg_ctr, top_post)
    return AnalyticsReport(
        summary_report=report_data,
        total_avg_ctr=round(avg_ctr, 4),
        top_performing_post_id=top_post,
    )
