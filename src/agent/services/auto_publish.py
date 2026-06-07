import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import requests
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import (
    FB_API_TIMEOUT_POST,
    FB_GRAPH_API_VERSION,
    RETRY_ATTEMPTS,
    RETRY_WAIT_MAX,
    RETRY_WAIT_MIN,
)
from .content_generation import AllMarketingContents, MarketingContent

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────

class FacebookPostRequest(BaseModel):
    marketing_data: AllMarketingContents
    page_id: str = Field(..., description="FB Page ID")
    access_token: str = Field(..., description="FB Page Access Token")


class SinglePostResult(BaseModel):
    content_id: str
    post_id: str = ""
    status: str
    response: dict


class DistributorOutput(BaseModel):
    results: List[SinglePostResult]


# ── Publisher interface ───────────────────────────────────────────────────────

class BasePublisher(ABC):
    """Abstract base for content publishers.

    Implement :meth:`can_publish` and :meth:`publish` to add a new platform
    (e.g. Twitter, LinkedIn) without touching the distribution orchestrator.
    """

    @abstractmethod
    def can_publish(self, item: MarketingContent) -> bool:
        """Return True if this publisher should handle *item*."""

    @abstractmethod
    def publish(self, item: MarketingContent, page_id: str, access_token: str) -> SinglePostResult:
        """Publish *item* and return the outcome."""


class FacebookPublisher(BasePublisher):
    """Publishes marketing content to a Facebook page feed."""

    def can_publish(self, item: MarketingContent) -> bool:
        return (
            "Poster" in item.content_format
            or "Facebook" in str(item.distribution_channel)
        )

    @retry(
        wait=wait_exponential(min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,
    )
    def _post_to_api(self, url: str, payload: dict) -> dict:
        resp = requests.post(url, data=payload, timeout=FB_API_TIMEOUT_POST)
        resp.raise_for_status()
        return resp.json()

    def publish(self, item: MarketingContent, page_id: str, access_token: str) -> SinglePostResult:
        url = f"https://graph.facebook.com/{FB_GRAPH_API_VERSION}/{page_id}/feed"
        message = f"{item.headline}\n\n{item.body_text}\n\n{item.call_to_action}"
        payload = {"message": message, "access_token": access_token}

        try:
            api_res = self._post_to_api(url, payload)
            logger.info("Published FB post %s for insight %s", api_res.get("id", ""), item.insight_id)
            return SinglePostResult(
                content_id=item.insight_id,
                post_id=api_res.get("id", ""),
                status="success",
                response=api_res,
            )
        except Exception:
            logger.exception("Failed to publish insight %s to Facebook", item.insight_id)
            return SinglePostResult(
                content_id=item.insight_id,
                status="failed",
                response={"error": "publish failed after retries"},
            )


# ── Orchestrator ──────────────────────────────────────────────────────────────

def distributor_agent(
    request: FacebookPostRequest,
    publisher: Optional[BasePublisher] = None,
) -> DistributorOutput:
    """Distribute marketing content via *publisher*.

    Args:
        request: Contains marketing content + Facebook credentials.
        publisher: Optional custom :class:`BasePublisher`. Defaults to
            :class:`FacebookPublisher`. Inject a different publisher to
            target other platforms (Twitter, LinkedIn, etc.).
    """
    pub = publisher or FacebookPublisher()
    results = [
        pub.publish(item, request.page_id, request.access_token)
        for item in request.marketing_data.contents
        if pub.can_publish(item)
    ]
    logger.info(
        "Distribution complete: %d/%d items published",
        sum(1 for r in results if r.status == "success"),
        len(results),
    )
    return DistributorOutput(results=results)
