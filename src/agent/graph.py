import logging
import operator
from typing import Annotated, Callable, List, Optional, Tuple, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.services.auto_analysis_report import AnalyticsReport, analytics_agent
from agent.services.auto_publish import DistributorOutput, FacebookPostRequest, distributor_agent
from agent.services.content_generation import AllMarketingContents, marketing_content_agent
from agent.services.insights_extract import AllStrategicInsights, insights_agent
from agent.services.local_doc_load import AllLocalDocResults, extract_text_from_pdf
from agent.services.rag_agent import RagResult, reduce_agent
from agent.services.search_agent import AllSearchResults, google_search
from agent.services.search_doc_load import AllSearchDocResults, text_loader

logger = logging.getLogger(__name__)


class MarketingState(TypedDict):
    """State object that flows through the multi-agent pipeline."""

    # Input parameters
    query: str
    local_pdf_path: Optional[str]
    facebook_page_id: Optional[str]
    facebook_access_token: Optional[str]
    db_path: str

    # Intermediate results
    search_results: Optional[AllSearchResults]
    web_documents: Optional[AllSearchDocResults]
    local_documents: Optional[AllLocalDocResults]
    rag_results: Optional[RagResult]
    strategic_insights: Optional[AllStrategicInsights]
    marketing_contents: Optional[AllMarketingContents]
    publish_results: Optional[DistributorOutput]
    analytics_report: Optional[AnalyticsReport]

    # Control flow
    skip_local_docs: bool
    skip_publishing: bool
    skip_analytics: bool
    require_human_approval: bool
    force_rebuild_db: bool  # If True, deletes and rebuilds the vector store on each run

    # Human-in-the-loop
    human_approval: Optional[str]   # "approved", "rejected", or None
    human_feedback: Optional[str]   # Optional feedback from human reviewer

    # Error tracking
    errors: Annotated[List[str], operator.add]


# ── Pipeline nodes ────────────────────────────────────────────────────────────

def search_node(state: MarketingState) -> dict:
    """Node 1: Use SerpAPI to find news/articles related to the query."""
    try:
        results = google_search(state["query"])
        logger.info("Search returned %d results", len(results.results))
        return {"search_results": results}
    except Exception as e:
        error_msg = f"Search node error: {e}"
        logger.error(error_msg)
        return {"errors": [error_msg]}  # errors is Annotated[List, add], will merge automatically

def web_loader_node(state: MarketingState) -> dict:
    """Node 2a: Load text content from web search result URLs."""
    try:
        if state.get("search_results") and state["search_results"].results:
            docs = text_loader(state["search_results"])
            logger.info("Loaded %d web documents", len(docs.results))
            return {"web_documents": docs}
        else:
            logger.warning("No search results to load — check search node")
            return {"web_documents": AllSearchDocResults()}
    except Exception as e:
        error_msg = f"Web loader error: {e}"
        logger.error(error_msg)
        return {"errors": [error_msg], "web_documents": AllSearchDocResults()}

def local_loader_node(state: MarketingState) -> dict:
    """Node 2b: Load content from a local PDF document."""
    if state.get("skip_local_docs", False) or not state.get("local_pdf_path"):
        return {"local_documents": AllLocalDocResults()}

    try:
        docs = extract_text_from_pdf(state["local_pdf_path"])
        logger.info("Loaded %d local documents from %s", len(docs.results), state["local_pdf_path"])
        return {"local_documents": docs}
    except Exception as e:
        error_msg = f"Local loader error: {e}"
        logger.error(error_msg)
        return {"errors": [error_msg], "local_documents": AllLocalDocResults()}


def rag_node(state: MarketingState) -> dict:
    """Node 3: Build vector store and retrieve relevant content."""
    try:
        web_docs = state.get("web_documents") or AllSearchDocResults()
        local_docs = state.get("local_documents") or AllLocalDocResults()
        db_path = state.get("db_path", "./chroma_db")

        rag_results = reduce_agent(
            web_docs,
            local_docs,
            db_path,
            query=state["query"],
            force_rebuild=state.get("force_rebuild_db", False),
        )
        logger.info("RAG retrieved %d chunks", len(rag_results.content))
        return {"rag_results": rag_results}
    except Exception as e:
        error_msg = f"RAG node error: {e}"
        logger.error(error_msg)
        return {"errors": [error_msg], "rag_results": RagResult(content=[])}


def insights_node(state: MarketingState) -> dict:
    """Node 4: Extract and synthesize strategic insights."""
    try:
        if state.get("rag_results") and state["rag_results"].content:
            insights = insights_agent(state["rag_results"])
            logger.info("Extracted %d strategic insights", len(insights.insights))
            return {"strategic_insights": insights}
        else:
            logger.warning("No RAG results available — skipping insight extraction")
            return {"strategic_insights": None}
    except Exception as e:
        error_msg = f"Insights node error: {e}"
        logger.error(error_msg)
        return {"errors": [error_msg]}


def content_generation_node(state: MarketingState) -> dict:
    """Node 5: Generate marketing content from insights."""
    try:
        if state.get("strategic_insights"):
            contents = marketing_content_agent(state["strategic_insights"])
            logger.info("Generated %d marketing content pieces", len(contents.contents))
            return {"marketing_contents": contents}
        else:
            logger.warning("No insights available — skipping content generation")
            return {"marketing_contents": None}
    except Exception as e:
        error_msg = f"Content generation error: {e}"
        logger.error(error_msg)
        return {"errors": [error_msg]}


def publishing_node(state: MarketingState) -> dict:
    """Node 6: Publish content to Facebook."""
    if state.get("skip_publishing", False):
        logger.info("Publishing skipped")
        return {}  

    try:
        if not state.get("marketing_contents"):
            logger.warning("No marketing content to publish")
            return {}

        if not state.get("facebook_page_id") or not state.get("facebook_access_token"):
            logger.warning("Facebook credentials not provided — skipping publishing")
            return {"skip_publishing": True}

        request = FacebookPostRequest(
            marketing_data=state["marketing_contents"],
            page_id=state["facebook_page_id"],
            access_token=state["facebook_access_token"],
        )
        results = distributor_agent(request)
        successful = sum(1 for r in results.results if r.status == "success")
        logger.info("Published %d/%d posts successfully", successful, len(results.results))
        return {"publish_results": results}
    except Exception as e:
        error_msg = f"Publishing error: {e}"
        logger.error(error_msg)
        return {"errors": [error_msg]}


def analytics_node(state: MarketingState) -> dict:
    """Node 7: Analyse published post performance."""
    if state.get("skip_analytics", False):
        logger.info("Analytics skipped")
        return {}

    try:
        if not state.get("publish_results"):
            logger.info("No published posts to analyse")
            return {}

        post_ids = [
            r.post_id for r in state["publish_results"].results
            if r.status == "success" and r.post_id
        ]

        if not post_ids:
            logger.info("No successful posts to analyse")
            return {}

        if not state.get("facebook_access_token"):
            logger.warning("Facebook access token not provided — skipping analytics")
            return {}

        report = analytics_agent(post_ids, state["facebook_access_token"])
        logger.info(
            "Analytics report: %d posts, avg CTR=%.2%%",
            len(report.summary_report),
            report.total_avg_ctr * 100,
        )
        if report.top_performing_post_id:
            logger.info("Top post: %s", report.top_performing_post_id)
        return {"analytics_report": report}
    except Exception as e:
        error_msg = f"Analytics error: {e}"
        logger.error(error_msg)
        return {"errors": [error_msg]}

# ── Human-in-the-loop ─────────────────────────────────────────────────────────

def _default_cli_callback(contents: AllMarketingContents) -> Tuple[str, str]:
    """Default approval callback that prompts via stdin.

    Not suitable for API or async deployments — pass a custom *approval_callback*
    to :func:`create_graph` or :func:`run_marketing_pipeline` instead.

    Returns:
        Tuple of (approval_status, feedback) where *approval_status* is
        ``"approved"`` or ``"rejected"``.
    """
    print("\nGenerated Marketing Content:")
    for idx, content in enumerate(contents.contents, 1):
        print(f"\n[Content #{idx}]")
        print(f"  Format: {content.content_format}")
        print(f"  Headline: {content.headline}")
        print(f"  Target Audience: {content.target_audience}")
        print(f"  Channels: {', '.join(content.distribution_channel)}")
        print(f"  Body: {content.body_text[:200]}...")
        print(f"  CTA: {content.call_to_action}")
        print("-" * 10)

    print("\nPlease review the content above.")
    print("  Type 'approve'/'yes' to publish, 'reject'/'no' to skip.")

    user_input = input("\nYour decision: ").strip().lower()

    if user_input in ("approve", "approved", "yes", "y") or "approve" in user_input or "yes" in user_input:
        approval = "approved"
    else:
        approval = "rejected"

    feedback = input("\nOptional feedback (press Enter to skip): ").strip()
    return approval, feedback


def _make_human_review_node(callback: Callable) -> Callable:
    """Return a human_review_node that delegates approval to *callback*.

    The callback signature is::

        callback(contents: AllMarketingContents) -> Tuple[str, str]

    returning ``(approval_status, feedback)``.
    """
    def human_review_node(state: MarketingState) -> dict:  # 返回类型改为 dict
        logger.info("HUMAN REVIEW REQUIRED")

        if not state.get("marketing_contents"):
            logger.warning("No marketing content available for review — rejecting")
            return {"human_approval": "rejected"}  # 只返回更新的字段

        try:
            approval, feedback = callback(state["marketing_contents"])
            result = {"human_approval": approval}
            if feedback:
                result["human_feedback"] = feedback
            logger.info("Content %s by human reviewer", approval)
            return result  # 只返回更新的字段
        except Exception as e:
            logger.error("Error in approval callback: %s — defaulting to rejection", e)
            return {"human_approval": "rejected"}  # 只返回更新的字段

    return human_review_node

def check_approval(state: MarketingState) -> str:
    """Conditional edge: route based on human approval."""
    return "publishing" if state.get("human_approval") == "approved" else "end"


# ── Graph factory ─────────────────────────────────────────────────────────────

def create_graph(
    require_human_approval: bool = False,
    approval_callback: Optional[Callable] = None,
) -> CompiledStateGraph:
    """Build and compile the multi-agent marketing intelligence graph.

    Args:
        require_human_approval: If True, inserts a human review step before publishing.
        approval_callback: Callable invoked for human review when
            *require_human_approval* is True. Must accept
            :class:`AllMarketingContents` and return ``Tuple[str, str]`` of
            ``(approval_status, feedback)``. Defaults to a CLI stdin prompt;
            supply a custom callback for API or async deployments.

    Returns:
        Compiled :class:`CompiledStateGraph` ready for invocation.
    """
    workflow = StateGraph(MarketingState)

    workflow.add_node("search", search_node)
    workflow.add_node("web_loader", web_loader_node)
    workflow.add_node("local_loader", local_loader_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("insights", insights_node)
    workflow.add_node("content_generation", content_generation_node)
    workflow.add_node("publishing", publishing_node)
    workflow.add_node("analytics", analytics_node)

    workflow.set_entry_point("search")
    workflow.add_edge("search", "web_loader")
    workflow.add_edge("search", "local_loader")
    workflow.add_edge("web_loader", "rag")
    workflow.add_edge("local_loader", "rag")
    workflow.add_edge("rag", "insights")
    workflow.add_edge("insights", "content_generation")

    if require_human_approval:
        review_node = _make_human_review_node(approval_callback or _default_cli_callback)
        workflow.add_node("human_review", review_node)
        workflow.add_edge("content_generation", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            check_approval,
            {"publishing": "publishing", "end": END},
        )
    else:
        workflow.add_edge("content_generation", "publishing")

    workflow.add_edge("publishing", "analytics")
    workflow.add_edge("analytics", END)

    return workflow.compile()


graph = create_graph(require_human_approval=False)


# ── Public entry point ────────────────────────────────────────────────────────

def run_marketing_pipeline(
    query: str,
    local_pdf_path: Optional[str] = None,
    facebook_page_id: Optional[str] = None,
    facebook_access_token: Optional[str] = None,
    db_path: str = "./chroma_db",
    skip_publishing: bool = False,
    skip_analytics: bool = False,
    require_human_approval: bool = False,
    force_rebuild_db: bool = False,
    approval_callback: Optional[Callable] = None,
) -> MarketingState:
    """Execute the complete marketing intelligence pipeline.

    Args:
        query: Search query for finding relevant news.
        local_pdf_path: Optional path to a local PDF for additional context.
        facebook_page_id: Facebook Page ID for publishing.
        facebook_access_token: Facebook access token.
        db_path: Path for the Chroma vector store.
        skip_publishing: Skip the publishing step.
        skip_analytics: Skip the analytics step.
        require_human_approval: If True, pauses for human review before publishing.
        force_rebuild_db: If True, deletes and rebuilds the vector store each run.
            Defaults to False so embeddings persist across runs.
        approval_callback: Custom callback for human review. Must accept
            :class:`AllMarketingContents` and return ``Tuple[str, str]`` of
            ``(approval_status, feedback)``. Defaults to CLI stdin prompt.

    Returns:
        Final :class:`MarketingState` containing all results and any errors.
    """
    execution_graph = create_graph(
        require_human_approval=require_human_approval,
        approval_callback=approval_callback,
    )

    initial_state: MarketingState = {
        "query": query,
        "local_pdf_path": local_pdf_path,
        "facebook_page_id": facebook_page_id,
        "facebook_access_token": facebook_access_token,
        "db_path": db_path,
        "search_results": None,
        "web_documents": None,
        "local_documents": None,
        "rag_results": None,
        "strategic_insights": None,
        "marketing_contents": None,
        "publish_results": None,
        "analytics_report": None,
        "skip_local_docs": local_pdf_path is None,
        "skip_publishing": skip_publishing,
        "skip_analytics": skip_analytics,
        "require_human_approval": require_human_approval,
        "force_rebuild_db": force_rebuild_db,
        "human_approval": None,
        "human_feedback": None,
        "errors": [],
    }

    logger.info("Starting Multi-Agent Marketing Pipeline (query=%r)", query)
    if require_human_approval:
        logger.info("Human-in-the-loop: ENABLED")

    final_state = execution_graph.invoke(initial_state)

    if final_state.get("errors"):
        logger.warning("Pipeline completed with %d error(s):", len(final_state["errors"]))
        for error in final_state["errors"]:
            logger.warning("  %s", error)
    else:
        logger.info("Pipeline completed successfully")

    return final_state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    result = run_marketing_pipeline(
        query="AI marketing trends 2025",
        # local_pdf_path="./document.pdf",  # optional local PDF
        require_human_approval=True,
        skip_publishing=False,
        skip_analytics=True,
        # facebook_page_id="your_page_id",
        # facebook_access_token="your_token",
    )

    if result.get("strategic_insights"):
        print("\nStrategic Insights:")
        for insight in result["strategic_insights"].insights:
            print(f"  {insight.key_insight_content[:100]}...")

    if result.get("marketing_contents"):
        print(f"\nGenerated {len(result['marketing_contents'].contents)} marketing pieces")
