#!/usr/bin/env python
"""Run each agent independently to see their individual outputs.

This script allows you to test and debug each agent separately without running
the entire pipeline. Each agent's output is saved to a JSON file for inspection.

Usage:
    python run_agents_separately.py
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到 Python 路径
# 假设脚本在项目根目录下运行
project_root = Path(__file__).parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from dotenv import load_dotenv

# 加载环境变量
load_dotenv(project_root / ".env")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 导入各个 Agent
from agent.services.search_agent import google_search
from agent.services.search_doc_load import text_loader
from agent.services.local_doc_load import extract_text_from_pdf, AllLocalDocResults
from agent.services.rag_agent import reduce_agent
from agent.services.insights_extract import insights_agent
from agent.services.content_generation import marketing_content_agent
from agent.services.auto_publish import distributor_agent, FacebookPostRequest
from agent.services.auto_analysis_report import analytics_agent


# ============================================================================
# 辅助函数
# ============================================================================

def save_output(filename: str, data):
    """保存输出到 JSON 文件。"""
    output_dir = Path("./outputs")
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        if hasattr(data, "model_dump"):
            json.dump(data.model_dump(), f, indent=2, ensure_ascii=False)
        elif isinstance(data, list):
            json.dump([item.model_dump() if hasattr(item, "model_dump") else item for item in data], 
                     f, indent=2, ensure_ascii=False)
        else:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"  💾 Saved to: {filepath}")


def print_section(title: str):
    """打印分节标题。"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def get_user_confirmation(prompt: str, default: str = "n") -> bool:
    """获取用户确认。"""
    response = input(f"\n⚠️  {prompt} [y/N]: ").strip().lower()
    return response in ("y", "yes", "ye")


# ============================================================================
# Agent 1: 搜索 Agent
# ============================================================================

def run_agent_1_search(query: str):
    """Agent 1: Google Search - 产出：搜索结果列表"""
    print_section("AGENT 1: SEARCH AGENT")
    print(f"   Query: {query}")
    
    try:
        results = google_search(query)
        print(f"   ✅ Total results: {len(results.results)}")
        
        print("\n   📋 Top 5 results:")
        for i, r in enumerate(results.results[:5], 1):
            print(f"      {i}. {r.title}")
            print(f"         URL: {r.link}")
            print(f"         Snippet: {r.snippet[:100]}...")
        
        save_output("agent_1_search_results.json", results)
        return results
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        logger.exception("Agent 1 failed")
        return None


# ============================================================================
# Agent 2: 网页加载 Agent
# ============================================================================
def run_agent_2_web_loader(search_results):
    """Agent 2: Web Loader - 产出：网页文档内容"""
    print_section("AGENT 2: WEB LOADER AGENT")
    
    if not search_results or not search_results.results:
        print("   ⚠️  No search results available. Please run Agent 1 first.")
        return None
    
    try:
        docs = text_loader(search_results)
        print(f"   ✅ Loaded {len(docs.results)} documents")
        
        print("\n   📄 Sample documents:")
        for i, doc in enumerate(docs.results[:3], 1):
            print(f"      {i}. {doc.title}")
            print(f"         Content preview: {doc.content[:150]}...")
        
        save_output("agent_2_web_documents.json", docs)
        return docs
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        logger.exception("Agent 2 failed")
        return None


# ============================================================================
# Agent 3: 本地 PDF 加载 Agent
# ============================================================================

def run_agent_3_local_loader(pdf_path: Optional[str] = None):
    """Agent 3: Local Loader - 产出：本地PDF提取内容"""
    print_section("AGENT 3: LOCAL LOADER AGENT")
    
    if not pdf_path or not os.path.exists(pdf_path):
        print(f"   ⚠️  PDF not found at: {pdf_path}")
        print("   📝 Returning empty result.")
        return AllLocalDocResults()
    
    try:
        docs = extract_text_from_pdf(pdf_path)
        print(f"   ✅ Loaded {len(docs.results)} pages/sections from {pdf_path}")
        
        print("\n   📄 Extracted content:")
        for i, doc in enumerate(docs.results[:3], 1):
            print(f"      {i}. {doc.title}")
            print(f"         Content preview: {doc.content[:150]}...")
        
        save_output("agent_3_local_documents.json", docs)
        return docs
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        logger.exception("Agent 3 failed")
        return AllLocalDocResults()


# ============================================================================
# Agent 4: RAG Agent
# ============================================================================

def run_agent_4_rag(web_docs, local_docs, query: str, force_rebuild: bool = True):
    """Agent 4: RAG Agent - 产出：向量检索的相关内容块"""
    print_section("AGENT 4: RAG AGENT")
    print(f"   Query: {query}")
    print(f"   Force rebuild DB: {force_rebuild}")
    
    web_count = len(web_docs.results) if web_docs else 0
    local_count = len(local_docs.results) if local_docs else 0
    print(f"   Web documents: {web_count}, Local documents: {local_count}")
    
    try:
        rag_results = reduce_agent(
            web_docs or AllLocalDocResults(),
            local_docs or AllLocalDocResults(),
            "./chroma_db",
            query=query,
            force_rebuild=force_rebuild,
        )
        print(f"   ✅ Retrieved {len(rag_results.content)} relevant chunks")
        
        print("\n   🔍 Retrieved chunks:")
        for i, chunk in enumerate(rag_results.content[:5], 1):
            print(f"      {i}. {chunk[:150]}...")
        
        save_output("agent_4_rag_results.json", rag_results)
        return rag_results
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        logger.exception("Agent 4 failed")
        return None


# ============================================================================
# Agent 5: 洞察提取 Agent
# ============================================================================

def run_agent_5_insights(rag_results):
    """Agent 5: Insights Agent - 产出：战略洞察"""
    print_section("AGENT 5: INSIGHTS AGENT")
    
    if not rag_results or not rag_results.content:
        print("   ⚠️  No RAG results available. Please run Agent 4 first.")
        return None
    
    try:
        insights = insights_agent(rag_results)
        print(f"   ✅ Generated {len(insights.insights)} strategic insights")
        
        print("\n   💡 Strategic Insights:")
        for i, insight in enumerate(insights.insights, 1):
            print(f"      {i}. {insight.key_insight_content[:150]}")
            if insight.strategic_relevance:
                print(f"         Supporting: {insight.strategic_relevance[:100]}...")
        
        save_output("agent_5_insights.json", insights)
        return insights
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        logger.exception("Agent 5 failed")
        return None


# ============================================================================
# Agent 6: 内容生成 Agent
# ============================================================================

def run_agent_6_content_generation(insights):
    """Agent 6: Content Generation - 产出：营销内容"""
    print_section("AGENT 6: CONTENT GENERATION AGENT")
    
    if not insights or not insights.insights:
        print("   ⚠️  No insights available. Please run Agent 5 first.")
        return None
    
    try:
        contents = marketing_content_agent(insights)
        print(f"   ✅ Generated {len(contents.contents)} marketing pieces")
        
        print("\n   📝 Marketing Content:")
        for i, content in enumerate(contents.contents, 1):
            print(f"\n      {'='*50}")
            print(f"      Content #{i}")
            print(f"      Headline: {content.headline}")
            print(f"      Format: {content.content_format}")
            print(f"      Audience: {content.target_audience}")
            print(f"      Channels: {', '.join(content.distribution_channel)}")
            print(f"      Body: {content.body_text[:200]}...")
            print(f"      CTA: {content.call_to_action}")
        
        save_output("agent_6_marketing_content.json", contents)
        return contents
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        logger.exception("Agent 6 failed")
        return None


# ============================================================================
# Agent 7: 发布 Agent
# ============================================================================

def run_agent_7_publishing(contents, page_id: str, access_token: str, dry_run: bool = True):
    """Agent 7: Publishing Agent - 产出：发布结果"""
    print_section("AGENT 7: PUBLISHING AGENT")
    
    if not contents or not contents.contents:
        print("   ⚠️  No marketing content available. Please run Agent 6 first.")
        return None
    
    if not page_id or not access_token:
        print("   ⚠️  Facebook credentials not provided.")
        print("   Please set FACEBOOK_PAGE_ID and FACEBOOK_ACCESS_TOKEN in .env file.")
        return None
    
    if dry_run:
        print("   🧪 DRY RUN MODE - No actual publishing will occur")
        print(f"   Would publish {len(contents.contents)} posts to Facebook")
        for content in contents.contents:
            print(f"      📤 Would publish: {content.headline}")
        return None
    
    print("   🔴 LIVE MODE - This will ACTUALLY publish to Facebook!")
    if not get_user_confirmation("Are you sure you want to publish?", default="n"):
        print("   ❌ Publishing cancelled.")
        return None
    
    try:
        request = FacebookPostRequest(
            marketing_data=contents,
            page_id=page_id,
            access_token=access_token,
        )
        results = distributor_agent(request)
        
        print(f"\n   📊 Publishing Results:")
        for res in results.results:
            status = "✅ SUCCESS" if res.status == "success" else "❌ FAILED"
            print(f"      {status}: {res.content_id} -> Post ID: {res.post_id}")
        
        save_output("agent_7_publish_results.json", results)
        return results
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        logger.exception("Agent 7 failed")
        return None


# ============================================================================
# Agent 8: 分析 Agent
# ============================================================================

def run_agent_8_analytics(post_ids: List[str], access_token: str):
    """Agent 8: Analytics Agent - 产出：数据分析报告"""
    print_section("AGENT 8: ANALYTICS AGENT")
    
    if not post_ids:
        print("   ⚠️  No post IDs provided.")
        return None
    
    if not access_token:
        print("   ⚠️  Facebook access token not provided.")
        return None
    
    print(f"   Analysing {len(post_ids)} posts: {post_ids}")
    
    try:
        report = analytics_agent(post_ids, access_token)
        print(f"   ✅ Analysed {len(report.summary_report)} posts")
        print(f"   📊 Average CTR: {report.total_avg_ctr * 100:.2f}%")
        print(f"   🏆 Top performing post: {report.top_performing_post_id}")
        
        print("\n   📈 Post Performance:")
        for post in report.summary_report:
            print(f"\n      Post: {post.post_id}")
            print(f"         Impressions: {post.impressions:,}")
            print(f"         Clicks: {post.clicks:,}")
            print(f"         Reach: {post.reach:,}")
            print(f"         Engagement: {post.engagement:,}")
            print(f"         CTR: {post.ctr * 100:.2f}%")
        
        save_output("agent_8_analytics_report.json", report)
        return report
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        logger.exception("Agent 8 failed")
        return None


# ============================================================================
# 主函数：顺序运行所有 Agent
# ============================================================================

def main():
    """主函数：顺序运行所有 Agent 并展示产出。"""
    print("\n" + "=" * 70)
    print("  🚀 MULTI-AGENT MARKETING SYSTEM - INDIVIDUAL AGENT RUNNER")
    print("=" * 70)
    
    # 配置参数
    QUERY = os.getenv("SEARCH_QUERY", "AI marketing trends 2025 for small businesses")
    PDF_PATH = os.getenv("LOCAL_PDF_PATH", "")
    FACEBOOK_PAGE_ID = os.getenv("FACEBOOK_PAGE_ID", "")
    FACEBOOK_ACCESS_TOKEN = os.getenv("FACEBOOK_ACCESS_TOKEN", "")
    DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
    
    print(f"\n📋 Configuration:")
    print(f"   Query: {QUERY}")
    print(f"   PDF Path: {PDF_PATH or '(not provided)'}")
    print(f"   Facebook Page ID: {FACEBOOK_PAGE_ID[:10] + '...' if FACEBOOK_PAGE_ID else '(not provided)'}")
    print(f"   Dry Run Mode: {DRY_RUN}")
    
    choice = input("\nYour choice (1 and 2, default=1): ").strip() or "1"
    
    # 存储中间结果
    search_results = None
    web_docs = None
    local_docs = None
    rag_results = None
    insights = None
    contents = None
    publish_results = None
    
    if choice=="1":
        search_results = run_agent_1_search(QUERY)
        print('########################Agent 1 Done########################')
        web_docs = run_agent_2_web_loader(search_results)
        print('########################Agent 2 Done########################')
        local_docs = run_agent_3_local_loader(PDF_PATH if PDF_PATH else None)
        print('########################Agent 3 Done########################')
        rag_results = run_agent_4_rag(web_docs, local_docs, QUERY)
        print('########################Agent 4 Done########################')
        insights = run_agent_5_insights(rag_results)
        print('########################Agent 5 Done########################')
        contents = run_agent_6_content_generation(insights)
        print('########################Agent 6 Done########################')
    
    # Agent 7
    if choice=="2":
        if DRY_RUN:
            run_agent_7_publishing(contents, FACEBOOK_PAGE_ID, FACEBOOK_ACCESS_TOKEN, dry_run=True)
        else:
            publish_results = run_agent_7_publishing(contents, FACEBOOK_PAGE_ID, FACEBOOK_ACCESS_TOKEN, dry_run=False)
        print('########################Agent 7 Done########################')


        if publish_results and not DRY_RUN:
            post_ids = [r.post_id for r in publish_results.results if r.status == "success"]
            if post_ids:
                run_agent_8_analytics(post_ids, FACEBOOK_ACCESS_TOKEN)
        print('########################Agent 8 Done########################')


if __name__ == "__main__":
    main()