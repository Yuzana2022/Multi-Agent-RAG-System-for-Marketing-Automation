"""Centralized configuration constants for the marketing pipeline.

All values can be overridden via environment variables using the same name
in upper-case (e.g. ``LLM_MODEL=gpt-4o`` overrides :data:`DEFAULT_LLM_MODEL`).
"""

import os

# ── LLM ──────────────────────────────────────────────────────────────────────
DEFAULT_LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
DEFAULT_LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE_STRICT", "0.2"))
INSIGHTS_LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE_INSIGHTS", "0.7"))
CONTENT_LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE_CONTENT", "0.7"))
EMBED_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

# ── Vector store ──────────────────────────────────────────────────────────────
DEFAULT_DB_PATH: str = os.getenv("DEFAULT_DB_PATH", "./chroma_db")
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "openai_embedding")
CHROMA_SPACE: str = os.getenv("CHROMA_SPACE", "cosine")
RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "1500"))
RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
RAG_SIMILARITY_THRESHOLD: float = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.5"))
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "50"))

# ── Web content loader ────────────────────────────────────────────────────────
MIN_CHUNK_WORD_COUNT: int = int(os.getenv("MIN_CHUNK_WORD_COUNT", "10"))
WEB_LOADER_MAX_WORKERS: int = int(os.getenv("WEB_LOADER_MAX_WORKERS", "5"))

# ── SerpAPI search ────────────────────────────────────────────────────────────
SERPAPI_LOCATION: str = os.getenv("SEARCH_LOCATION", "Dublin, Ireland")
SERPAPI_GL: str = os.getenv("SEARCH_GL", "IE")
SERPAPI_HL: str = os.getenv("SEARCH_HL", "en")
SERPAPI_TBM: str = os.getenv("SEARCH_TBM", "nws")
SERPAPI_NUM_RESULTS: int = int(os.getenv("SEARCH_NUM_RESULTS", "10"))

# ── Facebook Graph API ────────────────────────────────────────────────────────
FB_GRAPH_API_VERSION: str = os.getenv("FACEBOOK_API_VERSION", "v17.0")
FB_API_TIMEOUT_POST: int = int(os.getenv("FACEBOOK_TIMEOUT_POST", "15"))
FB_API_TIMEOUT_GET: int = int(os.getenv("FACEBOOK_TIMEOUT_GET", "10"))
FB_METRICS: str = (
    "post_impressions_unique,post_impressions,"
    "post_clicks_by_type,post_engagements"
)

# ── Retry policy ──────────────────────────────────────────────────────────────
RETRY_ATTEMPTS: int = int(os.getenv("RETRY_ATTEMPTS", "3"))
RETRY_WAIT_MIN: int = int(os.getenv("RETRY_WAIT_MIN", "1"))
RETRY_WAIT_MAX: int = int(os.getenv("RETRY_WAIT_MAX", "10"))
