import logging
import os

import dotenv

logger = logging.getLogger(__name__)

dotenv.load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_BASE_URL = os.getenv("TAVILY_BASE_URL")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set — LLM calls will fail")
if not SERPAPI_API_KEY:
    logger.warning("SERPAPI_API_KEY is not set — search will fail")
