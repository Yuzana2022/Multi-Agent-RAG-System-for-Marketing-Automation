import logging
import os
import shutil
from typing import List

from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from .config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_SPACE,
    EMBED_MODEL,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_SIMILARITY_THRESHOLD,
    RAG_TOP_K,
)
from .llm_model import _embed_model
from .local_doc_load import AllLocalDocResults
from .search_doc_load import AllSearchDocResults

logger = logging.getLogger(__name__)


class RagResult(BaseModel):
    content: List[str] = Field(..., description="The similar content list")


def map_agent(input_docs: AllSearchDocResults | AllLocalDocResults, db_path: str) -> Chroma:
    langchain_docs = [
        Document(page_content=res.content, metadata={"title": res.title})
        for res in input_docs.results
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE, chunk_overlap=RAG_CHUNK_OVERLAP
    )
    split_docs = text_splitter.split_documents(langchain_docs)

    embedding_func = _embed_model(model=EMBED_MODEL)

    return Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_func,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=db_path,
        collection_metadata={"hnsw:space": CHROMA_SPACE},
    )


def search_with_threshold(vectorstore: Chroma, query: str, threshold: float = RAG_SIMILARITY_THRESHOLD):
    results_with_scores = vectorstore.similarity_search_with_relevance_scores(query, k=RAG_TOP_K)
    return [doc for doc, score in results_with_scores if score >= threshold]


def reduce_agent(
    input_docs_1: AllSearchDocResults,
    input_docs_2: AllLocalDocResults,
    db_path: str,
    query: str,
    force_rebuild: bool = False,
) -> RagResult:
    """Build a vector store from web + local docs and retrieve relevant content.

    Args:
        input_docs_1: Web documents from the search loader.
        input_docs_2: Local PDF documents.
        db_path: Directory for the Chroma persistent store.
        query: The user search query used for similarity retrieval.
        force_rebuild: When True, deletes and re-creates the vector store.
            Defaults to False so embeddings persist across runs.
    """
    if not input_docs_1.results and not input_docs_2.results:
        logger.warning("No documents provided to RAG agent — returning empty result")
        return RagResult(content=[])

    if force_rebuild and os.path.exists(db_path):
        try:
            shutil.rmtree(db_path)
            logger.info("Deleted existing vector store at %s for forced rebuild", db_path)
        except PermissionError:
            logger.warning("Directory %s is in use; skipping deletion", db_path)

    if input_docs_1.results:
        map_agent(input_docs_1, db_path)
    if input_docs_2.results:
        map_agent(input_docs_2, db_path)

    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=_embed_model(model=EMBED_MODEL),
        collection_name=CHROMA_COLLECTION_NAME,
    )

    docs = search_with_threshold(vectorstore, query, threshold=RAG_SIMILARITY_THRESHOLD)
    logger.info("RAG retrieved %d chunks above threshold %.2f", len(docs), RAG_SIMILARITY_THRESHOLD)
    return RagResult(content=[d.page_content for d in docs])
