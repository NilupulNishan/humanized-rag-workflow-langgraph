"""
agent/tools/rag_tool.py
 
Wraps your existing SmartRetriever as a callable that LangGraph nodes can use.
 
This is NOT a LangChain @tool decorator — LangGraph nodes call it directly
as a plain function. The node handles orchestration; this handles retrieval.
 
Why not use LangChain's Tool wrapper?
    LangGraph nodes have full access to AgentState. Wrapping as a LangChain
    Tool would lose the source_nodes (citations) which are critical for this
    assistant. Direct function call keeps everything accessible.
 
Your retriever.py is 100% unchanged. This file is the only bridge.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Clean result contract between the tool and LangGraph nodes."""
    answer: str
    source_nodes: list
    collection_name: str
    successful: bool
    error: Optional[str] = None
    from_cache: bool = False

    @property
    def source_pages(self) -> list[int]:
        """Extract unique page numbers from source nodes for use in prompts."""
        pages = []
        for node in self.source_nodes:
            meta = getattr(node, 'metadata', {}) or {}
            page = meta.get('page_number') or meta.get('page') or meta.get('page_label')
            if page and page not in pages:
                pages.append(page)
        return sorted(pages)
    
    @property
    def formatted_sources(self) -> str:
        """Extract unique page numbers from source nodes for use in prompts."""
        pages = self.source_pages
        if not pages:
            return "Source: not available"
        return f"Sources: page {', '.join(str(p) for p in pages)}"


class RAGTool:
    """
    Lazy-loading wrapper around SmartRetriever.
 
    Lazy loading: the retriever is NOT initialised at import time.
    It's initialised on first call to retrieve(). This means graph.py
    imports cleanly even before any collections are loaded.
 
    Thread safety: SmartRetriever is stateless per-query (state is only
    the index, which is read-only after load). Safe for concurrent calls.
    """

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self._retriever = None

    def _get_retriever(self):
        """Initialise retriever on first use."""
        if self._retriever is None:
            from src.retriever import SmartRetriever 
            self._retriever = SmartRetriever(
                collection_name=self.collection_name,
                verbose=False,
                similarity_top_k=6,
            )
            logger.info(f"RAGTool initialised for collection: {self.collection_name}")
        return self._retriever
    
    def retrieve(self, query: str) -> RetrievalResult:
        """
        Run retrieval against the LlamaIndex collection.
        Uses SmartRetriever.query() (non-streaming) — streaming is handled
        at the API layer separately.
        """
        try:
            retriever = self._get_retriever()
            response = retriever.query(query)

            return RetrievalResult(
                answer=response.answer,
                source_nodes=response.source_nodes,
                collection_name=self.collection_name,
                successful=response.retrieval_successful,
                error=response.error_message,
                from_cache=response.from_cache,
            )
        except Exception as e:
            logger.error(f"RAGTool.retrieve failed: {e}")
            return RetrievalResult(
                answer="",
                source_nodes=[],
                collection_name=self.collection_name,
                successful=False,
                error=str(e),
            )
        
    def retrieve_expanded(self, queries: list[str]) -> RetrievalResult:
        """
        Run multiple query variants (from query expansion) and merge results.
        Used for short/vague queries like "wifi issue" → 4 expanded variants.
 
        Strategy: run all queries, deduplicate source nodes by node_id,
        use the answer from the highest-scoring retrieval.
        """
        if not queries:
            return RetrievalResult(
                answer="", source_nodes=[], collection_name=self.collection_name,
                successful=False, error="No queries provided"
            )
        results = [self.retrieve(q) for q in queries]
        successful = [r for r in results if r.successful and r.answer]

        if not successful:
            return results[0]  # Return first failure for error message
        
        # Merge: use best answer (longest successful), deduplicate nodes
        best = max(successful, key=lambda r: len(r.answer))
        seen_ids = set()
        merged_nodes = []
        for r in successful:
            for node in r.source_nodes:
                node_id = getattr(node, 'node_id', None) or id(node)
                if node_id not in seen_ids:
                    seen_ids.add(node_id)
                    merged_nodes.append(node)

        return RetrievalResult(
            answer=best.answer,
            source_nodes=merged_nodes,
            collection_name=self.collection_name,
            successful=True,
            from_cache=best.from_cache,
        )

# ─── Multi-collection tool ────────────────────────────────────────────────────
 
class MultiRAGTool:
    """
    Wraps MultiCollectionRetriever for sessions where collection is unknown.
    Falls back to best-answer selection across all collections.
    """
 
    def __init__(self):
        self._retriever = None
 
    def _get_retriever(self):
        if self._retriever is None:
            from src.retriever import MultiCollectionRetriever
            self._retriever = MultiCollectionRetriever()
        return self._retriever
 
    def retrieve(self, query: str) -> RetrievalResult:
        try:
            retriever = self._get_retriever()
            response = retriever.query_best(query)
            return RetrievalResult(
                answer=response.answer,
                source_nodes=response.source_nodes,
                collection_name=response.collection_name,
                successful=response.retrieval_successful,
                error=response.error_message,
            )
        except Exception as e:
            logger.error(f"MultiRAGTool.retrieve failed: {e}")
            return RetrievalResult(
                answer="", source_nodes=[], collection_name="unknown",
                successful=False, error=str(e)
            )