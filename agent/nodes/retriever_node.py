"""
LangGraph Node 3: Retrieval
 
Input:  AgentState.effective_query, AgentState.analysis, AgentState.collection_name
Output: AgentState.raw_answer, AgentState.source_nodes, AgentState.retrieval_successful
 
This node is the ONLY place in the graph that touches LlamaIndex.
All retrieval logic stays in src/retriever.py — untouched.
This node just calls it and maps results back into AgentState.
 
Two retrieval paths:
  1. Short/expanded query  → retrieve_expanded() runs all variants, merges
  2. Normal query          → retrieve() runs single effective_query
"""
from __future__ import annotations

import logging
from typing import Any

from agent.state import AgentState
from agent.tools.rag_tool import RAGTool, MultiRAGTool

logger = logging.getLogger(__name__)

# Module-level cache: one RAGTool per collection_name
# Avoids re-loading the LlamaIndex index on every graph invocation

_rag_tools: dict[str, RAGTool] = {}
_multi_tool: MultiRAGTool | None = None

def _get_rag_tool(collection_name: str | None) -> RAGTool | MultiRAGTool:
    global _multi_tool
    if not collection_name:
        if _multi_tool is None:
            _multi_tool = MultiRAGTool()
        return _multi_tool
    
    if collection_name not in _rag_tools:
        _rag_tools[collection_name] = RAGTool(collection_name)

    return _rag_tools[collection_name]

def retriever_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node function.
    """
    collection_name = state.get("collection_name")
    effective_query = state.get("effective_query", "")
    analysis = state.get("analysis", {})

    if not effective_query:
        logger.error("retriever_node: effective_query is empty")
        return {
            "raw_answer": "",
            "source_nodes": [],
            "retrieval_successful": False,
        }
    
    tool = _get_rag_tool(collection_name)
    expanded_queries = analysis.get("expanded_queries", []) if analysis else []
    specificity = analysis.get("specificity", "medium") if analysis else "medium"

    # ---- Choose retrieval strategy
    if specificity == "short" and expanded_queries and isinstance(tool, RAGTool):
        logger.info(
            f"retriever_node: expanded retrieval | "
            f"{len(expanded_queries)} variants | collection={collection_name}"
        )
        result = tool.retrieve_expanded(expanded_queries)

    else:
        logger.info(
            f"retriever_node: single retrieval | "
            f"query='{effective_query[:60]}...' | collection={collection_name}"
        )
        result = tool.retrieve(effective_query)
 
    if not result.successful:
        logger.warning(f"retriever_node: retrieval failed: {result.error}")
 
    return {
        "raw_answer": result.answer,
        "source_nodes": result.source_nodes,
        "retrieval_successful": result.successful,
    }