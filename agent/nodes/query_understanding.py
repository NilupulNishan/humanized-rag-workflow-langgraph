"""
agent/nodes/query_understanding.py

LangGraph Node 1: Query Understanding

Input:  AgentState.user_input, AgentState.session, AgentState.messages
Output: AgentState.analysis, AgentState.effective_query

What it does:
  1. Reads user message + session context + conversation history
  2. Classifies intent — including "general" for non-manual questions
  3. If general: sets route flag → graph skips retriever entirely
  4. If short/vague manual query: expands into search variants
  5. If needs clarification: flags it
  6. Builds effective_query for the retriever

Intent routing summary:
  "general"       → direct_answer_node  (LLM only, no manual)
  "clarify"       → skip_retrieval_node (ask user for more info)
  everything else → memory_read → retriever → planner → renderer
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_openai import AzureChatOpenAI

from agent.state import AgentState, QueryAnalysis
from agent.prompts.system_prompt import (
    QUERY_UNDERSTANDING_SYSTEM,
    QUERY_UNDERSTANDING_USER,
)

logger = logging.getLogger(__name__)


def _get_llm() -> AzureChatOpenAI:
    from config import settings
    return AzureChatOpenAI(
        azure_deployment=settings.AZURE_GPT4O_MINI_DEPLOYMENT,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        temperature=0,
        max_tokens=400,
    )


def query_understanding_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node function.
    Returns dict of keys to update in AgentState.
    """
    user_input = state.get("user_input", "").strip()
    session    = state.get("session", {})

    if not user_input:
        logger.warning("query_understanding: empty user_input")
        return {
            "analysis": QueryAnalysis(
                intent="faq",
                specificity="short",
                answer_mode="direct",
                expanded_queries=[],
                needs_clarification=False,
                clarification_question=None,
                inferred_topic="unknown",
            ),
            "effective_query": user_input,
        }

    # ── Hard-coded social bypass — never send these to the LLM classifier ──
    # Short social inputs are misclassified ~30% of the time by smaller models.
    # Catch them here before any LLM call — zero latency, zero cost.
    _SOCIAL_EXACT = {
        "hi", "hello", "hey", "hiya", "howdy",
        "thanks", "thank you", "thank you!", "thanks!", "thx",
        "ok", "okay", "ok!", "okay!", "got it", "got it!",
        "bye", "goodbye", "see you", "see ya",
        "yes", "no", "yep", "nope", "sure", "alright",
        "good morning", "good afternoon", "good evening",
        "nice", "great", "awesome", "perfect", "cool",
    }
    if user_input.lower().rstrip(".,!? ") in _SOCIAL_EXACT:
        logger.info(f"query_understanding: social bypass → general: '{user_input}'")
        return {
            "analysis": QueryAnalysis(
                intent="general",
                specificity="short",
                answer_mode="direct",
                expanded_queries=[],
                needs_clarification=False,
                clarification_question=None,
                inferred_topic="social/conversational",
            ),
            "effective_query": "",
        }

    # ── Session context string ─────────────────────────────────────────────
    session_context = ""
    if hasattr(session, 'to_context_string'):
        session_context = session.to_context_string()
    elif isinstance(session, dict) and session:
        parts = []
        if session.get("issue_summary"):
            parts.append(f"Issue: {session['issue_summary']}")
        if session.get("attempted_steps"):
            parts.append(f"Already tried: {', '.join(session['attempted_steps'])}")
        if parts:
            session_context = "[Session: " + " | ".join(parts) + "]"

    # ── Build LLM message list with conversation history ───────────────────
    messages_history = state.get("messages", [])
    prior_turns = messages_history[:-1]   # exclude current user turn

    user_prompt = QUERY_UNDERSTANDING_USER.format(
        session_context=session_context,
        user_input=user_input,
    )

    llm_messages = [{"role": "system", "content": QUERY_UNDERSTANDING_SYSTEM}]
    for m in prior_turns[-6:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            llm_messages.append({"role": m["role"], "content": m["content"]})
    llm_messages.append({"role": "user", "content": user_prompt})

    # ── Call LLM ──────────────────────────────────────────────────────────
    try:
        llm = _get_llm()
        response = llm.invoke(llm_messages)

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)

        analysis = QueryAnalysis(
            intent=data.get("intent", "faq"),
            specificity=data.get("specificity", "medium"),
            answer_mode=data.get("answer_mode", "direct"),
            expanded_queries=data.get("expanded_queries", []),
            needs_clarification=data.get("needs_clarification", False),
            clarification_question=data.get("clarification_question"),
            inferred_topic=data.get("inferred_topic", user_input),
        )

        # ── Build effective_query ──────────────────────────────────────────
        # General questions skip retrieval entirely — no effective_query needed.
        if analysis["intent"] == "general":
            effective_query = ""
        elif analysis["specificity"] == "short" and analysis["expanded_queries"]:
            effective_query = user_input
        elif session_context:
            effective_query = f"{session_context}\n{user_input}"
        else:
            effective_query = user_input

        logger.info(
            f"query_understanding: intent={analysis['intent']} "
            f"mode={analysis['answer_mode']} "
            f"specificity={analysis['specificity']} "
            f"general={analysis['intent'] == 'general'}"
        )

        return {
            "analysis":       analysis,
            "effective_query": effective_query,
        }

    except json.JSONDecodeError as e:
        logger.error(f"query_understanding: JSON parse failed: {e}")
        return _fallback(user_input)

    except Exception as e:
        logger.error(f"query_understanding: LLM call failed: {e}")
        return _fallback(user_input)


def _fallback(user_input: str) -> dict[str, Any]:
    """Graceful degradation — treat as plain FAQ, go through normal pipeline."""
    return {
        "analysis": QueryAnalysis(
            intent="faq",
            specificity="medium",
            answer_mode="direct",
            expanded_queries=[],
            needs_clarification=False,
            clarification_question=None,
            inferred_topic=user_input,
        ),
        "effective_query": user_input,
    }