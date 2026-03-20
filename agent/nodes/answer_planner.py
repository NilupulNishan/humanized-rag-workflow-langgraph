"""
LangGraph Node 4: Answer Planner
 
Input:  AgentState.raw_answer, AgentState.source_nodes,
        AgentState.analysis, AgentState.session
Output: AgentState.plan
 
The planner sits between retrieval and rendering.
It converts raw retrieved text into a STRUCTURED PLAN — not prose.
The renderer turns the plan into prose. This separation means:
  - The plan is testable (assert plan["mode"] == "troubleshoot")
  - The renderer is swappable (different tones, languages)
  - Confidence gates are explicit, not hidden inside a prompt
 
Confidence thresholds (tunable):
  > 0.75  → answer fully
  0.4-0.75 → answer but flag uncertainty
  < 0.4   → clarify or escalate

NOTE: With the LangChain retriever, raw_answer is always "".
Context for the planner LLM is built from source_nodes (LangChain Documents)
instead. The retrieval_successful + source_nodes check is the gate.
"""

from __future__ import annotations
 
import json
import logging
from typing import Any
 
from langchain_openai import AzureChatOpenAI
 
from agent.state import AgentState, AnswerPlan
from agent.prompts.system_prompt import PLANNER_SYSTEM, PLANNER_USER
 
logger = logging.getLogger(__name__)

# Confidence thresholds
CONF_HIGH  = 0.75
CONF_LOW   = 0.40


def _get_llm() -> AzureChatOpenAI:
    from config import settings
    return AzureChatOpenAI(
        azure_deployment=settings.AZURE_GPT4O_MINI_DEPLOYMENT,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        temperature=0.1,
        max_tokens=600,
    )


def _build_context_from_docs(source_nodes: list, max_chars: int = 3000) -> str:
    """
    Build a text context string from LangChain Document objects.
    Replaces the old raw_answer (which LlamaIndex pre-built for us).
    Each chunk is prefixed with its page number for the planner LLM to cite.
    """
    parts = []
    total = 0
    for doc in source_nodes:
        page = doc.metadata.get("page", "?")
        chunk = f"[Page {page}]\n{doc.page_content}"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n\n".join(parts)


def answer_planner_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node function.
    """
    source_nodes         = state.get("source_nodes", [])
    analysis             = state.get("analysis", {})
    session              = state.get("session")
    retrieval_successful = state.get("retrieval_successful", False)

    # ── BUG FIX 1: Gate on retrieval_successful + source_nodes, not raw_answer.
    # raw_answer is always "" with the new LangChain retriever.
    if not retrieval_successful or not source_nodes:
        logger.warning("answer_planner: retrieval failed or empty - escalating")
        return {
            "plan": AnswerPlan(
                mode="escalate",
                confidence=0.0,
                likely_goal=analysis.get("inferred_topic", "unknown") if analysis else "unknown",
                steps=None,
                expected_outcomes=None,
                safety_notes=[],
                citations=[],
                first_clarifying_question=None,
                escalation_message=(
                    "I wasn't able to find information about this in the manual. "
                    "Please contact technical support directly for further assistance."
                ),
            )
        }

    # ── BUG FIX 2: Build context from Documents, not raw_answer.
    # LangChain Documents have .page_content and .metadata — no .text / .get_content().
    raw_context = _build_context_from_docs(source_nodes, max_chars=3000)

    # ── Build source pages list for prompt
    # BUG FIX 3: source_pages_str was only assigned inside the loop,
    # so it was undefined if source_nodes was empty. Moved outside loop.
    source_pages = []
    for doc in source_nodes:
        meta = doc.metadata or {}
        page = meta.get("page_number") or meta.get("page") or meta.get("page_label")
        if page:
            source_pages.append(str(page))
    source_pages_str = ", ".join(sorted(set(source_pages))) if source_pages else "not available"

    # ── Build session context
    session_context = ""
    if session:
        if hasattr(session, 'to_context_string'):
            session_context = session.to_context_string()
        elif isinstance(session, dict):
            session_context = str(session)

    # ── Build prompt
    user_prompt = PLANNER_USER.format(
        session_context = session_context,
        intent          = analysis.get("intent", "faq") if analysis else "faq",
        answer_mode     = analysis.get("answer_mode", "direct") if analysis else "direct",
        inferred_topic  = analysis.get("inferred_topic", "") if analysis else "",
        raw_answer      = raw_context,   # was: raw_answer[:3000]
        source_pages    = source_pages_str,
    )

    # ── Call LLM
    try:
        llm = _get_llm()
        response = llm.invoke([
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user",   "content": user_prompt},
        ])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
 
        data = json.loads(raw)

        # Build citations from source nodes + LLM output
        llm_citations  = data.get("citations", [])
        node_citations = _citations_from_nodes(source_nodes)
        citations      = llm_citations if llm_citations else node_citations
 
        plan = AnswerPlan(
            mode=data.get("mode", "direct"),
            confidence=float(data.get("confidence", 0.5)),
            likely_goal=data.get("likely_goal", ""),
            steps=data.get("steps"),
            expected_outcomes=data.get("expected_outcomes"),
            safety_notes=data.get("safety_notes", []),
            citations=citations,
            first_clarifying_question=data.get("first_clarifying_question"),
            escalation_message=data.get("escalation_message"),
        )

        # ── Confidence-based mode override
        if plan["confidence"] < CONF_LOW and plan["mode"] not in ("clarify", "escalate"):
            logger.info(
                f"answer_planner: confidence {plan['confidence']:.2f} below threshold "
                f"— overriding mode from '{plan['mode']}' to 'clarify'"
            )
            plan["mode"] = "clarify"
            if not plan.get("first_clarifying_question"):
                plan["first_clarifying_question"] = (
                    "Could you give me a bit more detail about what you're seeing? "
                    "That'll help me point you to the right section of the manual."
                )

        logger.info(
            f"answer_planner: mode={plan['mode']} "
            f"confidence={plan['confidence']:.2f} "
            f"steps={len(plan.get('steps') or [])}"
        )
        return {"plan": plan}
    
    except json.JSONDecodeError as e:
        logger.error(f"answer_planner: JSON parse failed: {e}")
        return {"plan": _fallback_plan(analysis, source_pages_str)}
 
    except Exception as e:
        logger.error(f"answer_planner: LLM call failed: {e}")
        return {"plan": _fallback_plan(analysis, source_pages_str)}


def _citations_from_nodes(source_nodes: list) -> list[dict]:
    """
    Build citations from LangChain Document metadata.
    Documents use doc.metadata dict directly (no getattr needed).
    """
    citations = []
    seen = set()

    for doc in source_nodes:
        meta = doc.metadata or {}
        page    = meta.get("page_number") or meta.get("page") or meta.get("page_label")
        section = meta.get("section") or meta.get("header") or ""
        if page and page not in seen:
            seen.add(page)
            citations.append({"page": page, "section": section})
    return citations


def _fallback_plan(analysis: dict, source_pages: str) -> AnswerPlan:
    """Graceful fallback when planner LLM call fails."""
    return AnswerPlan(
        mode="direct",
        confidence=0.5,
        likely_goal=analysis.get("inferred_topic", "") if analysis else "",
        steps=None,
        expected_outcomes=None,
        safety_notes=[],
        citations=[],
        first_clarifying_question=None,
        escalation_message=None,
    )