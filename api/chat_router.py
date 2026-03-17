"""
api/chat_router.py

REST and WebSocket chat endpoints.

REST  POST /chat         — full response, returns JSON
WS         /ws/{session_id} — streaming tokens via WebSocket

WebSocket protocol (JSON frames):
  Client → Server:  {"message": "wifi not working", "collection": "hp_manual"}
  Server → Client:  {"type": "token",    "data": "Let"}           ← token stream
  Server → Client:  {"type": "token",    "data": "'s check..."}
  Server → Client:  {"type": "sources",  "data": [{"page": 12}]}  ← after stream
  Server → Client:  {"type": "plan",     "data": {...}}            ← plan metadata
  Server → Client:  {"type": "done",     "data": null}             ← end signal
  Server → Client:  {"type": "error",    "data": "message"}        ← on failure
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── REST models ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    collection: Optional[str] = None


class SourceInfo(BaseModel):
    page: Optional[int | str] = None
    section: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    mode: str
    confidence: float
    sources: list[SourceInfo]
    needs_followup: bool


# ─── REST endpoint ────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Synchronous REST endpoint. Returns complete response as JSON.
    Use this for simple integrations or testing.
    Use WebSocket for production UI (streaming).
    """
    import asyncio
    from agent.graph import chat

    session_id = request.session_id or str(uuid.uuid4())

    try:
        # graph.chat() is sync — run in thread pool to not block event loop
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: chat(
                user_input=request.message,
                session_id=session_id,
                collection_name=request.collection,
            )
        )

        plan = result.get("plan", {}) or {}
        source_nodes = result.get("source_nodes", []) or []

        sources = []
        for node in source_nodes:
            meta = getattr(node, 'metadata', {}) or {}
            page = meta.get('page_number') or meta.get('page') or meta.get('page_label')
            section = meta.get('section') or meta.get('header') or ""
            if page:
                sources.append(SourceInfo(page=page, section=section))

        return ChatResponse(
            session_id=session_id,
            response=result.get("final_response", ""),
            mode=plan.get("mode", "direct"),
            confidence=float(plan.get("confidence", 0.5)),
            sources=sources,
            needs_followup=plan.get("mode") in ("clarify", "troubleshoot"),
        )

    except Exception as e:
        logger.error(f"chat_endpoint error: {e}")
        return ChatResponse(
            session_id=session_id,
            response="I encountered an error processing your request. Please try again.",
            mode="error",
            confidence=0.0,
            sources=[],
            needs_followup=False,
        )


# ─── WebSocket endpoint ───────────────────────────────────────────────────────

@router.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    Streaming WebSocket endpoint.

    Flow per message:
      1. Run query_understanding + memory_read + retriever + answer_planner (sync)
      2. Stream renderer tokens via WebSocket
      3. Run memory_write (sync)
      4. Send sources + plan metadata
      5. Send done signal
    """
    await websocket.accept()
    logger.info(f"WebSocket connected: {session_id}")

    try:
        while True:
            # Receive message
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
                user_input   = data.get("message", "").strip()
                collection   = data.get("collection")
            except json.JSONDecodeError:
                user_input = raw.strip()
                collection = None

            if not user_input:
                continue

            logger.info(f"WS [{session_id}]: '{user_input[:60]}'")

            try:
                await _handle_ws_message(
                    websocket=websocket,
                    user_input=user_input,
                    session_id=session_id,
                    collection_name=collection,
                )
            except Exception as e:
                logger.error(f"WS message handler error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "data": str(e)
                }))

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")


async def _handle_ws_message(
    websocket: WebSocket,
    user_input: str,
    session_id: str,
    collection_name: str | None,
):
    """
    Handles one WebSocket message:
    - Runs the graph pipeline up to response_renderer
    - Streams renderer tokens
    - Sends metadata frames after stream completes
    """
    import asyncio
    from agent.graph import get_graph
    from agent.state import AgentState
    from agent.nodes.response_renderer import response_renderer_stream

    graph = get_graph()

    initial_state: AgentState = {
        "user_input":      user_input,
        "session_id":      session_id,
        "collection_name": collection_name or "",
        "messages":        [{"role": "user", "content": user_input}],
    }
    config = {"configurable": {"thread_id": session_id}}

    # ── Run pipeline up to (but not including) response_renderer ──────────
    # We do this by running individual nodes manually for streaming control.
    # The full graph.invoke() would block until renderer completes.

    from agent.nodes.query_understanding import query_understanding_node
    from agent.nodes.memory_node import memory_read_node, memory_write_node
    from agent.nodes.retriever_node import retriever_node
    from agent.nodes.answer_planner import answer_planner_node

    state = dict(initial_state)

    # Run pipeline nodes synchronously in executor (they're CPU/IO bound)
    loop = asyncio.get_event_loop()

    state.update(await loop.run_in_executor(None, lambda: query_understanding_node(state)))
    state.update(await loop.run_in_executor(None, lambda: memory_read_node(state)))

    # Check if clarification needed (skip retriever)
    analysis = state.get("analysis", {})
    if analysis and analysis.get("needs_clarification"):
        from agent.nodes.answer_planner import _fallback_plan
        from agent.state import AnswerPlan
        question = analysis.get("clarification_question", "Could you give me more detail?")
        state["plan"] = AnswerPlan(
            mode="clarify", confidence=0.0,
            likely_goal=analysis.get("inferred_topic", ""),
            steps=None, expected_outcomes=None, safety_notes=[],
            citations=[], first_clarifying_question=question, escalation_message=None,
        )
        state["raw_answer"] = ""
        state["source_nodes"] = []
        state["retrieval_successful"] = False
    else:
        state.update(await loop.run_in_executor(None, lambda: retriever_node(state)))
        state.update(await loop.run_in_executor(None, lambda: answer_planner_node(state)))

    # ── Stream renderer tokens ─────────────────────────────────────────────
    full_response = ""

    def _generate_tokens():
        """Runs in executor — yields tokens from streaming renderer."""
        return list(response_renderer_stream(state))

    tokens = await loop.run_in_executor(None, _generate_tokens)

    for token in tokens:
        full_response += token
        await websocket.send_text(json.dumps({"type": "token", "data": token}))

    state["final_response"] = full_response
    state["response_ready"] = True

    # ── Save memory ────────────────────────────────────────────────────────
    await loop.run_in_executor(None, lambda: memory_write_node(state))

    # ── Send metadata frames ───────────────────────────────────────────────
    plan = state.get("plan", {}) or {}
    source_nodes = state.get("source_nodes", []) or []

    sources = []
    for node in source_nodes:
        meta = getattr(node, 'metadata', {}) or {}
        page = meta.get('page_number') or meta.get('page') or meta.get('page_label')
        section = meta.get('section') or meta.get('header') or ""
        if page:
            sources.append({"page": page, "section": section})

    if sources:
        await websocket.send_text(json.dumps({"type": "sources", "data": sources}))

    await websocket.send_text(json.dumps({
        "type": "plan",
        "data": {
            "mode": plan.get("mode", "direct"),
            "confidence": plan.get("confidence", 0.5),
            "likely_goal": plan.get("likely_goal", ""),
        }
    }))

    await websocket.send_text(json.dumps({"type": "done", "data": None}))


# ─── Collection management ────────────────────────────────────────────────────

@router.get("/collections")
async def list_collections():
    """List available PDF collections."""
    try:
        from src.storage_manager import StorageManager
        sm = StorageManager()
        return {"collections": sm.list_collections()}
    except Exception as e:
        return {"collections": [], "error": str(e)}


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear session memory for a given session."""
    try:
        from agent.memory.session_store import get_session_store
        store = get_session_store()
        store.delete(session_id)
        return {"status": "cleared", "session_id": session_id}
    except Exception as e:
        return {"status": "error", "detail": str(e)}