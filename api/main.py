"""
api/main.py

FastAPI application entrypoint.

Endpoints:
  POST /chat           — single-turn REST query
  WS   /ws/{session_id} — WebSocket streaming (tokens arrive live)
  GET  /health         — liveness probe
  GET  /collections    — list available PDF collections
  DELETE /session/{id} — clear session memory

Run:
  uvicorn api.main:app --reload --port 8000
"""

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.chat_router import router as chat_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("VivoAssist API starting...")

    # Pre-warm: initialise graph singleton so first request isn't slow
    try:
        from agent.graph import get_graph
        get_graph()
        logger.info("LangGraph graph pre-warmed")
    except Exception as e:
        logger.warning(f"Graph pre-warm failed (non-fatal): {e}")

    yield

    logger.info("VivoAssist API shutting down")


app = FastAPI(
    title="VivoAssist",
    description="Technical support assistant powered by LlamaIndex + LangGraph",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "VivoAssist"}


@app.get("/session/new")
async def new_session():
    """Generate a fresh session ID for the client."""
    return {"session_id": str(uuid.uuid4())}