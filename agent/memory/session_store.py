from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

from agent.memory.schemas import SessionData

"""
agent/memory/session_store.py
 
In-process session store with a clean Redis upgrade path.
 
For single-process dev/testing:   MemorySessionStore  (default)
For production multi-worker:       RedisSessionStore   (swap in via settings)
 
Usage:
    from agent.memory.session_store import get_session_store
    store = get_session_store()
    session = store.get("thread_abc")
    store.save("thread_abc", updated_session)
 
Why this exists separately from LangGraph's checkpointer:
    LangGraph checkpointer persists the full AgentState (messages, plan, etc).
    This store persists ONLY the user's support session facts — product model,
    attempted steps, etc — in a shape that's easy to inject into prompts.
    They work together: checkpointer handles graph state, this handles domain memory.
"""

logger = logging.getLogger(__name__)

# Abstract base

class BaseSessionStore(ABC):

    @abstractmethod
    def get(self, session_id: str) -> Optional[SessionData]:
        """Return existing session or None."""
        # continue

    @abstractmethod
    def save(self, session_id: str, data: SessionData) -> None:
        """Persist session data."""
        # continue

    @abstractmethod
    def delete(self, session_id: str) -> None:
        """Remove session (e.g. on explicit reset)."""

    def get_or_create(self, session_id: str) -> SessionData:
        """Return existing session or initialise a fresh one."""
        existing = self.get(session_id)
        if existing:
            return existing
        fresh = SessionData(session_id=session_id)
        self.save(session_id, fresh)
        return fresh



class MemorySessionStore(BaseSessionStore):
    """
    Dict-backed store. Fast, zero dependencies.
    Suitable for: CLI, single-worker dev, testing.
    Lost on process restart — acceptable for short support sessions.
 
    TTL: sessions expire after `ttl_seconds` of inactivity (default 2h).
    """
    def __init__(self, ttl_seconds: int = 7200, max_sessions: int = 1000):
        self._store: dict[str, dict] = {}       # session_id → {data, last_access}
        self._ttl = ttl_seconds
        self._max = max_sessions


    def get(self, session_id: str) -> Optional[SessionData]:
        entry = self._store.get(session_id)
        if not entry:
            return None
        
        # TTL check

        if time.time() - entry["last_access"] > self._ttl:
            del self._store[session_id]
            logger.debug(f"Session expried: {session_id}")
            return None
        
        entry["last_access"] = time.time()
        return SessionData(**entry["data"])

        
    def save(self, session_id: str, data: SessionData) -> None:
        # Evict oldest if at capacity
        if len(self._store) >= self._max and session_id not in self._store:
            oldest = min(self._store, key=lambda k: self._store[k]["last_access"])
            del self._store[oldest]
            logger.warning(f"Session store at capacity, evicted: {oldest}")
 
        self._store[session_id] = {
            "data": data.model_dump(),
            "last_access": time.time()
        }

    def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)


    @property
    def active_sessions(self) -> int:
        return len(self._store)
    


#================================================================================
# ─── Redis implementation (production THIS REDIS FOR PRODUCTION IMPLEMENTATION)
#================================================================================

class RedisSessionStore(BaseSessionStore):
    """
    Redis-backed store. Required for multi-worker production deployments.
 
    Install: pip install redis
    Config:  REDIS_URL=redis://localhost:6379/0  in .env
 
    Sessions stored as JSON strings with TTL.
    Key format: vivoassist:session:{session_id}
    """
 
    KEY_PREFIX = "vivoassist:session:"
 
    def __init__(self, redis_url: str, ttl_seconds: int = 7200):
        try:
            import redis
            self._r = redis.from_url(redis_url, decode_responses=True)
            self._ttl = ttl_seconds
            self._r.ping()
            logger.info(f"RedisSessionStore connected: {redis_url}")
        except ImportError:
            raise RuntimeError("redis package not installed. Run: pip install redis")
        except Exception as e:
            raise RuntimeError(f"Redis connection failed: {e}")
 
    def _key(self, session_id: str) -> str:
        return f"{self.KEY_PREFIX}{session_id}"
 
    def get(self, session_id: str) -> Optional[SessionData]:
        raw = self._r.get(self._key(session_id))
        if not raw:
            return None
        try:
            return SessionData(**json.loads(raw))
        except Exception as e:
            logger.error(f"Failed to deserialize session {session_id}: {e}")
            return None
 
    def save(self, session_id: str, data: SessionData) -> None:
        self._r.setex(
            self._key(session_id),
            self._ttl,
            json.dumps(data.model_dump())
        )
 
    def delete(self, session_id: str) -> None:
        self._r.delete(self._key(session_id))
 

 # Factory

_store_instance: Optional[BaseSessionStore] = None

def get_session_store() -> BaseSessionStore:
    """
    Returns the configured session store singleton.
    Reads REDIS_URL from settings if available, falls back to in-memory.
    Call once at startup; reuse the instance everywhere.
    """
    global _store_instance
    if _store_instance is not None:
        return _store_instance
    
    try:
        from config import settings
        redis_url =getattr(settings, "REDIS_URL", None)
        if redis_url:
            _store_instance = RedisSessionStore(redis_url)
            logger.info("Using RedisSessionStore")
            return _store_instance
    
    except Exception as e:
        logger.warning(f"Redis init failed, falling back to memory: {e}")

    _store_instance = MemorySessionStore()
    logger.info("Using MemorySessionStore")
    return _store_instance
