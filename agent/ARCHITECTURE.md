# VivoAssist — Architecture Reference

## System Overview

```
USER MESSAGE
     │
     ▼
┌────────────────────────────────────────────────────────────────────┐
│                     LANGGRAPH AGENT LAYER                          │
│                                                                    │
│  ┌──────────────────────┐                                          │
│  │  1. query_understanding │  Intent + expansion (cheap LLM call)  │
│  └──────────┬───────────┘                                          │
│             │                                                      │
│   ┌─────────▼──────────┐   needs_clarification?                    │
│   │  2. memory_read    │──────────────────────────┐                │
│   └─────────┬──────────┘                          │                │
│             │ NO                              YES │                │
│   ┌─────────▼──────────┐              ┌────────────▼──────────┐    │
│   │  3. retriever_node │              │  skip_retrieval node  │    │
│   └─────────┬──────────┘              └────────────┬──────────┘    │
│             │                                      │               │
│   ┌─────────▼──────────┐                           │               │
│   │  4. answer_planner  │◄─────────────────────────┘               │
│   └─────────┬──────────┘                                           │
│             │                                                      │
│   ┌─────────▼──────────┐                                           │
│   │  5. response_renderer│  Mode-aware humanized prose             │
│   └─────────┬──────────┘                                           │
│             │                                                      │
│   ┌─────────▼──────────┐                                           │
│   │  6. memory_write    │  Extract facts, update session           │
│   └─────────────────────┘                                          │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
     │
     ▼  (only node 3 crosses this boundary)
┌────────────────────────────────────────────────────────────────────┐
│                    LLAMAINDEX LAYER (unchanged)                    │
│                                                                    │
│   RAGTool.retrieve() / retrieve_expanded()                         │
│        │                                                           │
│        ▼                                                           │
│   SmartRetriever.query()                                           │
│        │                                                           │
│        ├── EmbeddingCache (Azure OpenAI embed, cached)             │
│        ├── AutoMergingRetriever (hierarchical chunks)              │
│        ├── ChromaDB (vector store)                                 │
│        └── Docstore (node storage)                                 │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## File Map

```
pdf-embeddings-system/
│
├── config/settings.py              ← unchanged
│
├── src/                            ← LLAMAINDEX LAYER — ALL UNCHANGED
│   ├── retriever.py                ← SmartRetriever, EmbeddingCache, StreamResult
│   ├── embeddings.py
│   ├── storage_manager.py
│   ├── chunker.py
│   ├── pdf_loader.py
│   └── source_formatter.py
│
├── agent/                          ← NEW: LANGGRAPH LAYER
│   ├── __init__.py
│   ├── graph.py                    ← StateGraph assembly, build_graph(), chat()
│   ├── state.py                    ← AgentState TypedDict, QueryAnalysis, AnswerPlan
│   │
│   ├── nodes/
│   │   ├── query_understanding.py  ← Node 1: intent + query expansion
│   │   ├── memory_node.py          ← Node 2+6: read/write session
│   │   ├── retriever_node.py       ← Node 3: calls LlamaIndex via RAGTool
│   │   ├── answer_planner.py       ← Node 4: structured JSON plan
│   │   └── response_renderer.py   ← Node 5: humanized prose + streaming variant
│   │
│   ├── tools/
│   │   └── rag_tool.py             ← Wraps SmartRetriever, retrieve_expanded()
│   │
│   ├── memory/
│   │   ├── session_store.py        ← MemorySessionStore + RedisSessionStore
│   │   └── schemas.py              ← SessionData Pydantic model
│   │
│   └── prompts/
│       └── system_prompt.py        ← All LangGraph prompts (understanding/planner/renderer)
│
├── api/                            ← NEW: FastAPI
│   ├── main.py                     ← App entrypoint, lifespan, CORS
│   └── chat_router.py             ← POST /chat, WS /ws/{session_id}, /collections
│
├── scripts/
│   └── query.py                   ← Can call agent.graph.chat() directly
│
└── data/
    ├── pdfs/
    ├── chroma_db/
    └── docstore/
```

## Response Modes

| Mode          | Trigger                              | Response style                              |
|---------------|--------------------------------------|---------------------------------------------|
| `direct`      | Clear factual question, conf > 0.75  | 1-3 sentences, leads with answer            |
| `step_by_step`| How-to question                      | Numbered steps with expected outcomes       |
| `troubleshoot`| Vague problem statement              | "Let's check", First/Next/If, escalation    |
| `clarify`     | Ambiguous OR conf < 0.4              | One focused question, no guessing           |
| `escalate`    | No relevant content found            | Honest, warm handoff to support channel     |

## Session Memory Flow

```
Turn 1: "wifi not working"
  → session: {issue: "wifi not connecting", stage: "initial"}

Turn 2: "I already tried restarting"
  → memory_write extracts: attempted_steps: ["restarting"]
  → next query automatically: "wifi not connecting [already tried: restarting]"
  → response skips restart step entirely

Turn 3: "still not working"
  → stage advances to "resolving"
  → response continues from where we left off
```

## How to Run

```bash
# Install new dependencies
pip install -r requirements_agent.txt

# Start API
uvicorn api.main:app --reload --port 8000

# CLI test (quick)
python -c "
from agent.graph import chat
result = chat('wifi not working', session_id='test-1', collection_name='your_collection')
print(result['final_response'])
"
```

## Key Design Decisions

1. **LlamaIndex untouched** — `src/retriever.py` has zero changes. The agent layer
   calls it through `RAGTool`, which is the only bridge between the two layers.

2. **PromptManager preserved** — LlamaIndex's RAG synthesis still uses your existing
   `PromptManager` for document-grounded answers. The new prompts in `agent/prompts/`
   are only for the agent intelligence layer (planning, rendering).

3. **Streaming at API layer** — `response_renderer_stream()` is a generator called
   by the WebSocket handler, not by LangGraph. This gives streaming without fighting
   LangGraph's node execution model.

4. **Confidence gates are explicit** — `answer_planner.py` has `CONF_HIGH = 0.75` and
   `CONF_LOW = 0.40` as module-level constants. Easy to tune.

5. **Session store is swappable** — `get_session_store()` factory returns `MemorySessionStore`
   by default. Add `REDIS_URL` to `.env` to upgrade to Redis with zero code changes.
