# TASK-080: Build FastAPI backend API layer (api.py)

**Sprint:** 11 — React Frontend
**Priority:** 🔴 Critical
**Status:** Queued
**Dependencies:** None

## Description
Create `/Users/rajesh/athena/api.py` — a FastAPI server that exposes the Athena multi-agent pipeline to the React frontend via HTTP. Runs alongside the existing CLI. Reuses the same agent/memory wiring from `cli.py`.

## Endpoints

### POST /analyze
Request: `{ "symbol": "TSLA", "verbose": true }`
Response: Server-Sent Events (SSE) stream. Each event is a JSON object:
```json
{"type": "agent_step", "agent": "market_analyst", "step": "think", "content": "...", "elapsed_ms": 120}
{"type": "agent_step", "agent": "risk_manager", "step": "act", "content": "...", "elapsed_ms": 340}
{"type": "result", "data": { ...full coordinator output... }}
{"type": "memory", "ops": {"add": 3, "retrieve": 2}}
{"type": "done"}
```

### GET /memory
Query params: `limit=20`
Response: `{ "episodes": [{ "id": "...", "content": "...", "timestamp": "...", "source": "..." }] }`

### GET /stats
Response: `{ "memory": { "add": N, "retrieve": N, ... }, "learning": { "agent_id": { "inner_steps": N, "outer_updates": N } }, "query_count": N }`

### GET /health
Response: `{ "status": "ok", "neo4j": true/false }`

## Acceptance Criteria
- `uvicorn api:app --reload` starts on port 8000
- CORS configured for `http://localhost:3000`
- SSE stream works end-to-end: POST /analyze TSLA streams events, final `{"type":"done"}`
- GET /memory returns Neo4j episodes
- GET /stats returns memory + learning counts

## Files
- `api.py` (new, project root)
- Update `requirements.txt` or add `fastapi uvicorn[standard]` comment
