# TASK-087: Verification gate — build, integration, deployment check

**Sprint:** 11 — React Frontend
**Priority:** 🔴 Critical
**Status:** Queued
**Dependencies:** TASK-079 through TASK-086

## Description
Final verification gate for Sprint 11. Confirms the full stack (FastAPI + Next.js) works end-to-end and the Python test suite has no regressions.

## Checklist

### Python Backend
- [ ] `python3 -m pytest tests/ -q` → 173 passed, 4 skipped (no regressions)
- [ ] `uvicorn api:app` starts without errors
- [ ] GET /health returns `{"status":"ok"}`
- [ ] POST /analyze TSLA streams events and closes with `{"type":"done"}`
- [ ] GET /memory returns episode list
- [ ] GET /stats returns counts

### Next.js Frontend
- [ ] `npm run build` in `frontend/` succeeds with no TypeScript errors
- [ ] `npm run dev` starts on port 3000
- [ ] DragonLogo visible in header
- [ ] Symbol input → Enter → agent stream appears → results rendered
- [ ] Memory panel shows episodes
- [ ] Stats panel shows learning data
- [ ] Mobile layout works (sidebar collapses)

### End-to-End
- [ ] Full flow: open browser → type TSLA → see stream → see recommendation → check memory panel shows new episode

## Acceptance Criteria
All checklist items pass. No console errors in browser. Backend logs show no unhandled exceptions.
