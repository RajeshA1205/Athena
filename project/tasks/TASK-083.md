# TASK-083: Symbol analysis — input form + agent thought stream

**Sprint:** 11 — React Frontend
**Priority:** 🟡 High
**Status:** Queued
**Dependencies:** TASK-080, TASK-082

## Description
Build the core query interface: a symbol input form that fires a POST /analyze request and streams the 5-agent thought process in real time as the SSE events arrive.

## Components

### `frontend/src/components/analyze/SymbolInput.tsx`
- Text input for ticker symbol (e.g. "TSLA")
- "Analyze" submit button
- Keyboard shortcut: Enter to submit
- Disabled while analysis is running
- Error state if API unreachable

### `frontend/src/components/analyze/AgentStream.tsx`
- Shows agent steps as they stream in
- One row per event: `[agent_name] [step] content ... Xms`
- Color-coded by agent:
  - market_analyst → blue
  - risk_manager → amber
  - strategy_agent → purple
  - execution_agent → green
  - coordinator → white/default
- Monospace font for the content text
- Auto-scrolls to bottom as new steps arrive
- Shows a pulsing cursor while streaming

### `frontend/src/hooks/useAnalysis.ts`
- Custom hook wrapping the SSE fetch
- Returns: `{ stream, result, isLoading, error, analyze(symbol) }`
- Uses `EventSource` or `fetch` with `ReadableStream` for SSE
- Accumulates steps into state array, updates `result` on `type: "result"` event

## Acceptance Criteria
- Typing "TSLA" and pressing Enter triggers streaming
- Agent steps appear in real time as SSE events arrive
- "Analyzing..." spinner shown during streaming
- Stream stops cleanly on `type: "done"` event
- Error banner shown if API call fails
