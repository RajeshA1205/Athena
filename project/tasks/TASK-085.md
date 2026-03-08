# TASK-085: Memory & stats panels

**Sprint:** 11 — React Frontend
**Priority:** 🟢 Medium
**Status:** Queued
**Dependencies:** TASK-082, TASK-080

## Description
Build two informational panels: Memory (shows recent Neo4j episodes) and Stats (shows memory operation counts + nested learning progress).

## Memory Panel (`/memory` section)

### `frontend/src/components/memory/MemoryPanel.tsx`
- Fetches GET /memory?limit=20 on mount and after each analysis
- Displays episode list: content (truncated to 120 chars), source agent, timestamp (relative: "2 min ago")
- Empty state: "No episodes yet — run an analysis to start building memory."
- Auto-refreshes every 30 seconds

### `frontend/src/components/memory/EpisodeCard.tsx`
- Minimal card: monospace content preview, source badge, timestamp
- Subtle left border colored by source agent

## Stats Panel (`/stats` section)

### `frontend/src/components/stats/StatsPanel.tsx`
- Fetches GET /stats on mount and after each analysis
- Three sub-sections: Memory Operations, Learning Progress, Query Count

### `frontend/src/components/stats/MemoryOpsTable.tsx`
- Table: Operation | Count
- Rows: ADD, UPDATE, DELETE, RETRIEVE, SUMMARY, FILTER

### `frontend/src/components/stats/LearningTable.tsx`
- Table: Agent | Inner Steps | Outer Updates
- One row per agent (market_analyst, risk_manager, strategy_agent, execution_agent, coordinator)

## Acceptance Criteria
- Memory panel shows real Neo4j episodes after analysis
- Stats panel shows real memory operation counts
- Both panels update after each analysis without page reload
- Loading skeleton shown while fetching
