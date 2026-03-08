# TASK-090: Implement MarketDataMode.FILE in MarketDataFeed

## Status
- **State:** Queued
- **Priority:** 🔴 Critical
- **Depends on:** TASK-088, TASK-089
- **Created:** 2026-03-01

## Objective
Add `MarketDataMode.FILE` to the `MarketDataMode` enum in `trading/market_data.py`. When this mode is active, `MarketDataFeed`:
- `get_realtime_data(symbol)` — reads the latest OHLCV bar (last row) from `data/market/{SYMBOL}_ohlcv.parquet`
- `get_historical_data(symbol, days)` — returns all rows from the parquet file up to a `days` limit
- Falls back to MOCK silently if the parquet file does not exist for the requested symbol

## Context
The schema for parquet files is defined in `ingest/SCHEMA.md` (TASK-088): columns `symbol`, `timestamp`, `open`, `high`, `low`, `close`, `volume`.

`MarketDataFeed` currently supports `MOCK` and `LIVE` modes. The new `FILE` mode is a read-only mode — it never writes. It is the bridge between the ingest pipeline and the agent system.

Project root for parquet files: `/Users/rajesh/athena/data/market/`

The `MarketBar` (or equivalent named tuple / dataclass) returned by `get_realtime_data()` must have fields: `symbol`, `timestamp`, `open`, `high`, `low`, `close`, `volume`. Check the existing return type and use it.

## Scope & Constraints
- **May modify:** `trading/market_data.py` only
- **May NOT modify:** any test files, `ingest/` files, `agents/` files, `main.py`
- Use `polars` for parquet reads (guard import with `try/except ImportError`; if polars unavailable, FILE mode falls back to MOCK with a WARNING log)
- All parquet reads must be wrapped in `asyncio.to_thread()` to avoid blocking the event loop
- The fallback to MOCK must be silent (log at DEBUG level, do not raise an exception)
- `get_historical_data()` must respect the `days` parameter: return at most the last `days` rows sorted ascending by timestamp

## Input
- `trading/market_data.py` — existing implementation to extend
- `ingest/SCHEMA.md` — canonical schema (from TASK-088)

## Expected Output
`trading/market_data.py` with:
1. `MarketDataMode.FILE = "file"` added to the enum
2. `MarketDataFeed.get_realtime_data()` — FILE branch reads last row from parquet; falls back to MOCK if file missing
3. `MarketDataFeed.get_historical_data()` — FILE branch reads all rows up to `days` limit from parquet; falls back to MOCK if file missing
4. Helper `_parquet_path(symbol)` private method returning the path `data/market/{symbol.upper()}_ohlcv.parquet` relative to project root

## Acceptance Criteria
- [ ] `MarketDataMode.FILE` exists in the enum
- [ ] `get_realtime_data(symbol)` returns a non-None bar with valid `open`, `high`, `low`, `close`, `volume` fields when a parquet file exists
- [ ] `get_realtime_data(symbol)` falls back to MOCK (no exception) when file is missing
- [ ] `get_historical_data(symbol, days)` returns a list/DataFrame of rows from the parquet file, capped at `days`
- [ ] `get_historical_data(symbol, days)` falls back to MOCK when file is missing
- [ ] Fallback is logged at DEBUG level (not WARNING or ERROR)
- [ ] `_parquet_path(symbol)` constructs path with uppercase symbol
- [ ] Existing `MOCK` and `LIVE` mode behavior is unchanged
- [ ] `pytest tests/ -q` passes with no new failures (173 passed, 4 skipped baseline)

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
