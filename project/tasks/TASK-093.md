# TASK-093: Integration Test — Ingest to MarketDataFeed

## Status
- **State:** Queued
- **Priority:** 🟡 High
- **Depends on:** TASK-089, TASK-090
- **Created:** 2026-03-01

## Objective
Write `tests/test_ingest_integration.py` that verifies the full path from ingest collector through to `MarketDataFeed` FILE mode:
1. Run collect_data for one symbol (AAPL) — or create a minimal synthetic parquet if network access is unavailable
2. Verify the parquet file lands at the correct path with the correct schema
3. Instantiate `MarketDataFeed` in FILE mode
4. Call `get_realtime_data("AAPL")` and assert it returns a non-None OHLCV bar with valid fields

## Context
This is the acceptance gate for Sprint 12. It proves the schema contract (TASK-088), the path fix (TASK-089), and the FILE mode implementation (TASK-090) all work together end-to-end.

The test must be self-contained and not depend on live network access. Use a pytest fixture that writes a minimal synthetic parquet file matching the schema from `ingest/SCHEMA.md` to `data/market/AAPL_ohlcv.parquet` (in a temporary directory), then tears it down after the test.

Schema: `symbol` (str), `timestamp` (datetime64[ns, UTC]), `open` (float64), `high` (float64), `low` (float64), `close` (float64), `volume` (float64).

Existing test baseline: 173 passed, 4 skipped. This test file adds new passing tests on top.

## Scope & Constraints
- **May create:** `tests/test_ingest_integration.py`
- **May NOT modify:** `trading/market_data.py`, `ingest/collect_data.py`, any other test file
- Tests must be runnable offline (no real YFinance API calls in the test itself)
- Use `tmp_path` pytest fixture or `monkeypatch` to redirect parquet file path during tests
- Do not hardcode `/Users/rajesh/athena/` — use the `tmp_path` fixture or `monkeypatch`
- Test file must follow existing test conventions (see `tests/test_trading.py` for style)

## Input
- `trading/market_data.py` — `MarketDataFeed` and `MarketDataMode` interfaces (after TASK-090)
- `ingest/SCHEMA.md` — canonical schema (from TASK-088)
- `tests/test_trading.py` — style reference

## Expected Output
`tests/test_ingest_integration.py` containing:

1. **`test_parquet_schema`** — creates a synthetic parquet at a tmp path; reads it back with pandas; asserts all 7 columns exist with correct dtypes
2. **`test_market_data_feed_file_mode`** — monkeypatches `MarketDataFeed._parquet_path()` to return a tmp parquet; calls `get_realtime_data("AAPL")`; asserts return value is not None and has `open`, `high`, `low`, `close`, `volume` attributes/keys with numeric values
3. **`test_market_data_feed_file_mode_fallback`** — instantiates `MarketDataFeed` in FILE mode with a non-existent parquet path; calls `get_realtime_data("MISSING")`; asserts no exception is raised and a fallback MOCK bar is returned
4. **`test_get_historical_data_days_limit`** — creates a synthetic parquet with 30 rows; calls `get_historical_data("AAPL", days=7)`; asserts at most 7 rows returned

## Acceptance Criteria
- [ ] `tests/test_ingest_integration.py` exists and is syntactically valid
- [ ] `test_parquet_schema` passes: all 7 columns present with correct dtypes
- [ ] `test_market_data_feed_file_mode` passes: non-None bar with numeric OHLCV fields
- [ ] `test_market_data_feed_file_mode_fallback` passes: no exception, valid MOCK bar returned
- [ ] `test_get_historical_data_days_limit` passes: at most 7 rows returned for `days=7`
- [ ] No real network calls in any test
- [ ] `pytest tests/ -q` shows at least 4 new passing tests (total >= 177 passed, 4 skipped)
- [ ] All existing 173 tests continue to pass

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
