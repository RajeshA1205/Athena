# TASK-088: Define canonical parquet schema for ingest OHLCV output

## Status
- **State:** Queued
- **Priority:** 🔴 Critical
- **Depends on:** None
- **Created:** 2026-03-01

## Objective
Define and document the canonical parquet schema that both the ingest pipeline and `MarketDataFeed`
will agree on. Write a small schema-validation helper module at
`/Users/rajesh/athena/ingest/src/data/schema.py` that other modules import to validate conformance.
The schema is the contract between the two systems — every downstream task depends on it being
correct and locked in first.

## Context
The ingest pipeline (polars-based) currently writes parquet files from `YFinanceCollector.save_to_parquet()`.
The base class `BaseCollector.save_to_parquet()` in `/Users/rajesh/athena/ingest/src/data/collectors/base.py`
saves whatever columns the DataFrame has. There is no enforcement of a shared schema.

`MarketDataFeed` (in `/Users/rajesh/athena/trading/market_data.py`) consumes `OHLCV` dataclasses with
these fields:
  - `symbol: str`
  - `timestamp: str`  (ISO-8601 UTC string)
  - `open: float`
  - `high: float`
  - `low: float`
  - `close: float`
  - `volume: float`
  - `interval: str`  (e.g. "1d")

The YFinance collector currently produces a DataFrame with columns:
  `date`, `open`, `high`, `low`, `close`, `volume`, `symbol`
where `date` is a Polars `Date` or `Datetime` type (timezone-aware from yfinance).

The canonical schema must bridge these two representations without loss.

## Scope & Constraints
- **Create:** `/Users/rajesh/athena/ingest/src/data/schema.py`
- **Modify:** `/Users/rajesh/athena/ingest/src/data/collectors/base.py` — add schema cast step inside `save_to_parquet()`
- **Do NOT modify:** `trading/market_data.py`, any test files, any other collector files
- Use `polars` (already a dependency); do not add new third-party packages
- The `timestamp` column in parquet must be stored as `Utf8`/`String` in ISO-8601 UTC format
  (`"2025-01-15T16:00:00+00:00"` style) so `MarketDataFeed` can consume it as a plain string
- The `interval` column must default to `"1d"` when not present in the source DataFrame
- All price columns (`open`, `high`, `low`, `close`) must be `Float64`
- `volume` must be `Float64`
- `symbol` must be `Utf8`/`String`

## Input
- `/Users/rajesh/athena/ingest/src/data/collectors/base.py` — existing `save_to_parquet()` to extend
- `/Users/rajesh/athena/ingest/src/data/collectors/yfinance_collector.py` — shows actual column names and types produced
- `/Users/rajesh/athena/trading/market_data.py` — the `OHLCV` dataclass (lines 28–50) is the consumption target

## Expected Output

### `/Users/rajesh/athena/ingest/src/data/schema.py`
```python
"""Canonical parquet schema for ATHENA market data."""
import polars as pl

# Ordered column list — parquet files must contain exactly these columns in this order
OHLCV_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume", "interval"]

OHLCV_SCHEMA = {
    "symbol":    pl.Utf8,
    "timestamp": pl.Utf8,    # ISO-8601 UTC string
    "open":      pl.Float64,
    "high":      pl.Float64,
    "low":       pl.Float64,
    "close":     pl.Float64,
    "volume":    pl.Float64,
    "interval":  pl.Utf8,
}

def cast_to_canonical(df: pl.DataFrame, interval: str = "1d") -> pl.DataFrame:
    """
    Cast a raw collector DataFrame to the canonical OHLCV schema.

    Handles:
    - Renaming 'date' -> 'timestamp' if present
    - Converting date/datetime column to ISO-8601 UTC string
    - Adding 'interval' literal column if absent
    - Casting all columns to declared types
    - Selecting only OHLCV_COLUMNS in declared order

    Args:
        df: Raw collector DataFrame (must have symbol, date/timestamp, open, high, low, close, volume)
        interval: Bar interval string to embed (default "1d")

    Returns:
        Canonical polars DataFrame conforming to OHLCV_SCHEMA

    Raises:
        ValueError: If required source columns are missing after rename
    """
    ...

def validate_canonical(df: pl.DataFrame) -> bool:
    """
    Return True if df conforms to OHLCV_SCHEMA; False otherwise.
    Logs specific violations via the module logger.
    """
    ...
```

### Modified `base.py`
- `save_to_parquet()` calls `cast_to_canonical(df, interval=interval_hint)` before writing
- Accepts an optional `interval: str = "1d"` parameter passed through from collectors
- Existing callers that omit `interval` get `"1d"` default (backward-compatible)

## Acceptance Criteria
- [ ] `/Users/rajesh/athena/ingest/src/data/schema.py` exists and exports `OHLCV_COLUMNS`, `OHLCV_SCHEMA`, `cast_to_canonical`, `validate_canonical`
- [ ] `cast_to_canonical` correctly renames `date` -> `timestamp` and formats as ISO-8601 UTC string
- [ ] `cast_to_canonical` adds `interval` column with the supplied default when not present in source
- [ ] `cast_to_canonical` casts price columns to `Float64` and `symbol` to `Utf8`
- [ ] `cast_to_canonical` raises `ValueError` if required columns (`symbol`, one of `date`/`timestamp`, `open`, `high`, `low`, `close`, `volume`) are missing
- [ ] `validate_canonical` returns `True` for a correctly shaped frame and `False` with log output for violations
- [ ] `base.py` `save_to_parquet()` calls `cast_to_canonical` before writing and accepts optional `interval` param
- [ ] A parquet file written by the updated `save_to_parquet()` can be read by polars and passes `validate_canonical`
- [ ] `python3 -m pytest tests/ -q` still reports 173 passed, 4 skipped (no regressions)

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
