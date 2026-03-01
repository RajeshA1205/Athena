# TASK-089: Update ingest path structure to write under /data/market/

## Status
- **State:** Queued
- **Priority:** 🔴 Critical
- **Depends on:** TASK-088
- **Created:** 2026-03-01

## Objective
Change the ingest pipeline so that canonical OHLCV parquet files land at a predictable path that
`MarketDataFeed` can discover without knowing which data source produced them. The target layout is:

```
/Users/rajesh/athena/data/market/{SYMBOL}_ohlcv.parquet
```

One file per symbol, overwritten on each run (not date-stamped). This makes reads simple: the
reader looks for `data/market/AAPL_ohlcv.parquet` and either finds it or falls back to mock.

Additionally update `ingest/config.yaml` to reflect the new output path.

## Context
Currently `BaseCollector.save_to_parquet()` (in `/Users/rajesh/athena/ingest/src/data/collectors/base.py`)
writes to a path like:
```
data/raw/yfinance/AAPL_ohlcv_20260301.parquet
```
The date-stamp in the filename means a reader cannot know which file is current without listing
the directory. `MarketDataFeed` needs a stable, predictable path.

After TASK-088, `save_to_parquet()` already casts to the canonical schema. This task adds a second
write that emits the canonical file to the shared `data/market/` location, in addition to (not
replacing) the existing raw write. This preserves the raw archive for debugging.

The `collect_data.py` script resolves paths relative to its working directory (the `ingest/`
folder). The canonical write must resolve to an absolute path anchored at the ATHENA project root
(`/Users/rajesh/athena/`), not relative to `ingest/`.

## Scope & Constraints
- **Modify:** `/Users/rajesh/athena/ingest/src/data/collectors/base.py`
- **Modify:** `/Users/rajesh/athena/ingest/config.yaml` — add `canonical_data_dir` key
- **Modify:** `/Users/rajesh/athena/ingest/collect_data.py` — pass `canonical_data_dir` to collectors if needed
- **Do NOT modify:** any ATHENA agent files, `trading/market_data.py`, test files
- **Do NOT delete** the existing raw write; the canonical write is additive
- The canonical filename pattern must be exactly `{SYMBOL}_ohlcv.parquet` (no date stamp, no source prefix)
- The `data/market/` directory must be created by the code if it does not exist (using `Path.mkdir(parents=True, exist_ok=True)`)
- Use only stdlib + polars; no new packages

## Input
- `/Users/rajesh/athena/ingest/src/data/collectors/base.py` — after TASK-088 changes
- `/Users/rajesh/athena/ingest/src/data/schema.py` — `cast_to_canonical` from TASK-088
- `/Users/rajesh/athena/ingest/config.yaml` — add `canonical_data_dir` key here
- `/Users/rajesh/athena/ingest/collect_data.py` — understand how `data_dir` is passed

## Expected Output

### Updated `base.py` — `save_to_parquet()` signature
```python
def save_to_parquet(
    self,
    df: pl.DataFrame,
    symbol: str,
    data_type: str = "ohlcv",
    interval: str = "1d",
    canonical_dir: Optional[Path] = None,
) -> Path:
    """
    Save DataFrame to parquet.

    Writes two files:
    1. Raw archive: self.data_dir / f"{symbol}_{data_type}_{date}.parquet"  (existing behaviour)
    2. Canonical: canonical_dir / f"{symbol}_ohlcv.parquet"  (new, only when data_type=="ohlcv" and canonical_dir is not None)

    Returns path to the canonical file if written, else the raw archive path.
    """
```

### Updated `ingest/config.yaml`
Add under `data_processing`:
```yaml
canonical_data_dir: "/Users/rajesh/athena/data/market"
```

### Updated `collect_data.py`
- Read `canonical_data_dir` from the loaded config (or fall back to the hardcoded path above)
- Convert it to an absolute `Path` object
- Pass it as `canonical_dir=canonical_path` to each `save_to_parquet()` call

## Acceptance Criteria
- [ ] `base.py` `save_to_parquet()` accepts `canonical_dir: Optional[Path]` parameter
- [ ] When `canonical_dir` is provided and `data_type == "ohlcv"`, a file named `{SYMBOL}_ohlcv.parquet` is written to `canonical_dir`
- [ ] The canonical file passes `validate_canonical()` from `schema.py`
- [ ] The existing raw archive write still happens (no regression on the old write path)
- [ ] `ingest/config.yaml` contains `canonical_data_dir: "/Users/rajesh/athena/data/market"`
- [ ] `collect_data.py` reads `canonical_data_dir` from config and passes it to collectors
- [ ] Running `collect_data.py` with `use_yfinance=True` for a single symbol creates `/Users/rajesh/athena/data/market/{SYMBOL}_ohlcv.parquet`
- [ ] `python3 -m pytest tests/ -q` still reports 173 passed, 4 skipped

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
