# TASK-068: Fix main.py market data format mismatch with agent interface

## Status
- **State:** Queued
- **Priority:** Major
- **Depends on:** None
- **Created:** 2026-02-26

## Objective
Align the market data format that `main.py` writes into `AgentContext.metadata["market_data"]` with the format that `MarketAnalystAgent.think()` reads, so that market analysis is no longer silently skipped in paper-trade and backtest modes.

## Context
`main.py` (paper-trade and backtest modes) populates context as:
```python
context.metadata["market_data"] = {"AAPL": {"open": ..., "close": ...}, "MSFT": {...}}
```
`MarketAnalystAgent.think()` reads:
```python
market_data = context.metadata.get("market_data", {})
prices = market_data.get("prices", [])   # always returns []
```
`"prices"` is never a key in the dict-of-dicts format, so `prices` is always `[]` and the entire analysis block is skipped. All market analysis output is empty for every paper-trade and backtest run.

**Preferred fix** (consistent with single-symbol-per-run design): Update `main.py` to run the coordinator loop once per symbol per bar, and pass context as:
```python
context.metadata["market_data"] = {"symbol": sym, "prices": [bar_dict]}
```
This matches what `MarketAnalystAgent.think()` already expects.

If the multi-symbol-per-run design is preferred instead, update `MarketAnalystAgent.think()` to iterate the dict-of-dicts format — but document the design decision clearly and update `context.md`.

## Scope & Constraints
- **May modify:** `main.py`, `agents/market_analyst.py`
- **Must NOT modify:** Any other file
- Choose one consistent format and apply it end-to-end — do not create a hybrid
- If changing `main.py`, the loop structure changes but the overall dry-run / paper-trade / backtest flow must remain intact
- Do not change `MarketAnalystAgent`'s public `think()` signature

## Input
- `main.py` — paper-trade and backtest market data population code
- `agents/market_analyst.py` — `think()` method, `market_data` consumption logic

## Expected Output
- Either: `main.py` updated to pass `{"symbol": sym, "prices": [bar_dict]}` per symbol per iteration
- Or: `market_analyst.py` updated to consume the dict-of-dicts format and iterate all symbols
- One approach implemented consistently; code comment explains the design choice

## Acceptance Criteria
- [ ] `market_data.get("prices", [])` in `market_analyst.py` returns a non-empty list during paper-trade/backtest runs
- [ ] Market analysis executes (does not silently skip) when market data is present
- [ ] `python3 -m pytest tests/ -q` passes with 173 passed, 4 skipped (no regressions)
- [ ] `python3 main.py --mode dry-run` completes without error
- [ ] `python3 main.py --mode paper-trade` runs at least one analysis cycle without empty prices

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
