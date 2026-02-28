# TASK-060: Fix MarketDataFeed placement and private attribute access in main.py

## Status
- **State:** Queued
- **Priority:** 🟡 Minor
- **Depends on:** None
- **Created:** 2026-02-25

## Objective
Fix two related issues in `main.py`:
(a) `MarketDataFeed` is instantiated before the dry-run early-return branch, so it runs even when no feed is needed.
(b) `main.py` reads the private attribute `feed._MOCK_SYMBOLS` directly, violating encapsulation.

## Context
Issue (a): `main.py` creates a `MarketDataFeed` instance and resolves symbols near the top of `main()`, then checks `if args.mode == "dry-run": return`. The feed instantiation therefore always executes, including in dry-run mode where it is never used. This wastes resources and may cause unintended side effects (e.g. network calls, file I/O) in dry-run.

Issue (b): `_MOCK_SYMBOLS` is a class-level constant prefixed with `_` to signal it is private. External access via `feed._MOCK_SYMBOLS` bypasses the intended API boundary.

## Scope & Constraints
- **May modify:** `main.py`
- **Must NOT modify:** `trading/market_data.py` or any other file
- If `MarketDataFeed` has no public symbols accessor, define a local fallback list inside the branch rather than changing the class

## Input
- `main.py` — `main()` function, feed instantiation and dry-run branch

## Expected Output
- `MarketDataFeed(...)` instantiation moved inside the `paper-trade` / `backtest` branches
- Symbol list resolved without accessing `_MOCK_SYMBOLS` directly (use a public accessor if available, otherwise define a local default list such as `["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]`)

## Acceptance Criteria
- [ ] `MarketDataFeed` is not instantiated when `args.mode == "dry-run"`
- [ ] No reference to `feed._MOCK_SYMBOLS` (or any `._` private attribute of `MarketDataFeed`) remains in `main.py`
- [ ] `python3 main.py --mode dry-run` still exits cleanly
- [ ] `python3 -m pytest tests/ -q` stays green (173 passed, 4 skipped)

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
