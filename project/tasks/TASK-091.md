# TASK-091: Reconcile Symbol List

## Status
- **State:** Queued
- **Priority:** 🟡 High
- **Depends on:** None
- **Created:** 2026-03-01

## Objective
Merge the 60+ symbols from `ingest/symbols.yaml` into ATHENA's hardcoded symbol list in `trading/market_data.py`. If `cli.py` defines a `SUPPORTED_SYMBOLS` list, update it to match. The combined list must be the union of both sources — no symbol from either file should be dropped.

## Context
The ingest pipeline collects data for a large universe of symbols defined in `ingest/symbols.yaml`. `MarketDataFeed` in `trading/market_data.py` currently has its own hardcoded symbol list (the `MOCK_SYMBOLS` constant, made public in TASK-060). These two lists diverged and need to be unified so that FILE mode (TASK-090) can serve data for any symbol the ingest pipeline collected.

`cli.py` may also contain a `SUPPORTED_SYMBOLS` list used for input validation. All three locations must agree.

## Scope & Constraints
- **May modify:** `trading/market_data.py`, `cli.py` (if `SUPPORTED_SYMBOLS` exists there)
- **May NOT modify:** `ingest/symbols.yaml`, any test files, any agent files
- Symbols must be uppercase strings
- The merged list should be sorted alphabetically for readability and determinism
- Do not remove any symbol that is already in `trading/market_data.py`
- Do not remove any symbol from `ingest/symbols.yaml`

## Input
- `ingest/symbols.yaml` — source of truth for ingest symbols
- `trading/market_data.py` — current `MOCK_SYMBOLS` or equivalent list
- `cli.py` — check if `SUPPORTED_SYMBOLS` exists

## Expected Output
- `trading/market_data.py` — `MOCK_SYMBOLS` (or equivalent constant) updated to the sorted union of both symbol lists
- `cli.py` — `SUPPORTED_SYMBOLS` updated to match if it exists; file unchanged if it does not

## Acceptance Criteria
- [ ] Every symbol in `ingest/symbols.yaml` appears in `trading/market_data.py`'s symbol list
- [ ] Every symbol previously in `trading/market_data.py`'s symbol list is still present
- [ ] All symbols are uppercase
- [ ] Symbol list is sorted alphabetically
- [ ] If `cli.py` has `SUPPORTED_SYMBOLS`, it matches the merged list in `trading/market_data.py`
- [ ] `pytest tests/ -q` passes with no new failures (173 passed, 4 skipped baseline)

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
