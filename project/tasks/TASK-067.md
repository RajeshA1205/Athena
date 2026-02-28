# TASK-067: Fix portfolio check_limits double-counting exposure on sells

## Status
- **State:** Queued
- **Priority:** Major
- **Depends on:** None
- **Created:** 2026-02-26

## Objective
Fix `check_limits` in `trading/portfolio.py` to compute the signed exposure delta when evaluating a new order — so that a sell that reduces an existing long position decreases (rather than increases) total exposure, preventing false limit violations on valid risk-reducing trades.

## Context
Current logic in `check_limits` adds `abs(quantity * price)` to `current_exposure` unconditionally for every order. For a sell order with `quantity = -100` and `price = 50.0`, this adds `5000.0` to exposure even when the agent holds a long position of 200 shares. The correct behavior is:

1. Look up the current position quantity for the symbol: `current_qty`
2. Compute the post-order position: `new_pos_qty = current_qty + quantity` (negative for sell)
3. Compute the exposure delta: `delta = abs(new_pos_qty * price) - abs(current_qty * price)`
4. Add the signed `delta` to `current_exposure` — sells that reduce the position produce a negative delta, correctly reducing exposure

This produces false `POSITION_LIMIT_EXCEEDED` rejections for every sell order, making the risk manager block all risk-reducing trades.

## Scope & Constraints
- **May modify:** `trading/portfolio.py` only
- **Must NOT modify:** Any other file
- The fix must handle the case where `symbol` has no existing position (treat `current_qty = 0`)
- Do not change the method signature or return type of `check_limits`
- Be careful not to break the existing behavior for new positions (where `current_qty = 0`, the delta equals `abs(quantity * price)` — same as before)

## Input
- `trading/portfolio.py` — `check_limits` method

## Expected Output
- `trading/portfolio.py` — `check_limits` uses the signed delta calculation described above; sells that reduce an existing long no longer inflate exposure

## Acceptance Criteria
- [ ] `check_limits` computes `new_pos_qty = current_qty + quantity` and `delta = abs(new_pos_qty * price) - abs(current_qty * price)`
- [ ] Sell orders that fully close a long position result in `delta <= 0` (exposure decreases)
- [ ] Buy orders for a new position result in `delta = abs(quantity * price)` (same as before)
- [ ] `python3 -m pytest tests/ -q` passes with 173 passed, 4 skipped (no regressions)
- [ ] `python3 main.py --mode dry-run` completes without error

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
