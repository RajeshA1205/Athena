# TASK-069: Fix Sharpe ratio mean_return calculation in StrategyAgent

## Status
- **State:** Queued
- **Priority:** Major (financial correctness)
- **Depends on:** None
- **Created:** 2026-02-26

## Objective
Fix the `mean_return` calculation in `agents/strategy_agent.py` used for the Sharpe ratio. Replace the incorrect `total_return / len(returns)` (compounded total divided by period count) with the arithmetic mean `sum(returns) / len(returns)`.

## Context
The Sharpe ratio formula is: `(mean_period_return - risk_free_rate) / std(period_returns)`. The numerator requires the arithmetic mean of period return floats (e.g., `[0.01, -0.005, 0.02, ...]`).

Around line 629 of `agents/strategy_agent.py`, the code computes:
```python
mean_return = total_return / len(returns)
```
`total_return` is calculated via a geometric product (e.g., `(1+r1)*(1+r2)*...*(1+rN) - 1`), not a sum. Dividing this compounded scalar by the number of periods does not produce the arithmetic mean — it produces a number that is arbitrarily wrong depending on the magnitude and sign of returns. This means every backtest Sharpe ratio is incorrect and cannot be used to compare strategies.

Sprint 6 TASK-032 fixed the `total_return` formula itself to be geometric; this task fixes the downstream `mean_return` consumer.

## Scope & Constraints
- **May modify:** `agents/strategy_agent.py` only
- **Must NOT modify:** Any other file
- Change is exactly one line: `mean_return = total_return / len(returns)` → `mean_return = sum(returns) / len(returns)`
- `returns` must already be the list of per-period return floats at this point in the code — verify this before making the change
- Add a comment explaining the distinction between arithmetic mean (for Sharpe) and geometric total (for CAGR)

## Input
- `agents/strategy_agent.py` — Sharpe ratio calculation block (~line 629)

## Expected Output
- `agents/strategy_agent.py` — `mean_return = sum(returns) / len(returns)` with explanatory comment; `total_return` is not used in the Sharpe numerator

## Acceptance Criteria
- [ ] `mean_return = sum(returns) / len(returns)` is present at the Sharpe calculation site
- [ ] `total_return` is NOT divided by `len(returns)` anywhere in the Sharpe calculation path
- [ ] A comment distinguishes arithmetic mean (Sharpe) from geometric total (CAGR/drawdown)
- [ ] `python3 -m pytest tests/ -q` passes with 173 passed, 4 skipped (no regressions)
- [ ] `python3 main.py --mode dry-run` completes without error

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
