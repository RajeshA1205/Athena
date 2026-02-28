# TASK-069: Fix Sharpe Ratio mean_return Using Compounded Return Instead of Arithmetic Mean

## Summary

`StrategyAgent._calculate_sharpe_ratio()` (around line 629 in `agents/strategy_agent.py`) computes `mean_return = total_return / len(returns)` where `total_return` is the **compounded** return (`∏(1+r) - 1`). The Sharpe ratio formula requires the **arithmetic mean** of period returns. Using a compounded return divided by the count gives a different value than the arithmetic mean — smaller for positive returns (geometric vs arithmetic), potentially negative when many losing periods make the product contract. The `std_return` is already computed correctly using the arithmetic mean, so the numerator and denominator of the Sharpe ratio are inconsistent. Fix: replace `total_return / len(returns)` with `sum(returns) / len(returns)`.

## Current State

**File:** `agents/strategy_agent.py`, lines ~624-633:

```python
        cumulative_for_return = 1.0
        for r in returns:
            cumulative_for_return *= (1 + r)
        total_return = cumulative_for_return - 1.0
        mean_return = total_return / len(returns)   # BUG: compounded return ÷ count
        std_return = math.sqrt(sum((r - mean_return) ** 2 for r in returns) / len(returns))
        sharpe_ratio = (mean_return / (std_return + 1e-10)) * math.sqrt(252)
```

The `std_return` calculation uses `mean_return` in its deviation term. Because `mean_return` is wrong, `std_return` is also wrong — the deviations are measured from the wrong center. Both the numerator and denominator of the Sharpe ratio are therefore incorrect.

**Why the bug exists:** `total_return` was likely intended for a separate "total cumulative return" metric. It was then reused (incorrectly) as the basis for `mean_return` instead of computing `sum(returns) / len(returns)` directly.

**Magnitude of the error (example):**

| Period returns | Arithmetic mean | Compounded / N | Error |
|---------------|-----------------|----------------|-------|
| [0.01, 0.02, -0.005, 0.015] | 0.010 | (1.01·1.02·0.995·1.015 − 1) / 4 ≈ 0.0098 | ~2% |
| [0.05, 0.05, 0.05, -0.10] | 0.0125 | (1.05³·0.90 − 1) / 4 ≈ 0.0076 | ~39% |

The error grows with return volatility and the number of periods, causing the Sharpe ratio to be systematically underestimated.

## Proposed Change

**File:** `agents/strategy_agent.py`

Replace the `mean_return` computation:

```python
# BEFORE
total_return = cumulative_for_return - 1.0
mean_return = total_return / len(returns)

# AFTER
total_return = cumulative_for_return - 1.0          # keep for any other use
mean_return = sum(returns) / len(returns)            # arithmetic mean of period returns
```

The `total_return` variable may be used elsewhere in the method (e.g., logged or returned as a separate metric), so it is retained. Only the `mean_return` line changes.

### Full corrected block

```python
        cumulative_for_return = 1.0
        for r in returns:
            cumulative_for_return *= (1 + r)
        total_return = cumulative_for_return - 1.0
        mean_return = sum(returns) / len(returns)    # arithmetic mean of period returns
        std_return = math.sqrt(sum((r - mean_return) ** 2 for r in returns) / len(returns))
        sharpe_ratio = (mean_return / (std_return + 1e-10)) * math.sqrt(252)
```

## Files Modified

| File | Line | Change |
|------|------|--------|
| `agents/strategy_agent.py` | ~629 | `mean_return = total_return / len(returns)` → `mean_return = sum(returns) / len(returns)` |

## Acceptance Criteria

- `mean_return` is computed as `sum(returns) / len(returns)` (arithmetic mean).
- `total_return` (cumulative) is still computed and available for logging/return.
- `std_return` uses the corrected `mean_return` as its center, making Sharpe numerator and denominator consistent.
- For a flat sequence `[0.01, 0.01, 0.01]`, `mean_return == 0.01` exactly.
- For a mixed sequence `[0.05, -0.05]`, `mean_return == 0.0` (symmetric, not slightly negative as the compounded approach gives).
- 173 tests pass, 4 skipped.

## Edge Cases & Risks

1. **`len(returns) == 0` guard**: The method already has an early-exit guard (`if not prices or len(prices) < 30`) that ensures `prices` has at least 30 elements before computing `returns`. Since `returns = [prices[i]/prices[i-1] - 1 for i in range(1, len(prices))]`, `len(returns) >= 29`. Division by zero is not possible.
2. **`total_return` retained**: If `total_return` is logged or included in the returned strategy dict, it remains correct (compounded cumulative return). Only its misuse as `mean_return` is removed.
3. **Sign of Sharpe**: With the arithmetic mean, a portfolio of all-negative returns correctly yields a negative Sharpe. The old code could yield a less-negative (or spuriously positive) Sharpe by understating the magnitude of losses.
4. **One-line change**: No new imports, no structural changes, no logic branches added.

## Test Notes

- Manually verify: `returns = [0.01, 0.02, -0.01, 0.02]` → `mean_return = 0.04/4 = 0.01`. Old code: `total_return = (1.01·1.02·0.99·1.02) − 1 ≈ 0.0406`, `mean_return = 0.0406/4 ≈ 0.01015`. Error ≈ 1.5%.
- Verify `std_return` decreases when `mean_return` is the correct center of the distribution.
- Run `python3 -m pytest tests/ -q` — expect 173 passed, 4 skipped.
