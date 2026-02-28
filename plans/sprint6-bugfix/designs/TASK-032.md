# TASK-032: Fix backtest total_return to use geometric model

## Summary
`StrategyAgent._run_backtest()` computes `total_return = sum(returns)` (arithmetic/additive), but the `max_drawdown` calculation on lines 620–629 uses `cumulative *= (1 + r)` (geometric/multiplicative). The two metrics are now on incompatible scales: `total_return = 0.10` could mean 10% additive return while `max_drawdown` measures drawdown from a compound equity curve. For a strategy with alternating +5%/-4% returns, additive total return = +1% but compound total return = (1.05 × 0.96 - 1) ≈ +0.8%. The fix is to derive `total_return` from the same `cumulative` variable already computed for drawdown.

## Current State

**File:** `agents/strategy_agent.py` (lines 612–636)

```python
if not returns:
    return empty

total_return = sum(returns)                                          # line 615 — additive
mean_return = total_return / len(returns)                           # line 616
std_return = math.sqrt(sum((r - mean_return) ** 2 for r in returns) / len(returns))
sharpe_ratio = (mean_return / (std_return + 1e-10)) * math.sqrt(252)

cumulative = 1.0                                                     # line 620
peak = 1.0
max_dd = 0.0
for r in returns:
    cumulative *= (1 + r)                                            # line 624 — geometric
    if cumulative > peak:
        peak = cumulative
    dd = (peak - cumulative) / peak
    if dd > max_dd:
        max_dd = dd
```

## Proposed Change

Move the `cumulative` loop before the `total_return` assignment and derive `total_return` geometrically:

```python
if not returns:
    return empty

# Compute geometric equity curve (used for both total_return and drawdown)
cumulative = 1.0
peak = 1.0
max_dd = 0.0
for r in returns:
    cumulative *= (1 + r)
    if cumulative > peak:
        peak = cumulative
    dd = (peak - cumulative) / peak
    if dd > max_dd:
        max_dd = dd

total_return = cumulative - 1.0          # geometric total return
mean_return = sum(returns) / len(returns)  # arithmetic mean for Sharpe (industry standard)
std_return = math.sqrt(sum((r - mean_return) ** 2 for r in returns) / len(returns))
sharpe_ratio = (mean_return / (std_return + 1e-10)) * math.sqrt(252)
```

`mean_return` intentionally stays as arithmetic mean — the Sharpe ratio convention uses arithmetic mean of periodic returns.

## Files Modified

- `agents/strategy_agent.py`
  - Lines 615–629: reorder and fix `total_return` computation as shown above

## Acceptance Criteria

- [ ] `total_return` equals `cumulative - 1.0` (geometric), not `sum(returns)` (additive)
- [ ] `max_drawdown` still uses the same geometric `cumulative` variable
- [ ] `sharpe_ratio` calculation is unchanged (arithmetic mean is correct for Sharpe)
- [ ] All existing tests pass

## Edge Cases & Risks

- **All-zero returns:** `cumulative` remains `1.0`, `total_return = 0.0`. Same as `sum([0, 0, ...]) = 0.0`. No change.
- **Single return:** `cumulative = 1 + r[0]`, `total_return = r[0]`. Same as `sum([r[0]]) = r[0]`. No change for single-period.
- **Large negative returns:** `(1 + r)` can go negative if `r < -1.0` (>100% loss). This should not occur in a well-behaved backtest, but consider a clamp: `cumulative *= max(1 + r, 0.0)`.
- **Test expected values:** If any test asserts `result["total_return"] == some_additive_value`, it will fail after this change. Search tests for such assertions.

## Test Notes

- Existing `test_agents.py` tests call `think()` and check return type, not backtest values.
- Add a numerical test: for returns `[0.05, -0.04, 0.05, -0.04]`, assert `total_return ≈ 0.0192` (geometric) not `0.02` (additive).
