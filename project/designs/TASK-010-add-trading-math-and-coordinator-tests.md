# TASK-010: Add tests for trading math and coordinator conflict resolution

## Problem

Critical financial math and coordination logic lacks test coverage:
- VaR calculation accuracy (`RiskManagerAgent._calculate_var`)
- Sharpe ratio edge cases (zero volatility, single return, negative mean)
- FIFO cost-basis (if implemented in `trading/order_management.py`)
- Position sizing boundary conditions (`StrategyAgent._calculate_position_size`)
- Coordinator role-priority conflict resolution (`CoordinatorAgent._resolve_conflicts`)

Incorrect financial calculations or conflict resolution logic could produce harmful investment recommendations.

## Files to change

| File | Lines | Change |
|------|-------|--------|
| `tests/test_risk_math.py` | (new) | VaR accuracy, Expected Shortfall, portfolio metrics tests |
| `tests/test_strategy_math.py` | (new) | Position sizing, momentum signals, mean reversion signal boundary tests |
| `tests/test_coordinator_resolution.py` | (new) | Conflict detection, priority-weighted resolution, edge case tests |

## Approach

### 1. VaR accuracy tests (`tests/test_risk_math.py`)

Test `RiskManagerAgent._calculate_var()`:
- **Known distribution**: Feed a uniform return series `[-0.05, -0.04, ..., 0.04, 0.05]` and verify VaR(95%) equals the expected percentile value.
- **Single asset**: Returns dict with one key; verify no crash and correct scalar output.
- **Empty returns**: `{}` should return `0.0`.
- **All positive returns**: VaR should be 0 or near-0 (no losses in tail).
- **Parametric method**: Test with known normal distribution parameters.

### 2. Expected Shortfall tests

Test `RiskManagerAgent._calculate_expected_shortfall()`:
- Verify ES >= VaR for the same confidence level.
- Known distribution: manually compute the average of returns below the VaR cutoff.

### 3. Sharpe ratio edge cases

Test `RiskManagerAgent._calculate_portfolio_metrics()`:
- **Zero volatility**: All returns identical -> Sharpe should be `0.0` (not `inf`).
- **Single return**: Length-1 series -> no crash.
- **Negative mean**: Sharpe should be negative.
- **Max drawdown**: Feed a peak-trough-recovery pattern, verify correct drawdown.

### 4. Position sizing boundaries (`tests/test_strategy_math.py`)

Test `StrategyAgent._calculate_position_size()`:
- `signal_strength=0` -> returns `0.0`.
- `volatility=0` -> returns `0.0` (not division-by-zero).
- High volatility -> position size clamped at `0.2` (the `min(..., 0.2)` cap).
- Normal case: verify `risk_budget / volatility * signal_strength`.

### 5. Momentum and mean-reversion signal tests

- Too few data points -> empty signals list.
- Exact threshold (`momentum == 0.02`) -> verify boundary behavior.
- Extreme z-score -> strength clamped at `1.0`.

### 6. Coordinator conflict resolution (`tests/test_coordinator_resolution.py`)

Test `CoordinatorAgent._resolve_conflicts()`:
- **Risk > Strategy > Analyst**: Risk says "sell" with high confidence, strategy says "buy" -> risk should win.
- **Unanimous**: All agents agree "buy" -> decision is "buy" with high confidence.
- **No valid recommendations**: Empty dict -> "hold" with 0.0 confidence.
- **Unknown agent name**: Agent not in `self.agents` dict; falls back to substring match on line 448-450 -- verify this path.
- **All hold**: No conflict detected, majority path taken.

Test `CoordinatorAgent._detect_conflicts()`:
- Buy vs sell -> direct conflict detected.
- Buy vs hold -> no conflict (only buy vs sell triggers).
- All same action -> empty conflicts list.

## Edge cases / risks

- `_calculate_var` uses integer indexing: `int(len(sorted_returns) * (1 - confidence))`. For small return series (e.g., 10 values), the index rounding can produce unexpected results. Test with small series sizes.
- `_calculate_portfolio_metrics` uses population std dev (divides by N, not N-1). Tests should use the same formula for expected values.
- Coordinator's `_resolve_conflicts` is an async method; tests need `pytest-asyncio`.

## Acceptance criteria

- [ ] At least 3 VaR accuracy tests covering historical method with known distributions.
- [ ] At least 2 Expected Shortfall tests verifying ES >= VaR.
- [ ] At least 3 Sharpe edge-case tests (zero vol, single return, negative mean).
- [ ] At least 3 position sizing boundary tests.
- [ ] At least 4 coordinator conflict resolution tests covering priority weighting.
- [ ] All new tests pass: `pytest tests/test_risk_math.py tests/test_strategy_math.py tests/test_coordinator_resolution.py -q`.
- [ ] `pytest tests/ -q` remains green.
