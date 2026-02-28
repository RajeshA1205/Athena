# TASK-033: Seed random in execution_agent._simulate_fill

## Summary
`ExecutionAgent._simulate_fill()` uses `random.random()` from the global `random` module to generate slippage variance. This makes paper-trading fill simulations non-deterministic across runs, which prevents reproducible backtesting and makes tests fragile. The fix adds a per-instance `random.Random` instance seeded from a config value, following the same pattern already used in `trading/market_data.py` and `trading/order_management.py`.

> **Role change note (OQ-3 decision):** ExecutionAgent still receives this seeded RNG fix, but its role is changing in Sprint 6. Instead of submitting trade orders, ExecutionAgent now produces **recommended investment plans / advice** that a human trader acts on manually. The `_simulate_fill` code and slippage simulation remain in place and are used for recommendation reasoning (e.g., estimating realistic fill prices to inform the recommendation), but they no longer drive actual order submission. The seeded RNG fix is still valuable because it makes recommendation reasoning reproducible. See `project/decisions/sprint6-open-questions.md` (OQ-3) for the full decision rationale.

## Current State

**File:** `agents/execution_agent.py` (line 490)

```python
async def _simulate_fill(
    self, order: Order, ...
) -> Dict[str, Any]:
    """Simulate a single fill with slippage."""
    base_price = order.metadata.get("current_price", 100.0)
    estimated_slippage = order.metadata.get(
        "estimated_slippage_bps", self.default_slippage_bps
    )

    actual_slippage = estimated_slippage * (0.5 + random.random())   # line 490
```

## Proposed Change

**Step 1 — Add `seed` to `ExecutionAgent.__init__` config handling:**

```python
# In __init__, after existing config parsing:
seed = int(self.config.get("simulation_seed", 0)) or None
self._rng = random.Random(seed)
```

**Step 2 — Replace `random.random()` with `self._rng.random()`:**

```python
actual_slippage = estimated_slippage * (0.5 + self._rng.random())
```

**Config key:** `simulation_seed` (int, default `0`). `0` means use the current time (non-deterministic), matching `random.Random(None)` semantics. Pass a fixed integer for reproducible fills.

## Files Modified

- `agents/execution_agent.py`
  - `__init__`: add `self._rng = random.Random(...)` after config parsing
  - Line 490: `random.random()` → `self._rng.random()`

## Acceptance Criteria

- [ ] Two `ExecutionAgent` instances with `simulation_seed=42` produce identical `actual_slippage` for identical inputs
- [ ] `ExecutionAgent` with `simulation_seed=0` (default) behaves non-deterministically (for production use)
- [ ] All existing tests pass

## Edge Cases & Risks

- **Existing tests:** Tests that check fill price ranges (e.g. fill price within bounds of limit price) will pass regardless of the seed. Tests that check exact fill prices will need `simulation_seed` set.
- **Thread safety:** `random.Random` instances are not thread-safe. In async context (single event loop thread), this is not an issue.

## Test Notes

- Add test: create two `ExecutionAgent(config={"simulation_seed": 42})`, call `_simulate_fill` with identical input, assert identical `actual_slippage`.
- Existing fill tests need no changes.
