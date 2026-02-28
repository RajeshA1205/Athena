# TASK-027: Replace hash() with _stable_hash in order_management._simulate_fill

## Summary
`OrderManager._simulate_fill()` uses Python's built-in `hash(order.symbol)` to derive a deterministic base price for paper trading fills. Since Python 3.3, string `hash()` is randomized per-process by default (`PYTHONHASHSEED`). This means fill prices differ across Python sessions even for the same symbol, breaking reproducibility of paper-trading simulations and backtests. The file already defines `_stable_hash()` using SHA-256 (added in an earlier fix session) — it just needs to be used at lines 272 and 275.

## Current State

**File:** `trading/order_management.py`

```python
# Line 19 — _stable_hash already exists
def _stable_hash(s: str) -> int:
    """Return a stable, process-invariant hash using SHA-256."""
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)

# Lines 271-275 — still using hash()
if order.order_type == OrderType.MARKET:
    base_price = 50.0 + hash(order.symbol) % 451        # <-- line 272
else:
    base_price = order.limit_price if order.limit_price is not None \
        else 50.0 + hash(order.symbol) % 451             # <-- line 275
```

## Proposed Change

```python
if order.order_type == OrderType.MARKET:
    base_price = 50.0 + _stable_hash(order.symbol) % 451
else:
    base_price = order.limit_price if order.limit_price is not None \
        else 50.0 + _stable_hash(order.symbol) % 451
```

Two-character change per line. No other modifications needed.

## Files Modified

- `trading/order_management.py`
  - Line 272: `hash(order.symbol)` → `_stable_hash(order.symbol)`
  - Line 275: `hash(order.symbol)` → `_stable_hash(order.symbol)`

## Acceptance Criteria

- [ ] No calls to bare `hash()` remain in `order_management.py`
- [ ] `_simulate_fill` produces identical `fill_price` values across separate Python processes for the same symbol and order type
- [ ] All existing tests pass

## Edge Cases & Risks

- **Price value change:** The SHA-256-based `_stable_hash` returns a very large integer. `_stable_hash(symbol) % 451` still produces a value in `[0, 450]`, so `base_price` remains in `[50.0, 500.0]` — same range as before, just different specific values. Any test that asserts an exact fill price for a specific symbol will need updating.
- **`PYTHONHASHSEED=0`:** If tests were previously run with `PYTHONHASHSEED=0` (making `hash()` deterministic), existing expected values in tests would be wrong after this change regardless. Verify test assertions.

## Test Notes

- `tests/test_trading.py::TestOrderManager::test_simulate_fill_produces_fill` checks that a fill is produced but not the exact price — should pass without changes.
- Add a determinism test: run `_simulate_fill` for `"AAPL"` MARKET order, assert fill price equals a known expected value (computed once with the SHA-256 hash).
