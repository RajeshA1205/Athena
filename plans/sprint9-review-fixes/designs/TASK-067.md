# TASK-067: Fix Portfolio check_limits Double-Counting Exposure on Sells

## Summary

`Portfolio.check_limits()` (lines 199-207) computes `proposed_value = abs(quantity * price)` and adds it unconditionally to `exposure["total_exposure"]`. For a sell (quantity < 0), this adds the absolute notional of the sale to the current exposure instead of subtracting it, causing valid risk-reducing trades to be flagged as limit violations. Fix: compute the net delta in market exposure caused by the proposed trade and apply that signed delta.

## Current State

**File:** `trading/portfolio.py`, lines 199-207:

```python
        # Total exposure check
        proposed_value = abs(quantity * price)
        exposure = await self.get_exposure()
        new_exposure = exposure["total_exposure"] + proposed_value
        if new_exposure > self.max_total_exposure:
            reasons.append(
                f"Total exposure {new_exposure:.0f} would exceed limit "
                f"{self.max_total_exposure:.0f}"
            )
```

`get_exposure()` (lines 148-173) returns:
```python
        return {
            "total_exposure": round(long_exp + short_exp, 4),
            ...
            "per_symbol": per_symbol,   # market_value per symbol
        }
```

Where `long_exp = sum(p.market_value for p in self._positions.values() if p.quantity > 0)`.

`Position.market_value` is a property: `quantity * last_price` (or `quantity * avg_cost` as fallback).

## Proposed Change

Replace the `proposed_value` / `new_exposure` block with a net-delta calculation:

```python
        # Total exposure check — compute net change in exposure
        exposure = await self.get_exposure()
        current_qty = self._positions.get(symbol, Position(symbol=symbol)).quantity
        # Market value of current position for this symbol
        current_pos_value = abs(current_qty * price)
        # Market value of new position after this trade
        new_qty = current_qty + quantity
        new_pos_value = abs(new_qty * price)
        # Net delta: positive means exposure increases, negative means it decreases
        exposure_delta = new_pos_value - current_pos_value
        new_exposure = exposure["total_exposure"] + exposure_delta
        if new_exposure > self.max_total_exposure:
            reasons.append(
                f"Total exposure {new_exposure:.0f} would exceed limit "
                f"{self.max_total_exposure:.0f}"
            )
```

Note: `current_qty` is already retrieved above (line 192) for the position-size check:
```python
        current_qty = self._positions.get(symbol, Position(symbol=symbol)).quantity
```

So we reuse that variable rather than looking up again.

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `trading/portfolio.py` | 199-207 | Replace `proposed_value` / `new_exposure` block with net-delta calculation |

## Acceptance Criteria

- A buy trade that increases exposure is correctly flagged when `new_exposure > max_total_exposure`.
- A sell trade that reduces exposure does NOT trigger a false violation.
- A short sale that opens a new short position increases exposure correctly.
- A cover trade that reduces a short position decreases exposure correctly.
- `new_exposure` equals `exposure["total_exposure"] + (abs(current_qty + quantity) - abs(current_qty)) * price`.
- 173 tests pass, 4 skipped.

## Edge Cases & Risks

1. **Price approximation**: The fix uses the proposed execution `price` to estimate both current and new position values. The current position's actual market value may differ if `last_price` in the Position object differs from `price`. This is consistent with the original approach (both old and new code use `price` for the check) and acceptable for a pre-trade limit check.
2. **Reversing positions**: If `current_qty = +100` and `quantity = -200` (reversing to a -100 short), `new_qty = -100`. `current_pos_value = 100*price`, `new_pos_value = 100*price`, `delta = 0`. The exposure is unchanged — this is mathematically correct for a position reversal of equal size.
3. **`current_qty` reuse**: Line 192 already reads `current_qty = self._positions.get(symbol, Position(symbol=symbol)).quantity`. The new code reuses this variable; no extra dict lookup needed.

## Test Notes

- Test case: portfolio with `total_exposure = 90_000`, `max_total_exposure = 100_000`. Long 1000 shares at $100 (exposure = $100K used). Sell 100 shares at $100 → `exposure_delta = abs(900*100) - abs(1000*100) = -10_000`. `new_exposure = 90_000 + (-10_000) = 80_000 < 100_000` → approved. Old code would give `new_exposure = 90_000 + 10_000 = 100_000` → edge of violation or false flag.
- Run `python3 -m pytest tests/ -q` — expect 173 passed, 4 skipped.
