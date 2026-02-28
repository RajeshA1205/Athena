# TASK-054: Replace hash() with _stable_hash() in scrape_macro_indicators

## Summary

`scrape_macro_indicators()` in `training/data/scrapers/market.py` uses `hash(indicator)` at line 202 for the fallback value of unknown macro indicators. `hash()` is non-deterministic across Python processes (depends on `PYTHONHASHSEED`). The file already defines `_stable_hash()` at line 20 using SHA-256. Replace `hash()` with `_stable_hash()`.

## Current State

**File:** `/Users/rajesh/athena/training/data/scrapers/market.py`, line 202

```python
value = mock_values.get(indicator, round(1.0 + hash(indicator) % 10, 2))
```

The `_stable_hash` function is already defined at line 20 in the same file:

```python
def _stable_hash(s: str) -> int:
    """Return a stable, process-invariant hash using SHA-256."""
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)
```

Other functions in the same file already use `_stable_hash`:
- `_mock_ohlcv()` at line 118: `rng = random.Random(_stable_hash(symbol) % 2**32)`
- `scrape_fundamentals()` at lines 162-166: uses `_stable_hash(symbol)` for all fundamental values.

## Proposed Change

Modify `/Users/rajesh/athena/training/data/scrapers/market.py` only.

### Change line 202

**Before:**
```python
value = mock_values.get(indicator, round(1.0 + hash(indicator) % 10, 2))
```

**After:**
```python
value = mock_values.get(indicator, round(1.0 + _stable_hash(indicator) % 10, 2))
```

One word change: `hash` -> `_stable_hash`.

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `/Users/rajesh/athena/training/data/scrapers/market.py` | 202 | `hash(indicator)` -> `_stable_hash(indicator)` |

## Acceptance Criteria

- Line 202 uses `_stable_hash(indicator)` instead of `hash(indicator)`.
- Output of `scrape_macro_indicators()` is deterministic across Python processes (same values regardless of `PYTHONHASHSEED`).
- No other uses of `hash()` for value generation remain in this file.
- All 171 tests pass, 6 skipped.

## Edge Cases & Risks

1. **Value change**: The fallback values for unknown indicators will change because `_stable_hash` produces different integers than `hash`. This is acceptable -- the values are mock data and no test or downstream code depends on specific mock indicator values.

2. **Modulo arithmetic**: `_stable_hash` returns a very large integer (SHA-256 = 256 bits). `% 10` works correctly on large Python integers, producing a value in [0, 9]. `1.0 + result` is in [1.0, 10.0], rounded to 2 decimal places. This matches the original intent.

3. **No other `hash()` calls**: Verify no other `hash()` calls exist in this file. The only `hash()` usage is on line 202.

## Test Notes

- Verify determinism: run `scrape_macro_indicators(["unknown_indicator"])` twice in separate Python processes and confirm identical output.
- Run `python3 -m pytest tests/ -q` to confirm all tests pass.
- This is a one-word mechanical change with zero risk of regression.
