# TASK-031: Optimize MACD calculation to O(n)

## Summary
`MarketAnalystAgent._calculate_indicators()` builds the MACD series by calling `_calculate_ema(price_data[:i], 12)` and `_calculate_ema(price_data[:i], 26)` inside a loop from index 26 to `len(price_data)`. Each call recomputes the full EMA from scratch on a growing slice, making the algorithm O(n²) in the number of price bars. For 252 daily bars (one trading year) this means ~63,000 EMA recomputations per indicator update call. The fix is to compute the full EMA-12 and EMA-26 series in a single forward pass each using the standard incremental formula, then derive MACD by subtraction.

## Current State

**File:** `agents/market_analyst.py` (lines 338–354)

```python
if indicators.ema_12 is not None and indicators.ema_26 is not None:
    # Build MACD series across price history for proper signal line
    macd_series = []
    for i in range(26, len(price_data) + 1):
        ema12 = self._calculate_ema(price_data[:i], 12)   # O(i) each iteration
        ema26 = self._calculate_ema(price_data[:i], 26)   # O(i) each iteration
        macd_series.append(ema12 - ema26)

    indicators.macd = macd_series[-1]
    if len(macd_series) >= 9:
        indicators.macd_signal = self._calculate_ema(macd_series, 9)
    else:
        indicators.macd_signal = sum(macd_series) / len(macd_series)
    indicators.macd_histogram = indicators.macd - indicators.macd_signal
```

## Proposed Change

Replace the O(n²) loop with a single-pass incremental EMA computation:

```python
if len(price_data) >= 26:
    # Compute EMA-12 and EMA-26 series in a single O(n) pass each
    alpha12 = 2.0 / (12 + 1)
    alpha26 = 2.0 / (26 + 1)

    ema12_series: List[float] = []
    ema26_series: List[float] = []
    ema12_val = price_data[0]
    ema26_val = price_data[0]

    for price in price_data:
        ema12_val = alpha12 * price + (1 - alpha12) * ema12_val
        ema26_val = alpha26 * price + (1 - alpha26) * ema26_val
        ema12_series.append(ema12_val)
        ema26_series.append(ema26_val)

    macd_series = [e12 - e26 for e12, e26 in zip(ema12_series[25:], ema26_series[25:])]

    indicators.macd = macd_series[-1]
    if len(macd_series) >= 9:
        indicators.macd_signal = self._calculate_ema(macd_series, 9)
    else:
        indicators.macd_signal = sum(macd_series) / len(macd_series)
    indicators.macd_histogram = indicators.macd - indicators.macd_signal

    # Also update the scalar EMA fields from the series
    indicators.ema_12 = ema12_series[-1]
    indicators.ema_26 = ema26_series[-1]
```

Note: The seed value `ema12_val = price_data[0]` (simple initialization) produces slightly different initial values compared to the old `_calculate_ema` which also uses simple initialization (`ema = price_data[0]`). After 26+ bars, both methods converge to the same values. The difference is only in the warm-up period.

## Files Modified

- `agents/market_analyst.py`
  - Lines 338–354: replace the O(n²) loop block with the incremental implementation above
  - The existing `indicators.ema_12` / `indicators.ema_26` assignments at lines 335–339 can be removed or kept (they will be overwritten by the new block)

## Acceptance Criteria

- [ ] MACD value (`indicators.macd`) is numerically equal (within 1e-6 tolerance) to the old implementation for inputs of ≥ 50 price bars
- [ ] Time complexity is O(n) — no nested loop over price data
- [ ] Signal line and histogram still computed correctly
- [ ] All existing tests pass

## Edge Cases & Risks

- **Warm-up period difference:** For the first 26 bars, the incremental EMA uses a different initialization than repeated `_calculate_ema` calls. This is acceptable — both are approximations; the incremental method is actually the standard industry approach.
- **`price_data` with fewer than 26 bars:** The `if len(price_data) >= 26` guard is unchanged. If fewer bars are available, `macd_series` is not computed (same behavior as before).
- **`_calculate_ema` still used for signal line:** The 9-period signal line EMA reuses the existing `_calculate_ema(macd_series, 9)` call — no change needed there.

## Test Notes

- Existing tests in `test_agents.py` call `think()` with mock data; they assert the return type but not specific MACD values. These will pass unchanged.
- Add a numerical correctness test: generate 100 price bars, compute MACD with old and new methods, assert results agree within 1e-6 for the last 74 values (after 26-bar warm-up).
