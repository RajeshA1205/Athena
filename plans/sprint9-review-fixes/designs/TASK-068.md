# TASK-068: Fix main.py Market Data Format Mismatch with Agent Interface

## Summary

`main.py` paper-trade and backtest modes build `context.metadata["market_data"]` as a dict-of-dicts keyed by symbol: `{"AAPL": {"open": ..., "high": ..., "close": ...}}`. But `MarketAnalystAgent.think()` reads `market_data.get("prices", [])` expecting a flat structure with a `"prices"` key containing a list of closing prices. The result is that `price_data` is always `[]`, all technical analysis is skipped with "insufficient data", and both paper-trade and backtest modes are functionally inoperative. Fix: restructure `main.py` to run the coordinator once per symbol per bar with the flat format the agent expects.

## Current State

**File:** `main.py`, paper-trade loop (lines 113-125):

```python
                market_data: Dict[str, dict] = {}
                for symbol in symbols:
                    bar = await feed.get_realtime_data(symbol)
                    if bar is not None:
                        market_data[symbol] = asdict(bar)   # {"AAPL": {open, high, low, close, ...}}

                context = AgentContext(
                    task=f"market cycle {iteration}",
                    metadata={"market_data": market_data, "symbols": symbols},
                )
                thought = await coordinator.think(context)
                await coordinator.act(thought)
```

**File:** `agents/market_analyst.py`, lines 120-122:

```python
        market_data = context.metadata.get("market_data", {})
        price_data = market_data.get("prices", [])   # always [] with current main.py format
        text_data = market_data.get("news", [])
```

The agent expects: `{"symbol": "AAPL", "prices": [100.0, 101.5, ...], "news": [...]}`.
A `MarketDataBar` from `asdict(bar)` contains: `{"symbol": "AAPL", "open": ..., "high": ..., "low": ..., "close": ..., "volume": ..., "timestamp": ...}`.

## Proposed Change

Restructure both the paper-trade and backtest loops in `main.py` to run the coordinator once per symbol per bar, passing a per-symbol context with the format the agent expects.

### Paper-trade loop (replaces lines 113-125)

```python
        while True:
            iteration += 1
            for symbol in symbols:
                bar = await feed.get_realtime_data(symbol)
                if bar is None:
                    continue
                bar_dict = asdict(bar)
                context = AgentContext(
                    task=f"market cycle {iteration}: {symbol}",
                    metadata={
                        "market_data": {
                            "symbol": symbol,
                            "prices": [bar_dict["close"]],
                            "bar": bar_dict,
                        },
                        "symbol": symbol,
                    },
                )
                thought = await coordinator.think(context)
                await coordinator.act(thought)
            await asyncio.sleep(1.0)
```

### Backtest loop (replaces lines 144-156)

```python
        for i in range(num_bars):
            for symbol in symbols:
                bars = historical[symbol]
                if i >= len(bars):
                    continue
                bar_dict = bars[i]
                # Build a rolling price window for technical analysis
                window = [b["close"] for b in bars[: i + 1]]
                context = AgentContext(
                    task=f"backtest bar {i + 1}/{num_bars}: {symbol}",
                    metadata={
                        "market_data": {
                            "symbol": symbol,
                            "prices": window,
                            "bar": bar_dict,
                        },
                        "symbol": symbol,
                        "bar_index": i,
                    },
                )
                thought = await coordinator.think(context)
                await coordinator.act(thought)
```

Note: the backtest now passes a growing `window` of closing prices, giving `MarketAnalystAgent` enough history to compute indicators (SMA-20 needs 20+ bars, which is reached after bar 20).

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `main.py` | ~113-125 (paper-trade inner loop) | Per-symbol iteration with flat market_data format |
| `main.py` | ~144-156 (backtest inner loop) | Per-symbol iteration with rolling price window |

## Acceptance Criteria

- `context.metadata["market_data"]` always has a `"prices"` key containing a non-empty list.
- `context.metadata["market_data"]["symbol"]` contains the ticker string.
- `MarketAnalystAgent.think()` reaches the technical analysis code path (does not short-circuit with "insufficient data" after bar 50).
- `python3 main.py --mode dry-run` still works cleanly (dry-run path unchanged).
- 173 tests pass, 4 skipped.

## Edge Cases & Risks

1. **Single-bar price window in paper-trade**: Paper-trade only passes `[bar_dict["close"]]` (one price). `MarketAnalystAgent` needs 50+ prices for most indicators. This is correct for a live streaming scenario — indicators build up over time as the loop runs. The agent will log "insufficient data" for the first ~50 bars per symbol, then start producing analysis.
2. **Per-symbol coordinator calls**: The coordinator is now called once per symbol per bar instead of once per bar for all symbols. This multiplies the number of coordinator `think+act` cycles by `len(symbols)`. For 7 MOCK_SYMBOLS and 30 days × bars per day, this is still fast in mock mode.
3. **Backtest window grows**: Passing `bars[:i+1]` gives the full historical window, not a fixed lookback. This is intentional — it gives the agent maximum context. For a fixed lookback, change to `bars[max(0, i-99):i+1]`.
4. **`bar` key in metadata**: The raw OHLCV bar dict is included under `"bar"` for agents that need open/high/low/volume (e.g., risk agent). This is additive and does not break existing agent code that only reads `"prices"`.

## Test Notes

- Run `python3 main.py --mode dry-run` — should complete without error.
- Inspect `context.metadata["market_data"]` structure in a debug run to confirm `"prices"` key is present.
- Run `python3 -m pytest tests/ -q` — expect 173 passed, 4 skipped.
