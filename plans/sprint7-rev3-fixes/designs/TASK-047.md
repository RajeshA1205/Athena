# TASK-047: Wire MarketDataFeed into main.py Paper-Trade/Backtest Loops

## Summary

The paper-trade and backtest loops in `main.py` (lines 103-114) create `AgentContext` with only a task string and no market data. `MarketDataFeed` exists in `trading/market_data.py` with a working MOCK mode but is never imported or used. This makes paper-trade/backtest modes effectively inert -- agents receive no market information.

## Current State

**File:** `/Users/rajesh/athena/main.py`, lines 103-114

```python
# paper-trade / backtest: placeholder for event loop
logger.info("Mode '%s' main loop -- press Ctrl+C to stop", mode)
iteration = 0
try:
    while True:
        iteration += 1
        context = AgentContext(task=f"market cycle {iteration}")
        thought = await coordinator.think(context)
        await coordinator.act(thought)
        await asyncio.sleep(1.0)  # 1-second tick
except asyncio.CancelledError:
    logger.info("Main loop cancelled after %d iterations", iteration)
```

Issues:
1. `context.metadata` is empty -- no market data for any agent to analyze.
2. Both paper-trade and backtest use the same infinite loop. Backtest should replay finite historical data and terminate.
3. `MarketDataFeed` (at `/Users/rajesh/athena/trading/market_data.py`) is never imported.

**MarketDataFeed API** (from `trading/market_data.py`):
- `get_realtime_data(symbol) -> Optional[OHLCV]` -- returns latest bar (MOCK mode).
- `get_historical_data(symbol, days, interval) -> List[OHLCV]` -- returns historical bars.
- `_MOCK_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "SPY", "QQQ"]`
- `OHLCV` is a dataclass with: symbol, timestamp, open, high, low, close, volume, interval.

## Proposed Change

Modify `/Users/rajesh/athena/main.py` only.

### 1. Add imports (inside `run()`, after line 79)

```python
from trading.market_data import MarketDataFeed, MarketDataMode
from dataclasses import asdict
```

### 2. Instantiate MarketDataFeed (after line 93, after agent registration)

```python
# Market data feed (MOCK mode for paper-trade and backtest)
feed = MarketDataFeed(mode=MarketDataMode.MOCK)
symbols = config.trading.markets if config.trading.markets != ["stocks"] else feed._MOCK_SYMBOLS
```

Note: `config.trading.markets` defaults to `["stocks"]` which is not a symbol list. If the user has not overridden it with actual symbols, fall back to `_MOCK_SYMBOLS`.

### 3. Replace the paper-trade/backtest block (lines 103-114)

```python
if mode == "paper-trade":
    logger.info("Paper-trade loop -- press Ctrl+C to stop")
    iteration = 0
    try:
        while True:
            iteration += 1
            # Fetch latest market data for all symbols
            market_data = {}
            for symbol in symbols:
                bar = await feed.get_realtime_data(symbol)
                if bar is not None:
                    market_data[symbol] = asdict(bar)

            context = AgentContext(
                task=f"market cycle {iteration}",
                metadata={"market_data": market_data, "symbols": symbols},
            )
            thought = await coordinator.think(context)
            await coordinator.act(thought)
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        logger.info("Paper-trade loop cancelled after %d iterations", iteration)

elif mode == "backtest":
    backtest_days = 30  # Default backtest window
    logger.info("Backtest mode: replaying %d days of historical data", backtest_days)

    # Pre-fetch historical data for all symbols
    historical: Dict[str, list] = {}
    for symbol in symbols:
        bars = await feed.get_historical_data(symbol, days=backtest_days)
        historical[symbol] = [asdict(bar) for bar in bars]

    # Replay bar-by-bar
    num_bars = max((len(bars) for bars in historical.values()), default=0)
    for i in range(num_bars):
        market_data = {}
        for symbol in symbols:
            bars = historical[symbol]
            if i < len(bars):
                market_data[symbol] = bars[i]

        context = AgentContext(
            task=f"backtest bar {i + 1}/{num_bars}",
            metadata={"market_data": market_data, "symbols": symbols, "bar_index": i},
        )
        thought = await coordinator.think(context)
        await coordinator.act(thought)

    logger.info("Backtest complete: replayed %d bars", num_bars)
```

### 4. Add `Dict` import at top of `run()` if not already imported

The function already uses `from typing import Optional` at the module level. `Dict` needs to be added for the type hint on `historical`. Alternatively, use a bare dict annotation or no annotation.

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `/Users/rajesh/athena/main.py` | After line 79 | Add `MarketDataFeed`, `MarketDataMode`, `asdict` imports |
| `/Users/rajesh/athena/main.py` | After line 93 | Instantiate `feed` and resolve `symbols` |
| `/Users/rajesh/athena/main.py` | Lines 103-114 | Replace with separate paper-trade and backtest logic |

## Acceptance Criteria

- `python main.py --mode dry-run` behavior is unchanged (single think-act cycle, no MarketDataFeed).
- `python main.py --mode paper-trade` creates a `MarketDataFeed` in MOCK mode, populates `context.metadata["market_data"]` with a dict mapping symbol to OHLCV dict each iteration, and loops indefinitely until SIGINT.
- `python main.py --mode backtest` pre-fetches 30 days of historical data, replays bar-by-bar through the coordinator, and terminates when all bars are processed.
- `context.metadata["market_data"]` contains at least one symbol with keys: `symbol`, `timestamp`, `open`, `high`, `low`, `close`, `volume`, `interval`.
- `context.metadata["symbols"]` is the list of symbols being tracked.
- All 171 tests pass, 6 skipped.

## Edge Cases & Risks

1. **`config.trading.markets` content**: The default is `["stocks"]`, which is a category label not a ticker symbol. The implementation checks for this default and falls back to `_MOCK_SYMBOLS`. If the user provides actual symbols (e.g., `["AAPL", "MSFT"]`), those are used directly. Risk: user might provide a mix of categories and symbols. Mitigation: document that `markets` should be ticker symbols when used with paper-trade/backtest.

2. **Empty historical data**: If `get_historical_data` returns an empty list for a symbol, that symbol is simply absent from `market_data` for that bar. `max(... default=0)` handles the case where all symbols return empty lists.

3. **Backtest window configurability**: Hardcoding 30 days is a simplification. A future enhancement could read from `config` or CLI args. For now, 30 is reasonable for MOCK mode.

4. **Performance**: In paper-trade mode, `get_realtime_data()` is called for each symbol sequentially. With 7 mock symbols this is instant. For LIVE mode (future), consider `asyncio.gather()`.

5. **`asdict` on OHLCV**: The OHLCV dataclass is simple (all primitive types), so `asdict()` is safe with no recursive complexity.

## Test Notes

- Existing tests do not exercise `main.py` modes beyond dry-run (the dry-run path is unchanged).
- Manual smoke test: `python3 main.py --mode paper-trade` should print log lines showing market data population. Ctrl+C to stop.
- Manual smoke test: `python3 main.py --mode backtest` should complete and log "Backtest complete: replayed N bars".
- No new unit tests are strictly required since this is integration wiring, but a test could mock `MarketDataFeed` and verify `context.metadata` population.
