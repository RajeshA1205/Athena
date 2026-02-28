# TASK-060: Fix MarketDataFeed Placement and Private Attribute Access in main.py

## Summary

Two issues in `main.py`:
1. `feed = MarketDataFeed(...)` and symbol resolution happen at lines 98-99, before the `if mode == "dry-run"` branch. In dry-run mode the feed is instantiated but never used.
2. `feed._MOCK_SYMBOLS` accesses a private class attribute from outside the class.

## Current State

**File:** `/Users/rajesh/athena/main.py`, lines 97-99

```python
    # Market data feed (MOCK mode for paper-trade and backtest)
    feed = MarketDataFeed(mode=MarketDataMode.MOCK)
    symbols = config.trading.markets if config.trading.markets != ["stocks"] else feed._MOCK_SYMBOLS

    if mode == "dry-run":
        # Single think-act cycle with empty context to confirm wiring
        context = AgentContext(task="dry-run health check")
        thought = await coordinator.think(context)
        action = await coordinator.act(thought)
        logger.info("Dry-run complete. Action type: %s", action.action_type)
        return

    if mode == "paper-trade":
        ...
        for symbol in symbols:
            bar = await feed.get_realtime_data(symbol)
        ...

    elif mode == "backtest":
        ...
        for symbol in symbols:
            bars = await feed.get_historical_data(symbol, days=backtest_days)
        ...
```

**File:** `/Users/rajesh/athena/trading/market_data.py`, lines 72-74 and 237-240

```python
    _MOCK_SYMBOLS: List[str] = [
        "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "SPY", "QQQ"
    ]

    def get_available_symbols(self) -> List[str]:
        if self.mode == MarketDataMode.MOCK:
            return list(self._MOCK_SYMBOLS)
        return []
```

`_MOCK_SYMBOLS` is a class attribute with a leading underscore (private). The public `get_available_symbols()` method already exposes it.

## Proposed Change

### 1. Make `_MOCK_SYMBOLS` public in `trading/market_data.py`

Rename `_MOCK_SYMBOLS` to `MOCK_SYMBOLS` (remove leading underscore). This is a class-level constant that is legitimately useful to callers. Update the internal reference in `get_available_symbols()` too.

```python
# BEFORE
    _MOCK_SYMBOLS: List[str] = [
        "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "SPY", "QQQ"
    ]

    def get_available_symbols(self) -> List[str]:
        if self.mode == MarketDataMode.MOCK:
            return list(self._MOCK_SYMBOLS)
        return []

# AFTER
    MOCK_SYMBOLS: List[str] = [
        "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "SPY", "QQQ"
    ]

    def get_available_symbols(self) -> List[str]:
        if self.mode == MarketDataMode.MOCK:
            return list(self.MOCK_SYMBOLS)
        return []
```

### 2. Refactor `main.py`: move feed inside branches, use public attribute

Remove lines 97-99. Inside each branch, instantiate the feed and resolve symbols locally:

```python
    if mode == "dry-run":
        # Single think-act cycle — no feed needed
        context = AgentContext(task="dry-run health check")
        thought = await coordinator.think(context)
        action = await coordinator.act(thought)
        logger.info("Dry-run complete. Action type: %s", action.action_type)
        return

    if mode == "paper-trade":
        feed = MarketDataFeed(mode=MarketDataMode.MOCK)
        symbols = config.trading.markets if config.trading.markets != ["stocks"] else MarketDataFeed.MOCK_SYMBOLS
        logger.info("Paper-trade loop -- press Ctrl+C to stop")
        ...

    elif mode == "backtest":
        feed = MarketDataFeed(mode=MarketDataMode.MOCK)
        symbols = config.trading.markets if config.trading.markets != ["stocks"] else MarketDataFeed.MOCK_SYMBOLS
        ...
```

`MarketDataFeed.MOCK_SYMBOLS` is a class attribute access (no instance needed for symbol resolution), so it can be used before `feed` is instantiated if needed.

## Files Modified

| File | Line(s) | Change |
|------|---------|--------|
| `trading/market_data.py` | ~72 | Rename `_MOCK_SYMBOLS` → `MOCK_SYMBOLS` |
| `trading/market_data.py` | ~239 | Update internal `self._MOCK_SYMBOLS` → `self.MOCK_SYMBOLS` |
| `main.py` | 97-99 | Remove feed instantiation and symbol resolution from global scope |
| `main.py` | `paper-trade` branch | Add `feed = MarketDataFeed(...)` and `symbols = ...` |
| `main.py` | `backtest` branch | Add `feed = MarketDataFeed(...)` and `symbols = ...` |

## Acceptance Criteria

- `MarketDataFeed` is not instantiated in `main.py` before the mode branch.
- `_MOCK_SYMBOLS` (with leading underscore) no longer exists in `trading/market_data.py`.
- `MOCK_SYMBOLS` (public) is accessible as `MarketDataFeed.MOCK_SYMBOLS`.
- `main.py --mode dry-run` runs cleanly without instantiating a feed.
- `main.py --mode paper-trade` and `--mode backtest` still work correctly (feed used as before).
- All 173 tests pass, 4 skipped.

## Edge Cases & Risks

1. **`_MOCK_SYMBOLS` rename is a breaking change**: Any external code referencing `feed._MOCK_SYMBOLS` would break. In this codebase, only `main.py` references it. Verify with a grep before renaming.

2. **Duplicate `feed = ...` lines**: After the fix, `feed` is instantiated twice (once in paper-trade, once in backtest). This is fine — they're in separate branches and the feed is a lightweight mock object.

3. **`config.trading.markets` sentinel**: The check `config.trading.markets != ["stocks"]` is the sentinel for "no explicit markets configured". This is pre-existing; TASK-060 does not change this logic.

## Test Notes

- Run `python3 main.py --mode dry-run` and confirm no `MarketDataFeed` instantiation occurs (add a debug print temporarily if needed).
- Run `python3 -m pytest tests/ -q` to confirm 173 passed, 4 skipped.
- Grep for `_MOCK_SYMBOLS` after the change — should find zero results.
