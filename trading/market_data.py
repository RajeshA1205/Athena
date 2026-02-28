"""
Market Data Module
==================
Provides OHLCV data structures and a MarketDataFeed for real-time and historical
market data, with MOCK and LIVE operating modes. MOCK mode generates deterministic
synthetic price series using a seeded random walk; LIVE mode is reserved for
future integration with external data providers.
"""

import asyncio
import hashlib
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


def _stable_hash(s: str) -> int:
    """Return a stable (PYTHONHASHSEED-independent) integer hash of a string."""
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)

logger = logging.getLogger(__name__)


@dataclass
class OHLCV:
    """
    A single OHLCV (Open/High/Low/Close/Volume) price bar.

    Attributes:
        symbol: Ticker symbol (e.g. "AAPL")
        timestamp: ISO-8601 UTC timestamp for the bar open
        open: Opening price
        high: Highest price during the interval
        low: Lowest price during the interval
        close: Closing price
        volume: Total volume traded during the interval
        interval: Bar interval string (e.g. "1d", "1h")
    """

    symbol: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval: str = "1d"


class MarketDataMode(Enum):
    """Operating mode for MarketDataFeed."""

    MOCK = "mock"
    LIVE = "live"


class MarketDataFeed:
    """
    Provides OHLCV market data in MOCK or LIVE mode.

    In MOCK mode the feed generates fully deterministic synthetic price series
    seeded per symbol, suitable for development, backtesting, and unit testing
    without any external dependencies.

    In LIVE mode the feed is a stub that logs a warning; real provider
    integration is deferred to a future task.
    """

    MOCK_SYMBOLS: List[str] = [
        "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "SPY", "QQQ"
    ]

    def __init__(
        self,
        mode: MarketDataMode = MarketDataMode.MOCK,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the market data feed.

        Args:
            mode: Operating mode — MOCK or LIVE.
            config: Optional configuration dictionary (reserved for future use).
        """
        self.mode = mode
        self.config = config or {}
        self._subscribers: Dict[str, Dict[str, Callable]] = {}
        self._next_sub_id: int = 0
        self._mock_data: Dict[str, List[OHLCV]] = {}
        self._streaming: bool = False
        logger.info("MarketDataFeed initialized in %s mode", self.mode.value)

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def get_realtime_data(self, symbol: str) -> Optional[OHLCV]:
        """
        Fetch the most recent OHLCV bar for a symbol.

        In MOCK mode returns the last bar of a freshly-generated single-day
        series. In LIVE mode logs a warning and returns None.

        Args:
            symbol: Ticker symbol to fetch.

        Returns:
            Most recent OHLCV bar, or None on error or in LIVE mode.
        """
        try:
            if self.mode == MarketDataMode.LIVE:
                logger.warning("Live mode not implemented")
                return None
            bars = self._generate_mock_data(symbol, days=1)
            return bars[-1] if bars else None
        except Exception as e:
            logger.error("get_realtime_data failed for %r: %s", symbol, e)
            return None

    async def get_historical_data(
        self, symbol: str, days: int = 30, interval: str = "1d"
    ) -> List[OHLCV]:
        """
        Fetch historical OHLCV bars for a symbol.

        In MOCK mode returns a deterministic synthetic series. In LIVE mode
        logs a warning and returns an empty list.

        Args:
            symbol: Ticker symbol to fetch.
            days: Number of bars to return.
            interval: Bar interval string (e.g. "1d", "1h").

        Returns:
            List of OHLCV bars ordered oldest-first, or [] in LIVE mode.
        """
        if self.mode == MarketDataMode.LIVE:
            logger.warning("Live mode not implemented")
            return []
        return self._generate_mock_data(symbol, days=days, interval=interval)

    async def subscribe(self, symbol: str, callback: Callable) -> str:
        """
        Register a callback to receive streaming ticks for a symbol.

        Args:
            symbol: Ticker symbol to subscribe to.
            callback: Async or sync callable invoked with each new OHLCV bar.

        Returns:
            Subscription ID string that can be passed to unsubscribe().
        """
        if symbol not in self._subscribers:
            self._subscribers[symbol] = {}
        subscription_id = f"sub_{self._next_sub_id:08d}"
        self._next_sub_id += 1
        self._subscribers[symbol][subscription_id] = callback
        logger.debug("Subscribed to %r with id %r", symbol, subscription_id)
        return subscription_id

    async def unsubscribe(self, symbol: str, subscription_id: str) -> bool:
        """
        Remove a previously registered callback by subscription ID.

        Args:
            symbol: Ticker symbol the subscription belongs to.
            subscription_id: ID returned by subscribe().

        Returns:
            True if the callback was found and removed, False otherwise.
        """
        if symbol not in self._subscribers:
            logger.warning("unsubscribe: no subscribers found for %r", symbol)
            return False
        callbacks = self._subscribers[symbol]
        if subscription_id in callbacks:
            del callbacks[subscription_id]
            logger.debug("Unsubscribed %r from %r", subscription_id, symbol)
            return True
        logger.warning(
            "unsubscribe: subscription_id %r not found for symbol %r",
            subscription_id, symbol,
        )
        return False

    async def stream_start(
        self, symbols: List[str], interval_seconds: float = 1.0
    ) -> None:
        """
        Start a mock streaming loop that emits synthetic ticks per symbol.

        Loops until stream_stop() is called. Each iteration generates one
        mock OHLCV bar per symbol and invokes all registered callbacks.

        Args:
            symbols: List of ticker symbols to stream.
            interval_seconds: Seconds to sleep between tick batches.
        """
        self._streaming = True
        logger.info("Streaming started for %s at %.1fs intervals", symbols, interval_seconds)
        while self._streaming:
            for symbol in symbols:
                if not self._streaming:
                    break
                if self.mode == MarketDataMode.MOCK:
                    bars = self._generate_mock_data(symbol, days=1)
                    tick = bars[-1] if bars else None
                    if tick and symbol in self._subscribers:
                        for callback in list(self._subscribers[symbol].values()):
                            try:
                                result = callback(tick)
                                if asyncio.iscoroutine(result):
                                    await result
                            except Exception as e:
                                logger.warning(
                                    "Subscriber callback error for %r: %s", symbol, e
                                )
            await asyncio.sleep(interval_seconds)

    async def stream_stop(self) -> None:
        """Stop the streaming loop started by stream_start()."""
        self._streaming = False
        logger.info("Streaming stopped")

    async def get_symbols(self) -> List[str]:
        """
        Return the list of available symbols.

        In MOCK mode returns the built-in universe of 7 symbols.
        In LIVE mode returns an empty list (provider integration pending).

        Returns:
            List of ticker symbol strings.
        """
        if self.mode == MarketDataMode.MOCK:
            return list(self.MOCK_SYMBOLS)
        return []

    def get_stats(self) -> Dict[str, Any]:
        """Return operational statistics for the feed."""
        return {
            "mode": self.mode.value,
            "subscriber_counts": {
                symbol: len(cbs) for symbol, cbs in self._subscribers.items()
            },
            "mock_cache_size": len(self._mock_data),
            "streaming": self._streaming,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_mock_data(
        self, symbol: str, days: int = 30, interval: str = "1d"
    ) -> List[OHLCV]:
        """
        Generate a deterministic random-walk OHLCV series for a symbol.

        Results are cached in _mock_data. The random seed is derived from the
        symbol name so the same symbol always produces the same price series.

        Args:
            symbol: Ticker symbol used to seed the random walk.
            days: Number of bars to generate.
            interval: Bar interval label stored in each OHLCV record.

        Returns:
            List of OHLCV bars ordered oldest-first.
        """
        cache_key = f"{symbol}_{days}_{interval}"
        if cache_key in self._mock_data:
            return self._mock_data[cache_key]

        # Use a local RNG seeded by a stable hash to avoid corrupting global RNG state
        rng = random.Random(_stable_hash(symbol) % 2**32)

        # Base price in [50, 500] derived from symbol characters
        char_sum = sum(ord(c) for c in symbol if c.isalpha())
        base_price: float = 50.0 + (char_sum % 451)

        bars: List[OHLCV] = []
        prev_close = base_price

        for i in range(days):
            close = prev_close * (1 + rng.gauss(0.0002, 0.015))
            open_price = prev_close * (1 + rng.gauss(0, 0.003))
            high = max(open_price, close) * (1 + abs(rng.gauss(0, 0.005)))
            low = min(open_price, close) * (1 - abs(rng.gauss(0, 0.005)))
            volume = float(rng.randint(500_000, 5_000_000))
            timestamp = (
                datetime.now(timezone.utc) - timedelta(days=days - i)
            ).isoformat()

            bars.append(
                OHLCV(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=open_price,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                    interval=interval,
                )
            )
            prev_close = close

        self._mock_data[cache_key] = bars
        return bars
