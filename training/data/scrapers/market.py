"""
Market Data Scraper
===================
Scrapes OHLCV price data, fundamental data, and macro indicators.
Uses yfinance when available; generates deterministic mock data otherwise.

NOTE: Respect the terms of service for any external data providers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional


def _stable_hash(s: str) -> int:
    """Return a stable, process-invariant hash using SHA-256."""
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

logger = logging.getLogger(__name__)


class MarketScraper:
    """
    Scraper for market OHLCV data, fundamentals, and macro indicators.

    Attempts live data via yfinance when installed; falls back to deterministic
    mock data on ImportError or network failures.

    Args:
        config: Optional configuration dict with keys:
            - rate_limit_delay (float): Seconds between requests (default 0.5)
            - max_retries (int): Request retries (default 3)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.rate_limit_delay: float = self.config.get("rate_limit_delay", 0.5)
        self.max_retries: int = self.config.get("max_retries", 3)
        self._request_count: int = 0
        self.logger = logging.getLogger(__name__)

    async def scrape(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Scrape OHLCV data for a ticker symbol.

        Args:
            query: Ticker symbol (e.g. "AAPL").
            limit: Number of OHLCV bars to return.

        Returns:
            List of ScrapedItem-compatible dicts with OHLCV content.
        """
        return await self.scrape_ohlcv(query, period="1mo")

    async def scrape_ohlcv(
        self, symbol: str, period: str = "1y"
    ) -> List[Dict[str, Any]]:
        """
        Scrape historical OHLCV bars for a symbol.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").
            period: History period string (e.g. "1y", "6mo", "1mo").

        Returns:
            List of ScrapedItem-compatible dicts, one per bar.
        """
        if HAS_YFINANCE:
            try:
                return await self._live_ohlcv(symbol, period)
            except Exception as e:
                self.logger.warning(
                    "yfinance OHLCV fetch failed for %r (%s), using mock", symbol, e
                )
        return self._mock_ohlcv(symbol, period)

    async def _live_ohlcv(self, symbol: str, period: str) -> List[Dict[str, Any]]:
        """Fetch live OHLCV via yfinance."""
        self._request_count += 1
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        items = []
        for ts, row in hist.iterrows():
            ohlcv = {
                "date": str(ts.date()),
                "open": float(row.get("Open", 0)),
                "high": float(row.get("High", 0)),
                "low": float(row.get("Low", 0)),
                "close": float(row.get("Close", 0)),
                "volume": float(row.get("Volume", 0)),
            }
            items.append({
                "source": "yfinance",
                "content_type": "market",
                "content": json.dumps(ohlcv),
                "metadata": {"symbol": symbol, "period": period},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "url": None,
            })
        return items

    def _mock_ohlcv(self, symbol: str, period: str) -> List[Dict[str, Any]]:
        """Generate deterministic mock OHLCV bars seeded by symbol."""
        period_days = {"1d": 1, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 252}.get(
            period, 60
        )
        rng = random.Random(_stable_hash(symbol) % 2**32)
        char_sum = sum(ord(c) for c in symbol if c.isalpha())
        base_price = 50.0 + (char_sum % 451)

        items = []
        prev = base_price
        for i in range(period_days):
            close = prev * (1 + rng.gauss(0.0002, 0.015))
            open_p = prev * (1 + rng.gauss(0, 0.003))
            high = max(open_p, close) * (1 + abs(rng.gauss(0, 0.005)))
            low = min(open_p, close) * (1 - abs(rng.gauss(0, 0.005)))
            volume = rng.randint(500_000, 5_000_000)
            date = (datetime.now(timezone.utc) - timedelta(days=period_days - i)).date()
            ohlcv = {
                "date": str(date),
                "open": round(open_p, 4),
                "high": round(high, 4),
                "low": round(low, 4),
                "close": round(close, 4),
                "volume": volume,
            }
            items.append({
                "source": "mock_market",
                "content_type": "market",
                "content": json.dumps(ohlcv),
                "metadata": {"symbol": symbol, "period": period},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "url": None,
            })
            prev = close
        return items

    async def scrape_fundamentals(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Scrape fundamental data for a symbol.

        Args:
            symbol: Ticker symbol.

        Returns:
            Single-element list with fundamental metrics as a ScrapedItem dict.
        """
        fundamentals = {
            "symbol": symbol,
            "pe_ratio": round(15.0 + _stable_hash(symbol) % 30, 2),
            "eps": round(1.5 + _stable_hash(symbol) % 5, 2),
            "market_cap_billions": round(10.0 + _stable_hash(symbol) % 2000, 1),
            "dividend_yield": round(0.5 + (_stable_hash(symbol) % 30) / 10, 2),
            "debt_to_equity": round(0.2 + (_stable_hash(symbol) % 20) / 10, 2),
        }
        return [{
            "source": "mock_fundamentals",
            "content_type": "market",
            "content": json.dumps(fundamentals),
            "metadata": {"symbol": symbol, "data_type": "fundamentals"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "url": None,
        }]

    async def scrape_macro_indicators(
        self, indicators: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Scrape macroeconomic indicators.

        Args:
            indicators: List of indicator names (defaults to GDP, CPI, Fed rate).

        Returns:
            List of ScrapedItem dicts, one per indicator.
        """
        if indicators is None:
            indicators = ["GDP_growth", "CPI_YoY", "Fed_funds_rate"]

        mock_values = {
            "GDP_growth": 2.8,
            "CPI_YoY": 3.1,
            "Fed_funds_rate": 5.25,
            "unemployment_rate": 3.7,
            "10y_treasury_yield": 4.3,
        }

        items = []
        for indicator in indicators:
            value = mock_values.get(indicator, round(1.0 + _stable_hash(indicator) % 10, 2))
            items.append({
                "source": "mock_macro",
                "content_type": "market",
                "content": json.dumps({"indicator": indicator, "value": value}),
                "metadata": {"indicator": indicator, "data_type": "macro"},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "url": None,
            })
        return items

    def get_stats(self) -> Dict[str, Any]:
        """Return scraper statistics."""
        return {
            "request_count": self._request_count,
            "has_yfinance": HAS_YFINANCE,
            "rate_limit_delay": self.rate_limit_delay,
        }
