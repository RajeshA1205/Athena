"""
ATHENA Data Scrapers
====================
Scrapers for collecting training data: news/SEC filings, market OHLCV,
and social sentiment. All scrapers use async I/O with rate limiting and
fall back to mock data when external libraries or network are unavailable.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .news import NewsScraper
from .market import MarketScraper
from .social import SocialScraper


@dataclass
class ScrapedItem:
    """
    A single scraped data item from any source.

    Attributes:
        source: Data source identifier (e.g. "reuters", "reddit", "yfinance")
        content_type: High-level type ("news", "market", "social")
        content: Raw text or JSON string payload
        metadata: Arbitrary extra metadata (sentiment, ticker, etc.)
        timestamp: UTC ISO timestamp of collection
        url: Optional source URL
    """

    source: str
    content_type: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    url: Optional[str] = None


class BaseScraper(ABC):
    """
    Abstract base class for all ATHENA data scrapers.

    Provides rate limiting, retry configuration, and request counting.
    Subclasses implement scrape() for their specific data source.

    Args:
        config: Optional configuration dict with keys:
            - rate_limit_delay (float): Seconds to sleep between requests (default 1.0)
            - max_retries (int): Maximum request retries (default 3)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.rate_limit_delay: float = self.config.get("rate_limit_delay", 1.0)
        self.max_retries: int = self.config.get("max_retries", 3)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._request_count: int = 0

    @abstractmethod
    async def scrape(self, query: str, limit: int = 10) -> List[ScrapedItem]:
        """
        Scrape items for a given query string.

        Args:
            query: Search query or ticker symbol.
            limit: Maximum number of items to return.

        Returns:
            List of ScrapedItem objects.
        """
        ...

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        await asyncio.sleep(self.rate_limit_delay)
        self._request_count += 1

    def get_stats(self) -> Dict[str, Any]:
        """Return scraper statistics."""
        return {
            "request_count": self._request_count,
            "rate_limit_delay": self.rate_limit_delay,
            "max_retries": self.max_retries,
        }


__all__ = [
    "NewsScraper",
    "MarketScraper",
    "SocialScraper",
    "ScrapedItem",
    "BaseScraper",
]
