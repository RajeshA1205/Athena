"""
Social Sentiment Scraper
========================
Scrapes social media sentiment from Reddit (r/wallstreetbets, r/investing, etc.)
and mock Twitter sentiment. Falls back to mock data when aiohttp is unavailable
or requests fail.

NOTE: Respect robots.txt and platform terms of service.
Twitter/X API requires authentication — this module always returns mock data
for Twitter to avoid auth complexity in this skeleton implementation.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

logger = logging.getLogger(__name__)


class SocialScraper:
    """
    Scraper for social media sentiment data.

    Supports Reddit (via public JSON API, no auth required) and mock Twitter
    sentiment. Falls back to mock data on network errors or missing aiohttp.

    Args:
        config: Optional configuration dict with keys:
            - rate_limit_delay (float): Seconds between requests (default 2.0)
            - max_retries (int): Request retries (default 3)
            - user_agent (str): HTTP User-Agent header for Reddit API
    """

    _REDDIT_BASE = "https://www.reddit.com"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.rate_limit_delay: float = self.config.get("rate_limit_delay", 2.0)
        self.max_retries: int = self.config.get("max_retries", 3)
        self._user_agent: str = self.config.get(
            "user_agent", "ATHENA/1.0 (research bot)"
        )
        self._request_count: int = 0
        self.logger = logging.getLogger(__name__)

    async def scrape(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Scrape social sentiment for a query (defaults to Reddit).

        Args:
            query: Search query or ticker symbol.
            limit: Maximum number of items to return.

        Returns:
            List of ScrapedItem-compatible dicts.
        """
        return await self.scrape_reddit(
            subreddit="wallstreetbets", query=query, limit=limit
        )

    async def scrape_reddit(
        self,
        subreddit: str = "wallstreetbets",
        query: str = "",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Scrape posts from a Reddit subreddit.

        Attempts a live request to the Reddit JSON API (no auth required for
        public subreddits); falls back to mock posts on any failure.

        Args:
            subreddit: Subreddit name without the r/ prefix.
            query: Optional search query to filter posts.
            limit: Maximum number of posts to return.

        Returns:
            List of ScrapedItem-compatible dicts.
        """
        if HAS_AIOHTTP:
            try:
                return await self._live_reddit(subreddit, query, limit)
            except Exception as e:
                self.logger.warning(
                    "Reddit scrape failed for r/%s (%s), using mock", subreddit, e
                )
        return self._mock_reddit(subreddit, query, limit)

    async def _live_reddit(
        self, subreddit: str, query: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Fetch live Reddit posts via the public JSON endpoint."""
        url = (
            f"{self._REDDIT_BASE}/r/{subreddit}/search.json"
            f"?q={query}&limit={limit}&sort=new&restrict_sr=1"
        )
        headers = {"User-Agent": self._user_agent}
        self._request_count += 1

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                resp.raise_for_status()
                data = await resp.json()

        items = []
        for child in data.get("data", {}).get("children", []):
            post = child.get("data", {})
            sentiment = self._estimate_sentiment(post.get("title", "") + " " + post.get("selftext", ""))
            items.append({
                "source": "reddit",
                "content_type": "social",
                "content": post.get("title", "") + "\n" + post.get("selftext", ""),
                "metadata": {
                    "subreddit": subreddit,
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "sentiment": sentiment,
                    "author": post.get("author", "unknown"),
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "url": f"https://reddit.com{post.get('permalink', '')}",
            })
        return items

    def _mock_reddit(
        self, subreddit: str, query: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Generate mock Reddit posts with varying sentiments."""
        sentiments = [0.7, -0.3, 0.5, 0.1, -0.6, 0.4, -0.1, 0.8, -0.4, 0.2]
        items = []
        for i in range(limit):
            sentiment = sentiments[i % len(sentiments)]
            mood = "bullish" if sentiment > 0 else "bearish"
            items.append({
                "source": "reddit",
                "content_type": "social",
                "content": (
                    f"[Mock r/{subreddit} post {i + 1}] {query} is looking {mood} "
                    f"— {'to the moon! 🚀' if sentiment > 0 else 'puts printing 📉'}. "
                    f"TA shows {'breakout above resistance' if sentiment > 0 else 'breakdown below support'}."
                ),
                "metadata": {
                    "subreddit": subreddit,
                    "score": random.randint(10, 5000),
                    "num_comments": random.randint(5, 500),
                    "sentiment": sentiment,
                    "query": query,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "url": None,
            })
        return items

    async def scrape_twitter_sentiment(
        self, query: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Scrape Twitter/X sentiment for a query.

        Note: Twitter API requires authentication. This method always returns
        mock sentiment data to avoid auth complexity in this skeleton.

        Args:
            query: Search query or cashtag (e.g. "$AAPL").
            limit: Number of mock tweets to generate.

        Returns:
            List of ScrapedItem-compatible dicts with sentiment metadata.
        """
        self.logger.info(
            "Twitter API requires auth — returning mock sentiment for %r", query
        )
        items = []
        for i in range(min(limit, 20)):  # cap mock at 20
            sentiment = round(random.uniform(-1.0, 1.0), 3)
            items.append({
                "source": "mock_twitter",
                "content_type": "social",
                "content": (
                    f"Mock tweet {i + 1} about {query}: "
                    f"{'Feeling very bullish on' if sentiment > 0 else 'Worried about'} "
                    f"{query} today. #investing"
                ),
                "metadata": {
                    "query": query,
                    "sentiment": sentiment,
                    "likes": random.randint(0, 1000),
                    "retweets": random.randint(0, 200),
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "url": None,
            })
        return items

    async def compute_sentiment_summary(
        self, items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute aggregate sentiment statistics across a list of scraped items.

        Args:
            items: List of ScrapedItem-compatible dicts. Items with no
                   ``sentiment`` in metadata default to 0.0.

        Returns:
            Dict with mean_sentiment, positive_count, negative_count,
            neutral_count, total.
        """
        sentiments = [item.get("metadata", {}).get("sentiment", 0.0) for item in items]
        total = len(sentiments)
        if total == 0:
            return {
                "mean_sentiment": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "total": 0,
            }
        mean_sentiment = sum(sentiments) / total
        positive_count = sum(1 for s in sentiments if s > 0.1)
        negative_count = sum(1 for s in sentiments if s < -0.1)
        neutral_count = total - positive_count - negative_count
        return {
            "mean_sentiment": round(mean_sentiment, 4),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "total": total,
        }

    def _estimate_sentiment(self, text: str) -> float:
        """Very basic lexicon-based sentiment for live Reddit posts."""
        positive_words = {"bull", "bullish", "moon", "buy", "long", "calls", "up", "gain", "profit"}
        negative_words = {"bear", "bearish", "crash", "sell", "short", "puts", "down", "loss", "dump"}
        words = set(text.lower().split())
        score = len(words & positive_words) - len(words & negative_words)
        return max(-1.0, min(1.0, score * 0.2))

    def get_stats(self) -> Dict[str, Any]:
        """Return scraper statistics."""
        return {
            "request_count": self._request_count,
            "has_aiohttp": HAS_AIOHTTP,
            "rate_limit_delay": self.rate_limit_delay,
        }
