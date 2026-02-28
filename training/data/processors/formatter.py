"""
Data Formatter
==============
Transforms cleaned ScrapedItem dicts into structured training records.
Handles content-type-specific formatting (news, market, social) and
produces normalized feature dicts ready for dataset construction.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Sentinel for unknown numeric values
_UNKNOWN = 0.0


class DataFormatter:
    """
    Formats cleaned ScrapedItem dicts into training-ready records.

    Dispatches on the 'content_type' field:
      - 'news'   → text + sentiment label
      - 'market' → OHLCV numeric features
      - 'social' → text + sentiment score + engagement features

    Config keys:
        max_text_length (int): Truncate text to this many characters (default 512).
        include_metadata (bool): Embed metadata fields in output (default True).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.max_text_length: int = int(self.config.get("max_text_length", 512))
        self.include_metadata: bool = bool(self.config.get("include_metadata", True))
        self._stats: Dict[str, int] = {"formatted": 0, "skipped": 0}

    def format_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Format a single cleaned ScrapedItem into a training record.

        Args:
            item: Cleaned ScrapedItem dict.

        Returns:
            Formatted training record dict, or None if unrecognised content_type.
        """
        content_type = item.get("content_type", "")
        dispatcher = {
            "news": self._format_news,
            "market": self._format_market,
            "social": self._format_social,
        }
        handler = dispatcher.get(content_type)
        if handler is None:
            logger.debug("Unknown content_type %r — skipping", content_type)
            self._stats["skipped"] += 1
            return None

        record = handler(item)
        record["source"] = item.get("source", "unknown")
        record["timestamp"] = item.get("timestamp", "")
        if self.include_metadata:
            record["metadata"] = item.get("metadata", {})
        self._stats["formatted"] += 1
        return record

    def format_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format a batch of cleaned ScrapedItem dicts.

        Args:
            items: List of cleaned ScrapedItem dicts.

        Returns:
            List of formatted training records (None items excluded).
        """
        results = []
        for item in items:
            record = self.format_item(item)
            if record is not None:
                results.append(record)
        return results

    # ------------------------------------------------------------------
    # Content-type handlers
    # ------------------------------------------------------------------

    def _format_news(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Format a news article into a text + sentiment record."""
        text = item.get("content", "")[: self.max_text_length]
        metadata = item.get("metadata", {})
        sentiment = float(metadata.get("sentiment", _UNKNOWN))
        return {
            "content_type": "news",
            "text": text,
            "text_length": len(text),
            "sentiment": sentiment,
            "label": self._sentiment_label(sentiment),
        }

    def _format_market(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Format a market OHLCV item into a numeric feature record."""
        try:
            ohlcv = json.loads(item.get("content", "{}"))
        except (json.JSONDecodeError, TypeError):
            ohlcv = {}

        metadata = item.get("metadata", {})
        return {
            "content_type": "market",
            "symbol": metadata.get("symbol", ""),
            "date": ohlcv.get("date", ""),
            "open": float(ohlcv.get("open", _UNKNOWN)),
            "high": float(ohlcv.get("high", _UNKNOWN)),
            "low": float(ohlcv.get("low", _UNKNOWN)),
            "close": float(ohlcv.get("close", _UNKNOWN)),
            "volume": float(ohlcv.get("volume", _UNKNOWN)),
            "features": [
                float(ohlcv.get("open", _UNKNOWN)),
                float(ohlcv.get("high", _UNKNOWN)),
                float(ohlcv.get("low", _UNKNOWN)),
                float(ohlcv.get("close", _UNKNOWN)),
                float(ohlcv.get("volume", _UNKNOWN)),
            ],
        }

    def _format_social(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Format a social media post into text + engagement features."""
        text = item.get("content", "")[: self.max_text_length]
        metadata = item.get("metadata", {})
        sentiment = float(metadata.get("sentiment", _UNKNOWN))
        score = float(metadata.get("score", _UNKNOWN))
        num_comments = float(metadata.get("num_comments", _UNKNOWN))
        likes = float(metadata.get("likes", _UNKNOWN))
        retweets = float(metadata.get("retweets", _UNKNOWN))
        return {
            "content_type": "social",
            "text": text,
            "text_length": len(text),
            "sentiment": sentiment,
            "label": self._sentiment_label(sentiment),
            "engagement": score + num_comments + likes + retweets,
            "features": [sentiment, score, num_comments, likes, retweets],
        }

    @staticmethod
    def _sentiment_label(score: float) -> int:
        """Convert sentiment score to class label: 0=negative, 1=neutral, 2=positive."""
        if score > 0.1:
            return 2
        if score < -0.1:
            return 0
        return 1

    def get_stats(self) -> Dict[str, int]:
        """Return cumulative formatting statistics."""
        return dict(self._stats)
