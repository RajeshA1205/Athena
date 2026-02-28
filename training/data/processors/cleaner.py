"""
Data Cleaner
============
Cleans and normalizes raw scraped text data for training pipeline ingestion.
Handles HTML stripping, whitespace normalization, length filtering, and
basic deduplication.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Cleans raw scraped items for the training pipeline.

    Operates on ScrapedItem-compatible dicts produced by the scrapers.
    Filters out items that are too short, too long, or duplicate.

    Config keys:
        min_content_length (int): Minimum content length in characters (default 10).
        max_content_length (int): Maximum content length in characters (default 4096).
        strip_html (bool): Remove HTML tags (default True).
        normalize_whitespace (bool): Collapse whitespace (default True).
        deduplicate (bool): Drop duplicate content within a batch (default True).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.min_length: int = int(self.config.get("min_content_length", 10))
        self.max_length: int = int(self.config.get("max_content_length", 4096))
        self.strip_html: bool = bool(self.config.get("strip_html", True))
        self.normalize_whitespace: bool = bool(
            self.config.get("normalize_whitespace", True)
        )
        self.deduplicate: bool = bool(self.config.get("deduplicate", True))
        self._stats: Dict[str, int] = {
            "total": 0,
            "cleaned": 0,
            "dropped_short": 0,
            "dropped_long": 0,
            "dropped_duplicate": 0,
        }

    def clean_text(self, text: str) -> str:
        """
        Apply cleaning transforms to a single text string.

        Args:
            text: Raw text content.

        Returns:
            Cleaned text string.
        """
        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)

        if self.strip_html:
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"&[a-z]+;", " ", text)

        if self.normalize_whitespace:
            text = re.sub(r"\s+", " ", text).strip()

        return text

    def clean_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Clean a single ScrapedItem dict.

        Args:
            item: ScrapedItem-compatible dict with at least a 'content' key.

        Returns:
            Cleaned item dict, or None if the item should be dropped.
        """
        content = item.get("content", "")
        cleaned = self.clean_text(str(content))

        if len(cleaned) < self.min_length:
            self._stats["dropped_short"] += 1
            return None

        if len(cleaned) > self.max_length:
            cleaned = cleaned[: self.max_length]
            self._stats["dropped_long"] += 1

        result = dict(item)
        result["content"] = cleaned
        return result

    def clean_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean a batch of ScrapedItem dicts.

        Applies per-item cleaning and optional deduplication across the batch.

        Args:
            items: List of ScrapedItem-compatible dicts.

        Returns:
            List of cleaned items with duplicates and invalid items removed.
        """
        self._stats["total"] += len(items)
        cleaned_items: List[Dict[str, Any]] = []
        seen: set = set()

        for item in items:
            result = self.clean_item(item)
            if result is None:
                continue

            if self.deduplicate:
                content_key = result["content"]
                if content_key in seen:
                    self._stats["dropped_duplicate"] += 1
                    continue
                seen.add(content_key)

            cleaned_items.append(result)
            self._stats["cleaned"] += 1

        logger.debug(
            "Cleaned batch: %d in → %d out (%d short, %d dup)",
            len(items),
            len(cleaned_items),
            self._stats["dropped_short"],
            self._stats["dropped_duplicate"],
        )
        return cleaned_items

    def get_stats(self) -> Dict[str, int]:
        """Return cumulative cleaning statistics."""
        return dict(self._stats)
