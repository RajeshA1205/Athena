# TASK-094: Wire Finnhub Sentiment Data into MarketAnalystAgent

## Status
- **State:** Queued
- **Priority:** 🟡 High
- **Depends on:** TASK-089, TASK-090
- **Created:** 2026-03-01

## Objective
Load Finnhub news articles collected by the ingest pipeline and pass them into `MarketAnalystAgent` via `market_data["news"]` so that sentiment analysis runs on real news instead of being skipped.

## Context
The ingest pipeline already saves Finnhub news to `data/raw/finnhub/{SYMBOL}_news_{YYYYMMDD}.json`. `MarketAnalystAgent` already has `_analyze_sentiment()` and reads `market_data.get("news", [])` — but `cli.py` never populates that key, so sentiment is always `None`.

This is a `cli.py`-only change. No agent code needs to be modified.

Finnhub JSON structure:
```json
{
  "symbol": "AAPL",
  "articles": [
    {"headline": "Apple hits record high", "summary": "Shares rose...", "datetime": 1700000000}
  ],
  "avg_sentiment": 0.12,
  "article_count": 23
}
```

Design document: `plans/sprint12-ingest/designs/sentiment_wiring.md`

## Scope & Constraints
- **May modify:** `cli.py` only
- **May NOT modify:** any agent files, any test files, `trading/market_data.py`, ingest files
- Path to Finnhub data must be computed dynamically: `Path(__file__).resolve().parent / "data" / "raw" / "finnhub"`
- No hardcoded absolute paths
- Cap headlines at 50 articles to avoid context bloat
- Symbol must be normalised to uppercase when globbing for files
- Malformed JSON must be caught and logged as WARNING, not raised

## Input
- `cli.py` — the `analyze()` method that builds `market_data`
- `data/raw/finnhub/{SYMBOL}_news_*.json` — latest file per symbol
- `agents/market_analyst.py` — existing `_analyze_sentiment()` interface (read-only reference)

## Expected Output
`cli.py` with:
1. A helper (inline or small private method) that loads the latest Finnhub news file for a symbol and returns `List[str]` of `"headline summary"` strings
2. `market_data["news"]` populated before agents are called when a Finnhub file exists
3. Graceful fallback (no key, no exception) when no file exists

## Acceptance Criteria
- [ ] `market_data["news"]` is a non-empty `List[str]` when a Finnhub file exists for the symbol
- [ ] Each string in the list is `headline + " " + summary` (blank parts skipped)
- [ ] List is capped at 50 items
- [ ] Symbol is uppercased when searching for files
- [ ] If no Finnhub file exists, no exception is raised and sentiment degrades to `None`
- [ ] Malformed JSON is caught and logged at WARNING level
- [ ] Path uses `Path(__file__).resolve().parent`, not a hardcoded absolute path
- [ ] `pytest tests/ -q` passes — 181 passed, 4 skipped baseline

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
