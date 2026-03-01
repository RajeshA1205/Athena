# Design: Wire Finnhub Sentiment Data into MarketAnalystAgent

**Task:** TASK-094
**File to modify:** `cli.py` only
**Status:** Design approved, pending implementation

---

## 1. Data Flow

```
data/raw/finnhub/{SYMBOL}_news_{YYYYMMDD}.json
        │
        │  (cli.py loads latest file for symbol)
        ▼
market_data["news"] = ["headline summary", "headline summary", ...]
        │
        │  (passed to MarketAnalystAgent.think())
        ▼
market_analyst.py line 122:
    text_data = market_data.get("news", [])   # ← was always []
        │
        ▼
_analyze_sentiment(text_data) → float (-1.0 to +1.0)
        │
        ▼
_interpret_sentiment(score) → "Bullish signal" / "Bearish signal" / "Neutral"
        │
        ▼
recommendation["supporting_evidence"]["sentiment"] = score
recommendation["reasons"].append(sentiment_signal)
```

---

## 2. Finnhub JSON Structure

Files saved by the ingest pipeline at:
```
data/raw/finnhub/{SYMBOL}_news_{YYYYMMDD}.json
```

Structure:
```json
{
  "symbol": "AAPL",
  "timestamp": "2026-03-01T15:20:12.345678",
  "date_range": {"from": "2026-02-22", "to": "2026-03-01"},
  "article_count": 23,
  "avg_sentiment": 0.12,
  "articles": [
    {
      "headline": "Apple hits record high after earnings beat",
      "summary": "Shares rose 3% as quarterly revenue exceeded expectations...",
      "datetime": 1740000000,
      "source": "Reuters",
      "url": "https://..."
    }
  ]
}
```

---

## 3. File Selection Logic

```python
_FINNHUB_DIR = Path(__file__).resolve().parent / "data" / "raw" / "finnhub"

def _load_news(symbol: str) -> list[str]:
    """Load headlines+summaries from the latest Finnhub news file for symbol."""
    finnhub_dir = _FINNHUB_DIR
    if not finnhub_dir.exists():
        return []
    files = sorted(finnhub_dir.glob(f"{symbol.upper()}_news_*.json"))
    if not files:
        return []
    try:
        with open(files[-1]) as f:
            data = json.load(f)
        texts = []
        for article in data.get("articles", [])[:50]:   # cap at 50
            headline = article.get("headline", "").strip()
            summary  = article.get("summary", "").strip()
            text = f"{headline} {summary}".strip()
            if text:
                texts.append(text)
        return texts
    except Exception:
        logger.warning("Failed to load Finnhub news for %s", symbol)
        return []
```

**Key decisions:**
- `sorted(...).glob(...)[-1]` — filenames contain `YYYYMMDD` so lexicographic sort = chronological sort
- Cap at 50 articles — `_analyze_sentiment` is O(n × keywords), 50 is ample signal without bloat
- `symbol.upper()` — normalises lowercase user input ("aapl") to match filesystem ("AAPL")
- Returns `[]` (not raises) on any failure — agent falls back to `sentiment=None` gracefully

---

## 4. Integration Point in cli.py

In the `analyze()` method, after `market_data` dict is constructed (around line 220), before agents are called:

```python
market_data = {
    "symbol": symbol,
    "prices": prices,
    "bar":    bar_dict,
}

# --- NEW: load Finnhub news for sentiment analysis ---
news = _load_news(symbol)
if news:
    market_data["news"] = news
# (no key if empty — agent handles missing key gracefully)
```

`_load_news` can be a module-level private function or an inline helper; a module-level function is cleaner and testable.

---

## 5. Edge Cases

| Scenario | Behaviour |
|----------|-----------|
| `data/raw/finnhub/` directory doesn't exist | `finnhub_dir.exists()` is False → return `[]` → no `"news"` key → `sentiment=None` |
| No file for this symbol | `glob()` returns empty list → return `[]` |
| File exists but `articles` is empty list | Loop produces no items → return `[]` |
| File is malformed JSON | `json.load()` raises → caught by `except Exception` → WARNING logged → return `[]` |
| `headline` and `summary` both empty | `text.strip()` is `""` → skipped by `if text:` guard |
| Symbol passed as lowercase | `.upper()` normalises before glob |
| >50 articles | `articles[:50]` slice limits processing |

---

## 6. What Does NOT Change

- `agents/market_analyst.py` — zero changes; already handles `market_data.get("news", [])`
- `trading/market_data.py` — zero changes
- Any test files — existing tests unchanged
- Ingest files — zero changes
- The behaviour under `--data-mode mock` — news loads from disk regardless of data mode (orthogonal concerns)

---

## 7. Before vs After

**Before (sentiment always None):**
```
Market Analysis for AAPL
  trend: bullish
  indicators: {"rsi": 62.3, "macd": "positive"}
  sentiment: null
  recommendation: BUY (based on price trend, RSI)
```

**After (sentiment from real news):**
```
Market Analysis for AAPL
  trend: bullish
  indicators: {"rsi": 62.3, "macd": "positive"}
  sentiment: 0.34
  recommendation: BUY (based on price trend, RSI, Bullish news sentiment)
  supporting_evidence:
    sentiment: 0.34   ← "Apple posts record revenue; analysts raise targets"
```

---

## 8. Test Coverage

**Existing tests** (181 passed baseline) — no tests for `cli.py` directly; agent unit tests use mocked `market_data` dicts.

**New tests to add (TASK-095, optional):**
- `test_load_news_returns_headlines` — fixture creates a synthetic JSON file in `tmp_path`; asserts list of strings returned
- `test_load_news_missing_dir` — no directory → returns `[]`
- `test_load_news_malformed_json` — writes bad JSON → returns `[]`, no exception
- `test_load_news_caps_at_50` — 100-article file → asserts `len(result) == 50`

TASK-094 itself does not add tests (scope is `cli.py` only), but the implementation should be structured to make these testable without mocking the filesystem (using the `_FINNHUB_DIR` constant).
