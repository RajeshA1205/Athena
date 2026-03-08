# Design: Fundamentals Data Wiring

> **Note:** This document is an aspirational design (v2) covering deeper integration and more fields than Sprint 13 will implement. The **canonical implementation spec is `project/tasks/TASK-095.md`**. Sections 4.3 (`act()` changes), 4.4 (risk alerts), 5.3 (signal-strength modulation), 5.4 (`act()` changes), and 6 (CLI display) are **deferred** — do not implement them as part of TASK-095. Use this document as a reference for future sprints.

## Goal

Wire yfinance fundamentals data from `data/raw/yfinance/{SYMBOL}_fundamentals_{YYYYMMDD}.json` into the agent pipeline so that RiskManagerAgent and StrategyAgent can use financial ratios, valuation metrics, and balance sheet data in their reasoning.

## Background

The ingest pipeline already collects fundamentals data per symbol via the yfinance collector. Each file is a flat JSON with ~28 keys covering valuation (PE, P/B, P/S), profitability (margins, ROE, ROA), leverage (debt-to-equity, current ratio), growth (revenue, earnings), and market context (beta, 52-week range). The existing `_load_news()` function in `cli.py` provides a reference pattern for file loading.

Currently:
- `cli.py:229-253` builds `market_data` dict with `symbol`, `prices`, `bar`, and optionally `news`
- `market_data` is passed to every agent via `AgentContext.metadata["market_data"]`
- No agent currently uses fundamentals data

---

## 1. Available Data

Each fundamentals JSON contains these keys (observed from AAPL, TSLA, IONQ samples):

| Key | Type | Example (AAPL) | Nullable | Actionable | Target Agent |
|-----|------|-----------------|----------|------------|--------------|
| `symbol` | str | "AAPL" | No | Metadata only | -- |
| `timestamp` | str | "2026-03-01T15:17:32" | No | Staleness check | -- |
| `market_cap` | int | 3882897899520 | No | Yes - size context | Strategy |
| `enterprise_value` | int | 3902065606656 | No | Yes - EV ratios | Strategy |
| `pe_ratio` | float/null | 33.44 | Yes (IONQ) | Yes - valuation | Strategy |
| `forward_pe` | float/null | 28.41 | Rarely null | Yes - forward valuation | Strategy |
| `peg_ratio` | float/null | null | Often null | Skip - too sparse | -- |
| `price_to_book` | float | 44.04 | No | Yes - valuation | Strategy |
| `price_to_sales` | float | 8.91 | No | Yes - valuation | Strategy |
| `profit_margin` | float | 0.2704 | No | Yes - quality | Strategy |
| `operating_margin` | float | 0.3537 | No | Yes - quality | Strategy |
| `roe` | float | 1.52 | No | Yes - efficiency | Strategy |
| `roa` | float | 0.244 | No | Yes - efficiency | Strategy |
| `total_cash` | int | 66907000832 | No | Yes - liquidity | Risk |
| `total_debt` | int | 90509000704 | No | Yes - leverage | Risk |
| `debt_to_equity` | float | 102.63 | No | Yes - leverage | Risk |
| `current_ratio` | float | 0.974 | No | Yes - liquidity | Risk |
| `quick_ratio` | float | 0.845 | No | Yes - liquidity | Risk |
| `revenue_growth` | float | 0.157 | No | Yes - growth | Strategy |
| `earnings_growth` | float/null | 0.183 | Yes (IONQ) | Yes - growth | Strategy |
| `dividend_yield` | float/null | 0.39 | Yes (TSLA) | Informational | Strategy |
| `payout_ratio` | float | 0.1304 | No | Skip - low signal | -- |
| `sector` | str | "Technology" | No | Yes - context | Both |
| `industry` | str | "Consumer Electronics" | No | Yes - context | Both |
| `employees` | int | 150000 | No | Skip - not actionable | -- |
| `beta` | float | 1.107 | No | Yes - systematic risk | Risk |
| `52w_high` | float | 288.62 | No | Yes - range context | Strategy |
| `52w_low` | float | 169.21 | No | Yes - range context | Strategy |

**Fields to skip** (not extracted): `peg_ratio` (too often null), `payout_ratio` (low signal), `employees` (not actionable for trading), `timestamp` (used only for staleness check).

---

## 2. Data Flow

```
data/raw/yfinance/{SYMBOL}_fundamentals_{YYYYMMDD}.json
    |
    v
cli.py: _load_fundamentals(symbol) -> dict | None
    |
    v
cli.py: market_data["fundamentals"] = {...}
    |
    +------> RiskManagerAgent.think(context)
    |           context.metadata["market_data"]["fundamentals"]
    |           Uses: debt_to_equity, current_ratio, quick_ratio, beta, total_debt, total_cash
    |
    +------> StrategyAgent.think(context)
                context.metadata["market_data"]["fundamentals"]
                Uses: pe_ratio, forward_pe, price_to_book, revenue_growth, earnings_growth,
                      profit_margin, operating_margin, market_cap, 52w_high, 52w_low
```

---

## 3. `_load_fundamentals(symbol)` Function Design

**Location**: `/Users/rajesh/athena/cli.py`, new function after `_load_news()` (after line 73).

**Constants to add** (near line 49):
```
_YFINANCE_DIR = Path(__file__).resolve().parent / "data" / "raw" / "yfinance"
```

**Behavior**:
1. Check `_YFINANCE_DIR.exists()` -- return `None` if missing.
2. Glob for `{SYMBOL.upper()}_fundamentals_*.json`, sorted ascending.
3. Take the last file (latest by date suffix).
4. Load JSON, extract the subset of actionable fields listed below.
5. For each field, if the value is `None` or missing, omit it from the returned dict (do NOT substitute a default -- let consuming agents handle absence).
6. Return the dict, or `None` on any exception.

**Extracted fields** (key in returned dict -> key in JSON):
```
symbol            -> symbol
market_cap        -> market_cap
enterprise_value  -> enterprise_value
pe_ratio          -> pe_ratio
forward_pe        -> forward_pe
price_to_book     -> price_to_book
price_to_sales    -> price_to_sales
profit_margin     -> profit_margin
operating_margin  -> operating_margin
roe               -> roe
roa               -> roa
total_cash        -> total_cash
total_debt        -> total_debt
debt_to_equity    -> debt_to_equity
current_ratio     -> current_ratio
quick_ratio       -> quick_ratio
revenue_growth    -> revenue_growth
earnings_growth   -> earnings_growth
dividend_yield    -> dividend_yield
sector            -> sector
industry          -> industry
beta              -> beta
high_52w          -> 52w_high    (rename to valid Python-style key)
low_52w           -> 52w_low     (rename to valid Python-style key)
```

**Return type**: `Optional[dict]` -- `None` means no data available.

**Integration point in `cli.py`**: In `analyze_symbol()`, after `_load_news()` (line 252), add:
```
fundamentals = _load_fundamentals(symbol)
if fundamentals:
    market_data["fundamentals"] = fundamentals
```

---

## 4. RiskManagerAgent Changes

**File**: `/Users/rajesh/athena/agents/risk_manager.py`

### 4.1 New method: `_assess_fundamental_risk()`

Add a new private method that takes a fundamentals dict and returns a sub-assessment dict.

**Input**: `fundamentals: dict` (the dict from `market_data["fundamentals"]`)

**Logic and thresholds**:

| Metric | Condition | Flag | Severity |
|--------|-----------|------|----------|
| `debt_to_equity` | > 200 | `"high_leverage"` | high |
| `debt_to_equity` | > 100 | `"elevated_leverage"` | medium |
| `current_ratio` | < 1.0 | `"weak_liquidity"` | high |
| `current_ratio` | < 1.5 | `"marginal_liquidity"` | medium |
| `quick_ratio` | < 0.5 | `"cash_poor"` | high |
| `beta` | > 2.0 | `"high_systematic_risk"` | high |
| `beta` | > 1.5 | `"elevated_systematic_risk"` | medium |
| `profit_margin` | < 0 | `"unprofitable"` | medium |
| `operating_margin` | < 0 | `"negative_operations"` | high |

**Return type**:
```
{
    "flags": ["high_leverage", ...],
    "severity": "high" | "medium" | "low",
    "beta": 1.107,
    "debt_to_equity": 102.63,
    "current_ratio": 0.974,
}
```

Severity logic:
- Any "high" flag present -> `"high"`
- Any "medium" flag present -> `"medium"`
- Otherwise -> `"low"`

### 4.2 Integration into `think()` (line 90-182)

At `/Users/rajesh/athena/agents/risk_manager.py:100` (after retrieving memory context and before the early return for empty positions), add:

1. Extract fundamentals: `fundamentals = context.metadata.get("market_data", {}).get("fundamentals", {})`
2. If fundamentals is non-empty, call `fundamental_risk = self._assess_fundamental_risk(fundamentals)`
3. Add `"fundamental_risk"` key to the returned dict.

### 4.3 Integration into `_determine_risk_level()` (line 636-651)

Modify `_determine_risk_level` to accept an optional `fundamental_severity` parameter.
- If fundamental severity is `"high"`, escalate the risk level by one step (low->medium, medium->high).
- This requires passing the fundamental assessment through from `think()`.

**Alternative (simpler)**: Instead of modifying `_determine_risk_level`, append fundamental flags to the `compliance_issues` list before calling `_determine_risk_level`. This way the existing logic (3+ issues -> critical, any issues -> medium) naturally incorporates fundamentals. This is the recommended approach.

### 4.4 Integration into `act()` (line 184-306)

In the `result` dict (line 238-243), add `"fundamental_risk"` from the thought dict so it flows through to the coordinator.

### 4.5 Risk alerts from fundamentals

In `_generate_risk_alerts()` (line 570-596), add checks:
- If `fundamental_risk` is present in the metrics dict and has high severity, add an alert string.

---

## 5. StrategyAgent Changes

**File**: `/Users/rajesh/athena/agents/strategy_agent.py`

### 5.1 New method: `_assess_valuation()`

Add a new private method for fundamental valuation assessment.

**Input**: `fundamentals: dict`

**Logic**:

| Check | Condition | Assessment |
|-------|-----------|------------|
| PE cheap | `pe_ratio` exists and < 15 | `"undervalued_pe"` |
| PE expensive | `pe_ratio` exists and > 40 | `"expensive_pe"` |
| Forward PE discount | `forward_pe` < `pe_ratio` * 0.8 | `"earnings_improving"` |
| Revenue growing | `revenue_growth` > 0.10 | `"strong_revenue_growth"` |
| Revenue declining | `revenue_growth` < -0.05 | `"revenue_declining"` |
| High profitability | `profit_margin` > 0.20 | `"high_margin"` |
| 52w range position | current price / `high_52w` < 0.7 | `"well_below_52w_high"` |
| 52w range position | current price / `low_52w` > 1.5 | `"extended_from_52w_low"` |

Note: For the 52-week range checks, the current price is available from `market_data["bar"]["close"]` and should be passed in.

**Return type**:
```
{
    "valuation": "cheap" | "fair" | "expensive",
    "signals": ["undervalued_pe", "strong_revenue_growth", ...],
    "pe_ratio": 33.44,
    "forward_pe": 28.41,
    "revenue_growth": 0.157,
    "profit_margin": 0.2704,
}
```

Valuation determination:
- If any "undervalued" signal -> `"cheap"`
- If any "expensive" signal -> `"expensive"`
- Otherwise -> `"fair"`

### 5.2 Integration into `think()` (line 113-196)

At `/Users/rajesh/athena/agents/strategy_agent.py:125` (after extracting market_data), add:

1. Extract fundamentals: `fundamentals = market_data.get("fundamentals", {})`
2. If fundamentals is non-empty, call `valuation = self._assess_valuation(fundamentals)`
3. Include `valuation` in the `thought` dict returned.

### 5.3 Influence on signal generation

In `_generate_momentum_signals()` (line 376-433) and `_generate_mean_reversion_signals()` (line 435-496):

The strategy params dict already flows through. Add a `valuation` key to `strategy_params` in `think()`. Then in signal generation:

- If `valuation["valuation"] == "expensive"` and signal is a buy, reduce `strength` by 20%: `strength *= 0.8`
- If `valuation["valuation"] == "cheap"` and signal is a buy, boost `strength` by 10%: `strength = min(strength * 1.1, 1.0)`
- Append valuation context to `reasoning` string.

This is a lightweight influence -- fundamentals modulate signal strength rather than override technical signals.

### 5.4 Integration into `act()` result

Include `valuation` in the result dict (line 216-232) alongside `signals` and `strategy`, so it flows through to the coordinator for display.

---

## 6. Display in CLI

**File**: `/Users/rajesh/athena/cli.py`

In `_format_result()` (line 548-681), add a new section after "Market Analysis" and before "Risk Assessment" (around line 592):

**Fundamentals section** (if `result["analyst"]` or a new `result["fundamentals"]` key exists):
- Show: PE ratio, Forward PE, Revenue Growth, Profit Margin, Debt/Equity, Current Ratio, Beta
- Show valuation assessment if available from strategy result
- Show sector and industry

This is lower priority and can be deferred to a follow-up task.

---

## 7. Edge Cases

| Edge Case | Handling |
|-----------|----------|
| No fundamentals file for symbol | `_load_fundamentals()` returns `None`; `market_data` has no `"fundamentals"` key; agents skip fundamentals logic via `.get("fundamentals", {})` |
| JSON has `null` for `pe_ratio` | Field omitted from returned dict; agent checks `fundamentals.get("pe_ratio")` and skips if None |
| Multiple date files exist | Take last in sorted glob (latest date) |
| Non-US stock (e.g., BAYRY) | Some fields may be null or have unusual values (e.g., very different D/E norms); thresholds still apply but may trigger more flags -- acceptable for v1 |
| Corrupted JSON | `_load_fundamentals()` catches all exceptions and returns `None` |
| Very old file (weeks stale) | v1: no staleness check on fundamentals (they change quarterly). Future: check `timestamp` field and warn if > 30 days old |
| `52w_high` / `52w_low` key naming | Rename to `high_52w` / `low_52w` in loader to avoid keys starting with digits |

---

## Files to Modify

| File | Change |
|------|--------|
| `/Users/rajesh/athena/cli.py` | Add `_YFINANCE_DIR` constant, `_load_fundamentals()` function, call it in `analyze_symbol()` |
| `/Users/rajesh/athena/agents/risk_manager.py` | Add `_assess_fundamental_risk()`, integrate into `think()` and `act()` |
| `/Users/rajesh/athena/agents/strategy_agent.py` | Add `_assess_valuation()`, integrate into `think()`, modulate signal strength |

No new files are created. No dependencies are added.
