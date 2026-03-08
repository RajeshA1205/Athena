# Design: FRED Macro Data Wiring

> **Note:** This document is an aspirational design (v2) covering deeper integration than Sprint 13 will implement. The **canonical implementation spec is `project/tasks/TASK-096.md`**. Sections 4 (pandas option — use polars per the task brief), 5.3 (`_make_final_decision()` confidence multipliers), 5.4 (`_build_reasoning()` changes), and 6 (CLI display section) are **deferred**. The column names in Section 1 use the correct **parquet-friendly names** (e.g. `VIX`, `Yield_Curve_Spread`) — do not use raw FRED IDs like `VIXCLS` or `T10Y2Y`. Use this document as a reference for future sprints.

## Goal

Wire FRED macroeconomic data from `data/raw/fred/fred/economic_indicators_fred_{YYYYMMDD}.parquet` into the agent pipeline so that CoordinatorAgent and RiskManagerAgent can incorporate macro regime context (recession risk, volatility environment, monetary policy stance) into their reasoning.

## Background

The ingest pipeline collects 14 FRED economic series into a single parquet file via `FREDCollector` (`/Users/rajesh/athena/ingest/src/data/collectors/fred_collector.py`). The data is a time-indexed DataFrame where each column is one series. Series have different frequencies (daily, monthly, quarterly), so most columns contain many NaN values on any given row. The latest value must be extracted by scanning backward from the last row.

Currently:
- `cli.py:229-253` builds `market_data` with `symbol`, `prices`, `bar`, and optionally `news`
- No agent currently receives macro data
- CoordinatorAgent makes final decisions based on agent recommendations but has no macro context
- RiskManagerAgent assesses position-level risk but has no systemic/macro risk overlay

---

## 1. Available Data

The parquet file contains 14 series, stored with friendly column names per `FREDCollector.DEFAULT_SERIES`:

| FRED ID | Column Name | Frequency | Description | Investment Relevance |
|---------|-------------|-----------|-------------|---------------------|
| `GDP` | `GDP` | Quarterly | Gross Domestic Product | Economic growth proxy |
| `CPIAUCSL` | `CPI` | Monthly | Consumer Price Index | Inflation |
| `UNRATE` | `Unemployment` | Monthly | Unemployment Rate | Labor market health |
| `FEDFUNDS` | `Fed_Funds_Rate` | Daily | Federal Funds Rate | Monetary policy stance |
| `DGS10` | `Treasury_10Y` | Daily | 10-Year Treasury Yield | Risk-free rate benchmark |
| `DGS2` | `Treasury_2Y` | Daily | 2-Year Treasury Yield | Short-term rate expectations |
| `T10Y2Y` | `Yield_Curve_Spread` | Daily | 10Y-2Y Spread | Recession predictor |
| `VIXCLS` | `VIX` | Daily | CBOE Volatility Index | Market fear gauge |
| `DEXUSEU` | `USD_EUR` | Daily | USD/EUR Exchange Rate | Currency context |
| `DTWEXBGS` | `Dollar_Index` | Daily | Trade-Weighted Dollar | Dollar strength |
| `INDPRO` | `Industrial_Production` | Monthly | Industrial Production Index | Manufacturing activity |
| `PAYEMS` | `Nonfarm_Payrolls` | Monthly | Total Nonfarm Payrolls | Employment level |
| `RSXFS` | `Retail_Sales` | Monthly | Retail Sales | Consumer spending |
| `UMCSENT` | `Consumer_Sentiment` | Monthly | U of Michigan Consumer Sentiment | Consumer confidence |

**High-priority series for equity decisions** (used in regime classification):
- `VIX` -- market volatility / fear
- `Yield_Curve_Spread` -- recession signal
- `Fed_Funds_Rate` -- monetary tightening/easing
- `Unemployment` -- labor market
- `Treasury_10Y` -- discount rate context
- `CPI` -- inflation context

**Lower priority** (informational, included but not used in regime classification):
- `GDP`, `Industrial_Production`, `Nonfarm_Payrolls`, `Retail_Sales`, `Consumer_Sentiment`, `USD_EUR`, `Dollar_Index`, `Treasury_2Y`

---

## 2. Data Flow

```
data/raw/fred/fred/economic_indicators_fred_{YYYYMMDD}.parquet
    |
    v
cli.py: _load_macro() -> dict | None
    |  Extracts latest non-null value for each series
    |  Computes: CPI YoY, yield curve slope
    |  Classifies macro regime
    |
    v
cli.py: market_data["macro"] = {
    "indicators": {...},    # raw latest values
    "regime": "expansion",  # classified regime string
    "regime_details": {...} # supporting details
}
    |
    +------> CoordinatorAgent.think(context)
    |           Uses: regime string for decision weighting
    |           Uses: regime_details for reasoning text
    |
    +------> RiskManagerAgent.think(context)
                Uses: VIX as systemic risk multiplier
                Uses: yield curve as recession probability
```

---

## 3. `_load_macro()` Function Design

**Location**: `/Users/rajesh/athena/cli.py`, new function after `_load_fundamentals()`.

**Constants to add**:
```
_FRED_DIR = Path(__file__).resolve().parent / "data" / "raw" / "fred" / "fred"
```

Note the double `fred` -- the actual files are at `data/raw/fred/fred/economic_indicators_fred_*.parquet`.

**Behavior**:
1. Check `_FRED_DIR.exists()` -- return `None` if missing.
2. Glob for `economic_indicators_fred_*.parquet`, sorted ascending.
3. Take the last file (latest by date suffix).
4. Read parquet with `pandas.read_parquet()` (pandas is already an implicit dependency via the data pipeline; if preferred, polars can be used but pandas is simpler for this read-only case).
5. For each column (excluding `date`/index), find the latest non-null value by iterating backward from the last row. Store as `indicators[column_name] = float_value`.
6. Compute derived values (see below).
7. Classify macro regime (see section 4).
8. Return the assembled dict, or `None` on any exception.

**Dependency note**: This requires `pandas` and `pyarrow` (or `fastparquet`). Both should already be available since the ingest pipeline uses them. Add a try/except import guard:
```
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
```
If `HAS_PANDAS` is False, `_load_macro()` returns `None` immediately.

### 3.1 Extracting Latest Non-Null Values

For each column in the DataFrame:
1. Drop NaN values from the column.
2. If any values remain, take the last one (most recent).
3. Store as a float in the `indicators` dict.

Pseudocode:
```
indicators = {}
for col in df.columns:
    if col in ("date",):
        continue
    series = df[col].dropna()
    if len(series) > 0:
        indicators[col] = float(series.iloc[-1])
```

### 3.2 Derived Values

**CPI Year-over-Year**:
1. Get the CPI column, drop NaN.
2. If at least 13 values exist, take the last value and the value 12 rows back (monthly data, so 12 rows = 12 months).
3. `cpi_yoy = (cpi_latest - cpi_12m_ago) / cpi_12m_ago`
4. Add to indicators as `"CPI_YoY"`.

**Real Fed Funds Rate**:
1. If both `Fed_Funds_Rate` and `CPI_YoY` exist: `real_fed_funds = Fed_Funds_Rate - CPI_YoY * 100`
2. Add as `"Real_Fed_Funds_Rate"`.

### 3.3 Return Type

```python
{
    "indicators": {
        "VIX": 18.5,
        "Yield_Curve_Spread": 0.42,
        "Fed_Funds_Rate": 4.75,
        "Treasury_10Y": 4.35,
        "Treasury_2Y": 3.93,
        "Unemployment": 4.1,
        "CPI": 315.2,
        "CPI_YoY": 0.028,        # derived
        "Real_Fed_Funds_Rate": 1.95,  # derived
        "GDP": 28800.0,
        "Industrial_Production": 103.5,
        "Nonfarm_Payrolls": 157000.0,
        "Retail_Sales": 720000.0,
        "Consumer_Sentiment": 67.2,
        "USD_EUR": 1.08,
        "Dollar_Index": 105.3,
    },
    "regime": "expansion",         # from classifier
    "regime_details": {            # from classifier
        "yield_curve": "normal",
        "volatility": "low",
        "monetary_policy": "neutral",
        "labor_market": "healthy",
    },
    "data_date": "2026-03-01",     # date of the parquet file (from filename)
}
```

---

## 4. Macro Regime Classification

Add a standalone function `_classify_macro_regime(indicators: dict) -> tuple[str, dict]` in `cli.py` that returns `(regime_string, details_dict)`.

### 4.1 Classification Rules (evaluated in priority order)

The classifier checks conditions in order. The first matching regime wins.

**Regime 1: `"recession_risk"`**
- Condition: `Yield_Curve_Spread < 0` AND (`Unemployment > 5.0` OR unemployment is rising)
- Since we only have a snapshot (no history), simplify to: `Yield_Curve_Spread < 0` AND `Unemployment > 5.0`
- Details: `{"yield_curve": "inverted", "volatility": ..., "monetary_policy": ..., "labor_market": "weak"}`

**Regime 2: `"high_volatility"`**
- Condition: `VIX > 25`
- Details: `{"yield_curve": ..., "volatility": "high", "monetary_policy": ..., "labor_market": ...}`

**Regime 3: `"tightening"`**
- Condition: `Fed_Funds_Rate > 4.0` AND `Yield_Curve_Spread < 0.5`
- Rationale: High rates with flat/inverted curve = restrictive policy
- Details: `{"yield_curve": "flat", "volatility": ..., "monetary_policy": "tightening", "labor_market": ...}`

**Regime 4: `"expansion"` (default)**
- Condition: None of the above
- Details: `{"yield_curve": "normal", "volatility": "low", "monetary_policy": "neutral", "labor_market": "healthy"}`

### 4.2 Details Sub-Fields

Each sub-field is determined independently:

| Sub-field | Logic |
|-----------|-------|
| `yield_curve` | `"inverted"` if spread < 0, `"flat"` if spread < 0.5, `"normal"` otherwise |
| `volatility` | `"high"` if VIX > 25, `"elevated"` if VIX > 20, `"low"` otherwise |
| `monetary_policy` | `"tightening"` if Fed Funds > 4.0, `"neutral"` if 2.0-4.0, `"accommodative"` if < 2.0 |
| `labor_market` | `"weak"` if unemployment > 6.0, `"softening"` if > 5.0, `"healthy"` otherwise |

### 4.3 Missing Data Handling

If a key indicator is missing from `indicators`:
- `VIX` missing: skip high_volatility check, set `volatility` detail to `"unknown"`
- `Yield_Curve_Spread` missing: skip recession_risk and tightening checks using yield curve, set `yield_curve` detail to `"unknown"`
- `Fed_Funds_Rate` missing: skip tightening check, set `monetary_policy` detail to `"unknown"`
- `Unemployment` missing: set `labor_market` detail to `"unknown"`
- If too many indicators are missing (3+ of the 4 key ones), return `("unknown", {...})` regime

---

## 5. CoordinatorAgent Changes

**File**: `/Users/rajesh/athena/agents/coordinator.py`

### 5.1 Integration into `think()` (line 143-230)

At `/Users/rajesh/athena/agents/coordinator.py:179` (after building initial `orchestration_plan` dict), add:

1. Extract macro data: `macro = context.metadata.get("market_data", {}).get("macro", {})`
2. If macro is non-empty, add `"macro_regime"` and `"macro_details"` to `orchestration_plan`:
   ```
   orchestration_plan["macro_regime"] = macro.get("regime", "unknown")
   orchestration_plan["macro_details"] = macro.get("regime_details", {})
   ```

### 5.2 Integration into LLM synthesis prompt

At line 221-226, where the LLM synthesis prompt is built, append macro context:
```
Current macro regime: {regime}. Yield curve: {yield_curve}. VIX: {vix}. Fed Funds: {fed_funds}.
```

This gives the LLM (when available) richer context for its synthesis.

### 5.3 Integration into `_make_final_decision()` (line 545-593)

At `/Users/rajesh/athena/agents/coordinator.py:545`, the method currently takes `resolved` and `risk_assessment`. It needs to also accept an optional `macro_regime` parameter (default `None`).

Add macro-aware decision adjustments after the risk-level check (line 573-578):

```
if macro_regime == "recession_risk" and decision == "buy":
    confidence *= 0.6  # Significant confidence reduction
    # Optionally downgrade to hold if confidence drops below 0.3

if macro_regime == "high_volatility" and decision != "hold":
    confidence *= 0.8  # Moderate confidence reduction

if macro_regime == "tightening" and decision == "buy":
    confidence *= 0.85  # Slight confidence reduction
```

### 5.4 Integration into `_build_reasoning()` (line 595-626)

Add macro regime to the reasoning string:
```
if macro_regime and macro_regime != "unknown":
    parts.append(f"Macro regime: {macro_regime}")
```

### 5.5 Passing macro_regime through act()

In `act()` at line 296, pass `macro_regime` from the thought dict:
```
macro_regime = thought.get("macro_regime")
final_decision = await self._make_final_decision(resolved, risk_assessment, macro_regime)
```

---

## 6. RiskManagerAgent Changes

**File**: `/Users/rajesh/athena/agents/risk_manager.py`

### 6.1 New method: `_assess_macro_risk()`

**Input**: `macro: dict` (the full `market_data["macro"]` dict)

**Logic**:

| Check | Condition | Effect |
|-------|-----------|--------|
| VIX multiplier | VIX > 30 | systemic_multiplier = 1.5 |
| VIX multiplier | VIX > 25 | systemic_multiplier = 1.25 |
| VIX multiplier | VIX > 20 | systemic_multiplier = 1.1 |
| VIX multiplier | VIX <= 20 | systemic_multiplier = 1.0 |
| Recession signal | Yield_Curve_Spread < 0 | recession_probability = "elevated" |
| Recession signal | Yield_Curve_Spread < 0.5 | recession_probability = "moderate" |
| Recession signal | Yield_Curve_Spread >= 0.5 | recession_probability = "low" |
| Inflation risk | CPI_YoY > 0.04 (4%) | inflation_flag = True |

**Return type**:
```
{
    "systemic_multiplier": 1.25,
    "recession_probability": "moderate",
    "inflation_flag": False,
    "vix": 26.5,
    "yield_curve_spread": 0.42,
    "regime": "high_volatility",
}
```

### 6.2 Integration into `think()` (line 90-182)

At line 100 (after memory retrieval), add:

1. Extract macro: `macro = context.metadata.get("market_data", {}).get("macro", {})`
2. If macro is non-empty, call `macro_risk = self._assess_macro_risk(macro)`
3. Add `"macro_risk"` key to the returned dict.

### 6.3 VIX as Systemic Risk Multiplier

The `systemic_multiplier` from macro risk should scale the VaR output. In `think()`, after computing `var_95` and `var_99` (lines 143-144), apply:

```
if macro_risk:
    var_95 *= macro_risk["systemic_multiplier"]
    var_99 *= macro_risk["systemic_multiplier"]
```

This means that in a high-VIX environment, the reported VaR is scaled up, which naturally flows into `_determine_risk_level()` and may push the risk level from "medium" to "high".

### 6.4 Yield Curve in Risk Alerts

In `_generate_risk_alerts()` (line 570-596), add:
- If `macro_risk["recession_probability"]` is `"elevated"`, add alert: `"Yield curve inverted - elevated recession risk"`
- If `macro_risk["inflation_flag"]` is True, add alert: `"CPI YoY > 4% - inflation risk elevated"`

### 6.5 Integration into `act()` result

Include `"macro_risk"` from the thought dict in the result dict (line 238-243).

---

## 7. Integration in `cli.py` `analyze_symbol()`

**File**: `/Users/rajesh/athena/cli.py`, method `analyze_symbol()` (line 229-499)

After the fundamentals loading (added in the fundamentals design), add:

```
macro = _load_macro()
if macro:
    market_data["macro"] = macro
```

Note: `_load_macro()` takes no symbol argument -- macro data is symbol-independent.

Also, in the `_format_result()` method, add a "Macro Environment" section (after fundamentals, before Risk Assessment):
- Show regime, VIX, yield curve spread, Fed Funds Rate, unemployment
- This is lower priority and can be deferred.

---

## 8. Edge Cases

| Edge Case | Handling |
|-----------|----------|
| Parquet file missing | `_load_macro()` returns `None`; `market_data` has no `"macro"` key; agents skip macro logic via `.get("macro", {})` |
| Series with all nulls | That indicator is absent from `indicators` dict; regime classifier handles missing keys gracefully |
| Stale data (parquet > 7 days old) | v1: no staleness check. The `data_date` field is included so consumers could check. Future: add a warning if `data_date` is > 7 days before today |
| pandas not installed | `HAS_PANDAS` guard returns `None` immediately; no crash |
| Parquet has different column names than expected | Use the friendly names from `FREDCollector.DEFAULT_SERIES` mapping. If the collector changes its rename logic, the column names would change. The classifier checks for specific keys and handles missing ones. |
| Monthly/quarterly series lag | GDP may be 1-2 months behind daily series. The "latest non-null" extraction handles this naturally -- each series independently reports its most recent observation. |
| All key indicators missing | Regime classifier returns `("unknown", {...})` with all details set to `"unknown"` |
| VIX is 0 or negative | Treat as missing (skip VIX-based checks). VIX should never be 0 in real FRED data. |
| CPI_YoY calculation edge case | If fewer than 13 CPI observations exist, skip CPI_YoY derivation |

---

## Files to Modify

| File | Change |
|------|--------|
| `/Users/rajesh/athena/cli.py` | Add `_FRED_DIR` constant, `_load_macro()` function, `_classify_macro_regime()` function, call `_load_macro()` in `analyze_symbol()` |
| `/Users/rajesh/athena/agents/coordinator.py` | Add macro regime to `think()` orchestration plan, pass to `_make_final_decision()`, add to `_build_reasoning()` |
| `/Users/rajesh/athena/agents/risk_manager.py` | Add `_assess_macro_risk()`, integrate into `think()` for VaR scaling, add macro alerts |

**New dependencies**: `pandas` (for parquet reading). Already available in the environment via the ingest pipeline. Guarded with `HAS_PANDAS` flag.

No new files are created.

---

## Interaction Between Fundamentals and Macro

Both designs share the same integration pattern:
1. Loader function in `cli.py` extracts and normalizes data
2. Data is injected into `market_data` dict (separate keys: `"fundamentals"` and `"macro"`)
3. Agents extract their relevant data via `context.metadata["market_data"].get(key, {})`
4. Missing data results in graceful degradation (agent logic is skipped)

The two data sources complement each other:
- **Fundamentals** are stock-specific: "Is AAPL's leverage safe? Is it cheap?"
- **Macro** is market-wide: "Are we heading into recession? Is volatility elevated?"

RiskManagerAgent uses both: fundamentals for company-specific risk, macro for systemic risk.
StrategyAgent uses fundamentals only (macro regime is handled at the coordinator level).
CoordinatorAgent uses macro only (fundamentals assessment flows in via strategy/risk agent results).
