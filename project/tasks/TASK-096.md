# TASK-096: Wire FRED Macro Data into CoordinatorAgent and RiskManager

## Status
- **State:** Queued
- **Priority:** 🟡 High
- **Depends on:** TASK-094 (hard), TASK-095 (soft — `risk_manager.py` changes call `_apply_fundamental_risk_adjustments` which must exist before `_apply_macro_risk_adjustments` is added)
- **Created:** 2026-03-01

## Objective
Load the latest FRED macroeconomic indicator row from the ingest pipeline's parquet output and inject a curated `market_data["macro"]` dict into the pipeline. `CoordinatorAgent` uses it to annotate the final synthesis with a macro regime label. `RiskManagerAgent` uses VIX and yield curve spread as systemic risk multipliers on the computed risk score.

## Context

### Where the data comes from
The ingest pipeline saves FRED data to:
```
data/raw/fred/fred/economic_indicators_fred_{YYYYMMDD}.parquet
```
Each parquet file has one row per date. Read the file with `polars` (already a project dependency via the ingest pipeline) and take the **last row** (most recent date). The columns of interest are:

| Column name in parquet | Meaning | Key in macro dict |
|---|---|---|
| `VIX` | CBOE VIX | `vix` |
| `Yield_Curve_Spread` | 10Y−2Y Treasury spread (yield curve) | `yield_curve_spread` |
| `Fed_Funds_Rate` | Fed Funds Rate | `fed_funds_rate` |
| `Unemployment` | Unemployment rate | `unemployment` |
| `CPI` | CPI (monthly level) | `cpi_level` |
| `Treasury_10Y` | 10-Year Treasury yield | `treasury_10y` |
| `Treasury_2Y` | 2-Year Treasury yield | `treasury_2y` |
| `GDP` | GDP level (quarterly) | `gdp` |

Note: The FRED collector (`ingest/src/data/collectors/fred_collector.py`) renames raw FRED series IDs to friendly names before writing to parquet. The parquet therefore contains `VIX`, `Yield_Curve_Spread`, etc. — **not** the raw FRED IDs like `VIXCLS`, `T10Y2Y`, `FEDFUNDS`. Verified against `data/raw/fred/fred/economic_indicators_fred_20260301.parquet`.

`cpi_yoy` (year-over-year CPI change) must be **computed** from the parquet data: read the last 13 rows, compute `(last_row["CPI"] / 12_rows_ago["CPI"] - 1)`. If fewer than 13 rows are available, set `cpi_yoy` to `None`.

All values may be `NaN` or missing in the parquet if FRED had no data for that date. Map any `NaN` / `null` to `None` in the Python dict.

### Parquet reading approach
Use `polars`:
```python
import polars as pl

df = pl.read_parquet(path)
if df.is_empty():
    return {}
last_row = df.tail(1).to_dicts()[0]
```
Use `asyncio.to_thread()` to wrap the polars file read since it is blocking I/O (consistent with the project pattern in `trading/market_data.py`).

### How sentiment wiring was done (follow the same pattern)
TASK-094 added `_FINNHUB_DIR`, `_MAX_NEWS_ARTICLES`, and `_load_news()` as module-level items in `cli.py`. Follow the same structure for FRED:
- Module-level `_FRED_DIR = Path(__file__).resolve().parent / "data" / "raw" / "fred" / "fred"`
- Module-level async `_load_macro(symbol: str = None) -> dict` (symbol is unused but kept for API consistency; pass `None`)
- In `analyze_symbol()`, call `macro = await _load_macro()` and set `market_data["macro"] = macro` if the result is non-empty

Note: `_load_macro` is async (unlike `_load_news` which is sync) because it uses `asyncio.to_thread`.

### What market_data will look like after this task
```python
market_data = {
    "symbol": symbol,
    "prices": prices,
    "bar": bar_dict,
    "news": [...],            # from TASK-094
    "fundamentals": {...},    # from TASK-095
    "macro": {
        "vix": float | None,
        "yield_curve_spread": float | None,
        "fed_funds_rate": float | None,
        "unemployment": float | None,
        "cpi_yoy": float | None,
        "cpi_level": float | None,
        "treasury_10y": float | None,
        "treasury_2y": float | None,
        "gdp": float | None,
        "macro_regime": str,   # derived label, always present
    }
}
```

### macro_regime derivation
The `_load_macro()` function itself (not the agents) should derive `macro_regime` and include it in the returned dict. This is a pre-computed string label the agents can read directly without needing to implement the logic themselves.

Rules (evaluated top-to-bottom; first match wins):
1. `yield_curve_spread is not None and yield_curve_spread < 0` → `"recession_risk"`
2. `vix is not None and vix > 30` → `"high_volatility"`
3. `vix is not None and vix < 15` → `"low_volatility"`
4. `fed_funds_rate is not None and fed_funds_rate > 5.0` → `"restrictive_monetary"`
5. Fallback → `"neutral"`

### How CoordinatorAgent.think() currently works
`CoordinatorAgent.think()` is in `/Users/rajesh/athena/agents/coordinator.py` (around line 143). It:
1. Retrieves memory context
2. Receives LatentMAS messages
3. Collects `recommendations` from `context.messages`
4. Detects conflicts
5. Calls `_llm_reason()` with the recommendations
6. Returns `orchestration_plan` dict

The implementation must:
1. Read `macro = context.metadata.get("market_data", {}).get("macro", {})` in `think()`.
2. Add a **synchronous** helper method `_build_macro_context_note(macro: dict) -> str` that returns a human-readable string describing the macro regime, for example:
   - `"Macro regime: recession_risk (inverted yield curve: -0.45). VIX: 24.3. Fed funds: 5.25%."`
   - `"Macro regime: high_volatility (VIX: 38.2). Unemployment: 4.1%."`
   - `"No macro data available."` if `macro` is empty.
   The exact format is flexible as long as it includes the `macro_regime` value and at least two other populated metrics.
3. Add `"macro_context"` to `orchestration_plan` with the return value of `_build_macro_context_note(macro)`.
4. When constructing the LLM prompt in `think()`, prepend the macro context note:
   ```python
   llm_synthesis = await self._llm_reason(
       f"Macro context: {macro_context_note}. "
       f"Given these agent recommendations: {recommendations}, "
       f"what is the best trading decision?"
   )
   ```
   (Previously the prompt was just the `recommendations` part.)

### How RiskManagerAgent uses macro data
In `RiskManagerAgent.think()`, after calling `_apply_fundamental_risk_adjustments` (added by TASK-095), add a further adjustment step:

```python
macro = context.metadata.get("market_data", {}).get("macro", {})
if macro:
    risk_level = self._apply_macro_risk_adjustments(risk_level, macro, alerts, all_issues)
```

Add a new synchronous method `_apply_macro_risk_adjustments(self, risk_level: str, macro: dict, alerts: list, compliance_issues: list) -> str` that:
- If `macro.get("vix") is not None and macro["vix"] > 30`: elevate risk level one step (low→medium, medium→high, high stays high); append `f"High VIX ({macro['vix']:.1f}): systemic volatility elevated"` to `alerts`.
- If `macro.get("yield_curve_spread") is not None and macro["yield_curve_spread"] < 0`: elevate risk level one step; append `f"Inverted yield curve ({macro['yield_curve_spread']:.2f}): recession risk signal"` to `compliance_issues`.
- Both conditions can apply in the same call (evaluate independently, apply sequentially).
- `None` values for any field are skipped.
- Risk level is capped at `"high"`.
- Returns the (possibly elevated) `risk_level` string.

Also add `"macro_used": bool` to the return dict from `think()` — set `True` if `macro` is non-empty.

## Scope & Constraints
- **May modify:** `cli.py`, `agents/coordinator.py`, `agents/risk_manager.py`
- **May NOT modify:** any test files, `agents/market_analyst.py`, `agents/strategy_agent.py`, `agents/execution_agent.py`, `trading/`, `memory/`, `training/`, `communication/`, `core/`, `models/`, `ingest/`
- No hardcoded absolute paths. Use `Path(__file__).resolve().parent` as the anchor in `cli.py`.
- All new code must use `%s`-style logger calls (not f-strings): e.g. `logger.warning("Failed to load FRED data: %s", e)`.
- `_build_macro_context_note` must be synchronous (no `async`).
- `_apply_macro_risk_adjustments` must be synchronous (no `async`).
- `_load_macro` in `cli.py` must be `async` and use `asyncio.to_thread` for the polars file read.
- If `polars` is not installed, `_load_macro` must catch `ImportError`, log at `WARNING`, and return `{}`.
- Any `NaN` or `null` in the parquet must be mapped to Python `None` in the dict (use `math.isnan()` guarded by `isinstance(v, float)`, or polars null handling before `.to_dicts()`).
- Risk elevation in `_apply_macro_risk_adjustments` is independent of and sequential to the fundamentals elevation in `_apply_fundamental_risk_adjustments` (both methods are called in the same `think()`, macro after fundamentals).
- Both `_apply_fundamental_risk_adjustments` and `_apply_macro_risk_adjustments` must pass `"critical"` through unchanged — if `_determine_risk_level()` returns `"critical"`, neither adjustment method should elevate it further (cap logic applies only to the `low`/`medium`/`high` range).
- **This task brief is the canonical implementation spec.** The design document at `plans/sprint13-agent-intelligence/designs/macro_wiring.md` describes an aspirational v2 with more complexity (confidence multipliers, VaR scaling, backward-scan extraction). Do not implement features from the design doc that are not in this brief.

## Input
- `/Users/rajesh/athena/cli.py` — `analyze_symbol()`, existing `_load_news` pattern
- `/Users/rajesh/athena/agents/coordinator.py` — `think()`, `act()`, `_llm_reason` usage
- `/Users/rajesh/athena/agents/risk_manager.py` — `think()`, `_apply_fundamental_risk_adjustments` (added by TASK-095), `act()`
- `data/raw/fred/fred/economic_indicators_fred_{YYYYMMDD}.parquet` — input data (may not exist if ingest not run; loader must handle absence gracefully)

## Expected Output

### cli.py additions
1. Module-level constant:
   ```python
   _FRED_DIR = Path(__file__).resolve().parent / "data" / "raw" / "fred" / "fred"
   ```
2. Async function `_load_macro() -> dict` that:
   - Returns `{}` if `_FRED_DIR` does not exist or polars is not importable (log WARNING in the latter case)
   - Globs for `economic_indicators_fred_*.parquet`, sorts, takes the latest file
   - Returns `{}` if no files found
   - Reads the file via `asyncio.to_thread(pl.read_parquet, path)`, takes the last row
   - For `cpi_yoy`: reads up to 13 rows (`df.tail(13)`); if `len(df) >= 13` computes YoY change, else sets `None`
   - Maps NaN/null to `None` for all values
   - Derives `macro_regime` using the rules described in Context
   - Returns `{}` on any unhandled `Exception`, logging at `WARNING` level
3. In `analyze_symbol()`, immediately after the `_load_fundamentals` block:
   ```python
   macro = await _load_macro()
   if macro:
       market_data["macro"] = macro
   ```

### agents/coordinator.py additions
1. New synchronous method `_build_macro_context_note(self, macro: dict) -> str`
2. Modified `think()`: reads macro from `context.metadata.get("market_data", {}).get("macro", {})`, calls `_build_macro_context_note`, adds `"macro_context"` to `orchestration_plan`, prepends macro note to LLM prompt.

### agents/risk_manager.py additions
1. New synchronous method `_apply_macro_risk_adjustments(self, risk_level: str, macro: dict, alerts: list, compliance_issues: list) -> str`
2. Modified `think()`: reads macro from `context.metadata.get("market_data", {}).get("macro", {})`, calls `_apply_macro_risk_adjustments` after `_apply_fundamental_risk_adjustments`, adds `"macro_used"` bool to returned dict.

## Acceptance Criteria
- [ ] `_load_macro()` returns a non-empty dict with `"macro_regime"` key when a FRED parquet file exists
- [ ] `macro_regime` is `"recession_risk"` when `Yield_Curve_Spread < 0`, `"high_volatility"` when `VIX > 30`, `"low_volatility"` when `VIX < 15`, `"restrictive_monetary"` when `Fed_Funds_Rate > 5.0`, `"neutral"` otherwise
- [ ] `cpi_yoy` is computed as `(latest_CPI / CPI_12_months_ago) - 1` when at least 13 rows are available
- [ ] `cpi_yoy` is `None` when fewer than 13 rows are available
- [ ] NaN/null parquet values are mapped to `None` (not `float("nan")`)
- [ ] `_load_macro()` returns `{}` and logs a WARNING when `polars` is not importable
- [ ] When no FRED parquet exists, no exception is raised and `market_data` does not get a `"macro"` key
- [ ] `CoordinatorAgent.think()` returns a dict that includes `"macro_context"` key (a non-empty string) when `market_data["macro"]` is populated
- [ ] `_build_macro_context_note({})` returns `"No macro data available."` (or equivalent sentinel string)
- [ ] `_apply_macro_risk_adjustments`: `vix=35` with `risk_level="low"` elevates to `"medium"` and appends a string containing "VIX" to `alerts`
- [ ] `_apply_macro_risk_adjustments`: `yield_curve_spread=-0.3` with `risk_level="medium"` elevates to `"high"` and appends a string containing "yield curve" (case-insensitive) to `compliance_issues`
- [ ] `_apply_macro_risk_adjustments`: `vix=35, yield_curve_spread=-0.3` with `risk_level="low"` results in `"high"` (two elevations) and both lists populated
- [ ] `RiskManagerAgent.think()` result contains `"macro_used": True` when macro data is present, `"macro_used": False` when absent
- [ ] `pytest tests/ -q` passes — all existing tests green, no new failures (baseline: 181 passed, 4 skipped per TASK-094)

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes

**Senior-dev review 2026-03-01 — APPROVED WITH NOTES (after revisions)**

Revisions applied:
- Fixed column name table: replaced raw FRED IDs (`VIXCLS`, `T10Y2Y`, `FEDFUNDS`, `UNRATE`, `CPIAUCSL`, `DGS10`, `DGS2`) with actual parquet column names (`VIX`, `Yield_Curve_Spread`, `Fed_Funds_Rate`, `Unemployment`, `CPI`, `Treasury_10Y`, `Treasury_2Y`) — verified against live parquet
- Fixed CPI YoY formula: `last_row["CPIAUCSL"]` → `last_row["CPI"]`
- Fixed acceptance criterion to use `Yield_Curve_Spread` not `T10Y2Y`
- Added soft dependency on TASK-095
- Added `"critical"` pass-through to Scope & Constraints
- Added design-doc-is-aspirational note to Scope & Constraints
- Design doc marked with deferred sections note
