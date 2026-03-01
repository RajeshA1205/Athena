# TASK-095: Wire Fundamentals Data into RiskManager and StrategyAgent

## Status
- **State:** Queued
- **Priority:** 🟡 High
- **Depends on:** TASK-094
- **Created:** 2026-03-01

## Objective
Load per-symbol YFinance fundamentals data collected by the ingest pipeline and make it available to `RiskManagerAgent` (for fundamental risk adjustment) and `StrategyAgent` (for valuation context). The data is injected at the `cli.py` layer — no agent constructor signatures change.

## Context

### Where the data comes from
The ingest pipeline saves fundamentals to:
```
data/raw/yfinance/{SYMBOL}_fundamentals_{YYYYMMDD}.json
```
A typical file (e.g. `AAPL_fundamentals_20260301.json`) looks like:
```json
{
  "symbol": "AAPL",
  "pe_ratio": 28.4,
  "forward_pe": 25.1,
  "price_to_book": 45.2,
  "debt_to_equity": 1.73,
  "current_ratio": 0.98,
  "revenue_growth": 0.062,
  "earnings_growth": 0.11,
  "profit_margin": 0.253,
  "return_on_equity": 1.45,
  "market_cap": 2850000000000,
  "sector": "Technology",
  "industry": "Consumer Electronics",
  "beta": 1.24,
  "52_week_high": 198.23,
  "52_week_low": 142.50,
  "analyst_target_price": 205.00,
  "recommendation_mean": 1.8
}
```
Any field may be `null` if YFinance could not fetch it. The loader must handle missing keys gracefully.

### How sentiment wiring was done (follow the same pattern)
TASK-094 added two module-level constants and a helper function in `cli.py`:
```python
_FINNHUB_DIR = Path(__file__).resolve().parent / "data" / "raw" / "finnhub"
_MAX_NEWS_ARTICLES = 50

def _load_news(symbol: str) -> list:
    ...
```
Then in `analyze_symbol()`, just before agents are called:
```python
news = _load_news(symbol)
if news:
    market_data["news"] = news
```
Follow the exact same structure for fundamentals.

### How agents currently receive market_data
In `cli.py::analyze_symbol()` the `market_data` dict is passed to every agent via `AgentContext.metadata["market_data"]`. As of TASK-094 completion it contains:
```python
market_data = {
    "symbol": symbol,
    "prices": prices,
    "bar": bar_dict,
    "news": [...]   # populated by TASK-094 if file exists
}
```
After this task, it will also contain:
```python
market_data["fundamentals"] = {
    "pe_ratio": float | None,
    "forward_pe": float | None,
    "debt_to_equity": float | None,
    "current_ratio": float | None,
    "revenue_growth": float | None,
    "earnings_growth": float | None,
    "profit_margin": float | None,
    "return_on_equity": float | None,
    "beta": float | None,
    "analyst_target": float | None,
    "sector": str | None,
}
```
Note: `analyst_target_price` from the JSON is mapped to the key `analyst_target` in the dict.

### How RiskManagerAgent.think() currently works
`RiskManagerAgent.think()` is in `/Users/rajesh/athena/agents/risk_manager.py` (around line 90).
It receives `context.metadata` which contains `"market_data"`, `"portfolio"`, `"positions"`, and `"returns"`.

Currently it reads `context.metadata.get("positions", [])` and `context.metadata.get("returns", {})` and computes VaR, expected shortfall, compliance checks. The result dict returned from `think()` includes `"risk_level"`.

The `_determine_risk_level()` method (in the same file) converts a combination of VaR, expected shortfall, portfolio metrics, and compliance issues into `"low"`, `"medium"`, or `"high"`.

The implementation must add a **new method** `_apply_fundamental_risk_adjustments(risk_level: str, fundamentals: dict) -> str` that nudges the computed `risk_level` upward based on:
- `debt_to_equity > 2.0` → elevate one level (low→medium, medium→high)
- `current_ratio < 1.0` → elevate one level
- `beta > 1.5` → add `"high_beta"` to `alerts` list and append `"High beta (>1.5): elevated volatility"` to `compliance_issues`

Call this method at the end of `think()` just before constructing the return dict, using:
```python
fundamentals = context.metadata.get("market_data", {}).get("fundamentals", {})
if fundamentals:
    risk_level = self._apply_fundamental_risk_adjustments(risk_level, fundamentals)
```

The `alerts` list should be extended (not replaced) if high beta is detected. The `compliance_issues` list should also be extended (not replaced).

Also add `"fundamentals_used": bool` to the return dict from `think()` — set `True` if the `fundamentals` dict is non-empty.

### How StrategyAgent.think() currently works
`StrategyAgent.think()` is in `/Users/rajesh/athena/agents/strategy_agent.py` (around line 113).

It reads `market_data` from `context.metadata.get("market_data", {})` and uses `prices`, `volatility`, `trend_strength` to build `strategy_params`.

The returned `thought` dict contains `"strategy"`, `"rationale"`, `"action"`, `"memory_context"`, `"latent_messages"`, `"done"`, and optionally `"llm_analysis"`.

The implementation must:
1. Read `fundamentals = market_data.get("fundamentals", {})` in `think()`.
2. If `fundamentals` is non-empty, construct a `valuation_context` dict:
   ```python
   valuation_context = {
       "pe_ratio": fundamentals.get("pe_ratio"),
       "forward_pe": fundamentals.get("forward_pe"),
       "revenue_growth": fundamentals.get("revenue_growth"),
       "earnings_growth": fundamentals.get("earnings_growth"),
       "analyst_target": fundamentals.get("analyst_target"),
       "valuation_signal": self._derive_valuation_signal(fundamentals),
   }
   ```
3. Add `"valuation_context": valuation_context` to the `thought` dict (or `{}` if no fundamentals).
4. Add a private helper `_derive_valuation_signal(fundamentals: dict) -> str` that returns:
   - `"cheap"` if `forward_pe` is not None and `forward_pe < 15`
   - `"expensive"` if `forward_pe` is not None and `forward_pe > 35`
   - `"growth"` if `revenue_growth` is not None and `revenue_growth > 0.20`
   - `"value"` if `pe_ratio` is not None and `pe_ratio < 12`
   - `"neutral"` otherwise (including when all relevant fields are None)
   - If multiple conditions match, return the first that matches in the order listed above.

The `valuation_context` should also be passed to the LLM prompt in `think()` when `llm_analysis` is requested. Update the `_llm_reason` call to include:
```python
f"Valuation context: {valuation_context}. " + existing_prompt
```

## Scope & Constraints
- **May modify:** `cli.py`, `agents/risk_manager.py`, `agents/strategy_agent.py`
- **May NOT modify:** any test files, `agents/coordinator.py`, `agents/market_analyst.py`, `agents/execution_agent.py`, `trading/`, `memory/`, `training/`, `communication/`, `core/`, `models/`, `ingest/`
- No hardcoded absolute paths. Use `Path(__file__).resolve().parent` as the anchor in `cli.py`.
- All new code must use `%s`-style logger calls (not f-strings): e.g. `logger.warning("Failed ...: %s", e)`.
- `_apply_fundamental_risk_adjustments` must be a synchronous method (no `async`).
- `_derive_valuation_signal` must be a synchronous method (no `async`).
- Risk elevation must be capped at `"high"` — never elevate beyond `"high"`. If `_determine_risk_level()` returns `"critical"`, pass it through unchanged (no elevation needed; `"critical"` is already the maximum).
- Conditions are evaluated independently. If both `debt_to_equity > 2.0` AND `current_ratio < 1.0` apply, apply both elevations (so low → high is possible in one call).
- When a fundamentals field is `None` or missing, that condition is simply skipped (do not treat None as 0 or as meeting a threshold).
- **This task brief is the canonical implementation spec.** The design document at `plans/sprint13-agent-intelligence/designs/fundamentals_wiring.md` describes an aspirational v2 (23 extracted fields, signal-strength modulation, `_assess_fundamental_risk()` returning a flags dict, changes to `act()`, `_generate_risk_alerts()`, and `_generate_momentum_signals()`). Do not implement features from the design doc that are not in this brief.

## Input
- `/Users/rajesh/athena/cli.py` — `analyze_symbol()` method; `_FINNHUB_DIR` and `_load_news()` pattern to replicate
- `/Users/rajesh/athena/agents/risk_manager.py` — `think()`, `_determine_risk_level()`, `act()` to understand existing risk flow
- `/Users/rajesh/athena/agents/strategy_agent.py` — `think()`, `act()`, `_generate_signals()` to understand existing strategy flow
- `data/raw/yfinance/{SYMBOL}_fundamentals_{YYYYMMDD}.json` — input data (may not exist if ingest not run; loader must handle absence gracefully)

## Expected Output

### cli.py additions
1. Two new module-level constants after the existing `_FINNHUB_DIR` / `_MAX_NEWS_ARTICLES` block:
   ```python
   _YFINANCE_DIR = Path(__file__).resolve().parent / "data" / "raw" / "yfinance"

   _FUNDAMENTALS_KEYS = [
       "pe_ratio", "forward_pe", "debt_to_equity", "current_ratio",
       "revenue_growth", "earnings_growth", "profit_margin",
       "return_on_equity", "beta", "sector",
   ]
   ```
2. A new `_load_fundamentals(symbol: str) -> dict` function that:
   - Returns `{}` if `_YFINANCE_DIR` does not exist
   - Globs for `{symbol.upper()}_fundamentals_*.json`, sorts, takes the latest
   - Returns `{}` if no files found
   - Loads JSON; extracts only the keys in `_FUNDAMENTALS_KEYS` plus `"analyst_target_price"` (remapped to `"analyst_target"` in the returned dict)
   - Returns `{}` on any `Exception`, logging at `WARNING` level
3. In `analyze_symbol()`, immediately after the `_load_news` block:
   ```python
   fundamentals = _load_fundamentals(symbol)
   if fundamentals:
       market_data["fundamentals"] = fundamentals
   ```

### agents/risk_manager.py additions
1. New synchronous method `_apply_fundamental_risk_adjustments(self, risk_level: str, fundamentals: dict) -> str`
2. Modified `think()`: reads fundamentals from `context.metadata.get("market_data", {}).get("fundamentals", {})`, calls the new method, adds `"fundamentals_used"` key to returned dict.

### agents/strategy_agent.py additions
1. New synchronous method `_derive_valuation_signal(self, fundamentals: dict) -> str`
2. Modified `think()`: reads fundamentals from `market_data`, builds `valuation_context`, adds it to `thought`, prepends valuation context to the `_llm_reason` prompt.

## Acceptance Criteria
- [ ] `_load_fundamentals(symbol)` returns a non-empty dict when a `{SYMBOL}_fundamentals_*.json` file exists under `data/raw/yfinance/`
- [ ] `market_data["fundamentals"]` contains `"analyst_target"` (not `"analyst_target_price"`) and all keys from `_FUNDAMENTALS_KEYS`
- [ ] When no fundamentals file exists, no exception is raised and `market_data` does not get a `"fundamentals"` key
- [ ] Malformed fundamentals JSON is caught and logs a WARNING (not raises)
- [ ] `_apply_fundamental_risk_adjustments`: `debt_to_equity=2.5` with initial risk_level `"low"` returns `"medium"`
- [ ] `_apply_fundamental_risk_adjustments`: `debt_to_equity=2.5` AND `current_ratio=0.8` with initial `"low"` returns `"high"` (two elevations)
- [ ] `_apply_fundamental_risk_adjustments`: `beta=1.8` appends a string containing "High beta" to the alerts list and compliance_issues list (via side-effect on the lists passed in, OR via the return value — see implementation note below)
- [ ] `_apply_fundamental_risk_adjustments`: None values for any metric are skipped without error
- [ ] `_derive_valuation_signal`: `forward_pe=12` returns `"cheap"`; `forward_pe=40` returns `"expensive"`; `revenue_growth=0.25` returns `"growth"`; `pe_ratio=10` (no forward_pe) returns `"value"`; all None returns `"neutral"`
- [ ] `thought["valuation_context"]` is present in StrategyAgent output (as `{}` if no fundamentals file)
- [ ] `pytest tests/ -q` passes — all existing tests green, no new failures (baseline: 181 passed, 4 skipped per TASK-094)

**Implementation note on beta side-effects:** Because `_apply_fundamental_risk_adjustments` only returns `risk_level` (a string), the beta alert must be added to `self`-owned state or the lists must be passed in. The recommended approach is: pass `alerts` and `compliance_issues` as additional parameters and mutate them in-place:
```python
def _apply_fundamental_risk_adjustments(
    self, risk_level: str, fundamentals: dict,
    alerts: list, compliance_issues: list
) -> str:
```
Then call it as:
```python
risk_level = self._apply_fundamental_risk_adjustments(
    risk_level, fundamentals, alerts, all_issues
)
```
where `alerts` and `all_issues` are the lists already constructed by `think()`. Adjust the acceptance criterion accordingly — the test for beta will check the lists returned in the `think()` output dict.

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes

**Senior-dev review 2026-03-01 — APPROVED WITH NOTES (after revisions)**

Revisions applied:
- Added `"critical"` pass-through to Scope & Constraints
- Corrected `_derive_valuation_signal(fundamentals)` → `self._derive_valuation_signal(fundamentals)` in the valuation_context construction
- Added design-doc-is-aspirational note to Scope & Constraints
- Design doc marked with deferred sections note

Remaining known design consideration: `_apply_fundamental_risk_adjustments` mutates `alerts` and `compliance_issues` lists in-place while also returning `risk_level`. This is pragmatic but note the side-effect pattern — documented in the Implementation Note above.
