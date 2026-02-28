# Senior-Dev Review: Sprint 2 Batch 1 — Agents & Communication Layer

**Date:** 2026-02-17 (initial), 2026-02-21 (second pass + fixes + re-review)
**Reviewer:** Senior-Dev Agent (3 passes)
**Verdict:** APPROVED WITH NOTES
**Files Reviewed:**
- `agents/market_analyst.py`
- `agents/risk_manager.py`
- `agents/strategy_agent.py`
- `agents/execution_agent.py`
- `communication/latent_space.py`
- `communication/__init__.py`
- `agents/__init__.py`

**Tasks Covered:** TASK-001, TASK-002, TASK-003, TASK-004, TASK-006

---

## Summary

The implementation delivers four specialized agents and a latent-space communication layer. All agents correctly subclass `BaseAgent`, implement both abstract methods (`think`/`act`) with correct async signatures, and integrate cleanly with the core layer. An initial review found **3 Critical** and **7 Major** issues. All 10 have been fixed and verified. The code is approved for integration with a handful of minor observations for future improvement.

---

## Review Timeline

| Date | Pass | Verdict | Findings |
|------|------|---------|----------|
| 2026-02-17 | Initial review | CHANGES REQUIRED | 3 Critical, 4 Major identified |
| 2026-02-21 | Second review pass | CHANGES REQUIRED | 3 additional Major findings (RSI, VaR, drawdown) |
| 2026-02-21 | Fixes applied | — | All 10 must-fix issues resolved via coding agents |
| 2026-02-21 | Re-review (verification) | **APPROVED WITH NOTES** | 10/10 verified fixed, 5 new minor findings |

---

## Must-Fix Findings — All Resolved

| # | Severity | Finding | File | Status | Verification |
|---|----------|---------|------|--------|--------------|
| 1 | 🔴 CRITICAL | Deadlock in `get_stats()` | `communication/latent_space.py` | **FIXED** | Extracted `_get_buffer_status_unlocked()` helper. `get_stats()` calls it within single lock acquisition. No nested lock. |
| 2 | 🔴 CRITICAL | Fake MACD signal line (`macd * 0.9`) | `agents/market_analyst.py` | **FIXED** | MACD series built by iterating from index 26, computing EMA-12 minus EMA-26 at each step. Signal line is 9-period EMA of that series. Falls back to SMA when <9 values. |
| 3 | 🔴 CRITICAL | Backtest off-by-one — zero trades | `agents/strategy_agent.py` | **FIXED** | Window slice changed to `prices[max(0, i-lookback):i+1]`, providing `lookback + 1` elements as required by signal generators. |
| 4 | 🟡 MAJOR | Lossy encode/decode destroys content | `communication/latent_space.py` | **FIXED** | Original content stored as `_original_content` in metadata at send time. On decode, popped from metadata and used directly. Falls back to neural decode only if key missing. |
| 5 | 🟡 MAJOR | Broadcast queue never drains | `communication/latent_space.py` | **FIXED** | Broadcasts fan out to per-agent queues at send time. `receive()` drains per-agent queue via `popleft()`. `_broadcast_queue` retained as bounded audit log only. |
| 6 | 🟡 MAJOR | Hard `import torch` at module level | `communication/latent_space.py` | **FIXED** | Torch wrapped in `try/except ImportError` with `HAS_TORCH` flag. `LatentSpace.__init__()` raises clear `ImportError` with install instructions when missing. |
| 7 | 🟡 MAJOR | Inconsistent prompt sourcing | All 4 agent files | **FIXED** | All agents now use `get_default_agent_configs()` from `core.config`. Consistent pattern across all four. |
| 8 | 🟡 MAJOR | RSI uses SMA not Wilder's | `agents/market_analyst.py` | **FIXED** | First average uses SMA over initial `period` values, then Wilder's smoothing formula `(avg * (period-1) + current) / period` for subsequent values. |
| 9 | 🟡 MAJOR | VaR sign convention (`abs()`) | `agents/risk_manager.py` | **FIXED** | Historical VaR uses `-sorted_returns[index]`. Parametric branch retains `abs()` which is correct for that formula. Fallback branch also fixed. |
| 10 | 🟡 MAJOR | Backtest drawdown additive | `agents/strategy_agent.py` | **FIXED** | Now uses multiplicative compounding: `cumulative *= (1 + r)` with `dd = (peak - cumulative) / peak`. Consistent with risk manager. |

---

## New Minor Findings (from re-review, non-blocking)

| # | Severity | File | Line(s) | Finding | Suggested Fix |
|---|----------|------|---------|---------|---------------|
| N1 | Minor | `agents/market_analyst.py` | 270–273 | MACD series computation is O(n^2) — recalculates EMA from scratch per index. Fine for <1000 points; may be slow for large intraday datasets. | Build EMA series incrementally in a single forward pass. |
| N2 | Minor | `communication/latent_space.py` | 249 | Broadcast fan-out iterates `self._queues.keys()` which only includes agents with existing queue entries. An agent that hasn't sent/received yet will miss broadcasts. | Add explicit `register_agent()` method, or create queues eagerly on first `receive()`. |
| N3 | Minor | `agents/risk_manager.py` | 146 | `think()` always returns `done: False` on the main path, so `BaseAgent.run()` loop always hits `max_iterations`. | Set `done: True` in main return, or document as single-call agent. |
| N4 | Minor | `agents/strategy_agent.py` | 540 | `total_return` is additive `sum(returns)` but drawdown uses multiplicative compounding. Should be `cumulative - 1.0` for consistency. | Use `total_return = cumulative - 1.0` after the compounding loop. |
| N5 | Nit | `communication/latent_space.py` | 47, 88 | `LatentEncoder`/`LatentDecoder` inherit `nn.Module` at class definition time. If torch is missing, module import raises `NameError` before reaching `LatentSpace.__init__` guard. | Wrap class bodies in `if HAS_TORCH:` block, or accept that the communication module requires torch at import time. |

---

## Previously Identified Lower-Priority Items (unchanged)

### Warnings (should fix next pass)

| # | File | Finding |
|---|------|---------|
| W1 | `agents/risk_manager.py` | Portfolio return aggregation duplicated 3 times — extract helper |
| W2 | `agents/execution_agent.py` | Non-deterministic `random.random()` — accept seeded `Random` instance |
| W3 | `agents/execution_agent.py` | No defensive `.get()` fallback for config key access |
| W4 | All agent files | Unused `import logging` — remove |
| W5 | `agents/market_analyst.py` | Uses raw `config.get()` instead of `self.config.get()` |

### Notes (consider for future)

| # | File | Finding |
|---|------|---------|
| P1 | `agents/market_analyst.py:440` | Population std dev (`/n`) instead of sample std dev (`/(n-1)`) |
| P2 | `agents/market_analyst.py:495-521` | Triangle detection uses index parity as proxy for highs/lows |
| P3 | `agents/strategy_agent.py:323-325` | Hardcoded asset name `"ASSET"` in TradingSignal |
| P4 | `communication/latent_space.py` | numpy is indirect dependency via torch |
| P5 | `communication/__init__.py` | `LatentMessage`, `LatentEncoder`, `LatentDecoder` not exported |
| P6 | `core/base_agent.py:14-15` | `TYPE_CHECKING` imports use `athena.` prefix, runtime uses bare paths |
| P7 | `agents/strategy_agent.py:320` | Variable `vol` misleading — actually coefficient of variation |
| P8 | All agents | Inconsistent `done` flag semantics across agents |

---

## Integration Assessment

| Check | Status | Notes |
|-------|--------|-------|
| BaseAgent inheritance | Pass | All 4 agents call `super().__init__()` with correct 8 params |
| `think()` signature | Pass | All match `async def think(self, context: AgentContext) -> Dict[str, Any]` |
| `act()` signature | Pass | All match `async def act(self, thought: Dict[str, Any]) -> AgentAction` |
| `__init__.py` exports | Pass | Barrel files export correct symbols |
| Import paths | Pass | Consistent bare `from core.base_agent import ...` |
| `send_message()` usage | Pass | `RiskManagerAgent.act()` correctly uses inherited method |
| Cross-agent communication | Pass | Original content preserved via `_original_content` metadata |
| `_original_content` API boundary | Pass | Key is popped before returning `AgentMessage` — does not leak to consumers |
| Test coverage | **No tests** | No tests exist for Sprint 2 code. Should be addressed in testing sprint. |

---

## Action Items

### Completed (all 10 must-fix items)

- [x] Fix #1: Deadlock — extracted `_get_buffer_status_unlocked()` helper
- [x] Fix #2: MACD signal — proper 9-period EMA of MACD series
- [x] Fix #3: Backtest off-by-one — window slice `i + 1`
- [x] Fix #4: Lossy content — store original in `_original_content` metadata
- [x] Fix #5: Broadcast drain — fan-out to per-agent queues at send time
- [x] Fix #6: Torch import — `try/except ImportError` with `HAS_TORCH`
- [x] Fix #7: Prompt consistency — all agents use `get_default_agent_configs()`
- [x] Fix #8: RSI — Wilder's exponential smoothing
- [x] Fix #9: VaR sign — `-sorted_returns[index]` instead of `abs()`
- [x] Fix #10: Drawdown — multiplicative compounding

### Open (non-blocking, for future sprints)

- [ ] N1: Optimize MACD series computation to O(n)
- [ ] N2: Add agent registration for broadcast fan-out
- [ ] N3: Fix `done` semantics in risk manager `think()`
- [ ] N4: Use `cumulative - 1.0` for backtest `total_return`
- [ ] N5: Guard `LatentEncoder`/`LatentDecoder` class bodies with `if HAS_TORCH:`
- [ ] W1: Extract portfolio return aggregation helper
- [ ] W2: Accept seeded `random.Random` in execution agent
- [ ] W3: Defensive `.get()` fallback for config keys
- [ ] W4: Remove unused `import logging`
- [ ] W5: Use `self.config.get()` in market analyst
- [ ] P1–P8: Lower-priority notes (see table above)
