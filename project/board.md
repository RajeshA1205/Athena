# Task Board

## Sprint 2: Parallel Layer Implementation ✅ COMPLETE

### 🔴 Critical
- [x] TASK-005: Implement Coordinator Agent ✅ Accepted

### 🟡 High
- [x] TASK-001: Implement Market Analyst Agent ✅ Accepted
- [x] TASK-002: Implement Risk Manager Agent ✅ Accepted
- [x] TASK-003: Implement Strategy Agent ✅ Accepted
- [x] TASK-004: Implement Execution Agent ✅ Accepted
- [x] TASK-006: Implement LatentMAS Shared Latent Space ✅ Accepted
- [x] TASK-007: Implement LatentMAS Encoder ✅ Accepted
- [x] TASK-008: Implement LatentMAS Decoder ✅ Accepted

### 🟢 Medium
- [x] TASK-009: Implement LatentMAS Router ✅ Accepted
- [x] TASK-010: Implement AgentEvolver Workflow Discovery ✅ Accepted
- [x] TASK-011: Implement AgentEvolver Agent Generator ✅ Accepted

### ⚪ Low
- [x] TASK-012: Implement AgentEvolver Cooperative Evolution ✅ Accepted

---

## Sprint 3: Layer Integration ✅ COMPLETE

### 🔴 Critical
- [x] TASK-015: Integrate Agents with AgeMem Memory Layer ✅ Accepted
- [x] TASK-016: Integrate Agents with LatentMAS Communication ✅ Accepted

### 🟡 High
- [x] TASK-017: Implement End-to-End Pipeline Integration Test ✅ Accepted

---

## Sprint 4: Advanced Features & Learning Layer ✅ COMPLETE

### 🟢 Medium
- [x] TASK-013: Implement Nested Learning Framework ✅ Accepted
- [x] TASK-014: Implement RepExp Exploration Module ✅ Accepted

---

## Sprint 5: Trading Domain & Testing ✅ COMPLETE

### 🟡 High
- [x] TASK-018: Create Trading Market Data Module ✅ Accepted
- [x] TASK-019: Create Trading Order Management Module ✅ Accepted
- [x] TASK-020: Create Trading Portfolio Module ✅ Accepted
- [x] TASK-023: Create Comprehensive Test Suite ✅ Accepted

### 🟢 Medium
- [x] TASK-021: Create Data Scrapers for Training Pipeline ✅ Accepted
- [x] TASK-022: Create Data Processors and Dataset Classes ✅ Accepted

---

---

## Sprint 6: Code Quality, Security, and Correctness Fixes -- COMPLETE

Goal: Address all confirmed findings from two external code reviews. Bring ATHENA from
"feature-complete prototype" to "production-ready baseline." Must not break any of the
existing 171 tests.

All 22 tasks implemented directly in main context (sub-agents lacked write permissions).
Tests confirmed green at completion: 171 passed, 6 skipped.

Resolution notes for previously flagged gaps:
- TASK-038: Docstring now reads "demand-proportional allocation" (fixed 2026-02-24).
- TASK-029: Confirmed complete — OLMoE wired via `llm` param and `_llm_reason()` in
  BaseAgent; active in CoordinatorAgent and StrategyAgent.

### 🔴 Critical

- [x] TASK-024: Fix GRPO reference model aliasing and action log-prob placeholder ✅ Accepted
  - Files: `training/stage2_agemem/grpo.py`
  - Fix alias `self.reference_model = self.policy_model` to use `copy.deepcopy`;
    remove hardcoded-zero log-prob stubs from `_get_action_logprob` and `_get_reference_logprob`
  - Dependencies: None

- [x] TASK-025: Add weights_only=True to torch.load (unsafe deserialization) ✅ Accepted
  - Files: `training/stage2_agemem/grpo.py` (line 326)
  - Prevents arbitrary code execution via malicious pickle payloads
  - Dependencies: None

### 🟡 High

- [x] TASK-026: Unify trading enums into a canonical module ✅ Accepted
  - Files: `trading/enums.py` (created), `trading/__init__.py`, `agents/execution_agent.py`,
    `trading/order_management.py`
  - Eliminates duplicate `OrderType`/`OrderSide`/`OrderStatus` definitions in two files
  - Dependencies: None

- [x] TASK-027: Replace hash() with _stable_hash in order_management ✅ Accepted
  - Files: `trading/order_management.py` (lines 272, 275)
  - Makes fill simulation deterministic across Python sessions
  - Dependencies: None

- [x] TASK-028: Align AgentStateEncoder default latent_dim to 512 ✅ Accepted
  - Files: `communication/encoder.py` (line 52)
  - Fixes dimension mismatch between encoder (256) and LatentSpace default (512)
  - Dependencies: None

- [x] TASK-029: Wire OLMoE model into the agent framework ✅ Accepted
  - Files: `models/olmoe.py`, `core/config.py`, `core/base_agent.py`, `agents/coordinator.py`,
    `agents/strategy_agent.py`
  - OLMoE wired via `llm` param + `_llm_reason()` helper in BaseAgent; used in
    CoordinatorAgent and StrategyAgent; guarded behind TRANSFORMERS_AVAILABLE
  - Dependencies: None

- [x] TASK-030: Create production entry point (main.py) ✅ Accepted
  - Files: `main.py` (created at project root)
  - CLI with `--mode` (paper-trade/backtest/dry-run), graceful shutdown, full wiring
  - Dependencies: TASK-026, TASK-036

### 🟢 Medium

- [x] TASK-031: Optimize MACD calculation to O(n) incremental EMA ✅ Accepted
  - Files: `agents/market_analyst.py`
  - Replace O(n^2) per-index EMA recomputation with a single forward pass each
  - Dependencies: None

- [x] TASK-032: Fix backtest total_return to use geometric model ✅ Accepted
  - Files: `agents/strategy_agent.py`
  - Aligned `total_return` with the geometric model already used for drawdown
  - Dependencies: None

- [x] TASK-033: Seed random in execution_agent._simulate_fill ✅ Accepted
  - Files: `agents/execution_agent.py`
  - Added per-instance `self._rng = random.Random(seed)` for reproducible fill simulation
  - Dependencies: None

- [x] TASK-034: Bound adaptation_history and action_history ✅ Accepted
  - Files: `learning/nested_learning.py`, `core/base_agent.py`
  - Converted unbounded lists to `collections.deque` with maxlen caps
  - Dependencies: None

- [x] TASK-035: Replace blocking I/O in async methods with asyncio.to_thread ✅ Accepted
  - Files: `learning/nested_learning.py`, `learning/repexp.py`
  - Wrapped `json.dump` / `open()` calls in `asyncio.to_thread`
  - Dependencies: None

- [x] TASK-036: Fix duplicate logging handlers in setup_logging ✅ Accepted
  - Files: `core/utils.py`
  - Guard against re-adding handlers on repeated `setup_logging()` calls (idempotent)
  - Dependencies: None

- [x] TASK-037: Deduplicate coordinator memory writes ✅ Accepted
  - Files: `agents/coordinator.py`
  - Consolidated two consecutive `memory.add()` calls into one per `act()` invocation
  - Dependencies: None

- [x] TASK-038: Fix _allocate_resources docstring mismatch ✅ Accepted
  - Files: `agents/coordinator.py`
  - Docstring now reads "demand-proportional allocation"; `"method"` metadata string updated
  - Dependencies: None

- [x] TASK-039: Replace weak ID generation with secrets/uuid ✅ Accepted
  - Files: `core/utils.py`, `memory/operations.py`, `communication/latent_space.py`
  - Replaced `random.random()`-based IDs with `secrets.token_hex`
  - Dependencies: None

### ⚪ Low

- [x] TASK-040: Use timezone-aware timestamps throughout ✅ Accepted
  - Files: `core/utils.py`, `memory/operations.py`
  - Changed `datetime.now()` to `datetime.now(timezone.utc)` for UTC-aware output
  - Dependencies: None

- [x] TASK-041: Guard numpy import in utils.py ✅ Accepted
  - Files: `core/utils.py`
  - Wrapped `import numpy as np` in try/except so the module is importable without numpy
  - Dependencies: None

- [x] TASK-042: Add recursion depth limit to deep_merge ✅ Accepted
  - Files: `core/utils.py`
  - Added `max_depth=20` parameter to prevent stack overflow on deep/circular dicts
  - Dependencies: None

- [x] TASK-043: Fix RiskManager.think() hardcoded done flag ✅ Accepted
  - Files: `agents/risk_manager.py`
  - Changed `"done": False` to `"done": True` since `think()` completes synchronously
  - Dependencies: None

- [x] TASK-044: Document hardcoded quality rewards in agemem ✅ Accepted
  - Files: `memory/agemem.py`
  - Added docstring noting placeholder values (0.8/0.9) and made them configurable
  - Dependencies: None

- [x] TASK-045: Convert f-string logging to lazy %s-style in agent files ✅ Accepted
  - Files: `agents/coordinator.py`, `agents/execution_agent.py`, `agents/market_analyst.py`,
    `agents/risk_manager.py`, `agents/strategy_agent.py`
  - Replaced `logger.info(f"...")` with `logger.info("...", arg)` throughout
  - Dependencies: None

---

## Sprint 7: Rev 3 Review Fixes — COMPLETE ✅

*Addressing 11 confirmed findings from the Rev 3 post-Sprint-6 code review. 2 false positives (S-2 and I-1) were already fixed in Sprint 6.*
*Plan: /Users/rajesh/athena/plans/sprint7-rev3-fixes/ | Designs: plans/sprint7-rev3-fixes/designs/*

### 🔴 Blocker
- [x] TASK-046: Wire OLMoE `encode()` and `action_head` for GRPO ✅ Accepted
  - `models/olmoe.py`: add `encode(text) -> torch.Tensor`; attach `MemoryActionHead` as `self.action_head`
  - Dependencies: None
- [x] TASK-047: Wire `MarketDataFeed` into `main.py` paper-trade/backtest loops ✅ Accepted
  - `main.py`: populate `AgentContext.metadata` with real mock market data per mode
  - Dependencies: None

### 🟡 Should-fix
- [x] TASK-048: Secure `torch.load` fallback in GRPO checkpoint loading ✅ Accepted
  - `training/stage2_agemem/grpo.py`: removed unsafe `except TypeError` fallback; `weights_only=True` enforced; PyTorch < 2.0 version warning added
  - Dependencies: None
- [x] TASK-049: Wire real AgeMem operations into trajectory collection ✅ Accepted
  - `training/stage2_agemem/trainer.py`: `_execute_operation()` dispatches to real async AgeMem calls; trajectory collectors and `train()`/`train_stage()` made async
  - Dependencies: TASK-046 ✅
- [x] TASK-050: Implement distinct unified trajectory logic for Stage 3 ✅ Accepted
  - `training/stage2_agemem/trainer.py`: `_collect_unified_trajectory()` now distinct (4-8 steps, model-driven op selection, trajectory bonus)
  - Dependencies: TASK-049 ✅
- [x] TASK-051: Convert f-string logger calls to `%s`-style (finetune.py + trainer.py) ✅ Accepted
  - `training/stage1_finetune/finetune.py` (3 calls), `training/stage2_agemem/trainer.py` (5 calls)
  - Dependencies: None

### 🟢 Informational
- [x] TASK-052: Parse `olmoe` key in `AthenaConfig._from_dict()` ✅ Accepted
  - `core/config.py`: `olmoe` section now parsed so YAML/JSON config populates OLMoEIntegrationConfig
  - Dependencies: None
- [x] TASK-053: Fix missing `await` on async `get_stats()` in test_memory.py ✅ Accepted
  - `tests/test_memory.py`: `await` added; `test_get_stats` converted to async; two tests no longer skip
  - Note: baseline is now 173 passed, 4 skipped
  - Dependencies: None
- [x] TASK-054: Replace `hash()` with `_stable_hash()` in `scrape_macro_indicators` ✅ Accepted
  - `training/data/scrapers/market.py:202`: deterministic `_stable_hash(indicator)` replaces `hash()`
  - Dependencies: None

### Gate
- [x] TASK-055: Full test suite verification gate ✅ Accepted
  - `pytest tests/ -q` → 173 passed, 4 skipped; `main.py --mode dry-run` → clean shutdown
  - Dependencies: TASK-046 through TASK-054 — all complete

---

## Sprint 8: Rev 4 Review Fixes — COMPLETE ✅

*Addressing 6 findings from the Rev 4 post-Sprint-7 senior-dev code review.*
*Plan: /Users/rajesh/athena/plans/sprint8-review-fixes/ | Designs: plans/sprint8-review-fixes/designs/*

### 🔴 Critical

- [x] TASK-056: Fix `_ACTION_TO_IDX` key case mismatch in GRPO ✅ Accepted
  - `training/stage2_agemem/grpo.py`: `op.name` → `op.value`; action lookups now correctly index all 6 operations
  - Dependencies: None

- [x] TASK-057: Replace `hash()` with stable hash in trainer delete path ✅ Accepted
  - `training/stage2_agemem/trainer.py`: added `import hashlib`; `str(hash(content))` → `hashlib.sha256(content.encode()).hexdigest()[:16]`
  - Dependencies: None

### 🟡 Major / Minor

- [x] TASK-058: Compute real action log-probs at trajectory collection time ✅ Accepted
  - `training/stage2_agemem/trainer.py`: added `_get_logprob_for_step()`; all three collectors now use real log-probs (fallback to -1.0 when model not loaded)
  - Dependencies: TASK-056 ✅

- [x] TASK-059: Log debug message on model-driven selection failure ✅ Accepted
  - `training/stage2_agemem/trainer.py`: `except Exception: pass` → `except Exception as e: self.logger.debug(...)`
  - Dependencies: None

- [x] TASK-060: Fix MarketDataFeed placement and private attribute access in main.py ✅ Accepted
  - `trading/market_data.py`: `_MOCK_SYMBOLS` → `MOCK_SYMBOLS` (public); `main.py`: feed moved into paper-trade/backtest branches only
  - Dependencies: None

### ⚪ Nit

- [x] TASK-061: Convert remaining f-string logger calls in olmoe.py and grpo.py ✅ Accepted
  - `models/olmoe.py` (8 calls), `training/stage2_agemem/grpo.py` (1 call) converted to `%s`-style
  - Dependencies: None

---

## Sprint 9: Rev 5 Review Fixes — IN PROGRESS

*Addressing Critical and Major findings from the Rev 5 full-codebase senior-dev review (Sprints 2–8 scope).*
*Plan: /Users/rajesh/athena/plans/sprint9-review-fixes/*

Goal: Fix correctness bugs in the LLM reasoning path, market data wiring, order type unification, GRPO logprob sentinel, and portfolio exposure calculation.

### 🔴 Critical

- [ ] TASK-062: Fix timezone-naive datetime.now() in ExecutionAgent [ ] Queued
  - `agents/execution_agent.py`: replace all `datetime.now()` with `datetime.now(timezone.utc)`
  - Dependencies: None

- [ ] TASK-063: Fix LatentSpace broadcast missing unregistered agents [ ] Queued
  - `communication/latent_space.py`: add `register_agent()` method; `agents/coordinator.py`: call it in `initialize_communication()`
  - Dependencies: None

- [ ] TASK-064: Fix GRPO sentinel logprob corrupting gradients [ ] Queued
  - `training/stage2_agemem/trainer.py`: change `_get_logprob_for_step` fallback from `-1.0` to `0.0`; PPO ratio becomes neutral `exp(0-0)=1.0`
  - Dependencies: None

### 🟡 Major

- [ ] TASK-065: Remove duplicate Order dataclass from ExecutionAgent [ ] Queued
  - `agents/execution_agent.py`: remove local `Order`; import from `trading.order_management`; `filled_qty` → `filled_quantity`
  - Dependencies: TASK-062

- [ ] TASK-066: Fix _llm_reason async/sync mismatch [ ] Queued
  - `core/base_agent.py`: `return await self.llm.generate(prompt)` directly; remove `asyncio.to_thread` wrapper
  - Dependencies: None

- [ ] TASK-067: Fix portfolio check_limits double-counting exposure on sells [ ] Queued
  - `trading/portfolio.py`: compute signed exposure delta so sells reduce (not inflate) total exposure
  - Dependencies: None

- [ ] TASK-068: Fix main.py market data format mismatch with agent interface [ ] Queued
  - `main.py` / `agents/market_analyst.py`: align context format so `prices` key is non-empty during paper-trade/backtest
  - Dependencies: None

- [ ] TASK-069: Fix Sharpe ratio mean_return calculation in StrategyAgent [ ] Queued
  - `agents/strategy_agent.py` (~line 629): `mean_return = sum(returns) / len(returns)` (arithmetic mean, not compounded total / N)
  - Dependencies: None

### Minor Findings (informational only — no tasks)

- config.py `_from_dict` extra-key fragility
- AgentMessage.timestamp defaults to None
- portfolio.py update_from_fill no lock
- trainer.py type annotation mismatch (List[str] = None)
- coordinator.py max() key style
- base_agent.py TYPE_CHECKING import path inconsistency
- test_agents.py behavioral coverage gaps

---

## Sprint 10: mlx-lm Migration (Apple Silicon Inference) — QUEUED

*Migrate OLMoE inference from HuggingFace transformers + bitsandbytes to mlx-lm as the primary backend on Apple Silicon. Keep transformers as a fallback. Tests must stay at 173 passed, 4 skipped.*

### 🔴 Critical

- [ ] TASK-072: Install mlx-lm and mlx packages for Apple Silicon inference
  - Run `pip install mlx-lm`; verify `from mlx_lm import load, generate` succeeds
  - Dependencies: None

- [ ] TASK-073: Add mlx-lm backend fields to OLMoEConfig and OLMoEIntegrationConfig
  - `models/olmoe.py` (OLMoEConfig) + `core/config.py` (OLMoEIntegrationConfig): add `use_mlx: bool = True` and `mlx_model_path: Optional[str] = None`
  - Dependencies: TASK-072

- [ ] TASK-074: Rewrite OLMoEModel.load() to use mlx-lm as primary backend
  - `models/olmoe.py`: mlx-first load path with transformers fallback; sets `self._backend`; MemoryActionHead only attached on transformers path
  - Dependencies: TASK-073

- [ ] TASK-075: Rewrite OLMoEModel.generate() to use mlx-lm with async wrapper
  - `models/olmoe.py`: mlx path calls `mlx_lm.generate()` via `asyncio.to_thread()`; transformers path unchanged
  - Dependencies: TASK-074

- [ ] TASK-076: Fix OLMoEModel.encode() to bridge mlx arrays to torch tensors
  - `models/olmoe.py`: mlx path converts mlx array → numpy → torch.Tensor (CPU, no grad); transformers path unchanged
  - Dependencies: TASK-074

### 🟡 High

- [ ] TASK-077: Verify MemoryActionHead and GRPO pipeline with new encode() output
  - `models/olmoe.py`: add `get_action_logits()` helper with guard; update MemoryActionHead docstring noting transformers-only availability
  - Dependencies: TASK-076

- [ ] TASK-078: End-to-end smoke test gate for mlx-lm migration
  - `pytest tests/ -q` → 173 passed, 4 skipped; `main.py --mode dry-run` → clean exit, no traceback
  - Dependencies: TASK-072 through TASK-077

---

## Task Summary

**Total Tasks:** 76
- **Queued:** 15 (Sprint 9: 8, Sprint 10: 7)
- **In Progress:** 0
- **Accepted:** 61 (Sprints 2–8: all complete ✅)
- **Rejected:** 0

**By Priority:**
- Critical/Blocker: 16 tasks (9 accepted, 7 queued)
- High/Should-fix/Major: 30 tasks (23 accepted, 7 queued)
- Medium/Informational: 20 tasks (all accepted)
- Low/Gate/Nit: 10 tasks (all accepted)

**By Sprint:**
- Sprint 2 (Parallel Layer Implementation): 12 tasks ✅ COMPLETE
- Sprint 3 (Layer Integration): 3 tasks ✅ COMPLETE
- Sprint 4 (Advanced Features & Learning): 2 tasks ✅ COMPLETE
- Sprint 5 (Trading & Testing): 6 tasks ✅ COMPLETE
- Sprint 6 (Code Quality, Security, Correctness): 22 tasks ✅ COMPLETE
- Sprint 7 (Rev 3 Review Fixes): 10 tasks ✅ COMPLETE
- Sprint 8 (Rev 4 Review Fixes): 6 tasks ✅ COMPLETE
- Sprint 9 (Rev 5 Review Fixes): 8 tasks — IN PROGRESS (0/8 complete)
- Sprint 10 (mlx-lm Migration): 7 tasks — QUEUED (0/7 complete)

---

## Test Results (as of Sprint 8 completion — 2026-02-25)

**173 passed, 4 skipped** (skips = torch not installed in test environment)

| Test File | Tests | Status |
|-----------|-------|--------|
| test_integration_e2e.py | 8 | ✅ All pass |
| test_agents.py | 30 | ✅ All pass |
| test_memory.py | 12 | ✅ All pass |
| test_communication.py | 13 (+4 skip) | ✅ Pass |
| test_evolution.py | 28 | ✅ All pass |
| test_learning.py | 36 | ✅ All pass |
| test_trading.py | 46 | ✅ All pass |
