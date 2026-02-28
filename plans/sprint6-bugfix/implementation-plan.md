# Implementation Plan: Sprint 6 Bug-Fix and Improvements

> Tasks are numbered TASK-024 onward, continuing from the existing board.
> Dependencies are listed explicitly. Size: S (< 1 hr), M (1-4 hr), L (4-8 hr), XL (> 8 hr).

---

## Phase 1: Critical / Blocking Fixes

These must land first -- GRPO is non-functional and the security hole is exploitable.

### TASK-024: Fix GRPO reference model aliasing and action log-prob placeholder [Size: M]
- **Priority:** Critical
- **Findings:** CQ-4, CQ-8
- **Files:**
  - `training/stage2_agemem/grpo.py` (modify)
- **Description:**
  1. Line 312: Replace `self.reference_model = self.policy_model` with
     `import copy; self.reference_model = copy.deepcopy(self.policy_model)`.
     Alternatively, if `policy_model` has a `state_dict()`, use
     `self.reference_model.load_state_dict(self.policy_model.state_dict())`.
     The comment on line 313 already describes the correct approach -- implement
     it instead of leaving the alias.
  2. Line 297-301: `_get_action_logprob` currently returns
     `torch.tensor(0.0, requires_grad=True)`. Either implement a real forward
     pass through `self.policy_model` to obtain the log-probability of the given
     action, or raise `NotImplementedError("Action log-prob must be implemented
     for the specific policy model architecture")` so that the bug surfaces
     immediately instead of silently producing zero gradients.
  3. Line 303-306: Similarly, `_get_reference_logprob` returns a constant 0.0.
     Apply the same treatment -- forward pass through `self.reference_model` or
     `NotImplementedError`.
- **Acceptance Criteria:**
  - `self.reference_model is not self.policy_model` after
    `_update_reference_model()` is called.
  - Mutations to `policy_model` do not affect `reference_model`.
  - `_get_action_logprob` no longer returns a hardcoded constant.
  - Existing tests still pass.
- **Dependencies:** None

### TASK-025: Fix unsafe torch.load deserialization [Size: S]
- **Priority:** Critical
- **Findings:** SEC-1
- **Files:**
  - `training/stage2_agemem/grpo.py` (modify, line 326)
- **Description:**
  Change `torch.load(path)` on line 326 to
  `torch.load(path, weights_only=True)`. This prevents arbitrary code execution
  via malicious pickle payloads. Verify that the saved checkpoint dict (keys:
  `step_count`, `optimizer_state`, `config`) loads correctly with
  `weights_only=True`. If `config` contains non-tensor types that fail under
  `weights_only=True`, restructure the save to separate config from weights, or
  use `torch.load(path, weights_only=False)` with an explicit allowlist comment
  and a TODO to migrate.
- **Acceptance Criteria:**
  - `torch.load` call includes `weights_only=True`.
  - Round-trip save/load still works for GRPO state.
- **Dependencies:** None (can be done in parallel with TASK-024)

---

## Phase 2: High-Priority Fixes

### TASK-026: Unify trading enums into a canonical module [Size: M]
- **Priority:** High
- **Findings:** CQ-6
- **Files:**
  - `trading/enums.py` (create)
  - `trading/__init__.py` (modify -- re-export enums)
  - `agents/execution_agent.py` (modify -- remove local enum definitions,
    import from `trading.enums`)
  - `trading/order_management.py` (modify -- remove local enum definitions,
    import from `trading.enums`)
- **Description:**
  1. Create `trading/enums.py` containing the unified enums:
     - `OrderType`: MARKET, LIMIT, STOP, STOP_LIMIT, TWAP, VWAP (union of both
       current definitions).
     - `OrderSide`: BUY, SELL (identical in both -- no merge needed).
     - `OrderStatus`: PENDING, SUBMITTED, PARTIAL, PARTIALLY_FILLED, FILLED,
       CANCELLED, REJECTED (union of both; decide whether PARTIAL and
       PARTIALLY_FILLED should be consolidated into one -- recommend keeping
       PARTIALLY_FILLED as the canonical name and aliasing PARTIAL for backwards
       compatibility).
  2. In `agents/execution_agent.py`: delete lines 23-46 (local enum
     definitions) and add `from trading.enums import OrderType, OrderSide,
     OrderStatus`.
  3. In `trading/order_management.py`: delete lines 26-47 (local enum
     definitions) and add `from trading.enums import OrderType, OrderSide,
     OrderStatus`.
  4. Update `trading/__init__.py` to re-export the enums.
  5. Run full test suite to verify no comparisons break.
- **Acceptance Criteria:**
  - Only one definition of each enum exists in the codebase.
  - `execution_agent.OrderType.TWAP == order_management.OrderType.TWAP` is
    `True` (same object).
  - All 171 existing tests pass.
- **Dependencies:** None

### TASK-027: Replace hash() with _stable_hash in order_management [Size: S]
- **Priority:** High
- **Findings:** S5
- **Files:**
  - `trading/order_management.py` (modify, lines 272, 275)
- **Description:**
  Lines 272 and 275 use `hash(order.symbol)` for deterministic price generation
  in `_simulate_fill`. The same file already defines `_stable_hash(s)` on
  line 19 using `hashlib.sha256`. Replace both occurrences:
  - `hash(order.symbol)` --> `_stable_hash(order.symbol)`
- **Acceptance Criteria:**
  - No calls to bare `hash()` remain in `order_management.py`.
  - Fill simulation produces identical prices across Python sessions with the
    same symbol.
- **Dependencies:** None

### TASK-028: Align AgentStateEncoder default latent_dim to 512 [Size: S]
- **Priority:** High
- **Findings:** I5
- **Files:**
  - `communication/encoder.py` (modify, line 52)
- **Description:**
  Change the default value of `latent_dim` in `AgentStateEncoder.__init__` from
  `256` to `512` to match the `LatentSpace` default. The `LatentSpace` class
  (lines 58, 101, 161 of `communication/latent_space.py`) consistently uses
  512 as the default. If both are instantiated without explicit arguments, the
  encoder would produce 256-dim vectors that cannot be consumed by a 512-dim
  latent space.
- **Acceptance Criteria:**
  - `AgentStateEncoder()` (no args) produces latent vectors of dimension 512.
  - `AgentStateEncoder().latent_dim == LatentSpace().latent_dim`.
- **Dependencies:** None

### TASK-029: Wire OLMoE model into the agent framework [Size: L]
- **Priority:** High
- **Findings:** B1
- **Files:**
  - `models/olmoe.py` (modify -- ensure public API is clean)
  - `core/config.py` (modify -- add OLMoE config section)
  - `agents/coordinator.py` or a new integration module (modify)
- **Description:**
  The `models/olmoe.py` module defines `OLMoEConfig` and the full model wrapper
  but no agent or training code imports it. The minimum viable integration is:
  1. Add an `olmoe` config entry to `core/config.py` so the model path/device
     can be configured.
  2. Import `OLMoE` in at least the Coordinator or StrategyAgent and expose an
     optional `llm_backend` parameter that, when set to `"olmoe"`, routes
     natural-language reasoning through the model.
  3. Guard behind `TRANSFORMERS_AVAILABLE` so the system still works without
     transformers installed.
  This is a design-heavy task. If full integration exceeds the sprint, the
  minimum deliverable is a documented import path and a smoke test.
- **Acceptance Criteria:**
  - At least one agent can be configured to use OLMoE.
  - System still runs without transformers installed (graceful fallback).
- **Dependencies:** None

### TASK-030: Create production entry point (main.py) [Size: M]
- **Priority:** High
- **Findings:** B3
- **Files:**
  - `main.py` (create at project root `/Users/rajesh/athena/main.py`)
- **Description:**
  There is no top-level script to run the system end-to-end. Create `main.py`
  that:
  1. Parses CLI arguments (config file path, mode: paper-trade / backtest /
     dry-run).
  2. Calls `setup_logging()` (after TASK-036 fix).
  3. Loads config via `core/config.py`.
  4. Instantiates Coordinator, sub-agents, memory, communication layers.
  5. Runs the main loop (`coordinator.act()` in a loop or event-driven).
  6. Handles graceful shutdown (SIGINT/SIGTERM).
  Reference the integration test `tests/test_integration_e2e.py` for the
  existing wiring pattern.
- **Acceptance Criteria:**
  - `python main.py --mode dry-run` completes without error.
  - `python main.py --help` shows available options.
- **Dependencies:** TASK-026 (enums must be unified before wiring everything),
  TASK-036 (logging fix)

---

## Phase 3: Medium-Priority Fixes

### TASK-031: Optimize MACD calculation to O(n) [Size: S]
- **Priority:** Medium
- **Findings:** S2
- **Files:**
  - `agents/market_analyst.py` (modify, lines 344-347)
- **Description:**
  The current MACD loop at lines 344-347 calls `_calculate_ema(price_data[:i],
  period)` for every index `i` from 26 to `len(price_data)`, resulting in O(n^2)
  work. Rewrite to compute the full EMA-12 and EMA-26 series in a single forward
  pass each (standard incremental EMA formula: `ema[i] = alpha * price[i] +
  (1 - alpha) * ema[i-1]`), then derive the MACD series by subtraction.
- **Acceptance Criteria:**
  - MACD values are numerically identical (within floating-point tolerance) to
    the current implementation.
  - Time complexity is O(n).
- **Dependencies:** None

### TASK-032: Fix backtest total_return to use geometric model [Size: S]
- **Priority:** Medium
- **Findings:** S3
- **Files:**
  - `agents/strategy_agent.py` (modify, line 615)
- **Description:**
  Line 615: `total_return = sum(returns)` uses additive returns, but the
  drawdown calculation at lines 620-629 uses `cumulative *= (1 + r)` (geometric
  / multiplicative). Fix `total_return` to use the same geometric model:
  `total_return = cumulative - 1.0` (where `cumulative` is the product of
  `(1 + r)` for all `r` in `returns`, which is already computed on line 624 for
  drawdown). Move the cumulative computation before the `total_return`
  assignment, or compute it separately.
  Also update `mean_return` on line 616 if it is used downstream (it feeds into
  Sharpe ratio; decide whether Sharpe should use arithmetic or geometric mean --
  arithmetic mean of returns is standard for Sharpe, so `mean_return` can stay
  as `sum(returns) / len(returns)` for Sharpe purposes, but `total_return`
  itself should be geometric).
- **Acceptance Criteria:**
  - `total_return` and `max_drawdown` use the same return model (geometric).
  - Sharpe ratio calculation remains correct.
- **Dependencies:** None

### TASK-033: Seed random in execution_agent._simulate_fill [Size: S]
- **Priority:** Medium
- **Findings:** S4
- **Files:**
  - `agents/execution_agent.py` (modify, around line 490)
- **Description:**
  Line 490: `actual_slippage = estimated_slippage * (0.5 + random.random())`
  uses the global unseeded `random` module, producing non-deterministic fills.
  Fix by:
  1. Adding a `seed` parameter to the `ExecutionAgent.__init__` (or accepting it
     via config).
  2. Creating a per-instance `self._rng = random.Random(seed)` in `__init__`.
  3. Replacing `random.random()` with `self._rng.random()`.
- **Acceptance Criteria:**
  - Two `ExecutionAgent` instances with the same seed produce identical fill
    simulations for the same inputs.
- **Dependencies:** None

### TASK-034: Bound adaptation_history and action_history [Size: S]
- **Priority:** Medium
- **Findings:** S7, CQ-7
- **Files:**
  - `learning/nested_learning.py` (modify, line 160)
  - `core/base_agent.py` (modify, line 102)
- **Description:**
  1. `nested_learning.py:160` -- `self.adaptation_history` is an unbounded
     `List`. Convert to `collections.deque(maxlen=1000)` (or another reasonable
     cap). Ensure all `.append()` callers still work (deque supports `.append`).
  2. `core/base_agent.py:102` -- `self.action_history` is an unbounded `List`,
     but only the last 10 are ever read (line 193). Convert to
     `collections.deque(maxlen=100)` (keeping more than 10 for potential future
     use, but capping growth). Update the `reset()` method (line 317) to use
     `self.action_history.clear()` or re-initialize the deque. Verify line 310
     (`len(self.action_history)`) still works (it does -- `deque` supports
     `len`).
- **Acceptance Criteria:**
  - Both collections have a finite upper bound.
  - No existing functionality breaks (append, len, slicing last N).
- **Dependencies:** None

### TASK-035: Replace blocking I/O in async methods with asyncio.to_thread [Size: S]
- **Priority:** Medium
- **Findings:** S8
- **Files:**
  - `learning/nested_learning.py` (modify, line 424)
  - `learning/repexp.py` (modify, line 333)
- **Description:**
  Both files use `with open(path, "w") as f: json.dump(...)` inside `async def`
  methods. This blocks the event loop. Wrap each in `asyncio.to_thread`:
  ```
  # Pseudocode
  def _write_json_sync(path, data):
      with open(path, "w") as f:
          json.dump(data, f, indent=2)

  await asyncio.to_thread(_write_json_sync, path, data)
  ```
  Apply the same pattern to the corresponding `load_state` / `load` methods if
  they also use blocking `open()`.
- **Acceptance Criteria:**
  - No blocking `open()` calls remain inside `async def` methods in these files.
  - Save/load still works correctly.
- **Dependencies:** None

### TASK-036: Fix duplicate logging handlers in setup_logging [Size: S]
- **Priority:** Medium
- **Findings:** INF-2
- **Files:**
  - `core/utils.py` (modify, lines 32-49)
- **Description:**
  `setup_logging()` unconditionally adds a `StreamHandler` (and optionally a
  `FileHandler`) every time it is called. Add a guard:
  ```
  # Pseudocode
  if logger.handlers:
      return logger  # already configured
  ```
  Or check specifically for existing handler types before adding.
- **Acceptance Criteria:**
  - Calling `setup_logging()` multiple times does not produce duplicate log
    output.
- **Dependencies:** None

### TASK-037: Deduplicate coordinator memory writes [Size: S]
- **Priority:** Medium
- **Findings:** S10
- **Files:**
  - `agents/coordinator.py` (modify, lines 318-346)
- **Description:**
  The `act()` method makes two consecutive `memory.add()` calls (lines 320-327
  and 334-344) that store overlapping data. The first stores the full
  coordination result + thought; the second stores a "coordination_summary" with
  `final_decision` and `agents_queried`. Consolidate into a single
  `memory.add()` call that includes both the result and the summary metadata.
- **Acceptance Criteria:**
  - Only one `memory.add()` call per successful `act()` invocation.
  - All previously stored information is still captured.
- **Dependencies:** None

### TASK-038: Fix _allocate_resources docstring to match implementation [Size: S]
- **Priority:** Medium
- **Findings:** S11
- **Files:**
  - `agents/coordinator.py` (modify, line 505)
- **Description:**
  Line 505 docstring says "round-robin approach" but the implementation
  (lines 530-533) computes proportional shares
  (`requested_amount / total_requests`). Additionally, line 539 returns
  `"method": "round_robin"` in the result dict. Fix both:
  1. Change docstring to "proportional allocation" or "demand-weighted
     allocation".
  2. Change the returned method string to `"proportional"`.
- **Acceptance Criteria:**
  - Docstring and returned metadata accurately describe the algorithm.
- **Dependencies:** None

### TASK-039: Replace weak ID generation with secrets/uuid [Size: S]
- **Priority:** Medium
- **Findings:** SEC-2, I3
- **Files:**
  - `core/utils.py` (modify, line 107 -- `generate_id`)
  - `memory/operations.py` (modify, line 416 -- `_generate_id`)
  - `communication/latent_space.py` (modify, line 447 --
    `_generate_message_id`)
- **Description:**
  All three functions use `random.random()` as entropy for ID generation, which
  is predictable and not collision-resistant. Replace with:
  - `core/utils.py:107`: Use `secrets.token_hex(8)` instead of
    `hashlib.md5(str(random.random()).encode()).hexdigest()[:8]`.
  - `memory/operations.py:416`: Use `secrets.token_hex(6)` instead of
    `hashlib.sha256(f"...:{random.random()}").hexdigest()[:12]`.
  - `communication/latent_space.py:447`: Use `secrets.token_hex(8)` instead of
    `hashlib.sha256(f"...:{random.random()}").hexdigest()[:16]`.
  Add `import secrets` to each file.
- **Acceptance Criteria:**
  - No `random.random()` calls remain in ID generation functions.
  - Generated IDs are still unique strings of similar length.
- **Dependencies:** None

---

## Phase 4: Low-Priority / Code Quality

### TASK-040: Use timezone-aware timestamps throughout [Size: S]
- **Priority:** Low
- **Findings:** CQ-3, INF-1
- **Files:**
  - `core/utils.py` (modify, line 199 -- `format_timestamp`)
  - `memory/operations.py` (modify, line 26 -- `ContextItem.timestamp`)
- **Description:**
  1. `core/utils.py:199`: Change `datetime.now()` to
     `datetime.now(timezone.utc)`. Add `from datetime import timezone` if not
     already imported.
  2. `memory/operations.py:26`: Change `datetime.now().isoformat()` to
     `datetime.now(timezone.utc).isoformat()`.
- **Acceptance Criteria:**
  - All default timestamps include timezone info (`+00:00` suffix).
- **Dependencies:** None

### TASK-041: Guard numpy import in utils.py [Size: S]
- **Priority:** Low
- **Findings:** CQ-1
- **Files:**
  - `core/utils.py` (modify, line 9)
- **Description:**
  Line 9 does `import numpy as np` unconditionally. Other optional deps (like
  torch) use try/except guards throughout the codebase. Wrap the numpy import:
  ```
  # Pseudocode
  try:
      import numpy as np
      HAS_NUMPY = True
  except ImportError:
      np = None
      HAS_NUMPY = False
  ```
  Then guard `cosine_similarity` and any other numpy-dependent functions with an
  `if not HAS_NUMPY: raise ImportError(...)` check.
- **Acceptance Criteria:**
  - `core/utils.py` can be imported without numpy installed.
  - Functions that need numpy raise a clear error if it is missing.
- **Dependencies:** None

### TASK-042: Add recursion depth limit to deep_merge [Size: S]
- **Priority:** Low
- **Findings:** CQ-2
- **Files:**
  - `core/utils.py` (modify, lines 165-184)
- **Description:**
  `deep_merge` recursively merges dicts with no depth limit. Add a `max_depth`
  parameter (default 20) and raise `RecursionError` or return the override
  value when depth is exceeded.
- **Acceptance Criteria:**
  - `deep_merge(a, b)` still works for normal inputs (< 20 levels).
  - Deeply nested or circular-reference dicts do not cause a stack overflow.
- **Dependencies:** None

### TASK-043: Fix RiskManager.think() hardcoded done flag [Size: S]
- **Priority:** Low
- **Findings:** S1
- **Files:**
  - `agents/risk_manager.py` (modify, line 181)
- **Description:**
  `think()` returns `"done": False` unconditionally. This should reflect whether
  the risk assessment is actually complete. Change to `"done": True` since the
  `think()` method completes its full analysis before returning. If there is a
  scenario where `think()` should return `done: False` (e.g., async analysis
  pending), document it.
- **Acceptance Criteria:**
  - `think()` returns `"done": True` when analysis is complete.
- **Dependencies:** None

### TASK-044: Document hardcoded quality rewards in agemem [Size: S]
- **Priority:** Low
- **Findings:** I1
- **Files:**
  - `memory/agemem.py` (modify, lines 393-402)
- **Description:**
  `_calculate_quality_reward` returns hardcoded values (0.8 for SUMMARY, 0.9
  for FILTER) with TODO comments. At minimum, add a docstring noting these are
  placeholder values. Optionally, make them configurable via an `AgeMem` config
  parameter. The existing TODOs on lines 399 and 401 should remain as they
  describe the intended future behavior.
- **Acceptance Criteria:**
  - Hardcoded values are documented and/or configurable.
- **Dependencies:** None

### TASK-045: Convert f-string logging to lazy %s-style in agent files [Size: M]
- **Priority:** Low
- **Findings:** INF-3
- **Files:**
  - `agents/coordinator.py` (modify)
  - `agents/execution_agent.py` (modify)
  - `agents/market_analyst.py` (modify)
  - `agents/risk_manager.py` (modify)
  - `agents/strategy_agent.py` (modify)
- **Description:**
  Throughout the agent files, logging calls use f-strings:
  `self.logger.info(f"Broadcast final decision to all agents")` etc.
  Convert to lazy formatting: `self.logger.info("Broadcast final decision to %s agents", count)`.
  This avoids string formatting overhead when the log level is disabled.
  This is a mechanical, low-risk change but touches many lines across 5 files.
- **Acceptance Criteria:**
  - No f-string interpolation inside `logger.debug/info/warning/error` calls in
    agent files.
  - Log output is unchanged.
- **Dependencies:** None

---

## Dependency Graph

```
TASK-024 (GRPO fix)            -- no deps
TASK-025 (torch.load)          -- no deps
TASK-026 (enum unify)          -- no deps
TASK-027 (stable hash)         -- no deps
TASK-028 (latent dim)          -- no deps
TASK-029 (OLMoE integration)   -- no deps
TASK-030 (main.py)             -- depends on TASK-026, TASK-036
TASK-031 (MACD perf)           -- no deps
TASK-032 (backtest return)     -- no deps
TASK-033 (seed random)         -- no deps
TASK-034 (bound lists)         -- no deps
TASK-035 (async I/O)           -- no deps
TASK-036 (logging handlers)    -- no deps
TASK-037 (coordinator dedup)   -- no deps
TASK-038 (docstring fix)       -- no deps
TASK-039 (weak IDs)            -- no deps
TASK-040 (timestamps)          -- no deps
TASK-041 (numpy guard)         -- no deps
TASK-042 (deep_merge depth)    -- no deps
TASK-043 (done flag)           -- no deps
TASK-044 (quality rewards)     -- no deps
TASK-045 (f-string logging)    -- no deps
```

Only TASK-030 has dependencies. All others are independent and can be parallelized.

---

## Summary Table

| Task ID  | Title                                          | Priority | Size | Dependencies     |
|----------|------------------------------------------------|----------|------|------------------|
| TASK-024 | Fix GRPO reference model alias + log-prob stub | Critical | M    | None             |
| TASK-025 | Add weights_only=True to torch.load            | Critical | S    | None             |
| TASK-026 | Unify trading enums into canonical module       | High     | M    | None             |
| TASK-027 | Replace hash() with _stable_hash in order mgmt | High     | S    | None             |
| TASK-028 | Align AgentStateEncoder latent_dim to 512       | High     | S    | None             |
| TASK-029 | Wire OLMoE model into agent framework           | High     | L    | None             |
| TASK-030 | Create production entry point (main.py)         | High     | M    | TASK-026, 036    |
| TASK-031 | Optimize MACD to O(n) incremental EMA           | Medium   | S    | None             |
| TASK-032 | Fix backtest total_return to geometric model    | Medium   | S    | None             |
| TASK-033 | Seed random in execution_agent._simulate_fill   | Medium   | S    | None             |
| TASK-034 | Bound adaptation_history and action_history      | Medium   | S    | None             |
| TASK-035 | Replace blocking I/O in async with to_thread    | Medium   | S    | None             |
| TASK-036 | Fix duplicate logging handlers in setup_logging  | Medium   | S    | None             |
| TASK-037 | Deduplicate coordinator memory writes            | Medium   | S    | None             |
| TASK-038 | Fix _allocate_resources docstring mismatch        | Medium   | S    | None             |
| TASK-039 | Replace weak ID generation with secrets/uuid     | Medium   | S    | None             |
| TASK-040 | Use timezone-aware timestamps throughout          | Low      | S    | None             |
| TASK-041 | Guard numpy import in utils.py                   | Low      | S    | None             |
| TASK-042 | Add recursion depth limit to deep_merge           | Low      | S    | None             |
| TASK-043 | Fix RiskManager.think() hardcoded done flag       | Low      | S    | None             |
| TASK-044 | Document hardcoded quality rewards in agemem      | Low      | S    | None             |
| TASK-045 | Convert f-string logging to lazy %s-style         | Low      | M    | None             |
