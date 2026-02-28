# Implementation Plan: Sprint 7 -- Rev 3 Code Review Fixes

## Dependency Graph

```
TASK-046 (B-1: OLMoE encode + action_head)
    |
    +---> TASK-048 (S-1: torch.load security) [no hard dep, but same file]
    +---> TASK-049 (S-3: real AgeMem trajectories) [needs encode/action_head for logprobs]
              |
              +---> TASK-050 (S-4: unified trajectory) [extends S-3 work]

TASK-047 (B-2: main.py market data wiring) [independent]

TASK-051 (S-5+I-4: logging format) [independent]

TASK-052 (I-2: config olmoe parsing) [independent]

TASK-053 (I-3: test_memory await) [independent]

TASK-054 (residual S-2: stable hash in scraper) [independent]

TASK-055 (test: verify full green) [depends on all above]
```

---

## Phase 1: Blockers

### TASK-046: Wire OLMoE encode() and action_head for GRPO training [Size: M]

**Finding**: B-1
**Priority**: Blocker
**Files**:
- `/Users/rajesh/athena/models/olmoe.py` (modify)

**Description**:
The GRPO training pipeline (`grpo.py:327-330`) calls `model.encode(state_text)` and
`model.action_head(embedding)`. Neither exists on `OLMoEModel`. The fallback at `grpo.py:349-350`
returns `torch.tensor(0.0, requires_grad=True)`, making all training updates no-ops (zero
gradients).

Changes required in `/Users/rajesh/athena/models/olmoe.py`:

1. **Add `encode()` method to `OLMoEModel`** (after `embed()`, around line 295):
   - Signature: `def encode(self, text: str) -> torch.Tensor`
   - Must be **synchronous** (not async) -- GRPO calls it inside sync code paths.
   - Tokenize `text`, run through `self.model` with `output_hidden_states=True`,
     extract `outputs.hidden_states[-1]`, mean-pool over the sequence dimension,
     squeeze to 1-D tensor.
   - Return the tensor **on the model's device** (do NOT call `.cpu()` -- gradients
     must flow through it).
   - Raise `RuntimeError("Model not loaded")` if `not self._is_loaded`.
   - Note: This is similar to the existing `embed()` method but (a) synchronous,
     (b) returns `torch.Tensor` not `List[float]`, (c) keeps the tensor on-device
     with gradient support.

2. **Attach `MemoryActionHead` as `self.action_head` in `__init__`** (around line 84):
   - After `self._has_lora = False`, add `self.action_head = None`.
   - In `load()`, after `self._is_loaded = True` (line 149), instantiate:
     `self.action_head = MemoryActionHead(self.model.config.hidden_size)`
     and move it to the same device: `self.action_head = self.action_head.to(self.model.device)`.
   - Guard with `if TRANSFORMERS_AVAILABLE:` (the real `MemoryActionHead` is only
     defined when torch is available).

3. **Override `parameters()` to include action_head params**:
   - Add a `parameters()` method that yields from both `self.model.parameters()`
     and `self.action_head.parameters()` (if action_head is not None).
   - This is needed so `StepwiseGRPO.setup()` at `grpo.py:115` picks up all trainable
     params in the optimizer.
   - Similarly add `state_dict()` and `load_state_dict()` that include action_head,
     so reference model snapshotting works at `grpo.py:377-380`.

**Acceptance Criteria**:
- `OLMoEModel` has a sync `encode(str) -> torch.Tensor` method.
- `OLMoEModel().action_head` is `None` before `load()`, and a `MemoryActionHead` instance after.
- `grpo.py:349` hasattr checks pass (no more zero-tensor fallback).
- `parameters()` yields params from both the base model and the action head.
- Existing tests pass (171 passed, 6 skipped).

**Dependencies**: None

---

### TASK-047: Wire MarketDataFeed into main.py paper-trade/backtest loops [Size: M]

**Finding**: B-2
**Priority**: Blocker
**Files**:
- `/Users/rajesh/athena/main.py` (modify)

**Description**:
The paper-trade and backtest loops (`main.py:103-114`) create `AgentContext` with only a task
string. No market data is fetched or passed to agents. The `MarketDataFeed` class exists at
`/Users/rajesh/athena/trading/market_data.py` with a working MOCK mode but is never imported
or used in `main.py`.

Changes required in `/Users/rajesh/athena/main.py`:

1. **Import MarketDataFeed and MarketDataMode** at the top of `run()` (line 79 area):
   ```
   from trading.market_data import MarketDataFeed, MarketDataMode
   ```

2. **Instantiate MarketDataFeed** after agent registration (around line 93):
   - Use `MarketDataMode.MOCK` for both paper-trade and backtest modes.
   - Pull symbol list from `config.trading.markets` if non-empty, else use the feed's
     default `_MOCK_SYMBOLS`.

3. **Populate context.metadata with market data** inside the while loop (line 109):
   - For each iteration, call `await feed.get_realtime_data(symbol)` for each symbol.
   - Store in `context.metadata["market_data"]` as a dict mapping symbol to OHLCV
     (converted to dict via `dataclasses.asdict`).
   - This gives all downstream agents (via coordinator) access to market prices.

4. **Differentiate backtest from paper-trade**:
   - For backtest mode: use `get_historical_data()` to prefetch a date range, then
     iterate over bars sequentially (replay) instead of an infinite loop.
   - For paper-trade mode: keep the infinite loop with `asyncio.sleep(1.0)`.

**Acceptance Criteria**:
- `python main.py --mode paper-trade` creates a `MarketDataFeed` and populates
  `context.metadata["market_data"]` each iteration.
- `python main.py --mode backtest` replays historical bars and terminates.
- `python main.py --mode dry-run` is unchanged.
- Existing tests pass.

**Dependencies**: None

---

## Phase 2: Should-Fix

### TASK-048: Secure torch.load in GRPO checkpoint loading [Size: S]

**Finding**: S-1
**Priority**: Should-fix
**Files**:
- `/Users/rajesh/athena/training/stage2_agemem/grpo.py` (modify)

**Description**:
`StepwiseGRPO.load()` at line 394 calls `torch.load(path, weights_only=True)` but catches
`TypeError` at line 395-396 and falls back to bare `torch.load(path)`, which allows arbitrary
code execution via pickle. This fallback was intended for PyTorch < 2.0 compatibility.

Changes required in `/Users/rajesh/athena/training/stage2_agemem/grpo.py`:

1. **Remove the `except TypeError` fallback** (lines 395-396).
2. **Add a version check at module level** (after the torch import block, ~line 18):
   - After `TORCH_AVAILABLE = True`, check `torch.__version__` and warn/error if < 2.0.
   - Pseudocode: if major version < 2, log a warning that GRPO requires PyTorch >= 2.0
     for safe checkpoint loading.
3. **Keep only `torch.load(path, weights_only=True)`** in the `load()` method.

**Acceptance Criteria**:
- `StepwiseGRPO.load()` never calls `torch.load` without `weights_only=True`.
- A clear error message is raised if PyTorch < 2.0 is detected.
- Existing tests pass.

**Dependencies**: None (same file as TASK-046 but different functions, no conflict)

---

### TASK-049: Wire real AgeMem operations into trajectory collection [Size: L]

**Finding**: S-3
**Priority**: Should-fix
**Files**:
- `/Users/rajesh/athena/training/stage2_agemem/trainer.py` (modify)

**Description**:
All three `_collect_*_trajectory()` methods in `trainer.py` simulate operation outcomes using
`random.random()` (e.g., `success = random.random() > 0.2` at lines 246, 291). No actual
AgeMem operations are executed, so the GRPO policy never learns from real memory behavior.

The trainer's `__init__` already receives an `agemem` instance (`self.agemem`) but never uses
it in trajectory collection.

Changes required in `/Users/rajesh/athena/training/stage2_agemem/trainer.py`:

1. **Make `_collect_single_tool_trajectory()` async** and execute real AgeMem operations:
   - For `add`: call `await self.agemem.add(content, metadata)` with synthetic training content.
   - For `update`: call `await self.agemem.update(entry_id, content)`.
   - For `delete`: call `await self.agemem.delete(entry_id)`.
   - For `retrieve`: call `await self.agemem.retrieve(query)`.
   - For `summary`: call `await self.agemem.summary(context)`.
   - For `filter`: call `await self.agemem.filter(context)`.
   - Capture actual success/failure and latency from `MemoryOperationResult` or timing.
   - Use `self.agemem.operation_stats` to get real performance data.

2. **Make `_collect_multi_tool_trajectory()` async** similarly, executing sequences of real
   operations.

3. **Update `_collect_trajectories()`** to be async and await the trajectory collectors.

4. **Update `train()` and `train_stage()`** to use `asyncio.run()` or be async themselves
   to support the async trajectory collection. Since the rest of the GRPO pipeline is sync,
   wrap the async trajectory collection with `asyncio.get_event_loop().run_until_complete()`
   or make the caller async.

5. **Generate synthetic training content** for operations:
   - Create a helper `_generate_training_content()` that produces realistic financial text
     snippets (e.g., "AAPL Q3 earnings beat expectations by 5%") for add/update operations.
   - Use `_stable_hash` or the existing seeded RNG pattern for determinism in tests.

**Acceptance Criteria**:
- Trajectory collection calls real AgeMem async methods.
- `OperationOutcome` objects reflect actual success/failure from AgeMem.
- Latency measurements come from real operation timing.
- The `random.random() > 0.2` simulation pattern is fully removed.
- Existing tests pass.

**Dependencies**: TASK-046 (needs working encode/action_head for meaningful logprobs, though
trajectory collection itself only depends on AgeMem)

---

### TASK-050: Implement distinct unified trajectory logic for Stage 3 [Size: M]

**Finding**: S-4
**Priority**: Should-fix
**Files**:
- `/Users/rajesh/athena/training/stage2_agemem/trainer.py` (modify)

**Description**:
`_collect_unified_trajectory()` (lines 308-314) is a direct passthrough to
`_collect_multi_tool_trajectory()`. Stage 3 (UNIFIED) should implement distinct behavior
from Stage 2 (MULTI_TOOL) as described in the trainer's docstring: "Full autonomous memory
decisions" and "End-to-end optimization."

Changes required in `/Users/rajesh/athena/training/stage2_agemem/trainer.py`:

1. **Replace the passthrough** with distinct Stage 3 logic:
   - Longer operation sequences (4-8 steps, vs 2-4 for Stage 2).
   - Autonomous operation selection: instead of random choice, use the policy model to
     select the next operation based on current state (call `self.model` or
     `self.grpo._get_action_logprob` to get the model's preferred action, then sample).
   - End-to-end quality evaluation: after the full sequence, call `self.agemem.retrieve()`
     with the original query and measure retrieval quality as a trajectory-level bonus reward.
   - Include a "plan" state at the start of the trajectory that describes the overall
     memory management goal, giving the model context for multi-step planning.

2. **Add a trajectory-level reward bonus** based on end-to-end outcome:
   - After executing the full operation sequence, run a retrieval query and score the
     result quality (e.g., number of relevant results, latency).
   - Distribute this bonus across all steps in the trajectory.

**Acceptance Criteria**:
- `_collect_unified_trajectory()` no longer delegates to `_collect_multi_tool_trajectory()`.
- Stage 3 trajectories are longer (4-8 steps) and include model-driven operation selection.
- A trajectory-level quality bonus is computed and added to per-step rewards.
- Existing tests pass.

**Dependencies**: TASK-049 (builds on the real AgeMem operation wiring)

---

## Phase 3: Informational / Cleanup

### TASK-051: Convert f-string logger calls to lazy %s formatting [Size: S]

**Finding**: S-5 + I-4 (combined)
**Priority**: Should-fix / Informational
**Files**:
- `/Users/rajesh/athena/training/stage1_finetune/finetune.py` (modify)
- `/Users/rajesh/athena/training/stage2_agemem/trainer.py` (modify)

**Description**:
Python logging best practice is to use `%s`-style lazy formatting to avoid string
interpolation when the log level is disabled. The following calls use f-strings:

In `finetune.py`:
- Line 75: `self.logger.info(f"Loading model: {self.config.model_name}")`
- Line 124: `self.logger.error(f"Setup failed: {e}")`
- Line 264: `self.logger.info(f"Model saved to {save_path}")`

In `trainer.py`:
- Line 108: `self.logger.info(f"AgeMem trainer ready, starting at {self.current_stage.name}")`
- Lines 148-149: `self.logger.info(f"Step {self._step_count}, Stage {self.current_stage.name}, " f"Loss: {metrics['total_loss']:.4f}")`
- Line 201: `self.logger.info(f"Transitioning from {self.current_stage.name} to {new_stage.name}")`
- Line 326: `self.logger.info(f"Checkpoint saved: {path}")`
- Line 331: `self.logger.info(f"Checkpoint loaded: {path}")`

Changes: Convert each to `self.logger.info("Loading model: %s", self.config.model_name)` style.
For the loss formatting case, use `"Step %d, Stage %s, Loss: %.4f"` with positional args.

**Acceptance Criteria**:
- Zero f-string logger calls remain in `finetune.py` and `trainer.py`.
- Existing tests pass.

**Dependencies**: None

---

### TASK-052: Parse olmoe key in AthenaConfig._from_dict() [Size: S]

**Finding**: I-2
**Priority**: Informational
**Files**:
- `/Users/rajesh/athena/core/config.py` (modify)

**Description**:
`AthenaConfig._from_dict()` (lines 150-178) parses `model`, `memory`, `communication`,
`evolution`, `learning`, `trading`, and `agents` keys, but skips `olmoe`. The `olmoe` field
exists on `AthenaConfig` (line 125) as `OLMoEIntegrationConfig` but is always left at its
default when loading from YAML/JSON.

Changes required in `/Users/rajesh/athena/core/config.py`:

1. **Add olmoe parsing** in `_from_dict()`, after the `trading` block (line 166):
   ```
   if "olmoe" in data:
       config.olmoe = OLMoEIntegrationConfig(**data["olmoe"])
   ```

**Acceptance Criteria**:
- A YAML/JSON config with an `olmoe:` section populates `config.olmoe` correctly.
- Missing `olmoe` key still produces the default `OLMoEIntegrationConfig`.
- Existing tests pass.

**Dependencies**: None

---

### TASK-053: Fix missing await on async get_stats() in test_memory.py [Size: S]

**Finding**: I-3
**Priority**: Informational
**Files**:
- `/Users/rajesh/athena/tests/test_memory.py` (modify)

**Description**:
`AgeMem.get_stats()` is an `async def` method (see `memory/agemem.py:441`). It is called
without `await` in two places in `test_memory.py`:

1. Line 99: `stats = mem.get_stats() if hasattr(mem, "get_stats") else {}`
   - This test (`test_add_multiple_entries`) is already `async def` with `@pytest.mark.asyncio`.
   - Fix: `stats = await mem.get_stats() if hasattr(mem, "get_stats") else {}`

2. Line 108: `stats = mem.get_stats()`
   - This test (`test_get_stats`) is a sync `def` method.
   - Fix: Either make it async with `@pytest.mark.asyncio` and `await`, or use
     `asyncio.run(mem.get_stats())`.

Note: These tests currently pass because they're wrapped in try/except that skips on any
exception, so the coroutine-not-awaited issue is silently swallowed. The `isinstance(stats, dict)`
check happens to pass because an unawaited coroutine is truthy (but is NOT a dict -- the
assertion would fail if the try/except were removed).

**Acceptance Criteria**:
- Both `get_stats()` calls are properly awaited.
- `test_get_stats` is async or uses `asyncio.run()`.
- Tests pass (these specific tests may still skip if AgeMem can't initialize without Neo4j,
  but they should not produce coroutine-not-awaited warnings).

**Dependencies**: None

---

### TASK-054: Replace hash() with _stable_hash() in scrape_macro_indicators [Size: S]

**Finding**: Residual from S-2
**Priority**: Informational
**Files**:
- `/Users/rajesh/athena/training/data/scrapers/market.py` (modify)

**Description**:
`scrape_macro_indicators()` at line 202 uses `hash(indicator)` for the fallback value of
unknown macro indicators:
```python
value = mock_values.get(indicator, round(1.0 + hash(indicator) % 10, 2))
```

`hash()` is non-deterministic across Python processes (depends on `PYTHONHASHSEED`). The file
already defines `_stable_hash()` at line 20 using SHA-256.

Change line 202:
```python
value = mock_values.get(indicator, round(1.0 + _stable_hash(indicator) % 10, 2))
```

**Acceptance Criteria**:
- `hash(indicator)` is replaced with `_stable_hash(indicator)` on line 202.
- Output is deterministic across Python processes.
- Existing tests pass.

**Dependencies**: None

---

## Phase 4: Verification

### TASK-055: Full test suite verification [Size: S]

**Priority**: Gate
**Files**: None (test-only)

**Description**:
Run the full test suite to confirm all Sprint 7 changes are green:
```
python3 -m pytest tests/ -q
```
Expected: 171 passed, 6 skipped (unchanged baseline).

Also run a quick smoke test of the main entry point:
```
python3 main.py --mode dry-run
```

**Acceptance Criteria**:
- `pytest tests/ -q` shows 171 passed, 6 skipped, 0 failed.
- `python3 main.py --mode dry-run` exits cleanly.

**Dependencies**: TASK-046 through TASK-054 (all must be complete)

---

## Summary Table

| Task     | Finding      | Title                                        | Priority      | Size | Dependencies    |
|----------|-------------|----------------------------------------------|---------------|------|-----------------|
| TASK-046 | B-1         | Wire OLMoE encode() and action_head          | Blocker       | M    | None            |
| TASK-047 | B-2         | Wire MarketDataFeed into main.py loops       | Blocker       | M    | None            |
| TASK-048 | S-1         | Secure torch.load in GRPO                    | Should-fix    | S    | None            |
| TASK-049 | S-3         | Wire real AgeMem ops into trajectories       | Should-fix    | L    | TASK-046        |
| TASK-050 | S-4         | Implement distinct unified trajectory logic  | Should-fix    | M    | TASK-049        |
| TASK-051 | S-5 + I-4   | Convert f-string logger calls to %s style    | Should-fix    | S    | None            |
| TASK-052 | I-2         | Parse olmoe key in AthenaConfig._from_dict   | Informational | S    | None            |
| TASK-053 | I-3         | Fix missing await on get_stats in tests      | Informational | S    | None            |
| TASK-054 | Residual S-2| Replace hash() with _stable_hash() in scraper| Informational | S    | None            |
| TASK-055 | --          | Full test suite verification                 | Gate          | S    | TASK-046..054   |

## Recommended Execution Order

**Wave 1** (independent, can run in parallel):
- TASK-046 (B-1), TASK-047 (B-2), TASK-048 (S-1), TASK-051, TASK-052, TASK-053, TASK-054

**Wave 2** (depends on TASK-046):
- TASK-049 (S-3)

**Wave 3** (depends on TASK-049):
- TASK-050 (S-4)

**Wave 4** (depends on all):
- TASK-055 (verification gate)
