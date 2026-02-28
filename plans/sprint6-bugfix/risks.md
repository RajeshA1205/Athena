# Risks and Trade-offs: Sprint 6 Bug-Fix and Improvements

## Risks

| ID   | Risk                                                        | Likelihood | Impact | Mitigation                                                        |
|------|-------------------------------------------------------------|------------|--------|-------------------------------------------------------------------|
| R-1  | ~~GRPO `_get_action_logprob` architecture unknown~~ **RESOLVED:** Architecture decided (OLMoE + MLP head, 6-dim action space). Implementation can proceed. | ~~High~~ | ~~High~~ | N/A -- resolved via OQ-1a/OQ-1b |
| R-2  | `weights_only=True` may reject non-tensor objects in saved checkpoint (e.g., `config` dict) | Medium | Medium | Test round-trip save/load; if it fails, separate config from weights in the save format |
| R-3  | Unifying enums may break downstream code that compares enum values by string rather than identity | Low | High | Run full test suite; grep for string comparisons like `== "pending"` across the codebase |
| R-4  | Changing `AgentStateEncoder` default from 256 to 512 may break existing serialized models or configs that relied on the old default | Low | Medium | This is pre-production code with no serialized models in the wild; risk is minimal |
| R-5  | ~~OLMoE integration (TASK-029) underspecified~~ **RESOLVED:** Scope defined -- CoordinatorAgent + StrategyAgent + MLP action head for GRPO. Shared model instance. | ~~High~~ | ~~Low~~ | N/A -- resolved via OQ-1b/OQ-5 |
| R-6  | `main.py` creation (TASK-030) requires design decisions about CLI interface and run modes that are not fully specified | Medium | Medium | Start with a minimal `--mode dry-run` path; extend later |
| R-7  | Converting `action_history` from `list` to `deque` may break code that uses list-specific operations (e.g., `action_history[0:5]`) | Low | Low | `deque` supports indexing; slicing requires `list(deque)[0:5]` -- check all usage sites |
| R-8  | f-string to %s-style logging conversion (TASK-045) is high line count and tedious, increasing risk of copy-paste errors | Medium | Low | Use a linter or automated tool; review carefully |

## Trade-offs

- **TD-1: ~~NotImplementedError vs. stub implementation for GRPO log-probs.~~**
  **Superseded by OQ-1 resolution.** The architecture is now fully specified
  (OLMoE + MLP head), so `_get_action_logprob` will have a real implementation
  rather than `NotImplementedError`. The `_update_reference_model` deep-copy fix
  remains unchanged.

- **TD-2: Single canonical enum file vs. keeping enums in their original
  modules.** Creating `trading/enums.py` adds a new file but eliminates the
  divergence problem permanently. Keeping enums in `order_management.py` and
  importing elsewhere was considered but creates an awkward dependency from the
  agents layer to a trading infrastructure module.

- **TD-3: `deque(maxlen=100)` for action_history vs. explicit pruning.** Deque
  with maxlen silently drops old entries, which is fine since only the last 10
  are ever read. Explicit pruning (e.g., `if len(history) > 100:
  history = history[-100:]`) is more visible but adds boilerplate.

- **TD-4: `asyncio.to_thread` for file I/O vs. `aiofiles`.** `aiofiles` is the
  purpose-built solution but adds a new dependency. `asyncio.to_thread` is
  stdlib-only and sufficient for occasional JSON saves.

## Open Questions

- **OQ-1:** ~~What is the intended policy model architecture for GRPO?~~
  **RESOLVED (2026-02-23).** MLP action head (6-dim, matching `MemoryOperation` enum)
  sits on top of OLMoE. OLMoE encodes state, MLP outputs action logits. Shared model
  instance between TASK-024 and TASK-029. ADD and UPDATE are separate GRPO actions;
  both map to Graphiti's ADD API at the memory layer.
  See `project/decisions/sprint6-open-questions.md` (OQ-1a, OQ-1b).

- **OQ-2:** ~~Should `PARTIAL` and `PARTIALLY_FILLED` be consolidated?~~
  **RESOLVED.** Consolidated into a single enum value during TASK-026 design.

- **OQ-3:** ~~What CLI interface and run modes should `main.py` support?~~
  **RESOLVED (2026-02-23).** ExecutionAgent role changed to advisory output
  (recommended investment plans, not trade orders). All Sprint 6 bug fixes still
  apply. `act()` returns structured recommendations `{symbol, action_type,
  suggested_quantity, rationale}`.
  See `project/decisions/sprint6-open-questions.md` (OQ-3).

- **OQ-4:** ~~Is there an existing deployment target?~~
  **RESOLVED.** No existing deployment target; TASK-030 proceeds with standalone CLI.

- **OQ-5:** ~~Should OLMoE be used for all agents or only specific ones?~~
  **RESOLVED (2026-02-23).** CoordinatorAgent and StrategyAgent get OLMoE integration.
  OLMoE also serves as GRPO policy encoder (shared instance).
  See `project/decisions/sprint6-open-questions.md` (OQ-1b).
