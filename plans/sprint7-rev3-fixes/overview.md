# Sprint 7: Rev 3 Post-Sprint-6 Code Review Fixes

## Goal
Fix the 11 confirmed real issues identified in the Rev 3 code review of the ATHENA codebase,
organized into 10 tasks (TASK-046 through TASK-055). Two issues are blockers that render GRPO
training and the paper-trade/backtest main loop non-functional. The remaining are should-fix
and informational items covering test correctness, config parsing, logging hygiene, and
non-deterministic hashing.

## Background
After Sprint 6 completed all 22 bugfix tasks (TASK-024 through TASK-045), a Rev 3 code review
found 13 issues. Two were false positives (S-2 fixed, I-1 fixed), leaving 11 real findings.
The two blockers (B-1 and B-2) mean that GRPO training silently produces zero gradients and
the paper-trade/backtest loops run with empty context -- both rendering those code paths
effectively inert.

## Requirements

### Functional
- FR-1: `OLMoEModel` must expose a synchronous `encode(text) -> torch.Tensor` method returning
  the last-hidden-state embedding, and must attach `MemoryActionHead` as `self.action_head`
  in `__init__`, so that `grpo.py` `_compute_action_logprob` receives real gradients (B-1).
- FR-2: The paper-trade and backtest loops in `main.py` must instantiate `MarketDataFeed` and
  populate `AgentContext.metadata` with market data each iteration (B-2).
- FR-3: `StepwiseGRPO.load()` must not fall back to `torch.load()` without `weights_only=True`
  on any PyTorch version; enforce minimum PyTorch 2.0 or use `safetensors` (S-1).
- FR-4: All three `_collect_*_trajectory()` methods in `trainer.py` must execute real AgeMem
  operations instead of simulating with `random.random()` (S-3).
- FR-5: `_collect_unified_trajectory()` must implement distinct Stage 3 logic (longer sequences,
  autonomous operation selection, end-to-end quality evaluation) rather than delegating to
  `_collect_multi_tool_trajectory()` (S-4).
- FR-6: All f-string logger calls in `finetune.py` (3 calls) and `trainer.py` (5 calls) must
  use `%s`-style lazy formatting (S-5 + I-4, combined).
- FR-7: `AthenaConfig._from_dict()` must parse the `olmoe` key from input dicts to populate
  `OLMoEIntegrationConfig` (I-2).
- FR-8: `test_memory.py` must correctly `await` the async `AgeMem.get_stats()` calls at
  lines 99 and 108 (I-3).
- FR-9: `scrape_macro_indicators` in `training/data/scrapers/market.py` must replace
  `hash(indicator)` with `_stable_hash(indicator)` for deterministic output across processes
  (residual from S-2).

### Non-Functional
- NFR-1: All 171 existing tests must remain green after each task.
- NFR-2: No new external dependencies introduced.
- NFR-3: Changes must follow existing patterns (try/except guards for torch, `_stable_hash`
  for deterministic hashing, `%s`-style logging).

### Constraints
- CON-1: `OLMoEModel.encode()` must be synchronous (not async) since GRPO's
  `_compute_action_logprob` is synchronous and called inside `torch.no_grad()` blocks.
- CON-2: The existing `OLMoEModel.embed()` (async, returns `List[float]`) must not be removed
  or broken -- `encode()` is a separate method.
- CON-3: `MemoryActionHead` hidden_dim must match OLMoE's actual hidden size (auto-detected
  from `model.config.hidden_size` after loading).

## Assumptions
- ASM-1: The OLMoE base model's HuggingFace config exposes `hidden_size` after loading.
- ASM-2: `MarketDataFeed` in MOCK mode is sufficient for paper-trade/backtest wiring (LIVE
  mode integration is out of scope).
- ASM-3: The AgeMem mock backend (GraphitiBackend) can handle the async operations called
  from trajectory collection without a live Neo4j instance, since tests already pass without one.
- ASM-4: PyTorch >= 2.0 is a reasonable minimum version requirement for the training pipeline.

## Out of Scope
- Live market data provider integration (LIVE mode for MarketDataFeed).
- Learned quality metrics for SUMMARY/FILTER rewards (existing placeholders are acceptable).
- OLMoE model downloading, training convergence, or GPU performance optimization.
- UI, API, or deployment concerns.
