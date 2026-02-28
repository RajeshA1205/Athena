# ATHENA Project Roadmap

## Project Status: ~91% Complete (Sprints 1-8 done; Sprints 9-10 in progress)

---

## Sprint 1: Foundation & Core Abstractions — ✅ COMPLETED
**Completion: 100%**

- ✅ Project scaffolding with full directory structure
- ✅ OLMoE 1B model integration and inference pipeline
- ✅ BaseAgent abstract class with think() and act() methods
- ✅ Configuration system (core/config.py)
- ✅ Shared utilities (core/utils.py)
- ✅ Embedding models setup (models/embeddings.py)
- ✅ AgeMem memory layer foundation (memory/agemem.py)
- ✅ Graphiti backend integration (memory/graphiti_backend.py)
- ✅ Memory operations implementation (memory/operations.py)
- ✅ Stage 1 fine-tuning infrastructure (training/stage1_finetune/)
- ✅ Stage 2 AgeMem GRPO training infrastructure (training/stage2_agemem/)

---

## Sprint 2: Parallel Layer Implementation — ✅ COMPLETED
**Completion: 100%**

### Layer A: Agent Layer (/agents/)
- [x] market_analyst.py — Market data analysis agent
- [x] risk_manager.py — Portfolio risk assessment agent
- [x] strategy_agent.py — Strategy formulation agent
- [x] execution_agent.py — Order execution agent
- [x] coordinator.py — Orchestration and conflict resolution

### Layer B: Communication Layer (/communication/)
- [x] latent_space.py — Shared embedding space for LatentMAS
- [x] encoder.py — Encode agent outputs to latent representations
- [x] decoder.py — Decode latent messages for receiving agents
- [x] router.py — Message routing with attention-based priority

### Layer C: Evolution Layer (/evolution/)
- [x] workflow_discovery.py — Extract workflow patterns from agent history
- [x] agent_generator.py — Auto-generate agent configurations
- [x] cooperative_evolution.py — Multi-agent experience replay

---

## Sprint 3: Layer Integration — ✅ COMPLETED
**Completion: 100%**

- [x] Agent ↔ Memory integration (connect agents to AgeMem)
- [x] Agent ↔ Communication integration (LatentMAS messaging)
- [x] Coordinator orchestration workflow
- [x] End-to-end pipeline test with simple scenario

---

## Sprint 4: Advanced Features & Learning Layer — ✅ COMPLETED
**Completion: 100%**

### Learning Layer (/learning/)
- [x] nested_learning.py — Inner/outer loop meta-learning
- [x] repexp.py — Representation-space exploration with diversity bonuses

### Advanced Features
- [x] AgeMem RL training with real agent trajectories
- [x] LatentMAS advanced routing (attention-based, priority channels)
- [x] AgentEvolver workflow discovery from real histories
- [x] RepExp exploration bonuses in agent decision-making

---

## Sprint 5: Trading Domain & Testing — ✅ COMPLETED
**Completion: 100%**

### Trading Infrastructure
- [x] trading/market_data.py — Real-time and historical data feeds
- [x] trading/order_management.py — Order execution interface
- [x] trading/portfolio.py — Position tracking and P&L calculation

### Data Pipeline
- [x] training/data/scrapers/news.py — News and SEC filings scraper
- [x] training/data/scrapers/market.py — Market data scraper
- [x] training/data/scrapers/social.py — Social sentiment scraper
- [x] training/data/processors/cleaner.py — Data cleaning
- [x] training/data/processors/formatter.py — Format for training
- [x] training/data/datasets.py — PyTorch datasets

### Testing & Validation
- [x] Comprehensive test suite in tests/ (171 tests, 6 skipped)
- [x] Backtesting framework
- [x] Paper trading validation

---

## Sprint 6: Code Quality, Security, and Correctness Fixes — ✅ COMPLETE
**Completed: 2026-02-24** | 22/22 tasks | Tests: 171 passed, 6 skipped
**Source:** 28 findings from two external code reviews

### Phase 1: Critical / Blocking (2 tasks)
- [ ] TASK-024: Fix GRPO reference model aliasing and action log-prob placeholder
- [ ] TASK-025: Add weights_only=True to torch.load (unsafe deserialization)

### Phase 2: High Priority (5 tasks)
- [ ] TASK-026: Unify trading enums into canonical module (trading/enums.py)
- [ ] TASK-027: Replace hash() with _stable_hash in order_management
- [ ] TASK-028: Align AgentStateEncoder default latent_dim to 512
- [ ] TASK-029: Wire OLMoE model into the agent framework
- [ ] TASK-030: Create production entry point (main.py) [depends on 026, 036]

### Phase 3: Medium Priority (9 tasks)
- [ ] TASK-031: Optimize MACD to O(n) incremental EMA
- [ ] TASK-032: Fix backtest total_return to use geometric model
- [ ] TASK-033: Seed random in execution_agent._simulate_fill
- [ ] TASK-034: Bound adaptation_history and action_history
- [ ] TASK-035: Replace blocking I/O in async methods with asyncio.to_thread
- [ ] TASK-036: Fix duplicate logging handlers in setup_logging
- [ ] TASK-037: Deduplicate coordinator memory writes
- [ ] TASK-038: Fix _allocate_resources docstring mismatch
- [ ] TASK-039: Replace weak ID generation with secrets/uuid

### Phase 4: Low Priority / Code Quality (6 tasks)
- [ ] TASK-040: Use timezone-aware timestamps throughout
- [ ] TASK-041: Guard numpy import in utils.py
- [ ] TASK-042: Add recursion depth limit to deep_merge
- [ ] TASK-043: Fix RiskManager.think() hardcoded done flag
- [ ] TASK-044: Document hardcoded quality rewards in agemem
- [ ] TASK-045: Convert f-string logging to lazy %s-style in agent files

**Sprint 6 Constraint:** Must not break any of the 171 existing tests. No new third-party
dependencies. Public API signatures must remain backwards-compatible where possible.

---

## Sprint 7: Rev 3 Review Fixes — ✅ COMPLETE
**Completed: 2026-02-25** | 10/10 tasks | Tests: 173 passed, 4 skipped
**Source:** 11 confirmed findings from Rev 3 post-Sprint-6 review (2 false positives already fixed in Sprint 6)
**Plan:** /Users/rajesh/athena/plans/sprint7-rev3-fixes/ | Designs: plans/sprint7-rev3-fixes/designs/

### Phase 1: Blockers
- [x] TASK-046: Wire OLMoE `encode()` and `action_head` for GRPO
- [x] TASK-047: Wire `MarketDataFeed` into `main.py` paper-trade/backtest loops
- [x] TASK-048: Secure `torch.load` fallback in GRPO checkpoint loading
- [x] TASK-051: Convert f-string logger calls to `%s`-style (finetune.py + trainer.py)
- [x] TASK-052: Parse `olmoe` key in `AthenaConfig._from_dict()`
- [x] TASK-053: Fix missing `await` on `get_stats()` in test_memory.py
- [x] TASK-054: Replace `hash()` with `_stable_hash()` in `scrape_macro_indicators`

### Phase 2: Depends on TASK-046
- [x] TASK-049: Wire real AgeMem operations into trajectory collection

### Phase 3: Depends on TASK-049
- [x] TASK-050: Implement distinct unified trajectory logic for Stage 3

### Gate
- [x] TASK-055: Full test suite verification gate (173 passed, 4 skipped)

---

## Sprint 8: Rev 4 Review Fixes — ✅ COMPLETE
**Source:** 6 findings from Rev 4 post-Sprint-7 senior-dev code review
**Plan:** /Users/rajesh/athena/plans/sprint8-review-fixes/ | Designs: plans/sprint8-review-fixes/designs/

### Phase 1: Critical (parallelizable)
- [x] TASK-056: Fix `_ACTION_TO_IDX` key case mismatch in GRPO
- [x] TASK-057: Replace `hash()` with stable hash in trainer delete path

### Phase 2: Major / Minor (TASK-058 depends on TASK-056)
- [x] TASK-058: Compute real action log-probs at trajectory collection time
- [x] TASK-059: Log debug message on model-driven selection failure
- [x] TASK-060: Fix MarketDataFeed placement and private attribute access in main.py

### Phase 3: Nit (independent)
- [x] TASK-061: Convert remaining f-string logger calls in olmoe.py and grpo.py

---

## Sprint 9: Rev 5 Review Fixes — IN PROGRESS
**Source:** Critical and Major findings from Rev 5 full-codebase senior-dev review (Sprints 2–8 scope)
**Plan:** /Users/rajesh/athena/plans/sprint9-review-fixes/
**Target test baseline:** 173 passed, 4 skipped (no regressions)

Goal: Fix correctness bugs in the LLM reasoning path, market data wiring, order type unification, GRPO logprob sentinel, and portfolio exposure calculation.

### Phase 1: Critical (all parallelizable)
- [ ] TASK-062: Fix timezone-naive datetime.now() in ExecutionAgent
- [ ] TASK-063: Fix LatentSpace broadcast missing unregistered agents
- [ ] TASK-064: Fix GRPO sentinel logprob corrupting gradients

### Phase 2: Major (TASK-065 depends on TASK-062; others parallelizable)
- [ ] TASK-065: Remove duplicate Order dataclass from ExecutionAgent [depends on TASK-062]
- [ ] TASK-066: Fix _llm_reason async/sync mismatch
- [ ] TASK-067: Fix portfolio check_limits double-counting exposure on sells
- [ ] TASK-068: Fix main.py market data format mismatch with agent interface
- [ ] TASK-069: Fix Sharpe ratio mean_return calculation in StrategyAgent

---

## Sprint 10: mlx-lm Migration (Apple Silicon Inference) — QUEUED
**Source:** Platform constraint — bitsandbytes/CUDA unavailable on Apple Silicon Mac; mlx-lm is the correct inference stack
**Target test baseline:** 173 passed, 4 skipped (no regressions)
**Decision:** mlx-lm primary; transformers kept as fallback; MemoryActionHead/GRPO training on transformers path only

Goal: Make OLMoE inference actually runnable on the development machine (Apple Silicon Mac) by replacing the CUDA-only HuggingFace path with mlx-lm as the primary backend.

### Phase 1: Install + Config (parallelizable after TASK-072)
- [ ] TASK-072: Install mlx-lm and mlx packages for Apple Silicon inference
- [ ] TASK-073: Add mlx-lm backend fields to OLMoEConfig and OLMoEIntegrationConfig [depends on TASK-072]

### Phase 2: Core migration (sequential, each builds on prior)
- [ ] TASK-074: Rewrite OLMoEModel.load() to use mlx-lm as primary backend [depends on TASK-073]
- [ ] TASK-075: Rewrite OLMoEModel.generate() to use mlx-lm with async wrapper [depends on TASK-074]
- [ ] TASK-076: Fix OLMoEModel.encode() to bridge mlx arrays to torch tensors [depends on TASK-074]

### Phase 3: Hardening + Verification
- [ ] TASK-077: Verify MemoryActionHead and GRPO pipeline with new encode() output [depends on TASK-076]
- [ ] TASK-078: End-to-end smoke test gate for mlx-lm migration [depends on TASK-072 through TASK-077]

---

## Milestones

| Milestone | Target | Status |
|-----------|--------|--------|
| Sprint 1: Foundation Complete | ✅ | DONE |
| Sprint 2: All 4 Layers Implemented | ✅ | DONE |
| Sprint 3: Layers Integrated | ✅ | DONE |
| Sprint 4: Advanced Features Active | ✅ | DONE |
| Sprint 5: Trading System Operational | ✅ | DONE |
| Sprint 6: Production-Ready Baseline | ✅ 2026-02-24 | DONE (22/22 tasks) |
| Sprint 7: Rev 3 Review Fixes | ✅ 2026-02-25 | DONE (10/10 tasks) |
| Sprint 8: Rev 4 Review Fixes | ✅ 2026-02-25 | DONE (6/6 tasks) |
| Sprint 9: Rev 5 Review Fixes | 2026-02-26 | IN PROGRESS (0/8 tasks) |
| Sprint 10: mlx-lm Migration | 2026-02-26 | QUEUED (0/7 tasks) |

---

## Key Dependencies

- **Sprint 2** depends on Sprint 1 completion ✅
- **Sprint 3** depends on Sprint 2 completion ✅
- **Sprint 4** depends on Sprint 3 completion ✅
- **Sprint 5** developed in parallel with Sprint 4 ✅
- **Sprint 6** depends on Sprints 2-5 completion ✅ — COMPLETE
- **Sprint 7** depends on Sprint 6 completion ✅ — COMPLETE
- **Sprint 8** depends on Sprint 7 completion ✅ — COMPLETE
- **Sprint 9** depends on Sprint 8 completion ✅ — IN PROGRESS
- **Sprint 10** can run in parallel with Sprint 9 (different files; no overlap)

---

## Research Papers Reference

Located at `/Users/rajesh/athena/architecture/base/`:
- AgeMem.pdf — Memory layer with RL-trained operations
- AgentEvolver — Workflow discovery and agent generation
- LatentMAS — Latent space inter-agent communication
- RepExp — Representation-space exploration bonuses
