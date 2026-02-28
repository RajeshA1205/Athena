# Sprint 9: Rev 5 Review Fixes

**Status:** IN PROGRESS
**Started:** 2026-02-26
**Source:** Full-codebase Rev 5 senior-dev review covering Sprints 2–8 scope
**Test baseline:** 173 passed, 4 skipped — must not regress

---

## Sprint Goal

Address all Critical and Major findings from the Rev 5 review. Fix correctness bugs in five
areas: the LLM reasoning path, market data wiring, order type unification, GRPO logprob
sentinel, and portfolio exposure calculation.

---

## Task List

### Critical (Phase 1 — all parallelizable)

| Task | File(s) | Issue | Status |
|------|---------|-------|--------|
| TASK-062 | `agents/execution_agent.py` | `datetime.now()` timezone-naive; breaks timestamp ordering and P&L | Queued |
| TASK-063 | `communication/latent_space.py`, `agents/coordinator.py` | `broadcast()` silently misses agents with no prior direct message (defaultdict gap) | Queued |
| TASK-064 | `training/stage2_agemem/trainer.py`, `grpo.py` | Fallback logprob `-1.0` makes PPO ratio `exp(1.0)=2.718`, outside clip range; corrupts gradients | Queued |

### Major (Phase 2 — TASK-065 depends on TASK-062; rest parallelizable)

| Task | File(s) | Issue | Status |
|------|---------|-------|--------|
| TASK-065 | `agents/execution_agent.py` | Local `Order` dataclass shadows `trading.order_management.Order`; `filled_qty` vs `filled_quantity` mismatch | Queued |
| TASK-066 | `core/base_agent.py` | `asyncio.to_thread(self.llm.generate, ...)` wraps an `async def` — returns coroutine object, never a string; OLMoE reasoning path silently broken | Queued |
| TASK-067 | `trading/portfolio.py` | `check_limits` adds `abs(qty*price)` unconditionally; sell orders inflate instead of reduce exposure | Queued |
| TASK-068 | `main.py`, `agents/market_analyst.py` | `context.metadata["market_data"]` format mismatch: `main.py` writes dict-of-dicts; `think()` reads `prices` key → always empty | Queued |
| TASK-069 | `agents/strategy_agent.py` | Sharpe `mean_return = total_return / len(returns)` uses geometric total, not arithmetic mean; Sharpe values incorrect | Queued |

---

## Dependency Graph

```
TASK-062 (UTC timestamps)
    └── TASK-065 (remove duplicate Order)

TASK-063 — independent
TASK-064 — independent
TASK-066 — independent
TASK-067 — independent
TASK-068 — independent
TASK-069 — independent
```

Recommended execution order:
1. Wave 1 (parallel): TASK-062, TASK-063, TASK-064, TASK-066, TASK-067, TASK-068, TASK-069
2. Wave 2 (after TASK-062 accepted): TASK-065

---

## Verification

After each task (and after all tasks complete):

```bash
python3 -m pytest tests/ -q
# Expected: 173 passed, 4 skipped

python3 main.py --mode dry-run
# Expected: clean shutdown, no errors
```

---

## Minor Findings (informational — no tasks created)

These were identified in the Rev 5 review but are low-severity and do not require Sprint 9 tasks:

- `config.py` `_from_dict` extra-key fragility (silently ignores unknown keys)
- `AgentMessage.timestamp` defaults to `None` instead of UTC now
- `portfolio.py` `update_from_fill` has no lock (thread safety risk in concurrent use)
- `trainer.py` type annotation mismatch (`List[str] = None` should be `Optional[List[str]] = None`)
- `coordinator.py` `max()` key style (minor readability)
- `base_agent.py` `TYPE_CHECKING` import path inconsistency
- `test_agents.py` behavioral coverage gaps (happy-path only)

These can be addressed in a future Sprint 10 polish pass if needed.

---

## Task Files

- `/Users/rajesh/athena/project/tasks/TASK-062.md`
- `/Users/rajesh/athena/project/tasks/TASK-063.md`
- `/Users/rajesh/athena/project/tasks/TASK-064.md`
- `/Users/rajesh/athena/project/tasks/TASK-065.md`
- `/Users/rajesh/athena/project/tasks/TASK-066.md`
- `/Users/rajesh/athena/project/tasks/TASK-067.md`
- `/Users/rajesh/athena/project/tasks/TASK-068.md`
- `/Users/rajesh/athena/project/tasks/TASK-069.md`
