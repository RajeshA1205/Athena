# Sprint 8: Rev 4 Review Fixes

**Goal:** Address 6 findings from the Rev 4 post-Sprint-7 senior-dev code review. Two are Critical (blocking production use of GRPO training), one Major, two Minor, and one Nit.

**Plan directory:** `/Users/rajesh/athena/plans/sprint8-review-fixes/`
**Design docs:** `/Users/rajesh/athena/plans/sprint8-review-fixes/designs/`
**Test baseline:** 173 passed, 4 skipped

---

## Task List

| Task | Severity | File(s) | Description | Depends On |
|------|----------|---------|-------------|------------|
| TASK-056 | Critical | `training/stage2_agemem/grpo.py` | Fix `_ACTION_TO_IDX` key case: `op.name` → `op.value` | — |
| TASK-057 | Critical | `training/stage2_agemem/trainer.py` | Replace `hash(content)` with stable SHA-256 hash in delete path | — |
| TASK-058 | Major | `training/stage2_agemem/trainer.py` | Compute real action log-probs instead of `-1.0` placeholder at collection time | TASK-056 |
| TASK-059 | Minor | `training/stage2_agemem/trainer.py` | Log debug on model-driven op selection failure instead of silently passing | — |
| TASK-060 | Minor | `main.py`, `trading/market_data.py` | Move `MarketDataFeed` instantiation out of dry-run path; remove private `_MOCK_SYMBOLS` access | — |
| TASK-061 | Nit | `models/olmoe.py`, `training/stage2_agemem/grpo.py` | Convert remaining f-string logger calls to `%s`-style | — |

## Execution Order

**Wave 1 (parallel):** TASK-056, TASK-057, TASK-059, TASK-060, TASK-061
**Wave 2 (after TASK-056):** TASK-058
**Gate:** Re-run `pytest tests/ -q` and `python3 main.py --mode dry-run` — expect 173 passed, 4 skipped
