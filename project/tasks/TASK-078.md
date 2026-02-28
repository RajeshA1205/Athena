# TASK-078: End-to-end smoke test gate for mlx-lm migration

## Status
- **State:** Queued
- **Priority:** 🟡 High
- **Depends on:** TASK-072, TASK-073, TASK-074, TASK-075, TASK-076, TASK-077
- **Created:** 2026-02-26

## Objective
Run the full test suite and the production entry point to confirm that the mlx-lm migration is complete and nothing is broken. This is a verification-only gate task — no code changes are expected unless a regression is discovered.

## Context
All Sprint 10 implementation tasks (TASK-072 through TASK-077) must be accepted before this gate runs. The test baseline before Sprint 10 is 173 passed, 4 skipped (the 4 skips are torch-related tests that skip when torch is not installed).

Production entry point: `/Users/rajesh/athena/main.py --mode dry-run`
Test suite: `python3 -m pytest tests/ -q` from `/Users/rajesh/athena/`

If mlx-lm loads the OLMoE model successfully during `--mode dry-run`, the agent LLM reasoning path will exercise `generate()`. If the model weights are not yet downloaded, it is acceptable for `OLMoEModel.load()` to log a warning and return False (agents fall back to heuristic reasoning) — the dry-run must still complete without an uncaught exception.

See `/Users/rajesh/athena/main.py` and `/Users/rajesh/athena/project/context.md`.

## Scope & Constraints
- May modify: nothing, unless a regression requires a targeted fix — in that case document the fix clearly in the Agent Log
- Must NOT make speculative code changes "while you're in there"
- If a test regression is found, fix only the specific broken behaviour and log it

## Input
- All files modified in TASK-072 through TASK-077
- `/Users/rajesh/athena/main.py`
- `/Users/rajesh/athena/tests/` (full test suite)

## Expected Output
1. Test run output showing `173 passed, 4 skipped` (or better — no regressions)
2. `main.py --mode dry-run` output showing clean startup and shutdown with no uncaught exceptions
3. A one-paragraph summary of the mlx-lm migration status:
   - Which backend was active during dry-run (mlx or transformers fallback)
   - Whether the model loaded successfully or fell back gracefully
   - Any warnings logged during startup

## Acceptance Criteria
- [ ] `python3 -m pytest tests/ -q` exits 0 with 173 passed, 4 skipped (no regressions from Sprint 10 changes)
- [ ] `python3 main.py --mode dry-run` exits 0 (clean shutdown) with no uncaught exceptions in stderr
- [ ] Log output from dry-run shows either "Loading OLMoE via mlx-lm" or a graceful fallback/warning — not a Python traceback
- [ ] `import mlx_lm` succeeds in the project Python environment (confirms TASK-072 is in effect)
- [ ] `OLMoEConfig().use_mlx` is True (confirms TASK-073 is in effect)

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
