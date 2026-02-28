# TASK-064: Fix GRPO sentinel logprob corrupting gradients

## Status
- **State:** Queued
- **Priority:** Critical
- **Depends on:** None
- **Created:** 2026-02-26

## Objective
Change the fallback sentinel value in `_get_logprob_for_step` from `-1.0` to `0.0` so that when the model is not loaded, the PPO importance-sampling ratio equals `exp(0 - 0) = 1.0` (neutral) rather than `exp(0 - (-1.0)) = 2.718` (outside clip range), preventing corrupt gradient updates.

## Context
`_get_logprob_for_step` in `training/stage2_agemem/trainer.py` returns `-1.0` when the policy model is not loaded. `_get_action_logprob` in `training/stage2_agemem/grpo.py` returns `tensor(0.0)` as the reference logprob sentinel. PPO computes the ratio `exp(logprob - ref_logprob)`. With the current sentinels: `exp(0.0 - (-1.0)) = exp(1.0) ≈ 2.718`, which is outside the PPO clip range of `[1-epsilon, 1+epsilon]` (typically `[0.8, 1.2]`). This means every training step that runs without a loaded model produces a maximally-clipped gradient update — silently corrupting training.

Fixed in Sprint 8 TASK-058 to return real log-probs when the model is loaded, but the fallback sentinel was not updated.

## Scope & Constraints
- **May modify:** `training/stage2_agemem/trainer.py`, `training/stage2_agemem/grpo.py`
- **Must NOT modify:** Any other file
- Preferred fix: change the `_get_logprob_for_step` fallback return from `-1.0` to `0.0`
- Acceptable alternative: add a guard in `train_step` to skip trajectories containing the sentinel value; if this approach is chosen, document the sentinel constant clearly

## Input
- `training/stage2_agemem/trainer.py` — `_get_logprob_for_step` method
- `training/stage2_agemem/grpo.py` — `_get_action_logprob` and `train_step` methods

## Expected Output
- `training/stage2_agemem/trainer.py` — `_get_logprob_for_step` returns `0.0` (not `-1.0`) when model is not loaded
- OR: `train_step` skips/filters trajectories where any step logprob equals the sentinel, with a debug-level log
- Sentinel values must be consistent between trainer.py and grpo.py (both `0.0`, or both guarded)

## Acceptance Criteria
- [ ] `_get_logprob_for_step` fallback is `0.0` (or trajectory is skipped on sentinel detection)
- [ ] The PPO ratio for a no-model fallback step evaluates to `1.0` (neutral)
- [ ] No occurrence of `-1.0` as a logprob sentinel remains without a corresponding skip guard
- [ ] `python3 -m pytest tests/ -q` passes with 173 passed, 4 skipped (no regressions)
- [ ] `python3 main.py --mode dry-run` completes without error

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
