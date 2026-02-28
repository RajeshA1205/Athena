# TASK-058: Compute real action log-probs at trajectory collection time

## Status
- **State:** Queued
- **Priority:** 🟡 Major
- **Depends on:** TASK-056
- **Created:** 2026-02-25

## Objective
Replace the hardcoded `-1.0` log-probability placeholders in the three trajectory collector methods of `training/stage2_agemem/trainer.py` with real log-probs computed from the policy model at collection time.

## Context
All three collectors — `_collect_single_tool_trajectory`, `_collect_multi_tool_trajectory`, and `_collect_unified_trajectory` — currently append a constant `-1.0` as the log-prob for every action step. GRPO's importance-sampling ratio is `exp(new_lp - old_lp)`. When `old_lp` is always `-1.0`, the denominator is always `exp(-1.0)` regardless of the actual action, making the ratio meaningless and the policy gradient signal corrupted.

TASK-056 must be accepted first so that `_ACTION_TO_IDX` lookups inside `_compute_action_logprob` return the correct index for the action.

The model may not be loaded (e.g. during unit tests or CPU-only runs). In that case fall back to `-1.0` to preserve existing behaviour.

## Scope & Constraints
- **May modify:** `training/stage2_agemem/trainer.py`
- **Must NOT modify:** `training/stage2_agemem/grpo.py` or any other file
- Wrap the model call in `torch.no_grad()` to avoid accumulating gradients
- Use `.item()` to convert tensor to Python float before appending

## Input
- `training/stage2_agemem/trainer.py` — all three `_collect_*_trajectory` methods
- `training/stage2_agemem/grpo.py` — `_compute_action_logprob(model, state, action)` signature

## Expected Output
Each `logprobs.append(-1.0)` replaced with logic similar to:
```python
try:
    with torch.no_grad():
        lp = self.grpo._compute_action_logprob(self.model, state, action)
    logprobs.append(lp.item() if hasattr(lp, "item") else float(lp))
except Exception:
    logprobs.append(-1.0)
```

## Acceptance Criteria
- [ ] All three collector methods compute real log-probs when `self.model` is available
- [ ] Graceful fallback to `-1.0` when model is unavailable or raises
- [ ] `torch.no_grad()` context used around the model call
- [ ] `python3 -m pytest tests/ -q` stays green (173 passed, 4 skipped)

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
