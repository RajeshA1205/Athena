# TASK-064: Fix GRPO Sentinel Logprob Corrupting Gradients

## Summary

`_get_logprob_for_step()` in `trainer.py` returns `-1.0` when the model is not loaded (the common case in testing and early training). In `grpo.py`, `_get_action_logprob()` returns `tensor(0.0)` as its fallback. The GRPO loss computes the PPO importance-sampling ratio as `exp(new_logprob - old_logprob)`. With `new=0.0` and `old=-1.0`, this gives `exp(1.0) = 2.718` — outside the PPO clip range of [0.8, 1.2] — causing the clipped surrogate to always hit the clip boundary and produce systematically biased gradients even with no real training signal. Fix: change the fallback in `_get_logprob_for_step` from `-1.0` to `0.0` so the ratio starts at `exp(0-0) = 1.0` (neutral).

## Current State

**File:** `training/stage2_agemem/trainer.py`, lines 378-389:

```python
    def _get_logprob_for_step(self, state: Dict[str, Any], action: str) -> float:
        """
        Compute the current policy log-probability for (state, action).
        Returns -1.0 if the model is not loaded or torch is unavailable.
        """
        try:
            import torch
            with torch.no_grad():
                lp = self.grpo._compute_action_logprob(self.model, state, action)
            return float(lp.item())
        except Exception:
            return -1.0
```

**File:** `training/stage2_agemem/grpo.py`, lines 355-370 (approximate, `_get_action_logprob` fallback):

The fallback when model lacks `encode` returns `torch.tensor(0.0)`:
```python
        if not (hasattr(model, "encode") and hasattr(model, "action_head") ...):
            return torch.tensor(0.0)
```

**GRPO loss computation (`compute_loss`, grpo.py ~line 240-260):**
```python
            new_logprob = self.grpo._get_action_logprob(self.policy_model, state, action)
            old_logprob = traj.action_logprobs[t]   # float from trajectory
            ratio = torch.exp(new_logprob - old_logprob)
            # ratio = exp(0.0 - (-1.0)) = exp(1.0) = 2.718 → always clipped
```

## Proposed Change

**File:** `training/stage2_agemem/trainer.py`

Change the fallback return value in `_get_logprob_for_step` from `-1.0` to `0.0`:

```python
    def _get_logprob_for_step(self, state: Dict[str, Any], action: str) -> float:
        """
        Compute the current policy log-probability for (state, action).
        Returns 0.0 if the model is not loaded or torch is unavailable,
        matching the _get_action_logprob fallback so the PPO ratio starts at 1.0.
        """
        try:
            import torch
            with torch.no_grad():
                lp = self.grpo._compute_action_logprob(self.model, state, action)
            return float(lp.item())
        except Exception:
            return 0.0
```

Also update the docstring on line 381 to say `Returns 0.0` instead of `Returns -1.0`.

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `training/stage2_agemem/trainer.py` | 381, 389 | Docstring + `return -1.0` → `return 0.0` |

## Acceptance Criteria

- `_get_logprob_for_step` returns `0.0` (not `-1.0`) when model is not loaded.
- PPO ratio when model is absent: `exp(0.0 - 0.0) = 1.0` (neutral, no gradient).
- The docstring says `Returns 0.0`.
- No other `-1.0` logprob fallbacks remain in `trainer.py`.
- 173 tests pass, 4 skipped.

## Edge Cases & Risks

1. **Neutral ratio means zero gradient**: `ratio=1.0` means `min(ratio * advantage, clip(...) * advantage) = 1.0 * advantage`. The GRPO loss still computes correctly — it simply produces no policy update from the sentinel trajectories, which is the correct behaviour when no real model signal is available.
2. **Existing trajectories**: Any checkpoint that saved trajectories with `-1.0` logprobs will produce `exp(0.0 - (-1.0)) = 2.718` on the first training step after loading. This is unavoidable for pre-existing checkpoints; after one step the trajectories are replaced and the issue resolves.
3. **One-line change**: This is the minimum safe fix. A more robust solution would filter trajectories with sentinel logprobs in `compute_loss`, but that requires more extensive changes and is out of scope.

## Test Notes

- Verify: `trainer._get_logprob_for_step({}, "add")` returns `0.0` when `trainer.model` is None.
- Verify: `trainer._get_logprob_for_step({}, "add")` returns a float close to 0 when model is loaded (log-prob of uniform-ish distribution over 6 actions ≈ -log(6) ≈ -1.79, but the exact value depends on model weights).
- Run `python3 -m pytest tests/ -q` — expect 173 passed, 4 skipped.
