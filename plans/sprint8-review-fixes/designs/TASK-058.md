# TASK-058: Compute Real Action Log-Probs at Trajectory Collection Time

## Summary

All three trajectory collectors (`_collect_single_tool_trajectory`, `_collect_multi_tool_trajectory`, `_collect_unified_trajectory`) hardcode `logprobs.append(-1.0)`. When GRPO computes the importance-sampling ratio `exp(new_lp - old_lp)`, the denominator `old_lp` is always `-1.0` regardless of action, producing a biased constant offset of `exp(1.0) ≈ 2.72` instead of the true policy ratio. This makes the policy gradient signal meaningless. Fix: at collection time, call `self.grpo._compute_action_logprob()` under `torch.no_grad()` and store the real scalar.

**Depends on TASK-056** — the key case fix must be applied first so that `_ACTION_TO_IDX.get(action)` returns the correct index.

## Current State

**File:** `/Users/rajesh/athena/training/stage2_agemem/trainer.py`

Three occurrences of the placeholder pattern:

**Line 253** (`_collect_single_tool_trajectory`):
```python
        state = {"operation": operation, "context": "training"}
        states.append(state)
        actions.append(operation)
        logprobs.append(-1.0)  # Placeholder until model provides real logprobs
```

**Line 287** (`_collect_multi_tool_trajectory`):
```python
            state = {"operation": operation, "step": step}
            states.append(state)
            actions.append(operation)
            logprobs.append(-1.0)
```

**Line 358** (`_collect_unified_trajectory`):
```python
            state["operation"] = operation
            states.append(state)
            actions.append(operation)
            logprobs.append(-1.0)  # Placeholder
```

The GRPO method signature (grpo.py:329):
```python
def _compute_action_logprob(self, model: Any, state: Dict[str, Any], action: str) -> "torch.Tensor":
```
It takes the state dict directly (not a string); it calls `self._format_state(state)` internally.

## Proposed Change

Replace each `logprobs.append(-1.0)` with a real log-prob computation. Wrap in `try/except` so the placeholder `-1.0` is used when the model is not loaded or torch is unavailable.

### Helper method `_get_logprob_for_step()`

Add a new private method to `AgeMemTrainer` to avoid repeating the try/except at all three sites:

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

### Replace all three placeholder lines

**Line 253:**
```python
        logprobs.append(self._get_logprob_for_step(state, operation))
```

**Line 287:**
```python
            logprobs.append(self._get_logprob_for_step(state, operation))
```

**Line 358:**
```python
            logprobs.append(self._get_logprob_for_step(state, operation))
```

## Files Modified

| File | Line(s) | Change |
|------|---------|--------|
| `training/stage2_agemem/trainer.py` | 253 | `logprobs.append(-1.0)` → `logprobs.append(self._get_logprob_for_step(state, operation))` |
| `training/stage2_agemem/trainer.py` | 287 | Same |
| `training/stage2_agemem/trainer.py` | 358 | Same |
| `training/stage2_agemem/trainer.py` | New method | Add `_get_logprob_for_step(self, state, action) -> float` |

## Acceptance Criteria

- No `logprobs.append(-1.0)` literal exists in any trajectory collector.
- `_get_logprob_for_step` returns `float(lp.item())` when the model has `encode` and `action_head`.
- `_get_logprob_for_step` returns `-1.0` when torch is unavailable or any exception occurs.
- TASK-056 must be applied first (`_ACTION_TO_IDX` uses `op.value` keys).
- All 173 tests pass, 4 skipped.

## Edge Cases & Risks

1. **Model not loaded**: The most common case in testing. `_compute_action_logprob` will raise (no `encode` method or model not loaded). The `except Exception` returns `-1.0` safely.

2. **Gradient accumulation**: `torch.no_grad()` ensures no computation graph is built during collection. The returned `.item()` is a Python float, not a tensor. This is correct — old log-probs should be detached scalars.

3. **TASK-056 dependency**: If TASK-056 has not been applied, `_ACTION_TO_IDX.get(action, 0)` still returns `0` for all actions. The log-prob computation would return a real number but always for action index 0. Verify TASK-056 is committed first.

4. **State dict at logprob time**: The state dict passed to `_get_logprob_for_step` is the same dict appended to `states`. In `_collect_unified_trajectory`, `state["operation"]` is set before calling `_get_logprob_for_step`, so the state reflects the chosen operation. This is correct.

## Test Notes

- Existing tests do not exercise trajectory collection with a real model, so they are unaffected.
- To verify: create a mock `grpo` object with a `_compute_action_logprob` method that returns `torch.tensor(-2.5)`, inject it into a trainer, call `_get_logprob_for_step({"op": "add"}, "add")`, assert result is `-2.5`.
