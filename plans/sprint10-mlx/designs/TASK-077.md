# TASK-077: Verify MemoryActionHead and GRPO Pipeline -- Defensive Hardening

## Summary

Add a `get_action_logits(embedding)` helper method to `OLMoEModel` that checks `action_head` availability before forwarding, raising an informative `RuntimeError` on the mlx path (where `action_head is None`). Update the `MemoryActionHead` class docstring to document transformers-only availability. No functional logic changes to GRPO or `MemoryActionHead.forward()`.

## Current State

**File:** `models/olmoe.py`

After TASK-074, the mlx path leaves `self.action_head = None`. After TASK-076, `encode()` on the mlx path returns a `(vocab_size,)` tensor.

If GRPO trainer code (`grpo.py` line 343) calls `model.action_head(embedding)` when `action_head is None`, the result is:
```
TypeError: 'NoneType' object is not callable
```
This is cryptic and unhelpful.

**GRPO call sites** (both in `training/stage2_agemem/`):

`grpo.py` line 343:
```python
logits = model.action_head(embedding)
```

`trainer.py` line 432:
```python
logits = self.model.action_head(embedding)
```

Both are guarded by `hasattr(self.model, "action_head") and self.model.action_head is not None` checks, so they will not crash at runtime. However, the guard pattern is duplicated and the failure mode (silent fallback to random selection / zero tensor) is not well documented.

**`MemoryActionHead` class docstring** (lines 425-439):
```python
class MemoryActionHead(nn.Module):
    """
    MLP classification head on top of OLMoE for GRPO memory action selection.
    ...
    Note: Doubles GPU memory for the policy model copy during GRPO training.
    Training and inference must run in separate phases to avoid gradient corruption.
    """
```
No mention of mlx-backend unavailability.

## Proposed Change

### 1. New method on `OLMoEModel` (after `encode()`, before `parameters()`)

```python
def get_action_logits(self, embedding: "torch.Tensor") -> "torch.Tensor":
    """
    Run the MemoryActionHead on an embedding from encode().

    Only available on the transformers backend (action_head is None on
    the mlx path). GRPO training must use the transformers backend.

    Args:
        embedding: 1-D tensor from encode(), shape (hidden_size,).

    Returns:
        Logit tensor of shape (ACTION_DIM,).

    Raises:
        RuntimeError: If action_head is not available (mlx backend
            or model not loaded with transformers).
    """
    if self.action_head is None:
        raise RuntimeError(
            "action_head is not available. GRPO training requires the "
            "transformers backend (use_mlx=False or mlx-lm not installed)."
        )
    return self.action_head(embedding)
```

### 2. Update `MemoryActionHead` class docstring (lines 425-439)

Add after the existing "Training and inference must run in separate phases" note:

```
Note: Only instantiated on the transformers backend. The mlx-lm backend
does not attach an action_head (it remains None on OLMoEModel). GRPO
training is therefore only supported with the transformers backend.
```

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `models/olmoe.py` | after `encode()` (~346) | Add `get_action_logits()` method |
| `models/olmoe.py` | 425-439 | Update `MemoryActionHead` class docstring |

## Acceptance Criteria

- `OLMoEModel.get_action_logits(embedding)` exists and returns `self.action_head(embedding)` when `action_head` is a `MemoryActionHead`.
- `get_action_logits()` raises `RuntimeError` with an informative message when `self.action_head is None`.
- `MemoryActionHead` class docstring includes a note about transformers-only availability.
- No changes to `MemoryActionHead.forward()`, GRPO trainer, or GRPO config.
- 173 tests pass, 4 skipped.

## Edge Cases & Risks

1. **Trainer not updated to use `get_action_logits()`**: The GRPO trainer and `AgeMemTrainer` currently call `model.action_head(embedding)` directly, guarded by `action_head is not None` checks. Updating them to use `get_action_logits()` is out of scope for this task (as stated in the task brief). The existing guards prevent crashes, but a follow-on task should migrate callers to use the new method for consistency. Document this in review notes.
2. **Stub `MemoryActionHead`**: When `TRANSFORMERS_AVAILABLE=False`, the stub class (lines 456-464) also needs the docstring update for completeness. However, the stub raises `ImportError` on instantiation, so the note about mlx is less relevant. Update both docstrings for consistency anyway.
3. **No functional changes**: This task is purely additive -- new method + docstring updates. Zero risk of breaking existing behavior.

## Test Notes

- No new tests required by the task brief. The method can be tested by constructing an `OLMoEModel()` (unloaded) and verifying `get_action_logits()` raises `RuntimeError` (since `action_head` defaults to `None`).
- Run `python3 -m pytest tests/ -q` -- expect 173 passed, 4 skipped.
