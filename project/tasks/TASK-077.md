# TASK-077: Verify MemoryActionHead and GRPO pipeline with new encode() output

## Status
- **State:** Queued
- **Priority:** 🟡 High
- **Depends on:** TASK-076
- **Created:** 2026-02-26

## Objective
Audit the GRPO training pipeline and `MemoryActionHead` to confirm they continue to work correctly after the mlx-lm migration. Specifically:
1. Confirm that `MemoryActionHead` input dimension assumptions are clearly documented and that a dimension mismatch between mlx-path `encode()` output and the action head will fail fast with a clear error (not silently corrupt gradients)
2. Add a runtime guard in `MemoryActionHead.forward()` or `OLMoEModel` that raises a clear `RuntimeError` if `encode()` is called on the mlx backend and the output is fed into `action_head` (since `action_head` is None on the mlx path and this would otherwise produce an obscure `AttributeError`)
3. Document in `MemoryActionHead` class docstring and in `OLMoEModel.encode()` that GRPO training (which requires action_head) is only supported on the transformers backend

No functional logic changes to GRPO are needed — this task is defensive hardening and documentation only.

## Context
After TASK-074:
- mlx path: `self.action_head is None`
- transformers path: `self.action_head` is a `MemoryActionHead` instance

After TASK-076:
- `encode()` on mlx path returns a vocab-size tensor (not hidden_size)
- `encode()` on transformers path returns a hidden_size tensor (correct for action_head)

The GRPO trainer (`/Users/rajesh/athena/training/stage2_agemem/trainer.py`) calls:
```python
embedding = self.policy_model.encode(text)
logits = self.policy_model.action_head(embedding)
```
If `action_head` is None (mlx path), this produces `TypeError: 'NoneType' object is not callable`. This should be a clear, informative error.

See `/Users/rajesh/athena/models/olmoe.py`, `/Users/rajesh/athena/training/stage2_agemem/grpo.py`, and `/Users/rajesh/athena/training/stage2_agemem/trainer.py`.

## Scope & Constraints
- May modify: `models/olmoe.py` — add a `get_action_logits(embedding)` helper method to OLMoEModel that checks action_head availability before calling forward; update MemoryActionHead class docstring
- Must NOT modify: GRPO trainer, GRPO config, MemoryActionHead forward logic
- Changes must be purely additive (new helper method + docstring update only)
- If changing trainer.py to use `get_action_logits()` is out of scope for this task, document the risk in Review Notes and create an /idea for a follow-on task instead

## Input
- `/Users/rajesh/athena/models/olmoe.py` (after TASK-076 changes)
- `/Users/rajesh/athena/training/stage2_agemem/trainer.py`
- `/Users/rajesh/athena/training/stage2_agemem/grpo.py`

## Expected Output

### Addition to `models/olmoe.py`
New method on `OLMoEModel`:
```python
def get_action_logits(self, embedding: "torch.Tensor") -> "torch.Tensor":
    """
    Run the MemoryActionHead on an embedding from encode().

    Only available on the transformers backend (action_head is None on the mlx path).
    GRPO training must use the transformers backend.

    Raises:
        RuntimeError: If action_head is not available (mlx backend or model not loaded).
    """
    if self.action_head is None:
        raise RuntimeError(
            "action_head is not available. GRPO training requires the transformers "
            "backend (use_mlx=False or mlx-lm not installed)."
        )
    return self.action_head(embedding)
```

### Update to `MemoryActionHead` class docstring
Add a note:
```
Note: Only instantiated on the transformers backend. The mlx-lm backend does not
attach an action_head. GRPO training is therefore only supported with transformers.
```

## Acceptance Criteria
- [ ] `OLMoEModel.get_action_logits()` method exists and raises `RuntimeError` when `self.action_head is None`
- [ ] `MemoryActionHead` class docstring includes a note about transformers-only availability
- [ ] `python3 -m pytest tests/ -q` shows 173 passed, 4 skipped

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
