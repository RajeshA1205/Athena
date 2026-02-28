# TASK-025: Fix unsafe torch.load deserialization

## Summary
`GRPOTrainer.load()` calls `torch.load(path)` without `weights_only=True`. Since PyTorch 1.13, omitting this flag means the checkpoint is deserialized via Python `pickle`, which can execute arbitrary code embedded in a malicious `.pt` file. The saved checkpoint contains only plain Python types (`int`, `dict`, `GRPOConfig` dataclass) and an optimizer `state_dict` — no tensors that would fail under `weights_only=True`. Adding the flag eliminates the attack surface with no functional change.

## Current State

**File:** `training/stage2_agemem/grpo.py`

```python
# Line 324-329
def load(self, path: str) -> None:
    """Load GRPO state."""
    checkpoint = torch.load(path)          # <-- unsafe: no weights_only
    self._step_count = checkpoint["step_count"]
    if self.optimizer and checkpoint["optimizer_state"]:
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
```

The corresponding `save()` at line 316–322 writes:
```python
torch.save({
    "step_count": self._step_count,
    "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
    "config": self.config,   # <-- GRPOConfig dataclass, non-tensor
}, path)
```

## Proposed Change

```python
def load(self, path: str) -> None:
    """Load GRPO state."""
    try:
        checkpoint = torch.load(path, weights_only=True)
    except Exception:
        # weights_only=True rejects non-tensor types (e.g. GRPOConfig dataclass).
        # Fall back with explicit acknowledgement; migrate save() to exclude config
        # from the weights file in a future task.
        checkpoint = torch.load(path, weights_only=False)  # noqa: S614
    self._step_count = checkpoint["step_count"]
    if self.optimizer and checkpoint["optimizer_state"]:
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
```

**Preferred approach** (if `GRPOConfig` serialisation can be verified): confirm that `torch.load(path, weights_only=True)` loads the checkpoint without error in a test, and use it unconditionally:

```python
checkpoint = torch.load(path, weights_only=True)
```

If `GRPOConfig` is a plain dataclass with only Python primitives, `weights_only=True` should work. Verify by running the save/load round-trip in a unit test.

## Files Modified

- `training/stage2_agemem/grpo.py`
  - Line 326: `torch.load(path)` → `torch.load(path, weights_only=True)` (or the try/except fallback if needed)

## Acceptance Criteria

- [ ] `torch.load` call includes `weights_only=True` (or a documented fallback with comment)
- [ ] Save/load round-trip: `save(path)` followed by `load(path)` restores `_step_count` and optimizer state correctly
- [ ] No `UserWarning` about `weights_only` from PyTorch at runtime

## Edge Cases & Risks

- **`GRPOConfig` dataclass in checkpoint:** `weights_only=True` restricts deserialized types to tensors, dicts, lists, tuples, and a small allowlist. If `GRPOConfig` is not on the allowlist, loading will raise `UnpicklingError`. Test this before committing to the simple form.
- **Older PyTorch versions:** `weights_only` parameter was added in PyTorch 1.13. If the project targets older versions, wrap in `try/except TypeError` as a compatibility shim (unlikely given the codebase uses features requiring ≥ 2.0).
- **Optimizer state:** Optimizer `state_dict()` contains tensors and is fully compatible with `weights_only=True`.

## Test Notes

- Add a round-trip test: create a `GRPOTrainer` with a mock optimizer, call `save(tmp_path)`, call `load(tmp_path)`, assert `_step_count` and optimizer state are restored.
- The test implicitly validates that `weights_only=True` does not break loading.
