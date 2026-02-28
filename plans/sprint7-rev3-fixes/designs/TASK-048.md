# TASK-048: Secure torch.load in GRPO Checkpoint Loading

## Summary

`StepwiseGRPO.load()` catches `TypeError` when calling `torch.load(path, weights_only=True)` and falls back to bare `torch.load(path)` (without `weights_only`), which allows arbitrary code execution via pickle deserialization. Remove the unsafe fallback and enforce PyTorch >= 2.0.

## Current State

**File:** `/Users/rajesh/athena/training/stage2_agemem/grpo.py`, lines 391-399

```python
def load(self, path: str) -> None:
    """Load GRPO state."""
    try:
        checkpoint = torch.load(path, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path)
    self._step_count = checkpoint["step_count"]
    if self.optimizer and checkpoint["optimizer_state"]:
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
```

The `except TypeError` on line 395-396 was a compatibility shim for PyTorch < 2.0 where `weights_only` was not a recognized parameter. However, this silently enables unsafe deserialization of arbitrary pickle payloads, which is a security vulnerability if checkpoints come from untrusted sources.

The torch import block is at lines 14-19:

```python
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
```

## Proposed Change

Modify `/Users/rajesh/athena/training/stage2_agemem/grpo.py` only.

### 1. Add version check after the torch import block (after line 19)

```python
if TORCH_AVAILABLE:
    _torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
    if _torch_version < (2, 0):
        import warnings
        warnings.warn(
            f"ATHENA GRPO requires PyTorch >= 2.0 for safe checkpoint loading. "
            f"Found {torch.__version__}. Training pipeline may not function correctly.",
            UserWarning,
            stacklevel=1,
        )
```

### 2. Remove the `except TypeError` fallback in `load()` (lines 391-399)

Replace with:

```python
def load(self, path: str) -> None:
    """Load GRPO state."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for checkpoint loading")
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    self._step_count = checkpoint["step_count"]
    if self.optimizer and checkpoint["optimizer_state"]:
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
```

Notes:
- `weights_only=True` is the only safe option -- no fallback.
- Added `map_location="cpu"` to avoid errors when loading a GPU checkpoint on a CPU-only machine. This is safe because the optimizer states and step count are small scalar/tensor data.
- Removed the try/except entirely. If PyTorch < 2.0 is installed, `weights_only` raises `TypeError`, which will now propagate as an unhandled exception -- correctly signaling that the version is unsupported.

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `/Users/rajesh/athena/training/stage2_agemem/grpo.py` | After line 19 | Add `_torch_version` check with warning |
| `/Users/rajesh/athena/training/stage2_agemem/grpo.py` | Lines 391-399 | Remove `except TypeError` fallback, add `map_location="cpu"` |

## Acceptance Criteria

- `StepwiseGRPO.load()` never calls `torch.load` without `weights_only=True`.
- No `except TypeError` fallback exists in the `load()` method.
- `map_location="cpu"` is included for cross-device portability.
- A `UserWarning` is emitted at import time if PyTorch < 2.0 is detected.
- If PyTorch < 2.0 is actually used at runtime, `load()` raises `TypeError` (from torch itself) instead of silently falling back to unsafe loading.
- All 171 tests pass, 6 skipped.

## Edge Cases & Risks

1. **PyTorch version string parsing**: `torch.__version__` can be `"2.1.0"`, `"2.0.0+cu117"`, or nightly builds like `"2.4.0.dev20240101"`. Splitting on `"."` and taking the first two components, then parsing as int, handles all standard formats. The `+cu117` suffix is ignored because `split(".")[:2]` returns `["2", "1"]` for `"2.1.0+cu117"` (the third element `"0+cu117"` is discarded). Edge case: if a version string is malformed (e.g., `"2.x.y"`), the `int()` call raises `ValueError`. This is acceptable -- it surfaces a clear error at import time.

2. **Checkpoint format compatibility**: Existing checkpoints saved by `save()` use `torch.save({...})`, which produces standard pickle format. `weights_only=True` only allows loading tensors, primitive Python types, and dicts/lists. The saved checkpoint contains `step_count` (int), `optimizer_state` (dict of tensors), and `config` (GRPOConfig dataclass). The `config` field is a dataclass, which `weights_only=True` may not be able to deserialize. This needs verification. If it fails, `save()` should be updated to serialize `config` as a plain dict via `dataclasses.asdict()`. **This is a known risk -- verify during implementation.**

3. **`map_location="cpu"`**: The `save()` method at lines 383-389 saves optimizer state which may contain GPU tensors. Adding `map_location="cpu"` ensures these load correctly on CPU-only machines. On GPU machines, the optimizer state will be on CPU after loading but will be moved to GPU when the optimizer is used.

## Test Notes

- Existing tests do not call `StepwiseGRPO.load()` directly, so the change is safe.
- If adding a test: create a temporary file with `torch.save({"step_count": 0, "optimizer_state": None, "config": None}, path)`, then call `grpo.load(path)` and verify `_step_count == 0`.
- Verify the `GRPOConfig` dataclass serialization with `weights_only=True`. If it fails, update `save()` to use `dataclasses.asdict(self.config)` and `load()` to reconstruct with `GRPOConfig(**checkpoint["config"])`.
