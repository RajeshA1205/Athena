# TASK-056: Fix `_ACTION_TO_IDX` key case mismatch in GRPO

## Summary

`_ACTION_TO_IDX` is built using `op.name` (uppercase enum member names: `"ADD"`, `"RETRIEVE"`, etc.) but all callers pass `op.value` (lowercase: `"add"`, `"retrieve"`) from `MemoryOperation`. The `.get(action, 0)` fallback silently maps every action to index 0, meaning every operation is treated as `ADD` during GRPO training. This completely breaks the policy gradient signal.

## Current State

**File:** `/Users/rajesh/athena/training/stage2_agemem/grpo.py`

The dict comprehension at line 37 builds keys from `op.name`:

```python
# lines 36-38
from memory.agemem import MemoryOperation
_ACTION_TO_IDX: Dict[str, int] = {op.name: i for i, op in enumerate(MemoryOperation)}
_MEMORY_OP_AVAILABLE = True
```

This produces: `{"ADD": 0, "UPDATE": 1, "DELETE": 2, "RETRIEVE": 3, "SUMMARY": 4, "FILTER": 5}`

The consumer at line 349 looks up with `action` which is always a lowercase string:

```python
# line 349
action_idx = _ACTION_TO_IDX.get(action, 0)
```

The `action` parameter comes from `Trajectory.actions` which are populated by the trainer using `self.config.ltm_operations` and `self.config.stm_operations`. These are defined in `/Users/rajesh/athena/training/stage2_agemem/trainer.py` lines 50-53:

```python
# trainer.py lines 50-53
if self.ltm_operations is None:
    self.ltm_operations = ["add", "update", "delete"]
if self.stm_operations is None:
    self.stm_operations = ["retrieve", "summary", "filter"]
```

The `MemoryOperation` enum in `/Users/rajesh/athena/memory/agemem.py` lines 23-30 confirms the values are lowercase:

```python
class MemoryOperation(Enum):
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    RETRIEVE = "retrieve"
    SUMMARY = "summary"
    FILTER = "filter"
```

Since `_ACTION_TO_IDX` keys are uppercase (`"ADD"`) but lookups use lowercase (`"add"`), `.get(action, 0)` always returns `0`.

## Proposed Change

**File:** `/Users/rajesh/athena/training/stage2_agemem/grpo.py`, line 37.

Replace `op.name` with `op.value`:

```python
# BEFORE (line 37)
_ACTION_TO_IDX: Dict[str, int] = {op.name: i for i, op in enumerate(MemoryOperation)}

# AFTER
_ACTION_TO_IDX: Dict[str, int] = {op.value: i for i, op in enumerate(MemoryOperation)}
```

This produces: `{"add": 0, "update": 1, "delete": 2, "retrieve": 3, "summary": 4, "filter": 5}` which matches all caller conventions.

## Files Modified

| File | Line(s) | Change |
|------|---------|--------|
| `/Users/rajesh/athena/training/stage2_agemem/grpo.py` | 37 | `op.name` -> `op.value` in dict comprehension |

## Acceptance Criteria

- `_ACTION_TO_IDX` maps lowercase strings to correct indices: `{"add": 0, "update": 1, "delete": 2, "retrieve": 3, "summary": 4, "filter": 5}`.
- `_ACTION_TO_IDX.get("add", 0)` returns `0`, `_ACTION_TO_IDX.get("retrieve", 0)` returns `3`, etc.
- `_compute_action_logprob` indexes into the correct position of the log-softmax vector for each operation.
- All 173 tests pass, 4 skipped.

## Edge Cases & Risks

1. **Fallback default still 0**: If an unknown action string is passed, `.get(action, 0)` still defaults to index 0 (`"add"`). This is acceptable as a safe default -- the alternative would be raising a `KeyError`, but the training pipeline should be resilient to unexpected inputs.

2. **Import failure path unchanged**: If `MemoryOperation` cannot be imported (line 39-41), `_ACTION_TO_IDX` remains `{}` and the fallback to index 0 applies for all actions. This is the same behavior as before, just now the happy path works correctly.

3. **TASK-058 dependency**: TASK-058 (computing real action log-probs at collection time) depends on this fix being applied first. Without it, `_compute_action_logprob` would index incorrectly even with real log-probs.

## Test Notes

- No new tests required; this is a one-line semantic fix.
- Existing GRPO tests exercise the `_ACTION_TO_IDX` lookup indirectly through `_compute_action_logprob`. With the model fallback path (line 362-363), the lookup is only reached when a real model is available, so existing tests pass unchanged.
- To verify manually: `from memory.agemem import MemoryOperation; d = {op.value: i for i, op in enumerate(MemoryOperation)}; assert d["retrieve"] == 3`.
