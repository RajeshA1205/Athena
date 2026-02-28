# TASK-044: Document hardcoded quality rewards in agemem

## Summary
`AgeMem._calculate_quality_reward()` returns hardcoded values (0.8 for SUMMARY, 0.9 for FILTER) with `# TODO` comments acknowledging they are placeholders. These values directly affect the GRPO reward signal used to train memory operation policies. The task is to make the placeholders explicit via docstring documentation and optionally configurable via `AgeMem` config, so future implementers know these values are not empirically derived and the system's reward function is not yet fully implemented.

## Current State

**File:** `memory/agemem.py` (lines 393–402)

```python
def _calculate_quality_reward(self, operation: MemoryOperation, result: MemoryOperationResult) -> float:
    """Calculate quality component of reward."""
    if operation == MemoryOperation.RETRIEVE:
        if result.data and isinstance(result.data, list):
            return min(1.0, len(result.data) / 5)
    elif operation == MemoryOperation.SUMMARY:
        return 0.8  # TODO: Compute based on compression quality
    elif operation == MemoryOperation.FILTER:
        return 0.9  # TODO: Compute based on noise reduction
    return 1.0 if result.success else 0.0
```

## Proposed Change

**Option A (documentation only — minimal change):**

```python
def _calculate_quality_reward(self, operation: MemoryOperation, result: MemoryOperationResult) -> float:
    """
    Calculate quality component of the GRPO reward signal.

    NOTE: SUMMARY and FILTER rewards are hardcoded placeholder values.
    These should be replaced with learned metrics once training data is available:
    - SUMMARY: should reflect compression quality (e.g. ROUGE score vs. original)
    - FILTER: should reflect noise reduction (e.g. relevance improvement post-filter)
    See training/stage2_agemem/ for the reward model training infrastructure.
    """
    if operation == MemoryOperation.RETRIEVE:
        if result.data and isinstance(result.data, list):
            return min(1.0, len(result.data) / 5)
    elif operation == MemoryOperation.SUMMARY:
        return self._quality_rewards.get("summary", 0.8)  # placeholder
    elif operation == MemoryOperation.FILTER:
        return self._quality_rewards.get("filter", 0.9)   # placeholder
    return 1.0 if result.success else 0.0
```

**Option B (configurable — slightly more work):**

Add to `AgeMem.__init__` config handling:
```python
self._quality_rewards = {
    "summary": float(config.get("summary_quality_reward", 0.8)),
    "filter": float(config.get("filter_quality_reward", 0.9)),
}
```

Option B is preferred as it makes the values adjustable without a code change.

## Files Modified

- `memory/agemem.py`
  - `__init__`: add `self._quality_rewards` dict from config (Option B)
  - Lines 393–402: update `_calculate_quality_reward` docstring and use `self._quality_rewards`

## Acceptance Criteria

- [ ] `_calculate_quality_reward` has a docstring explaining the placeholder nature of SUMMARY/FILTER values
- [ ] Values are either configurable (Option B) or prominently documented (Option A)
- [ ] Existing TODO comments are preserved or replaced with the docstring explanation
- [ ] All existing tests pass

## Edge Cases & Risks

- **Reward value change:** If the config doesn't specify overrides, the default values (0.8, 0.9) are unchanged. No behaviour change for existing code.
- **Training impact:** These values influence GRPO training rewards. Documenting them as placeholders sets expectations correctly for when real training begins.

## Test Notes

- No new tests needed — this is a documentation/configurability improvement.
- If Option B is chosen, add a test: `AgeMem(config={"summary_quality_reward": 0.5})` and verify the reward value changes.
