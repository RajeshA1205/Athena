# TASK-050: Implement Distinct Unified Trajectory Logic for Stage 3

## Summary

`_collect_unified_trajectory()` is a direct passthrough to `_collect_multi_tool_trajectory()`. Stage 3 (UNIFIED) should implement distinct behavior: longer operation sequences (4-8 steps), model-driven operation selection, and a trajectory-level quality bonus based on end-to-end retrieval quality.

## Current State

**File:** `/Users/rajesh/athena/training/stage2_agemem/trainer.py`, lines 308-314

```python
def _collect_unified_trajectory(self) -> Optional[Trajectory]:
    """
    Collect trajectory for unified management.
    Full autonomous memory decisions.
    """
    # Similar to multi-tool but with more complex scenarios
    return self._collect_multi_tool_trajectory()
```

After TASK-049, this will be:

```python
async def _collect_unified_trajectory(self) -> Optional[Trajectory]:
    return await self._collect_multi_tool_trajectory()
```

Still a passthrough. Stage 3 should be meaningfully different from Stage 2:
- Stage 2 (MULTI_TOOL): Random operation selection, 2-4 steps, per-step rewards only.
- Stage 3 (UNIFIED): Model-driven operation selection, 4-8 steps, both per-step and trajectory-level rewards, includes both LTM and STM operations.

## Proposed Change

Modify `/Users/rajesh/athena/training/stage2_agemem/trainer.py` only. This task depends on TASK-049 being complete (real AgeMem operations via `_execute_operation()`).

### Replace `_collect_unified_trajectory()` with distinct Stage 3 logic

```python
async def _collect_unified_trajectory(self) -> Optional[Trajectory]:
    """
    Collect trajectory for unified (Stage 3) memory management.

    Distinct from Stage 2 multi-tool:
    - Longer sequences (4-8 steps)
    - Model-driven operation selection when available (uses action_head logits
      to sample the next operation, falling back to random if model lacks
      encode/action_head)
    - Mandatory mix of both LTM and STM operations
    - Trajectory-level quality bonus based on end-to-end retrieval quality
    - Planning context: initial state describes the overall memory goal
    """
    import random

    states = []
    actions = []
    logprobs = []
    rewards = []

    seq_length = random.randint(4, 8)
    all_ops = self.config.ltm_operations + self.config.stm_operations

    # Planning context: describe the overall memory management goal
    plan = random.choice([
        "Ingest new market data and verify retrieval quality",
        "Update stale entries and consolidate memory via summary",
        "Add diverse data points then filter for relevance",
        "Build knowledge base with add operations then retrieve and summarize",
    ])

    # Track which operation categories have been used
    used_ltm = False
    used_stm = False

    for step in range(seq_length):
        # Build state with planning context
        state = {
            "operation": "pending",
            "step": step,
            "total_steps": seq_length,
            "plan": plan,
            "history": [a for a in actions],  # copy of actions so far
        }

        # Model-driven operation selection
        operation = self._select_operation_unified(
            state, all_ops, step, seq_length, used_ltm, used_stm
        )

        # Track LTM/STM usage
        if operation in self.config.ltm_operations:
            used_ltm = True
        if operation in self.config.stm_operations:
            used_stm = True

        state["operation"] = operation
        states.append(state)
        actions.append(operation)
        logprobs.append(-1.0)  # Placeholder

        outcome = await self._execute_operation(operation)
        rewards.append(self.reward_fn.compute(outcome))

    # Trajectory-level quality bonus: retrieve and score end-to-end quality
    bonus = await self._compute_trajectory_bonus()
    # Distribute bonus across all steps
    if rewards:
        per_step_bonus = bonus / len(rewards)
        rewards = [r + per_step_bonus for r in rewards]

    return Trajectory(
        states=states,
        actions=actions,
        action_logprobs=logprobs,
        rewards=rewards,
    )
```

### Add `_select_operation_unified()` helper

```python
def _select_operation_unified(
    self,
    state: Dict[str, Any],
    all_ops: List[str],
    step: int,
    total_steps: int,
    used_ltm: bool,
    used_stm: bool,
) -> str:
    """
    Select the next operation for Stage 3 unified trajectories.

    Uses the policy model's action_head when available to sample an operation
    from the model's learned distribution. Falls back to heuristic selection
    that ensures both LTM and STM operations are represented.

    Args:
        state: Current state dict.
        all_ops: All available operation names.
        step: Current step index.
        total_steps: Total steps in this trajectory.
        used_ltm: Whether an LTM operation has been used so far.
        used_stm: Whether an STM operation has been used so far.

    Returns:
        Selected operation name.
    """
    import random

    # Try model-driven selection if available
    if (
        hasattr(self.model, "encode")
        and hasattr(self.model, "action_head")
        and self.model.action_head is not None
    ):
        try:
            import torch
            state_text = self.grpo._format_state(state)
            with torch.no_grad():
                embedding = self.model.encode(state_text)
                logits = self.model.action_head(embedding)
                probs = torch.softmax(logits, dim=-1)
                action_idx = torch.multinomial(probs, 1).item()
            # Map index back to operation name
            from memory.agemem import MemoryOperation
            ops_list = list(MemoryOperation)
            if action_idx < len(ops_list):
                return ops_list[action_idx].value
        except Exception:
            pass  # Fall back to heuristic

    # Heuristic: ensure both LTM and STM are used in the trajectory
    remaining = total_steps - step
    if not used_ltm and remaining <= 2:
        # Force an LTM operation
        return random.choice(self.config.ltm_operations)
    if not used_stm and remaining <= 2:
        # Force an STM operation
        return random.choice(self.config.stm_operations)

    return random.choice(all_ops)
```

### Add `_compute_trajectory_bonus()` async helper

```python
async def _compute_trajectory_bonus(self) -> float:
    """
    Compute a trajectory-level quality bonus by running an end-to-end
    retrieval quality check.

    After a full sequence of operations, retrieve a known snippet and
    score how well the memory system performs. This bonus rewards
    trajectories that leave the memory system in a good state.

    Returns:
        Bonus reward value in [0.0, 1.0].
    """
    import random

    try:
        # Pick a snippet that was likely added during this trajectory
        query = random.choice(self._TRAINING_SNIPPETS).split(",")[0]
        results = await self.agemem.retrieve(query, top_k=3)

        if not isinstance(results, list):
            return 0.0

        # Score: fraction of requested results actually returned
        retrieval_quality = min(1.0, len(results) / 3.0)

        # Latency bonus: fast retrieval is better (not measured here,
        # since per-step latency is already in step rewards)
        return retrieval_quality * 0.5  # Scale to [0, 0.5] to not overwhelm step rewards

    except Exception:
        return 0.0
```

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `/Users/rajesh/athena/training/stage2_agemem/trainer.py` | Lines 308-314 (post-TASK-049) | Replace passthrough with distinct Stage 3 logic |
| `/Users/rajesh/athena/training/stage2_agemem/trainer.py` | New method | Add `_select_operation_unified()` |
| `/Users/rajesh/athena/training/stage2_agemem/trainer.py` | New method | Add `_compute_trajectory_bonus()` |

## Acceptance Criteria

- `_collect_unified_trajectory()` does NOT delegate to `_collect_multi_tool_trajectory()`.
- Stage 3 trajectories have 4-8 steps (compared to 2-4 in Stage 2).
- Each trajectory includes a `plan` field in the initial state for planning context.
- When the policy model has `encode` and `action_head`, operations are sampled from the model's logits via `torch.multinomial`.
- When the model lacks encode/action_head, a heuristic ensures both LTM and STM operations are represented.
- A trajectory-level quality bonus is computed via `_compute_trajectory_bonus()` and distributed across all steps.
- The bonus is scaled to [0, 0.5] to avoid overwhelming per-step rewards.
- All 171 tests pass, 6 skipped.

## Edge Cases & Risks

1. **Model not loaded**: When the model lacks `encode`/`action_head` (which is the common case in testing), the method falls back to random+heuristic selection. This is safe and provides a meaningful baseline.

2. **MemoryOperation import**: The model-driven selection imports `MemoryOperation` to map action indices back to operation names. This import is inside a try/except, so if the import fails, it falls back gracefully.

3. **Trajectory bonus computation**: The bonus calls `self.agemem.retrieve()`, which may fail or return empty results. The try/except returns 0.0 on failure. The bonus is capped at 0.5 to prevent it from dominating the per-step rewards.

4. **LTM/STM coverage**: The heuristic forces LTM and STM operations when the trajectory is nearly complete and one category is unused. With 4-8 steps, there is always room for at least one forced operation in the last 2 steps.

5. **Interaction with TASK-049**: This task depends on `_execute_operation()` from TASK-049. If TASK-049 is not complete, this task cannot be implemented. The `_TRAINING_SNIPPETS` class attribute from TASK-049 is also used here.

## Test Notes

- Test with a mock model that has `encode` and `action_head` attributes to verify model-driven selection path.
- Test with a model that lacks these attributes to verify heuristic fallback.
- Verify trajectory length is in [4, 8] range.
- Verify that `plan` appears in `states[0]`.
- Verify that the trajectory-level bonus is added to each step's reward.
- Verify that both LTM and STM operations appear in a trajectory (at least with the heuristic path).
