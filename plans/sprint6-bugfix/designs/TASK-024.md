# TASK-024: Fix GRPO reference model aliasing and action log-prob placeholder

## Summary
`GRPOTrainer._update_reference_model()` assigns `self.reference_model = self.policy_model`, creating a reference alias rather than an independent copy. As a result the KL divergence term `log π_θ(a|s) - log π_ref(a|s)` always evaluates to zero because both names point to the same object — GRPO training is a functional no-op. Additionally, `_get_action_logprob` and `_get_reference_logprob` return the constant `0.0`, compounding the problem. This task replaces the alias with a real deep copy and replaces the stub return with a real implementation using OLMoE + MLP action head architecture (6-dim output over `MemoryOperation` actions).

## Current State

**File:** `training/stage2_agemem/grpo.py`

```python
# Line 297-301
def _get_action_logprob(self, state: Dict[str, Any], action: str) -> torch.Tensor:
    """Get log probability of action given state from policy model."""
    # This should be implemented based on how the policy model works
    # Placeholder returning a learnable parameter for demonstration
    return torch.tensor(0.0, requires_grad=True)

# Line 303-306
def _get_reference_logprob(self, state: Dict[str, Any], action: str) -> torch.Tensor:
    """Get log probability from reference model."""
    # Placeholder
    return torch.tensor(0.0)

# Line 308-314
def _update_reference_model(self) -> None:
    """Update reference model with current policy weights."""
    if self.reference_model is None:
        # Create reference model (simplified - in practice would deep copy)
        self.reference_model = self.policy_model
    # In practice: self.reference_model.load_state_dict(self.policy_model.state_dict())
    self.logger.debug("Reference model updated")
```

The comment on line 313 explicitly describes the correct approach but it is never executed.

## Proposed Change

**`_update_reference_model`:** Replace the alias with a proper deep copy on first creation, then use `load_state_dict` for subsequent updates:

```python
import copy

def _update_reference_model(self) -> None:
    """Snapshot current policy weights into the reference model."""
    if self.reference_model is None:
        self.reference_model = copy.deepcopy(self.policy_model)
    else:
        self.reference_model.load_state_dict(self.policy_model.state_dict())
    self.logger.debug("Reference model updated at step %d", self._step_count)
```

**`_get_action_logprob` and `_get_reference_logprob`:** Replace the constant `0.0` return with a real implementation. The policy model architecture is now resolved (see OQ-1a/OQ-1b in `project/decisions/sprint6-open-questions.md`):

- **Action space:** 6 discrete actions matching `MemoryOperation` enum exactly: `ADD`, `UPDATE`, `DELETE`, `RETRIEVE`, `SUMMARY`, `FILTER`. MLP output dimension = 6.
- **Graphiti mapping note:** ADD and UPDATE are separate GRPO actions (separate MLP outputs). At the Graphiti layer, both map to Graphiti's `add_episode` API -- this is an implementation detail inside the memory layer, not reflected in the action space.
- **Architecture:** The MLP is a classification head sitting ON TOP of OLMoE. The `policy_model` passed to `StepwiseGRPO` is OLMoE + MLP head (not a standalone MLP). This means TASK-024 (GRPO) and TASK-029 (OLMoE wiring) share the same model instance.

Pseudocode for the implementation:

```python
# Action enum indices (matching MemoryOperation):
# 0=ADD, 1=UPDATE, 2=DELETE, 3=RETRIEVE, 4=SUMMARY, 5=FILTER
ACTION_DIM = 6

def _get_action_logprob(self, state: Dict[str, Any], action: str) -> torch.Tensor:
    """Get log probability of action given state from policy model.

    Architecture: OLMoE encodes the state into an embedding, then the MLP
    action head maps it to a 6-dim logit vector over MemoryOperation actions.
    """
    # 1. Encode state with OLMoE backbone
    state_text = self._format_state(state)
    embedding = self.policy_model.encode(state_text)  # OLMoE hidden state

    # 2. Pass embedding through MLP action head -> logits of shape (6,)
    logits = self.policy_model.action_head(embedding)

    # 3. log_softmax over action dimension
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # 4. Index by the action taken
    action_idx = MemoryOperation[action].value  # or a lookup dict
    return log_probs[action_idx]

def _get_reference_logprob(self, state: Dict[str, Any], action: str) -> torch.Tensor:
    """Get log probability from reference model (same architecture, frozen weights)."""
    with torch.no_grad():
        state_text = self._format_state(state)
        embedding = self.reference_model.encode(state_text)
        logits = self.reference_model.action_head(embedding)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        action_idx = MemoryOperation[action].value
        return log_probs[action_idx]
```

## Files Modified

- `training/stage2_agemem/grpo.py`
  - Line 297–301: `_get_action_logprob` — replace `return torch.tensor(0.0, ...)` with real OLMoE encode -> MLP head -> log_softmax implementation
  - Line 303–306: `_get_reference_logprob` — replace `return torch.tensor(0.0)` with frozen reference model forward pass (same architecture)
  - Line 308–314: `_update_reference_model` — replace alias assignment with `copy.deepcopy` + `load_state_dict`
  - Line 1 (top of file): add `import copy`
  - Add `ACTION_DIM = 6` constant and action-to-index mapping for `MemoryOperation` enum
  - **Note:** The `policy_model` passed to `StepwiseGRPO` must be the OLMoE+MLP composite model from TASK-029 (shared instance)

## Acceptance Criteria

- [ ] `self.reference_model is not self.policy_model` after `_update_reference_model()` is called
- [ ] Mutating a parameter of `policy_model` does not affect `reference_model`
- [ ] `_get_action_logprob(state, "ADD")` returns a scalar log-probability tensor with `requires_grad=True`
- [ ] `_get_action_logprob` output shape is scalar (indexed from a 6-dim log_softmax output)
- [ ] `_get_reference_logprob(state, "ADD")` returns a scalar tensor with `requires_grad=False`
- [ ] All 6 MemoryOperation actions (ADD, UPDATE, DELETE, RETRIEVE, SUMMARY, FILTER) are valid action inputs
- [ ] All existing tests still pass

## Edge Cases & Risks

- **Model without `state_dict()`:** If `policy_model` is not a `torch.nn.Module` (e.g. a mock), `load_state_dict` will fail. The `copy.deepcopy` path on first creation is safe for any object. The subsequent `load_state_dict` path should be guarded: `if hasattr(self.policy_model, 'state_dict')`.
- **Memory:** `deepcopy` doubles the GPU memory footprint for the model. If memory is tight, document this in the method docstring.
- **Shared model instance:** Since the OLMoE model is shared between GRPO (TASK-024) and agent reasoning (TASK-029), care must be taken that GRPO gradient updates do not corrupt in-flight inference. In practice this is safe because training and inference run in separate phases, but document the constraint.
- **ADD vs UPDATE at Graphiti layer:** Both ADD and UPDATE GRPO actions map to Graphiti's `add_episode` API. The memory layer must handle this mapping internally. The GRPO reward signal should still distinguish between ADD and UPDATE based on the memory state (e.g., UPDATE gets higher reward when modifying stale entries).

## Test Notes

- Existing tests do not appear to exercise `_update_reference_model` directly (they test `train_step` at a higher level with mocked models).
- Add a unit test: instantiate `GRPOTrainer` with a mock `nn.Module` policy model, call `_update_reference_model()`, assert `trainer.reference_model is not trainer.policy_model`.
- Add a test: mutate a parameter in `policy_model`, assert `reference_model` is unchanged.
