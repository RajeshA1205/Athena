# TASK-046: Wire OLMoE encode() and action_head for GRPO Training

## Summary

OLMoEModel lacks the `encode()` method and `action_head` attribute that GRPO's `_compute_action_logprob` (grpo.py:327-330) depends on. Without them, the hasattr check at grpo.py:349 fails and all training updates produce zero gradients via the fallback `torch.tensor(0.0, requires_grad=True)`.

## Current State

**File:** `/Users/rajesh/athena/models/olmoe.py`

The `OLMoEModel.__init__` (line 71-85) does not create an `action_head` attribute:

```python
# lines 81-85
self.model = None
self.tokenizer = None
self.device = None
self._is_loaded = False
self._has_lora = False
```

The `load()` method (line 87-155) loads the HuggingFace model but never instantiates `MemoryActionHead`:

```python
# line 149
self._is_loaded = True
```

The `embed()` method (line 267-295) is async and returns `List[float]` with `.cpu().tolist()`, which detaches from the computation graph. GRPO needs a synchronous method that returns `torch.Tensor` on-device with gradients intact.

`MemoryActionHead` is defined at line 352-380 but is a standalone class never attached to `OLMoEModel`.

**Consumer in grpo.py** (lines 316-337, 349-350):

```python
def _compute_action_logprob(self, model, state, action):
    state_text = self._format_state(state)
    embedding = model.encode(state_text)       # <-- does not exist
    logits = model.action_head(embedding)       # <-- does not exist
    log_probs = F.log_softmax(logits, dim=-1)
    action_idx = _ACTION_TO_IDX.get(action, 0)
    return log_probs[action_idx]

def _get_action_logprob(self, state, action):
    ...
    if not hasattr(self.policy_model, "encode") or not hasattr(self.policy_model, "action_head"):
        return torch.tensor(0.0, requires_grad=True)  # <-- always hits this
```

GRPO's `setup()` at line 115 also calls `self.policy_model.parameters()`, and `_update_reference_model()` at line 377-380 calls `state_dict()` / `load_state_dict()`. These must include the action_head's parameters.

## Proposed Change

Modify `/Users/rajesh/athena/models/olmoe.py` only.

### 1. Add `self.action_head = None` in `__init__` (after line 85)

```python
self._has_lora = False
self.action_head = None  # MemoryActionHead, attached after load()
```

### 2. Instantiate `MemoryActionHead` in `load()` (after line 149)

```python
self._is_loaded = True

# Attach MemoryActionHead for GRPO training pipeline
if TRANSFORMERS_AVAILABLE:
    hidden_size = self.model.config.hidden_size
    self.action_head = MemoryActionHead(hidden_size)
    # Move to same device as the base model
    model_device = next(self.model.parameters()).device
    self.action_head = self.action_head.to(model_device)
```

Using `next(self.model.parameters()).device` is more reliable than `self.device` because when `device_map="auto"` is used, `self.device` is the string `"auto"`, not an actual torch device.

### 3. Add synchronous `encode()` method (after `embed()`, around line 296)

```python
def encode(self, text: str) -> "torch.Tensor":
    """
    Encode text into a 1-D embedding tensor (synchronous, on-device).

    Unlike embed() which is async and returns List[float], this method
    is synchronous and returns a torch.Tensor on the model's device
    with the computation graph intact for gradient flow.

    Args:
        text: Input text to encode.

    Returns:
        1-D tensor of shape (hidden_size,) on the model's device.

    Raises:
        RuntimeError: If the model has not been loaded.
    """
    if not self._is_loaded:
        raise RuntimeError("Model not loaded")

    inputs = self.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=self.config.max_length,
    )

    # Move inputs to model device
    model_device = next(self.model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    outputs = self.model(**inputs, output_hidden_states=True)
    # Mean-pool last hidden state over the sequence dimension, squeeze to 1-D
    embedding = outputs.hidden_states[-1].mean(dim=1).squeeze(0)

    return embedding
```

Key differences from `embed()`:
- Synchronous (not async) -- GRPO calls it in sync code paths.
- Returns `torch.Tensor` on-device, not `List[float]`.
- No `torch.no_grad()` context -- gradients must flow through for GRPO training.
- No `.cpu()` call -- tensor stays on the model device.

### 4. Add `parameters()` method (after `encode()`)

```python
def parameters(self):
    """Yield parameters from both the base model and the action head."""
    if self.model is not None:
        yield from self.model.parameters()
    if self.action_head is not None:
        yield from self.action_head.parameters()
```

### 5. Add `state_dict()` and `load_state_dict()` methods

```python
def state_dict(self):
    """Return combined state dict of base model and action head."""
    sd = {}
    if self.model is not None:
        sd["model"] = self.model.state_dict()
    if self.action_head is not None:
        sd["action_head"] = self.action_head.state_dict()
    return sd

def load_state_dict(self, state_dict):
    """Load combined state dict into base model and action head."""
    if "model" in state_dict and self.model is not None:
        self.model.load_state_dict(state_dict["model"])
    if "action_head" in state_dict and self.action_head is not None:
        self.action_head.load_state_dict(state_dict["action_head"])
```

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `/Users/rajesh/athena/models/olmoe.py` | After line 85 | Add `self.action_head = None` |
| `/Users/rajesh/athena/models/olmoe.py` | After line 149 | Instantiate `MemoryActionHead` in `load()` |
| `/Users/rajesh/athena/models/olmoe.py` | After line 295 | Add `encode()` method |
| `/Users/rajesh/athena/models/olmoe.py` | After `encode()` | Add `parameters()`, `state_dict()`, `load_state_dict()` |

## Acceptance Criteria

- `OLMoEModel().action_head` is `None` before `load()`.
- After `load()`, `action_head` is a `MemoryActionHead` instance with `hidden_dim == model.config.hidden_size`.
- `encode("some text")` returns a 1-D `torch.Tensor` of shape `(hidden_size,)` on the model's device.
- `encode()` is synchronous (not a coroutine).
- `encode()` does NOT wrap in `torch.no_grad()` -- gradients flow through.
- `encode()` raises `RuntimeError("Model not loaded")` if called before `load()`.
- `parameters()` yields parameters from both `self.model` and `self.action_head`.
- `state_dict()` and `load_state_dict()` handle both `model` and `action_head` sub-dicts.
- The existing `embed()` method is unchanged and still works.
- `hasattr(model, "encode")` and `hasattr(model, "action_head")` both return `True` after `load()`, so grpo.py:349 no longer falls back to zero tensor.
- All 171 tests pass, 6 skipped.

## Edge Cases & Risks

1. **Device mismatch**: If the base model uses `device_map="auto"` (multi-GPU sharding), `next(self.model.parameters()).device` returns the device of the first parameter, which may differ from other shards. The action_head is small (2 linear layers), so placing it on the first device is acceptable. The embedding tensor from `encode()` will be on whatever device the last hidden state lands on, which should match the action_head's device in single-GPU scenarios. For multi-GPU, an explicit `.to()` may be needed in `_compute_action_logprob` -- but multi-GPU is out of scope for now.

2. **Memory overhead**: `MemoryActionHead` is tiny (hidden_size/2 * hidden_size + hidden_size/2 * 6 parameters). For hidden_size=2048 (OLMoE), that is ~2M params (~8MB in float32). Negligible.

3. **Gradient flow through `encode()`**: The method deliberately omits `torch.no_grad()`. During GRPO's `train_step()`, this is correct -- gradients must backprop through the encoder. During `_get_reference_logprob()`, the caller already wraps in `torch.no_grad()` (grpo.py:366), so no double-counting.

4. **`_update_reference_model` uses `copy.deepcopy`**: This will now deep-copy the entire OLMoEModel including `action_head`. The `state_dict()`/`load_state_dict()` pair is used for subsequent updates (grpo.py:379-380). The nested dict structure (`{"model": ..., "action_head": ...}`) must match between policy and reference models.

5. **LoRA interaction**: If LoRA is applied via `prepare_for_training()`, `self.model` becomes a PeftModel. `self.model.config.hidden_size` should still be accessible (PeftModel delegates to the base model's config). Verify this assumption in testing.

## Test Notes

- Existing tests do not load the real OLMoE model (requires GPU + model download). Tests pass because GRPO falls back to zero tensor. After this change, the fallback still exists (grpo.py:349) for when the model is not loaded, so existing tests remain unaffected.
- A unit test for `encode()` would require mocking the HuggingFace model. Consider adding a test that creates a mock `OLMoEModel` with a small `nn.Module` as `self.model` and verifies `encode()` returns a tensor with correct shape.
- Verify `parameters()` yields from both model and action_head by counting parameter groups.
- Verify `state_dict()` / `load_state_dict()` round-trip preserves weights.
