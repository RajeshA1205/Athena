# TASK-076: Fix OLMoEModel.encode() to Bridge mlx Arrays to torch Tensors

## Summary

Add an mlx backend branch to `OLMoEModel.encode()` that tokenizes with the mlx tokenizer, runs a forward pass to get logits, mean-pools over the sequence dimension, and converts the result to a `torch.Tensor` via numpy. The transformers path (gradient-capable, hidden-state embedding) remains unchanged. The mlx path returns a CPU tensor with no gradient -- GRPO training requires the transformers backend.

## Current State

**File:** `models/olmoe.py`, lines 308-345 -- `encode()`:

```python
def encode(self, text: str) -> "torch.Tensor":
    if not self._is_loaded:
        raise RuntimeError("Model not loaded")

    inputs = self.tokenizer(text, return_tensors="pt", ...)
    model_device = next(self.model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    outputs = self.model(**inputs, output_hidden_states=True)
    embedding = outputs.hidden_states[-1].mean(dim=1).squeeze(0)
    return embedding  # shape (hidden_size,), on model device, grad-capable
```

**Callers:**
- `training/stage2_agemem/grpo.py` `_compute_action_logprob()` (line 340): `embedding = model.encode(state_text)` then `logits = model.action_head(embedding)`.
- `training/stage2_agemem/trainer.py` `_select_operation_unified()` (line 431): same pattern.

Both callers expect `torch.Tensor`. The GRPO path also requires `action_head` (which is `None` on mlx).

## Proposed Change

### `encode()` -- add mlx branch before existing transformers code

```python
def encode(self, text: str) -> "torch.Tensor":
    """
    Encode text into a 1-D embedding tensor (synchronous, on-device).

    On the transformers backend: returns a hidden-state embedding of shape
    (hidden_size,) with the computation graph intact for gradient flow.

    On the mlx backend: tokenizes, runs a forward pass to get logits,
    mean-pools to a 1-D tensor of shape (vocab_size,), and converts
    to a torch.Tensor via numpy. No gradient flow -- this is an
    inference-only path. GRPO training requires the transformers backend.

    Args:
        text: Input text to encode.

    Returns:
        1-D torch.Tensor.

    Raises:
        RuntimeError: If the model has not been loaded.
    """
    if not self._is_loaded:
        raise RuntimeError("Model not loaded")

    if self._backend == "mlx":
        import mlx.core as mx
        import numpy as np

        token_ids = self.tokenizer.encode(text)
        inputs = mx.array([token_ids])

        # mlx model forward: returns logits of shape (1, seq_len, vocab_size)
        logits = self.model(inputs)

        # Mean-pool over sequence dimension -> (vocab_size,) as proxy embedding.
        # NOTE: This returns a logit-space embedding (vocab_size dim), NOT a
        # hidden_size embedding. The dimensionality differs from the transformers
        # path. This is acceptable for inference-only use (e.g., similarity
        # comparisons). GRPO training, which feeds encode() output into
        # action_head (expecting hidden_size), MUST use the transformers backend.
        embedding_np = np.array(logits[0].mean(axis=0))
        return torch.from_numpy(embedding_np).float()

    # --- transformers path (unchanged) ---
    inputs = self.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=self.config.max_length,
    )
    model_device = next(self.model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    outputs = self.model(**inputs, output_hidden_states=True)
    embedding = outputs.hidden_states[-1].mean(dim=1).squeeze(0)
    return embedding
```

Key design decisions:
- `import mlx.core as mx` and `import numpy as np` are local imports inside the mlx branch. This avoids hard dependencies when mlx is not installed.
- `torch` is already imported at module level (guarded by `TRANSFORMERS_AVAILABLE`). Since we need `torch.from_numpy`, and torch is a hard dependency of the GRPO pipeline regardless, this is safe. If torch is not available, `encode()` on the mlx path would fail at `torch.from_numpy` -- but this is acceptable because any code calling `encode()` to get a `torch.Tensor` implicitly requires torch.
- `logits[0].mean(axis=0)` pools over sequence length. Shape: `(vocab_size,)`. This is dimensionally different from `hidden_size` on the transformers path, but TASK-077 adds a guard so `action_head` is never called with this tensor.
- The mlx tensor is converted via numpy to avoid any direct mlx-to-torch dependency.

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `models/olmoe.py` | 308-345 | Add mlx branch at top of `encode()`, before existing transformers code |

## Acceptance Criteria

- `encode()` returns `torch.Tensor` on both backends.
- On mlx backend: tensor is on CPU, `requires_grad=False`, shape `(vocab_size,)`.
- On transformers backend: existing behaviour unchanged (on model device, grad-capable, shape `(hidden_size,)`).
- `RuntimeError("Model not loaded")` raised when `_is_loaded` is False.
- Comment in mlx path documents the vocab-size dimensionality limitation and that GRPO requires the transformers backend.
- 173 tests pass, 4 skipped.

## Edge Cases & Risks

1. **Dimensionality mismatch**: The mlx path returns `(vocab_size,)` while the transformers path returns `(hidden_size,)`. For OLMoE-1B, `vocab_size=50280` and `hidden_size=2048`. Any code that assumes a specific embedding size will break on the mlx path. Mitigation: TASK-077 adds a `get_action_logits()` guard that raises a clear error if `action_head` is called on the mlx path.
2. **numpy dependency**: `np.array(mlx_tensor)` requires numpy. numpy is a transitive dependency of both torch and mlx, so it is always available when either backend is loaded. No risk.
3. **torch not available**: If torch is not installed, `torch.from_numpy` will `NameError`. This is acceptable -- `encode()` returning `torch.Tensor` implies torch is required. The existing module-level guard already sets `TRANSFORMERS_AVAILABLE=False` when torch is missing.
4. **Empty input text**: `self.tokenizer.encode("")` may return an empty list or a list with a single BOS token. `mx.array([[]])` would have shape `(1, 0, vocab_size)`, and `mean(axis=0)` on an empty sequence would produce NaN. Mitigation: the tokenizer typically adds at least one special token, so this is unlikely. If it occurs, the NaN tensor will propagate and surface as an obvious bug. No guard added -- same risk exists on the transformers path.
5. **Performance**: The mlx forward pass computes full logits (vocabulary projection) rather than returning hidden states. This is slightly more compute than needed for a pure embedding. Acceptable for Sprint 10 -- hidden state access can be added if mlx-lm exposes it in a future version.

## Test Notes

- Existing tests do not call `encode()` with a real loaded model. The mlx branch will not execute in CI.
- Manual testing on Apple Silicon: `model.encode("Hello")` should return a `torch.Tensor` of shape `(vocab_size,)` on CPU.
- Run `python3 -m pytest tests/ -q` -- expect 173 passed, 4 skipped.
