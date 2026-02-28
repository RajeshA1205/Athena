# TASK-076: Fix OLMoEModel.encode() to bridge mlx arrays to torch tensors

## Status
- **State:** Queued
- **Priority:** 🔴 Critical
- **Depends on:** TASK-074
- **Created:** 2026-02-26

## Objective
Rewrite `OLMoEModel.encode()` in `models/olmoe.py` so that it:
1. Produces a valid `torch.Tensor` on both backends (mlx and transformers)
2. On the mlx backend: tokenize with the mlx tokenizer, run a forward pass to get hidden states, mean-pool to a 1-D mlx array, then convert to a `torch.Tensor` via numpy
3. On the transformers backend: existing logic unchanged

The GRPO training pipeline (`training/stage2_agemem/trainer.py`) calls `model.encode(text)` and feeds the result directly into `model.action_head(embedding)` which is a `torch.nn.Module`. This means `encode()` MUST return a `torch.Tensor` regardless of which inference backend is active.

## Context
Current `encode()` (lines 308–345 of `/Users/rajesh/athena/models/olmoe.py`):
- Synchronous method
- Runs a forward pass on `self.model` with `output_hidden_states=True`
- Mean-pools last hidden state to 1-D
- Returns `torch.Tensor` on the model device
- Gradients flow through (no `torch.no_grad()`) for GRPO training

On the mlx backend after TASK-074:
- `self.model` is an mlx model (not a PyTorch nn.Module)
- mlx models return mlx arrays, not torch tensors
- `self.action_head` is None (not attached on mlx path — GRPO training requires the transformers path)
- However, `encode()` must still be callable and must still return a `torch.Tensor` so that any code path that calls `encode()` does not crash

Important constraint: GRPO training (which needs gradients through encode) only works on the transformers backend where `action_head` is a real `torch.nn.Module`. On the mlx backend, `encode()` is used for inference-time embedding only (no gradient requirement). Therefore, when backend is mlx, the conversion can use `torch.from_numpy(numpy_array)` (a no-grad path is acceptable).

mlx-lm does not expose a direct "get hidden states" API identical to transformers. Acceptable fallback for the mlx encode path:
- Tokenize with `self.tokenizer.encode(text)` → list of ints
- Run `self.model(mlx.array([token_ids]))` to get logits (mlx models return logits by default)
- Use the logit matrix as a proxy embedding: mean-pool over the sequence dimension
- Convert via `numpy`: `embedding_np = np.array(mlx_output.mean(axis=1).squeeze())` then `torch.from_numpy(embedding_np).float()`
- This is dimensionally different from hidden_size but functionally adequate for inference-only use; document the limitation in a comment

See `/Users/rajesh/athena/models/olmoe.py` and `/Users/rajesh/athena/training/stage2_agemem/trainer.py` for context.

## Scope & Constraints
- May modify: `models/olmoe.py` — `OLMoEModel.encode()` method only
- Must NOT modify: `MemoryActionHead`, `generate()`, `load()`, GRPO trainer
- `encode()` must remain synchronous (callers in GRPO are synchronous)
- Return type must be `torch.Tensor` on both paths
- On the mlx path the returned tensor may be on CPU and detached (no gradient requirement)
- On the transformers path the existing gradient-capable behaviour must be preserved exactly
- Must import `numpy as np` inside the mlx branch (not at module level) to avoid a hard numpy dependency when mlx is not in use; or use a guarded top-level import

## Input
- `/Users/rajesh/athena/models/olmoe.py` (after TASK-074 changes)
- `/Users/rajesh/athena/training/stage2_agemem/trainer.py` — shows how `model.encode()` output is used

## Expected Output
Modified `OLMoEModel.encode()` with structure:
```python
def encode(self, text: str) -> "torch.Tensor":
    if not self._is_loaded:
        raise RuntimeError("Model not loaded")

    if self._backend == "mlx":
        import mlx.core as mx
        import numpy as np
        token_ids = self.tokenizer.encode(text)
        inputs = mx.array([token_ids])
        # mlx model forward: returns logits of shape (1, seq_len, vocab_size)
        logits = self.model(inputs)
        # mean-pool over sequence dimension → (vocab_size,) as proxy embedding
        embedding_np = np.array(logits[0].mean(axis=0))
        # NOTE: mlx path returns logit-space embedding (vocab_size dim), not hidden_size.
        # This is inference-only; GRPO training requires the transformers backend.
        return torch.from_numpy(embedding_np).float()

    # --- transformers path (unchanged) ---
    ...existing code...
```

## Acceptance Criteria
- [ ] `encode()` returns a `torch.Tensor` on both backends
- [ ] On the mlx backend, the tensor is on CPU and has `requires_grad=False`
- [ ] On the transformers backend, the existing behaviour (gradient-capable, on model device) is unchanged
- [ ] `encode()` raises `RuntimeError("Model not loaded")` when `self._is_loaded` is False
- [ ] A comment in the mlx path notes the vocab-size dimensionality limitation and that GRPO requires the transformers backend
- [ ] `python3 -m pytest tests/ -q` shows 173 passed, 4 skipped

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
