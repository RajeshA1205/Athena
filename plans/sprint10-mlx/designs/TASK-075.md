# TASK-075: Rewrite OLMoEModel.generate() to Use mlx-lm with Async Wrapper

## Summary

Add an mlx backend branch to `OLMoEModel.generate()`. When `self._backend == "mlx"`, call the synchronous `mlx_lm.generate()` wrapped in `asyncio.to_thread()` to preserve the `async def` contract. The transformers path remains unchanged. Map `temperature` to mlx-lm's `temp` parameter.

## Current State

**File:** `models/olmoe.py`, lines 224-276 -- `generate()`:

```python
async def generate(self, prompt: str, max_new_tokens: int = 256, **kwargs) -> str:
    if not self._is_loaded:
        raise RuntimeError("Model not loaded")

    # Tokenize
    inputs = self.tokenizer(prompt, return_tensors="pt", ...)
    ...
    # Generate (torch)
    with torch.no_grad():
        outputs = self.model.generate(**inputs, **gen_kwargs)
    # Decode
    generated = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], ...)
    return generated.strip()
```

All callers use `await self.llm.generate(prompt)` (see `core/base_agent.py` `_llm_reason()`).

**mlx-lm generate API:**
```python
output: str = mlx_lm.generate(
    model, tokenizer, prompt=prompt,
    max_tokens=max_new_tokens, temp=temperature, top_p=top_p,
    verbose=False,
)
```
Returns a plain string (generated text only, no prompt prefix).

## Proposed Change

### `generate()` -- add mlx branch before existing transformers code

```python
async def generate(self, prompt: str, max_new_tokens: int = 256, **kwargs) -> str:
    """
    Generate text from prompt.

    On the mlx backend, delegates to mlx_lm.generate() via asyncio.to_thread().
    On the transformers backend, uses the existing torch generation pipeline.
    """
    if not self._is_loaded:
        raise RuntimeError("Model not loaded")

    if self._backend == "mlx":
        import asyncio
        temperature = kwargs.get("temperature", self.config.temperature)
        top_p = kwargs.get("top_p", self.config.top_p)

        def _run():
            return mlx_lm.generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_new_tokens,
                temp=temperature,
                top_p=top_p,
                verbose=False,
            )

        return await asyncio.to_thread(_run)

    # --- transformers path (unchanged) ---
    inputs = self.tokenizer(
        prompt,
        return_tensors="pt",
        ...
    )
    ...  # existing code exactly as-is
```

Key design decisions:
- `asyncio` is already in the stdlib; the local import avoids any overhead at module level.
- `_run()` is a closure capturing `self.model`, `self.tokenizer`, and the parameter values. It runs in a thread pool via `to_thread()`.
- `mlx_lm.generate` returns a string directly -- no decoding step needed.
- `top_k` is not forwarded: `mlx_lm.generate` does not accept a `top_k` parameter. This is documented in a comment.

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `models/olmoe.py` | 224-276 | Add mlx branch at top of `generate()`, before existing transformers code |

## Acceptance Criteria

- `generate()` remains `async def` and returns `str`.
- On mlx backend, `mlx_lm.generate` is called inside `asyncio.to_thread()`.
- `temperature` kwarg maps to `temp` parameter in `mlx_lm.generate`.
- `top_p` kwarg forwards directly.
- On transformers backend, existing logic is byte-for-byte unchanged.
- `RuntimeError("Model not loaded")` raised when `_is_loaded` is False, regardless of backend.
- 173 tests pass, 4 skipped.

## Edge Cases & Risks

1. **Thread safety of mlx models**: `mlx_lm.generate` is synchronous and should be safe to run in a single worker thread. However, if multiple `generate()` calls overlap, the mlx model may not be thread-safe. Risk is low in ATHENA's current single-agent-at-a-time flow, but document as a known limitation for future concurrent use.
2. **`top_k` not forwarded**: `mlx_lm.generate` does not support `top_k`. The transformers path uses `top_k=50` by default. On the mlx path, nucleus sampling (`top_p`) is the only sampling strategy. This may produce slightly different outputs. Acceptable for Sprint 10.
3. **`do_sample` not forwarded**: `mlx_lm.generate` always samples when `temp > 0`. Setting `temperature=0` produces greedy decoding. No explicit `do_sample` flag exists. Compatible with current defaults (`do_sample=True`, `temperature=0.7`).
4. **Return value**: `mlx_lm.generate` returns the generated text without the prompt prefix. The transformers path also strips the prompt via `outputs[0][inputs["input_ids"].shape[1]:]`. Both paths return only the generated portion. Consistent.
5. **Error handling**: If `mlx_lm.generate` raises inside `to_thread`, the exception propagates to the awaiter. No special handling needed -- callers already expect exceptions from `generate()`.

## Test Notes

- Existing tests mock `generate()` or skip model loading. The mlx branch will not execute in CI (`MLX_AVAILABLE=False`), so the transformers path is tested as before.
- Manual testing on Apple Silicon: `await model.generate("Hello")` should return a string without blocking the event loop.
- Run `python3 -m pytest tests/ -q` -- expect 173 passed, 4 skipped.
