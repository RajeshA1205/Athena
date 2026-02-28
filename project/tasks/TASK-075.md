# TASK-075: Rewrite OLMoEModel.generate() to use mlx-lm with async wrapper

## Status
- **State:** Queued
- **Priority:** 🔴 Critical
- **Depends on:** TASK-074
- **Created:** 2026-02-26

## Objective
Rewrite `OLMoEModel.generate()` in `models/olmoe.py` to:
1. Detect which backend is active (`self._backend`)
2. When `_backend == "mlx"`: call `mlx_lm.generate()` (which is synchronous) wrapped in `asyncio.to_thread()` so the method remains `async def` and does not block the event loop
3. When `_backend == "transformers"`: keep the existing torch-based generation logic unchanged
4. Raise `RuntimeError("Model not loaded")` if `self._is_loaded` is False (unchanged)

## Context
The current `generate()` signature (lines 224–276 of `/Users/rajesh/athena/models/olmoe.py`):
```python
async def generate(self, prompt: str, max_new_tokens: int = 256, **kwargs) -> str:
```
All callers use `await self.llm.generate(prompt)` — so the method MUST remain `async def`. The mlx-lm `generate()` is a synchronous call, so wrapping it in `asyncio.to_thread()` is the correct pattern.

mlx-lm generate API:
```python
# mlx_lm.generate is synchronous
output: str = mlx_lm.generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=max_new_tokens,
    temp=temperature,
    top_p=top_p,
    verbose=False,
)
```
Note: mlx_lm.generate returns a plain string (the generated text, not including the prompt).

See `/Users/rajesh/athena/models/olmoe.py` for full current code.
See `/Users/rajesh/athena/core/base_agent.py` `_llm_reason()` for the call site pattern.

## Scope & Constraints
- May modify: `models/olmoe.py` — `OLMoEModel.generate()` method only
- Must NOT modify the method signature: must remain `async def generate(self, prompt, max_new_tokens=256, **kwargs) -> str`
- Must NOT change the transformers-path logic (only the mlx branch is new code)
- `asyncio` is already imported via the stdlib; no new imports needed beyond what TASK-074 adds
- The mlx-lm generate call must run in `asyncio.to_thread()` to avoid blocking the async event loop
- `temperature`, `top_p`, `top_k` kwargs must be forwarded to mlx_lm.generate using the correct mlx_lm parameter names (`temp` for temperature)

## Input
- `/Users/rajesh/athena/models/olmoe.py` (after TASK-074 changes)
- mlx-lm generate API: `mlx_lm.generate(model, tokenizer, prompt, max_tokens, temp, top_p, verbose=False) -> str`

## Expected Output
Modified `OLMoEModel.generate()` with structure:
```python
async def generate(self, prompt: str, max_new_tokens: int = 256, **kwargs) -> str:
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
    ...existing code...
```

## Acceptance Criteria
- [ ] `generate()` is `async def` and returns `str`
- [ ] On mlx backend, `mlx_lm.generate` is called inside `asyncio.to_thread()` (not called directly at await site)
- [ ] `temperature` and `top_p` kwargs are forwarded to mlx_lm.generate as `temp` and `top_p`
- [ ] On transformers backend, the existing generation logic is unchanged
- [ ] `generate()` raises `RuntimeError("Model not loaded")` when `self._is_loaded` is False regardless of backend
- [ ] `python3 -m pytest tests/ -q` shows 173 passed, 4 skipped

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
