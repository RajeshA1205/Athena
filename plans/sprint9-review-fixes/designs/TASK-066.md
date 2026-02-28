# TASK-066: Fix _llm_reason Async/Sync Mismatch

## Summary

`BaseAgent._llm_reason()` (line 183) uses `asyncio.to_thread(self.llm.generate, prompt)` to call OLMoE. But `OLMoEModel.generate` is `async def` — passing an async function to `asyncio.to_thread` runs it in a thread pool where it returns a coroutine object immediately without executing it. That coroutine is then awaited by `_llm_reason`, returning the coroutine object itself as the "result". Both `CoordinatorAgent` and `StrategyAgent` silently receive a coroutine object instead of a string, making all LLM-assisted reasoning return garbage. Fix: remove `asyncio.to_thread` and `await self.llm.generate(prompt)` directly.

## Current State

**File:** `core/base_agent.py`, lines 178-186:

```python
    async def _llm_reason(self, prompt: str) -> Optional[str]:
        """Route a reasoning prompt through OLMoE if available, else return None."""
        if self.llm is None:
            return None
        try:
            return await asyncio.to_thread(self.llm.generate, prompt)
        except Exception as e:
            self.logger.warning("LLM reasoning failed: %s", e)
            return None
```

**File:** `models/olmoe.py` — `generate` signature:

```python
    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> str:
```

`generate` is `async def`, confirming it must be awaited directly, not dispatched to a thread pool.

## Proposed Change

**File:** `core/base_agent.py`, lines 183:

```python
    async def _llm_reason(self, prompt: str) -> Optional[str]:
        """Route a reasoning prompt through OLMoE if available, else return None."""
        if self.llm is None:
            return None
        try:
            return await self.llm.generate(prompt)
        except Exception as e:
            self.logger.warning("LLM reasoning failed: %s", e)
            return None
```

Replace `await asyncio.to_thread(self.llm.generate, prompt)` with `await self.llm.generate(prompt)`.

## Files Modified

| File | Line | Change |
|------|------|--------|
| `core/base_agent.py` | 183 | `await asyncio.to_thread(self.llm.generate, prompt)` → `await self.llm.generate(prompt)` |

## Acceptance Criteria

- `_llm_reason` no longer uses `asyncio.to_thread`.
- When `self.llm` is a loaded `OLMoEModel`, `_llm_reason(prompt)` returns a `str` (the generated text), not a coroutine object.
- When `self.llm` is None, `_llm_reason` returns None (unchanged).
- When `generate` raises an exception, `_llm_reason` logs the warning and returns None (unchanged).
- `asyncio` import in `base_agent.py` is still used elsewhere — do NOT remove it.
- 173 tests pass, 4 skipped.

## Edge Cases & Risks

1. **`asyncio` still needed**: `base_agent.py` imports `asyncio` and uses it in `reset()` and elsewhere. Only remove the `asyncio.to_thread` call, not the import.
2. **Model not loaded**: `OLMoEModel.generate` raises `RuntimeError("Model not loaded")` if called before `load()`. This is caught by the `except Exception` and results in `None` being returned — correct behaviour.
3. **Thread safety**: `asyncio.to_thread` was not needed in the first place (generate is async). Removing it has no thread-safety implications.
4. **One-line change, zero risk**: This is a pure correctness fix with no logic change.

## Test Notes

- Without a real OLMoE model, the code path through `generate` raises `RuntimeError("Model not loaded")` and `_llm_reason` returns `None` — same observable behaviour as before the fix from the test perspective.
- To verify the fix manually: pass a mock `llm` where `generate` is an `async def` returning a string and assert `_llm_reason` returns that string (not a coroutine).
- Run `python3 -m pytest tests/ -q` — expect 173 passed, 4 skipped.
