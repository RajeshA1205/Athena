# TASK-066: Fix _llm_reason async/sync mismatch

## Status
- **State:** Queued
- **Priority:** Major
- **Depends on:** None
- **Created:** 2026-02-26

## Objective
Fix `_llm_reason` in `core/base_agent.py` to call `await self.llm.generate(prompt)` directly instead of wrapping it in `asyncio.to_thread(...)`. Since `OLMoEModel.generate` is `async def`, wrapping it in `asyncio.to_thread` returns an unawaited coroutine object — the entire OLMoE reasoning path is silently broken.

## Context
`BaseAgent._llm_reason` contains:
```python
return await asyncio.to_thread(self.llm.generate, prompt)
```
`OLMoEModel.generate` is declared `async def`. `asyncio.to_thread` is designed for synchronous blocking callables; when passed an async function, it executes `self.llm.generate(prompt)` in a thread pool, which merely creates and immediately returns a coroutine object without ever awaiting it. The caller receives a `Coroutine` object instead of a string. CoordinatorAgent and StrategyAgent — the only two agents that use OLMoE — silently receive a coroutine object for their LLM synthesis step, meaning no actual reasoning occurs.

## Scope & Constraints
- **May modify:** `core/base_agent.py` only
- **Must NOT modify:** `models/olmoe.py` or any other file
- The fix is a one-line change: remove `asyncio.to_thread(` wrapper and add direct `await`
- If `asyncio` is no longer needed after the fix, remove the import to keep the file clean (check for other usages first)
- Do not change the `_llm_reason` method signature

## Input
- `core/base_agent.py` — `_llm_reason` method, current implementation

## Expected Output
- `core/base_agent.py` — `_llm_reason` body changed to `return await self.llm.generate(prompt)` (direct await, no `asyncio.to_thread`)

## Acceptance Criteria
- [ ] `_llm_reason` uses `return await self.llm.generate(prompt)` with no `asyncio.to_thread` wrapper
- [ ] `asyncio` import removed if it has no other usages in `base_agent.py`; retained if used elsewhere
- [ ] `python3 -m pytest tests/ -q` passes with 173 passed, 4 skipped (no regressions)
- [ ] `python3 main.py --mode dry-run` completes without error

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
