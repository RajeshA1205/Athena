# TASK-011: Fix CLI _memory_initialized silently set on init failure

## Problem

In `cli.py` lines 221-222, `_ensure_memory()` calls `ok = await self.memory.initialize()` and then unconditionally sets `self._memory_initialized = True` regardless of whether `ok` is `True` or `False`. This means if memory initialization fails (e.g., Neo4j is down), the flag is still set to `True`, and all subsequent calls to `_ensure_memory()` skip re-initialization. The agent pipeline then runs with a broken memory backend, silently producing degraded results (no memory retrieval, failed memory stores).

## Files to change

| File | Lines | Change |
|------|-------|--------|
| `cli.py` | 220-227 | Gate `self._memory_initialized = True` on `ok` being truthy |

## Approach

1. Change `_ensure_memory()` to gate the flag:
   ```python
   async def _ensure_memory(self):
       if not self._memory_initialized:
           ok = await self.memory.initialize()
           if ok:
               self._memory_initialized = True
               if self.verbose:
                   stats = await self.memory.get_stats()
                   using = "Graphiti/Neo4j" if stats["backend"].get("using_graphiti") else "in-memory fallback"
                   episodes = stats["backend"].get("episode_count", 0)
                   print(f"  {DIM}Memory: {using} ({episodes} episodes stored){RESET}")
           else:
               # Log warning but don't set flag -- will retry on next call
               print(f"  {YELLOW}Warning: Memory initialization failed. Will retry on next query.{RESET}")
   ```

2. Consider adding a retry limit to prevent infinite retry loops:
   ```python
   _memory_init_attempts: int = 0
   MAX_MEMORY_INIT_ATTEMPTS: int = 3
   ```
   After 3 failed attempts, set the flag and log that memory is running in degraded mode.

3. Add a test: mock `self.memory.initialize()` returning `False`, verify `_memory_initialized` stays `False` after first call, then verify the next `_ensure_memory()` call attempts initialization again.

## Edge cases / risks

- **Retry storm**: Without an attempt limit, every user query triggers a re-initialization attempt against a down Neo4j. The retry limit (3 attempts) prevents this. After exhaustion, set `_memory_initialized = True` to stop retrying, and log a clear message.
- **`initialize()` raising an exception** vs returning `False`: Currently the code does not wrap `initialize()` in try/except. If it raises, the exception propagates to `analyze_symbol()`. Add a try/except around the `initialize()` call for safety.
- **Thread safety**: `cli.py` is single-threaded (asyncio), so no race condition on `_memory_initialized`.

## Acceptance criteria

- [ ] When `memory.initialize()` returns `False`, `_memory_initialized` remains `False`.
- [ ] On next `_ensure_memory()` call, initialization is reattempted.
- [ ] After `MAX_MEMORY_INIT_ATTEMPTS` failures, retries stop and a warning is logged.
- [ ] When `memory.initialize()` raises an exception, it is caught and logged (not propagated).
- [ ] When `memory.initialize()` returns `True`, behavior is unchanged.
- [ ] `pytest tests/ -q` remains green.
