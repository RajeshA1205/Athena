# TASK-053: Fix Missing await on Async get_stats() in test_memory.py

## Summary

`AgeMem.get_stats()` is `async def` (memory/agemem.py:441), but `test_memory.py` calls it without `await` at lines 99 and 108. The unawaited coroutine silently passes because both calls are inside try/except blocks that skip on any exception, and the `isinstance(stats, dict)` check happens to not crash because a coroutine object is truthy (but is NOT a dict).

## Current State

**File:** `/Users/rajesh/athena/tests/test_memory.py`

### Line 99 (inside `test_add_multiple_entries`, which IS async):

```python
@pytest.mark.asyncio
async def test_add_multiple_entries(self):
    try:
        from memory.agemem import AgeMem
        mem = AgeMem()
        for i in range(5):
            await mem.add(_make_memory_entry(f"entry {i}", f"agent_{i}"))
        stats = mem.get_stats() if hasattr(mem, "get_stats") else {}
        assert isinstance(stats, dict)
    except Exception as e:
        pytest.skip(f"AgeMem not available: {e}")
```

Problem: `mem.get_stats()` returns a coroutine object (not a dict). The `isinstance(stats, dict)` assertion would fail, but it never runs because the unawaited coroutine (or any other issue) triggers the `except Exception` which calls `pytest.skip`. Additionally, unawaited coroutines generate `RuntimeWarning: coroutine 'AgeMem.get_stats' was never awaited`.

### Lines 104-111 (`test_get_stats`, which is SYNC):

```python
def test_get_stats(self):
    try:
        from memory.agemem import AgeMem
        mem = AgeMem()
        stats = mem.get_stats()
        assert isinstance(stats, dict)
    except Exception as e:
        pytest.skip(f"AgeMem not available: {e}")
```

Problem: Same issue -- `mem.get_stats()` returns a coroutine. Additionally, the test is synchronous (`def` not `async def`), so it cannot use `await` without being converted.

### AgeMem.get_stats() signature (memory/agemem.py:441):

```python
async def get_stats(self) -> Dict[str, Any]:
```

## Proposed Change

Modify `/Users/rajesh/athena/tests/test_memory.py` only.

### Fix line 99: Add `await` to get_stats() call

```python
stats = (await mem.get_stats()) if hasattr(mem, "get_stats") else {}
```

The parentheses are needed because `await` has lower precedence than the ternary operator. Without them, Python would parse it as `await (mem.get_stats() if hasattr(...) else {})`, which would attempt to await `{}` on the else branch.

### Fix lines 104-111: Convert `test_get_stats` to async

```python
@pytest.mark.asyncio
async def test_get_stats(self):
    try:
        from memory.agemem import AgeMem
        mem = AgeMem()
        stats = await mem.get_stats()
        assert isinstance(stats, dict)
    except Exception as e:
        pytest.skip(f"AgeMem not available: {e}")
```

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `/Users/rajesh/athena/tests/test_memory.py` | 99 | Add `await` with parentheses |
| `/Users/rajesh/athena/tests/test_memory.py` | 104 | Add `@pytest.mark.asyncio` decorator |
| `/Users/rajesh/athena/tests/test_memory.py` | 105 | Change `def test_get_stats` to `async def test_get_stats` |
| `/Users/rajesh/athena/tests/test_memory.py` | 108 | Add `await` before `mem.get_stats()` |

## Acceptance Criteria

- Both `get_stats()` calls are properly awaited.
- `test_get_stats` is an `async def` decorated with `@pytest.mark.asyncio`.
- No `RuntimeWarning: coroutine ... was never awaited` warnings from these tests.
- Both tests still skip gracefully when AgeMem cannot initialize (no Neo4j backend).
- The `isinstance(stats, dict)` assertion would actually validate a dict (not a coroutine object) if the tests ever run past the try/except.
- All 171 tests pass, 6 skipped (these two tests are likely among the 6 skipped).

## Edge Cases & Risks

1. **Precedence of `await` with ternary**: The expression `await mem.get_stats() if hasattr(mem, "get_stats") else {}` without parentheses would be parsed as `await (mem.get_stats() if ... else {})`. If `hasattr` is False, Python would `await {}` which raises `TypeError`. The fix uses `(await mem.get_stats()) if hasattr(mem, "get_stats") else {}` which correctly awaits only the coroutine.

2. **Test still skips**: These tests are wrapped in try/except that calls `pytest.skip()`. Even with the await fix, they will likely skip if the AgeMem backend (Neo4j/Graphiti) is not available. The fix ensures correct behavior when the backend IS available and eliminates the coroutine warning.

3. **pytest-asyncio version**: The existing test file already uses `@pytest.mark.asyncio` for other tests (lines 70, 82, 92), so the pytest-asyncio plugin is already configured. Adding another `@pytest.mark.asyncio` to `test_get_stats` follows the established pattern.

4. **Mock-based tests unaffected**: The `TestMockMemoryLayer` class (lines 118-150) uses `MagicMock` for `get_stats` (not `AsyncMock`), which is intentional -- the mock tests validate the sync interface contract. The mock's `get_stats` is defined as `MagicMock(return_value={"total_memories": 1})` at line 126, which is sync and correct for the mock test class.

## Test Notes

- Run `python3 -m pytest tests/test_memory.py -q -W error::RuntimeWarning` to verify no unawaited coroutine warnings.
- Run `python3 -m pytest tests/ -q` to verify all 171 pass, 6 skipped.
- The fix is trivially correct -- adding `await` to an async function call and converting a sync test to async.
