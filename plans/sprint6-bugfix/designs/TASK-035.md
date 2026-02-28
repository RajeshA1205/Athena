# TASK-035: Replace blocking I/O in async methods with asyncio.to_thread

## Summary
`NestedLearning.save_state()` and `RepExp.save()` are declared `async def` but use synchronous `open()` + `json.dump()` for file I/O. When these methods are awaited, the blocking I/O runs on the event loop thread, stalling all other coroutines for the duration of the write. The same issue applies to the `load_state()` and `load()` counterparts. The fix is to wrap the synchronous file operations in `asyncio.to_thread()`, which offloads them to a thread pool without changing the calling interface.

## Current State

**File:** `learning/nested_learning.py` (lines 424–425)

```python
async def save_state(self, path: str) -> None:
    ...
    with open(path, "w") as f:           # blocking — line 424
        json.dump(data, f, indent=2)     # blocking — line 425
```

**File:** `learning/nested_learning.py` (lines ~437–445, `load_state`)

```python
async def load_state(self, path: str) -> None:
    ...
    with open(path, "r") as f:           # blocking
        data = json.load(f)
```

**File:** `learning/repexp.py` (lines 333–334)

```python
async def save(self, path: str) -> None:
    ...
    with open(path, "w") as f:           # blocking — line 333
        json.dump(data, f, indent=2)     # blocking — line 334
```

**File:** `learning/repexp.py` (lines 347–348, `load`)

```python
async def load(self, path: str) -> None:
    ...
    with open(path, "r") as f:           # blocking
        data = json.load(f)
```

## Proposed Change

Extract the I/O into a private sync helper and call it via `asyncio.to_thread`:

**`learning/nested_learning.py`:**

```python
async def save_state(self, path: str) -> None:
    ...
    def _write() -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    await asyncio.to_thread(_write)
    self.logger.info("NestedLearning state saved to %s", path)

async def load_state(self, path: str) -> None:
    ...
    def _read() -> dict:
        with open(path, "r") as f:
            return json.load(f)
    data = await asyncio.to_thread(_read)
    # ... rest of the method uses `data`
```

**`learning/repexp.py`:** Same pattern for `save()` and `load()`.

`asyncio` is already imported in both files (used for `asyncio.sleep` etc.), so no new import is needed.

## Files Modified

- `learning/nested_learning.py`
  - `save_state()`: wrap `open/json.dump` in `asyncio.to_thread`
  - `load_state()`: wrap `open/json.load` in `asyncio.to_thread`
- `learning/repexp.py`
  - `save()`: wrap `open/json.dump` in `asyncio.to_thread`
  - `load()`: wrap `open/json.load` in `asyncio.to_thread`

## Acceptance Criteria

- [ ] No synchronous `open()` calls remain inside `async def` methods in these two files
- [ ] `save_state()` / `save()` still write valid JSON files
- [ ] `load_state()` / `load()` still restore state correctly
- [ ] All existing tests pass (tests call `await save_state(...)` / `await load(...)`)

## Edge Cases & Risks

- **`asyncio.to_thread` requires Python 3.9+.** The project already uses Python 3.10+ per `context.md`, so this is safe.
- **Exception handling:** The existing `try/except` block wraps the I/O. When using `to_thread`, the exception is raised in the awaiting coroutine, so the existing `except` clauses continue to work unchanged.
- **Thread safety of `data` dict:** The `data` dict is constructed before `to_thread` is called and is not modified concurrently. Safe.
- **Nested function capturing variables:** The inner `_write()` closure captures `data` and `path` by closure. This is correct Python — they are read-only within `_write`.

## Test Notes

- `test_learning.py::TestNestedLearning::test_save_and_load_state` already tests the round-trip. It will pass unchanged since `asyncio.to_thread` is transparent to the caller.
- No new tests needed; the existing round-trip tests provide sufficient coverage.
