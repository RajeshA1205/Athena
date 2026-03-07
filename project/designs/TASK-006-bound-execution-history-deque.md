# TASK-006: Bound WorkflowDiscovery.execution_history to a deque

## Problem

`WorkflowDiscovery.execution_history` (line 102 in `evolution/workflow_discovery.py`) is declared as `List[Dict[str, Any]]` and grows unbounded via `self.execution_history.append(execution_trace)` on line 127. In long-running sessions (paper-trade mode), this is a memory leak proportional to the number of completed execution traces.

## Files to change

| File | Lines | Change |
|------|-------|--------|
| `evolution/workflow_discovery.py` | 8 | Add `from collections import deque` |
| `evolution/workflow_discovery.py` | 102 | Change `self.execution_history: List[...]` to `self.execution_history: deque = deque(maxlen=N)` |
| `evolution/workflow_discovery.py` | 413 | `get_stats()` uses `len(self.execution_history)` -- no change needed, `deque` supports `len()` |

## Approach

1. Import `deque` from `collections`.
2. Add a `max_history` config key (default 1000) read from the config dict in `__init__`.
3. Replace `self.execution_history: List[Dict[str, Any]] = []` with `self.execution_history: deque = deque(maxlen=self.max_history)`.
4. Update the type annotation in the class docstring / type hints. The `Deque` type from `typing` should be used: `Deque[Dict[str, Any]]`.
5. Verify that `get_stats()` (line 413) which calls `len(self.execution_history)` still works (it does -- `deque` supports `len()`).

## Edge cases / risks

- `deque(maxlen=N).append()` silently drops the oldest item when full. This is the desired behavior for a sliding window, but any code that assumes the full history is available will silently get truncated data. Audit callers -- only `get_stats()` references `execution_history`, and it only calls `len()`, so this is safe.
- Serialization: if `save_library()` is extended to persist execution history, `deque` needs `list()` conversion. Not currently an issue since `save_library` only serializes `workflow_library`.
- The `max_history` default of 1000 is generous. At ~1KB per trace, this caps memory at ~1MB.

## Acceptance criteria

- [ ] `execution_history` is a `deque` with configurable `maxlen`.
- [ ] `get_stats()["executions_analyzed"]` returns accurate count (capped at `maxlen`).
- [ ] Existing tests pass without modification.
- [ ] `pytest tests/ -q` remains green.
