# TASK-034: Bound adaptation_history and action_history

## Summary
Two unbounded lists accumulate entries indefinitely with no eviction policy. `BaseAgent.action_history` (`core/base_agent.py:102`) grows forever, but only the last 10 entries are ever read (line 193: `for a in self.action_history[-10:]`). `NestedLearning.adaptation_history` (`learning/nested_learning.py:160`) also grows forever with no cap. In a long-running agent process, these lists will slowly exhaust memory. Converting both to `collections.deque` with a `maxlen` cap is a one-line fix per site that preserves all existing access patterns (`append`, `len`, `[-10:]` slicing).

## Current State

**File:** `core/base_agent.py`
```python
# Line 102
self.action_history: List[AgentAction] = []

# Line 193 (only usage of the list contents)
for a in self.action_history[-10:]:
    ...

# Line 317 (reset)
self.action_history = []
```

**File:** `learning/nested_learning.py`
```python
# Line 160
self.adaptation_history: List[Dict[str, Any]] = []
```

## Proposed Change

**`core/base_agent.py`:**

```python
# Add to imports at top of file
from collections import deque

# Line 102 — change list to bounded deque
self.action_history: deque = deque(maxlen=100)

# Line 317 — reset: deque.clear() works, or re-initialize
self.action_history = deque(maxlen=100)
```

`maxlen=100` retains the last 100 actions (far more than the 10 that are read), providing a reasonable buffer without unbounded growth.

**`learning/nested_learning.py`:**

```python
# Add to imports at top of file
from collections import deque

# Line 160 — change list to bounded deque
self.adaptation_history: deque = deque(maxlen=1000)
```

`maxlen=1000` provides a 1000-entry rolling window of adaptation history.

## Files Modified

- `core/base_agent.py`
  - Line 5–12 (imports): add `from collections import deque`
  - Line 102: `List[AgentAction] = []` → `deque = deque(maxlen=100)`
  - Line 317: `self.action_history = []` → `self.action_history = deque(maxlen=100)`
- `learning/nested_learning.py`
  - Imports: add `from collections import deque`
  - Line 160: `List[Dict[str, Any]] = []` → `deque = deque(maxlen=1000)`

## Acceptance Criteria

- [ ] `len(agent.action_history)` never exceeds 100
- [ ] `agent.action_history[-10:]` slicing still works correctly
- [ ] `len(learner.adaptation_history)` never exceeds 1000
- [ ] `agent.reset()` correctly re-initializes `action_history` as a new bounded deque
- [ ] All existing tests pass

## Edge Cases & Risks

- **`deque` type annotation:** The `List[AgentAction]` type hint will need updating to `Deque[AgentAction]` from `collections` (or `from typing import Deque`). In Python 3.9+ `deque[AgentAction]` works directly. Use `from collections import deque` and annotate as `deque` for simplicity.
- **Negative indexing:** `deque` supports `[-1]` but **not arbitrary negative slices like `[-10:]`**. Actually, `deque[-10:]` does work in Python — `deque` supports slicing. Verify with `list(deque_obj)[-10:]` if needed, but direct slicing works.
- **`action_count` in `get_stats()`:** `len(self.action_history)` still works on `deque` — returns current length, capped at 100. Semantics change: previously returned total lifetime count, now returns current window size. This is acceptable (and arguably more useful).
- **`reset()` re-initialization:** Using `self.action_history = deque(maxlen=100)` in `reset()` creates a new deque, discarding old entries. `self.action_history.clear()` could also be used — it preserves the same deque object and clears contents, keeping `maxlen`.

## Test Notes

- Existing tests call `len(agent.action_history)` and iterate it — these work unchanged with `deque`.
- Add test: append 200 actions, assert `len(action_history) == 100`.
- Add test: call `reset()`, assert `len(action_history) == 0`.
