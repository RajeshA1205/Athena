# TASK-062: Fix Timezone-Naive datetime.now() in ExecutionAgent

## Summary

`agents/execution_agent.py` uses `datetime.now()` (timezone-naive) in four places: `Order.__post_init__` (lines 43-45), `act()` (line 243), `_simulate_fill()` (line 487), and `cancel_order()` (line 571). The rest of the codebase — including `trading/order_management.py` — uses `datetime.now(timezone.utc)`. Mixed naive/aware datetimes cause `TypeError` on comparison and produce incorrect timestamp ordering in fills and order records.

## Current State

**File:** `agents/execution_agent.py`

**Line 10 — import (no `timezone`):**
```python
from datetime import datetime
```

**Lines 43-45 — Order.__post_init__:**
```python
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
```

**Line 243 — act():**
```python
            order.updated_at = datetime.now()
```

**Line 487 — _simulate_fill():**
```python
            "timestamp": datetime.now().isoformat(),
```

**Line 571 — cancel_order():**
```python
        order.updated_at = datetime.now()
```

## Proposed Change

### 1. Update import (line 10)

```python
from datetime import datetime, timezone
```

### 2. Order.__post_init__ (lines 43-45)

```python
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = datetime.now(timezone.utc)
```

### 3. act() (line 243)

```python
            order.updated_at = datetime.now(timezone.utc)
```

### 4. _simulate_fill() (line 487)

```python
            "timestamp": datetime.now(timezone.utc).isoformat(),
```

### 5. cancel_order() (line 571)

```python
        order.updated_at = datetime.now(timezone.utc)
```

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `agents/execution_agent.py` | 10 | Add `timezone` to datetime import |
| `agents/execution_agent.py` | 43, 45 | `datetime.now()` → `datetime.now(timezone.utc)` |
| `agents/execution_agent.py` | 243 | `datetime.now()` → `datetime.now(timezone.utc)` |
| `agents/execution_agent.py` | 487 | `datetime.now().isoformat()` → `datetime.now(timezone.utc).isoformat()` |
| `agents/execution_agent.py` | 571 | `datetime.now()` → `datetime.now(timezone.utc)` |

## Acceptance Criteria

- Zero `datetime.now()` calls remain in `execution_agent.py`.
- All datetime values in `Order` are UTC-aware (`tzinfo=timezone.utc`).
- `order.created_at.isoformat()` produces a string ending in `+00:00`.
- No `TypeError` when comparing timestamps between `execution_agent.py` and `trading/order_management.py`.
- 173 tests pass, 4 skipped.

## Edge Cases & Risks

1. **`Order.created_at` type**: Currently `Optional[datetime]`. After this fix it is still `Optional[datetime]`, but the datetime object is now UTC-aware. The `_order_to_dict` serialisation at line 551 (`order.created_at.isoformat()`) will now include the `+00:00` suffix, which is correct and more informative.
2. **`Order.updated_at` type**: Same as above.
3. **TASK-065 dependency**: TASK-065 removes this local `Order` dataclass entirely. This fix is needed first so the timestamps are correct before the dataclass is replaced.

## Test Notes

- Existing tests instantiate `Order` objects indirectly via `_create_order`. After this fix, `order.created_at` will be a UTC-aware datetime rather than a naive one — verify no test asserts on the exact value or tzinfo absence.
- Run `python3 -m pytest tests/ -q` — expect 173 passed, 4 skipped.
