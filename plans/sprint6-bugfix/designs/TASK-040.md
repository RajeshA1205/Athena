# TASK-040: Use timezone-aware timestamps throughout

## Summary
Two remaining locations produce naive (timezone-unaware) `datetime` objects, violating the project convention of always using `datetime.now(timezone.utc)`. `core/utils.py:199` (`format_timestamp`) and `memory/operations.py:26` (`ContextItem.timestamp` default factory) both call `datetime.now()` without a timezone argument. These produce local-time timestamps that are incompatible with the UTC timestamps used everywhere else in the codebase, and will cause incorrect comparisons and sorting if the host timezone differs from UTC.

## Current State

**File:** `core/utils.py` (line 199)
```python
def format_timestamp(dt: Optional[datetime] = None, ...) -> str:
    if dt is None:
        dt = datetime.now()    # <-- naive, local time
    return dt.strftime(format_str)
```

**File:** `memory/operations.py` (line 26)
```python
@dataclass
class ContextItem:
    ...
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())  # <-- naive
```

**File:** `core/utils.py` (line 11)
```python
from datetime import datetime    # <-- no `timezone` imported
```

**File:** `memory/operations.py` — check existing imports for `timezone`.

## Proposed Change

**`core/utils.py`:**

```python
from datetime import datetime, timezone   # add timezone

def format_timestamp(dt: Optional[datetime] = None, ...) -> str:
    if dt is None:
        dt = datetime.now(timezone.utc)   # UTC-aware
    return dt.strftime(format_str)
```

**`memory/operations.py`:**

```python
# Verify timezone is already imported (it likely is, given other UTC usage)
timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
```

## Files Modified

- `core/utils.py`
  - Line 11: `from datetime import datetime` → `from datetime import datetime, timezone`
  - Line 199: `datetime.now()` → `datetime.now(timezone.utc)`
- `memory/operations.py`
  - Line 26: `datetime.now().isoformat()` → `datetime.now(timezone.utc).isoformat()`
  - Verify `timezone` is in the imports (add if missing)

## Acceptance Criteria

- [ ] `format_timestamp()` with no arguments returns a UTC-aware timestamp (string contains `+00:00`)
- [ ] `ContextItem()` with no arguments has a timezone-aware `timestamp` field
- [ ] All existing tests pass

## Edge Cases & Risks

- **`strftime` and timezone:** `datetime.now(timezone.utc).strftime(format_str)` formats identically to the naive version for most format strings. If `format_str` includes `%z` (timezone offset), it will now produce `+0000` instead of an empty string. This is an improvement.
- **Callers that pass a naive `dt`:** `format_timestamp(naive_dt)` will still work — the guard only applies when `dt is None`. Callers that pass aware datetimes are unaffected.

## Test Notes

- Add assertion: `format_timestamp()` result (when parsed back) has `tzinfo` set.
- Existing tests that check `ContextItem.timestamp` format will pass — it's still an ISO format string, just with `+00:00` suffix.
