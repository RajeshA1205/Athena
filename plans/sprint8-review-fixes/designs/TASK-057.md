# TASK-057: Replace `hash()` with stable hash in trainer delete path

## Summary

`_execute_operation()` in the trainer uses `str(hash(content))` to generate a delete ID when exercising the `"delete"` operation. Python's `hash()` is non-deterministic across processes when `PYTHONHASHSEED` is set (the default since Python 3.3). This means delete operations target different IDs in different runs, making training non-reproducible. TASK-054 fixed the same class of bug in the scrapers but missed this instance.

## Current State

**File:** `/Users/rajesh/athena/training/stage2_agemem/trainer.py`

Line 532 in `_execute_operation()`:

```python
# lines 526-533
elif op == "delete":
    # Add then delete -- AgeMem's delete takes an entry_id
    content = random.choice(self._TRAINING_SNIPPETS)
    await self.agemem.add(content, metadata={"source": "training"})
    # Note: delete requires an entry_id; we use the content hash as a proxy.
    # If AgeMem doesn't track IDs this way, delete may return False,
    # which is acceptable -- the model learns that delete needs a valid ID.
    result = await self.agemem.delete(str(hash(content)))
    success = bool(result)
```

The `_stable_hash` function already exists in `/Users/rajesh/athena/trading/market_data.py` lines 20-22:

```python
def _stable_hash(s: str) -> int:
    """Return a stable (PYTHONHASHSEED-independent) integer hash of a string."""
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)
```

However, importing from `trading.market_data` into `training.stage2_agemem.trainer` would create an awkward cross-domain dependency. A self-contained inline approach using `hashlib` is cleaner and matches the existing import style in the training package.

## Proposed Change

**File:** `/Users/rajesh/athena/training/stage2_agemem/trainer.py`

### 1. Add `hashlib` import at the top of the file (after line 9)

The file currently imports `logging` on line 10. Add `hashlib` alongside:

```python
# BEFORE (lines 8-10)
from dataclasses import dataclass
from enum import Enum
import logging

# AFTER
from dataclasses import dataclass
from enum import Enum
import hashlib
import logging
```

### 2. Replace `hash()` call on line 532

```python
# BEFORE (line 532)
                result = await self.agemem.delete(str(hash(content)))

# AFTER
                delete_id = hashlib.sha256(content.encode()).hexdigest()[:16]
                result = await self.agemem.delete(delete_id)
```

The 16-character hex prefix (64 bits of entropy) is sufficient for a training-time proxy ID and matches the style used elsewhere in the codebase (e.g., `_stable_hash` truncation patterns).

## Files Modified

| File | Line(s) | Change |
|------|---------|--------|
| `/Users/rajesh/athena/training/stage2_agemem/trainer.py` | After line 9 | Add `import hashlib` |
| `/Users/rajesh/athena/training/stage2_agemem/trainer.py` | 532 | Replace `str(hash(content))` with `hashlib.sha256(content.encode()).hexdigest()[:16]` |

## Acceptance Criteria

- `hash()` is no longer called anywhere in `trainer.py`.
- The delete operation uses a SHA-256 based ID that is identical across Python processes and `PYTHONHASHSEED` values.
- `hashlib` is imported at the module level (not inside the function).
- All 173 tests pass, 4 skipped.

## Edge Cases & Risks

1. **ID mismatch with AgeMem backend**: The delete ID was already a proxy -- AgeMem's `add()` returns a `bool`, not an entry ID, so the delete path was already best-effort. Changing from `hash()` to SHA-256 does not make this worse; it just makes the proxy stable.

2. **Encoding assumption**: `content.encode()` uses UTF-8 by default. All `_TRAINING_SNIPPETS` are ASCII, so this is safe. If non-ASCII content were introduced later, UTF-8 encoding would still work correctly.

3. **Truncation to 16 chars**: 16 hex characters = 64 bits. Collision probability for the ~10 training snippets is negligible (`~10^2 / 2^64`).

## Test Notes

- No new tests required; this is a determinism fix for an already-best-effort code path.
- Existing tests do not exercise the delete path with a real AgeMem backend (it would need a running Graphiti instance). The mock path returns `False` regardless of the ID, so test behavior is unchanged.
- To verify stability: run `python3 -c "import hashlib; print(hashlib.sha256('test'.encode()).hexdigest()[:16])"` twice and confirm identical output.
