# TASK-041: Guard numpy import in utils.py

## Summary
`core/utils.py` imports `numpy as np` unconditionally at line 9. Every other optional dependency in the codebase (`torch`, `aiohttp`, `yfinance`, `graphiti_core`) uses a `try/except ImportError` guard. The inconsistency means `core/utils.py` — which is imported by virtually every module — will crash at import time if numpy is not installed, taking down the entire system. Wrapping the numpy import in a try/except and guarding the `cosine_similarity` function restores consistency and allows the system to run without numpy.

## Current State

**File:** `core/utils.py` (lines 7–13)

```python
import logging
import random
import numpy as np          # <-- unconditional, line 9
from typing import Optional, Any, Dict, List
from datetime import datetime
import hashlib
import json
```

**File:** `core/utils.py` (lines 114–130, the only numpy-dependent function)

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    ...
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
```

## Proposed Change

**Step 1 — Guard the import:**

```python
import logging
import random
from typing import Optional, Any, Dict, List
from datetime import datetime
import hashlib
import json

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None      # type: ignore[assignment]
    HAS_NUMPY = False
```

**Step 2 — Guard `cosine_similarity`:**

```python
def cosine_similarity(a: Any, b: Any) -> float:
    """Compute cosine similarity between two vectors. Requires numpy."""
    if not HAS_NUMPY:
        raise ImportError(
            "numpy is required for cosine_similarity. Install with: pip install numpy"
        )
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
```

## Files Modified

- `core/utils.py`
  - Lines 7–13 (imports): wrap `import numpy as np` in try/except, add `HAS_NUMPY` flag
  - Lines 114–130 (`cosine_similarity`): add `if not HAS_NUMPY: raise ImportError(...)`

## Acceptance Criteria

- [ ] `import core.utils` succeeds even when numpy is not installed
- [ ] `cosine_similarity(a, b)` raises `ImportError` with a clear message when numpy is absent
- [ ] `cosine_similarity(a, b)` works correctly when numpy is present
- [ ] All existing tests pass (numpy is installed in the test environment)

## Edge Cases & Risks

- **Other numpy usages in utils.py:** Grep the file for `np.` to confirm `cosine_similarity` is the only numpy-dependent function. If others exist, guard those too.
- **Type annotations:** `a: np.ndarray` won't work when `np` is `None`. Use `a: Any` (already imported from `typing`) or add `TYPE_CHECKING` guard for the annotation.
- **Downstream callers:** Any module that calls `cosine_similarity` will now get a clear `ImportError` instead of a confusing `AttributeError: 'NoneType' object has no attribute 'linalg'`.

## Test Notes

- Existing tests that call `cosine_similarity` (if any) will pass when numpy is installed.
- Add test: mock `HAS_NUMPY = False`, call `cosine_similarity`, assert `ImportError` is raised.
