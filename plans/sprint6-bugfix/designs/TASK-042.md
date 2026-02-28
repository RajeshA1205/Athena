# TASK-042: Add recursion depth limit to deep_merge

## Summary
`deep_merge()` in `core/utils.py` recursively merges nested dicts with no depth limit. A config dict with circular references or very deep nesting will cause a stack overflow (`RecursionError`) with no useful error message. Adding a `max_depth` parameter (default 20) allows the function to detect and fail gracefully on pathological inputs while being completely transparent for normal usage (real config dicts are rarely more than 5–6 levels deep).

## Current State

**File:** `core/utils.py` (lines 165–184)

```python
def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)   # <-- no depth limit
        else:
            result[key] = value

    return result
```

## Proposed Change

```python
def deep_merge(
    base: Dict[str, Any],
    override: Dict[str, Any],
    max_depth: int = 20,
    _current_depth: int = 0,
) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.

    Args:
        base: Base dictionary
        override: Override dictionary (takes precedence)
        max_depth: Maximum recursion depth (default 20). Raises RecursionError if exceeded.
        _current_depth: Internal depth counter — do not pass externally.

    Returns:
        Merged dictionary
    """
    if _current_depth > max_depth:
        raise RecursionError(
            f"deep_merge exceeded maximum depth of {max_depth}. "
            "Check for circular references or overly nested config."
        )

    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(
                result[key], value,
                max_depth=max_depth,
                _current_depth=_current_depth + 1,
            )
        else:
            result[key] = value

    return result
```

## Files Modified

- `core/utils.py`
  - Lines 165–184: add `max_depth` and `_current_depth` parameters, add depth guard at top of function body

## Acceptance Criteria

- [ ] `deep_merge(a, b)` with no extra arguments still works for all normal inputs
- [ ] `deep_merge(a, b)` with dicts nested > 20 levels raises `RecursionError` with a descriptive message
- [ ] The `max_depth` parameter is respected when passed explicitly
- [ ] All existing tests pass

## Edge Cases & Risks

- **`_current_depth` as public API:** Prefixing with `_` signals it is internal. External callers should only use `max_depth`. Document this in the docstring.
- **Python's native recursion limit:** Python's default recursion limit is ~1000. For `max_depth=20`, we hit the explicit guard long before Python's own limit. No change to `sys.setrecursionlimit` is needed.
- **Real config depth:** ATHENA's `ATHENAConfig` is at most 3–4 levels deep. `max_depth=20` is conservative — any legitimate config is far shallower.

## Test Notes

- Existing tests using `deep_merge` (if any) will pass unchanged.
- Add test: create a dict 21 levels deep, call `deep_merge`, assert `RecursionError` is raised.
- Add test: create a dict 5 levels deep, call `deep_merge`, assert result is correct.
