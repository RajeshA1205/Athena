# TASK-070: Fix Canonical Order Dataclass (fills field + created_at/updated_at defaults)

## Summary

The canonical `Order` in `trading/order_management.py` has two defects found during Sprint 9 review:

1. **Missing `fills` field**: `execution_agent.py` sets `order.fills = fills` as a dynamic attribute after construction. The canonical `Order` dataclass has no `fills` field, so any code path that accesses `order.fills` before `act()` sets it (e.g. `_order_to_dict`) will raise `AttributeError`. Add `fills: List[Any] = field(default_factory=list)`.

2. **Empty-string defaults for timestamps**: `created_at: str = ""` and `updated_at: str = ""`. Orders constructed without explicit timestamps silently carry empty strings through serialization and logging. The defaults should be the current UTC time in ISO-8601 format.

## Current State

**File:** `trading/order_management.py`, lines 56-60:

```python
    created_at: str = ""
    updated_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
```

No `fills` field declared anywhere in the dataclass (lines 48-60).

`execution_agent.py` line 212 sets:
```python
order.fills = fills  # dynamic attribute — not in dataclass
```

`_order_to_dict` at line 526 reads:
```python
len(order.fills)  # AttributeError if fills not set
```

## Proposed Change

**File:** `trading/order_management.py`

```python
# BEFORE (lines 56-60)
    created_at: str = ""
    updated_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

# AFTER
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    fills: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

Also update the class docstring to mention `fills`:
```
fills: List of fill records accumulated during execution.
```

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `trading/order_management.py` | 56-60 | Replace empty-string defaults with `default_factory` timestamps; add `fills` field |
| `trading/order_management.py` | docstring (~44) | Add `fills` attribute description |

## Acceptance Criteria

- `Order()` constructed with no arguments has `created_at` and `updated_at` as valid ISO-8601 UTC strings (not `""`).
- `Order()` has a `fills` attribute defaulting to `[]` — no `AttributeError` when accessing `order.fills` on a freshly constructed Order.
- `order.to_dict()` and `order.from_dict()` continue to work correctly.
- `execution_agent.py` line 212 (`order.fills = fills`) now sets a declared dataclass field, not a monkey-patched attribute.
- 173 tests pass, 4 skipped.

## Edge Cases & Risks

1. **`from_dict` backward compat**: `from_dict` uses `data.get("created_at", "")` — old serialized orders with `created_at: ""` will deserialize to `""`. Acceptable: the default_factory only applies to freshly constructed Orders, not deserialized ones.
2. **`fills` in `to_dict`/`from_dict`**: The canonical `to_dict` does not currently serialize `fills`. It can remain that way — `execution_agent._order_to_dict` handles fills serialization for the execution agent's own purposes.
3. **Field ordering**: New `fills` field is inserted before `metadata` so required fields (no default) still come before optional fields (with default). This maintains valid dataclass field ordering.
