# TASK-065: Remove Duplicate Order Dataclass from ExecutionAgent

## Summary

`agents/execution_agent.py` defines its own `Order` dataclass (lines 23-45) with fields `filled_qty` (float) and `created_at: Optional[datetime]`. The canonical `trading/order_management.py` defines a different `Order` with fields `filled_quantity` (float) and `created_at: str` (ISO timestamp string). The two types are incompatible — passing orders between the execution agent and order management module will cause `AttributeError` on field access. Remove the local dataclass and import the canonical `Order` from `trading.order_management`. Update all internal field references accordingly.

**Dependency:** TASK-062 must be applied first (UTC timestamps) so the datetime values are already correct when we unify types.

## Current State

**File:** `agents/execution_agent.py`, lines 23-45 — local Order dataclass:
```python
@dataclass
class Order:
    """Order representation."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0          # ← different from canonical
    avg_fill_price: float = 0.0
    created_at: Optional[datetime] = None   # ← datetime obj vs str
    updated_at: Optional[datetime] = None   # ← not in canonical
    fills: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)   # after TASK-062
        if self.updated_at is None:
            self.updated_at = datetime.now(timezone.utc)   # after TASK-062
```

**Internal uses of local-only fields:**

| Line | Usage | Canonical replacement |
|------|-------|-----------------------|
| 238 | `order.filled_qty = sum(...)` | `order.filled_quantity = sum(...)` |
| 239 | `if order.filled_qty >= order.quantity` | `if order.filled_quantity >= order.quantity` |
| 242 | `elif order.filled_qty > 0` | `elif order.filled_quantity > 0` |
| 243 | `order.updated_at = datetime.now(timezone.utc)` | remove (canonical has no `updated_at`) |
| 246 | `if order.filled_qty > 0` | `if order.filled_quantity > 0` |
| 247 | `/ order.filled_qty` | `/ order.filled_quantity` |
| 259 | f-string: `order.filled_qty` | `order.filled_quantity` |
| 549 | `"filled_qty": order.filled_qty` | `"filled_qty": order.filled_quantity` *(keep key name for API compat)* |
| 551 | `order.created_at.isoformat() if order.created_at else None` | `order.created_at` *(already a str in canonical)* |
| 571 | `order.updated_at = datetime.now(timezone.utc)` | remove |

**File:** `trading/order_management.py` — canonical Order fields (reference):
```python
@dataclass
class Order:
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    fees: float = 0.0
    fills: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

Note: canonical `Order` has no `updated_at` field. Code that sets `order.updated_at` must be removed.

## Proposed Change

### 1. Remove local Order dataclass (lines 23-45)

Delete the entire `@dataclass class Order:` block and its `__post_init__`.

### 2. Update import (line 14 area)

Add to existing imports:
```python
from trading.order_management import Order
```

Remove `from dataclasses import dataclass, field` if no longer needed elsewhere (check — `field` is used only in the local Order).

### 3. Update `_create_order()` (around line 336-358)

The canonical `Order` uses `created_at` as a str (auto-populated by `default_factory`), not `Optional[datetime]`. The `Order(...)` construction already doesn't pass `created_at`, so the default will apply. Remove any explicit `created_at=` argument if present.

### 4. Update `act()` field references (lines 238-248, 259)

```python
# BEFORE
order.filled_qty = sum(f["quantity"] for f in fills)
if order.filled_qty >= order.quantity:
    ...
elif order.filled_qty > 0:
    order.status = OrderStatus.PARTIAL
order.updated_at = datetime.now(timezone.utc)   # REMOVE - no such field

if order.filled_qty > 0:
    order.avg_fill_price = sum(...) / order.filled_qty

# f-string at line 259: order.filled_qty

# AFTER
order.filled_quantity = sum(f["quantity"] for f in fills)
if order.filled_quantity >= order.quantity:
    ...
elif order.filled_quantity > 0:
    order.status = OrderStatus.PARTIAL
# (remove updated_at line)

if order.filled_quantity > 0:
    order.avg_fill_price = sum(...) / order.filled_quantity
```

### 5. Update `cancel_order()` (line 571)

```python
# BEFORE
order.updated_at = datetime.now(timezone.utc)
# AFTER
# Remove — canonical Order has no updated_at field
```

### 6. Update `_order_to_dict()` (lines 540-553)

```python
# BEFORE
"filled_qty": order.filled_qty,
...
"created_at": order.created_at.isoformat() if order.created_at else None,

# AFTER
"filled_qty": order.filled_quantity,   # keep key name for API compat
...
"created_at": order.created_at,        # already a str in canonical Order
```

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `agents/execution_agent.py` | 23-45 | Delete local `Order` dataclass |
| `agents/execution_agent.py` | ~14 | Add `from trading.order_management import Order` |
| `agents/execution_agent.py` | 238-248 | `filled_qty` → `filled_quantity`; remove `updated_at` set |
| `agents/execution_agent.py` | 259 | `filled_qty` → `filled_quantity` in f-string |
| `agents/execution_agent.py` | 549, 551 | `_order_to_dict`: update field accesses |
| `agents/execution_agent.py` | 571 | Remove `order.updated_at = ...` |

## Acceptance Criteria

- No local `Order` dataclass defined in `execution_agent.py`.
- `from trading.order_management import Order` is present.
- `order.filled_quantity` is used everywhere (not `filled_qty`).
- No `order.updated_at` assignments remain.
- `_order_to_dict` serializes correctly (canonical Order fields).
- 173 tests pass, 4 skipped.

## Edge Cases & Risks

1. **`updated_at` consumers**: If any test or downstream code reads `order.updated_at` from an `ExecutionAgent` order, it will get `AttributeError` after this fix. Search for `.updated_at` in tests before applying.
2. **`fills` field**: Canonical `Order.fills` is `List[Any]`. The execution agent's code sets `order.fills = fills` (a list of dicts). This is compatible.
3. **`metadata` field**: Canonical `Order` has `metadata: Dict[str, Any]`. ExecutionAgent passes metadata in `_create_order`. Compatible.
4. **`dataclass` / `field` imports**: After removing the local dataclass, `from dataclasses import dataclass, field` may be unused. Remove if so to avoid lint warnings.

## Test Notes

- After the change, `ExecutionAgent._create_order()` produces `trading.order_management.Order` instances.
- Verify `order.filled_quantity` is set correctly after `_simulate_execution`.
- Verify `_order_to_dict` returns `"filled_qty": N` (keeping the external key name).
- Run `python3 -m pytest tests/ -q` — expect 173 passed, 4 skipped.
