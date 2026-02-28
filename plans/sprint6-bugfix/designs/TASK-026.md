# TASK-026: Unify trading enums into a canonical module

## Summary
`OrderType`, `OrderSide`, and `OrderStatus` are independently defined in both `agents/execution_agent.py` and `trading/order_management.py`. The two definitions have **diverged**: `execution_agent` has `TWAP` and `VWAP` order types that `order_management` lacks, and `order_management` has `SUBMITTED` and `PARTIALLY_FILLED` statuses that `execution_agent` lacks (using `PARTIAL` instead). Any code that compares an `execution_agent.OrderStatus` with an `order_management.OrderStatus` will silently fail because they are different enum classes. This task creates a single canonical `trading/enums.py` and updates both files to import from it.

## Current State

**File:** `agents/execution_agent.py` (lines 23–45)

```python
class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TWAP = "twap"      # <-- only here
    VWAP = "vwap"      # <-- only here

class OrderStatus(Enum):
    PENDING = "pending"
    PARTIAL = "partial"          # <-- different name
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    # missing: SUBMITTED, PARTIALLY_FILLED

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"
```

**File:** `trading/order_management.py` (lines 26–47)

```python
class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    # missing: TWAP, VWAP

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"         # <-- only here
    PARTIALLY_FILLED = "partially_filled"  # <-- different name
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
```

## Proposed Change

**Step 1 — Create `trading/enums.py`:**

```python
"""Canonical trading enum definitions shared across all trading modules."""
from enum import Enum


class OrderType(Enum):
    """Supported order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TWAP = "twap"
    VWAP = "vwap"


class OrderSide(Enum):
    """Direction of an order."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Lifecycle states of an order."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    # Backwards-compatibility alias: PARTIAL -> PARTIALLY_FILLED
    PARTIAL = "partially_filled"
```

> **Decision point:** `PARTIAL` (execution_agent) vs `PARTIALLY_FILLED` (order_management). Recommendation: canonical name is `PARTIALLY_FILLED` (more explicit). Add `PARTIAL = "partially_filled"` as an alias so existing code using `OrderStatus.PARTIAL` continues to work without changes.

**Step 2 — Update `agents/execution_agent.py`:**
- Delete lines 23–45 (the three enum class definitions)
- Add import: `from trading.enums import OrderType, OrderSide, OrderStatus`

**Step 3 — Update `trading/order_management.py`:**
- Delete lines 26–47 (the three enum class definitions)
- Add import: `from trading.enums import OrderType, OrderSide, OrderStatus`

**Step 4 — Update `trading/__init__.py`:**
- Re-export: `from trading.enums import OrderType, OrderSide, OrderStatus`

## Files Modified

- `trading/enums.py` — **create new file**
- `agents/execution_agent.py` — remove local enum definitions, add import
- `trading/order_management.py` — remove local enum definitions, add import
- `trading/__init__.py` — add re-export

## Acceptance Criteria

- [ ] Only one definition of each enum exists in the codebase (`trading/enums.py`)
- [ ] `from agents.execution_agent import OrderType` and `from trading.order_management import OrderType` resolve to the same class object
- [ ] `OrderStatus.PARTIAL` and `OrderStatus.PARTIALLY_FILLED` both exist and compare equal (`OrderStatus.PARTIAL == OrderStatus.PARTIALLY_FILLED` → `True`, same value `"partially_filled"`)
- [ ] `OrderType.TWAP` and `OrderType.VWAP` are accessible from both modules
- [ ] All 171 existing tests pass

## Edge Cases & Risks

- **Enum alias semantics in Python:** In Python's `enum.Enum`, two members with the same value create an alias — the second name is an alias for the first. `OrderStatus.PARTIAL is OrderStatus.PARTIALLY_FILLED` will be `True`. This is intentional. Verify that no code does `if status == OrderStatus.PARTIAL` in a way that breaks with the alias.
- **`Order` dataclass uses these enums:** `execution_agent.Order` and `order_management.Order` both have `side: OrderSide`, `order_type: OrderType`, `status: OrderStatus` fields. After the change both use the same classes, so cross-module order passing will work correctly.
- **Import cycles:** `agents/execution_agent.py` importing from `trading/` is fine — `trading` does not import from `agents`. No cycle.
- **Test file imports:** `tests/test_trading.py` imports `OrderStatus` from `trading.order_management`. After the change this still works (re-exported from `trading.enums` via `order_management`). No test changes needed unless tests reference `OrderStatus.PARTIAL` (check and update if so).

## Test Notes

- Run full suite after the change: `pytest tests/ -v`
- Add one test: `assert OrderStatus.PARTIAL == OrderStatus.PARTIALLY_FILLED`
- Add one test: `assert OrderType.TWAP.value == "twap"` accessible from both `execution_agent` and `order_management` import paths
