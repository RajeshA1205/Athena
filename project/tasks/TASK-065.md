# TASK-065: Remove duplicate Order dataclass from ExecutionAgent

## Status
- **State:** Queued
- **Priority:** Major
- **Depends on:** TASK-062
- **Created:** 2026-02-26

## Objective
Remove the local `Order` dataclass defined inside `agents/execution_agent.py` and replace all its usages with the canonical `Order` from `trading.order_management`. Update all field references to match the canonical field names (`filled_quantity` instead of `filled_qty`, `created_at` as a string or UTC datetime per the canonical definition).

## Context
`execution_agent.py` defines a local `Order` dataclass with fields including `filled_qty` (int/float) and `created_at: Optional[datetime]`. The canonical `trading/order_management.Order` uses `filled_quantity` and `created_at` as a string timestamp. When orders are passed between the execution layer and the order management layer, attribute lookups fail with `AttributeError`. This prevents the system from actually tracking fill quantities.

TASK-062 must complete first because the canonical Order's `created_at` field must already use UTC-aware datetimes before the local Order class is removed — otherwise the timestamp format mismatch will resurface immediately.

Sprint 6 (TASK-026) unified trading enums; this task completes the unification by removing the duplicate Order type.

## Scope & Constraints
- **May modify:** `agents/execution_agent.py` only
- **Must NOT modify:** `trading/order_management.py` or any other file
- After removing the local `Order`, all code in `execution_agent.py` must reference `trading.order_management.Order` exclusively
- Field name migration: `filled_qty` → `filled_quantity` throughout the file
- Do not change any method signatures visible to callers outside the file

## Input
- `agents/execution_agent.py` — contains local `Order` dataclass and all usages
- `trading/order_management.py` — canonical `Order` definition (reference only, do not modify)

## Expected Output
- `agents/execution_agent.py` — local `Order` dataclass removed; `from trading.order_management import Order` added; all `filled_qty` references changed to `filled_quantity`

## Acceptance Criteria
- [ ] No local `Order` class or dataclass definition remains in `execution_agent.py`
- [ ] `from trading.order_management import Order` is present in the imports
- [ ] Zero occurrences of `filled_qty` remain in `execution_agent.py` (grep clean)
- [ ] All replaced with `filled_quantity`
- [ ] `python3 -m pytest tests/ -q` passes with 173 passed, 4 skipped (no regressions)
- [ ] `python3 main.py --mode dry-run` completes without error

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
