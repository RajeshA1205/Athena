# TASK-020: Create Trading Portfolio Module

## Status
- **State:** Queued
- **Priority:** 🟡 High
- **Depends on:** TASK-019
- **Created:** 2026-02-15

## Objective
Create trading/portfolio.py module for position tracking, P&L calculation, and risk monitoring.

## Context
Part of Sprint 5 trading infrastructure. Provides portfolio state to Risk Manager and Strategy agents.

## Scope & Constraints
**Files to Create:** `/Users/rajesh/athena/trading/portfolio.py`
**Files to Modify:** `/Users/rajesh/athena/trading/__init__.py`
**Constraints:** Async operations, real-time P&L updates, position limits

## Expected Output
Portfolio class with:
- get_positions() — Current positions
- calculate_pnl() — Real-time P&L
- get_exposure() — Market exposure
- check_limits() — Position limit checks
- update_from_fill() — Update on order fills

## Acceptance Criteria
- [ ] Portfolio class created
- [ ] Position tracking
- [ ] P&L calculation
- [ ] Exposure monitoring
- [ ] Position limit checking

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|

## Review Notes
