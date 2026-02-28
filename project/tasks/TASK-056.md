# TASK-056: Fix `_ACTION_TO_IDX` key case mismatch in GRPO

## Status
- **State:** Queued
- **Priority:** 🔴 Critical
- **Depends on:** None
- **Created:** 2026-02-25

## Objective
Fix the `_ACTION_TO_IDX` dict in `training/stage2_agemem/grpo.py` so its keys match what callers actually pass. Currently the dict is built with `op.name` (uppercase, e.g. `"ADD"`), but callers look up by `op.value` (lowercase, e.g. `"add"`). The `.get(action, 0)` default silently returns 0 for every action, meaning all GRPO log-probability lookups return ADD's probability regardless of the true action.

## Context
`_ACTION_TO_IDX` is used inside `_get_action_logprob` (and related helpers) to map an action string to the correct index into the log-softmax output of `MemoryActionHead`. Because every lookup silently falls back to index 0 (ADD), the GRPO importance-sampling ratios computed in TASK-058 will also be wrong until this is fixed first. See `project/context.md` for the full enum layout: `MemoryActionHead.ACTION_DIM = 6`, values are `"add"`, `"update"`, `"delete"`, `"retrieve"`, `"summary"`, `"filter"`.

## Scope & Constraints
- **May modify:** `training/stage2_agemem/grpo.py`
- **Must NOT modify:** any other file
- One-line change to the dict comprehension

## Input
- `training/stage2_agemem/grpo.py` — locate `_ACTION_TO_IDX` dict comprehension

## Expected Output
Dict comprehension changed from:
```python
_ACTION_TO_IDX = {op.name: i for i, op in enumerate(MemoryOperation)}
```
to:
```python
_ACTION_TO_IDX = {op.value: i for i, op in enumerate(MemoryOperation)}
```
(or equivalent inline construction).

## Acceptance Criteria
- [ ] `_ACTION_TO_IDX` keys are lowercase strings matching `MemoryOperation` `.value` fields
- [ ] No other lines in `grpo.py` are changed
- [ ] `python3 -m pytest tests/ -q` stays green (173 passed, 4 skipped)

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
