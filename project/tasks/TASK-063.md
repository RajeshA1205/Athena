# TASK-063: Fix LatentSpace broadcast missing unregistered agents

## Status
- **State:** Queued
- **Priority:** Critical
- **Depends on:** None
- **Created:** 2026-02-26

## Objective
Add an explicit `register_agent(agent_name: str)` method to `LatentSpace` that pre-creates the agent's deque entry. Call it from `coordinator.py`'s `initialize_communication()` for every registered agent so that `broadcast()` reaches all agents regardless of prior direct-message history.

## Context
`LatentSpace._queues` is a `defaultdict(deque)`. `broadcast()` iterates `self._queues.keys()` to deliver messages. Because the defaultdict only creates an entry on first access, agents that have never received a direct point-to-point message have no key in `_queues` and silently miss every broadcast. In practice, the coordinator's final-decision broadcast — the most important message in the pipeline — reaches zero agents when the system starts fresh.

This was missed in the Sprint 6/7/8 rounds because the unit tests for broadcast only verify delivery to agents that had prior direct messages.

## Scope & Constraints
- **May modify:** `communication/latent_space.py`, `agents/coordinator.py`
- **Must NOT modify:** Any other file
- The `register_agent` method must be idempotent (calling it twice for the same name is a no-op)
- Do not change the existing `send()`, `receive()`, or `broadcast()` signatures

## Input
- `communication/latent_space.py` — current source
- `agents/coordinator.py` — `initialize_communication()` method

## Expected Output
- `communication/latent_space.py` — new `register_agent(agent_name: str) -> None` method added; method creates the deque entry for `agent_name` if not already present
- `agents/coordinator.py` — `initialize_communication()` calls `self.latent_space.register_agent(name)` for each agent in the registry before any messages are sent

## Acceptance Criteria
- [ ] `LatentSpace.register_agent(name)` exists and creates the deque entry idempotently
- [ ] `coordinator.py`'s `initialize_communication()` calls `register_agent` for every agent
- [ ] A broadcast sent before any direct messages still delivers to all registered agents
- [ ] `python3 -m pytest tests/ -q` passes with 173 passed, 4 skipped (no regressions)
- [ ] `python3 main.py --mode dry-run` completes without error

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
