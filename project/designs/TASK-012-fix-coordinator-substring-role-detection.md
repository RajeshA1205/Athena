# TASK-012: Fix coordinator substring role detection fallback

## Problem

`CoordinatorAgent._resolve_conflicts()` at lines 448-451 and 483-485 in `agents/coordinator.py` uses a substring match `if role in agent_name` to guess an agent's role when the agent is not found in `self.agents`. This is fragile: an agent named `"risk_strategy_hybrid"` would match `"risk"` first (iteration order of `agent_priority.keys()`), and an agent named `"executor"` would match nothing because the role key is `"execution"`, not `"executor"`.

The substring fallback runs when `agent_name not in self.agents`, which happens if the coordinator receives recommendations from agents it did not register. This is a legitimate scenario in dynamic agent configurations.

## Files to change

| File | Lines | Change |
|------|-------|--------|
| `agents/coordinator.py` | 444-451 | Replace substring fallback with explicit role mapping or remove it |
| `agents/coordinator.py` | 478-485 | Same pattern (duplicated in winning_agent detection) |

## Approach

### Option A: Strict lookup, no fallback (recommended)

Remove the substring fallback entirely. If `agent_name` is not in `self.agents`, assign `agent_role = None` and use the default priority of 1 (already happens via `self.agent_priority.get(agent_role, 1)` on line 453/486).

```python
agent_role = None
if agent_name in self.agents:
    agent_role = self.agents[agent_name].role
# No substring fallback -- unknown agents get default priority 1
```

Log a warning when an unregistered agent sends a recommendation:
```python
if agent_role is None:
    self.logger.warning("Unregistered agent '%s' in recommendations; using default priority", agent_name)
```

### Option B: Explicit role-from-name mapping

If substring detection is desired, make it explicit rather than iterating dict keys:

```python
_NAME_TO_ROLE = {
    "risk_manager": "risk",
    "strategy_agent": "strategy",
    "market_analyst": "analyst",
    "execution_agent": "execution",
}
agent_role = _NAME_TO_ROLE.get(agent_name)
```

This is deterministic and won't accidentally match substrings.

### Recommendation: Option A

The substring fallback solves a problem that should not exist. If an agent is sending recommendations to the coordinator, it should be registered. The fallback masks configuration errors.

### Deduplication

The role-detection logic is duplicated between lines 444-451 and 478-485. Extract into a private helper:

```python
def _get_agent_role(self, agent_name: str) -> Optional[str]:
    if agent_name in self.agents:
        return self.agents[agent_name].role
    self.logger.warning("Unregistered agent '%s'; using default priority", agent_name)
    return None
```

Call from both locations.

## Edge cases / risks

- **Breaking change**: If any existing code relies on the substring fallback to assign elevated priority to unregistered agents, this change will reduce their priority to 1. This is unlikely in practice since the CLI and `main.py` both register all agents before use.
- **Dict iteration order**: In Python 3.7+, `dict.keys()` preserves insertion order. The current `agent_priority` dict has `risk` first, so an agent matching multiple substrings would always get `risk` priority. Removing the fallback eliminates this fragile dependency.

## Acceptance criteria

- [ ] Substring-based role detection is removed from `_resolve_conflicts()`.
- [ ] A warning is logged when an unregistered agent name appears in recommendations.
- [ ] Unregistered agents receive default priority 1.
- [ ] Role detection logic is extracted into a single `_get_agent_role()` helper (no duplication).
- [ ] Existing tests with registered agents pass unchanged.
- [ ] New test: unregistered agent recommendation uses default priority, warning is logged.
- [ ] `pytest tests/ -q` remains green.
