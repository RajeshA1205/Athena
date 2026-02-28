# TASK-043: Fix RiskManager.think() hardcoded done flag

## Summary
`RiskManagerAgent.think()` returns `"done": False` unconditionally (line 181), regardless of whether the risk analysis completed successfully. The `done` field is part of the `AgentContext` / `think()` return convention used by the Coordinator to determine whether an agent has finished processing. Returning `False` always signals "still working" and could cause the Coordinator to treat risk assessment as perpetually incomplete. Since `think()` runs synchronously to completion before returning, the correct value is `True`.

## Current State

**File:** `agents/risk_manager.py` (lines 170–182)

```python
return {
    "task": context.task,
    "risk_level": risk_level,
    "var_95": var_95,
    "expected_shortfall": expected_shortfall,
    "portfolio_metrics": portfolio_metrics,
    "exposures": self._calculate_exposures(positions),
    "compliance_issues": all_issues,
    "alerts": alerts,
    "memory_context": memory_context,
    "latent_messages": latent_messages,
    "done": False,    # <-- line 181, always False
}
```

## Proposed Change

```python
return {
    "task": context.task,
    "risk_level": risk_level,
    "var_95": var_95,
    "expected_shortfall": expected_shortfall,
    "portfolio_metrics": portfolio_metrics,
    "exposures": self._calculate_exposures(positions),
    "compliance_issues": all_issues,
    "alerts": alerts,
    "memory_context": memory_context,
    "latent_messages": latent_messages,
    "done": True,    # Analysis is complete when think() returns
}
```

## Files Modified

- `agents/risk_manager.py`
  - Line 181: `"done": False` → `"done": True`

## Acceptance Criteria

- [ ] `risk_manager.think(context)["done"]` is `True`
- [ ] All existing tests pass

## Edge Cases & Risks

- **Other agents:** Check whether other agents also have `"done": False`. MarketAnalyst, StrategyAgent, ExecutionAgent should all return `"done": True` when `think()` completes synchronously. This task only changes RiskManager; a follow-up task could audit others.
- **Coordinator dependency:** If the Coordinator checks `thought.get("done")` to decide whether to proceed, fixing this will change Coordinator behavior. Review `coordinator.py`'s handling of the `done` field before merging.

## Test Notes

- `test_agents.py::TestRiskManagerAgent::test_think_returns_dict` likely checks keys but not the `done` value. Update to assert `result["done"] is True`.
