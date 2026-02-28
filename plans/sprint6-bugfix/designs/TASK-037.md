# TASK-037: Deduplicate coordinator memory writes

## Summary
`CoordinatorAgent.act()` makes two consecutive `memory.add()` calls on lines 320–327 and 334–344. The first stores `{"coordination": result, "thought": thought}` with success metadata. The second stores `{"final_decision": ..., "agents_queried": ...}` with an `"operation": "coordination_summary"` tag. The `final_decision` and `agents_queried` data is already present in `result` (the first write's content), making the second write redundant. This doubles memory storage per decision cycle and increases memory operation latency for no informational gain.

## Current State

**File:** `agents/coordinator.py` (lines 317–346)

```python
# First memory write (lines 320-327)
if self.memory:
    try:
        await self.memory.add(
            content={"coordination": result, "thought": thought},
            metadata={
                "agent": self.name,
                "role": self.role,
                "success": True,
            }
        )
    except Exception as e:
        self.logger.warning(f"Memory store failed: {e}")

# Second memory write (lines 332-346) — overlapping data
if self.memory:
    try:
        await self.memory.add(
            content={
                "final_decision": result.get("final_decision"),
                "agents_queried": list(self.agents.keys()),
            },
            metadata={
                "agent": self.name,
                "role": self.role,
                "operation": "coordination_summary",
            }
        )
    except Exception as e:
        self.logger.warning(f"Memory store (summary) failed: {e}")
```

## Proposed Change

Merge into a single `memory.add()` call that captures all previously stored information:

```python
if self.memory:
    try:
        await self.memory.add(
            content={
                "coordination": result,
                "thought": thought,
                "final_decision": result.get("final_decision"),
                "agents_queried": list(self.agents.keys()),
            },
            metadata={
                "agent": self.name,
                "role": self.role,
                "success": True,
                "operation": "coordination_summary",
            }
        )
    except Exception as e:
        self.logger.warning("Memory store failed: %s", e)
```

## Files Modified

- `agents/coordinator.py`
  - Lines 317–346: replace two `memory.add()` blocks with one consolidated call
  - Also fix f-string in warning (line 329): `f"Memory store failed: {e}"` → `"Memory store failed: %s", e`
  - Also fix f-string in warning (line 346): `f"Memory store (summary) failed: {e}"` → `"Memory store failed: %s", e`

## Acceptance Criteria

- [ ] Exactly one `memory.add()` call per successful `act()` invocation (when `self.memory` is set)
- [ ] The stored content includes `coordination`, `thought`, `final_decision`, and `agents_queried`
- [ ] All existing tests pass

## Edge Cases & Risks

- **Memory retrieval queries:** If any downstream code queries memory for entries with `"operation": "coordination_summary"` metadata specifically, it will still find them (metadata field is preserved in the merged call).
- **Content size:** Merging means the stored content dict is slightly larger. `result` already contains `final_decision`, so storing it separately under `"final_decision"` is a minor duplication — acceptable for retrieval convenience.

## Test Notes

- `test_agents.py::TestCoordinatorAgent::test_act_returns_action` uses a mock memory. After this change, `mock_memory.add.call_count` should be 1 (not 2) per `act()` call. Update this assertion if present.
