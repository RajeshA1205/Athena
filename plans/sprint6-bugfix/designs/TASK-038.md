# TASK-038: Fix _allocate_resources docstring to match implementation

## Summary
`CoordinatorAgent._allocate_resources()` documents itself as using a "round-robin approach" (docstring line 505) and returns `"method": "round_robin"` in each allocation result (line 539). However, the actual implementation at lines 518–535 is **proportional allocation**: each agent receives a share proportional to its request relative to the total (`share = requested_amount / total_requests[resource_type]`). This is a purely informational fix — no logic changes — but the mismatch misleads anyone reading the code or using the returned metadata to understand how resources were divided.

## Current State

**File:** `agents/coordinator.py`

```python
async def _allocate_resources(
    self, requests: Dict[str, Dict]
) -> Dict[str, Dict]:
    """
    Allocate resources to agents using round-robin approach.   # <-- line 505, wrong

    Args:
        requests: Dictionary of resource requests from agents

    Returns:
        Dictionary mapping agent names to allocated resources
    """
    ...
    allocations[agent_name] = {
        "allocated_resources": allocated,
        "method": "round_robin",    # <-- line 539, wrong
    }
```

The actual logic:
```python
share = requested_amount / total_requests[resource_type]  # proportional, not round-robin
```

## Proposed Change

```python
async def _allocate_resources(
    self, requests: Dict[str, Dict]
) -> Dict[str, Dict]:
    """
    Allocate resources to agents using demand-proportional allocation.

    Each agent receives a share of each resource type proportional to its
    requested amount relative to the total demand across all agents.

    Args:
        requests: Dictionary of resource requests from agents

    Returns:
        Dictionary mapping agent names to allocated resources
    """
    ...
    allocations[agent_name] = {
        "allocated_resources": allocated,
        "method": "proportional",    # changed from "round_robin"
    }
```

## Files Modified

- `agents/coordinator.py`
  - Line 505: docstring `"round-robin approach"` → `"demand-proportional allocation"` + expanded description
  - Line 539: `"method": "round_robin"` → `"method": "proportional"`

## Acceptance Criteria

- [ ] Docstring accurately describes the proportional allocation algorithm
- [ ] `"method"` value in returned dict is `"proportional"` not `"round_robin"`
- [ ] No logic changes — all existing tests pass

## Edge Cases & Risks

- **Downstream consumers of `"method"` field:** If any code checks `allocation["method"] == "round_robin"`, it will break. Search the codebase for this string before merging.
- **Pure documentation change:** Zero risk to runtime behaviour since only docstring and a metadata string value change.

## Test Notes

- No new tests needed.
- Check if any test asserts `result["method"] == "round_robin"` — update to `"proportional"` if so.
