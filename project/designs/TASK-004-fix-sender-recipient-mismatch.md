# TASK-004: Fix sender/recipient vs from/to key mismatch in WorkflowDiscovery

## Problem

`WorkflowDiscovery._extract_interaction_pattern()` (line 241-242 in `evolution/workflow_discovery.py`) reads `interaction.get("sender")` and `interaction.get("recipient")`, but the `AgentMessage` dataclass in `core/base_agent.py` uses fields named `sender` and `recipient`. The CLI and coordinator pass `AgentMessage` objects via `AgentContext.messages`, but when execution traces are built manually (e.g., for evolution analysis), callers may use `from`/`to` keys instead. This causes `_extract_interaction_pattern` to silently skip all interactions (the `if not sender or not recipient: continue` guard on line 245 drops them), producing empty communication graphs and preventing workflow pattern discovery.

## Files to change

| File | Lines | Change |
|------|-------|--------|
| `evolution/workflow_discovery.py` | 241-242 | Accept both `sender`/`from` and `recipient`/`to` keys |
| `evolution/workflow_discovery.py` | 243 | Similarly normalize `message_type` / `type` |

## Approach

1. In `_extract_interaction_pattern`, normalize each interaction dict at the top of the loop:
   ```
   sender = interaction.get("sender") or interaction.get("from")
   recipient = interaction.get("recipient") or interaction.get("to")
   message_type = interaction.get("message_type") or interaction.get("type", "default")
   ```
2. Add a `logger.debug` when falling back to `from`/`to` keys so silent mismatches are visible at debug level.
3. Audit all call sites that build `interactions` lists passed to `analyze_execution()`. Ensure each either uses `sender`/`recipient` keys consistently, or document the fallback in `_extract_interaction_pattern`'s docstring.
4. Add unit test: pass interactions with `from`/`to` keys and verify the communication graph is non-empty.

## Edge cases / risks

- If a caller passes both `sender` AND `from` on the same dict, the `or` fallback still picks the canonical `sender` key first -- correct behavior.
- `None` vs missing key: `interaction.get("sender")` returns `None` for both cases; the `or` fallback handles this cleanly.
- Empty string values: `""` is falsy in Python, so `interaction.get("sender") or interaction.get("from")` will fall through to `from` if sender is `""`. This is acceptable since an empty-string sender is invalid anyway.

## Acceptance criteria

- [ ] `_extract_interaction_pattern` produces non-empty `communication_graph` when interactions use `from`/`to` keys.
- [ ] Existing tests using `sender`/`recipient` keys continue to pass unchanged.
- [ ] New test covers the `from`/`to` fallback path.
- [ ] `pytest tests/ -q` remains green.
