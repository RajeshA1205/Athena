# TASK-045: Convert f-string logging to lazy %s-style in agent files

## Summary
All five agent files use f-strings inside `logger.info/warning/error` calls (40 instances total across 5 files). When a log level is disabled (e.g. DEBUG messages in production), Python's logging module skips the handler but still evaluates the f-string before the call, wasting CPU on string formatting that produces no output. The standard fix is `%s`-style lazy formatting: `logger.warning("Failed: %s", e)` — the string is only formatted if the message will actually be emitted. This is a purely mechanical, low-risk change with no behaviour impact.

## Current State (representative examples)

**File:** `agents/coordinator.py` (10 instances)
```python
self.logger.info(f"Registered agent: {name} ({agent.role})")         # line 113
self.logger.info(f"Coordinating agents for task: {context.task}")    # line 150
self.logger.warning(f"Memory retrieve failed: {e}")                  # line 163
self.logger.warning(f"Memory store failed: {e}")                     # line 329
self.logger.error(f"Error in act(): {e}")                            # line 359
self.logger.info(f"Allocating resources for {len(requests)} requests") # line 513
```

**Counts per file:**
- `coordinator.py`: 10
- `execution_agent.py`: 9
- `market_analyst.py`: 7
- `risk_manager.py`: 7
- `strategy_agent.py`: 7
- **Total: 40 instances**

## Proposed Change

Convert each f-string call to `%s`-style. Patterns:

| Before | After |
|--------|-------|
| `logger.info(f"Registered agent: {name} ({agent.role})")` | `logger.info("Registered agent: %s (%s)", name, agent.role)` |
| `logger.warning(f"Memory retrieve failed: {e}")` | `logger.warning("Memory retrieve failed: %s", e)` |
| `logger.info(f"Allocating resources for {len(requests)} requests")` | `logger.info("Allocating resources for %d requests", len(requests))` |
| `logger.error(f"Error in act(): {e}")` | `logger.error("Error in act(): %s", e)` |

Use `%d` for integer values and `%s` for everything else (Python's `%` operator will call `str()` on the argument).

**Note:** F-strings with no interpolation (e.g. `logger.info(f"Starting coordination")`) can simply drop the `f` prefix: `logger.info("Starting coordination")`.

## Files Modified

- `agents/coordinator.py` — 10 logger call sites
- `agents/execution_agent.py` — 9 logger call sites
- `agents/market_analyst.py` — 7 logger call sites
- `agents/risk_manager.py` — 7 logger call sites
- `agents/strategy_agent.py` — 7 logger call sites

## Acceptance Criteria

- [ ] Zero f-string interpolations inside `logger.debug/info/warning/error/critical` calls across all 5 files
- [ ] Log output is identical to before (same messages, same values)
- [ ] All existing tests pass

## Edge Cases & Risks

- **Exception objects as `%s`:** `logger.error("Failed: %s", e)` calls `str(e)` lazily. This is identical to `f"Failed: {e}"` in output but only evaluated when the message is emitted.
- **Multi-argument format strings:** `f"Agent {name} responded in {duration:.2f}s"` becomes `logger.info("Agent %s responded in %.2fs", name, duration)`. The `:.2f` format specifier translates to `%.2f` in `%`-style formatting.
- **Grep for regressions:** After the change, grep for `logger\.[a-z]*(f"` across all agent files to confirm zero matches.
- **Low risk:** This is a pure string-formatting style change. No logic is altered.

## Test Notes

- No new tests needed — this is a style change only.
- Run `pytest tests/ -v` to confirm all 171 tests still pass.
- Optionally add a lint rule (e.g. `pylint W1202`) to prevent f-string logging from being reintroduced.
