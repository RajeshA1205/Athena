# TASK-059: Log Debug Message on Model-Driven Selection Failure

## Summary

`_select_operation_unified()` in `trainer.py` catches all exceptions from the model-driven operation selection path with `except Exception: pass`. This silently swallows real bugs (tensor shape mismatches, CUDA OOM, import errors) making them impossible to diagnose. Change to log at debug level.

## Current State

**File:** `/Users/rajesh/athena/training/stage2_agemem/trainer.py`, line 425

```python
            except Exception:
                pass  # Fall back to heuristic
```

The surrounding context (lines 410-427):

```python
    if (
        hasattr(self.model, "encode")
        and hasattr(self.model, "action_head")
        and self.model.action_head is not None
    ):
        try:
            import torch
            state_text = self.grpo._format_state(state)
            with torch.no_grad():
                embedding = self.model.encode(state_text)
                logits = self.model.action_head(embedding)
                probs = torch.softmax(logits, dim=-1)
                action_idx = torch.multinomial(probs, 1).item()
            # Map index back to operation name
            from memory.agemem import MemoryOperation
            ops_list = list(MemoryOperation)
            if action_idx < len(ops_list):
                return ops_list[action_idx].value
        except Exception:
            pass  # Fall back to heuristic
```

## Proposed Change

**File:** `/Users/rajesh/athena/training/stage2_agemem/trainer.py`, line 425.

```python
# BEFORE
        except Exception:
            pass  # Fall back to heuristic

# AFTER
        except Exception as e:
            self.logger.debug("Model-driven op selection failed, using heuristic: %s", e)
```

## Files Modified

| File | Line(s) | Change |
|------|---------|--------|
| `training/stage2_agemem/trainer.py` | 425 | `except Exception: pass` → `except Exception as e: self.logger.debug(...)` |

## Acceptance Criteria

- No bare `except Exception: pass` remains in `_select_operation_unified`.
- The exception is captured as `e` and logged at `DEBUG` level using `%s`-style formatting.
- The fallback to heuristic selection still occurs (the `except` block does not re-raise).
- All 173 tests pass, 4 skipped.

## Edge Cases & Risks

1. **Log spam**: If the model is partially loaded but the action_head forward pass fails on every step, this could produce many debug-level log lines. This is acceptable — debug logging is disabled by default and is the correct level for an expected fallback path.

2. **No behavioral change**: The heuristic fallback still runs. This is purely an observability fix.

## Test Notes

- No new tests needed. This is a one-line mechanical change.
- Run `python3 -m pytest tests/ -q` to confirm 173 passed, 4 skipped.
