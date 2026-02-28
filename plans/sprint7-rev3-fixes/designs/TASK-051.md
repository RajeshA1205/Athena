# TASK-051: Convert f-string Logger Calls to Lazy %s Formatting

## Summary

Eight logger calls across `finetune.py` (3) and `trainer.py` (5) use f-string interpolation instead of `%s`-style lazy formatting. F-strings are evaluated eagerly even when the log level is disabled, wasting CPU on string formatting. Convert all to `%s`-style.

## Current State

### File: `/Users/rajesh/athena/training/stage1_finetune/finetune.py`

**Line 75:**
```python
self.logger.info(f"Loading model: {self.config.model_name}")
```

**Line 124:**
```python
self.logger.error(f"Setup failed: {e}")
```

**Line 264:**
```python
self.logger.info(f"Model saved to {save_path}")
```

### File: `/Users/rajesh/athena/training/stage2_agemem/trainer.py`

**Line 108:**
```python
self.logger.info(f"AgeMem trainer ready, starting at {self.current_stage.name}")
```

**Lines 148-149:**
```python
self.logger.info(
    f"Step {self._step_count}, Stage {self.current_stage.name}, "
    f"Loss: {metrics['total_loss']:.4f}"
)
```

**Line 201:**
```python
self.logger.info(f"Transitioning from {self.current_stage.name} to {new_stage.name}")
```

**Line 326:**
```python
self.logger.info(f"Checkpoint saved: {path}")
```

**Line 331:**
```python
self.logger.info(f"Checkpoint loaded: {path}")
```

## Proposed Change

### finetune.py changes

**Line 75:**
```python
self.logger.info("Loading model: %s", self.config.model_name)
```

**Line 124:**
```python
self.logger.error("Setup failed: %s", e)
```

**Line 264:**
```python
self.logger.info("Model saved to %s", save_path)
```

### trainer.py changes

**Line 108:**
```python
self.logger.info("AgeMem trainer ready, starting at %s", self.current_stage.name)
```

**Lines 148-149:**
```python
self.logger.info(
    "Step %d, Stage %s, Loss: %.4f",
    self._step_count, self.current_stage.name, metrics["total_loss"]
)
```

**Line 201:**
```python
self.logger.info("Transitioning from %s to %s", self.current_stage.name, new_stage.name)
```

**Line 326:**
```python
self.logger.info("Checkpoint saved: %s", path)
```

**Line 331:**
```python
self.logger.info("Checkpoint loaded: %s", path)
```

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `/Users/rajesh/athena/training/stage1_finetune/finetune.py` | 75 | f-string -> `%s` |
| `/Users/rajesh/athena/training/stage1_finetune/finetune.py` | 124 | f-string -> `%s` |
| `/Users/rajesh/athena/training/stage1_finetune/finetune.py` | 264 | f-string -> `%s` |
| `/Users/rajesh/athena/training/stage2_agemem/trainer.py` | 108 | f-string -> `%s` |
| `/Users/rajesh/athena/training/stage2_agemem/trainer.py` | 148-149 | f-string -> `%d`, `%s`, `%.4f` |
| `/Users/rajesh/athena/training/stage2_agemem/trainer.py` | 201 | f-string -> `%s` |
| `/Users/rajesh/athena/training/stage2_agemem/trainer.py` | 326 | f-string -> `%s` |
| `/Users/rajesh/athena/training/stage2_agemem/trainer.py` | 331 | f-string -> `%s` |

## Acceptance Criteria

- Zero f-string logger calls remain in `finetune.py`.
- Zero f-string logger calls remain in `trainer.py`.
- All logger calls use `%s`/`%d`/`%.4f` lazy formatting with positional arguments.
- Log output is identical to before (same message content, same formatting).
- All 171 tests pass, 6 skipped.

## Edge Cases & Risks

1. **Loss formatting**: The `:.4f` format specifier in the f-string must be converted to `%.4f` in the format string. The value `metrics["total_loss"]` is a float, so `%.4f` is correct.

2. **`metrics['total_loss']` key access**: The f-string uses single quotes inside double-quoted f-string. The `%s`-style version passes it as a positional argument: `metrics["total_loss"]`. Quote style doesn't matter for dict access.

3. **No behavioral change**: This is purely a performance/style fix. The log messages appear identical to end users. The only difference is that the string formatting is deferred until the logger confirms the level is enabled.

4. **Note on trainer.py lines**: If TASK-049 modifies `trainer.py` (making methods async), the line numbers for logging calls may shift. The implementer should apply these changes based on the content pattern (the f-string text), not absolute line numbers.

## Test Notes

- No new tests needed. This is a mechanical text transformation.
- Verify by running `grep -n 'logger\.\(info\|error\|warning\|debug\)(f"' finetune.py trainer.py` and confirming zero matches.
- Run `python3 -m pytest tests/ -q` to confirm all tests pass.
