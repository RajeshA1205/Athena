# TASK-061: Convert Remaining f-string Logger Calls in olmoe.py and grpo.py

## Summary

TASK-051 converted f-string logger calls in `finetune.py` and `trainer.py`. TASK-045 covered the five agent files. But `models/olmoe.py` and `training/stage2_agemem/grpo.py` still contain f-string logger calls. Convert all remaining instances to lazy `%s`-style formatting for consistency and to avoid eager string evaluation when the log level is disabled.

## Current State

### File: `models/olmoe.py`

**Line 103:**
```python
self.logger.info(f"Loading OLMoE model: {self.config.model_name}")
```

**Line 161:**
```python
self.logger.info(f"OLMoE model loaded on {self.device}")
```

**Line 165:**
```python
self.logger.error(f"Failed to load OLMoE model: {e}")
```

**Line 177:**
```python
self.logger.info(f"Loaded LoRA adapter from {lora_path}")
```

**Line 180:**
```python
self.logger.error(f"Failed to load LoRA: {e}")
```

**Line 221:**
```python
self.logger.error(f"Failed to prepare for training: {e}")
```

**Line 386:**
```python
self.logger.info(f"LoRA adapter saved to {path}")
```

**Line 389:**
```python
self.logger.error(f"Failed to save LoRA: {e}")
```

### File: `training/stage2_agemem/grpo.py`

**Line 141:**
```python
self.logger.error(f"GRPO setup failed: {e}")
```

## Proposed Change

### olmoe.py changes

| Line | Before | After |
|------|--------|-------|
| 103 | `self.logger.info(f"Loading OLMoE model: {self.config.model_name}")` | `self.logger.info("Loading OLMoE model: %s", self.config.model_name)` |
| 161 | `self.logger.info(f"OLMoE model loaded on {self.device}")` | `self.logger.info("OLMoE model loaded on %s", self.device)` |
| 165 | `self.logger.error(f"Failed to load OLMoE model: {e}")` | `self.logger.error("Failed to load OLMoE model: %s", e)` |
| 177 | `self.logger.info(f"Loaded LoRA adapter from {lora_path}")` | `self.logger.info("Loaded LoRA adapter from %s", lora_path)` |
| 180 | `self.logger.error(f"Failed to load LoRA: {e}")` | `self.logger.error("Failed to load LoRA: %s", e)` |
| 221 | `self.logger.error(f"Failed to prepare for training: {e}")` | `self.logger.error("Failed to prepare for training: %s", e)` |
| 386 | `self.logger.info(f"LoRA adapter saved to {path}")` | `self.logger.info("LoRA adapter saved to %s", path)` |
| 389 | `self.logger.error(f"Failed to save LoRA: {e}")` | `self.logger.error("Failed to save LoRA: %s", e)` |

### grpo.py changes

| Line | Before | After |
|------|--------|-------|
| 141 | `self.logger.error(f"GRPO setup failed: {e}")` | `self.logger.error("GRPO setup failed: %s", e)` |

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `models/olmoe.py` | 103, 161, 165, 177, 180, 221, 386, 389 | 8 f-string logger calls → `%s`-style |
| `training/stage2_agemem/grpo.py` | 141 | 1 f-string logger call → `%s`-style |

## Acceptance Criteria

- Zero f-string logger calls remain in `models/olmoe.py`.
- Zero f-string logger calls remain in `training/stage2_agemem/grpo.py`.
- All logger calls use `%s`-style with positional arguments.
- Log output is semantically identical (same messages, same data).
- All 173 tests pass, 4 skipped.

## Edge Cases & Risks

1. **No behavioral change**: Purely a style/performance fix. Log messages appear identical to users.
2. **Line numbers may shift**: If TASK-056 or TASK-057 add import lines to `grpo.py` or `trainer.py`, line numbers in those files shift. The implementer should match by content, not absolute line number.

## Test Notes

- No new tests required.
- Verify with: `grep -n 'logger\.\(info\|error\|warning\|debug\)(f"' models/olmoe.py training/stage2_agemem/grpo.py` — expect zero matches.
- Run `python3 -m pytest tests/ -q` to confirm 173 passed, 4 skipped.
