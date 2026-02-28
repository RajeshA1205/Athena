# TASK-073: Add mlx-lm Backend Fields to Config Dataclasses

## Summary

Add two fields (`use_mlx: bool`, `mlx_model_path: Optional[str]`) to `OLMoEConfig` in `models/olmoe.py` and `OLMoEIntegrationConfig` in `core/config.py`. Both fields have defaults so all existing instantiation code remains unchanged. No mlx import at module level. Pure data-only change -- no logic.

## Current State

**File:** `models/olmoe.py`, lines 27-53 -- `OLMoEConfig` dataclass ends at `cache_dir`:
```python
@dataclass
class OLMoEConfig:
    ...
    use_cache: bool = True
    cache_dir: Optional[str] = None
```

**File:** `core/config.py`, lines 93-101 -- `OLMoEIntegrationConfig` dataclass ends at `lora_adapter_path`:
```python
@dataclass
class OLMoEIntegrationConfig:
    ...
    lora_adapter_path: Optional[str] = None
```

`Optional` is already imported in both files.

## Proposed Change

### 1. `models/olmoe.py` -- OLMoEConfig (after line 53)

```python
    # MLX backend settings (Apple Silicon)
    use_mlx: bool = True          # prefer mlx-lm when available
    mlx_model_path: Optional[str] = None  # local path or HF repo for mlx weights
```

### 2. `core/config.py` -- OLMoEIntegrationConfig (after line 101)

```python
    # MLX backend settings (Apple Silicon)
    use_mlx: bool = True          # prefer mlx-lm when available
    mlx_model_path: Optional[str] = None  # local path or HF repo for mlx weights
```

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `models/olmoe.py` | after 53 | Add `use_mlx` and `mlx_model_path` fields to `OLMoEConfig` |
| `core/config.py` | after 101 | Add `use_mlx` and `mlx_model_path` fields to `OLMoEIntegrationConfig` |

## Acceptance Criteria

- `OLMoEConfig()` has `use_mlx=True` and `mlx_model_path=None`.
- `OLMoEIntegrationConfig()` has `use_mlx=True` and `mlx_model_path=None`.
- `OLMoEIntegrationConfig(use_mlx=False, mlx_model_path="/some/path")` works.
- `AthenaConfig._from_dict({"olmoe": {"use_mlx": False}})` works without error.
- No mlx import in either file.
- 173 tests pass, 4 skipped.

## Edge Cases & Risks

1. **`_from_dict` passthrough**: `OLMoEIntegrationConfig(**data["olmoe"])` uses `**kwargs`, so any new field with a default is automatically accepted as a keyword arg. No change needed in `_from_dict`.
2. **`to_dict` / `asdict`**: Both dataclasses are converted via `dataclasses.asdict` in `AthenaConfig.to_dict()`. New fields will appear in YAML/JSON output automatically.
3. **No mlx dependency**: These are plain `bool` and `Optional[str]` fields -- no mlx import needed.

## Test Notes

- No new tests required for config fields. Existing config tests (if any) instantiate with defaults and will pass.
- Run `python3 -m pytest tests/ -q` -- expect 173 passed, 4 skipped.
