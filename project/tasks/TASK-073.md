# TASK-073: Add mlx-lm backend fields to OLMoEConfig and OLMoEIntegrationConfig

## Status
- **State:** Queued
- **Priority:** 🔴 Critical
- **Depends on:** TASK-072
- **Created:** 2026-02-26

## Objective
Extend the two config dataclasses in `core/config.py` and the `OLMoEConfig` dataclass in `models/olmoe.py` with fields that control mlx-lm backend selection, so callers can opt into or out of the mlx path via configuration rather than hardcoded logic.

## Context
Current state:
- `models/olmoe.py` defines `OLMoEConfig` with fields: `model_name`, `device`, `dtype`, `load_in_8bit`, `load_in_4bit`, `max_length`, generation params, LoRA params, `use_cache`, `cache_dir`.
- `core/config.py` defines `OLMoEIntegrationConfig` with fields: `enabled`, `model_name`, `device`, `dtype`, `load_in_4bit`, `max_length`, `lora_adapter_path`.

Neither has any mlx-specific fields. Subsequent tasks (TASK-074, TASK-075, TASK-076) will use these fields to select the inference backend.

See `/Users/rajesh/athena/project/context.md` and `/Users/rajesh/athena/models/olmoe.py` and `/Users/rajesh/athena/core/config.py` for full current code.

## Scope & Constraints
- May modify: `models/olmoe.py` (OLMoEConfig dataclass only), `core/config.py` (OLMoEIntegrationConfig dataclass only)
- Must NOT modify any other class or function in either file
- Must NOT break existing dataclass field defaults — all new fields must have defaults so existing instantiation code `OLMoEConfig()` and `OLMoEIntegrationConfig()` continue to work unchanged
- Must NOT import mlx at module level in config.py (config must be importable without mlx installed)
- Public API of AthenaConfig._from_dict() must continue to parse `olmoe` YAML/JSON keys via `OLMoEIntegrationConfig(**data["olmoe"])`; new fields must therefore be valid keyword arguments

## Input
- `/Users/rajesh/athena/models/olmoe.py` — current OLMoEConfig
- `/Users/rajesh/athena/core/config.py` — current OLMoEIntegrationConfig

## Expected Output

### Changes to `models/olmoe.py` — OLMoEConfig dataclass
Add these fields after `cache_dir`:
```python
# MLX backend settings (Apple Silicon)
use_mlx: bool = True          # prefer mlx-lm when available
mlx_model_path: Optional[str] = None  # local path or HF repo for mlx weights
```

### Changes to `core/config.py` — OLMoEIntegrationConfig dataclass
Add these fields after `lora_adapter_path`:
```python
# MLX backend settings (Apple Silicon)
use_mlx: bool = True          # prefer mlx-lm when available
mlx_model_path: Optional[str] = None  # local path or HF repo for mlx weights
```

Both `mlx_model_path` fields use `Optional[str]` which is already imported in both files.

## Acceptance Criteria
- [ ] `OLMoEConfig()` instantiates without error and has `use_mlx=True` and `mlx_model_path=None` attributes
- [ ] `OLMoEIntegrationConfig()` instantiates without error and has `use_mlx=True` and `mlx_model_path=None` attributes
- [ ] `OLMoEIntegrationConfig(use_mlx=False, mlx_model_path="/some/path")` works (keyword arg construction)
- [ ] `AthenaConfig._from_dict({"olmoe": {"use_mlx": False}})` works without KeyError
- [ ] `python3 -m pytest tests/ -q` still shows 173 passed, 4 skipped

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
