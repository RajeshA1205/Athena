# TASK-052: Parse olmoe Key in AthenaConfig._from_dict()

## Summary

`AthenaConfig._from_dict()` parses all component config keys (`model`, `memory`, `communication`, `evolution`, `learning`, `trading`, `agents`) but skips `olmoe`. The `olmoe` field exists on `AthenaConfig` (line 125) as `OLMoEIntegrationConfig` but is always left at its default when loading from YAML/JSON.

## Current State

**File:** `/Users/rajesh/athena/core/config.py`, lines 150-178

```python
@classmethod
def _from_dict(cls, data: Dict[str, Any]) -> "AthenaConfig":
    """Create config from dictionary."""
    config = cls()

    if "model" in data:
        config.model = ModelConfig(**data["model"])
    if "memory" in data:
        config.memory = MemoryConfig(**data["memory"])
    if "communication" in data:
        config.communication = CommunicationConfig(**data["communication"])
    if "evolution" in data:
        config.evolution = EvolutionConfig(**data["evolution"])
    if "learning" in data:
        config.learning = LearningConfig(**data["learning"])
    if "trading" in data:
        config.trading = TradingConfig(**data["trading"])
    if "agents" in data:
        config.agents = {
            name: AgentConfig(**cfg)
            for name, cfg in data["agents"].items()
        }

    # System settings
    for key in ["log_level", "checkpoint_dir", "data_dir", "seed"]:
        if key in data:
            setattr(config, key, data[key])

    return config
```

The `olmoe` key is missing between the `trading` and `agents` blocks.

`OLMoEIntegrationConfig` is defined at lines 93-101:

```python
@dataclass
class OLMoEIntegrationConfig:
    enabled: bool = False
    model_name: str = "allenai/OLMoE-1B-7B-0924"
    device: str = "auto"
    dtype: str = "float16"
    load_in_4bit: bool = False
    max_length: int = 2048
    lora_adapter_path: Optional[str] = None
```

## Proposed Change

Modify `/Users/rajesh/athena/core/config.py` only.

Add two lines after the `trading` block (after line 166) and before the `agents` block (line 167):

```python
    if "trading" in data:
        config.trading = TradingConfig(**data["trading"])
    if "olmoe" in data:
        config.olmoe = OLMoEIntegrationConfig(**data["olmoe"])
    if "agents" in data:
```

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `/Users/rajesh/athena/core/config.py` | After line 166 | Add `if "olmoe" in data:` block |

## Acceptance Criteria

- A YAML config with an `olmoe:` section correctly populates `config.olmoe` with the provided values.
- Example: `{"olmoe": {"enabled": true, "load_in_4bit": true}}` results in `config.olmoe.enabled == True` and `config.olmoe.load_in_4bit == True`, with all other fields at defaults.
- A config without an `olmoe` key produces the default `OLMoEIntegrationConfig()` (enabled=False).
- `to_dict()` round-trips correctly: `AthenaConfig._from_dict(config.to_dict())` preserves olmoe settings.
- All 171 tests pass, 6 skipped.

## Edge Cases & Risks

1. **Unknown keys in olmoe dict**: If the YAML contains keys not in `OLMoEIntegrationConfig` (e.g., a typo like `"devce"`), the `**data["olmoe"]` unpacking will raise `TypeError: __init__() got an unexpected keyword argument`. This is consistent with how all other config sections behave (they all use `**data[key]` unpacking). No special handling needed.

2. **Partial olmoe dict**: If only some fields are provided (e.g., `{"olmoe": {"enabled": true}}`), the remaining fields get their defaults from the dataclass. This is correct behavior.

3. **`to_dict()` consistency**: The existing `to_dict()` uses `dataclasses.asdict(self)`, which will include the `olmoe` field. The round-trip `_from_dict(config.to_dict())` will now correctly parse it back. Before this fix, the olmoe data in `to_dict()` output was silently ignored by `_from_dict()`.

## Test Notes

- Simple unit test: construct a dict with `"olmoe": {"enabled": True, "load_in_4bit": True}`, call `AthenaConfig._from_dict(data)`, assert `config.olmoe.enabled is True` and `config.olmoe.load_in_4bit is True`.
- Round-trip test: create config, set olmoe values, call `to_dict()`, then `_from_dict()`, assert olmoe values preserved.
- Existing tests pass unchanged.
