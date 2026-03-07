# TASK-009: Add mypy configuration and fix type annotation gaps

## Problem

The project has no `mypy.ini`, `pyproject.toml [tool.mypy]` section, or `py.typed` marker. `AthenaConfig._from_dict()` (line 155-184 in `core/config.py`) uses `**data["model"]` with no key validation, which will raise `TypeError` at runtime if the config file contains unexpected keys. There are approximately 20 annotation issues across the codebase (bare `Dict`, missing return types, `Any` overuse).

## Files to change

| File | Lines | Change |
|------|-------|--------|
| `pyproject.toml` | (new section) | Add `[tool.mypy]` configuration |
| `core/config.py` | 155-184 | Add key validation/filtering in `_from_dict()` before `**` unpacking |
| `core/config.py` | (various) | Fix annotation gaps |
| `agents/*.py` | (various) | Fix bare `Dict` annotations to `Dict[str, Any]` |
| `evolution/workflow_discovery.py` | 267 | `interaction_pattern: Dict` -> `Dict[str, Any]` |
| `agents/coordinator.py` | 385, 469 | `recommendations: Dict[str, Dict]` -> `Dict[str, Dict[str, Any]]` |
| `agents/coordinator.py` | 545-546 | `resolved: Dict, risk_assessment: Dict` -> typed |

## Approach

1. **Add `[tool.mypy]` to `pyproject.toml`**:
   ```toml
   [tool.mypy]
   python_version = "3.12"
   warn_return_any = true
   warn_unused_configs = true
   disallow_untyped_defs = false  # start permissive, tighten later
   ignore_missing_imports = true  # third-party stubs not yet available
   ```

2. **Fix `_from_dict()` key validation** in `core/config.py`:
   ```python
   if "model" in data:
       valid_keys = {f.name for f in fields(ModelConfig)}
       filtered = {k: v for k, v in data["model"].items() if k in valid_keys}
       config.model = ModelConfig(**filtered)
   ```
   Apply the same pattern to all sub-config unpacking (`memory`, `communication`, `evolution`, `learning`, `trading`, `olmoe`, `agents`).
   Log a warning for unexpected keys so config typos are visible.

3. **Fix bare `Dict` annotations** across the codebase:
   - `Dict` -> `Dict[str, Any]` where the value type is mixed
   - `Dict[str, Dict]` -> `Dict[str, Dict[str, Any]]`
   - Add missing return type annotations on private methods

4. **Add `py.typed` marker** file at the package root (empty file, signals PEP 561 compliance).

5. **Run `mypy core/ agents/ evolution/ trading/`** and fix reported issues iteratively.

## Edge cases / risks

- **Key filtering silently drops unknown keys**: A config with a typo like `"modle"` instead of `"model"` at the top level is already silently ignored. Sub-key typos like `"devce"` inside `"model"` would now be dropped with a warning rather than crashing. This is safer behavior.
- **dataclass `fields()` introspection**: Requires `from dataclasses import fields`. This is already available since all config classes are dataclasses.
- **`ignore_missing_imports = true`**: Necessary because `graphiti_core`, `mlx`, `neo4j`, etc. don't ship type stubs. Can be tightened per-module later.
- **`disallow_untyped_defs = false`**: Starting permissive avoids a huge diff. Can be tightened in a follow-up task.

## Acceptance criteria

- [ ] `mypy core/ agents/` runs without errors (warnings are OK for now).
- [ ] `_from_dict()` logs warnings for unexpected config keys instead of crashing.
- [ ] `_from_dict()` with extra keys in any sub-config does not raise `TypeError`.
- [ ] No bare `Dict` annotations remain in `core/`, `agents/`, `evolution/`.
- [ ] `py.typed` marker file exists at package root.
- [ ] `pytest tests/ -q` remains green.
