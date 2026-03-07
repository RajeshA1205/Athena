# TASK-007: Make MLX dependencies platform-conditional in pyproject.toml

## Problem

`pyproject.toml` lines 8-10 list `mlx>=0.30.6`, `mlx-lm>=0.30.7`, and `mlx-metal>=0.30.6` as hard dependencies. These packages only exist for macOS/Apple Silicon and cause `pip install` / `uv sync` to fail on Linux and Windows CI environments. Additionally, `main.py` registers `signal.SIGTERM` (line 191) which is available on Unix but not on Windows, though this is a secondary concern.

## Files to change

| File | Lines | Change |
|------|-------|--------|
| `pyproject.toml` | 8-10 | Move `mlx`, `mlx-lm`, `mlx-metal` from `dependencies` to a new `[project.optional-dependencies] mlx` group, or use platform markers |
| `pyproject.toml` | (new) | Add platform markers: `sys_platform == "darwin"` |
| `main.py` | 191 | Guard `signal.SIGTERM` registration with platform check |

## Approach

### Option A: Platform markers (preferred)

Use PEP 508 environment markers to keep MLX in `dependencies` but only install on macOS:

```toml
dependencies = [
    # Inference -- Apple Silicon only
    "mlx>=0.30.6; sys_platform == 'darwin'",
    "mlx-lm>=0.30.7; sys_platform == 'darwin'",
    "mlx-metal>=0.30.6; sys_platform == 'darwin'",
    ...
]
```

This keeps the install experience simple on macOS (no extras needed) while allowing Linux/Windows installs to succeed.

### Option B: Optional dependency group

Move MLX to `[project.optional-dependencies]`:

```toml
[project.optional-dependencies]
mlx = [
    "mlx>=0.30.6",
    "mlx-lm>=0.30.7",
    "mlx-metal>=0.30.6",
]
```

Then install with `pip install -e ".[mlx]"` on macOS. This requires users to know about the extra.

### Recommendation: Option A

Platform markers are transparent and require no user action.

### SIGTERM guard in main.py

```python
if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, _shutdown)
```

`signal.SIGINT` is available on all platforms; `SIGTERM` is Unix-only. The `hasattr` guard is the standard pattern.

### Runtime MLX import guard

Verify that `models/olmoe.py` and `core/config.py` already handle missing `mlx` gracefully. If they do `import mlx` at module level without a try/except, add a lazy import or conditional import guard:

```python
try:
    import mlx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
```

## Edge cases / risks

- `mlx-metal` is a Metal GPU backend; it may not be needed even on Intel Macs. The `sys_platform == 'darwin'` marker is broad enough; Apple Silicon detection would require a more complex marker (`platform_machine == 'arm64'`) but `mlx` itself handles this at runtime.
- CI pipelines on Linux will now skip MLX installation. Any tests that import MLX directly will need `pytest.importorskip("mlx")` guards.
- The `OLMoEIntegrationConfig.use_mlx` flag (line 104 in `core/config.py`) defaults to `True` but the OLMoE model loader should already fall back to transformers when MLX is unavailable.

## Acceptance criteria

- [ ] `pip install -e .` succeeds on Linux without MLX packages.
- [ ] `pip install -e .` on macOS still installs MLX packages automatically.
- [ ] `main.py` runs on Windows without `SIGTERM` `AttributeError`.
- [ ] All existing tests pass on macOS.
- [ ] `pytest tests/ -q` remains green.
