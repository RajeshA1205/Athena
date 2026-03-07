# TASK-013: Run ruff check --fix on CLI and clean up lint

## Problem

Approximately 20 unused imports exist across the codebase (visible when running `ruff check .`). The project has no `[tool.ruff]` configuration, so lint rules vary depending on developer environment defaults. This creates noise in diffs and can mask real issues.

## Files to change

| File | Lines | Change |
|------|-------|--------|
| `pyproject.toml` | (new section) | Add `[tool.ruff]` configuration |
| `cli.py` | (various) | Remove unused imports flagged by ruff |
| `agents/*.py` | (various) | Remove unused imports |
| `core/*.py` | (various) | Remove unused imports |
| `evolution/*.py` | (various) | Remove unused imports |
| `trading/*.py` | (various) | Remove unused imports |
| `learning/*.py` | (various) | Remove unused imports |

## Approach

1. **Add `[tool.ruff]` configuration to `pyproject.toml`**:
   ```toml
   [tool.ruff]
   target-version = "py312"
   line-length = 120

   [tool.ruff.lint]
   select = [
       "E",    # pycodestyle errors
       "F",    # pyflakes (includes F401 unused imports)
       "W",    # pycodestyle warnings
       "I",    # isort
       "UP",   # pyupgrade
   ]
   ignore = [
       "E501",  # line too long (handled by formatter, not linter)
   ]

   [tool.ruff.lint.isort]
   known-first-party = ["core", "agents", "evolution", "trading", "learning", "memory", "communication", "models"]
   ```

2. **Run `ruff check . --fix`** to auto-fix:
   - `F401`: unused imports
   - `I001`: import sorting
   - `UP` rules: Python 3.12 syntax upgrades (e.g., `Optional[X]` -> `X | None`)

3. **Review the diff manually** before committing:
   - Ensure no `TYPE_CHECKING`-guarded imports were removed (ruff respects this, but verify).
   - Ensure no re-exports from `__init__.py` files were removed (these may appear unused locally but are part of the public API). Use `# noqa: F401` for intentional re-exports.
   - Check that `__all__` lists in `__init__.py` files align with actual exports.

4. **Run full test suite** after the auto-fix to catch any import that was "unused" locally but actually needed at runtime (e.g., imported for side effects).

5. **Do NOT enable auto-format** (`ruff format`) in this task -- that's a separate, larger change. This task is lint-only.

## Edge cases / risks

- **Re-exports in `__init__.py`**: Files like `evolution/__init__.py` explicitly import and re-export classes. Ruff may flag these as unused. Add `# noqa: F401` comments or use `__all__` to signal intent.
- **`TYPE_CHECKING` imports**: These are only used for type annotations and are not available at runtime. Ruff handles these correctly but verify.
- **Side-effect imports**: Some imports may be needed for module registration or monkey-patching. Unlikely in this codebase but review each removal.
- **`UP` rules**: `Optional[X]` -> `X | None` changes are syntactically valid in Python 3.12+ but may surprise developers unfamiliar with the new union syntax. This is a style preference -- can be deferred if controversial.

## Acceptance criteria

- [ ] `ruff check .` produces zero errors with the new configuration.
- [ ] `[tool.ruff]` section exists in `pyproject.toml` with documented rule selections.
- [ ] All `__init__.py` re-exports have `# noqa: F401` or `__all__` coverage.
- [ ] No runtime imports were removed (verified by test suite).
- [ ] Import ordering is consistent across all files.
- [ ] `pytest tests/ -q` remains green.
