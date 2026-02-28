# TASK-036: Fix duplicate logging handlers in setup_logging

## Summary
`setup_logging()` in `core/utils.py` unconditionally creates and attaches a `StreamHandler` (and optionally a `FileHandler`) every time it is called, with no check for existing handlers. In a typical application startup where multiple modules call `setup_logging()`, this results in duplicate (or triplicate, etc.) log output for every log message. The fix is a one-line idempotency guard that returns early if the logger is already configured.

## Current State

**File:** `core/utils.py` (lines 30–49)

```python
def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    ...
    logger = logging.getLogger("athena")
    logger.setLevel(getattr(logging, level.upper()))

    # Console handler
    console_handler = logging.StreamHandler()         # always added — no guard
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)  # always added — no guard
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)

    return logger
```

## Proposed Change

Add an idempotency guard at the top of the function body, after the `logger` is retrieved:

```python
def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    ...
    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)s [%(name)s] %(message)s"

    logger = logging.getLogger("athena")
    logger.setLevel(getattr(logging, level.upper()))

    # Guard: only add handlers if none exist yet
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)

    return logger
```

## Files Modified

- `core/utils.py`
  - Lines 35–41: add `if logger.handlers: return logger` after `logger.setLevel(...)`

## Acceptance Criteria

- [ ] Calling `setup_logging()` twice produces no duplicate handlers (`len(logger.handlers) == 1` after two calls)
- [ ] Log messages appear exactly once per call site
- [ ] The `log_file` handler is still added on first call when specified
- [ ] All existing tests pass

## Edge Cases & Risks

- **Level change after first call:** The `if logger.handlers: return logger` guard returns early after setting the level but before adding handlers. If the first call uses `level="INFO"` and a second call uses `level="DEBUG"`, the level will be updated but no new handlers added. This is correct behaviour — only the level changes.
- **`log_file` on second call:** If the first call has no `log_file` and a second call specifies one, the guard returns early and the file handler is never added. This is acceptable — callers should configure file logging on the first call. Document this in the docstring.
- **Test isolation:** Tests that call `setup_logging()` in fixtures may accumulate handlers across tests if the logger is not reset. Use `logger.handlers.clear()` in test teardown if this becomes an issue.

## Test Notes

- Add test: call `setup_logging()` twice, assert `len(logging.getLogger("athena").handlers) == 1`.
- Existing tests that call `setup_logging()` should not be affected.
