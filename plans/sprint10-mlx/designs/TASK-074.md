# TASK-074: Rewrite OLMoEModel.load() to Use mlx-lm as Primary Backend

## Summary

Restructure `OLMoEModel.load()` to attempt `mlx_lm.load()` first (when `config.use_mlx=True` and mlx-lm is installed), falling back to the existing HuggingFace transformers path. Add a module-level `MLX_AVAILABLE` import guard and a `self._backend` attribute to track which backend was loaded. On the mlx path, `self.action_head` remains `None` (GRPO training requires transformers).

## Current State

**File:** `models/olmoe.py`

**Lines 12-17 -- existing import guards:**
```python
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
```

**Lines 71-86 -- `__init__`:**
```python
def __init__(self, config: Optional[OLMoEConfig] = None):
    self.config = config or OLMoEConfig()
    self.logger = logging.getLogger("athena.models.olmoe")
    self.model = None
    self.tokenizer = None
    self.device = None
    self._is_loaded = False
    self._has_lora = False
    self.action_head = None
```

**Lines 88-166 -- `load()`:**
- Guards on `TRANSFORMERS_AVAILABLE`; returns False if missing
- Loads tokenizer, model, optional LoRA, attaches `MemoryActionHead`
- Sets `self._is_loaded = True`

## Proposed Change

### 1. New module-level import guard (after line 17)

```python
try:
    import mlx_lm
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
```

### 2. `__init__` -- add `_backend` attribute (after line 86)

```python
    self._backend: Optional[str] = None
```

### 3. `load()` -- restructure with mlx-first logic

Replace the current `load()` body with:

```python
def load(self, lora_path: Optional[str] = None) -> bool:
    """
    Load the OLMoE model.

    Attempts mlx-lm first (when use_mlx=True and mlx-lm is installed),
    then falls back to the HuggingFace transformers backend.

    Args:
        lora_path: Optional path to LoRA adapter weights (transformers only)

    Returns:
        True if loaded successfully
    """
    # --- mlx-lm path ---
    if MLX_AVAILABLE and self.config.use_mlx:
        try:
            model_path = self.config.mlx_model_path or self.config.model_name
            self.logger.info("Loading OLMoE via mlx-lm: %s", model_path)
            self.model, self.tokenizer = mlx_lm.load(model_path)
            self._backend = "mlx"
            self._is_loaded = True
            self.device = "mlx"
            # action_head intentionally left as None on mlx path.
            # GRPO training requires the transformers backend.
            self.logger.info("OLMoE model loaded via mlx-lm")
            return True
        except Exception as e:
            self.logger.warning(
                "mlx-lm load failed, falling back to transformers: %s", e
            )

    # --- transformers path (existing logic, unchanged) ---
    if not TRANSFORMERS_AVAILABLE:
        self.logger.error(
            "No inference backend available "
            "(mlx-lm and transformers both missing)"
        )
        return False

    try:
        self.logger.info("Loading OLMoE model: %s", self.config.model_name)

        # ... (entire existing transformers load block unchanged) ...

        self._backend = "transformers"   # <-- NEW: set backend tag
        self._is_loaded = True

        # Attach MemoryActionHead (existing logic, unchanged)
        ...

        self.logger.info("OLMoE model loaded on %s", self.device)
        return True

    except Exception as e:
        self.logger.error("Failed to load OLMoE model: %s", e)
        return False
```

Key structural points:
- The mlx path is a **self-contained try/except block** before the transformers path. If it fails, execution falls through to transformers.
- `self._backend = "transformers"` is set inside the existing try block, just before `self._is_loaded = True`.
- `MemoryActionHead` attachment only runs in the transformers path (already guarded by `if TRANSFORMERS_AVAILABLE`).
- `lora_path` is only used in the transformers path. The mlx path ignores it (mlx-lm does not support PEFT LoRA).

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `models/olmoe.py` | after 17 | Add `MLX_AVAILABLE` import guard |
| `models/olmoe.py` | ~86 | Add `self._backend: Optional[str] = None` to `__init__` |
| `models/olmoe.py` | 88-166 | Restructure `load()` with mlx-first logic |

## Acceptance Criteria

- `MLX_AVAILABLE` and `mlx_lm` import guard exist at module level.
- When mlx-lm is installed and `use_mlx=True`, `load()` calls `mlx_lm.load()` and sets `_backend="mlx"`, `_is_loaded=True`.
- When mlx-lm is installed and `use_mlx=False`, `load()` skips to transformers path.
- When mlx-lm is NOT installed (`MLX_AVAILABLE=False`), `load()` falls through to transformers path.
- `self.action_head` is `None` after mlx-path load.
- `self.action_head` is a `MemoryActionHead` instance after transformers-path load.
- If both backends fail, `load()` returns `False`.
- 173 tests pass, 4 skipped.

## Edge Cases & Risks

1. **mlx-lm load failure**: If `mlx_lm.load()` raises (e.g., model not converted to mlx format), the code falls back to transformers. The `self.logger.warning` makes this visible. Risk: users may not notice they are on the slower backend. Mitigation: the log message is explicit.
2. **`lora_path` ignored on mlx path**: mlx-lm does not support PEFT LoRA adapters. If a user passes `lora_path` and mlx loads successfully, LoRA is silently skipped. This is acceptable for Sprint 10 -- document as known limitation.
3. **`self.device = "mlx"`**: The mlx backend does not use torch devices. Setting `self.device = "mlx"` prevents `_get_device()` from being called and avoids torch.cuda/mps detection on the mlx path. `get_info()` will show `device: mlx`.
4. **`self.model` type difference**: On mlx path, `self.model` is an mlx model object (not `torch.nn.Module`). Code that calls `self.model.parameters()` (e.g., `OLMoEModel.parameters()`) will fail on the mlx path. This is acceptable: the `parameters()` method is only used by GRPO training, which requires transformers. If this becomes a problem, `parameters()` can be guarded in a follow-on task.
5. **`self.tokenizer` type difference**: On mlx path, the tokenizer is from `mlx_lm` (typically a `SentencePieceProcessor` or similar). `generate()` and `encode()` (TASK-075, TASK-076) must use the correct API for each backend.

## Test Notes

- Existing tests mock or skip model loading. The mlx-lm import guard returns `MLX_AVAILABLE=False` in CI (mlx-lm not installed), so all existing tests will follow the transformers path unchanged.
- No new tests required in this task (mocking `mlx_lm.load` is deferred to integration tests).
- Run `python3 -m pytest tests/ -q` -- expect 173 passed, 4 skipped.
