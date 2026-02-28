# TASK-074: Rewrite OLMoEModel.load() to use mlx-lm as primary backend

## Status
- **State:** Queued
- **Priority:** 🔴 Critical
- **Depends on:** TASK-073
- **Created:** 2026-02-26

## Objective
Rewrite `OLMoEModel.load()` in `models/olmoe.py` so that it:
1. Attempts `mlx_lm.load()` first when `self.config.use_mlx` is True and mlx-lm is installed
2. Falls back to the existing HuggingFace `AutoModelForCausalLM` path when mlx is unavailable or `use_mlx=False`

After a successful mlx load, the model is stored internally and `self._is_loaded` is set to True, but `self.model` and `self.tokenizer` attributes follow the mlx-lm convention (model + tokenizer objects from `mlx_lm.load()`). The `MemoryActionHead` attachment logic (torch.nn.Module) must only run on the transformers path — the mlx path skips it because `encode()` (TASK-076) will handle the bridge separately.

## Context
Current `OLMoEModel.load()` (lines 88–166 of `/Users/rajesh/athena/models/olmoe.py`):
- Guards on `TRANSFORMERS_AVAILABLE`; returns False if missing
- Loads tokenizer via `AutoTokenizer.from_pretrained`
- Loads model via `AutoModelForCausalLM.from_pretrained` with optional BitsAndBytesConfig
- Calls `self._load_lora(lora_path)` if a path is given
- Attaches `MemoryActionHead` after load

New behaviour should be:
- Add module-level guard: `try: import mlx_lm; MLX_AVAILABLE = True except ImportError: MLX_AVAILABLE = False`
- `load()` checks `MLX_AVAILABLE and self.config.use_mlx`; if both true, runs mlx path
- mlx path: `self.model, self.tokenizer = mlx_lm.load(model_path)` where `model_path = self.config.mlx_model_path or self.config.model_name`
- mlx path sets `self._backend = "mlx"` and `self._is_loaded = True`; does NOT attach MemoryActionHead (action_head remains None on mlx path — GRPO training requires the transformers path)
- transformers fallback path: existing logic unchanged, sets `self._backend = "transformers"`
- If both paths fail: return False

See `/Users/rajesh/athena/models/olmoe.py` for full current code.

## Scope & Constraints
- May modify: `models/olmoe.py` — `OLMoEModel.load()` method, module-level import guard for mlx_lm, `__init__` to initialise `self._backend = None`
- Must NOT modify: `OLMoEConfig` (done in TASK-073), `MemoryActionHead`, any other method in this file
- Must NOT break the transformers path — existing code that runs with transformers installed must continue to work
- Must NOT import mlx_lm at module level unconditionally; use a try/except guard
- `self.model` and `self.tokenizer` are reused for both backends; the mlx-lm API `mlx_lm.load(path)` returns `(model, tokenizer)` — assign them to those same attributes

## Input
- `/Users/rajesh/athena/models/olmoe.py` — full current file
- mlx-lm API reference: `mlx_lm.load(model_path_or_hf_id)` returns `(model, tokenizer)`

## Expected Output
Modified `models/olmoe.py` with:
1. New module-level guard at the top:
   ```python
   try:
       import mlx_lm
       MLX_AVAILABLE = True
   except ImportError:
       MLX_AVAILABLE = False
   ```
2. `OLMoEModel.__init__` gains `self._backend: Optional[str] = None`
3. `OLMoEModel.load()` restructured with mlx-first logic as described above
4. Log messages distinguish the two paths:
   - mlx path: `self.logger.info("Loading OLMoE via mlx-lm: %s", model_path)`
   - transformers path: existing log message unchanged
   - fallback: `self.logger.error("No inference backend available (mlx-lm and transformers both missing)")`

## Acceptance Criteria
- [ ] When mlx-lm is installed and `use_mlx=True` (default), `load()` calls `mlx_lm.load()` and sets `self._backend = "mlx"` and `self._is_loaded = True`
- [ ] When mlx-lm is installed and `use_mlx=False`, `load()` uses the transformers path (or returns False if transformers not installed)
- [ ] When mlx-lm is NOT installed, `load()` falls through to transformers path
- [ ] `self.action_head` is None after an mlx-path load (not attached)
- [ ] `self.action_head` is a `MemoryActionHead` instance after a transformers-path load (existing behaviour unchanged)
- [ ] `python3 -m pytest tests/ -q` shows 173 passed, 4 skipped

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
