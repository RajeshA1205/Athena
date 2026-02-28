# TASK-072: Install mlx-lm and mlx packages for Apple Silicon inference

## Status
- **State:** Queued
- **Priority:** 🔴 Critical
- **Depends on:** None
- **Created:** 2026-02-26

## Objective
Install the `mlx` and `mlx-lm` Python packages into the ATHENA environment so that subsequent tasks can import them. Verify that `import mlx` and `import mlx_lm` both succeed. Confirm that `mlx_lm.load` and `mlx_lm.generate` are importable symbols.

## Context
ATHENA is running on Apple Silicon (Darwin/macOS). The existing OLMoE inference backend uses HuggingFace `transformers` + `bitsandbytes`, neither of which are installed (and bitsandbytes requires CUDA so cannot work on Mac). The decision is to migrate primary inference to `mlx-lm` (Apple's MLX framework), which is optimised for Apple Silicon's unified memory and does not require CUDA.

See `/Users/rajesh/athena/project/context.md` for overall architecture.

## Scope & Constraints
- May run `pip install` commands (mlx, mlx-lm)
- Must NOT modify any source files in this task — installation only
- Must NOT install bitsandbytes, CUDA packages, or anything CUDA-specific
- If `pip install mlx-lm` also installs `mlx` as a transitive dependency that is fine — do not install separately if already pulled in

## Input
- The Python interpreter / pip executable in the ATHENA project environment
- `/Users/rajesh/athena/` (project root for reference)

## Expected Output
- `mlx` and `mlx-lm` packages present in the Python environment
- A brief verification script output showing that the following imports succeed:
  ```
  import mlx
  import mlx.core
  import mlx_lm
  from mlx_lm import load as mlx_load, generate as mlx_generate
  ```
- A plain-text note (printed to stdout, not written to a file) listing the installed versions of `mlx` and `mlx-lm`

## Acceptance Criteria
- [ ] `pip install mlx-lm` completes without error
- [ ] `python3 -c "import mlx; print(mlx.__version__)"` exits 0 and prints a version string
- [ ] `python3 -c "from mlx_lm import load, generate; print('ok')"` exits 0 and prints `ok`
- [ ] No CUDA-specific packages (bitsandbytes, nvidia-*) are installed as side effects
- [ ] Existing test suite still passes: `python3 -m pytest tests/ -q` shows 173 passed, 4 skipped (package install must not break existing imports)

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
