# TASK-028: Align AgentStateEncoder default latent_dim to 512

## Summary
`AgentStateEncoder.__init__` defaults `latent_dim=256` while `LatentSpace` (and its internal `LatentEncoder`/`LatentDecoder`) defaults to `latent_dim=512`. If both are instantiated without explicit arguments, the encoder produces 256-dimensional vectors that cannot be consumed by a 512-dimensional latent space, causing a shape mismatch at runtime. Changing the `AgentStateEncoder` default to 512 aligns the two without breaking any caller that passes `latent_dim` explicitly.

## Current State

**File:** `communication/encoder.py` (line 53)

```python
def __init__(
    self,
    latent_dim: int = 256,    # <-- mismatched with LatentSpace default of 512
    input_dim: int = 512,
    hidden_dims: Optional[list] = None,
    text_embed_dim: int = 768,
    dropout: float = 0.1,
    device: Optional[str] = None,
):
```

**File:** `communication/latent_space.py` (line 58)

```python
def __init__(
    self, latent_dim: int = 512, ...   # LatentEncoder default
```

```python
class LatentSpace:
    def __init__(self, latent_dim: int = 512, ...  # LatentSpace default
```

## Proposed Change

**File:** `communication/encoder.py`, line 53:

```python
def __init__(
    self,
    latent_dim: int = 512,    # Changed from 256 to match LatentSpace default
    input_dim: int = 512,
    ...
):
```

One-line change. No other modifications needed.

## Files Modified

- `communication/encoder.py`
  - Line 53: `latent_dim: int = 256` → `latent_dim: int = 512`

## Acceptance Criteria

- [ ] `AgentStateEncoder().latent_dim == 512`
- [ ] `AgentStateEncoder().latent_dim == LatentSpace().latent_dim` (both 512)
- [ ] All existing tests pass

## Edge Cases & Risks

- **Tests that assert latent vector shape `== 256`:** Any test that creates `AgentStateEncoder()` without an explicit `latent_dim` and checks the output dimension will now expect 512 instead of 256. Search for such assertions before merging.
- **Memory:** A 512-dim latent space uses more memory than 256. Existing users who rely on the 256 default for efficiency should be unaffected since they can still pass `latent_dim=256` explicitly.
- **`input_dim` default is also 512:** The `input_dim=512` default (the MLP input size) is unrelated to `latent_dim` and does not need changing.

## Test Notes

- `tests/test_communication.py` has encoder tests that are skipped when torch is not installed. When run with torch, verify shape of `encode_agent_state()` output changes from 256 → 512.
- Add assertion: `assert AgentStateEncoder().latent_dim == LatentSpace().latent_dim`.
