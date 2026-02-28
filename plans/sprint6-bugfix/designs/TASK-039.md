# TASK-039: Replace weak ID generation with secrets module

## Summary
Three ID generation functions use `random.random()` as their entropy source, feeding it into MD5 or SHA-256 to produce an identifier. `random.random()` uses the Mersenne Twister PRNG, which is not cryptographically secure and produces predictable outputs if the seed is known or can be brute-forced. While these IDs are not used for security purposes (no authentication tokens), using a cryptographically strong source (`secrets.token_hex`) is straightforward, produces better collision resistance, and eliminates the dependency on the wall-clock timestamp as the primary uniqueness source.

## Current State

**File:** `core/utils.py` (lines 106–111)

```python
def generate_id(prefix: Optional[str] = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    random_part = hashlib.md5(str(random.random()).encode()).hexdigest()[:8]
    if prefix:
        return f"{prefix}_{timestamp}_{random_part}"
    return f"{timestamp}_{random_part}"
```

**File:** `memory/operations.py` (lines 411–416)

```python
def _generate_id(self) -> str:
    """Generate unique ID."""
    import hashlib
    import random
    timestamp = datetime.now().isoformat()
    return hashlib.sha256(f"{timestamp}:{random.random()}".encode()).hexdigest()[:12]
```

**File:** `communication/latent_space.py` (lines 444–447)

```python
def _generate_message_id(self) -> str:
    """Generate unique message ID."""
    timestamp = datetime.now(timezone.utc).isoformat()
    return hashlib.sha256(f"{timestamp}:{random.random()}".encode()).hexdigest()[:16]
```

## Proposed Change

**`core/utils.py`:**

```python
import secrets  # add to top-level imports

def generate_id(prefix: Optional[str] = None) -> str:
    """Generate a unique identifier."""
    unique_part = secrets.token_hex(8)   # 16 hex chars, 64 bits of randomness
    if prefix:
        return f"{prefix}_{unique_part}"
    return unique_part
```

Note: the timestamp prefix is dropped — `secrets.token_hex(8)` already provides sufficient uniqueness (2^64 possible values). If the timestamp component is important for sortability, keep it: `f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}_{secrets.token_hex(4)}"`.

**`memory/operations.py`:**

```python
import secrets  # add to module-level imports (remove inline imports)

def _generate_id(self) -> str:
    """Generate unique ID."""
    return secrets.token_hex(6)   # 12 hex chars
```

**`communication/latent_space.py`:**

```python
import secrets  # add to module-level imports

def _generate_message_id(self) -> str:
    """Generate unique message ID."""
    return secrets.token_hex(8)   # 16 hex chars
```

Also: `operations.py` has inline `import hashlib` and `import random` inside the method body — move these to module-level imports and remove them once the `random.random()` usage is gone.

## Files Modified

- `core/utils.py`
  - Imports: add `import secrets`
  - Lines 106–111: replace `generate_id` body
- `memory/operations.py`
  - Imports: remove inline `import hashlib; import random` from `_generate_id`, add `import secrets` at top
  - Lines 411–416: replace `_generate_id` body
- `communication/latent_space.py`
  - Imports: add `import secrets`
  - Lines 444–447: replace `_generate_message_id` body

## Acceptance Criteria

- [ ] No `random.random()` calls remain in ID generation functions in any of the three files
- [ ] Generated IDs are unique strings with no collisions in a 100,000-ID sample
- [ ] ID format is still a hex string (compatible with existing consumers)
- [ ] `import secrets` added at module level (not inline)
- [ ] All existing tests pass

## Edge Cases & Risks

- **ID length change:** `secrets.token_hex(8)` produces a 16-character hex string. Current `utils.py` produces `timestamp(20 chars) + "_" + md5[:8]` = 29 chars. If any code parses ID length or extracts a timestamp from the prefix portion, it will break. Search for ID parsing before merging.
- **`operations.py` timestamp extraction:** The current `_generate_id` encodes the timestamp in the hash input but the timestamp is not recoverable from the output (it's hashed). So no information is lost by dropping it.
- **`random` module still used elsewhere:** Only the ID generation usages are changed. Other `random` usages (RNG for mock data, etc.) are handled by separate tasks.

## Test Notes

- `test_agents.py` checks that agent IDs are unique strings — will pass unchanged.
- `test_communication.py` checks message IDs — will pass unchanged.
- Add test: generate 1000 IDs from `generate_id()`, assert all unique.
