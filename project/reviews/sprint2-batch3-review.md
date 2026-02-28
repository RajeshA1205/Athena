# Senior-Dev Review: Sprint 2 Batch 3 — Router, Agent Generator, Cooperative Evolution

**Date:** 2026-02-22
**Reviewer:** Senior-Dev Agent
**Verdict:** CHANGES REQUIRED
**Files Reviewed:**
- `communication/router.py` (NEW — TASK-009)
- `evolution/agent_generator.py` (NEW — TASK-011)
- `evolution/cooperative_evolution.py` (NEW — TASK-012)
- `communication/__init__.py` (UPDATED)
- `evolution/__init__.py` (UPDATED)

**Tasks Covered:** TASK-009, TASK-011, TASK-012

---

## Summary

Three new modules and two `__init__.py` updates. Overall code quality is solid — good documentation, correct `datetime.now(timezone.utc)` usage throughout, proper async patterns, and complete exports. However, there are 2 Critical and 2 Major issues that block acceptance: the priority-queue routing path in `router.py` silently encodes string messages as zero vectors (rendering all string messages indistinguishable), and `agent_generator.py` generates non-unique config IDs after pruning, causing silent overwrites. Two major issues cover non-atomic receive ordering (acceptable in asyncio's cooperative model but undocumented) and redundant N+1 encoding in broadcast.

---

## Findings Overview

| # | Severity | Category | File | Line(s) | Finding |
|---|----------|----------|------|---------|---------|
| 1 | 🔴 CRITICAL | Integration | `communication/router.py` | 151 | `encoder.encode_agent_state()` returns zero vector for string inputs — all string messages become identical and unroutable |
| 2 | 🔴 CRITICAL | Correctness | `evolution/agent_generator.py` | 163 | `config_id` uses `len(generated_configs)` — non-unique after pruning; silently overwrites entries |
| 3 | 🟡 MAJOR | Correctness | `communication/router.py` | 232–249 | `receive()` drain + LatentSpace fetch is non-atomic across an `await`; priority inversion possible |
| 4 | 🟡 MAJOR | Performance | `communication/router.py` | 362–366 | `broadcast_with_attention` re-encodes message N times for N recipients — N+1 encode calls |
| 5 | ⚠️ Minor | Style | `evolution/agent_generator.py` | throughout | f-string logger calls instead of `%s`-style (inconsistent with `router.py`) |
| 6 | ⚠️ Minor | Correctness | `evolution/cooperative_evolution.py` | 389 | `cross_pollinate` cutoff `avg_reward * 0.9` inverts for negative rewards |
| 7 | ⚠️ Minor | Async | `evolution/agent_generator.py`, `evolution/cooperative_evolution.py` | 266–284, 411–470 | Sync file I/O inside `async` methods blocks the event loop |
| 8 | ⚠️ Minor | Style | `communication/router.py` | 18–20 | Absolute imports; rest of package uses relative imports |
| 9 | 📝 Nit | Style | `communication/router.py` | 362–366 | `_dot` helper redefined on every loop iteration |
| 10 | 📝 Nit | Documentation | `evolution/cooperative_evolution.py` | 244–246 | `target_agent` param in `share_knowledge` is accepted but never used for routing |
| 11 | 📝 Nit | Style | `evolution/cooperative_evolution.py` | 178–179 | `agent_performance` list trimmed with O(n) `pop(0)` — could be `deque(maxlen=100)` |

---

## Detailed Findings

### 🔴 CRITICAL

#### #1 — String messages encoded as zero vectors in priority-queue path

**File:** `communication/router.py` line 151

When `send()` is called with a string `message`, `encoder.encode_agent_state()` (encoder.py:225–230) logs a warning and returns `torch.zeros(self.latent_dim)`. Every string message produces the same zero-vector representation, making them semantically identical after encoding. The LatentSpace fallback path (`router.py` lines 175–193) correctly wraps the message in `AgentMessage` and delegates to `LatentSpace.send()`, which handles strings properly via character-code tokenization.

The two routing paths produce fundamentally different encodings for the same string input, with the priority-queue path silently degrading string messages.

**Fix:** Add string handling to `AgentStateEncoder.encode_agent_state()` using the existing `encode_text()` method:

```python
# In communication/encoder.py, inside encode_agent_state():
elif isinstance(agent_output, str):
    return self.encode_text(agent_output)
```

---

#### #2 — Non-unique config IDs after pruning

**File:** `evolution/agent_generator.py` line 163

```python
config_id = f"generated_{agent_type}_{len(self.generated_configs)}"
```

After pruning removes configs, `len(self.generated_configs)` can return a value that was already used by a previously pruned config. The new config silently overwrites the existing entry in the dict.

**Fix:** Use a monotonically increasing counter:

```python
# In __init__:
self._next_config_id: int = 0

# In generate_from_pattern:
config_id = f"generated_{agent_type}_{self._next_config_id}"
self._next_config_id += 1
```

---

### 🟡 MAJOR

#### #3 — Non-atomic receive across priority queues and LatentSpace

**File:** `communication/router.py` lines 232–249

`receive()` drains priority queues HIGH → MEDIUM → LOW, then `await`s LatentSpace messages. The `await` at line 235 yields control to the event loop. If a concurrent `send()` inserts a HIGH-priority message between the queue drain and the LatentSpace call, that HIGH-priority message is processed in the next `receive()` call — after lower-priority LatentSpace messages from the current call. No data corruption (asyncio is single-threaded per loop), but priority ordering can be violated.

**Fix:** Document the non-atomicity with a comment, or add a router-level `asyncio.Lock` around the combined drain+receive.

---

#### #4 — N+1 encoding in `broadcast_with_attention`

**File:** `communication/router.py` lines 362–366

The message is encoded once at the top of `broadcast_with_attention` into `latent_message` for attention scoring. Then `self.send()` is called for each selected agent, which re-encodes the message via `encoder.encode_agent_state(message)` again. For N recipients this is N+1 total encodes.

**Fix:** Bypass `self.send()` and push `latent_message` directly into the priority queue for each recipient.

---

### ⚠️ Minor Findings

#### #5 — f-string logger calls

`agent_generator.py` and `cooperative_evolution.py` use f-string interpolation in `logger.info()` calls. This evaluates the f-string even when the log level is above INFO. `router.py` correctly uses `%s`-style deferred formatting. Standardize.

#### #6 — Negative reward cutoff in `cross_pollinate`

`evolution/cooperative_evolution.py` line 389: `reward_cutoff = avg_reward * 0.9`. When `avg_reward` is negative (valid for reward signals), multiplying by 0.9 makes the cutoff less negative (more restrictive), which is the opposite of intent.

**Fix:** `reward_cutoff = avg_reward - abs(avg_reward) * 0.1`

#### #7 — Synchronous file I/O in async methods

`save_configs`, `load_configs` (`agent_generator.py`), `save_experiences`, `load_experiences` (`cooperative_evolution.py`) perform blocking `open()` calls inside async methods. Either make these methods synchronous (they contain no `await`) or use `asyncio.to_thread(...)`.

#### #8 — Absolute imports in router.py

`router.py` imports via `from communication.encoder import ...` while all sibling modules use relative imports (`from .encoder import ...`). Use relative imports for consistency.

---

### 📝 Nits

| # | File | Description |
|---|------|-------------|
| 9 | `router.py:362–366` | `_dot` helper function redefined on every loop iteration. Move to module/class level. |
| 10 | `cooperative_evolution.py:244–246` | `target_agent` param in `share_knowledge` never used for routing — misleading API. Remove or document. |
| 11 | `cooperative_evolution.py:178–179` | `agent_performance` trimmed with O(n) `pop(0)`. Use `deque(maxlen=100)`. |

---

## Integration Assessment

| Check | Status | Notes |
|-------|--------|-------|
| `encoder.encode_agent_state()` signature | **WARN** | Signature matches but zero-vector output for strings (Finding #1) |
| `decoder.decode_messages()` signature | Pass | Matches `AgentStateDecoder.decode_messages(List[Tensor], mode)` |
| `latent_space.send()` signature | Pass | `LatentSpace.send(AgentMessage)` — router wraps correctly |
| `latent_space.receive()` signature | Pass | `LatentSpace.receive(agent_name)` — matches |
| `workflow_discovery.get_successful_patterns()` | Pass | Signature and return type match |
| `WorkflowPattern` field access | Pass | All fields accessed (`agent_sequence`, `pattern_id`, `success_rate`, etc.) exist |
| `communication/__init__.py` exports | Pass | `LatentSpace`, `AgentStateEncoder`, `AgentStateDecoder`, `MessageRouter`, `MessagePriority` — complete |
| `evolution/__init__.py` exports | Pass | `WorkflowDiscovery`, `WorkflowPattern`, `CooperativeEvolution`, `Experience`, `AgentGenerator`, `AgentConfiguration` — complete |
| HAS_TORCH pattern (router) | Pass | Correct `try/except ImportError` guard; fallback path implemented |
| `datetime.utcnow()` usage | Pass | None — all three files use `datetime.now(timezone.utc)` correctly |
| Test coverage | No tests | No test files for batch 3 code |

---

## Action Items

### Must Fix (blocks acceptance) — 2 items

| # | Finding | File(s) to Edit | Complexity |
|---|---------|-----------------|------------|
| 1 | String zero-vector encoding | `communication/encoder.py` | Low |
| 2 | Non-unique config IDs after pruning | `evolution/agent_generator.py` | Low |

### Should Fix (next pass) — 6 items

- [ ] #3: Document non-atomic receive in `router.py`
- [ ] #4: Eliminate N+1 encoding in `broadcast_with_attention`
- [ ] #5: Switch f-string logger calls to `%s`-style in evolution modules
- [ ] #6: Fix negative reward cutoff in `cross_pollinate`
- [ ] #7: Make file I/O methods synchronous or use `asyncio.to_thread`
- [ ] #8: Standardize to relative imports in `router.py`

### Nits (low priority) — 3 items

- [ ] #9: Move `_dot` helper out of loop body in `router.py`
- [ ] #10: Clarify `target_agent` in `share_knowledge`
- [ ] #11: Use `deque(maxlen=100)` for performance history

---

## Re-Review — 2026-02-22

**Verdict: APPROVED**

Both must-fix items confirmed FIXED:

| # | Finding | Status | Verification |
|---|---------|--------|--------------|
| 1 | String zero-vector encoding | ✅ FIXED | `encoder.py:225–231`: str converted to normalized char-code tensor `[input_dim]`, passed to `encode_numeric()`. `self.input_dim` valid (set at `__init__:80`). |
| 2 | Non-unique config IDs after pruning | ✅ FIXED | `agent_generator.py:137`: `self._next_config_id: int = 0`. `generate_from_pattern:164–165`: uses counter then increments. No `len(generated_configs)` for ID generation. |

No regressions detected.
