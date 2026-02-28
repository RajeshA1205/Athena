# Sprint 6: Code Quality, Security, and Correctness Fixes

## Goal
Address all confirmed findings from two external code reviews, covering critical
functional bugs in the GRPO training pipeline, a security vulnerability, enum
divergence across trading modules, determinism issues, performance problems, and
assorted code-quality improvements. The sprint goal is to bring ATHENA from
"feature-complete prototype" to "production-ready baseline."

## Background
Sprints 2-5 delivered the full ATHENA system: five agents, AgeMem memory,
LatentMAS communication, AgentEvolver evolution, nested learning, RepExp
exploration, trading infrastructure, and a 171-test suite. Two independent code
reviews identified 28 findings across security, correctness, performance, and
code quality. This sprint resolves all of them.

## Requirements

### Functional
- FR-1: GRPO training must produce meaningful policy updates (reference model
  must be a deep copy; action log-probs must come from the policy model).
- FR-2: Trading enum types (`OrderType`, `OrderSide`, `OrderStatus`) must be
  defined in exactly one canonical location and used consistently.
- FR-3: Backtest `total_return` must use the same geometric/multiplicative model
  as drawdown calculation.
- FR-4: OLMoE foundation model must be importable and callable by at least one
  agent or training entry point.
- FR-5: A production entry point (`main.py`) must exist to run the system
  end-to-end.

### Non-Functional
- NFR-1: All hash-based operations must be deterministic across Python sessions
  (use `_stable_hash` or `hashlib`, never bare `hash()`).
- NFR-2: All randomness in simulation/fill paths must be seeded and reproducible.
- NFR-3: Unbounded lists (`adaptation_history`, `action_history`) must be capped.
- NFR-4: No O(n^2) hot-path calculations (MACD).
- NFR-5: All IDs must use `uuid4` or `secrets.token_hex`, not `random.random()`.
- NFR-6: Async methods must not use blocking I/O.
- NFR-7: All timestamps must be timezone-aware (UTC).

### Constraints
- CON-1: Changes must not break any of the existing 171 tests.
- CON-2: Public API signatures should remain backwards-compatible where possible.
- CON-3: No new third-party dependencies.

## Assumptions
- ASM-1: `copy.deepcopy` or `state_dict()` copy is sufficient for reference
  model isolation in GRPO.
- ASM-2: `aiofiles` is not acceptable as a new dependency; `asyncio.to_thread`
  wrapping `open()` is the preferred async I/O pattern.
- ASM-3: The OLMoE integration (B1) and entry-point creation (B3) are design
  tasks that may span into a follow-up sprint if scope exceeds estimates.

## Out of Scope
- Full re-architecture of the GRPO training loop (only the aliasing and
  placeholder log-prob bugs are in scope).
- Replacing `random` module with `numpy.random` globally.
- Adding new tests for every fix (test updates are limited to ensuring existing
  tests still pass; new tests are recommended but not required this sprint).
