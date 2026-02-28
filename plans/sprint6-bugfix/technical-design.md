# Technical Design: Sprint 6 Bug-Fix and Improvements

## Current State

### GRPO Training Pipeline
- `training/stage2_agemem/grpo.py:312` -- `_update_reference_model` aliases
  `self.reference_model = self.policy_model` instead of deep-copying. Any
  mutation to `policy_model` immediately mutates `reference_model`, making the
  KL penalty trivially zero.
- `training/stage2_agemem/grpo.py:301` -- `_get_action_logprob` returns a
  constant `torch.tensor(0.0, requires_grad=True)` placeholder. Combined with
  the aliasing bug, GRPO produces no meaningful gradient signal.
- `training/stage2_agemem/grpo.py:326` -- `torch.load(path)` without
  `weights_only=True` allows arbitrary code execution via pickle.

### Enum Divergence
- `agents/execution_agent.py:23-46` defines `OrderType` (includes TWAP, VWAP),
  `OrderStatus` (uses PARTIAL), `OrderSide`.
- `trading/order_management.py:26-47` defines its own `OrderType` (no TWAP/VWAP),
  `OrderStatus` (uses PARTIALLY_FILLED, adds SUBMITTED), `OrderSide`.
- Cross-module enum comparisons will silently fail because they are different
  Python objects.

### Determinism Issues
- `trading/order_management.py:272` uses `hash(order.symbol)` despite the same
  file defining `_stable_hash` on line 19.
- `agents/execution_agent.py:490` uses unseeded `random.random()` in
  `_simulate_fill`.

### Dimension Mismatch
- `communication/encoder.py:52` -- `AgentStateEncoder.__init__` defaults
  `latent_dim=256`.
- `communication/latent_space.py:58,101,161` -- `LatentSpace` and its internal
  encoder/decoder default `latent_dim=512`.
- If both are instantiated with defaults, tensors will have incompatible shapes.

### Unbounded Collections
- `learning/nested_learning.py:160` -- `self.adaptation_history: List[...]` is
  appended to on every adaptation but never pruned.
- `core/base_agent.py:102` -- `self.action_history: List[AgentAction]` grows
  unboundedly but only the last 10 entries are ever read (line 193).

### Blocking I/O in Async
- `learning/nested_learning.py:424` -- `with open(path, "w")` inside
  `async def save_state`.
- `learning/repexp.py:333` -- `with open(path, "w")` inside `async def save`.

### Performance
- `agents/market_analyst.py:344-347` -- MACD loop recalculates EMA from scratch
  at each iteration, yielding O(n^2) complexity.

### Weak ID Generation
- `core/utils.py:107` -- `generate_id` uses `random.random()` for entropy.
- `memory/operations.py:416` -- `_generate_id` uses `random.random()`.
- `communication/latent_space.py:447` -- `_generate_message_id` uses
  `random.random()`.

### Naive Timestamps
- `core/utils.py:199` -- `format_timestamp` defaults to `datetime.now()` (no
  timezone).
- `memory/operations.py:26` -- `ContextItem.timestamp` defaults to
  `datetime.now().isoformat()`.

### Logging
- `core/utils.py:38-41` -- `setup_logging` always adds new handlers; repeated
  calls produce duplicate log lines.
- Multiple agent files use f-string logging instead of `%s`-style lazy
  formatting.

### Miscellaneous
- `core/utils.py:9` -- `import numpy as np` is a hard top-level import with no
  try/except guard (unlike torch imports elsewhere).
- `core/utils.py:165-184` -- `deep_merge` has no recursion depth limit.
- `agents/risk_manager.py:181` -- `think()` hardcodes `"done": False`.
- `memory/agemem.py:393-402` -- Quality rewards are hardcoded stubs.
- `agents/coordinator.py:318-346` -- Two consecutive `memory.add()` calls store
  overlapping data on every `act()` invocation.
- `agents/coordinator.py:505,539` -- `_allocate_resources` docstring says
  "round-robin" but implementation is proportional share.
- `models/olmoe.py` -- OLMoE model exists but no agent imports or uses it.
- No `main.py` or equivalent entry point exists.

## Proposed Changes (Summary)

Changes are grouped by subsystem:

1. **GRPO pipeline** -- deep-copy reference model, implement real action
   log-prob (or raise NotImplementedError), add `weights_only=True`.
2. **Trading enums** -- create canonical `trading/enums.py`, merge both sets,
   re-export from `execution_agent.py` and `order_management.py`.
3. **Determinism** -- replace `hash()` with `_stable_hash`, seed `random` in
   simulation.
4. **Dimension defaults** -- align `AgentStateEncoder` default `latent_dim` to
   512 to match `LatentSpace`.
5. **Bounded collections** -- convert unbounded lists to `deque(maxlen=N)` or
   add explicit pruning.
6. **Async I/O** -- wrap blocking `open()` calls with `asyncio.to_thread`.
7. **Performance** -- rewrite MACD to incremental EMA.
8. **ID generation** -- replace `random.random()` with `secrets.token_hex` or
   `uuid4`.
9. **Timestamps** -- switch all `datetime.now()` to `datetime.now(timezone.utc)`.
10. **Logging** -- add handler-exists guard in `setup_logging`; convert f-string
    logs to `%s`-style (lower priority).
11. **Misc code quality** -- guard numpy import, add depth limit to
    `deep_merge`, fix hardcoded `done: False`, fix docstring mismatch,
    deduplicate coordinator memory writes.
12. **OLMoE integration + entry point** -- larger scope items, planned but may
    carry over.

## Alternatives Considered

| Approach | Pros | Cons | Why not chosen |
|----------|------|------|----------------|
| Create `trading/enums.py` as canonical source | Single source of truth, clean imports | Requires updating all imports | **Chosen** -- cleanest long-term |
| Keep enums in `order_management.py` and import into `execution_agent.py` | Fewer new files | `order_management` becomes a dependency of agents layer | Violates layering |
| Use `aiofiles` for async I/O | Purpose-built, clean API | New dependency (CON-3 violation) | Not chosen per constraint |
| Use `asyncio.to_thread(open, ...)` | No new deps, standard library | Slightly more verbose | **Chosen** |
| Align encoder to 512 | Matches LatentSpace default | Could break existing code expecting 256 | **Chosen** -- 512 is the downstream consumer's expectation |
| Align LatentSpace to 256 | Smaller memory footprint | Encoder is the "input" side; LatentSpace is the "system" | Not chosen -- LatentSpace is the authority |
