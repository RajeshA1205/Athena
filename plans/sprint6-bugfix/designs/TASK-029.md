# TASK-029: Wire OLMoE model into the agent framework

## Summary
`models/olmoe.py` defines `OLMoEConfig` and `OLMoEModel` — a fully-featured wrapper for the OLMoE 1B Mixture-of-Experts foundation model. However, no agent, coordinator, or training code imports or uses it. The entire foundation model layer is dead code. This task wires `OLMoEModel` into `BaseAgent` as an optional `llm_backend` so agents can route natural-language reasoning through the real model when it is available. The integration is gated behind `TRANSFORMERS_AVAILABLE` so the system continues to work without transformers installed.

## Current State

**File:** `models/olmoe.py` — defines:
- `OLMoEConfig` dataclass (model_name, device, dtype, generation params, LoRA params)
- `OLMoEModel` class with `generate()`, `encode()`, `load_lora_adapter()`, `save_lora_adapter()`

**File:** `core/config.py` — defines `ModelConfig` with `model_name = "allenai/OLMo-1B"` (stale reference to OLMo, not OLMoE). No field for enabling/disabling the LLM backend.

**File:** `core/base_agent.py` — `BaseAgent.__init__` accepts `memory` and `router` but has no `llm` parameter. The `think()` and `act()` methods are fully synthetic (rule-based), never invoking a language model.

No agent file imports from `models/`.

**Decision (OQ-1b, OQ-5):** OLMoE is wired into **CoordinatorAgent** (for decision synthesis) and **StrategyAgent** (for analysis reasoning). OLMoE also serves as the GRPO policy encoder -- the same model instance is shared with TASK-024. An MLP action head (output dim = 6, matching `MemoryOperation` enum) is added on top of OLMoE for GRPO training purposes. See `project/decisions/sprint6-open-questions.md` for full rationale.

## Proposed Change

### Step 1 — Add `OLMoEConfig` to `core/config.py`

Add an `ollmoe` config section to `ATHENAConfig`:

```python
# In core/config.py

@dataclass
class OLMoEIntegrationConfig:
    """Controls OLMoE LLM backend wiring."""
    enabled: bool = False                          # off by default
    model_name: str = "allenai/OLMoE-1B-7B-0924"
    device: str = "auto"
    dtype: str = "float16"
    load_in_4bit: bool = False
    max_length: int = 2048
    lora_adapter_path: Optional[str] = None        # path to fine-tuned LoRA weights

# Add to ATHENAConfig dataclass:
#   olmoe: OLMoEIntegrationConfig = field(default_factory=OLMoEIntegrationConfig)
```

### Step 2 — Add optional `llm` parameter to `BaseAgent`

```python
# In core/base_agent.py

if TYPE_CHECKING:
    from models.olmoe import OLMoEModel   # add to existing TYPE_CHECKING block

class BaseAgent(ABC):
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        config: Optional[Dict[str, Any]] = None,
        memory: Optional[Any] = None,
        router: Optional[Any] = None,
        llm: Optional[Any] = None,        # <-- new parameter
    ):
        ...
        self.llm = llm   # OLMoEModel instance or None
```

Add a helper method:

```python
async def _llm_reason(self, prompt: str) -> Optional[str]:
    """Route a reasoning prompt through OLMoE if available, else return None."""
    if self.llm is None:
        return None
    try:
        return await asyncio.to_thread(self.llm.generate, prompt)
    except Exception as e:
        self.logger.warning("LLM reasoning failed: %s", e)
        return None
```

### Step 3 — Use `_llm_reason` in CoordinatorAgent and StrategyAgent

**CoordinatorAgent** (`agents/coordinator.py`): In the `think()` method, add an optional LLM reasoning call for the final decision synthesis:

```python
# After collecting agent responses, optionally use LLM to synthesize
llm_synthesis = await self._llm_reason(
    f"Given these agent recommendations: {responses}, what is the best trading decision?"
)
if llm_synthesis:
    thought["llm_synthesis"] = llm_synthesis
```

**StrategyAgent** (`agents/strategy_agent.py`): In the `think()` method, use LLM to enrich strategy analysis:

```python
# After computing technical indicators, optionally use LLM for deeper analysis
llm_analysis = await self._llm_reason(
    f"Analyze these market signals: {signals}. What patterns do you see?"
)
if llm_analysis:
    thought["llm_analysis"] = llm_analysis
```

Both integrations are additive -- they enrich the `thought` dict but do not change existing logic.

### Step 3b — Add MLP action head for GRPO integration

The OLMoE model instance shared with GRPO (TASK-024) needs an MLP action head. This head maps OLMoE's hidden state to a 6-dim logit vector over `MemoryOperation` actions:

```python
# In models/olmoe.py or a new models/action_head.py
class MemoryActionHead(torch.nn.Module):
    """MLP head on top of OLMoE for GRPO memory action selection."""
    def __init__(self, hidden_dim: int, action_dim: int = 6):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.mlp(embedding)
```

The composite model (OLMoE backbone + action head) is what gets passed to `StepwiseGRPO` as `policy_model`. The action head is only used during GRPO training; agent reasoning uses `OLMoEModel.generate()` and `OLMoEModel.encode()` directly.

### Step 4 — Smoke test

```python
# In tests/test_integration_e2e.py or a new test
def test_coordinator_works_without_llm():
    coordinator = CoordinatorAgent(...)  # no llm= argument
    # existing tests pass unchanged
```

## Files Modified

- `core/config.py` — add `OLMoEIntegrationConfig` dataclass and field on `ATHENAConfig`
- `core/base_agent.py` — add `llm` param to `__init__`, add `_llm_reason` helper
- `agents/coordinator.py` — use `_llm_reason` in `think()` for optional LLM synthesis
- `agents/strategy_agent.py` — use `_llm_reason` in `think()` for optional LLM analysis
- `models/olmoe.py` (or new `models/action_head.py`) — add `MemoryActionHead` MLP class (6-dim output for GRPO)
- **Dependency:** TASK-024 (`training/stage2_agemem/grpo.py`) expects the `policy_model` to have both `encode()` and `action_head()` methods — this task provides that composite model

## Acceptance Criteria

- [ ] `BaseAgent.__init__` accepts an optional `llm` parameter
- [ ] `_llm_reason(prompt)` returns `None` gracefully when `self.llm is None`
- [ ] `CoordinatorAgent` can be instantiated with a real `OLMoEModel` and calls `generate()` during `think()`
- [ ] `StrategyAgent` can be instantiated with a real `OLMoEModel` and calls `generate()` during `think()`
- [ ] `MemoryActionHead` produces a (6,) logit tensor from an OLMoE embedding
- [ ] The composite model (OLMoE + action head) can be passed to `StepwiseGRPO` as `policy_model`
- [ ] System runs end-to-end without transformers installed (all `llm=None` paths)
- [ ] All existing 171 tests pass unmodified

## Edge Cases & Risks

- **GPU memory:** Loading OLMoE 1B requires ~2–4 GB GPU RAM. Document this in `OLMoEIntegrationConfig`. The `load_in_4bit` option reduces this to ~1 GB.
- **Blocking `generate()`:** `OLMoEModel.generate()` is synchronous. Always call via `asyncio.to_thread()` to avoid blocking the event loop.
- **`_llm_reason` failure:** Wrap in try/except — if the model is loaded but inference fails (OOM, etc.), the agent should fall back to synthetic reasoning, not crash.
- **`lora_adapter_path`:** If set, the model should load LoRA weights via `load_lora_adapter()`. Add this logic in the wiring code that instantiates `OLMoEModel`.
- **Shared model instance with GRPO:** The OLMoE model is shared between agent reasoning (this task) and GRPO training (TASK-024). Training and inference must run in separate phases to avoid gradient corruption during in-flight inference. The action head weights are only updated during GRPO training; the OLMoE backbone may or may not be frozen depending on LoRA configuration.

## Test Notes

- No new tests required for the `llm=None` path (existing tests cover it).
- New test with a mock `OLMoEModel`: assert `_llm_reason("test")` returns the mock's output.
- New test: assert `_llm_reason("test")` returns `None` when `self.llm is None`.
