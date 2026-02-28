# Sprint 6 Open Questions -- Resolved Decisions

Date: 2026-02-23
Sprint: 6 (Bug-Fix and Improvements)
Status: All questions resolved

---

## OQ-1a: Action space for GRPO MLP

**Question:** What is the action space for the GRPO MLP head? How many discrete actions, and what are they?

**Decision:** 6 actions total, matching the `MemoryOperation` enum exactly:

| Index | Action   | Description                        |
|-------|----------|------------------------------------|
| 0     | ADD      | Add a new memory episode           |
| 1     | UPDATE   | Update an existing memory episode  |
| 2     | DELETE   | Delete a memory episode            |
| 3     | RETRIEVE | Retrieve from memory               |
| 4     | SUMMARY  | Summarize memory contents          |
| 5     | FILTER   | Filter memory by criteria          |

**MLP output dimension = 6.**

**Graphiti mapping detail:** ADD and UPDATE are separate GRPO actions (separate MLP outputs with independent learned policies). However, at the Graphiti API layer, both ADD and UPDATE map to Graphiti's `add_episode` operation. This is an implementation detail inside the memory layer -- the GRPO action space treats them as distinct actions because the agent's intent differs (creating vs. modifying), even though the underlying API call is the same.

**Impact:** `training/stage2_agemem/grpo.py` (TASK-024), `models/olmoe.py` or `models/action_head.py` (TASK-029)

---

## OQ-1b: MLP relationship to OLMoE

**Question:** Is the GRPO MLP a standalone model, or does it sit on top of OLMoE?

**Decision:** The MLP is a classification head sitting **on top of OLMoE**. The architecture is:

1. OLMoE encodes the agent state into a hidden embedding
2. The MLP action head maps that embedding to a 6-dim logit vector
3. `log_softmax` produces action probabilities
4. The selected action index is used to extract the log-probability

This means:
- **TASK-024 (GRPO) and TASK-029 (OLMoE wiring) share the same model instance.** The `policy_model` passed to `StepwiseGRPO` is the composite OLMoE + MLP head, not a standalone MLP.
- The `_get_action_logprob` method follows this flow: encode state with OLMoE -> pass embedding through MLP head -> `log_softmax` -> index by `action_idx`.
- The `_get_reference_logprob` method does the same with `self.reference_model` (a frozen deep copy of the composite model).

**Impact:** `training/stage2_agemem/grpo.py` (TASK-024), `models/olmoe.py` or `models/action_head.py` (TASK-029)

---

## OQ-3: ExecutionAgent scope

**Question:** Should ExecutionAgent be wired as a trade executor in Sprint 6, and which bug fixes apply to it?

**Decision:** Three parts:

1. **Bug fixes:** Apply ALL Sprint 6 bug fixes to `agents/execution_agent.py`:
   - TASK-026: Enum unification (use canonical `OrderStatus` / `OrderType` enums)
   - TASK-033: Seeded random for `_simulate_fill`
   - TASK-045: f-string to %-style logging conversion

2. **No trade execution wiring:** Do NOT wire ExecutionAgent as a live trade executor in Sprint 6.

3. **Role change to advisory output:** Instead of submitting orders, ExecutionAgent now produces a **recommended investment plan / advice** that a human trader acts on manually. Specifically:
   - `act()` returns a structured recommendation: `{symbol, action_type, suggested_quantity, rationale}` rather than calling any order submission API
   - The fill simulation code (`_simulate_fill`) remains in place and is used for recommendation reasoning (estimating realistic fill prices, slippage, etc.) but does not drive actual order submission
   - This positions ExecutionAgent as an advisory tool, reducing risk while the system is pre-production

**Impact:** `agents/execution_agent.py` (TASK-026, TASK-033, TASK-045, plus new advisory output refactor)

---

## Previously Resolved (not revisited)

- **OQ-2 (PARTIAL vs PARTIALLY_FILLED):** Resolved during TASK-026 design -- consolidated into a single enum value.
- **OQ-4 (Deployment target for main.py):** No existing deployment target; TASK-030 proceeds with a standalone CLI.
- **OQ-5 (Which agents get OLMoE):** Resolved as part of OQ-1b above -- CoordinatorAgent and StrategyAgent get OLMoE integration (TASK-029).
