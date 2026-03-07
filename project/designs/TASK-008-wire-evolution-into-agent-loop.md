# TASK-008: Wire evolution/RepExp/NestedLearning into the agent loop

## Problem

The evolution layer (`WorkflowDiscovery`, `CooperativeEvolution`), RepExp, and `NestedLearning` infrastructure exists but is only connected manually in `cli.py` lines 412-484. The `main.py` production entry point and the `CoordinatorAgent` have no awareness of these systems. This means paper-trade and backtest modes miss nested learning adaptation, workflow discovery, and experience replay entirely.

## Files to change

| File | Lines | Change |
|------|-------|--------|
| `agents/coordinator.py` | 41-93 | Add optional `learners` and `workflow_discovery` attributes to `CoordinatorAgent.__init__` |
| `agents/coordinator.py` | 232-383 | Wire learning trajectory creation + adaptation into `act()` after coordination result |
| `main.py` | 66-171 | Instantiate `NestedLearning` per agent and `WorkflowDiscovery`, pass to coordinator |
| `cli.py` | 412-484 | Refactor to delegate to coordinator's built-in learning instead of inline code |

## Approach

1. **Create a pipeline orchestrator** (new file `core/pipeline.py` or extend `CoordinatorAgent`):
   - Option A (recommended): Add a `PostCycleHook` protocol and register learning + evolution as hooks on the coordinator. This keeps the coordinator focused on coordination while allowing extensible post-cycle processing.
   - Option B: Inline the learning loop into `CoordinatorAgent.act()`. Simpler but couples concerns.

2. **For Option A**, add to `CoordinatorAgent`:
   ```python
   def __init__(self, ..., post_cycle_hooks: Optional[List[Callable]] = None):
       self._post_cycle_hooks = post_cycle_hooks or []

   async def _run_post_cycle(self, agent_actions: Dict[str, AgentAction], context: AgentContext):
       for hook in self._post_cycle_hooks:
           await hook(agent_actions, context)
   ```

3. **Create `core/learning_hook.py`** that encapsulates the logic currently in `cli.py` lines 412-484:
   - Build `TaskTrajectory` per agent from action results
   - Call `learner.adapt_to_task()` (inner loop)
   - Every N cycles, call `learner.update_meta_parameters()` (outer loop)
   - Every M cycles, call `learner.consolidate_knowledge()`

4. **Create `core/evolution_hook.py`** that wraps `WorkflowDiscovery.analyze_execution()`:
   - Build execution trace from coordinator's agent_recommendations and final_decision
   - Feed to `WorkflowDiscovery.analyze_execution()`

5. **Update `main.py`** `run()` function:
   - Instantiate `NestedLearning` per agent (same as `cli.py` lines 179-186)
   - Instantiate `WorkflowDiscovery`
   - Create hooks and pass to coordinator

6. **Refactor `cli.py`**:
   - Remove inline learning code (lines 412-484)
   - Let coordinator handle it via hooks
   - Keep verbose logging by passing a verbose callback to the hook

## Edge cases / risks

- **Backward compatibility**: Existing code that creates `CoordinatorAgent()` without hooks must still work. Default `post_cycle_hooks=[]` ensures this.
- **Coordinator doesn't have individual agent actions**: Currently `act()` only receives `thought` (the orchestration plan). The per-agent actions are only available in `cli.py`. The coordinator would need to receive or store agent results. Options: (a) pass agent_actions in `thought` metadata, (b) have coordinator store them during orchestration.
- **CLI verbose logging**: The hook needs access to the verbose flag and print functions. Pass a logger/callback rather than coupling to CLI internals.
- **Cycle counting** for outer-loop and consolidation triggers: Maintain a counter on the hook instance, not globally.

## Acceptance criteria

- [ ] `main.py --mode paper-trade` runs nested learning inner-loop on every cycle.
- [ ] `main.py --mode paper-trade` runs outer-loop meta-update every N cycles (configurable).
- [ ] `WorkflowDiscovery.analyze_execution()` is called after each coordination cycle.
- [ ] `cli.py` delegates to the same hook infrastructure (no duplicated learning code).
- [ ] `pytest tests/ -q` remains green.
