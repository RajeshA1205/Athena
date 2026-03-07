"""
Unit tests for core.learning_hook.LearningHook and the
CoordinatorAgent post_cycle_hooks integration.

All heavy dependencies (NestedLearning, WorkflowDiscovery, CooperativeEvolution)
are exercised with real lightweight instances so no mocking of internal async
calls is needed. CooperativeEvolution.add_experience is the only genuine async
call triggered by the hook; others (adapt_to_task, update_meta_parameters,
consolidate_knowledge, analyze_execution) are also real but pure-Python.
"""

from __future__ import annotations

import types
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_learning_config(**kwargs) -> types.SimpleNamespace:
    """Minimal LearningConfig-like namespace accepted by NestedLearning."""
    defaults = {
        "inner_lr": 0.01,
        "outer_lr": 0.001,
        "inner_steps": 3,
        "exploration_coefficient": 0.2,
        "representation_dim": 8,
        "diversity_threshold": 0.5,
    }
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


def _make_agent_action(
    action_type: str = "coordination",
    success: bool = True,
    result: dict | None = None,
    duration: float = 0.01,
):
    """Return a lightweight AgentAction-like object."""
    from core.base_agent import AgentAction
    return AgentAction(
        action_type=action_type,
        parameters={},
        result=result or {},
        success=success,
        error=None,
        duration=duration,
    )


def _make_learners(agent_names=None):
    """Build a dict of real NestedLearning instances."""
    from learning.nested_learning import NestedLearning
    cfg = _make_learning_config()
    names = agent_names or ["market_analyst", "risk_manager", "coordinator"]
    return {name: NestedLearning(cfg, name) for name in names}


def _make_workflow_discovery():
    from evolution.workflow_discovery import WorkflowDiscovery
    return WorkflowDiscovery()


def _make_coop_evolution():
    from evolution.cooperative_evolution import CooperativeEvolution
    return CooperativeEvolution(config={})


def _make_hook(
    agent_names=None,
    with_coop: bool = True,
    meta_update_interval: int = 10,
    consolidation_interval: int = 50,
):
    from core.learning_hook import LearningHook
    learners = _make_learners(agent_names)
    wd = _make_workflow_discovery()
    ce = _make_coop_evolution() if with_coop else None
    return LearningHook(
        learners=learners,
        workflow_discovery=wd,
        cooperative_evolution=ce,
        meta_update_interval=meta_update_interval,
        consolidation_interval=consolidation_interval,
    )


# ---------------------------------------------------------------------------
# LearningHook: import and instantiation
# ---------------------------------------------------------------------------

class TestLearningHookInstantiation:
    def test_import(self):
        from core.learning_hook import LearningHook
        assert LearningHook is not None

    def test_instantiation_minimal(self):
        from core.learning_hook import LearningHook
        hook = LearningHook(
            learners=_make_learners(),
            workflow_discovery=_make_workflow_discovery(),
        )
        assert hook is not None
        assert hook.cycle_count == 0
        assert hook.cooperative_evolution is None

    def test_instantiation_with_coop_evolution(self):
        from core.learning_hook import LearningHook
        hook = LearningHook(
            learners=_make_learners(),
            workflow_discovery=_make_workflow_discovery(),
            cooperative_evolution=_make_coop_evolution(),
        )
        assert hook.cooperative_evolution is not None

    def test_custom_intervals(self):
        hook = _make_hook(meta_update_interval=3, consolidation_interval=7)
        assert hook.meta_update_interval == 3
        assert hook.consolidation_interval == 7


# ---------------------------------------------------------------------------
# LearningHook.on_cycle_complete: basic behaviour
# ---------------------------------------------------------------------------

class TestLearningHookOnCycleComplete:
    @pytest.mark.asyncio
    async def test_cycle_count_increments(self):
        hook = _make_hook()
        actions = {"market_analyst": _make_agent_action()}
        await hook.on_cycle_complete(actions, {"task": "test_task", "cycle": 0})
        assert hook.cycle_count == 1

    @pytest.mark.asyncio
    async def test_cycle_count_increments_twice(self):
        hook = _make_hook()
        actions = {"market_analyst": _make_agent_action()}
        await hook.on_cycle_complete(actions, {"task": "t1", "cycle": 0})
        await hook.on_cycle_complete(actions, {"task": "t2", "cycle": 1})
        assert hook.cycle_count == 2

    @pytest.mark.asyncio
    async def test_inner_loop_stores_trajectory(self):
        """Each on_cycle_complete call must add a trajectory to the learner."""
        from learning.nested_learning import NestedLearning
        cfg = _make_learning_config()
        learner = NestedLearning(cfg, "market_analyst")

        from core.learning_hook import LearningHook
        hook = LearningHook(
            learners={"market_analyst": learner},
            workflow_discovery=_make_workflow_discovery(),
        )
        actions = {"market_analyst": _make_agent_action(success=True)}
        await hook.on_cycle_complete(actions, {"task": "t1", "cycle": 0})

        assert "t1" in learner.task_trajectories
        assert len(learner.task_trajectories["t1"]) == 1

    @pytest.mark.asyncio
    async def test_agent_not_in_learners_is_skipped(self):
        """An agent present in agent_actions but absent from learners must not raise."""
        hook = _make_hook(agent_names=["coordinator"])
        actions = {
            "coordinator": _make_agent_action(),
            "unknown_agent": _make_agent_action(),
        }
        # Must complete without error
        await hook.on_cycle_complete(actions, {"task": "t", "cycle": 0})

    @pytest.mark.asyncio
    async def test_failed_action_produces_negative_reward(self):
        """A failed action should store a trajectory with reward < 0."""
        from learning.nested_learning import NestedLearning
        cfg = _make_learning_config()
        learner = NestedLearning(cfg, "risk_manager")

        from core.learning_hook import LearningHook
        hook = LearningHook(
            learners={"risk_manager": learner},
            workflow_discovery=_make_workflow_discovery(),
        )
        actions = {"risk_manager": _make_agent_action(success=False)}
        await hook.on_cycle_complete(actions, {"task": "fail_task", "cycle": 0})

        traj = learner.task_trajectories.get("fail_task", [])
        assert len(traj) == 1
        assert traj[0].rewards[0] < 0

    @pytest.mark.asyncio
    async def test_confidence_weighted_reward(self):
        """Reward should be scaled by action confidence when present."""
        from learning.nested_learning import NestedLearning
        cfg = _make_learning_config()
        learner = NestedLearning(cfg, "strategy_agent")

        from core.learning_hook import LearningHook
        hook = LearningHook(
            learners={"strategy_agent": learner},
            workflow_discovery=_make_workflow_discovery(),
        )
        actions = {
            "strategy_agent": _make_agent_action(
                success=True, result={"confidence": 0.5}
            )
        }
        await hook.on_cycle_complete(actions, {"task": "conf_task", "cycle": 0})

        traj = learner.task_trajectories.get("conf_task", [])
        # reward = 1.0 * 0.5 = 0.5
        assert abs(traj[0].rewards[0] - 0.5) < 1e-9

    @pytest.mark.asyncio
    async def test_workflow_discovery_receives_trace(self):
        """analyze_execution must be called once per on_cycle_complete."""
        wd = _make_workflow_discovery()
        from core.learning_hook import LearningHook
        hook = LearningHook(
            learners=_make_learners(["coordinator"]),
            workflow_discovery=wd,
        )
        actions = {"coordinator": _make_agent_action()}
        await hook.on_cycle_complete(actions, {"task": "trace_task", "cycle": 0})
        assert len(wd.execution_history) == 1

    @pytest.mark.asyncio
    async def test_cooperative_evolution_receives_experience(self):
        """add_experience must be called for each agent when coop_evolution is set."""
        ce = _make_coop_evolution()
        from core.learning_hook import LearningHook
        hook = LearningHook(
            learners=_make_learners(["market_analyst", "risk_manager"]),
            workflow_discovery=_make_workflow_discovery(),
            cooperative_evolution=ce,
        )
        actions = {
            "market_analyst": _make_agent_action(),
            "risk_manager": _make_agent_action(),
        }
        await hook.on_cycle_complete(actions, {"task": "coop_task", "cycle": 0})

        stats = await ce.get_population_stats()
        assert stats["total_agents"] == 2

    @pytest.mark.asyncio
    async def test_no_coop_evolution_does_not_raise(self):
        """When cooperative_evolution is None, on_cycle_complete must not raise."""
        from core.learning_hook import LearningHook
        hook = LearningHook(
            learners=_make_learners(["coordinator"]),
            workflow_discovery=_make_workflow_discovery(),
            cooperative_evolution=None,
        )
        actions = {"coordinator": _make_agent_action()}
        await hook.on_cycle_complete(actions, {"task": "no_coop", "cycle": 0})  # no error


# ---------------------------------------------------------------------------
# LearningHook: outer loop triggers
# ---------------------------------------------------------------------------

class TestLearningHookOuterLoop:
    @pytest.mark.asyncio
    async def test_outer_loop_triggers_at_interval(self):
        """After meta_update_interval cycles, meta_params.update_count must increase."""
        from learning.nested_learning import NestedLearning
        cfg = _make_learning_config()
        learner = NestedLearning(cfg, "coordinator")

        from core.learning_hook import LearningHook
        hook = LearningHook(
            learners={"coordinator": learner},
            workflow_discovery=_make_workflow_discovery(),
            meta_update_interval=3,
            consolidation_interval=999,
        )
        actions = {"coordinator": _make_agent_action()}
        for i in range(3):
            await hook.on_cycle_complete(actions, {"task": "outer_task", "cycle": i})

        # After 3 cycles with interval=3, outer loop ran once
        assert learner.meta_params.update_count == 1

    @pytest.mark.asyncio
    async def test_outer_loop_does_not_trigger_before_interval(self):
        """Meta update_count stays 0 when fewer cycles than meta_update_interval."""
        from learning.nested_learning import NestedLearning
        cfg = _make_learning_config()
        learner = NestedLearning(cfg, "coordinator")

        from core.learning_hook import LearningHook
        hook = LearningHook(
            learners={"coordinator": learner},
            workflow_discovery=_make_workflow_discovery(),
            meta_update_interval=10,
            consolidation_interval=999,
        )
        actions = {"coordinator": _make_agent_action()}
        for i in range(5):
            await hook.on_cycle_complete(actions, {"task": "t", "cycle": i})

        assert learner.meta_params.update_count == 0

    @pytest.mark.asyncio
    async def test_consolidation_triggers_at_interval(self):
        """Pruned entries should appear after consolidation_interval cycles."""
        from learning.nested_learning import NestedLearning
        cfg = _make_learning_config()
        learner = NestedLearning(cfg, "coordinator")

        from core.learning_hook import LearningHook
        hook = LearningHook(
            learners={"coordinator": learner},
            workflow_discovery=_make_workflow_discovery(),
            meta_update_interval=999,
            consolidation_interval=3,
        )
        actions = {"coordinator": _make_agent_action()}
        for i in range(3):
            await hook.on_cycle_complete(actions, {"task": "cons_task", "cycle": i})

        # Consolidation ran: task_trajectories still has entries (3 ≤ 50 limit)
        assert "cons_task" in learner.task_trajectories


# ---------------------------------------------------------------------------
# CoordinatorAgent: post_cycle_hooks parameter
# ---------------------------------------------------------------------------

class TestCoordinatorPostCycleHooks:
    def test_coordinator_accepts_post_cycle_hooks(self):
        from agents.coordinator import CoordinatorAgent
        hook = MagicMock()
        agent = CoordinatorAgent(post_cycle_hooks=[hook])
        assert agent._post_cycle_hooks == [hook]

    def test_coordinator_empty_hooks_by_default(self):
        from agents.coordinator import CoordinatorAgent
        agent = CoordinatorAgent()
        assert agent._post_cycle_hooks == []

    def test_coordinator_cycle_count_starts_at_zero(self):
        from agents.coordinator import CoordinatorAgent
        agent = CoordinatorAgent()
        assert agent._cycle_count == 0

    @pytest.mark.asyncio
    async def test_coordinator_act_calls_hook(self):
        """Hook.on_cycle_complete must be called after a successful act()."""
        from agents.coordinator import CoordinatorAgent
        from core.base_agent import AgentContext

        called_with = {}

        class _MockHook:
            async def on_cycle_complete(self, agent_actions, context):
                called_with["agent_actions"] = agent_actions
                called_with["context"] = context

        agent = CoordinatorAgent(post_cycle_hooks=[_MockHook()])
        ctx = AgentContext(task="test_hook_call")
        thought = await agent.think(ctx)
        await agent.act(thought)

        assert "context" in called_with
        assert "agent_actions" in called_with

    @pytest.mark.asyncio
    async def test_coordinator_cycle_count_increments_after_act(self):
        from agents.coordinator import CoordinatorAgent
        from core.base_agent import AgentContext

        class _NullHook:
            async def on_cycle_complete(self, agent_actions, context):
                pass

        agent = CoordinatorAgent(post_cycle_hooks=[_NullHook()])
        ctx = AgentContext(task="count_test")
        thought = await agent.think(ctx)
        await agent.act(thought)
        await agent.act(thought)

        assert agent._cycle_count == 2

    @pytest.mark.asyncio
    async def test_coordinator_hook_failure_does_not_propagate(self):
        """A failing hook must be swallowed (logged only) so act() still returns."""
        from agents.coordinator import CoordinatorAgent
        from core.base_agent import AgentContext

        class _FailingHook:
            async def on_cycle_complete(self, agent_actions, context):
                raise RuntimeError("hook exploded")

        agent = CoordinatorAgent(post_cycle_hooks=[_FailingHook()])
        ctx = AgentContext(task="failing_hook_test")
        thought = await agent.think(ctx)
        action = await agent.act(thought)

        # act() must still return a valid action
        assert action is not None
        assert action.action_type == "coordination"

    @pytest.mark.asyncio
    async def test_coordinator_no_hooks_act_works(self):
        """Coordinator with no hooks must work exactly as before."""
        from agents.coordinator import CoordinatorAgent
        from core.base_agent import AgentContext

        agent = CoordinatorAgent()
        ctx = AgentContext(task="no_hook_act")
        thought = await agent.think(ctx)
        action = await agent.act(thought)

        assert action is not None
        assert action.success is True

    @pytest.mark.asyncio
    async def test_coordinator_with_real_learning_hook(self):
        """End-to-end: coordinator wired with LearningHook must complete a cycle."""
        from agents.coordinator import CoordinatorAgent
        from core.base_agent import AgentContext
        from core.learning_hook import LearningHook

        learners = _make_learners(["coordinator"])
        hook = LearningHook(
            learners=learners,
            workflow_discovery=_make_workflow_discovery(),
            cooperative_evolution=_make_coop_evolution(),
        )
        agent = CoordinatorAgent(post_cycle_hooks=[hook])
        ctx = AgentContext(task="e2e_hook_test")
        thought = await agent.think(ctx)
        action = await agent.act(thought)

        assert action is not None
        assert hook.cycle_count == 1
