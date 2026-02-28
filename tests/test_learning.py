"""
Unit tests for the ATHENA learning layer.

Tests cover: NestedLearning (bilevel meta-learning), RepExp (representation-
based exploration), RepresentationBuffer, TaskTrajectory, and MetaParameters.
All implementations are pure Python — no torch required.
"""

import asyncio
import types
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> types.SimpleNamespace:
    """Create a minimal LearningConfig-like namespace."""
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


def _make_trajectory(task_id="task_a", agent_id="agent_1", n=3) -> "TaskTrajectory":
    from learning.nested_learning import TaskTrajectory
    return TaskTrajectory(
        task_id=task_id,
        agent_id=agent_id,
        states=[{"step": i} for i in range(n)],
        actions=[{"a": i} for i in range(n)],
        rewards=[float(i) * 0.1 for i in range(n)],
    )


# ---------------------------------------------------------------------------
# NestedLearning
# ---------------------------------------------------------------------------

class TestNestedLearning:
    def test_import(self):
        from learning.nested_learning import NestedLearning
        assert NestedLearning is not None

    def test_instantiation(self):
        from learning.nested_learning import NestedLearning
        nl = NestedLearning(config=_make_config(), agent_id="agent_1")
        assert nl is not None
        assert nl.agent_id == "agent_1"

    @pytest.mark.asyncio
    async def test_adapt_to_task_returns_dict(self):
        from learning.nested_learning import NestedLearning
        nl = NestedLearning(config=_make_config(), agent_id="agent_1")
        traj = _make_trajectory("market_analysis")
        result = await nl.adapt_to_task("market_analysis", traj)
        assert isinstance(result, dict)
        assert result["task_id"] == "market_analysis"

    @pytest.mark.asyncio
    async def test_adapt_to_task_stores_trajectory(self):
        from learning.nested_learning import NestedLearning
        nl = NestedLearning(config=_make_config(), agent_id="agent_1")
        traj = _make_trajectory("risk_assessment")
        await nl.adapt_to_task("risk_assessment", traj)
        stored = await nl.get_task_trajectory("risk_assessment")
        assert len(stored) == 1

    @pytest.mark.asyncio
    async def test_update_meta_parameters_with_trajectories(self):
        from learning.nested_learning import NestedLearning
        nl = NestedLearning(config=_make_config(), agent_id="agent_1")
        trajs = [_make_trajectory(f"task_{i}") for i in range(3)]
        result = await nl.update_meta_parameters(trajs)
        assert isinstance(result, dict)
        assert "mean_performance" in result
        assert result["num_tasks"] == 3

    @pytest.mark.asyncio
    async def test_update_meta_parameters_empty(self):
        from learning.nested_learning import NestedLearning
        nl = NestedLearning(config=_make_config(), agent_id="agent_1")
        result = await nl.update_meta_parameters([])
        assert result["num_tasks"] == 0

    @pytest.mark.asyncio
    async def test_consolidate_knowledge(self):
        from learning.nested_learning import NestedLearning
        nl = NestedLearning(config=_make_config(), agent_id="agent_1")
        await nl.adapt_to_task("task_x", _make_trajectory("task_x"))
        result = await nl.consolidate_knowledge()
        assert isinstance(result, dict)
        assert "consolidated_tasks" in result

    @pytest.mark.asyncio
    async def test_get_task_trajectory_none_for_unknown(self):
        from learning.nested_learning import NestedLearning
        nl = NestedLearning(config=_make_config(), agent_id="agent_1")
        result = await nl.get_task_trajectory("nonexistent_task_xyz")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_exploration_weight(self):
        from learning.nested_learning import NestedLearning
        nl = NestedLearning(config=_make_config(), agent_id="agent_1")
        weight = await nl.get_exploration_weight("any_task")
        assert isinstance(weight, float)
        assert 0.0 <= weight <= 1.0

    @pytest.mark.asyncio
    async def test_get_exploration_weight_decays_with_familiarity(self):
        from learning.nested_learning import NestedLearning
        nl = NestedLearning(config=_make_config(), agent_id="agent_1")
        # Add many trajectories for the same task
        for _ in range(30):
            await nl.adapt_to_task("familiar_task", _make_trajectory("familiar_task"))
        weight_known = await nl.get_exploration_weight("familiar_task")
        weight_new = await nl.get_exploration_weight("brand_new_task")
        assert weight_known < weight_new

    def test_get_stats(self):
        from learning.nested_learning import NestedLearning
        nl = NestedLearning(config=_make_config(), agent_id="agent_1")
        stats = nl.get_stats()
        assert isinstance(stats, dict)
        assert "total_tasks" in stats
        assert "total_trajectories" in stats

    @pytest.mark.asyncio
    async def test_save_and_load_state(self, tmp_path):
        from learning.nested_learning import NestedLearning
        nl = NestedLearning(config=_make_config(), agent_id="agent_1")
        await nl.adapt_to_task("t1", _make_trajectory())
        save_path = str(tmp_path / "nl_state.json")
        await nl.save_state(save_path)
        import os
        assert os.path.exists(save_path)

        nl2 = NestedLearning(config=_make_config(), agent_id="agent_1")
        await nl2.load_state(save_path)
        assert nl2.meta_params.param_id == nl.meta_params.param_id


# ---------------------------------------------------------------------------
# TaskTrajectory
# ---------------------------------------------------------------------------

class TestTaskTrajectory:
    def test_import(self):
        from learning.nested_learning import TaskTrajectory
        assert TaskTrajectory is not None

    def test_create_trajectory(self):
        from learning.nested_learning import TaskTrajectory
        traj = TaskTrajectory(
            task_id="market_analysis",
            agent_id="market_analyst",
            states=[{"price": 100}, {"price": 101}],
            actions=[{"signal": "buy"}, {"signal": "hold"}],
            rewards=[0.5, 0.2],
        )
        assert traj.task_id == "market_analysis"
        assert len(traj.states) == 2
        assert len(traj.rewards) == 2

    def test_to_dict(self):
        from learning.nested_learning import TaskTrajectory
        traj = TaskTrajectory(
            task_id="t1", agent_id="a1",
            states=[{}], actions=[{}], rewards=[0.0],
        )
        d = traj.to_dict()
        assert d["task_id"] == "t1"
        assert "states" in d
        assert "rewards" in d

    def test_from_dict(self):
        from learning.nested_learning import TaskTrajectory
        data = {
            "task_id": "t2", "agent_id": "a2",
            "states": [{"x": 1}], "actions": [{"a": 1}],
            "rewards": [1.0], "metadata": {}, "created_at": "2026-01-01T00:00:00+00:00",
        }
        traj = TaskTrajectory.from_dict(data)
        assert traj.task_id == "t2"
        assert traj.rewards == [1.0]


# ---------------------------------------------------------------------------
# MetaParameters
# ---------------------------------------------------------------------------

class TestMetaParameters:
    def test_import(self):
        from learning.nested_learning import MetaParameters
        assert MetaParameters is not None

    def test_create(self):
        from learning.nested_learning import MetaParameters
        mp = MetaParameters(
            param_id="meta_001",
            values={"inner_lr": 0.01, "outer_lr": 0.001},
        )
        assert mp.param_id == "meta_001"
        assert mp.values["inner_lr"] == 0.01

    def test_to_dict(self):
        from learning.nested_learning import MetaParameters
        mp = MetaParameters(param_id="p1", values={"lr": 0.01})
        d = mp.to_dict()
        assert d["param_id"] == "p1"
        assert "values" in d

    def test_from_dict(self):
        from learning.nested_learning import MetaParameters
        data = {
            "param_id": "p2",
            "values": {"lr": 0.001},
            "performance_score": 0.85,
            "update_count": 5,
        }
        mp = MetaParameters.from_dict(data)
        assert mp.param_id == "p2"
        assert mp.performance_score == 0.85
        assert mp.update_count == 5


# ---------------------------------------------------------------------------
# RepresentationBuffer
# ---------------------------------------------------------------------------

class TestRepresentationBuffer:
    def test_import(self):
        from learning.repexp import RepresentationBuffer
        assert RepresentationBuffer is not None

    def test_instantiation(self):
        from learning.repexp import RepresentationBuffer
        buf = RepresentationBuffer(max_size=100, representation_dim=8)
        assert buf is not None
        assert len(buf) == 0

    def test_add_and_len(self):
        from learning.repexp import RepresentationBuffer
        buf = RepresentationBuffer(max_size=10, representation_dim=4)
        buf.add([1.0, 0.0, 0.0, 0.0], agent_id="a1", task_id="t1")
        assert len(buf) == 1

    def test_add_normalizes_vector(self):
        from learning.repexp import RepresentationBuffer
        from math import sqrt
        buf = RepresentationBuffer(max_size=10, representation_dim=4)
        buf.add([3.0, 4.0, 0.0, 0.0], agent_id="a1", task_id="t1")
        entries = buf.get_all()
        norm = sqrt(sum(x * x for x in entries[0]))
        assert abs(norm - 1.0) < 1e-6

    def test_max_size_evicts_old(self):
        from learning.repexp import RepresentationBuffer
        buf = RepresentationBuffer(max_size=3, representation_dim=2)
        for i in range(5):
            buf.add([float(i), 0.0], agent_id="a1", task_id="t1")
        assert len(buf) == 3

    def test_get_by_agent(self):
        from learning.repexp import RepresentationBuffer
        buf = RepresentationBuffer(max_size=10, representation_dim=2)
        buf.add([1.0, 0.0], agent_id="agent_x", task_id="t1")
        buf.add([0.0, 1.0], agent_id="agent_y", task_id="t1")
        results = buf.get_by_agent("agent_x")
        assert len(results) == 1
        # get_by_agent returns representation vectors (list of floats)
        assert isinstance(results[0], list)


# ---------------------------------------------------------------------------
# RepExp
# ---------------------------------------------------------------------------

class TestRepExp:
    def test_import(self):
        from learning.repexp import RepExp
        assert RepExp is not None

    def test_instantiation(self):
        from learning.repexp import RepExp
        exp = RepExp(config=_make_config())
        assert exp is not None

    @pytest.mark.asyncio
    async def test_compute_novelty_empty_buffer(self):
        from learning.repexp import RepExp
        exp = RepExp(config=_make_config())
        novelty = await exp.compute_novelty([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert isinstance(novelty, float)
        assert novelty == 1.0  # max novelty when buffer is empty

    @pytest.mark.asyncio
    async def test_compute_novelty_with_entries(self):
        from learning.repexp import RepExp
        exp = RepExp(config=_make_config())
        for i in range(10):
            exp.buffer.add([float(i), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           agent_id="a1", task_id="t1")
        novelty = await exp.compute_novelty([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        assert isinstance(novelty, float)
        assert 0.0 <= novelty <= 1.0

    @pytest.mark.asyncio
    async def test_compute_exploration_bonus(self):
        from learning.repexp import RepExp
        exp = RepExp(config=_make_config(exploration_coefficient=2.0))
        bonus = await exp.compute_exploration_bonus(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            agent_id="a1",
            task_id="t1",
        )
        assert isinstance(bonus, float)
        assert bonus >= 0.0

    @pytest.mark.asyncio
    async def test_compute_diversity(self):
        from learning.repexp import RepExp
        exp = RepExp(config=_make_config())
        reps = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        diversity = await exp.compute_diversity(reps)
        assert isinstance(diversity, float)
        assert 0.0 <= diversity <= 1.0

    @pytest.mark.asyncio
    async def test_compute_diversity_single(self):
        from learning.repexp import RepExp
        exp = RepExp(config=_make_config())
        diversity = await exp.compute_diversity([[1.0, 0.0]])
        assert diversity == 0.0  # single item → no pairs → 0

    @pytest.mark.asyncio
    async def test_select_diverse_subset_returns_indices(self):
        from learning.repexp import RepExp
        exp = RepExp(config=_make_config())
        reps = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.577, 0.577, 0.577, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        indices = await exp.select_diverse_subset(reps, k=2)
        assert isinstance(indices, list)
        assert len(indices) == 2
        for idx in indices:
            assert 0 <= idx < len(reps)

    @pytest.mark.asyncio
    async def test_get_exploration_strategy(self):
        from learning.repexp import RepExp
        exp = RepExp(config=_make_config())
        result = await exp.get_exploration_strategy(
            agent_id="a1", task_id="t1", current_performance=0.5
        )
        assert isinstance(result, dict)
        assert "strategy" in result
        assert result["strategy"] in ("explore", "exploit")

    def test_get_stats(self):
        from learning.repexp import RepExp
        exp = RepExp(config=_make_config())
        stats = exp.get_stats()
        assert isinstance(stats, dict)
