"""
Unit tests for the ATHENA evolution layer.

Tests cover: WorkflowDiscovery, AgentGenerator, CooperativeEvolution,
and their dataclasses. All tests use pure Python without torch dependencies.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_execution_trace(success: bool = True) -> dict:
    return {
        "agents": ["market_analyst", "risk_manager"],
        "interactions": [
            {"from": "market_analyst", "to": "risk_manager", "type": "signal"},
        ],
        "outcome": {"success": success, "pnl": 0.05},
        "metadata": {"regime": "trending"},
    }


# ---------------------------------------------------------------------------
# WorkflowDiscovery
# ---------------------------------------------------------------------------

class TestWorkflowDiscovery:
    def test_import(self):
        from evolution.workflow_discovery import WorkflowDiscovery
        assert WorkflowDiscovery is not None

    def test_instantiation(self):
        from evolution.workflow_discovery import WorkflowDiscovery
        wd = WorkflowDiscovery()
        assert wd is not None

    def test_instantiation_with_config(self):
        from evolution.workflow_discovery import WorkflowDiscovery
        wd = WorkflowDiscovery(config={"max_patterns": 50})
        assert wd is not None

    @pytest.mark.asyncio
    async def test_analyze_execution_returns_pattern_id(self):
        from evolution.workflow_discovery import WorkflowDiscovery
        wd = WorkflowDiscovery()
        result = await wd.analyze_execution(_make_execution_trace(success=True))
        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_analyze_execution_no_agents_returns_none(self):
        from evolution.workflow_discovery import WorkflowDiscovery
        wd = WorkflowDiscovery()
        result = await wd.analyze_execution({"interactions": [], "outcome": {"success": True}})
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_execution_accumulates_patterns(self):
        from evolution.workflow_discovery import WorkflowDiscovery
        wd = WorkflowDiscovery()
        for _ in range(3):
            await wd.analyze_execution(_make_execution_trace(success=True))
        stats = wd.get_stats()
        assert stats["total_patterns"] >= 1

    @pytest.mark.asyncio
    async def test_get_successful_patterns(self):
        from evolution.workflow_discovery import WorkflowDiscovery
        wd = WorkflowDiscovery(config={"min_success_rate": 0.0})
        await wd.analyze_execution(_make_execution_trace(success=True))
        patterns = await wd.get_successful_patterns()
        assert isinstance(patterns, list)

    def test_get_stats(self):
        from evolution.workflow_discovery import WorkflowDiscovery
        wd = WorkflowDiscovery()
        stats = wd.get_stats()
        assert isinstance(stats, dict)

    def test_workflow_pattern_dataclass(self):
        from evolution.workflow_discovery import WorkflowPattern
        pattern = WorkflowPattern(
            pattern_id="pat_001",
            agent_sequence=["market_analyst", "risk_manager"],
            interaction_pattern={"type": "sequential"},
            success_rate=0.75,
            use_count=10,
        )
        assert pattern.pattern_id == "pat_001"
        assert len(pattern.agent_sequence) == 2
        assert pattern.success_rate == 0.75

    def test_workflow_pattern_to_dict(self):
        from evolution.workflow_discovery import WorkflowPattern
        pattern = WorkflowPattern(
            pattern_id="pat_002",
            agent_sequence=["strategy"],
            interaction_pattern={},
        )
        d = pattern.to_dict()
        assert d["pattern_id"] == "pat_002"
        assert "agent_sequence" in d


# ---------------------------------------------------------------------------
# AgentGenerator
# ---------------------------------------------------------------------------

class TestAgentGenerator:
    def _make_generator(self):
        from evolution.workflow_discovery import WorkflowDiscovery
        from evolution.agent_generator import AgentGenerator
        wd = WorkflowDiscovery()
        return AgentGenerator(workflow_discovery=wd, config={})

    def test_import(self):
        from evolution.agent_generator import AgentGenerator
        assert AgentGenerator is not None

    def test_instantiation(self):
        gen = self._make_generator()
        assert gen is not None

    @pytest.mark.asyncio
    async def test_generate_from_pattern_returns_config(self):
        from evolution.workflow_discovery import WorkflowPattern
        gen = self._make_generator()
        pattern = WorkflowPattern(
            pattern_id="pat_001",
            agent_sequence=["market_analyst", "risk_manager"],
            interaction_pattern={},
            success_rate=0.9,
            use_count=5,
        )
        config = await gen.generate_from_pattern(pattern)
        assert config is not None
        assert hasattr(config, "config_id")
        assert hasattr(config, "agent_type")

    @pytest.mark.asyncio
    async def test_generate_unique_ids(self):
        from evolution.workflow_discovery import WorkflowPattern
        gen = self._make_generator()
        ids = set()
        for i in range(5):
            pattern = WorkflowPattern(
                pattern_id=f"pat_{i}",
                agent_sequence=["risk_manager"],
                interaction_pattern={},
                success_rate=0.8,
                use_count=1,
            )
            config = await gen.generate_from_pattern(pattern)
            ids.add(config.config_id)
        assert len(ids) == 5

    @pytest.mark.asyncio
    async def test_generate_unique_ids_after_prune(self):
        from evolution.workflow_discovery import WorkflowPattern
        gen = self._make_generator()

        ids_before = []
        for i in range(3):
            pattern = WorkflowPattern(
                pattern_id=f"pre_{i}",
                agent_sequence=["strategy"],
                interaction_pattern={},
                success_rate=0.85,
                use_count=2,
            )
            config = await gen.generate_from_pattern(pattern)
            ids_before.append(config.config_id)

        # Simulate pruning by clearing internal dict
        gen.generated_configs.clear()

        ids_after = []
        for i in range(3):
            pattern = WorkflowPattern(
                pattern_id=f"post_{i}",
                agent_sequence=["execution"],
                interaction_pattern={},
                success_rate=0.9,
                use_count=3,
            )
            config = await gen.generate_from_pattern(pattern)
            ids_after.append(config.config_id)

        assert set(ids_before).isdisjoint(set(ids_after))

    def test_get_stats(self):
        gen = self._make_generator()
        stats = gen.get_stats()
        assert isinstance(stats, dict)

    def test_agent_configuration_to_dict(self):
        from evolution.agent_generator import AgentConfiguration
        cfg = AgentConfiguration(
            config_id="cfg_001",
            agent_type="market_analyst",
            capabilities=["technical_analysis", "sentiment"],
        )
        d = cfg.to_dict()
        assert d["config_id"] == "cfg_001"
        assert d["agent_type"] == "market_analyst"

    def test_agent_configuration_from_dict(self):
        from evolution.agent_generator import AgentConfiguration
        data = {
            "config_id": "cfg_002",
            "agent_type": "risk_manager",
            "capabilities": ["risk_assessment"],
            "parameters": {"threshold": 0.05},
            "performance_score": 0.72,
            "source_pattern": "pat_001",
            "metadata": {},
        }
        cfg = AgentConfiguration.from_dict(data)
        assert cfg.config_id == "cfg_002"
        assert cfg.agent_type == "risk_manager"


# ---------------------------------------------------------------------------
# CooperativeEvolution
# ---------------------------------------------------------------------------

class TestCooperativeEvolution:
    def test_import(self):
        from evolution.cooperative_evolution import CooperativeEvolution
        assert CooperativeEvolution is not None

    def test_instantiation(self):
        from evolution.cooperative_evolution import CooperativeEvolution
        ce = CooperativeEvolution(config={})
        assert ce is not None

    @pytest.mark.asyncio
    async def test_add_experience(self):
        from evolution.cooperative_evolution import CooperativeEvolution, Experience
        ce = CooperativeEvolution(config={"min_reward_threshold": 0.0})
        exp = Experience(
            agent_id="market_analyst",
            state={"price": 185.0},
            action={"signal": "buy"},
            outcome={"pnl": 0.05},
            reward=0.5,
        )
        await ce.add_experience("market_analyst", exp)
        stats = await ce.get_population_stats()
        assert stats["total_agents"] >= 1

    @pytest.mark.asyncio
    async def test_replay_experiences(self):
        from evolution.cooperative_evolution import CooperativeEvolution, Experience
        ce = CooperativeEvolution(config={"replay_batch_size": 5})
        for i in range(10):
            exp = Experience(
                agent_id="risk_manager",
                state={"risk": float(i)},
                action={"approved": True},
                outcome={"result": "ok"},
                reward=float(i) * 0.1,
            )
            await ce.add_experience("risk_manager", exp)
        batch = await ce.replay_experiences("risk_manager", batch_size=3)
        assert isinstance(batch, list)
        assert len(batch) <= 3

    @pytest.mark.asyncio
    async def test_share_knowledge_between_agents(self):
        from evolution.cooperative_evolution import CooperativeEvolution, Experience
        ce = CooperativeEvolution(config={"min_reward_threshold": 0.0})
        exp_a = Experience("agent_a", {"x": 1}, {"a": 1}, {"ok": True}, 0.8)
        exp_b = Experience("agent_b", {"x": 3}, {"b": 1}, {"ok": True}, 0.6)
        await ce.add_experience("agent_a", exp_a)
        await ce.add_experience("agent_b", exp_b)
        result = await ce.share_knowledge(source_agent="agent_a", target_agent="agent_b")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_population_stats(self):
        from evolution.cooperative_evolution import CooperativeEvolution
        ce = CooperativeEvolution(config={})
        stats = await ce.get_population_stats()
        assert isinstance(stats, dict)
        assert "total_agents" in stats

    @pytest.mark.asyncio
    async def test_cross_pollinate(self):
        from evolution.cooperative_evolution import CooperativeEvolution, Experience
        ce = CooperativeEvolution(config={"min_reward_threshold": 0.0})
        for ag in ["a1", "a2", "a3"]:
            exp = Experience(ag, {"s": 1}, {"a": 1}, {"ok": True}, 0.9)
            await ce.add_experience(ag, exp)
        result = await ce.cross_pollinate(top_k=2)
        assert isinstance(result, dict)

    def test_experience_dataclass(self):
        from evolution.cooperative_evolution import Experience
        exp = Experience(
            agent_id="test",
            state={"s": 1},
            action={"a": 1},
            outcome={"result": "ok"},
            reward=0.5,
        )
        assert exp.agent_id == "test"
        assert exp.reward == 0.5
        assert isinstance(exp.experience_id, str)

    def test_experience_to_dict(self):
        from evolution.cooperative_evolution import Experience
        exp = Experience("a1", {}, {}, {}, 0.3)
        d = exp.to_dict()
        assert d["agent_id"] == "a1"
        assert d["reward"] == 0.3

    def test_experience_from_dict(self):
        from evolution.cooperative_evolution import Experience
        original = Experience("a2", {"s": 1}, {"a": 1}, {"r": "ok"}, 0.7)
        d = original.to_dict()
        restored = Experience.from_dict(d)
        assert restored.agent_id == "a2"
        assert restored.reward == 0.7
        assert restored.experience_id == original.experience_id
