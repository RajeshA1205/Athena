"""
Unit tests for ATHENA agent layer.

Tests cover MarketAnalystAgent, RiskManagerAgent, StrategyAgent,
ExecutionAgent, and CoordinatorAgent without external dependencies.
AgeMem and MessageRouter are mocked throughout.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(task: str = "analyze_market") -> "AgentContext":
    from core.base_agent import AgentContext
    return AgentContext(
        task=task,
        history=[],
        memory_context=[],
        messages=[],
        metadata={
            "market_data": {
                "AAPL": {"price": 185.0, "volume": 1_000_000, "prices": [180.0, 182.0, 185.0]},
            },
            "portfolio": {"cash": 50_000, "positions": {}},
            "risk_metrics": {},
        },
    )


@pytest.fixture
def mock_memory():
    mem = MagicMock()
    mem.retrieve = AsyncMock(return_value=[])
    mem.add = AsyncMock(return_value=None)
    return mem


@pytest.fixture
def mock_router():
    router = MagicMock()
    router.send = AsyncMock(return_value=None)
    router.receive = AsyncMock(return_value=[])
    return router


# ---------------------------------------------------------------------------
# MarketAnalystAgent
# ---------------------------------------------------------------------------

class TestMarketAnalystAgent:
    def test_import(self):
        from agents.market_analyst import MarketAnalystAgent
        assert MarketAnalystAgent is not None

    def test_instantiation(self):
        from agents.market_analyst import MarketAnalystAgent
        agent = MarketAnalystAgent()
        assert agent is not None
        assert agent.role == "analyst"

    def test_instantiation_with_memory(self, mock_memory):
        from agents.market_analyst import MarketAnalystAgent
        agent = MarketAnalystAgent(memory=mock_memory)
        assert agent.memory is mock_memory

    def test_instantiation_with_router(self, mock_router):
        from agents.market_analyst import MarketAnalystAgent
        agent = MarketAnalystAgent(router=mock_router)
        assert agent.router is mock_router

    @pytest.mark.asyncio
    async def test_think_returns_dict(self):
        from agents.market_analyst import MarketAnalystAgent
        agent = MarketAnalystAgent()
        result = await agent.think(_make_context())
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_think_with_memory(self, mock_memory):
        from agents.market_analyst import MarketAnalystAgent
        agent = MarketAnalystAgent(memory=mock_memory)
        result = await agent.think(_make_context())
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_act_returns_action(self):
        from agents.market_analyst import MarketAnalystAgent
        agent = MarketAnalystAgent()
        thoughts = await agent.think(_make_context())
        action = await agent.act(thoughts)
        assert action is not None

    @pytest.mark.asyncio
    async def test_act_with_router(self, mock_router):
        from agents.market_analyst import MarketAnalystAgent
        agent = MarketAnalystAgent(router=mock_router)
        thoughts = await agent.think(_make_context())
        action = await agent.act(thoughts)
        assert action is not None

    @pytest.mark.asyncio
    async def test_think_act_roundtrip(self):
        from agents.market_analyst import MarketAnalystAgent
        from core.base_agent import AgentAction
        agent = MarketAnalystAgent()
        ctx = _make_context()
        thoughts = await agent.think(ctx)
        action = await agent.act(thoughts)
        assert isinstance(action, AgentAction)


# ---------------------------------------------------------------------------
# RiskManagerAgent
# ---------------------------------------------------------------------------

class TestRiskManagerAgent:
    def test_import(self):
        from agents.risk_manager import RiskManagerAgent
        assert RiskManagerAgent is not None

    def test_instantiation(self):
        from agents.risk_manager import RiskManagerAgent
        agent = RiskManagerAgent()
        assert agent.role == "risk"

    @pytest.mark.asyncio
    async def test_think_returns_dict(self):
        from agents.risk_manager import RiskManagerAgent
        agent = RiskManagerAgent()
        result = await agent.think(_make_context("assess_risk"))
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_act_returns_action(self):
        from agents.risk_manager import RiskManagerAgent
        agent = RiskManagerAgent()
        thoughts = await agent.think(_make_context("assess_risk"))
        action = await agent.act(thoughts)
        assert action is not None

    @pytest.mark.asyncio
    async def test_think_with_memory_and_router(self, mock_memory, mock_router):
        from agents.risk_manager import RiskManagerAgent
        agent = RiskManagerAgent(memory=mock_memory, router=mock_router)
        result = await agent.think(_make_context("assess_risk"))
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# StrategyAgent
# ---------------------------------------------------------------------------

class TestStrategyAgent:
    def test_import(self):
        from agents.strategy_agent import StrategyAgent
        assert StrategyAgent is not None

    def test_instantiation(self):
        from agents.strategy_agent import StrategyAgent
        agent = StrategyAgent()
        assert agent.role == "strategy"

    @pytest.mark.asyncio
    async def test_think_returns_dict(self):
        from agents.strategy_agent import StrategyAgent
        agent = StrategyAgent()
        result = await agent.think(_make_context("generate_strategy"))
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_act_returns_action(self):
        from agents.strategy_agent import StrategyAgent
        agent = StrategyAgent()
        thoughts = await agent.think(_make_context("generate_strategy"))
        action = await agent.act(thoughts)
        assert action is not None

    @pytest.mark.asyncio
    async def test_think_with_memory(self, mock_memory):
        from agents.strategy_agent import StrategyAgent
        agent = StrategyAgent(memory=mock_memory)
        result = await agent.think(_make_context("generate_strategy"))
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# ExecutionAgent
# ---------------------------------------------------------------------------

class TestExecutionAgent:
    def test_import(self):
        from agents.execution_agent import ExecutionAgent
        assert ExecutionAgent is not None

    def test_instantiation(self):
        from agents.execution_agent import ExecutionAgent
        agent = ExecutionAgent()
        assert agent.role == "execution"

    @pytest.mark.asyncio
    async def test_think_returns_dict(self):
        from agents.execution_agent import ExecutionAgent
        agent = ExecutionAgent()
        result = await agent.think(_make_context("execute_order"))
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_act_returns_action(self):
        from agents.execution_agent import ExecutionAgent
        agent = ExecutionAgent()
        thoughts = await agent.think(_make_context("execute_order"))
        action = await agent.act(thoughts)
        assert action is not None

    @pytest.mark.asyncio
    async def test_think_with_router(self, mock_router):
        from agents.execution_agent import ExecutionAgent
        agent = ExecutionAgent(router=mock_router)
        result = await agent.think(_make_context("execute_order"))
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# CoordinatorAgent
# ---------------------------------------------------------------------------

class TestCoordinatorAgent:
    def test_import(self):
        from agents.coordinator import CoordinatorAgent
        assert CoordinatorAgent is not None

    def test_instantiation(self):
        from agents.coordinator import CoordinatorAgent
        agent = CoordinatorAgent()
        assert agent is not None
        assert agent.role == "coordinator"

    @pytest.mark.asyncio
    async def test_think_returns_dict(self):
        from agents.coordinator import CoordinatorAgent
        agent = CoordinatorAgent()
        result = await agent.think(_make_context("coordinate"))
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_act_returns_action(self):
        from agents.coordinator import CoordinatorAgent
        agent = CoordinatorAgent()
        thoughts = await agent.think(_make_context("coordinate"))
        action = await agent.act(thoughts)
        assert action is not None

    @pytest.mark.asyncio
    async def test_think_with_memory(self, mock_memory):
        from agents.coordinator import CoordinatorAgent
        agent = CoordinatorAgent(memory=mock_memory)
        result = await agent.think(_make_context("coordinate"))
        assert isinstance(result, dict)

    def test_has_agents_dict(self):
        from agents.coordinator import CoordinatorAgent
        agent = CoordinatorAgent()
        assert hasattr(agent, "agents")
        assert isinstance(agent.agents, dict)
