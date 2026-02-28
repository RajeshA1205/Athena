"""
End-to-End Pipeline Integration Test
=====================================
Validates the full 5-agent ATHENA pipeline with mocked external dependencies
(AgeMem memory backend, LatentMAS router, and torch).  All tests are designed
to run in under 30 seconds total and to be fully deterministic.

Tests:
    test_market_analyst_think_act     — MarketAnalystAgent standalone cycle
    test_risk_manager_think_act       — RiskManagerAgent standalone cycle
    test_strategy_agent_think_act     — StrategyAgent standalone cycle
    test_execution_agent_think_act    — ExecutionAgent standalone cycle
    test_coordinator_think_act        — CoordinatorAgent standalone cycle
    test_full_pipeline                — 5-agent sequential end-to-end run
    test_memory_persistence           — MockAgeMem stores and retrieves correctly
    test_communication_registered     — 4 specialist agents registered with router
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Torch guard: inject a minimal stub before any project import tries to load
# torch.  This keeps the test suite runnable in environments without torch.
# ---------------------------------------------------------------------------
import sys
import types

if "torch" not in sys.modules:
    _torch_stub = types.ModuleType("torch")
    _torch_stub.Tensor = object  # type: ignore[attr-defined]
    _torch_stub.nn = types.ModuleType("torch.nn")
    _torch_stub.nn.Module = object  # type: ignore[attr-defined]

    class _FakeLinear:  # minimal nn.Linear stub
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, x):
            return x

    _torch_stub.nn.Linear = _FakeLinear  # type: ignore[attr-defined]
    _torch_stub.no_grad = lambda: __import__("contextlib").nullcontext()
    _torch_stub.float32 = "float32"
    _torch_stub.zeros = lambda *a, **kw: []  # type: ignore[attr-defined]
    _torch_stub.cat = lambda *a, **kw: []  # type: ignore[attr-defined]
    _torch_stub.stack = lambda *a, **kw: []  # type: ignore[attr-defined]
    sys.modules["torch"] = _torch_stub
    sys.modules["torch.nn"] = _torch_stub.nn

import pytest

from core.base_agent import AgentContext, AgentAction
from agents.market_analyst import MarketAnalystAgent
from agents.risk_manager import RiskManagerAgent
from agents.strategy_agent import StrategyAgent
from agents.execution_agent import ExecutionAgent
from agents.coordinator import CoordinatorAgent


# ---------------------------------------------------------------------------
# Mock market / portfolio data
# ---------------------------------------------------------------------------

MOCK_MARKET_DATA = {
    "symbol": "AAPL",
    "price": 150.0,
    "prices": [145.0 + i * 0.5 for i in range(60)],   # 60-day uptrend
    "volumes": [1_000_000 + i * 1000 for i in range(60)],
    "high": [146.0 + i * 0.5 for i in range(60)],
    "low": [144.0 + i * 0.5 for i in range(60)],
    "timestamp": "2026-02-22T00:00:00Z",
}

MOCK_PORTFOLIO = {
    "total_value": 100_000.0,
    "cash": 50_000.0,
    "positions": {"AAPL": {"quantity": 100, "avg_price": 140.0}},
    "returns": [0.01, -0.005, 0.02, -0.01, 0.015, 0.008, -0.003, 0.012],
}


# ---------------------------------------------------------------------------
# MockAgeMem
# ---------------------------------------------------------------------------

class MockAgeMem:
    """
    In-memory drop-in replacement for AgeMem.

    Provides async add / retrieve / summary / filter / initialize methods that
    record all stored entries in a plain Python list for easy assertion in tests.
    Does NOT connect to any external service.
    """

    def __init__(self) -> None:
        self.entries: list = []
        self._initialized: bool = False

    async def initialize(self) -> bool:
        """Mark the mock as initialised."""
        self._initialized = True
        return True

    async def add(self, content, metadata=None) -> bool:
        """Append a new entry and return True."""
        self.entries.append({"content": content, "metadata": metadata})
        return True

    async def retrieve(self, query: str, top_k: int = 5) -> list:
        """Return the most recent min(top_k, len(entries)) entries."""
        count = min(top_k, len(self.entries))
        return self.entries[-count:] if count > 0 else []

    async def summary(self, context=None) -> str:
        """Return a fixed summary string."""
        return "memory_summary"

    async def filter(self, context=None, relevance_threshold: float = 0.5) -> list:
        """Return context unchanged (or empty list)."""
        return context or []


# ---------------------------------------------------------------------------
# MockRouter
# ---------------------------------------------------------------------------

class MockRouter:
    """
    In-memory drop-in replacement for MessageRouter.

    Records every send call in self.sent and every register_agent call in
    self.registered.  receive() always returns an empty list so agents do not
    block waiting for real messages.
    """

    def __init__(self) -> None:
        self.sent: list = []
        self.registered: dict = {}

    async def send(self, sender_id, receiver_id, message, priority=None) -> bool:
        """Log the outgoing message and return True."""
        self.sent.append(
            {
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "message": message,
                "priority": priority,
            }
        )
        return True

    async def receive(self, receiver_id, decode_mode="numeric") -> list:
        """Return an empty list — no real messages in tests."""
        return []

    def register_agent(self, agent_id, agent_info) -> None:
        """Record agent registration."""
        self.registered[agent_id] = agent_info

    def get_routing_stats(self) -> dict:
        """Return basic stats about registered agents."""
        return {"registered_agents": len(self.registered)}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def agent_system():
    """
    Synchronous fixture that creates all 5 agents sharing one MockAgeMem and
    one MockRouter, then yields a dict of all components.

    The coordinator is pre-wired with the mock router and has all 4 specialist
    agents registered.  Each specialist also gets the mock router injected so
    their think()/act() LatentMAS paths are exercised.
    """
    mock_memory = MockAgeMem()
    mock_router = MockRouter()

    analyst = MarketAnalystAgent(memory=mock_memory)
    risk_manager = RiskManagerAgent(memory=mock_memory)
    strategy = StrategyAgent(memory=mock_memory)
    executor = ExecutionAgent(memory=mock_memory)
    coordinator = CoordinatorAgent(memory=mock_memory)

    # Register specialist agents with coordinator
    coordinator.register_agent("market_analyst", analyst)
    coordinator.register_agent("risk_manager", risk_manager)
    coordinator.register_agent("strategy_agent", strategy)
    coordinator.register_agent("execution_agent", executor)

    # Inject the mock router everywhere
    coordinator.router = mock_router
    analyst.router = mock_router
    risk_manager.router = mock_router
    strategy.router = mock_router
    executor.router = mock_router

    # Register each specialist with the mock router (mirrors what
    # initialize_communication() would do with a real LatentSpace)
    for name, agent in coordinator.agents.items():
        mock_router.register_agent(name, {"role": agent.role})

    yield {
        "analyst": analyst,
        "risk_manager": risk_manager,
        "strategy": strategy,
        "executor": executor,
        "coordinator": coordinator,
        "mock_memory": mock_memory,
        "mock_router": mock_router,
    }


# ---------------------------------------------------------------------------
# Individual agent tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_market_analyst_think_act(agent_system):
    """
    MarketAnalystAgent should complete a full think-act cycle using market
    data provided in the AgentContext metadata, store at least one memory
    entry, and return a successful AgentAction.
    """
    analyst: MarketAnalystAgent = agent_system["analyst"]
    mock_memory: MockAgeMem = agent_system["mock_memory"]

    context = AgentContext(
        task="Analyse AAPL market conditions",
        metadata={"market_data": MOCK_MARKET_DATA},
    )

    thought = await analyst.think(context)
    assert isinstance(thought, dict), "think() must return a dict"
    assert "memory_context" in thought, "think() result must include 'memory_context'"

    action: AgentAction = await analyst.act(thought)
    assert action.success is True, f"act() failed: {action.error}"
    assert action.result is not None, "act() result must not be None"

    assert len(mock_memory.entries) >= 1, (
        "MarketAnalystAgent must store at least one memory entry after act()"
    )


@pytest.mark.asyncio
async def test_risk_manager_think_act(agent_system):
    """
    RiskManagerAgent should complete a full think-act cycle using portfolio
    data provided in the AgentContext metadata, store at least one memory
    entry, and return a successful AgentAction with a result dict.
    """
    risk_manager: RiskManagerAgent = agent_system["risk_manager"]
    mock_memory: MockAgeMem = agent_system["mock_memory"]
    entries_before = len(mock_memory.entries)

    # RiskManagerAgent reads 'positions' and 'returns' from metadata.
    # Passing an empty positions list triggers the early-return path which is
    # also a valid success case — so we provide a minimal portfolio to exercise
    # the full computation path.
    context = AgentContext(
        task="Assess portfolio risk",
        metadata={
            "portfolio": MOCK_PORTFOLIO,
            "positions": [
                {"symbol": "AAPL", "value": 14000.0, "sector": "Technology"},
            ],
            "returns": {"AAPL": MOCK_PORTFOLIO["returns"]},
        },
    )

    thought = await risk_manager.think(context)
    assert isinstance(thought, dict), "think() must return a dict"
    assert "memory_context" in thought, "think() result must include 'memory_context'"
    assert "risk_level" in thought, "think() result must include 'risk_level'"

    action: AgentAction = await risk_manager.act(thought)
    assert action.success is True, f"act() failed: {action.error}"
    assert action.result is not None, "act() result must not be None"
    assert "risk_level" in action.result, "result must include 'risk_level'"

    assert len(mock_memory.entries) > entries_before, (
        "RiskManagerAgent must store at least one memory entry after act()"
    )


@pytest.mark.asyncio
async def test_strategy_agent_think_act(agent_system):
    """
    StrategyAgent should complete a full think-act cycle using market data
    in context metadata, store at least one memory entry, and return a
    successful AgentAction.
    """
    strategy: StrategyAgent = agent_system["strategy"]
    mock_memory: MockAgeMem = agent_system["mock_memory"]
    entries_before = len(mock_memory.entries)

    context = AgentContext(
        task="Generate trading strategy for AAPL",
        metadata={"market_data": MOCK_MARKET_DATA},
    )

    thought = await strategy.think(context)
    assert isinstance(thought, dict), "think() must return a dict"
    assert "memory_context" in thought, "think() result must include 'memory_context'"
    assert "strategy" in thought, "think() result must include 'strategy'"

    action: AgentAction = await strategy.act(thought)
    assert action.success is True, f"act() failed: {action.error}"

    assert len(mock_memory.entries) > entries_before, (
        "StrategyAgent must store at least one memory entry after act()"
    )


@pytest.mark.asyncio
async def test_execution_agent_think_act(agent_system):
    """
    ExecutionAgent should complete a full think-act cycle given a buy order
    request in the AgentContext metadata, store at least one memory entry,
    and return a successful AgentAction.
    """
    executor: ExecutionAgent = agent_system["executor"]
    mock_memory: MockAgeMem = agent_system["mock_memory"]
    entries_before = len(mock_memory.entries)

    context = AgentContext(
        task="Execute buy order for AAPL",
        metadata={
            "trade_request": {
                "action": "buy",
                "quantity": 10,
                "symbol": "AAPL",
            },
            "market_conditions": {
                "current_price": MOCK_MARKET_DATA["price"],
                "avg_volume": 1_000_000.0,
                "volatility": 0.02,
            },
        },
    )

    thought = await executor.think(context)
    assert isinstance(thought, dict), "think() must return a dict"
    assert "memory_context" in thought, "think() result must include 'memory_context'"
    assert "execution_plan" in thought, "think() result must include 'execution_plan'"

    action: AgentAction = await executor.act(thought)
    assert action.success is True, f"act() failed: {action.error}"
    assert action.result is not None, "act() result must not be None"

    assert len(mock_memory.entries) > entries_before, (
        "ExecutionAgent must store at least one memory entry after act()"
    )


@pytest.mark.asyncio
async def test_coordinator_think_act(agent_system):
    """
    CoordinatorAgent should complete a full think-act cycle given a
    coordination task, store at least one memory entry, and return a
    successful AgentAction.
    """
    coordinator: CoordinatorAgent = agent_system["coordinator"]
    mock_memory: MockAgeMem = agent_system["mock_memory"]
    entries_before = len(mock_memory.entries)

    context = AgentContext(
        task="Coordinate agents for AAPL analysis",
        metadata={
            "task": {
                "type": "coordinate",
                "agents": ["market_analyst", "risk_manager"],
            }
        },
    )

    thought = await coordinator.think(context)
    assert isinstance(thought, dict), "think() must return a dict"
    assert "memory_context" in thought, "think() result must include 'memory_context'"

    action: AgentAction = await coordinator.act(thought)
    assert action.success is True, f"act() failed: {action.error}"
    assert action.result is not None, "act() result must not be None"
    assert isinstance(action.result, dict), "act() result must be a dict"

    assert len(mock_memory.entries) > entries_before, (
        "CoordinatorAgent must store at least one memory entry after act()"
    )


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_pipeline(agent_system):
    """
    End-to-end sequential pipeline test.

    Runs the full 5-agent pipeline in the natural order:
        MarketAnalyst -> RiskManager -> StrategyAgent -> ExecutionAgent -> Coordinator

    All agents share the same MockAgeMem instance.  After the pipeline:
      - Each agent must have returned a successful AgentAction.
      - The shared memory must contain at least 5 entries (one per agent).
      - The coordinator action must include a result dict.
    """
    analyst: MarketAnalystAgent = agent_system["analyst"]
    risk_manager: RiskManagerAgent = agent_system["risk_manager"]
    strategy: StrategyAgent = agent_system["strategy"]
    executor: ExecutionAgent = agent_system["executor"]
    coordinator: CoordinatorAgent = agent_system["coordinator"]
    mock_memory: MockAgeMem = agent_system["mock_memory"]

    actions: list[AgentAction] = []

    # --- 1. Market Analyst ---
    analyst_context = AgentContext(
        task="Analyse AAPL market conditions",
        metadata={"market_data": MOCK_MARKET_DATA},
    )
    analyst_thought = await analyst.think(analyst_context)
    analyst_action = await analyst.act(analyst_thought)
    actions.append(analyst_action)

    # --- 2. Risk Manager ---
    risk_context = AgentContext(
        task="Assess portfolio risk for AAPL position",
        metadata={
            "portfolio": MOCK_PORTFOLIO,
            "positions": [
                {"symbol": "AAPL", "value": 14000.0, "sector": "Technology"},
            ],
            "returns": {"AAPL": MOCK_PORTFOLIO["returns"]},
        },
    )
    risk_thought = await risk_manager.think(risk_context)
    risk_action = await risk_manager.act(risk_thought)
    actions.append(risk_action)

    # --- 3. Strategy Agent ---
    strategy_context = AgentContext(
        task="Generate trading strategy for AAPL",
        metadata={"market_data": MOCK_MARKET_DATA},
    )
    strategy_thought = await strategy.think(strategy_context)
    strategy_action = await strategy.act(strategy_thought)
    actions.append(strategy_action)

    # --- 4. Execution Agent ---
    exec_context = AgentContext(
        task="Execute buy order for AAPL",
        metadata={
            "trade_request": {
                "action": "buy",
                "quantity": 10,
                "symbol": "AAPL",
            },
            "market_conditions": {
                "current_price": MOCK_MARKET_DATA["price"],
                "avg_volume": 1_000_000.0,
                "volatility": 0.02,
            },
        },
    )
    exec_thought = await executor.think(exec_context)
    exec_action = await executor.act(exec_thought)
    actions.append(exec_action)

    # --- 5. Coordinator ---
    coord_context = AgentContext(
        task="Coordinate agents and make final decision for AAPL",
        metadata={
            "task": {
                "type": "coordinate",
                "agents": ["market_analyst", "risk_manager", "strategy_agent", "execution_agent"],
            }
        },
    )
    coord_thought = await coordinator.think(coord_context)
    coord_action = await coordinator.act(coord_thought)
    actions.append(coord_action)

    # --- Assertions ---
    for i, action in enumerate(actions):
        assert action.success is True, (
            f"Agent #{i + 1} action failed: {action.error}"
        )

    assert len(mock_memory.entries) >= 5, (
        f"Expected at least 5 memory entries after full pipeline, "
        f"got {len(mock_memory.entries)}"
    )

    coord_result = coord_action.result
    assert coord_result is not None, "Coordinator action must have a non-None result"
    assert isinstance(coord_result, dict), "Coordinator result must be a dict"


# ---------------------------------------------------------------------------
# Memory persistence test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_memory_persistence(agent_system):
    """
    MockAgeMem must store entries via add() and make them accessible via
    retrieve().  The retrieved entries must contain the content that was
    stored.
    """
    mock_memory: MockAgeMem = agent_system["mock_memory"]

    observation = {"observation": "AAPL trending up"}
    meta = {"agent": "test"}
    stored = await mock_memory.add(observation, meta)
    assert stored is True, "add() must return True"

    results = await mock_memory.retrieve("AAPL analysis")
    assert len(results) > 0, "retrieve() must return a non-empty list after add()"

    # The most-recently stored entry should be in the results
    contents = [entry.get("content") for entry in results]
    assert observation in contents, (
        "The stored content must be accessible via retrieve()"
    )


# ---------------------------------------------------------------------------
# Communication registration test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_communication_registered(agent_system):
    """
    After fixture setup, the MockRouter must have exactly 4 specialist agents
    registered (market_analyst, risk_manager, strategy_agent, execution_agent).
    The coordinator itself is not registered as a receiver.
    """
    mock_router: MockRouter = agent_system["mock_router"]
    coordinator: CoordinatorAgent = agent_system["coordinator"]

    # The fixture registers all entries in coordinator.agents
    assert len(mock_router.registered) == 4, (
        f"Expected 4 registered agents, got {len(mock_router.registered)}: "
        f"{list(mock_router.registered.keys())}"
    )

    # Confirm the coordinator's internal agent registry also has 4 entries
    assert len(coordinator.agents) == 4, (
        f"Expected 4 agents in coordinator.agents, got {len(coordinator.agents)}"
    )
