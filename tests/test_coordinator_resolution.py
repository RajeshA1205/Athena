"""Tests for CoordinatorAgent conflict detection and priority-weighted resolution."""
import sys
import types

import pytest

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.nn = types.SimpleNamespace(Module=object, Linear=object)
    torch_stub.tensor = lambda *a, **kw: None
    torch_stub.no_grad = lambda: __import__("contextlib").nullcontext()
    sys.modules["torch"] = torch_stub


from agents.coordinator import CoordinatorAgent  # noqa: E402
from agents.risk_manager import RiskManagerAgent  # noqa: E402
from agents.strategy_agent import StrategyAgent  # noqa: E402
from agents.market_analyst import MarketAnalystAgent  # noqa: E402


@pytest.fixture
def coordinator():
    c = CoordinatorAgent()
    c.register_agent("risk_manager", RiskManagerAgent())
    c.register_agent("strategy_agent", StrategyAgent())
    c.register_agent("market_analyst", MarketAnalystAgent())
    return c


# ---------------------------------------------------------------------------
# _detect_conflicts
# ---------------------------------------------------------------------------
class TestDetectConflicts:
    def test_buy_vs_sell_is_conflict(self, coordinator):
        recs = {
            "risk_manager": {"action": "sell", "confidence": 0.8},
            "strategy_agent": {"action": "buy", "confidence": 0.7},
        }
        conflicts = coordinator._detect_conflicts(recs)
        assert len(conflicts) >= 1

    def test_all_same_action_no_conflict(self, coordinator):
        recs = {
            "risk_manager": {"action": "hold", "confidence": 0.6},
            "strategy_agent": {"action": "hold", "confidence": 0.7},
            "market_analyst": {"action": "hold", "confidence": 0.5},
        }
        conflicts = coordinator._detect_conflicts(recs)
        assert conflicts == []

    def test_buy_vs_hold_no_conflict(self, coordinator):
        # Only buy vs sell triggers a direct conflict
        recs = {
            "risk_manager": {"action": "hold", "confidence": 0.6},
            "strategy_agent": {"action": "buy", "confidence": 0.7},
        }
        conflicts = coordinator._detect_conflicts(recs)
        assert conflicts == []

    def test_missing_action_key_ignored(self, coordinator):
        recs = {
            "risk_manager": {"confidence": 0.8},  # no "action"
            "strategy_agent": {"action": "buy", "confidence": 0.7},
        }
        conflicts = coordinator._detect_conflicts(recs)
        assert conflicts == []


# ---------------------------------------------------------------------------
# _resolve_conflicts
# ---------------------------------------------------------------------------
class TestResolveConflicts:
    async def test_risk_beats_strategy(self, coordinator):
        # risk priority=3, strategy priority=2
        recs = {
            "risk_manager": {"action": "sell", "confidence": 1.0},
            "strategy_agent": {"action": "buy", "confidence": 1.0},
        }
        result = await coordinator._resolve_conflicts(recs)
        assert result["decision"] == "sell"

    async def test_unanimous_buy_high_confidence(self, coordinator):
        recs = {
            "risk_manager": {"action": "buy", "confidence": 0.9},
            "strategy_agent": {"action": "buy", "confidence": 0.8},
            "market_analyst": {"action": "buy", "confidence": 0.7},
        }
        result = await coordinator._resolve_conflicts(recs)
        assert result["decision"] == "buy"
        assert result["confidence"] > 0.5

    async def test_empty_recommendations_returns_hold(self, coordinator):
        result = await coordinator._resolve_conflicts({})
        assert result["decision"] == "hold"
        assert result["confidence"] == 0.0

    async def test_unregistered_agent_no_crash(self, coordinator):
        # Agent not in coordinator.agents — should use default priority 1
        recs = {
            "unknown_agent_xyz": {"action": "buy", "confidence": 0.9},
        }
        result = await coordinator._resolve_conflicts(recs)
        assert result["decision"] in ("buy", "sell", "hold")

    async def test_priority_weighting_risk_over_analyst(self, coordinator):
        # risk weight = 3 * 0.8 = 2.4; analyst weight = 1 * 1.0 = 1.0
        recs = {
            "risk_manager": {"action": "sell", "confidence": 0.8},
            "market_analyst": {"action": "buy", "confidence": 1.0},
        }
        result = await coordinator._resolve_conflicts(recs)
        assert result["decision"] == "sell"

    async def test_all_hold_returns_hold(self, coordinator):
        recs = {
            "risk_manager": {"action": "hold", "confidence": 0.6},
            "strategy_agent": {"action": "hold", "confidence": 0.7},
        }
        result = await coordinator._resolve_conflicts(recs)
        assert result["decision"] == "hold"
