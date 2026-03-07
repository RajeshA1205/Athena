"""Tests for StrategyAgent financial math: position sizing, momentum, mean reversion."""
import sys
import types

import pytest

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.nn = types.SimpleNamespace(Module=object, Linear=object)
    torch_stub.tensor = lambda *a, **kw: None
    torch_stub.no_grad = lambda: __import__("contextlib").nullcontext()
    sys.modules["torch"] = torch_stub


from agents.strategy_agent import StrategyAgent  # noqa: E402


@pytest.fixture
def agent():
    return StrategyAgent()


# ---------------------------------------------------------------------------
# _calculate_position_size
# ---------------------------------------------------------------------------
class TestCalculatePositionSize:
    async def test_zero_signal_strength(self, agent):
        size = await agent._calculate_position_size(signal_strength=0.0, volatility=0.1)
        assert size == 0.0

    async def test_zero_volatility(self, agent):
        # Should not raise ZeroDivisionError
        size = await agent._calculate_position_size(signal_strength=0.5, volatility=0.0)
        assert size == 0.0

    async def test_negative_signal_strength(self, agent):
        size = await agent._calculate_position_size(signal_strength=-0.5, volatility=0.1)
        assert size == 0.0

    async def test_cap_at_0_2(self, agent):
        # Very low volatility -> large raw size, but capped at 0.2
        size = await agent._calculate_position_size(
            signal_strength=1.0, risk_budget=0.02, volatility=0.0001
        )
        assert size == pytest.approx(0.2)

    async def test_normal_case_formula(self, agent):
        # base_size = 0.02 / (0.1 + 1e-10) ≈ 0.2; * 0.5 = 0.1
        size = await agent._calculate_position_size(
            signal_strength=0.5, risk_budget=0.02, volatility=0.1
        )
        expected = min(0.02 / 0.1 * 0.5, 0.2)
        assert size == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# _generate_momentum_signals
# ---------------------------------------------------------------------------
class TestMomentumSignals:
    async def test_too_few_prices(self, agent):
        signals = await agent._generate_momentum_signals([100.0, 101.0], lookback=20)
        assert signals == []

    async def test_minimum_length_boundary(self, agent):
        # Exactly lookback+1 prices required; lookback+1 prices should produce result
        prices = [100.0] * 21  # lookback=20 => need 21
        signals = await agent._generate_momentum_signals(prices, lookback=20)
        # All same prices => momentum = 0, no signal
        assert signals == []

    async def test_positive_momentum_generates_buy(self, agent):
        # Rising prices: last price >> lookback price
        base = [100.0] * 20
        prices = base + [103.0]  # 3% gain => momentum = 0.03 > 0.02
        signals = await agent._generate_momentum_signals(prices, lookback=20)
        assert len(signals) >= 1
        assert signals[0].signal_type == "buy"

    async def test_negative_momentum_generates_sell(self, agent):
        base = [100.0] * 20
        prices = base + [97.0]  # -3% => momentum = -0.03 < -0.02
        signals = await agent._generate_momentum_signals(prices, lookback=20)
        assert len(signals) >= 1
        assert signals[0].signal_type == "sell"

    async def test_signal_strength_clamped(self, agent):
        base = [100.0] * 20
        prices = base + [200.0]  # 100% gain => strength = min(10.0, 1.0) = 1.0
        signals = await agent._generate_momentum_signals(prices, lookback=20)
        assert signals[0].strength == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _generate_mean_reversion_signals
# ---------------------------------------------------------------------------
class TestMeanReversionSignals:
    async def test_too_few_prices(self, agent):
        signals = await agent._generate_mean_reversion_signals([100.0, 101.0], window=20)
        assert signals == []

    async def test_minimum_length_boundary(self, agent):
        prices = [100.0] * 21  # window=20 => need 21
        signals = await agent._generate_mean_reversion_signals(prices, window=20)
        # All same prices => z_score = 0, no signal
        assert signals == []

    async def test_below_lower_band_generates_buy(self, agent):
        # 20 prices at 100, then 1 price way below => z_score << -2
        prices = [100.0] * 20 + [80.0]
        signals = await agent._generate_mean_reversion_signals(prices, window=20, num_std=2.0)
        assert len(signals) >= 1
        assert signals[0].signal_type == "buy"

    async def test_above_upper_band_generates_sell(self, agent):
        prices = [100.0] * 20 + [120.0]
        signals = await agent._generate_mean_reversion_signals(prices, window=20, num_std=2.0)
        assert len(signals) >= 1
        assert signals[0].signal_type == "sell"

    async def test_extreme_z_score_strength_clamped(self, agent):
        # Extreme outlier => strength = min(|z|/num_std, 1.0) = 1.0
        prices = [100.0] * 20 + [0.001]  # near-zero price, extreme z-score
        signals = await agent._generate_mean_reversion_signals(prices, window=20, num_std=2.0)
        if signals:
            assert signals[0].strength == pytest.approx(1.0)
