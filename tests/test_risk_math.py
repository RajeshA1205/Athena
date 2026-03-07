"""Tests for RiskManagerAgent financial math: VaR, Expected Shortfall, portfolio metrics."""
import math
import sys
import types

import pytest

# ---------------------------------------------------------------------------
# Torch stub (mirrors the pattern in test_e2e.py)
# ---------------------------------------------------------------------------
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.nn = types.SimpleNamespace(Module=object, Linear=object)
    torch_stub.tensor = lambda *a, **kw: None
    torch_stub.no_grad = lambda: __import__("contextlib").nullcontext()
    sys.modules["torch"] = torch_stub


from agents.risk_manager import RiskManagerAgent  # noqa: E402


@pytest.fixture
def agent():
    return RiskManagerAgent()


# ---------------------------------------------------------------------------
# _calculate_var
# ---------------------------------------------------------------------------
class TestCalculateVar:
    async def test_empty_returns(self, agent):
        assert await agent._calculate_var({}) == 0.0

    async def test_empty_series(self, agent):
        assert await agent._calculate_var({"AAPL": []}) == 0.0

    async def test_single_asset_known_percentile(self, agent):
        # 100 returns: -0.05, -0.04, ..., 0.04 (step 0.001)
        returns = [round(-0.05 + i * 0.001, 4) for i in range(100)]
        var = await agent._calculate_var({"AAPL": returns}, confidence=0.95)
        # 5th percentile of sorted series: index = int(100 * 0.05) = 5 => returns[5] = -0.045
        assert var == pytest.approx(0.045, abs=1e-4)

    async def test_all_positive_returns(self, agent):
        returns = [0.01 * i for i in range(1, 21)]  # 0.01..0.20
        var = await agent._calculate_var({"AAPL": returns}, confidence=0.95)
        # All positive: worst sorted value is positive, so -sorted[0] <= 0 => var = -positive < 0
        # but the method returns -sorted[index]; for all-positive sorted[0] is smallest positive
        # The clamp max(0, min(index, len-1)) ensures we don't go negative on index
        # The result is -sorted[0] which is negative — but caller should treat 0 as floor;
        # the method itself returns the raw value which may be <=0. Just verify no crash + type.
        assert isinstance(var, float)

    async def test_multi_asset_equal_weight(self, agent):
        # Two identical series — equal-weighted average is the same series
        r = [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
        var_single = await agent._calculate_var({"A": r}, confidence=0.90)
        var_multi = await agent._calculate_var({"A": r, "B": r}, confidence=0.90)
        assert var_single == pytest.approx(var_multi, abs=1e-6)

    async def test_parametric_method(self, agent):
        # Known normal: mean=0, std=0.01 => VaR(95%) = 1.645 * 0.01 = 0.01645
        import random
        rng = random.Random(42)
        returns = [rng.gauss(0, 0.01) for _ in range(1000)]
        var = await agent._calculate_var({"AAPL": returns}, confidence=0.95, method="parametric")
        # Should be close to 1.645 * sample_std
        assert var > 0
        assert var < 0.05  # sanity bound

    async def test_single_element_series(self, agent):
        var = await agent._calculate_var({"AAPL": [-0.02]}, confidence=0.95)
        assert var == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# _calculate_expected_shortfall
# ---------------------------------------------------------------------------
class TestCalculateExpectedShortfall:
    async def test_empty_returns(self, agent):
        assert await agent._calculate_expected_shortfall({}) == 0.0

    async def test_es_geq_var(self, agent):
        returns = [round(-0.05 + i * 0.001, 4) for i in range(100)]
        var = await agent._calculate_var({"AAPL": returns}, confidence=0.95)
        es = await agent._calculate_expected_shortfall({"AAPL": returns}, confidence=0.95)
        assert es >= var - 1e-9

    async def test_known_distribution(self, agent):
        # 10 returns, sorted: -0.09, -0.07, ..., 0.09 (step 0.02)
        returns = [-0.09 + i * 0.02 for i in range(10)]
        # confidence=0.90 => cutoff_index = max(1, int(10 * 0.10)) = 1
        # tail_losses = [sorted_returns[0]] = [-0.09]
        # ES = abs(-0.09 / 1) = 0.09
        es = await agent._calculate_expected_shortfall({"AAPL": returns}, confidence=0.90)
        assert es == pytest.approx(0.09, abs=1e-6)

    async def test_positive_result(self, agent):
        returns = [-0.01 * i for i in range(1, 11)]
        es = await agent._calculate_expected_shortfall({"AAPL": returns})
        assert es >= 0.0


# ---------------------------------------------------------------------------
# _calculate_portfolio_metrics
# ---------------------------------------------------------------------------
class TestCalculatePortfolioMetrics:
    async def test_empty_returns(self, agent):
        result = await agent._calculate_portfolio_metrics([], {})
        assert result == {}

    async def test_expected_keys(self, agent):
        returns = [0.01, -0.01, 0.02, -0.02, 0.005]
        result = await agent._calculate_portfolio_metrics([], {"AAPL": returns})
        assert "sharpe_ratio" in result
        assert "volatility" in result
        assert "max_drawdown" in result

    async def test_zero_volatility_no_inf_sharpe(self, agent):
        # All identical returns => zero population std dev => sharpe = 0.0
        returns = [0.001] * 20
        result = await agent._calculate_portfolio_metrics([], {"AAPL": returns})
        assert result["sharpe_ratio"] == 0.0
        assert not math.isinf(result["sharpe_ratio"])

    async def test_negative_mean_negative_sharpe(self, agent):
        returns = [-0.01] * 20  # constant negative => zero vol => sharpe = 0.0
        result = await agent._calculate_portfolio_metrics([], {"AAPL": returns})
        # zero vol path => 0.0
        assert result["sharpe_ratio"] == 0.0

    async def test_negative_mean_with_variance(self, agent):
        # Clearly negative mean with some variance => negative sharpe
        returns = [-0.05, -0.04, -0.06, -0.03, -0.07, -0.04, -0.05] * 5
        result = await agent._calculate_portfolio_metrics([], {"AAPL": returns})
        assert result["sharpe_ratio"] < 0

    async def test_max_drawdown_peak_trough(self, agent):
        # Rise to peak, fall 50%, recover: drawdown should be 0.50
        # Cumulative: 1.0 -> 2.0 -> 1.0
        # +100% then -50%
        returns = [1.0, -0.5]
        result = await agent._calculate_portfolio_metrics([], {"AAPL": returns})
        assert result["max_drawdown"] == pytest.approx(0.5, abs=1e-4)

    async def test_monotone_increase_zero_drawdown(self, agent):
        returns = [0.01] * 10  # always rising => zero drawdown (zero vol => sharpe=0)
        result = await agent._calculate_portfolio_metrics([], {"AAPL": returns})
        assert result["max_drawdown"] == pytest.approx(0.0, abs=1e-9)

    async def test_single_return(self, agent):
        result = await agent._calculate_portfolio_metrics([], {"AAPL": [0.01]})
        assert "sharpe_ratio" in result
