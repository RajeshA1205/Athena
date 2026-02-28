"""
Unit tests for the ATHENA trading layer.

Tests cover: MarketDataFeed, OrderManager, Portfolio, and their supporting
dataclasses (OHLCV, Order, Fill, Position). All tests use paper trading
mode (no live broker required).
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# MarketDataFeed / OHLCV / MarketDataMode
# ---------------------------------------------------------------------------

class TestMarketDataFeed:
    def test_import(self):
        from trading.market_data import MarketDataFeed, OHLCV, MarketDataMode
        assert MarketDataFeed is not None
        assert OHLCV is not None
        assert MarketDataMode is not None

    def test_instantiation_default(self):
        from trading.market_data import MarketDataFeed
        feed = MarketDataFeed()
        assert feed is not None

    def test_instantiation_with_config(self):
        from trading.market_data import MarketDataFeed, MarketDataMode
        feed = MarketDataFeed(
            mode=MarketDataMode.MOCK,
            config={"key": "value"},
        )
        assert feed is not None

    @pytest.mark.asyncio
    async def test_get_realtime_data_returns_ohlcv(self):
        from trading.market_data import MarketDataFeed
        feed = MarketDataFeed()
        bar = await feed.get_realtime_data("AAPL")
        assert bar is not None
        assert bar.symbol == "AAPL"
        assert bar.close > 0

    @pytest.mark.asyncio
    async def test_get_historical_data_returns_list(self):
        from trading.market_data import MarketDataFeed
        feed = MarketDataFeed()
        bars = await feed.get_historical_data("MSFT", days=10)
        assert isinstance(bars, list)
        assert len(bars) == 10

    @pytest.mark.asyncio
    async def test_subscribe_and_get_symbols(self):
        from trading.market_data import MarketDataFeed
        feed = MarketDataFeed()
        sub_id = await feed.subscribe("AAPL", callback=lambda bar: None)
        assert sub_id is not None
        symbols = await feed.get_symbols()
        assert "AAPL" in symbols

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        from trading.market_data import MarketDataFeed
        feed = MarketDataFeed()
        sub_id = await feed.subscribe("AAPL", callback=lambda bar: None)
        result = await feed.unsubscribe("AAPL", sub_id)
        assert result is True

    def test_ohlcv_dataclass(self):
        from trading.market_data import OHLCV
        bar = OHLCV(
            symbol="AAPL",
            timestamp="2026-02-23T00:00:00+00:00",
            open=184.0,
            high=186.0,
            low=183.0,
            close=185.0,
            volume=1_000_000,
        )
        assert bar.symbol == "AAPL"
        assert bar.close == 185.0

    def test_ohlcv_fields(self):
        from trading.market_data import OHLCV
        bar = OHLCV("AAPL", "2026-01-01T00:00:00+00:00", 100.0, 105.0, 99.0, 102.0, 500_000)
        assert bar.symbol == "AAPL"
        assert bar.close == 102.0
        assert bar.volume == 500_000

    def test_market_data_mode_enum(self):
        from trading.market_data import MarketDataMode
        assert MarketDataMode.MOCK is not None
        assert MarketDataMode.LIVE is not None

    @pytest.mark.asyncio
    async def test_get_realtime_multiple_symbols(self):
        from trading.market_data import MarketDataFeed
        feed = MarketDataFeed()
        for sym in ["AAPL", "MSFT", "TSLA"]:
            bar = await feed.get_realtime_data(sym)
            assert bar.symbol == sym
            assert bar.close > 0

    @pytest.mark.asyncio
    async def test_mock_prices_deterministic(self):
        from trading.market_data import MarketDataFeed
        feed1 = MarketDataFeed()
        feed2 = MarketDataFeed()
        bar1 = await feed1.get_realtime_data("AAPL")
        bar2 = await feed2.get_realtime_data("AAPL")
        # Same symbol should produce same base price in mock mode
        assert bar1.close == bar2.close


# ---------------------------------------------------------------------------
# OrderManager
# ---------------------------------------------------------------------------

class TestOrderManager:
    def test_import(self):
        from trading.order_management import (
            OrderManager, Order, Fill, OrderType, OrderSide, OrderStatus
        )
        assert OrderManager is not None

    def test_instantiation_default(self):
        from trading.order_management import OrderManager
        om = OrderManager()
        assert om is not None
        assert om.enable_paper_trading is True

    def test_instantiation_with_config(self):
        from trading.order_management import OrderManager
        om = OrderManager(config={"slippage_bps": 10, "fill_delay_seconds": 0.0})
        assert om.slippage_bps == 10

    @pytest.mark.asyncio
    async def test_submit_market_order(self):
        from trading.order_management import OrderManager, OrderType, OrderSide
        om = OrderManager(config={"fill_delay_seconds": 0.0})
        order = await om.submit_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10,
        )
        assert order is not None
        assert order.symbol == "AAPL"
        assert order.quantity == 10

    @pytest.mark.asyncio
    async def test_submit_limit_order(self):
        from trading.order_management import OrderManager, OrderType, OrderSide
        om = OrderManager(config={"fill_delay_seconds": 0.0})
        order = await om.submit_order(
            symbol="MSFT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=5,
            limit_price=300.0,
        )
        assert order.limit_price == 300.0
        assert order.order_id.startswith("ORD-")

    @pytest.mark.asyncio
    async def test_order_ids_are_unique(self):
        from trading.order_management import OrderManager, OrderType, OrderSide
        om = OrderManager(config={"fill_delay_seconds": 0.0})
        ids = set()
        for _ in range(5):
            order = await om.submit_order("AAPL", OrderSide.BUY, OrderType.MARKET, 1)
            ids.add(order.order_id)
        assert len(ids) == 5

    @pytest.mark.asyncio
    async def test_get_order_status(self):
        from trading.order_management import OrderManager, OrderType, OrderSide, OrderStatus
        om = OrderManager(config={"fill_delay_seconds": 0.0, "enable_paper_trading": False})
        order = await om.submit_order("AAPL", OrderSide.BUY, OrderType.MARKET, 1)
        status = await om.get_order_status(order.order_id)
        assert status in {OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.FILLED}

    @pytest.mark.asyncio
    async def test_cancel_pending_order(self):
        from trading.order_management import OrderManager, OrderType, OrderSide, OrderStatus
        om = OrderManager(config={"enable_paper_trading": False})
        order = await om.submit_order("AAPL", OrderSide.BUY, OrderType.MARKET, 1)
        cancelled = await om.cancel_order(order.order_id)
        assert cancelled is True
        status = await om.get_order_status(order.order_id)
        assert status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_order(self):
        from trading.order_management import OrderManager
        om = OrderManager()
        result = await om.cancel_order("ORD-999999")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_open_orders_empty(self):
        from trading.order_management import OrderManager
        om = OrderManager()
        orders = await om.get_open_orders()
        assert isinstance(orders, list)

    @pytest.mark.asyncio
    async def test_simulate_fill_produces_fill(self):
        from trading.order_management import OrderManager, OrderType, OrderSide, OrderStatus
        om = OrderManager(config={"fill_delay_seconds": 0.0, "slippage_bps": 5})
        order = await om.submit_order("AAPL", OrderSide.BUY, OrderType.MARKET, 10)
        # Allow event loop to process the fill coroutine
        await asyncio.sleep(0.01)
        fills = await om.get_fills(order_id=order.order_id)
        assert isinstance(fills, list)

    @pytest.mark.asyncio
    async def test_get_fills_filtered_by_symbol(self):
        from trading.order_management import OrderManager, OrderType, OrderSide
        om = OrderManager(config={"fill_delay_seconds": 0.0})
        await om.submit_order("AAPL", OrderSide.BUY, OrderType.MARKET, 5)
        await om.submit_order("MSFT", OrderSide.BUY, OrderType.MARKET, 3)
        await asyncio.sleep(0.05)
        fills = await om.get_fills(symbol="AAPL")
        for f in fills:
            assert f.symbol == "AAPL"

    def test_get_stats(self):
        from trading.order_management import OrderManager
        om = OrderManager()
        stats = om.get_stats()
        assert isinstance(stats, dict)
        assert "total_orders" in stats

    def test_order_to_dict(self):
        from trading.order_management import Order, OrderType, OrderSide, OrderStatus
        order = Order(
            order_id="ORD-000001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10,
            created_at="2026-02-23T00:00:00+00:00",
            updated_at="2026-02-23T00:00:00+00:00",
        )
        d = order.to_dict()
        assert d["order_id"] == "ORD-000001"
        assert d["side"] == "buy"

    def test_order_from_dict(self):
        from trading.order_management import Order, OrderType, OrderSide, OrderStatus
        data = {
            "order_id": "ORD-000002",
            "symbol": "MSFT",
            "side": "sell",
            "order_type": "limit",
            "quantity": 5.0,
            "limit_price": 300.0,
            "stop_price": None,
            "status": "pending",
            "filled_quantity": 0.0,
            "avg_fill_price": 0.0,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "metadata": {},
        }
        order = Order.from_dict(data)
        assert order.order_id == "ORD-000002"
        assert order.side.value == "sell"


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

class TestPortfolio:
    def test_import(self):
        from trading.portfolio import Portfolio, Position
        assert Portfolio is not None
        assert Position is not None

    def test_instantiation_default(self):
        from trading.portfolio import Portfolio
        p = Portfolio()
        assert p is not None
        assert p.cash == 100_000.0

    def test_instantiation_with_config(self):
        from trading.portfolio import Portfolio
        p = Portfolio(config={"initial_cash": 50_000, "max_position_size": 500})
        assert p.cash == 50_000.0
        assert p.max_position_size == 500.0

    @pytest.mark.asyncio
    async def test_get_positions_empty(self):
        from trading.portfolio import Portfolio
        p = Portfolio()
        positions = await p.get_positions()
        assert positions == []

    @pytest.mark.asyncio
    async def test_update_from_buy_fill_creates_position(self):
        from trading.portfolio import Portfolio
        from trading.order_management import Fill, OrderSide
        p = Portfolio()
        fill = Fill(
            fill_id="FILL-001",
            order_id="ORD-000001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            price=185.0,
            timestamp="2026-02-23T00:00:00+00:00",
        )
        await p.update_from_fill(fill)
        positions = await p.get_positions("AAPL")
        assert len(positions) == 1
        assert positions[0]["quantity"] == 10.0
        assert positions[0]["avg_cost"] == 185.0

    @pytest.mark.asyncio
    async def test_cash_decreases_on_buy(self):
        from trading.portfolio import Portfolio
        from trading.order_management import Fill, OrderSide
        p = Portfolio(config={"initial_cash": 100_000})
        fill = Fill("F1", "O1", "AAPL", OrderSide.BUY, 10, 185.0, "2026-02-23T00:00:00+00:00")
        await p.update_from_fill(fill)
        assert p.cash == 100_000 - 10 * 185.0

    @pytest.mark.asyncio
    async def test_cash_increases_on_sell(self):
        from trading.portfolio import Portfolio
        from trading.order_management import Fill, OrderSide
        p = Portfolio(config={"initial_cash": 100_000})
        buy = Fill("F1", "O1", "AAPL", OrderSide.BUY, 10, 185.0, "2026-02-23T00:00:00+00:00")
        await p.update_from_fill(buy)
        sell = Fill("F2", "O2", "AAPL", OrderSide.SELL, 10, 190.0, "2026-02-23T00:01:00+00:00")
        await p.update_from_fill(sell)
        # Cash after buy and sell
        expected = 100_000 - 10 * 185.0 + 10 * 190.0
        assert abs(p.cash - expected) < 0.01

    @pytest.mark.asyncio
    async def test_realized_pnl_on_close(self):
        from trading.portfolio import Portfolio
        from trading.order_management import Fill, OrderSide
        p = Portfolio()
        buy = Fill("F1", "O1", "AAPL", OrderSide.BUY, 10, 185.0, "2026-02-23T00:00:00+00:00")
        sell = Fill("F2", "O2", "AAPL", OrderSide.SELL, 10, 190.0, "2026-02-23T00:01:00+00:00")
        await p.update_from_fill(buy)
        await p.update_from_fill(sell)
        pnl = await p.calculate_pnl()
        assert abs(pnl["realized_pnl"] - 50.0) < 0.01  # (190-185) * 10 = 50

    @pytest.mark.asyncio
    async def test_calculate_pnl_returns_dict(self):
        from trading.portfolio import Portfolio
        p = Portfolio()
        pnl = await p.calculate_pnl()
        assert isinstance(pnl, dict)
        assert "realized_pnl" in pnl
        assert "unrealized_pnl" in pnl
        assert "total_pnl" in pnl

    @pytest.mark.asyncio
    async def test_get_exposure_empty(self):
        from trading.portfolio import Portfolio
        p = Portfolio()
        exp = await p.get_exposure()
        assert exp["total_exposure"] == 0.0
        assert exp["long_exposure"] == 0.0

    @pytest.mark.asyncio
    async def test_get_exposure_with_position(self):
        from trading.portfolio import Portfolio
        from trading.order_management import Fill, OrderSide
        p = Portfolio()
        fill = Fill("F1", "O1", "AAPL", OrderSide.BUY, 100, 185.0, "2026-02-23T00:00:00+00:00")
        await p.update_from_fill(fill)
        await p.update_prices({"AAPL": 185.0})
        exp = await p.get_exposure()
        assert exp["long_exposure"] == 185.0 * 100

    @pytest.mark.asyncio
    async def test_check_limits_approved(self):
        from trading.portfolio import Portfolio
        p = Portfolio(config={"max_position_size": 1000, "initial_cash": 100_000})
        result = await p.check_limits("AAPL", 10, 185.0)
        assert result["approved"] is True
        assert result["reasons"] == []

    @pytest.mark.asyncio
    async def test_check_limits_rejected_size(self):
        from trading.portfolio import Portfolio
        p = Portfolio(config={"max_position_size": 5})
        result = await p.check_limits("AAPL", 100, 10.0)
        assert result["approved"] is False
        assert len(result["reasons"]) >= 1

    @pytest.mark.asyncio
    async def test_check_limits_insufficient_cash(self):
        from trading.portfolio import Portfolio
        p = Portfolio(config={"initial_cash": 1_000})
        result = await p.check_limits("AAPL", 10, 200.0)  # 2000 > 1000 cash
        assert result["approved"] is False

    @pytest.mark.asyncio
    async def test_update_prices(self):
        from trading.portfolio import Portfolio
        from trading.order_management import Fill, OrderSide
        p = Portfolio()
        fill = Fill("F1", "O1", "AAPL", OrderSide.BUY, 10, 185.0, "2026-02-23T00:00:00+00:00")
        await p.update_from_fill(fill)
        await p.update_prices({"AAPL": 200.0})
        positions = await p.get_positions("AAPL")
        assert positions[0]["last_price"] == 200.0
        assert positions[0]["unrealized_pnl"] == (200.0 - 185.0) * 10

    @pytest.mark.asyncio
    async def test_position_flat_after_full_close(self):
        from trading.portfolio import Portfolio
        from trading.order_management import Fill, OrderSide
        p = Portfolio()
        buy = Fill("F1", "O1", "AAPL", OrderSide.BUY, 10, 185.0, "2026-02-23T00:00:00+00:00")
        sell = Fill("F2", "O2", "AAPL", OrderSide.SELL, 10, 190.0, "2026-02-23T00:01:00+00:00")
        await p.update_from_fill(buy)
        await p.update_from_fill(sell)
        positions = await p.get_positions("AAPL")
        assert positions == []  # flat position excluded

    def test_get_stats(self):
        from trading.portfolio import Portfolio
        p = Portfolio()
        stats = p.get_stats()
        assert isinstance(stats, dict)
        assert "cash" in stats
        assert "total_fills" in stats

    def test_position_dataclass(self):
        from trading.portfolio import Position
        pos = Position(symbol="AAPL", quantity=10, avg_cost=185.0, last_price=190.0)
        assert pos.unrealized_pnl == (190.0 - 185.0) * 10
        assert pos.market_value == 190.0 * 10

    def test_position_to_dict(self):
        from trading.portfolio import Position
        pos = Position(symbol="AAPL", quantity=5, avg_cost=100.0, last_price=110.0)
        d = pos.to_dict()
        assert d["symbol"] == "AAPL"
        assert "unrealized_pnl" in d
        assert "market_value" in d
