"""
Portfolio Module
================
Tracks positions, computes P&L, monitors market exposure, and enforces
position limits for ATHENA paper trading. Updated via update_from_fill()
as orders are executed by the OrderManager.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .order_management import Fill, OrderSide

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """
    Represents a current holding in a single symbol.

    Attributes:
        symbol: Ticker symbol.
        quantity: Net quantity (positive = long, negative = short, 0 = flat).
        avg_cost: Volume-weighted average cost basis per unit.
        realized_pnl: Realized profit/loss accumulated from closed portions.
        last_price: Most recent market price used for unrealized P&L.
        opened_at: UTC ISO-8601 timestamp of first fill into this position.
        updated_at: UTC ISO-8601 timestamp of the last state change.
    """

    symbol: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0
    last_price: float = 0.0
    opened_at: str = ""
    updated_at: str = ""

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L based on last_price vs avg_cost."""
        if self.last_price == 0.0 or self.quantity == 0.0:
            return 0.0
        return (self.last_price - self.avg_cost) * self.quantity

    @property
    def market_value(self) -> float:
        """Current signed market value of the position."""
        return self.last_price * self.quantity

    def to_dict(self) -> Dict[str, Any]:
        """Serialize position to a plain dictionary."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_cost": round(self.avg_cost, 4),
            "realized_pnl": round(self.realized_pnl, 4),
            "unrealized_pnl": round(self.unrealized_pnl, 4),
            "last_price": round(self.last_price, 4),
            "market_value": round(self.market_value, 4),
            "opened_at": self.opened_at,
            "updated_at": self.updated_at,
        }


class Portfolio:
    """
    Manages the paper trading portfolio state.

    Tracks positions, computes real-time P&L, monitors exposure, and
    enforces position limits. Call update_from_fill() each time the
    OrderManager produces a Fill to keep the portfolio in sync.

    Config keys:
        max_position_size (float): Max absolute quantity per symbol (default 10_000).
        max_total_exposure (float): Max total absolute market value (default 1_000_000).
        initial_cash (float): Starting cash balance (default 100_000).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.max_position_size: float = float(
            self.config.get("max_position_size", 10_000)
        )
        self.max_total_exposure: float = float(
            self.config.get("max_total_exposure", 1_000_000)
        )
        self.cash: float = float(self.config.get("initial_cash", 100_000))
        self._positions: Dict[str, Position] = {}
        self._fill_history: List[Fill] = []
        logger.info(
            "Portfolio initialized — cash=%.2f max_pos=%.0f max_exposure=%.0f",
            self.cash,
            self.max_position_size,
            self.max_total_exposure,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_positions(
        self, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Return current non-zero positions.

        Args:
            symbol: If given, return only the position for this symbol.

        Returns:
            List of position dicts (empty list if no matching position).
        """
        if symbol is not None:
            pos = self._positions.get(symbol)
            return [pos.to_dict()] if pos and pos.quantity != 0.0 else []
        return [p.to_dict() for p in self._positions.values() if p.quantity != 0.0]

    async def calculate_pnl(self) -> Dict[str, Any]:
        """
        Calculate overall portfolio P&L.

        Returns:
            Dict with realized_pnl, unrealized_pnl, total_pnl, cash, and
            per-symbol breakdown.
        """
        realized = sum(p.realized_pnl for p in self._positions.values())
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        breakdown = {
            sym: {
                "realized_pnl": round(pos.realized_pnl, 4),
                "unrealized_pnl": round(pos.unrealized_pnl, 4),
            }
            for sym, pos in self._positions.items()
            if pos.quantity != 0.0 or pos.realized_pnl != 0.0
        }
        return {
            "realized_pnl": round(realized, 4),
            "unrealized_pnl": round(unrealized, 4),
            "total_pnl": round(realized + unrealized, 4),
            "cash": round(self.cash, 4),
            "breakdown": breakdown,
        }

    async def get_exposure(self) -> Dict[str, Any]:
        """
        Calculate total market exposure across all positions.

        Returns:
            Dict with total_exposure, long_exposure, short_exposure, cash,
            and per-symbol market values.
        """
        long_exp = sum(
            p.market_value for p in self._positions.values() if p.quantity > 0
        )
        short_exp = sum(
            abs(p.market_value) for p in self._positions.values() if p.quantity < 0
        )
        per_symbol = {
            sym: round(pos.market_value, 4)
            for sym, pos in self._positions.items()
            if pos.quantity != 0.0
        }
        return {
            "total_exposure": round(long_exp + short_exp, 4),
            "long_exposure": round(long_exp, 4),
            "short_exposure": round(short_exp, 4),
            "cash": round(self.cash, 4),
            "per_symbol": per_symbol,
        }

    async def check_limits(
        self, symbol: str, quantity: float, price: float
    ) -> Dict[str, Any]:
        """
        Check whether a proposed trade would violate portfolio limits.

        Args:
            symbol: Ticker symbol for the proposed trade.
            quantity: Signed quantity (positive = buy, negative = sell/short).
            price: Proposed execution price.

        Returns:
            Dict with 'approved' bool and 'reasons' list of violation strings.
        """
        reasons: List[str] = []

        # Position size check
        current_qty = self._positions.get(symbol, Position(symbol=symbol)).quantity
        new_qty = current_qty + quantity
        if abs(new_qty) > self.max_position_size:
            reasons.append(
                f"Position size {abs(new_qty):.0f} exceeds limit {self.max_position_size:.0f}"
            )

        # Total exposure check — compute net change in exposure
        exposure = await self.get_exposure()
        current_pos_value = abs(current_qty * price)
        new_pos_value = abs(new_qty * price)
        exposure_delta = new_pos_value - current_pos_value
        new_exposure = exposure["total_exposure"] + exposure_delta
        if new_exposure > self.max_total_exposure:
            reasons.append(
                f"Total exposure {new_exposure:.0f} would exceed limit "
                f"{self.max_total_exposure:.0f}"
            )

        # Cash check (only for buy orders)
        if quantity > 0:
            cost = quantity * price
            if cost > self.cash:
                reasons.append(
                    f"Insufficient cash: need {cost:.2f}, have {self.cash:.2f}"
                )

        return {"approved": len(reasons) == 0, "reasons": reasons}

    async def update_from_fill(self, fill: Fill) -> None:
        """
        Update portfolio state from an executed order fill.

        Adjusts position quantity, average cost, realized P&L, and cash balance
        using standard FIFO cost-basis accounting.

        Args:
            fill: Fill object produced by OrderManager._simulate_fill().
        """
        self._fill_history.append(fill)
        symbol = fill.symbol
        now = datetime.now(timezone.utc).isoformat()

        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol=symbol, opened_at=now)

        pos = self._positions[symbol]
        signed_qty = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
        trade_cost = fill.price * fill.quantity  # always positive

        if pos.quantity == 0.0:
            # Opening a new position
            pos.avg_cost = fill.price
            pos.quantity = signed_qty
            self.cash += -trade_cost if fill.side == OrderSide.BUY else trade_cost

        elif (pos.quantity > 0 and signed_qty > 0) or (
            pos.quantity < 0 and signed_qty < 0
        ):
            # Adding to an existing position — update VWAP cost basis
            total_cost = pos.avg_cost * abs(pos.quantity) + fill.price * fill.quantity
            pos.quantity += signed_qty
            pos.avg_cost = total_cost / abs(pos.quantity)
            self.cash += -trade_cost if fill.side == OrderSide.BUY else trade_cost

        else:
            # Reducing or closing/reversing an existing position
            close_qty = min(fill.quantity, abs(pos.quantity))

            if fill.side == OrderSide.SELL:
                # Selling from a long position: realize (sell_price - cost) * qty
                pos.realized_pnl += (fill.price - pos.avg_cost) * close_qty
                self.cash += trade_cost
            else:
                # Buying to cover a short: realize (short_price - buy_price) * qty
                pos.realized_pnl += (pos.avg_cost - fill.price) * close_qty
                self.cash -= trade_cost

            pos.quantity += signed_qty

            if pos.quantity == 0.0:
                pos.avg_cost = 0.0
            elif (pos.quantity > 0 and signed_qty < 0) or (
                pos.quantity < 0 and signed_qty > 0
            ):
                # Position reversed; cost basis resets to fill price for the remainder
                pos.avg_cost = fill.price

        pos.last_price = fill.price
        pos.updated_at = now

        logger.debug(
            "Portfolio updated: %s qty=%.4f avg_cost=%.4f realized_pnl=%.4f cash=%.2f",
            symbol,
            pos.quantity,
            pos.avg_cost,
            pos.realized_pnl,
            self.cash,
        )

    async def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update the last_price on positions for accurate unrealized P&L.

        Args:
            prices: Dict mapping symbol -> current market price.
        """
        for symbol, price in prices.items():
            if symbol in self._positions:
                self._positions[symbol].last_price = price

    def get_stats(self) -> Dict[str, Any]:
        """Return a summary of portfolio activity."""
        return {
            "total_positions": sum(
                1 for p in self._positions.values() if p.quantity != 0.0
            ),
            "total_fills": len(self._fill_history),
            "cash": round(self.cash, 4),
            "symbols_traded": list(self._positions.keys()),
        }
