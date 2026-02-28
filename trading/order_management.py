"""
Order Management Module
=======================
Provides order lifecycle management for ATHENA paper trading, including order
submission, fill simulation with configurable slippage, cancellation, and
status querying. All operations are async. Live broker integration is deferred;
this module supports paper trading mode only.
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from trading.enums import OrderType, OrderSide, OrderStatus


def _stable_hash(s: str) -> int:
    """Return a stable (PYTHONHASHSEED-independent) integer hash of a string."""
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """
    Represents a single trading order and its current state.

    Attributes:
        order_id: Unique identifier assigned by OrderManager.
        symbol: Ticker symbol (e.g. "AAPL").
        side: BUY or SELL direction.
        order_type: Market, limit, stop, or stop-limit.
        quantity: Requested number of units.
        limit_price: Required for LIMIT and STOP_LIMIT orders.
        stop_price: Trigger price for STOP and STOP_LIMIT orders.
        status: Current lifecycle state.
        filled_quantity: Units filled so far.
        avg_fill_price: Volume-weighted average fill price.
        created_at: UTC ISO-8601 timestamp of creation.
        updated_at: UTC ISO-8601 timestamp of last state change.
        fills: List of fill records accumulated during execution.
        metadata: Arbitrary caller-supplied key/value pairs.
    """

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    fills: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the order to a plain dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Order":
        """Restore an Order from a plain dictionary."""
        return Order(
            order_id=data["order_id"],
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            order_type=OrderType(data["order_type"]),
            quantity=data["quantity"],
            limit_price=data.get("limit_price"),
            stop_price=data.get("stop_price"),
            status=OrderStatus(data["status"]),
            filled_quantity=data.get("filled_quantity", 0.0),
            avg_fill_price=data.get("avg_fill_price", 0.0),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Fill:
    """
    Represents a single execution fill against an order.

    Attributes:
        fill_id: Unique identifier for this fill.
        order_id: Identifier of the parent order.
        symbol: Ticker symbol traded.
        side: BUY or SELL direction.
        quantity: Units filled in this execution.
        price: Price at which the fill was executed.
        timestamp: UTC ISO-8601 timestamp of the fill.
        metadata: Arbitrary key/value pairs (e.g. venue, commission).
    """

    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the fill to a plain dictionary."""
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class OrderManager:
    """
    Manages the full lifecycle of paper-trading orders.

    In paper trading mode (default) orders are immediately simulated:
    a configurable fill delay is applied, then a synthetic fill is generated
    using a mock price derived from the symbol hash plus slippage in bps.

    Config keys:
        slippage_bps (int): Slippage in basis points (default 5).
        fill_delay_seconds (float): Simulated fill latency (default 0.1).
        enable_paper_trading (bool): Auto-fill via simulation (default True).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the OrderManager.

        Args:
            config: Optional configuration dict.
        """
        self.config = config or {}
        self.slippage_bps: int = int(self.config.get("slippage_bps", 5))
        self.fill_delay_seconds: float = float(self.config.get("fill_delay_seconds", 0.1))
        self.enable_paper_trading: bool = bool(self.config.get("enable_paper_trading", True))
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        self._next_order_num: int = 0
        logger.info(
            "OrderManager initialized — paper_trading=%s slippage_bps=%d fill_delay=%.2fs",
            self.enable_paper_trading, self.slippage_bps, self.fill_delay_seconds,
        )

    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Order:
        """
        Submit a new order for execution.

        Generates a unique order ID, stores the order, and in paper trading
        mode immediately schedules fill simulation.

        Args:
            symbol: Ticker symbol to trade.
            side: BUY or SELL.
            order_type: Order type.
            quantity: Number of units to trade.
            limit_price: Limit price for LIMIT/STOP_LIMIT orders.
            stop_price: Stop trigger for STOP/STOP_LIMIT orders.
            metadata: Arbitrary caller-supplied key/value pairs.

        Returns:
            The created Order object.
        """
        order_id = f"ORD-{self._next_order_num:06d}"
        self._next_order_num += 1
        now = datetime.now(timezone.utc).isoformat()
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            status=OrderStatus.PENDING,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )
        self.orders[order_id] = order
        logger.info(
            "Order submitted: %s %s %s %.4f @ %s",
            order_id, side.value, symbol, quantity, limit_price or "MARKET",
        )
        if self.enable_paper_trading:
            task = asyncio.create_task(self._simulate_fill(order))
            task.add_done_callback(
                lambda t: (
                    logger.error("_simulate_fill task failed: %s", t.exception())
                    if not t.cancelled() and t.exception() is not None
                    else None
                )
            )
        return order

    async def _simulate_fill(self, order: Order) -> Optional[Fill]:
        """
        Simulate an order fill for paper trading.

        Waits fill_delay_seconds, computes a mock price from the symbol hash,
        applies slippage, creates a Fill, and updates the Order to FILLED.

        Args:
            order: The Order to fill.

        Returns:
            The Fill created, or None on unexpected error.
        """
        try:
            await asyncio.sleep(self.fill_delay_seconds)

            if order.order_type == OrderType.MARKET:
                base_price = 50.0 + _stable_hash(order.symbol) % 451
            else:
                base_price = order.limit_price if order.limit_price is not None \
                    else 50.0 + _stable_hash(order.symbol) % 451

            if order.side == OrderSide.BUY:
                fill_price = base_price * (1 + self.slippage_bps / 10_000)
            else:
                fill_price = base_price * (1 - self.slippage_bps / 10_000)

            now = datetime.now(timezone.utc).isoformat()
            fill = Fill(
                fill_id=f"FILL-{order.order_id}-001",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=fill_price,
                timestamp=now,
            )
            self.fills.append(fill)

            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.avg_fill_price = fill_price
            order.updated_at = now
            logger.debug("Fill: %s %.4f x %s @ %.4f", fill.fill_id, order.quantity, order.symbol, fill_price)
            return fill
        except Exception as e:
            logger.error("_simulate_fill failed for %r: %s", order.order_id, e)
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending or submitted order.

        Args:
            order_id: Identifier of the order to cancel.

        Returns:
            True if successfully cancelled, False otherwise.
        """
        order = self.orders.get(order_id)
        if order is None:
            logger.warning("cancel_order: order %r not found", order_id)
            return False
        if order.status in {OrderStatus.PENDING, OrderStatus.SUBMITTED}:
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now(timezone.utc).isoformat()
            logger.info("Order cancelled: %s", order_id)
            return True
        logger.warning("cancel_order: cannot cancel order %r in status %r", order_id, order.status.value)
        return False

    async def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """
        Return the current status of an order.

        Args:
            order_id: Identifier of the order.

        Returns:
            OrderStatus enum value, or None if not found.
        """
        order = self.orders.get(order_id)
        return order.status if order else None

    async def get_order(self, order_id: str) -> Optional[Order]:
        """Return a full Order object by ID, or None if not found."""
        return self.orders.get(order_id)

    async def get_fills(
        self, symbol: Optional[str] = None, order_id: Optional[str] = None
    ) -> List[Fill]:
        """
        Return fills, optionally filtered by symbol and/or order ID.

        Args:
            symbol: If given, only fills for this symbol.
            order_id: If given, only fills for this order.

        Returns:
            List of matching Fill objects.
        """
        results = list(self.fills)
        if symbol is not None:
            results = [f for f in results if f.symbol == symbol]
        if order_id is not None:
            results = [f for f in results if f.order_id == order_id]
        return results

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Return all unresolved orders (PENDING, SUBMITTED, PARTIALLY_FILLED).

        Args:
            symbol: If given, only open orders for this symbol.

        Returns:
            List of open Order objects.
        """
        open_statuses = {OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED}
        results = [o for o in self.orders.values() if o.status in open_statuses]
        if symbol is not None:
            results = [o for o in results if o.symbol == symbol]
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Return a summary of OrderManager activity."""
        return {
            "total_orders": len(self.orders),
            "filled_orders": sum(1 for o in self.orders.values() if o.status == OrderStatus.FILLED),
            "cancelled_orders": sum(1 for o in self.orders.values() if o.status == OrderStatus.CANCELLED),
            "total_fills": len(self.fills),
            "enable_paper_trading": self.enable_paper_trading,
        }
