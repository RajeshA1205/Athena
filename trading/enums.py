"""Canonical trading enum definitions shared across all trading modules."""
from enum import Enum


class OrderType(Enum):
    """Supported order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TWAP = "twap"
    VWAP = "vwap"


class OrderSide(Enum):
    """Direction of an order."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Lifecycle states of an order."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    # Backwards-compatibility alias: PARTIAL -> PARTIALLY_FILLED (same value)
    PARTIAL = "partially_filled"
