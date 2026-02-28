"""
ATHENA Execution Agent
======================
Handles order execution, timing optimization, and slippage minimization.
"""

import time
import random
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import math

from core.base_agent import BaseAgent, AgentContext, AgentAction
from core.config import get_default_agent_configs
from trading.enums import OrderType, OrderSide, OrderStatus
from trading.order_management import Order

if TYPE_CHECKING:
    from memory.agemem import AgeMem
    from communication.router import MessageRouter


class ExecutionAgent(BaseAgent):
    """
    Execution Agent for ATHENA multi-agent trading system.

    Responsibilities:
    - Order execution and management
    - Timing optimization
    - Slippage minimization
    - Transaction cost analysis
    """

    def __init__(
        self,
        name: str = "execution_agent",
        role: str = "execution",
        system_prompt: Optional[str] = None,
        model: Optional[Any] = None,
        memory: Optional["AgeMem"] = None,
        communication: Optional[Any] = None,
        router: Optional["MessageRouter"] = None,
        tools: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Execution Agent.

        Args:
            name: Agent name
            role: Agent role
            system_prompt: System prompt
            model: Language model instance
            memory: AgeMem memory instance
            communication: LatentSpace communication instance
            router: MessageRouter for LatentMAS routing (optional)
            tools: List of available tool names
            config: Additional configuration
        """
        if system_prompt is None:
            default_configs = get_default_agent_configs()
            system_prompt = default_configs["execution_agent"].system_prompt

        if tools is None:
            tools = ["place_order", "cancel_order", "get_execution_stats"]

        super().__init__(
            name=name,
            role=role,
            system_prompt=system_prompt,
            model=model,
            memory=memory,
            communication=communication,
            tools=tools,
            config=config,
        )

        self.memory: Optional["AgeMem"] = memory
        self.router: Optional["MessageRouter"] = router
        self.active_orders: Dict[str, Order] = {}
        self.completed_orders: Dict[str, Order] = {}
        self.order_counter = 0

        self.default_slippage_bps = (config or {}).get("default_slippage_bps", 5.0)
        self.fee_bps = (config or {}).get("fee_bps", 1.0)
        seed = int((config or {}).get("simulation_seed", 0)) or None
        self._rng = random.Random(seed)

    async def think(self, context: AgentContext) -> Dict[str, Any]:
        """
        Analyze trade request and formulate execution plan.

        Args:
            context: Current context including task, history, memory, messages

        Returns:
            Dictionary containing execution plan
        """
        self.logger.info("Planning execution for: %s", context.task)

        trade_request = self._parse_trade_request(context.task, context.metadata)
        market_conditions = context.metadata.get("market_conditions", {})

        # Retrieve relevant memory context
        memory_context = []
        if self.memory:
            try:
                market_data = context.metadata.get("market_data", {})
                memory_context = await self.memory.retrieve(
                    query=f"order execution {market_data.get('symbol', '') if market_data else ''}",
                    top_k=5
                )
            except Exception as e:
                self.logger.warning("Memory retrieve failed: %s", e)

        # Receive any LatentMAS messages
        latent_messages = []
        if self.router:
            try:
                latent_messages = await self.router.receive(
                    receiver_id=self.name,
                    decode_mode="structured",
                )
            except Exception as e:
                self.logger.warning("LatentMAS receive failed: %s", e)

        current_price = market_conditions.get("current_price", 100.0)
        avg_volume = market_conditions.get("avg_volume", 1000000.0)
        volatility = market_conditions.get("volatility", 0.02)

        size_pct_volume = trade_request["quantity"] / avg_volume if avg_volume > 0 else 0.0

        market_impact = await self._estimate_market_impact(
            order_size=trade_request["quantity"],
            avg_volume=avg_volume,
            volatility=volatility,
        )

        urgency = trade_request.get("urgency", "normal")
        optimal_order_type = await self._select_order_type(
            urgency=urgency,
            size_pct_volume=size_pct_volume,
            volatility=volatility,
        )

        execution_plan: Dict[str, Any] = {
            "symbol": trade_request["symbol"],
            "side": trade_request["side"],
            "quantity": trade_request["quantity"],
            "order_type": optimal_order_type,
            "market_impact_bps": market_impact * 10000,
            "estimated_slippage_bps": self.default_slippage_bps + market_impact * 10000,
            "current_price": current_price,
            "urgency": urgency,
        }

        if optimal_order_type == OrderType.TWAP:
            duration = trade_request.get("duration_minutes", 60)
            num_slices = trade_request.get("num_slices", 10)
            schedule = await self._calculate_twap_schedule(
                trade_request["quantity"], duration, num_slices
            )
            execution_plan["schedule"] = schedule
            execution_plan["duration_minutes"] = duration

        elif optimal_order_type == OrderType.VWAP:
            volume_profile = market_conditions.get("volume_profile", [1.0] * 10)
            schedule = await self._calculate_vwap_schedule(
                trade_request["quantity"], volume_profile
            )
            execution_plan["schedule"] = schedule

        elif optimal_order_type == OrderType.LIMIT:
            if trade_request["side"] == OrderSide.BUY:
                execution_plan["limit_price"] = current_price * 0.999
            else:
                execution_plan["limit_price"] = current_price * 1.001

        elif optimal_order_type == OrderType.STOP:
            execution_plan["stop_price"] = current_price * 0.98

        self.logger.info(
            f"Execution plan: {optimal_order_type.value} for "
            f"{trade_request['quantity']} {trade_request['symbol']}, "
            f"estimated impact: {market_impact * 10000:.2f}bps"
        )

        return {
            "execution_plan": execution_plan,
            "memory_context": memory_context,
            "latent_messages": latent_messages,
            "done": False,
        }

    async def act(self, thought: Dict[str, Any]) -> AgentAction:
        """
        Execute the planned order.

        Args:
            thought: Output from think()

        Returns:
            AgentAction describing execution result
        """
        start_time = time.perf_counter()
        execution_plan = thought["execution_plan"]

        try:
            order = await self._create_order(execution_plan)
            fills = await self._simulate_execution(order)

            order.fills = fills
            order.filled_quantity = sum(f["quantity"] for f in fills)
            if order.filled_quantity >= order.quantity:
                order.status = OrderStatus.FILLED
            elif order.filled_quantity > 0:
                order.status = OrderStatus.PARTIAL

            if order.filled_quantity > 0:
                order.avg_fill_price = sum(
                    f["quantity"] * f["price"] for f in fills
                ) / order.filled_quantity

            if order.status == OrderStatus.FILLED:
                self.completed_orders[order.order_id] = order
                self.active_orders.pop(order.order_id, None)

            metrics = await self._calculate_execution_metrics(order, fills)
            duration = time.perf_counter() - start_time

            self.logger.info(
                f"Order {order.order_id}: {order.status.value}, "
                f"filled {order.filled_quantity}/{order.quantity} @ {order.avg_fill_price:.2f}, "
                f"slippage: {metrics['slippage_bps']:.2f}bps"
            )

            result = {
                "order": self._order_to_dict(order),
                "metrics": metrics,
            }

            # Send output via LatentMAS to coordinator
            if self.router:
                try:
                    from communication.router import MessagePriority
                    await self.router.send(
                        sender_id=self.name,
                        receiver_id="coordinator",
                        message=result,
                        priority=MessagePriority.MEDIUM,
                    )
                except Exception as e:
                    self.logger.warning("LatentMAS send failed: %s", e)

            # Store decision to memory
            if self.memory:
                try:
                    await self.memory.add(
                        content={"execution": result, "thought": thought},
                        metadata={
                            "agent": self.name,
                            "role": self.role,
                            "success": True,
                        }
                    )
                except Exception as e:
                    self.logger.warning("Memory store failed: %s", e)

            return AgentAction(
                action_type="execute_order",
                parameters={
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "quantity": order.quantity,
                    "order_type": order.order_type.value,
                },
                result=result,
                success=True,
                duration=duration,
            )

        except Exception as e:
            duration = time.perf_counter() - start_time
            self.logger.error("Execution failed: %s", e)

            # Store failure to memory
            if self.memory:
                try:
                    await self.memory.add(
                        content={"execution": None, "thought": thought},
                        metadata={
                            "agent": self.name,
                            "role": self.role,
                            "success": False,
                        }
                    )
                except Exception as mem_e:
                    self.logger.warning("Memory store failed: %s", mem_e)

            return AgentAction(
                action_type="execute_order",
                parameters=execution_plan,
                result=None,
                success=False,
                error=str(e),
                duration=duration,
            )

    async def _create_order(self, execution_plan: Dict[str, Any]) -> Order:
        """Create an order from execution plan."""
        self.order_counter += 1
        order_id = f"ORD-{self.order_counter:06d}"

        order = Order(
            order_id=order_id,
            symbol=execution_plan["symbol"],
            side=execution_plan["side"],
            quantity=execution_plan["quantity"],
            order_type=execution_plan["order_type"],
            limit_price=execution_plan.get("limit_price"),
            stop_price=execution_plan.get("stop_price"),
            metadata={
                "urgency": execution_plan.get("urgency"),
                "estimated_slippage_bps": execution_plan.get("estimated_slippage_bps"),
                "current_price": execution_plan.get("current_price"),
                "schedule": execution_plan.get("schedule"),
            },
        )

        self.active_orders[order_id] = order
        return order

    async def _estimate_market_impact(
        self, order_size: float, avg_volume: float, volatility: float
    ) -> float:
        """
        Estimate market impact using square-root model.

        impact = volatility * sqrt(order_size / avg_volume)

        Args:
            order_size: Size of the order
            avg_volume: Average daily volume
            volatility: Asset volatility

        Returns:
            Estimated market impact as fraction
        """
        if avg_volume <= 0:
            return 0.01
        volume_fraction = order_size / avg_volume
        impact = volatility * math.sqrt(volume_fraction)
        return min(impact, 0.05)

    async def _select_order_type(
        self, urgency: str, size_pct_volume: float, volatility: float
    ) -> OrderType:
        """
        Select optimal order type based on conditions.

        Args:
            urgency: Urgency level (high, normal, low)
            size_pct_volume: Order size as percentage of volume
            volatility: Market volatility

        Returns:
            Optimal order type
        """
        if urgency == "high" or size_pct_volume < 0.01:
            return OrderType.MARKET
        if size_pct_volume > 0.1:
            return OrderType.VWAP
        if size_pct_volume > 0.05:
            return OrderType.TWAP
        if volatility < 0.03:
            return OrderType.LIMIT
        return OrderType.MARKET

    async def _calculate_twap_schedule(
        self, total_qty: float, duration_minutes: int, num_slices: int
    ) -> List[Dict[str, Any]]:
        """Calculate TWAP execution schedule."""
        slice_qty = total_qty / num_slices
        interval = duration_minutes / num_slices

        return [
            {
                "slice": i + 1,
                "quantity": slice_qty,
                "time_offset_minutes": i * interval,
            }
            for i in range(num_slices)
        ]

    async def _calculate_vwap_schedule(
        self, total_qty: float, volume_profile: List[float]
    ) -> List[Dict[str, Any]]:
        """Calculate VWAP execution schedule based on volume profile."""
        total_volume = sum(volume_profile)
        if total_volume <= 0:
            return await self._calculate_twap_schedule(
                total_qty, len(volume_profile) * 10, len(volume_profile)
            )

        normalized = [v / total_volume for v in volume_profile]
        return [
            {
                "slice": i + 1,
                "quantity": total_qty * weight,
                "time_offset_minutes": i * 10,
                "volume_weight": weight,
            }
            for i, weight in enumerate(normalized)
        ]

    async def _simulate_execution(self, order: Order) -> List[Dict[str, Any]]:
        """Simulate order execution with realistic slippage."""
        fills = []

        if order.order_type in [OrderType.TWAP, OrderType.VWAP]:
            schedule = order.metadata.get("schedule", [])
            for entry in schedule:
                fill = await self._simulate_fill(order, entry["quantity"], entry["slice"])
                fills.append(fill)
        else:
            fill = await self._simulate_fill(order, order.quantity)
            fills.append(fill)

        return fills

    async def _simulate_fill(
        self, order: Order, quantity: float, slice_index: int = 1
    ) -> Dict[str, Any]:
        """Simulate a single fill with slippage."""
        base_price = order.metadata.get("current_price", 100.0)
        estimated_slippage = order.metadata.get(
            "estimated_slippage_bps", self.default_slippage_bps
        )

        actual_slippage = estimated_slippage * (0.5 + self._rng.random())

        if order.side == OrderSide.BUY:
            fill_price = base_price * (1 + actual_slippage / 10000)
        else:
            fill_price = base_price * (1 - actual_slippage / 10000)

        if order.limit_price is not None:
            if order.side == OrderSide.BUY:
                fill_price = min(fill_price, order.limit_price)
            else:
                fill_price = max(fill_price, order.limit_price)

        fees = quantity * fill_price * (self.fee_bps / 10000)

        return {
            "fill_id": f"{order.order_id}-F{slice_index}",
            "order_id": order.order_id,
            "quantity": quantity,
            "price": fill_price,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fees": fees,
            "slippage_bps": actual_slippage,
        }

    async def _calculate_execution_metrics(
        self, order: Order, fills: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate execution quality metrics."""
        if not fills or order.filled_quantity == 0:
            return {
                "slippage_bps": 0.0,
                "total_fees": 0.0,
                "implementation_shortfall_bps": 0.0,
                "fill_rate": 0.0,
            }

        total_fees = sum(f["fees"] for f in fills)
        avg_slippage = sum(
            f["quantity"] * f.get("slippage_bps", 0.0) for f in fills
        ) / order.filled_quantity

        arrival_price = order.metadata.get("current_price", order.avg_fill_price)
        if order.side == OrderSide.BUY:
            shortfall = (order.avg_fill_price - arrival_price) / arrival_price
        else:
            shortfall = (arrival_price - order.avg_fill_price) / arrival_price

        return {
            "slippage_bps": avg_slippage,
            "total_fees": total_fees,
            "implementation_shortfall_bps": shortfall * 10000,
            "fill_rate": order.filled_quantity / order.quantity,
            "avg_fill_price": order.avg_fill_price,
            "num_fills": len(fills),
        }

    def _parse_trade_request(
        self, task: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse trade request from task and metadata."""
        trade_request = metadata.get("trade_request", {})
        return {
            "symbol": trade_request.get("symbol", "AAPL"),
            "side": trade_request.get("side", OrderSide.BUY),
            "quantity": trade_request.get("quantity", 100.0),
            "urgency": trade_request.get("urgency", "normal"),
            "limit_price": trade_request.get("limit_price"),
            "stop_price": trade_request.get("stop_price"),
            "duration_minutes": trade_request.get("duration_minutes", 60),
            "num_slices": trade_request.get("num_slices", 10),
        }

    def _order_to_dict(self, order: Order) -> Dict[str, Any]:
        """Convert order to dictionary for serialization."""
        return {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "order_type": order.order_type.value,
            "status": order.status.value,
            "filled_qty": order.filled_quantity,
            "avg_fill_price": order.avg_fill_price,
            "created_at": order.created_at,
            "num_fills": len(order.fills),
        }

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.active_orders.get(order_id) or self.completed_orders.get(order_id)

    def get_active_orders(self) -> List[Order]:
        """Get all active orders."""
        return list(self.active_orders.values())

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order."""
        if order_id not in self.active_orders:
            self.logger.warning("Order %s not found in active orders", order_id)
            return False

        order = self.active_orders[order_id]
        order.status = OrderStatus.CANCELLED
        self.completed_orders[order_id] = order
        del self.active_orders[order_id]

        self.logger.info("Order %s cancelled", order_id)
        return True
