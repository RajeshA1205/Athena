"""
ATHENA Strategy Agent
=====================
Strategy formulation, backtesting, and portfolio optimization agent.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
import logging
import math
import time

from core.base_agent import BaseAgent, AgentContext, AgentAction
from core.config import get_default_agent_configs

if TYPE_CHECKING:
    from memory.agemem import AgeMem
    from communication.router import MessageRouter


class StrategyType(Enum):
    """Available trading strategy types."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    PAIRS_TRADING = "pairs_trading"
    BREAKOUT = "breakout"


@dataclass
class TradingSignal:
    """Trading signal with entry/exit criteria."""
    asset: str
    signal_type: str  # buy, sell, hold
    strength: float  # 0.0 to 1.0
    entry_price: float
    exit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: float = 0.0
    reasoning: str = ""


@dataclass
class BacktestResult:
    """Results from strategy backtesting."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    avg_trade_duration: float
    profit_factor: float


class StrategyAgent(BaseAgent):
    """
    Strategy Agent for trading strategy formulation and optimization.

    Responsibilities:
    - Analyze market conditions and select optimal strategy type
    - Generate trading signals based on selected strategy
    - Backtest strategies on historical data
    - Optimize portfolio allocation
    - Calculate position sizes based on risk parameters
    """

    def __init__(
        self,
        name: str = "strategy_agent",
        role: str = "strategy",
        system_prompt: Optional[str] = None,
        model: Optional[Any] = None,
        memory: Optional["AgeMem"] = None,
        communication: Optional[Any] = None,
        router: Optional["MessageRouter"] = None,
        tools: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Strategy Agent.

        Args:
            name: Agent identifier
            role: Agent role
            system_prompt: System prompt defining behavior
            model: Language model instance
            memory: AgeMem memory instance
            communication: LatentSpace communication instance
            router: MessageRouter for LatentMAS routing (optional)
            tools: Available tool names
            config: Additional configuration
        """
        if system_prompt is None:
            default_configs = get_default_agent_configs()
            system_prompt = default_configs["strategy_agent"].system_prompt

        super().__init__(
            name=name,
            role=role,
            system_prompt=system_prompt,
            model=model,
            memory=memory,
            communication=communication,
            tools=tools or ["generate_signals", "backtest", "optimize_params"],
            config=config,
        )

        self.memory: Optional["AgeMem"] = memory
        self.router: Optional["MessageRouter"] = router

    async def think(self, context: AgentContext) -> Dict[str, Any]:
        """
        Process context and decide on strategy formulation approach.

        Args:
            context: Current context including task, history, memory, messages

        Returns:
            Dictionary containing strategy selection and parameters
        """
        self.logger.info("Thinking about task: %s", context.task)

        market_data = context.metadata.get("market_data", {})
        prices = market_data.get("prices", [])
        volatility = market_data.get("volatility", 0.0)
        trend_strength = market_data.get("trend_strength", 0.0)

        # Retrieve relevant memory context
        memory_context = []
        if self.memory:
            try:
                memory_context = await self.memory.retrieve(
                    query=f"trading strategy {market_data.get('symbol', '') if market_data else ''}",
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

        market_regime = self._classify_market_regime(prices, volatility, trend_strength)
        strategy_type = await self._select_strategy(market_regime, volatility, trend_strength)

        strategy_params: Dict[str, Any] = {
            "type": strategy_type.value,
            "market_regime": market_regime,
            "volatility": volatility,
            "trend_strength": trend_strength,
            "prices": prices,
        }

        if strategy_type == StrategyType.MOMENTUM:
            strategy_params["lookback_period"] = 20
            strategy_params["threshold"] = 0.02
        elif strategy_type == StrategyType.MEAN_REVERSION:
            strategy_params["window"] = 20
            strategy_params["num_std"] = 2.0
        elif strategy_type == StrategyType.BREAKOUT:
            strategy_params["window"] = 20
            strategy_params["breakout_threshold"] = 1.5

        rationale = (
            f"Selected {strategy_type.value} strategy based on "
            f"market regime: {market_regime}. "
            f"Volatility: {volatility:.4f}, Trend strength: {trend_strength:.4f}"
        )

        thought: Dict[str, Any] = {
            "strategy": strategy_params,
            "rationale": rationale,
            "action": "generate_signals",
            "memory_context": memory_context,
            "latent_messages": latent_messages,
            "done": False,
        }

        # Optional LLM analysis for deeper market pattern recognition
        signals_summary = {k: v for k, v in strategy_params.items() if k != "prices"}
        llm_analysis = await self._llm_reason(
            f"Analyze these market signals: {signals_summary}. What patterns do you see?"
        )
        if llm_analysis:
            thought["llm_analysis"] = llm_analysis

        return thought

    async def act(self, thought: Dict[str, Any]) -> AgentAction:
        """
        Execute strategy action to generate trade recommendations.

        Args:
            thought: Output from think()

        Returns:
            AgentAction with signals or backtest results
        """
        start_time = time.time()

        try:
            strategy_params = thought.get("strategy", {})
            action_type = thought.get("action", "generate_signals")

            if action_type == "generate_signals":
                signals = await self._generate_signals(strategy_params)
                result = {
                    "signals": [
                        {
                            "asset": s.asset,
                            "type": s.signal_type,
                            "strength": s.strength,
                            "entry_price": s.entry_price,
                            "exit_price": s.exit_price,
                            "stop_loss": s.stop_loss,
                            "take_profit": s.take_profit,
                            "position_size": s.position_size,
                            "reasoning": s.reasoning,
                        }
                        for s in signals
                    ],
                    "strategy": {k: v for k, v in strategy_params.items() if k != "prices"},
                }
                success = True
                error = None

            elif action_type == "backtest":
                backtest_result = await self._backtest_strategy(
                    strategy_params, strategy_params.get("prices", [])
                )
                result = {
                    "total_return": backtest_result.total_return,
                    "sharpe_ratio": backtest_result.sharpe_ratio,
                    "max_drawdown": backtest_result.max_drawdown,
                    "win_rate": backtest_result.win_rate,
                    "num_trades": backtest_result.num_trades,
                }
                success = True
                error = None
            else:
                result = None
                success = False
                error = f"Unknown action type: {action_type}"

            duration = time.time() - start_time

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
                        content={"strategy": result, "thought": thought},
                        metadata={
                            "agent": self.name,
                            "role": self.role,
                            "success": success,
                        }
                    )
                except Exception as e:
                    self.logger.warning("Memory store failed: %s", e)

            return AgentAction(
                action_type=action_type,
                parameters={k: v for k, v in strategy_params.items() if k != "prices"},
                result=result,
                success=success,
                error=error,
                duration=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error("Action failed: %s", str(e))

            # Store failure to memory
            if self.memory:
                try:
                    await self.memory.add(
                        content={"strategy": None, "thought": thought},
                        metadata={
                            "agent": self.name,
                            "role": self.role,
                            "success": False,
                        }
                    )
                except Exception as mem_e:
                    self.logger.warning("Memory store failed: %s", mem_e)

            return AgentAction(
                action_type=thought.get("action", "unknown"),
                parameters={},
                result=None,
                success=False,
                error=str(e),
                duration=duration,
            )

    def _classify_market_regime(
        self, prices: List[float], volatility: float, trend_strength: float
    ) -> str:
        """Classify current market regime."""
        if not prices or len(prices) < 2:
            return "unknown"
        if volatility > 0.3:
            return "high_volatility"
        if abs(trend_strength) > 0.5:
            return "trending" if trend_strength > 0 else "downtrending"
        if volatility < 0.1 and abs(trend_strength) < 0.2:
            return "mean_reverting"
        return "neutral"

    async def _select_strategy(
        self, market_regime: str, volatility: float, trend_strength: float
    ) -> StrategyType:
        """
        Select optimal strategy type based on market conditions.

        Args:
            market_regime: Current market regime
            volatility: Market volatility
            trend_strength: Trend strength indicator

        Returns:
            Selected strategy type
        """
        if market_regime in ["trending", "downtrending"]:
            if abs(trend_strength) > 0.7:
                return StrategyType.MOMENTUM
            return StrategyType.BREAKOUT

        if market_regime == "mean_reverting":
            return StrategyType.MEAN_REVERSION

        if market_regime == "high_volatility":
            return StrategyType.PAIRS_TRADING

        return StrategyType.MOMENTUM

    async def _generate_signals(self, strategy_params: Dict[str, Any]) -> List[TradingSignal]:
        """Generate trading signals based on strategy parameters."""
        strategy_type = strategy_params.get("type", "momentum")
        prices = strategy_params.get("prices", [])

        if strategy_type == "momentum":
            return await self._generate_momentum_signals(
                prices, strategy_params.get("lookback_period", 20)
            )
        elif strategy_type == "mean_reversion":
            return await self._generate_mean_reversion_signals(
                prices, strategy_params.get("window", 20),
                strategy_params.get("num_std", 2.0)
            )
        return []

    async def _generate_momentum_signals(
        self, prices: List[float], lookback: int = 20
    ) -> List[TradingSignal]:
        """
        Generate momentum-based trading signals.

        Args:
            prices: Historical price data
            lookback: Lookback period for momentum calculation

        Returns:
            List of momentum trading signals
        """
        if not prices or len(prices) < lookback + 1:
            return []

        signals = []
        current_price = prices[-1]
        lookback_price = prices[-lookback - 1]
        momentum = (current_price - lookback_price) / lookback_price

        mean_price = sum(prices[-lookback:]) / lookback
        std_price = math.sqrt(
            sum((p - mean_price) ** 2 for p in prices[-lookback:]) / lookback
        )
        vol = std_price / mean_price if mean_price > 0 else 0.1

        if momentum > 0.02:
            signal = TradingSignal(
                asset="ASSET",
                signal_type="buy",
                strength=min(abs(momentum) * 10, 1.0),
                entry_price=current_price,
                stop_loss=current_price * 0.98,
                take_profit=current_price * 1.05,
                reasoning=f"Positive momentum: {momentum:.4f}",
            )
            signal.position_size = await self._calculate_position_size(
                signal.strength, risk_budget=0.02, volatility=vol
            )
            signals.append(signal)

        elif momentum < -0.02:
            signal = TradingSignal(
                asset="ASSET",
                signal_type="sell",
                strength=min(abs(momentum) * 10, 1.0),
                entry_price=current_price,
                stop_loss=current_price * 1.02,
                take_profit=current_price * 0.95,
                reasoning=f"Negative momentum: {momentum:.4f}",
            )
            signal.position_size = await self._calculate_position_size(
                signal.strength, risk_budget=0.02, volatility=vol
            )
            signals.append(signal)

        return signals

    async def _generate_mean_reversion_signals(
        self, prices: List[float], window: int = 20, num_std: float = 2.0
    ) -> List[TradingSignal]:
        """
        Generate mean reversion trading signals.

        Args:
            prices: Historical price data
            window: Moving average window
            num_std: Number of standard deviations for bands

        Returns:
            List of mean reversion trading signals
        """
        if not prices or len(prices) < window + 1:
            return []

        signals = []
        recent_prices = prices[-window:]
        mean_price = sum(recent_prices) / len(recent_prices)
        std_price = math.sqrt(
            sum((p - mean_price) ** 2 for p in recent_prices) / len(recent_prices)
        )

        current_price = prices[-1]
        z_score = (current_price - mean_price) / (std_price + 1e-10)

        if z_score < -num_std:
            signal = TradingSignal(
                asset="ASSET",
                signal_type="buy",
                strength=min(abs(z_score) / num_std, 1.0),
                entry_price=current_price,
                exit_price=mean_price,
                stop_loss=current_price * 0.97,
                take_profit=mean_price,
                reasoning=f"Price {abs(z_score):.2f} std below mean",
            )
            signal.position_size = await self._calculate_position_size(
                signal.strength, risk_budget=0.02,
                volatility=std_price / mean_price if mean_price > 0 else 0.1,
            )
            signals.append(signal)

        elif z_score > num_std:
            signal = TradingSignal(
                asset="ASSET",
                signal_type="sell",
                strength=min(abs(z_score) / num_std, 1.0),
                entry_price=current_price,
                exit_price=mean_price,
                stop_loss=current_price * 1.03,
                take_profit=mean_price,
                reasoning=f"Price {abs(z_score):.2f} std above mean",
            )
            signal.position_size = await self._calculate_position_size(
                signal.strength, risk_budget=0.02,
                volatility=std_price / mean_price if mean_price > 0 else 0.1,
            )
            signals.append(signal)

        return signals

    async def _optimize_portfolio(
        self, assets: List[str], returns: List[List[float]],
        method: str = "mean_variance",
    ) -> Dict[str, float]:
        """
        Optimize portfolio allocation across assets.

        Args:
            assets: List of asset identifiers
            returns: List of return series per asset
            method: Optimization method (mean_variance, risk_parity, equal_weight)

        Returns:
            Dictionary mapping assets to allocation weights
        """
        if not assets or not returns:
            return {}

        num_assets = len(assets)

        if method == "equal_weight":
            weight = 1.0 / num_assets
            return {asset: weight for asset in assets}

        elif method == "risk_parity":
            volatilities = []
            for r in returns:
                mean_r = sum(r) / len(r) if r else 0
                vol = math.sqrt(sum((x - mean_r) ** 2 for x in r) / len(r)) if r else 1.0
                volatilities.append(max(vol, 1e-10))

            inv_vol = [1.0 / v for v in volatilities]
            total_inv = sum(inv_vol)
            weights = [iv / total_inv for iv in inv_vol]
            return {asset: w for asset, w in zip(assets, weights)}

        elif method == "mean_variance":
            # Simplified mean-variance: weight by return/risk ratio
            ratios = []
            for r in returns:
                if not r:
                    ratios.append(0.0)
                    continue
                mean_r = sum(r) / len(r)
                vol = math.sqrt(sum((x - mean_r) ** 2 for x in r) / len(r))
                ratios.append(max(mean_r / (vol + 1e-10), 0))

            total = sum(ratios)
            if total > 0:
                weights = [r / total for r in ratios]
            else:
                weights = [1.0 / num_assets] * num_assets

            return {asset: w for asset, w in zip(assets, weights)}

        return {asset: 1.0 / num_assets for asset in assets}

    async def _calculate_position_size(
        self, signal_strength: float, risk_budget: float = 0.02,
        volatility: float = 0.1,
    ) -> float:
        """
        Calculate position size using volatility-based sizing.

        Args:
            signal_strength: Signal strength (0.0 to 1.0)
            risk_budget: Maximum risk per trade as fraction of portfolio
            volatility: Asset volatility

        Returns:
            Position size as fraction of portfolio
        """
        if volatility <= 0 or signal_strength <= 0:
            return 0.0

        base_size = risk_budget / (volatility + 1e-10)
        position_size = base_size * signal_strength
        return min(position_size, 0.2)

    async def _backtest_strategy(
        self, strategy: Dict[str, Any], prices: List[float]
    ) -> BacktestResult:
        """
        Run backtest on strategy and return performance metrics.

        Args:
            strategy: Strategy configuration
            prices: Historical price data

        Returns:
            Backtest results with performance metrics
        """
        empty = BacktestResult(
            total_return=0.0, sharpe_ratio=0.0, max_drawdown=0.0,
            win_rate=0.0, num_trades=0, avg_trade_duration=0.0, profit_factor=0.0,
        )

        if not prices or len(prices) < 30:
            return empty

        strategy_type = strategy.get("type", "momentum")
        lookback = strategy.get("lookback_period", 20)
        returns = []
        trades = []

        for i in range(lookback, len(prices)):
            window_prices = prices[max(0, i - lookback):i + 1]

            if strategy_type == "momentum":
                signals = await self._generate_momentum_signals(window_prices, lookback)
            else:
                signals = await self._generate_mean_reversion_signals(window_prices, lookback)

            period_return = (prices[i] - prices[i - 1]) / prices[i - 1]

            if signals:
                signal = signals[0]
                position = signal.position_size if signal.signal_type == "buy" else -signal.position_size
                trade_return = position * period_return
                returns.append(trade_return)
                trades.append({"return": trade_return, "duration": 1})
            else:
                returns.append(0)

        if not returns:
            return empty

        cumulative_for_return = 1.0
        for r in returns:
            cumulative_for_return *= (1 + r)
        total_return = cumulative_for_return - 1.0
        mean_return = sum(returns) / len(returns)
        std_return = math.sqrt(sum((r - mean_return) ** 2 for r in returns) / len(returns))
        sharpe_ratio = (mean_return / (std_return + 1e-10)) * math.sqrt(252)

        cumulative = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in returns:
            cumulative *= (1 + r)
            if cumulative > peak:
                peak = cumulative
            dd = (peak - cumulative) / peak
            if dd > max_dd:
                max_dd = dd

        winning_trades = [t for t in trades if t["return"] > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0.0

        gross_profit = sum(t["return"] for t in trades if t["return"] > 0)
        gross_loss = abs(sum(t["return"] for t in trades if t["return"] < 0))
        profit_factor = gross_profit / (gross_loss + 1e-10)

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_dd,
            win_rate=win_rate,
            num_trades=len(trades),
            avg_trade_duration=1.0,
            profit_factor=profit_factor,
        )
