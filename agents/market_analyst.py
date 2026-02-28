"""
Market Analyst Agent
====================
Specialized agent for market data analysis, pattern recognition,
technical indicators, and sentiment analysis.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional
from dataclasses import dataclass
import logging
import math
import time

from core.base_agent import BaseAgent, AgentContext, AgentAction
from core.config import get_default_agent_configs

if TYPE_CHECKING:
    from memory.agemem import AgeMem
    from communication.router import MessageRouter


@dataclass
class TechnicalIndicators:
    """Technical indicators calculation results."""
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None


@dataclass
class MarketRegime:
    """Market regime classification."""
    regime_type: str  # trending, ranging, volatile
    trend_direction: Optional[str] = None  # bullish, bearish, neutral
    volatility_level: str = "normal"  # low, normal, high
    confidence: float = 0.0


class MarketAnalystAgent(BaseAgent):
    """
    Market Analyst Agent for ATHENA multi-agent system.

    Responsibilities:
    - Real-time market data analysis
    - Pattern recognition (head-and-shoulders, double-top, etc.)
    - Technical indicator calculation (MA, EMA, RSI, MACD, Bollinger Bands)
    - Sentiment analysis from news/social media
    - Market regime classification
    """

    def __init__(
        self,
        name: str = "market_analyst",
        role: str = "analyst",
        system_prompt: Optional[str] = None,
        model: Optional[Any] = None,
        memory: Optional["AgeMem"] = None,
        communication: Optional[Any] = None,
        router: Optional["MessageRouter"] = None,
        tools: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Market Analyst Agent.

        Args:
            name: Agent name
            role: Agent role
            system_prompt: System prompt for agent behavior
            model: Language model instance
            memory: AgeMem memory instance
            communication: LatentSpace communication instance
            router: MessageRouter for LatentMAS routing (optional)
            tools: List of available tool names
            config: Additional configuration
        """
        if system_prompt is None:
            default_configs = get_default_agent_configs()
            system_prompt = default_configs["market_analyst"].system_prompt

        if tools is None:
            tools = ["analyze_price", "detect_patterns", "analyze_sentiment"]

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
        self.min_data_points = config.get("min_data_points", 50) if config else 50
        self.pattern_lookback = config.get("pattern_lookback", 100) if config else 100

    async def think(self, context: AgentContext) -> Dict[str, Any]:
        """
        Analyze market data and generate insights.

        Args:
            context: Context containing task, history, memory, and messages

        Returns:
            Dictionary with analysis results, patterns detected, indicators, sentiment
        """
        self.logger.info("Analyzing market data for task: %s", context.task)

        market_data = context.metadata.get("market_data", {})
        price_data = market_data.get("prices", [])
        text_data = market_data.get("news", [])

        # Retrieve relevant memory context
        memory_context = []
        if self.memory:
            try:
                memory_context = await self.memory.retrieve(
                    query=f"market analysis {market_data.get('symbol', 'market') if market_data else 'market'}",
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

        analysis_result = {
            "task": context.task,
            "timestamp": time.time(),
            "indicators": None,
            "patterns": [],
            "sentiment": None,
            "market_regime": None,
            "confidence": 0.0,
            "done": False,
        }

        if price_data and len(price_data) >= self.min_data_points:
            indicators = await self._calculate_technical_indicators(price_data)
            analysis_result["indicators"] = indicators

            patterns = await self._detect_patterns(price_data)
            analysis_result["patterns"] = patterns

            market_regime = await self._calculate_market_regime(indicators, price_data)
            analysis_result["market_regime"] = market_regime

            analysis_result["confidence"] = self._calculate_confidence(
                len(price_data), indicators, patterns
            )
        else:
            self.logger.warning(
                f"Insufficient price data: {len(price_data)} points "
                f"(need {self.min_data_points})"
            )

        if text_data:
            sentiment_score = await self._analyze_sentiment(text_data)
            analysis_result["sentiment"] = sentiment_score

        if analysis_result["indicators"] or analysis_result["sentiment"]:
            analysis_result["done"] = True

        analysis_result["memory_context"] = memory_context
        analysis_result["latent_messages"] = latent_messages
        return analysis_result

    async def act(self, thought: Dict[str, Any]) -> AgentAction:
        """
        Formulate actionable recommendations based on analysis.

        Args:
            thought: Analysis results from think()

        Returns:
            AgentAction with recommendations and supporting evidence
        """
        start_time = time.time()

        try:
            indicators = thought.get("indicators")
            patterns = thought.get("patterns", [])
            sentiment = thought.get("sentiment")
            market_regime = thought.get("market_regime")
            confidence = thought.get("confidence", 0.0)

            recommendation = {
                "action": "hold",
                "confidence": confidence,
                "reasons": [],
                "supporting_evidence": {},
            }

            if indicators:
                signal = self._interpret_indicators(indicators)
                recommendation["action"] = signal["action"]
                recommendation["reasons"].extend(signal["reasons"])
                recommendation["supporting_evidence"]["indicators"] = {
                    "rsi_14": indicators.rsi_14,
                    "macd": indicators.macd,
                    "macd_signal": indicators.macd_signal,
                }

            if patterns:
                pattern_signal = self._interpret_patterns(patterns)
                if pattern_signal["action"] != "hold":
                    recommendation["action"] = pattern_signal["action"]
                recommendation["reasons"].extend(pattern_signal["reasons"])
                recommendation["supporting_evidence"]["patterns"] = patterns

            if sentiment is not None:
                sentiment_signal = self._interpret_sentiment(sentiment)
                recommendation["reasons"].append(sentiment_signal)
                recommendation["supporting_evidence"]["sentiment"] = sentiment

            if market_regime:
                regime_context = (
                    f"Market is in {market_regime.regime_type} regime "
                    f"with {market_regime.volatility_level} volatility"
                )
                recommendation["reasons"].append(regime_context)
                recommendation["supporting_evidence"]["regime"] = {
                    "type": market_regime.regime_type,
                    "trend": market_regime.trend_direction,
                    "volatility": market_regime.volatility_level,
                }

            duration = time.time() - start_time

            # Send output via LatentMAS to coordinator
            if self.router:
                try:
                    from communication.router import MessagePriority
                    await self.router.send(
                        sender_id=self.name,
                        receiver_id="coordinator",
                        message=recommendation,
                        priority=MessagePriority.MEDIUM,
                    )
                except Exception as e:
                    self.logger.warning("LatentMAS send failed: %s", e)

            # Store decision to memory
            if self.memory:
                try:
                    await self.memory.add(
                        content={"analysis": recommendation, "thought": thought},
                        metadata={
                            "agent": self.name,
                            "role": self.role,
                            "success": True,
                        }
                    )
                except Exception as e:
                    self.logger.warning("Memory store failed: %s", e)

            return AgentAction(
                action_type="market_analysis",
                parameters={"task": thought.get("task")},
                result=recommendation,
                success=True,
                error=None,
                duration=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error("Error in act(): %s", e)

            # Store failure to memory
            if self.memory:
                try:
                    await self.memory.add(
                        content={"analysis": None, "thought": thought},
                        metadata={
                            "agent": self.name,
                            "role": self.role,
                            "success": False,
                        }
                    )
                except Exception as mem_e:
                    self.logger.warning("Memory store failed: %s", mem_e)

            return AgentAction(
                action_type="market_analysis",
                parameters={"task": thought.get("task")},
                result=None,
                success=False,
                error=str(e),
                duration=duration,
            )

    async def _calculate_technical_indicators(
        self, price_data: List[float]
    ) -> TechnicalIndicators:
        """
        Calculate technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands).

        Args:
            price_data: List of closing prices

        Returns:
            TechnicalIndicators with calculated values
        """
        indicators = TechnicalIndicators()

        if len(price_data) < 2:
            return indicators

        if len(price_data) >= 20:
            indicators.sma_20 = sum(price_data[-20:]) / 20

        if len(price_data) >= 50:
            indicators.sma_50 = sum(price_data[-50:]) / 50

        if len(price_data) >= 26:
            # Compute EMA-12 and EMA-26 series in a single O(n) pass each
            alpha12 = 2.0 / (12 + 1)
            alpha26 = 2.0 / (26 + 1)

            ema12_val = price_data[0]
            ema26_val = price_data[0]
            ema12_series: List[float] = []
            ema26_series: List[float] = []

            for price in price_data:
                ema12_val = alpha12 * price + (1 - alpha12) * ema12_val
                ema26_val = alpha26 * price + (1 - alpha26) * ema26_val
                ema12_series.append(ema12_val)
                ema26_series.append(ema26_val)

            indicators.ema_12 = ema12_series[-1]
            indicators.ema_26 = ema26_series[-1]

            macd_series = [e12 - e26 for e12, e26 in zip(ema12_series[25:], ema26_series[25:])]
            indicators.macd = macd_series[-1]
            if len(macd_series) >= 9:
                indicators.macd_signal = self._calculate_ema(macd_series, 9)
            else:
                indicators.macd_signal = sum(macd_series) / len(macd_series)
            indicators.macd_histogram = indicators.macd - indicators.macd_signal
        elif len(price_data) >= 12:
            indicators.ema_12 = self._calculate_ema(price_data, 12)

        if len(price_data) >= 14:
            indicators.rsi_14 = self._calculate_rsi(price_data, 14)

        if len(price_data) >= 20 and indicators.sma_20 is not None:
            std_dev = self._calculate_std_dev(price_data[-20:], indicators.sma_20)
            indicators.bb_middle = indicators.sma_20
            indicators.bb_upper = indicators.sma_20 + (2 * std_dev)
            indicators.bb_lower = indicators.sma_20 - (2 * std_dev)

        return indicators

    async def _detect_patterns(self, price_data: List[float]) -> List[str]:
        """
        Detect chart patterns in price data.

        Args:
            price_data: List of closing prices

        Returns:
            List of detected pattern names
        """
        patterns = []

        if len(price_data) < 20:
            return patterns

        lookback = min(self.pattern_lookback, len(price_data))
        recent_prices = price_data[-lookback:]

        if self._is_double_top(recent_prices):
            patterns.append("double_top")
        if self._is_double_bottom(recent_prices):
            patterns.append("double_bottom")
        if self._is_head_and_shoulders(recent_prices):
            patterns.append("head_and_shoulders")
        if self._is_ascending_triangle(recent_prices):
            patterns.append("ascending_triangle")
        if self._is_descending_triangle(recent_prices):
            patterns.append("descending_triangle")

        return patterns

    async def _analyze_sentiment(self, text_data: List[str]) -> float:
        """
        Analyze sentiment from news/social media text.

        Args:
            text_data: List of text strings

        Returns:
            Sentiment score (-1.0 to 1.0)
        """
        if not text_data:
            return 0.0

        positive_keywords = [
            "bullish", "buy", "growth", "strong", "rally", "up", "gain",
            "positive", "profit", "surge", "rise", "boom", "outperform",
        ]
        negative_keywords = [
            "bearish", "sell", "decline", "weak", "crash", "down", "loss",
            "negative", "drop", "fall", "recession", "underperform",
        ]

        total_score = 0.0
        for text in text_data:
            text_lower = text.lower()
            pos_count = sum(1 for kw in positive_keywords if kw in text_lower)
            neg_count = sum(1 for kw in negative_keywords if kw in text_lower)
            if pos_count + neg_count > 0:
                total_score += (pos_count - neg_count) / (pos_count + neg_count)

        sentiment_score = total_score / len(text_data) if text_data else 0.0
        return max(-1.0, min(1.0, sentiment_score))

    async def _calculate_market_regime(
        self, indicators: Optional[TechnicalIndicators], price_data: List[float]
    ) -> MarketRegime:
        """
        Classify market regime based on indicators and price action.

        Args:
            indicators: Technical indicators
            price_data: Price data

        Returns:
            MarketRegime classification
        """
        if not indicators or len(price_data) < 20:
            return MarketRegime(regime_type="unknown", confidence=0.0)

        recent_prices = price_data[-20:]
        mean_price = sum(recent_prices) / len(recent_prices)
        volatility = self._calculate_std_dev(recent_prices, mean_price)
        volatility_pct = (volatility / mean_price) * 100

        if volatility_pct < 1.0:
            volatility_level = "low"
        elif volatility_pct < 3.0:
            volatility_level = "normal"
        else:
            volatility_level = "high"

        trend_direction = "neutral"
        if indicators.sma_20 and indicators.sma_50:
            if indicators.sma_20 > indicators.sma_50 * 1.02:
                trend_direction = "bullish"
            elif indicators.sma_20 < indicators.sma_50 * 0.98:
                trend_direction = "bearish"

        regime_type = "ranging"
        if trend_direction != "neutral" and volatility_level != "high":
            regime_type = "trending"
        elif volatility_level == "high":
            regime_type = "volatile"

        confidence = 0.7 if indicators.sma_50 else 0.5

        return MarketRegime(
            regime_type=regime_type,
            trend_direction=trend_direction,
            volatility_level=volatility_level,
            confidence=confidence,
        )

    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return sum(prices) / len(prices)

        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period

        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return ema

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index using Wilder's smoothing."""
        if len(prices) < period + 1:
            return 50.0

        gains = []
        losses = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(abs(change))

        # First average: SMA of first `period` values
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        # Wilder's smoothing for subsequent values
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_std_dev(self, values: List[float], mean: float) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

    def _is_double_top(self, prices: List[float]) -> bool:
        """Detect double top pattern."""
        if len(prices) < 20:
            return False
        peaks = []
        for i in range(5, len(prices) - 5):
            if prices[i] > max(prices[i-5:i]) and prices[i] > max(prices[i+1:i+6]):
                peaks.append((i, prices[i]))
        if len(peaks) >= 2:
            last_two = peaks[-2:]
            price_diff = abs(last_two[0][1] - last_two[1][1])
            avg_price = (last_two[0][1] + last_two[1][1]) / 2
            if price_diff / avg_price < 0.02:
                return True
        return False

    def _is_double_bottom(self, prices: List[float]) -> bool:
        """Detect double bottom pattern."""
        if len(prices) < 20:
            return False
        troughs = []
        for i in range(5, len(prices) - 5):
            if prices[i] < min(prices[i-5:i]) and prices[i] < min(prices[i+1:i+6]):
                troughs.append((i, prices[i]))
        if len(troughs) >= 2:
            last_two = troughs[-2:]
            price_diff = abs(last_two[0][1] - last_two[1][1])
            avg_price = (last_two[0][1] + last_two[1][1]) / 2
            if price_diff / avg_price < 0.02:
                return True
        return False

    def _is_head_and_shoulders(self, prices: List[float]) -> bool:
        """Detect head and shoulders pattern."""
        if len(prices) < 30:
            return False
        peaks = []
        for i in range(5, len(prices) - 5):
            if prices[i] > max(prices[i-5:i]) and prices[i] > max(prices[i+1:i+6]):
                peaks.append((i, prices[i]))
        if len(peaks) >= 3:
            last_three = peaks[-3:]
            left_shoulder = last_three[0][1]
            head = last_three[1][1]
            right_shoulder = last_three[2][1]
            if head > left_shoulder and head > right_shoulder:
                shoulder_diff = abs(left_shoulder - right_shoulder)
                avg_shoulder = (left_shoulder + right_shoulder) / 2
                if shoulder_diff / avg_shoulder < 0.05:
                    return True
        return False

    def _is_ascending_triangle(self, prices: List[float]) -> bool:
        """Detect ascending triangle pattern."""
        if len(prices) < 20:
            return False
        recent = prices[-20:]
        highs = [recent[i] for i in range(0, len(recent), 2)]
        lows = [recent[i] for i in range(1, len(recent), 2)]
        if len(highs) < 3 or len(lows) < 3:
            return False
        high_mean = sum(highs) / len(highs)
        high_variance = sum((h - high_mean) ** 2 for h in highs) / len(highs)
        low_slope = (lows[-1] - lows[0]) / len(lows)
        return high_variance / (high_mean ** 2) < 0.001 and low_slope > 0

    def _is_descending_triangle(self, prices: List[float]) -> bool:
        """Detect descending triangle pattern."""
        if len(prices) < 20:
            return False
        recent = prices[-20:]
        highs = [recent[i] for i in range(0, len(recent), 2)]
        lows = [recent[i] for i in range(1, len(recent), 2)]
        if len(highs) < 3 or len(lows) < 3:
            return False
        low_mean = sum(lows) / len(lows)
        low_variance = sum((l - low_mean) ** 2 for l in lows) / len(lows)
        high_slope = (highs[-1] - highs[0]) / len(highs)
        return low_variance / (low_mean ** 2) < 0.001 and high_slope < 0

    def _calculate_confidence(
        self, data_points: int, indicators: Optional[TechnicalIndicators],
        patterns: List[str],
    ) -> float:
        """Calculate overall confidence in analysis."""
        confidence = 0.0
        if data_points >= 100:
            confidence += 0.4
        elif data_points >= 50:
            confidence += 0.3
        else:
            confidence += 0.1

        if indicators:
            indicator_count = sum([
                indicators.sma_20 is not None,
                indicators.sma_50 is not None,
                indicators.rsi_14 is not None,
                indicators.macd is not None,
            ])
            confidence += (indicator_count / 4) * 0.4

        if patterns:
            confidence += min(len(patterns) * 0.1, 0.2)

        return min(confidence, 1.0)

    def _interpret_indicators(self, indicators: TechnicalIndicators) -> Dict[str, Any]:
        """Interpret technical indicators to generate trading signal."""
        signal = {"action": "hold", "reasons": []}

        if indicators.rsi_14 is not None:
            if indicators.rsi_14 > 70:
                signal["action"] = "sell"
                signal["reasons"].append(f"RSI overbought at {indicators.rsi_14:.1f}")
            elif indicators.rsi_14 < 30:
                signal["action"] = "buy"
                signal["reasons"].append(f"RSI oversold at {indicators.rsi_14:.1f}")

        if indicators.macd is not None and indicators.macd_signal is not None:
            if indicators.macd > indicators.macd_signal:
                if signal["action"] == "hold":
                    signal["action"] = "buy"
                signal["reasons"].append("MACD bullish crossover")
            elif indicators.macd < indicators.macd_signal:
                if signal["action"] == "hold":
                    signal["action"] = "sell"
                signal["reasons"].append("MACD bearish crossover")

        if indicators.sma_20 is not None and indicators.sma_50 is not None:
            if indicators.sma_20 > indicators.sma_50:
                signal["reasons"].append("Short-term MA above long-term MA (bullish)")
            else:
                signal["reasons"].append("Short-term MA below long-term MA (bearish)")

        return signal

    def _interpret_patterns(self, patterns: List[str]) -> Dict[str, Any]:
        """Interpret chart patterns to generate signal."""
        signal = {"action": "hold", "reasons": []}
        bullish_patterns = ["double_bottom", "ascending_triangle"]
        bearish_patterns = ["double_top", "head_and_shoulders", "descending_triangle"]

        for pattern in patterns:
            if pattern in bullish_patterns:
                signal["action"] = "buy"
                signal["reasons"].append(f"Bullish pattern detected: {pattern}")
            elif pattern in bearish_patterns:
                signal["action"] = "sell"
                signal["reasons"].append(f"Bearish pattern detected: {pattern}")

        return signal

    def _interpret_sentiment(self, sentiment_score: float) -> str:
        """Interpret sentiment score."""
        if sentiment_score > 0.3:
            return f"Positive market sentiment ({sentiment_score:.2f})"
        elif sentiment_score < -0.3:
            return f"Negative market sentiment ({sentiment_score:.2f})"
        else:
            return f"Neutral market sentiment ({sentiment_score:.2f})"

    def _get_tool_definition(self, tool_name: str) -> Dict[str, Any]:
        """Get tool definition for market analyst tools."""
        tools = {
            "analyze_price": {
                "name": "analyze_price",
                "description": "Analyze price data and calculate technical indicators",
                "parameters": {
                    "prices": {"type": "array", "description": "List of closing prices"},
                    "volumes": {"type": "array", "description": "List of volumes (optional)"},
                },
            },
            "detect_patterns": {
                "name": "detect_patterns",
                "description": "Detect chart patterns in price data",
                "parameters": {
                    "prices": {"type": "array", "description": "List of closing prices"},
                    "lookback": {"type": "integer", "description": "Lookback period"},
                },
            },
            "analyze_sentiment": {
                "name": "analyze_sentiment",
                "description": "Analyze sentiment from text data (news, social media)",
                "parameters": {
                    "texts": {"type": "array", "description": "List of text strings"},
                },
            },
        }
        return tools.get(tool_name, super()._get_tool_definition(tool_name))
