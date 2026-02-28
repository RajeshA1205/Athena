"""
ATHENA Risk Manager Agent
==========================
Portfolio risk assessment and compliance monitoring agent.
"""

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from dataclasses import dataclass
import logging
import math

from core.base_agent import BaseAgent, AgentContext, AgentAction
from core.config import get_default_agent_configs

if TYPE_CHECKING:
    from memory.agemem import AgeMem
    from communication.router import MessageRouter


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    max_position_pct: float = 0.20
    max_sector_pct: float = 0.30
    max_drawdown: float = 0.10
    max_var_pct: float = 0.05
    max_leverage: float = 2.0
    min_liquidity_ratio: float = 0.10
    max_correlation: float = 0.70


class RiskManagerAgent(BaseAgent):
    """
    Risk Manager Agent for ATHENA multi-agent system.

    Responsibilities:
    - Portfolio risk assessment (volatility, correlation, concentration)
    - Exposure monitoring across positions
    - Value at Risk (VaR) and Expected Shortfall calculations
    - Compliance checks with risk limits and regulations
    """

    def __init__(
        self,
        name: str = "risk_manager",
        role: str = "risk",
        system_prompt: Optional[str] = None,
        model: Optional[Any] = None,
        memory: Optional["AgeMem"] = None,
        communication: Optional[Any] = None,
        router: Optional["MessageRouter"] = None,
        tools: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Risk Manager Agent.

        Args:
            name: Agent identifier
            role: Agent role
            system_prompt: System prompt defining behavior
            model: Language model instance
            memory: AgeMem memory instance
            communication: LatentMAS communication instance
            router: MessageRouter for LatentMAS routing (optional)
            tools: List of available tool names
            config: Additional configuration
        """
        if system_prompt is None:
            default_configs = get_default_agent_configs()
            system_prompt = default_configs["risk_manager"].system_prompt

        super().__init__(
            name=name,
            role=role,
            system_prompt=system_prompt,
            model=model,
            memory=memory,
            communication=communication,
            tools=tools or ["calculate_var", "check_exposure", "verify_compliance"],
            config=config,
        )

        self.memory: Optional["AgeMem"] = memory
        self.router: Optional["MessageRouter"] = router
        risk_config = config.get("risk_limits", {}) if config else {}
        self.risk_limits = RiskLimits(**risk_config)

    async def think(self, context: AgentContext) -> Dict[str, Any]:
        """
        Assess portfolio risk and identify exposures.

        Args:
            context: Current context including task, history, memory, messages

        Returns:
            Dictionary with risk metrics, VaR, exposures, compliance status
        """
        self.logger.info("Assessing risk for task: %s", context.task)

        positions = context.metadata.get("positions", [])
        returns_data = context.metadata.get("returns", {})

        # Retrieve relevant memory context
        memory_context = []
        if self.memory:
            try:
                portfolio = context.metadata.get("portfolio", {})
                memory_context = await self.memory.retrieve(
                    query=f"risk assessment portfolio {portfolio.get('total_value', '') if portfolio else ''}",
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

        if not positions:
            return {
                "risk_level": "low",
                "var_95": 0.0,
                "var_99": 0.0,
                "expected_shortfall": 0.0,
                "portfolio_metrics": {},
                "exposures": {},
                "compliance_issues": [],
                "alerts": [],
                "memory_context": memory_context,
                "latent_messages": latent_messages,
                "done": True,
            }

        var_95 = await self._calculate_var(returns_data, confidence=0.95)
        var_99 = await self._calculate_var(returns_data, confidence=0.99)
        expected_shortfall = await self._calculate_expected_shortfall(returns_data)
        portfolio_metrics = await self._calculate_portfolio_metrics(positions, returns_data)
        compliance_issues = await self._check_position_limits(positions, self.risk_limits)

        correlation_issues = []
        if len(positions) > 1 and returns_data:
            correlation_matrix = self._calculate_correlation_matrix(returns_data)
            if correlation_matrix:
                correlation_issues = await self._check_correlation_risk(
                    positions, correlation_matrix
                )

        all_issues = compliance_issues + correlation_issues
        risk_level = self._determine_risk_level(
            var_95, expected_shortfall, portfolio_metrics, all_issues
        )

        alerts = await self._generate_risk_alerts({
            "var_95": var_95,
            "var_99": var_99,
            "expected_shortfall": expected_shortfall,
            "portfolio_metrics": portfolio_metrics,
            "compliance_issues": all_issues,
        })

        return {
            "risk_level": risk_level,
            "var_95": var_95,
            "var_99": var_99,
            "expected_shortfall": expected_shortfall,
            "portfolio_metrics": portfolio_metrics,
            "exposures": self._calculate_exposures(positions),
            "compliance_issues": all_issues,
            "alerts": alerts,
            "memory_context": memory_context,
            "latent_messages": latent_messages,
            "done": True,
        }

    async def act(self, thought: Dict[str, Any]) -> AgentAction:
        """
        Generate risk alerts and position adjustment recommendations.

        Args:
            thought: Risk assessment results from think()

        Returns:
            AgentAction with alerts, recommended actions, and justification
        """
        start_time = time.perf_counter()

        try:
            risk_level = thought["risk_level"]
            alerts = thought["alerts"]
            compliance_issues = thought["compliance_issues"]

            recommendations = []
            for issue in compliance_issues:
                issue_lower = issue.lower()
                if "position_size" in issue_lower or "exceeds size" in issue_lower:
                    recommendations.append({
                        "action": "reduce_position",
                        "reason": issue,
                        "urgency": "high",
                    })
                elif "correlation" in issue_lower:
                    recommendations.append({
                        "action": "diversify",
                        "reason": issue,
                        "urgency": "medium",
                    })
                elif "drawdown" in issue_lower:
                    recommendations.append({
                        "action": "reduce_exposure",
                        "reason": issue,
                        "urgency": "critical",
                    })

            if alerts and self.communication:
                for alert in alerts:
                    await self.send_message(
                        recipient="*",
                        content={
                            "alert": alert,
                            "risk_level": risk_level,
                            "recommendations": recommendations,
                        },
                        message_type="risk_alert",
                        priority=2 if risk_level == "high" else 1,
                    )

            duration = time.perf_counter() - start_time

            result = {
                "risk_level": risk_level,
                "alerts": alerts,
                "recommendations": recommendations,
                "metrics": thought.get("portfolio_metrics", {}),
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
                        content={"assessment": result, "thought": thought},
                        metadata={
                            "agent": self.name,
                            "role": self.role,
                            "success": True,
                        }
                    )
                except Exception as e:
                    self.logger.warning("Memory store failed: %s", e)

            return AgentAction(
                action_type="risk_assessment",
                parameters={"risk_level": risk_level},
                result=result,
                success=True,
                error=None,
                duration=duration,
            )

        except Exception as e:
            self.logger.error("Error in risk action: %s", str(e))
            duration = time.perf_counter() - start_time

            # Store failure to memory
            if self.memory:
                try:
                    await self.memory.add(
                        content={"assessment": None, "thought": thought},
                        metadata={
                            "agent": self.name,
                            "role": self.role,
                            "success": False,
                        }
                    )
                except Exception as mem_e:
                    self.logger.warning("Memory store failed: %s", mem_e)

            return AgentAction(
                action_type="risk_assessment",
                parameters={},
                result=None,
                success=False,
                error=str(e),
                duration=duration,
            )

    async def _calculate_var(
        self,
        returns: Dict[str, List[float]],
        confidence: float = 0.95,
        method: str = "historical",
    ) -> float:
        """
        Calculate Value at Risk at given confidence level.

        Args:
            returns: Dictionary mapping asset symbols to return series
            confidence: Confidence level (0.95 = 95%)
            method: VaR calculation method (historical, parametric)

        Returns:
            VaR as a positive number (loss magnitude)
        """
        if not returns:
            return 0.0

        # Aggregate portfolio returns (equal-weighted)
        all_returns = list(returns.values())
        if not all_returns:
            return 0.0

        if len(all_returns) == 1:
            return_series = all_returns[0]
        else:
            min_length = min(len(r) for r in all_returns)
            return_series = [
                sum(r[i] for r in all_returns) / len(all_returns)
                for i in range(min_length)
            ]

        if not return_series:
            return 0.0

        sorted_returns = sorted(return_series)

        if method == "historical":
            index = int(len(sorted_returns) * (1 - confidence))
            index = max(0, min(index, len(sorted_returns) - 1))
            var = -sorted_returns[index]
        elif method == "parametric":
            mean_r = sum(return_series) / len(return_series)
            std_r = math.sqrt(
                sum((r - mean_r) ** 2 for r in return_series) / len(return_series)
            )
            z_score = {0.90: 1.28, 0.95: 1.645, 0.99: 2.326}.get(confidence, 1.645)
            var = abs(mean_r - z_score * std_r)
        else:
            index = int(len(sorted_returns) * (1 - confidence))
            var = -sorted_returns[max(0, index)]

        return var

    async def _calculate_expected_shortfall(
        self,
        returns: Dict[str, List[float]],
        confidence: float = 0.95,
    ) -> float:
        """
        Calculate Expected Shortfall (CVaR) — average loss beyond VaR.

        Args:
            returns: Dictionary mapping asset symbols to return series
            confidence: Confidence level

        Returns:
            Expected Shortfall as a positive number
        """
        if not returns:
            return 0.0

        all_returns = list(returns.values())
        if len(all_returns) == 1:
            return_series = all_returns[0]
        else:
            min_length = min(len(r) for r in all_returns)
            return_series = [
                sum(r[i] for r in all_returns) / len(all_returns)
                for i in range(min_length)
            ]

        if not return_series:
            return 0.0

        sorted_returns = sorted(return_series)
        cutoff_index = int(len(sorted_returns) * (1 - confidence))
        cutoff_index = max(1, cutoff_index)

        tail_losses = sorted_returns[:cutoff_index]
        if tail_losses:
            return abs(sum(tail_losses) / len(tail_losses))
        return 0.0

    async def _check_position_limits(
        self,
        positions: List[Dict[str, Any]],
        limits: RiskLimits,
    ) -> List[str]:
        """
        Check if positions violate size, concentration, or exposure limits.

        Args:
            positions: List of position dictionaries
            limits: Risk limits configuration

        Returns:
            List of compliance issue descriptions
        """
        issues = []
        if not positions:
            return issues

        total_value = sum(p.get("value", 0) for p in positions)
        if total_value == 0:
            return issues

        for position in positions:
            symbol = position.get("symbol", "UNKNOWN")
            position_value = position.get("value", 0)
            position_pct = position_value / total_value
            if position_pct > limits.max_position_pct:
                issues.append(
                    f"Position {symbol} exceeds size limit: "
                    f"{position_pct:.1%} > {limits.max_position_pct:.1%}"
                )

        sector_exposure: Dict[str, float] = {}
        for position in positions:
            sector = position.get("sector", "Unknown")
            value = position.get("value", 0)
            sector_exposure[sector] = sector_exposure.get(sector, 0) + value

        for sector, exposure in sector_exposure.items():
            sector_pct = exposure / total_value
            if sector_pct > limits.max_sector_pct:
                issues.append(
                    f"Sector {sector} exceeds concentration limit: "
                    f"{sector_pct:.1%} > {limits.max_sector_pct:.1%}"
                )

        return issues

    async def _calculate_portfolio_metrics(
        self,
        positions: List[Dict[str, Any]],
        returns: Dict[str, List[float]],
    ) -> Dict[str, float]:
        """
        Calculate portfolio risk metrics (Sharpe ratio, volatility, max drawdown).

        Args:
            positions: List of position dictionaries
            returns: Dictionary mapping asset symbols to return series

        Returns:
            Dictionary of portfolio metrics
        """
        metrics: Dict[str, float] = {}

        if not returns:
            return metrics

        all_returns = list(returns.values())
        if len(all_returns) == 1:
            portfolio_returns = all_returns[0]
        else:
            min_length = min(len(r) for r in all_returns)
            portfolio_returns = [
                sum(r[i] for r in all_returns) / len(all_returns)
                for i in range(min_length)
            ]

        if not portfolio_returns:
            return metrics

        mean_return = sum(portfolio_returns) / len(portfolio_returns)
        daily_vol = math.sqrt(
            sum((r - mean_return) ** 2 for r in portfolio_returns) / len(portfolio_returns)
        )

        metrics["volatility"] = daily_vol * math.sqrt(252)

        risk_free_rate = (self.config or {}).get("risk_free_rate", 0.02) / 252
        if daily_vol > 0:
            metrics["sharpe_ratio"] = (
                (mean_return - risk_free_rate) / daily_vol
            ) * math.sqrt(252)
        else:
            metrics["sharpe_ratio"] = 0.0

        # Max drawdown
        cumulative = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in portfolio_returns:
            cumulative *= (1 + r)
            if cumulative > peak:
                peak = cumulative
            drawdown = (peak - cumulative) / peak
            if drawdown > max_dd:
                max_dd = drawdown

        metrics["max_drawdown"] = max_dd

        return metrics

    def _calculate_correlation_matrix(
        self, returns: Dict[str, List[float]]
    ) -> Optional[List[List[float]]]:
        """Calculate correlation matrix for portfolio positions."""
        if not returns or len(returns) < 2:
            return None

        symbols = list(returns.keys())
        min_length = min(len(returns[s]) for s in symbols)
        if min_length < 2:
            return None

        n = len(symbols)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    ri = returns[symbols[i]][:min_length]
                    rj = returns[symbols[j]][:min_length]
                    mean_i = sum(ri) / len(ri)
                    mean_j = sum(rj) / len(rj)
                    cov = sum((ri[k] - mean_i) * (rj[k] - mean_j) for k in range(min_length)) / min_length
                    std_i = math.sqrt(sum((r - mean_i) ** 2 for r in ri) / len(ri))
                    std_j = math.sqrt(sum((r - mean_j) ** 2 for r in rj) / len(rj))
                    if std_i > 0 and std_j > 0:
                        matrix[i][j] = cov / (std_i * std_j)

        return matrix

    async def _check_correlation_risk(
        self,
        positions: List[Dict[str, Any]],
        correlation_matrix: List[List[float]],
    ) -> List[str]:
        """Check for concentration risk via high correlations."""
        issues = []
        n = len(correlation_matrix)
        symbols = [p.get("symbol", f"pos_{i}") for i, p in enumerate(positions[:n])]

        for i in range(n):
            for j in range(i + 1, n):
                corr = correlation_matrix[i][j]
                if abs(corr) > self.risk_limits.max_correlation:
                    issues.append(
                        f"High correlation between {symbols[i]} and {symbols[j]}: "
                        f"{corr:.2f} > {self.risk_limits.max_correlation:.2f}"
                    )

        return issues

    async def _generate_risk_alerts(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate risk alerts based on metrics and thresholds."""
        alerts = []

        var_95 = metrics.get("var_95", 0)
        if var_95 > self.risk_limits.max_var_pct:
            alerts.append(
                f"VaR(95%) exceeds limit: {var_95:.2%} > {self.risk_limits.max_var_pct:.2%}"
            )

        portfolio_metrics = metrics.get("portfolio_metrics", {})
        max_drawdown = portfolio_metrics.get("max_drawdown", 0)
        if max_drawdown > self.risk_limits.max_drawdown:
            alerts.append(
                f"Maximum drawdown exceeds limit: "
                f"{max_drawdown:.2%} > {self.risk_limits.max_drawdown:.2%}"
            )

        compliance_issues = metrics.get("compliance_issues", [])
        if compliance_issues:
            alerts.extend([f"Compliance: {issue}" for issue in compliance_issues[:3]])

        volatility = portfolio_metrics.get("volatility", 0)
        if volatility > 0.40:
            alerts.append(f"High portfolio volatility: {volatility:.1%}")

        return alerts

    def _calculate_exposures(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate exposures by sector and asset class."""
        exposures: Dict[str, Any] = {
            "by_sector": {},
            "by_asset_class": {},
            "total_long": 0.0,
            "total_short": 0.0,
            "net_exposure": 0.0,
        }

        total_value = sum(p.get("value", 0) for p in positions)
        if total_value == 0:
            return exposures

        for position in positions:
            value = position.get("value", 0)
            sector = position.get("sector", "Unknown")
            exposures["by_sector"][sector] = exposures["by_sector"].get(sector, 0) + value

            asset_class = position.get("asset_class", "Unknown")
            exposures["by_asset_class"][asset_class] = (
                exposures["by_asset_class"].get(asset_class, 0) + value
            )

            if value > 0:
                exposures["total_long"] += value
            else:
                exposures["total_short"] += abs(value)

        exposures["net_exposure"] = exposures["total_long"] - exposures["total_short"]

        for sector in exposures["by_sector"]:
            exposures["by_sector"][sector] /= total_value
        for asset_class in exposures["by_asset_class"]:
            exposures["by_asset_class"][asset_class] /= total_value

        return exposures

    def _determine_risk_level(
        self, var: float, expected_shortfall: float,
        metrics: Dict[str, float], issues: List[str],
    ) -> str:
        """Determine overall risk level based on metrics."""
        if len(issues) >= 3:
            return "critical"

        max_drawdown = metrics.get("max_drawdown", 0)
        if var > self.risk_limits.max_var_pct or max_drawdown > self.risk_limits.max_drawdown:
            return "high"

        if issues or var > self.risk_limits.max_var_pct * 0.7:
            return "medium"

        return "low"
