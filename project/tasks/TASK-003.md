# TASK-003: Implement Strategy Agent

## Status
- **State:** Queued
- **Priority:** 🟡 High
- **Depends on:** None (Sprint 1 complete)
- **Created:** 2026-02-15

## Objective
Create the Strategy Agent that formulates trading strategies, performs backtesting logic, and optimizes portfolio allocation.

## Context
The Strategy Agent is one of 5 specialized agents in ATHENA. It synthesizes insights from the Market Analyst and Risk Manager to formulate actionable trading strategies.

This agent will:
- Formulate trading strategies based on market analysis
- Implement backtesting logic for strategy validation
- Optimize portfolio allocation (position sizing, diversification)
- Generate trade recommendations with entry/exit criteria
- Communicate strategies to Execution Agent via LatentMAS

Reference the architecture plan at `/Users/rajesh/athena/agent-plan/atomic-wibbling-simon.md` (lines 64-66, 311).

## Scope & Constraints

### Files to Create
- `/Users/rajesh/athena/agents/strategy_agent.py`

### Files to Modify
- `/Users/rajesh/athena/agents/__init__.py` — Add StrategyAgent import

### Files to Reference (DO NOT MODIFY)
- `/Users/rajesh/athena/core/base_agent.py` — Inherit from BaseAgent
- `/Users/rajesh/athena/core/config.py` — Use for configuration
- `/Users/rajesh/athena/models/olmoe.py` — OLMoE model interface

### Constraints
- Must inherit from BaseAgent and implement `think()` and `act()` methods
- Use **async/await** for all operations
- Follow existing code style in core/ modules
- Strategy logic should be modular (support multiple strategy types)
- No actual backtesting execution yet (interface only)

## Input
- BaseAgent abstract class definition
- OLMoE model interface
- Project configuration system
- Common trading strategy patterns (momentum, mean reversion, arbitrage)

## Expected Output

### File: `/Users/rajesh/athena/agents/strategy_agent.py`
```python
from core.base_agent import BaseAgent
from typing import Dict, Any, List, Optional
from enum import Enum
import asyncio

class StrategyType(Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    PAIRS_TRADING = "pairs_trading"

class StrategyAgent(BaseAgent):
    """
    Strategy Agent for ATHENA multi-agent system.

    Responsibilities:
    - Formulate trading strategies based on market analysis
    - Backtest strategies on historical data
    - Optimize portfolio allocation and position sizing
    - Generate trade recommendations with entry/exit criteria
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(agent_id="strategy_agent", config=config)
        # Initialize strategy library, optimization parameters

    async def think(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market conditions and formulate trading strategy.

        Args:
            observation: Dict containing market_analysis, risk_metrics, portfolio_state

        Returns:
            Dict with strategy_type, parameters, expected_performance, rationale
        """
        # Implement strategy formulation logic
        pass

    async def act(self, thought: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate concrete trade recommendations.

        Args:
            thought: Strategy formulation from think()

        Returns:
            Dict with trade_recommendations, position_sizes, entry/exit criteria
        """
        # Implement trade recommendation logic
        pass

    async def _select_strategy(self, market_regime: str) -> StrategyType:
        """Select optimal strategy based on market regime"""
        pass

    async def _optimize_portfolio(self, assets: List[str], returns: List[float]) -> Dict[str, float]:
        """Optimize portfolio weights (mean-variance, risk-parity, etc.)"""
        pass

    async def _calculate_position_size(self, signal_strength: float, risk_budget: float) -> float:
        """Calculate position size using Kelly criterion or similar"""
        pass

    async def _backtest_strategy(self, strategy: Dict[str, Any], historical_data: List[Dict]) -> Dict[str, float]:
        """Backtest strategy on historical data, return performance metrics"""
        pass
```

### Update: `/Users/rajesh/athena/agents/__init__.py`
Add StrategyAgent to imports and __all__.

## Acceptance Criteria
- [ ] StrategyAgent class created and inherits from BaseAgent
- [ ] `think()` method implemented with strategy formulation logic
- [ ] `act()` method implemented with trade recommendation generation
- [ ] Helper methods for strategy selection, portfolio optimization, position sizing, backtesting interface
- [ ] Support for multiple strategy types (enum or similar)
- [ ] All methods use async/await pattern
- [ ] Class is importable and instantiable
- [ ] Docstrings present for all public methods
- [ ] Code follows existing style conventions from core/ modules

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
