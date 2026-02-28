# TASK-002: Implement Risk Manager Agent

## Status
- **State:** Queued
- **Priority:** 🟡 High
- **Depends on:** None (Sprint 1 complete)
- **Created:** 2026-02-15

## Objective
Create the Risk Manager Agent that performs portfolio risk assessment, exposure monitoring, VaR calculations, and compliance checks.

## Context
The Risk Manager is one of 5 specialized agents in ATHENA. It inherits from BaseAgent and implements risk management logic to ensure trading stays within acceptable risk parameters.

This agent will:
- Assess portfolio risk (volatility, correlation, concentration)
- Monitor exposure across positions
- Calculate Value at Risk (VaR) and Expected Shortfall
- Check compliance with risk limits and regulations
- Communicate risk alerts to other agents via LatentMAS

Reference the architecture plan at `/Users/rajesh/athena/agent-plan/atomic-wibbling-simon.md` (lines 64-66, 311).

## Scope & Constraints

### Files to Create
- `/Users/rajesh/athena/agents/risk_manager.py`

### Files to Modify
- `/Users/rajesh/athena/agents/__init__.py` — Add RiskManagerAgent import

### Files to Reference (DO NOT MODIFY)
- `/Users/rajesh/athena/core/base_agent.py` — Inherit from BaseAgent
- `/Users/rajesh/athena/core/config.py` — Use for configuration
- `/Users/rajesh/athena/models/olmoe.py` — OLMoE model interface

### Constraints
- Must inherit from BaseAgent and implement `think()` and `act()` methods
- Use **async/await** for all operations
- Follow existing code style in core/ modules
- No external dependencies yet (mock portfolio data acceptable)
- Agent should be standalone, integration comes later

## Input
- BaseAgent abstract class definition
- OLMoE model interface
- Project configuration system
- Standard risk management formulas (VaR, Sharpe ratio, etc.)

## Expected Output

### File: `/Users/rajesh/athena/agents/risk_manager.py`
```python
from core.base_agent import BaseAgent
from typing import Dict, Any, List
import asyncio

class RiskManagerAgent(BaseAgent):
    """
    Risk Manager Agent for ATHENA multi-agent system.

    Responsibilities:
    - Portfolio risk assessment (volatility, correlation, concentration)
    - Exposure monitoring across positions
    - Value at Risk (VaR) and Expected Shortfall calculations
    - Compliance checks with risk limits
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(agent_id="risk_manager", config=config)
        # Initialize risk limits, compliance rules

    async def think(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess portfolio risk and identify exposures.

        Args:
            observation: Dict containing portfolio, positions, market_data

        Returns:
            Dict with risk metrics, VaR, exposures, compliance status
        """
        # Implement risk assessment logic
        pass

    async def act(self, thought: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate risk alerts and position adjustment recommendations.

        Args:
            thought: Risk assessment results from think()

        Returns:
            Dict with alerts, recommended actions, justification
        """
        # Implement action logic
        pass

    async def _calculate_var(self, positions: List[Dict], confidence: float = 0.95) -> float:
        """Calculate Value at Risk at given confidence level"""
        pass

    async def _calculate_expected_shortfall(self, positions: List[Dict]) -> float:
        """Calculate Expected Shortfall (CVaR)"""
        pass

    async def _check_position_limits(self, positions: List[Dict]) -> List[str]:
        """Check if positions violate size/concentration limits"""
        pass

    async def _calculate_portfolio_metrics(self, positions: List[Dict]) -> Dict[str, float]:
        """Calculate Sharpe ratio, volatility, beta, etc."""
        pass
```

### Update: `/Users/rajesh/athena/agents/__init__.py`
Add RiskManagerAgent to imports and __all__.

## Acceptance Criteria
- [ ] RiskManagerAgent class created and inherits from BaseAgent
- [ ] `think()` method implemented with risk assessment logic
- [ ] `act()` method implemented with alert generation
- [ ] Helper methods for VaR, Expected Shortfall, position limits, portfolio metrics
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
