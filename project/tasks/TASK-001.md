# TASK-001: Implement Market Analyst Agent

## Status
- **State:** Queued
- **Priority:** 🟡 High
- **Depends on:** None (Sprint 1 complete)
- **Created:** 2026-02-15

## Objective
Create the Market Analyst Agent that performs real-time market analysis, pattern recognition, technical indicators calculation, and sentiment analysis.

## Context
The Market Analyst is one of 5 specialized agents in ATHENA. It inherits from BaseAgent (`/Users/rajesh/athena/core/base_agent.py`) and implements the agent's decision-making logic for analyzing market data.

This agent will:
- Analyze market data (prices, volume, volatility)
- Recognize chart patterns and trends
- Calculate technical indicators (MA, RSI, MACD, Bollinger Bands, etc.)
- Perform sentiment analysis on news/social data
- Communicate findings to other agents via LatentMAS

Reference the architecture plan at `/Users/rajesh/athena/agent-plan/atomic-wibbling-simon.md` (lines 64-66, 311).

## Scope & Constraints

### Files to Create
- `/Users/rajesh/athena/agents/market_analyst.py`
- `/Users/rajesh/athena/agents/__init__.py` (if not exists)

### Files to Reference (DO NOT MODIFY)
- `/Users/rajesh/athena/core/base_agent.py` — Inherit from BaseAgent
- `/Users/rajesh/athena/core/config.py` — Use for configuration
- `/Users/rajesh/athena/models/olmoe.py` — OLMoE model interface
- `/Users/rajesh/athena/models/embeddings.py` — Embedding models

### Constraints
- Must inherit from BaseAgent and implement `think()` and `act()` methods
- Use **async/await** for all operations
- Follow existing code style in core/ modules
- No external API calls yet (mock data is acceptable for now)
- Agent should be instantiable and callable, integration comes later

## Input
- BaseAgent abstract class definition
- OLMoE model interface
- Project configuration system
- Research papers in `/Users/rajesh/athena/architecture/base/`

## Expected Output

### File: `/Users/rajesh/athena/agents/market_analyst.py`
```python
from core.base_agent import BaseAgent
from typing import Dict, Any, List
import asyncio

class MarketAnalystAgent(BaseAgent):
    """
    Market Analyst Agent for ATHENA multi-agent system.

    Responsibilities:
    - Real-time market data analysis
    - Pattern recognition (head-and-shoulders, double-top, etc.)
    - Technical indicator calculation
    - Sentiment analysis from news/social media
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(agent_id="market_analyst", config=config)
        # Initialize agent-specific components

    async def think(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data and generate insights.

        Args:
            observation: Dict containing market_data, news, social_sentiment

        Returns:
            Dict with analysis results, patterns detected, indicators, sentiment
        """
        # Implement market analysis logic
        pass

    async def act(self, thought: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formulate actionable recommendations based on analysis.

        Args:
            thought: Analysis results from think()

        Returns:
            Dict with recommendations, confidence scores, supporting evidence
        """
        # Implement action logic
        pass

    async def _calculate_technical_indicators(self, price_data: List[float]) -> Dict[str, float]:
        """Calculate technical indicators (MA, RSI, MACD, etc.)"""
        pass

    async def _detect_patterns(self, price_data: List[float]) -> List[str]:
        """Detect chart patterns"""
        pass

    async def _analyze_sentiment(self, text_data: List[str]) -> float:
        """Analyze sentiment from news/social media"""
        pass
```

### File: `/Users/rajesh/athena/agents/__init__.py`
```python
from .market_analyst import MarketAnalystAgent

__all__ = ['MarketAnalystAgent']
```

## Acceptance Criteria
- [ ] MarketAnalystAgent class created and inherits from BaseAgent
- [ ] `think()` method implemented with market analysis logic
- [ ] `act()` method implemented with recommendation formulation
- [ ] Helper methods for technical indicators, pattern detection, sentiment analysis
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
