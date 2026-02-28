# TASK-005: Implement Coordinator Agent

## Status
- **State:** Queued
- **Priority:** 🔴 Critical
- **Depends on:** TASK-001, TASK-002, TASK-003, TASK-004
- **Created:** 2026-02-15

## Objective
Create the Coordinator Agent that orchestrates all other agents, resolves conflicts, allocates resources, and manages the overall decision-making workflow.

## Context
The Coordinator is the central orchestrator in ATHENA. It manages the lifecycle of all other agents (Market Analyst, Risk Manager, Strategy, Execution), routes messages between them, resolves conflicts when agents disagree, and ensures cohesive system behavior.

This agent will:
- Orchestrate agent workflows (data flow, decision sequencing)
- Resolve conflicts when agents have contradictory recommendations
- Allocate resources (compute, memory, API calls) across agents
- Maintain system state and coordinate inter-agent communication
- Make final decisions on trading actions

Reference the architecture plan at `/Users/rajesh/athena/agent-plan/atomic-wibbling-simon.md` (lines 64-66, 311).

## Scope & Constraints

### Files to Create
- `/Users/rajesh/athena/agents/coordinator.py`

### Files to Modify
- `/Users/rajesh/athena/agents/__init__.py` — Add CoordinatorAgent import

### Files to Reference (DO NOT MODIFY)
- `/Users/rajesh/athena/core/base_agent.py` — Inherit from BaseAgent
- `/Users/rajesh/athena/agents/market_analyst.py` — MarketAnalystAgent
- `/Users/rajesh/athena/agents/risk_manager.py` — RiskManagerAgent
- `/Users/rajesh/athena/agents/strategy_agent.py` — StrategyAgent
- `/Users/rajesh/athena/agents/execution_agent.py` — ExecutionAgent

### Constraints
- Must inherit from BaseAgent and implement `think()` and `act()` methods
- Use **async/await** for all operations
- Manage lifecycle of all 4 specialized agents
- Implement conflict resolution logic
- Follow existing code style in core/ modules
- Full integration will happen in Sprint 3, but basic orchestration should work

## Input
- All 4 specialized agent classes (TASK-001 through TASK-004)
- BaseAgent abstract class definition
- OLMoE model interface
- Project configuration system

## Expected Output

### File: `/Users/rajesh/athena/agents/coordinator.py`
```python
from core.base_agent import BaseAgent
from typing import Dict, Any, List, Optional
import asyncio
from .market_analyst import MarketAnalystAgent
from .risk_manager import RiskManagerAgent
from .strategy_agent import StrategyAgent
from .execution_agent import ExecutionAgent

class CoordinatorAgent(BaseAgent):
    """
    Coordinator Agent for ATHENA multi-agent system.

    Responsibilities:
    - Orchestrate workflow across all agents
    - Resolve conflicts between agent recommendations
    - Allocate resources (compute, memory, API calls)
    - Make final decisions on trading actions
    - Maintain system state and coordination
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(agent_id="coordinator", config=config)

        # Initialize specialized agents
        self.market_analyst = MarketAnalystAgent(config.get('market_analyst', {}))
        self.risk_manager = RiskManagerAgent(config.get('risk_manager', {}))
        self.strategy_agent = StrategyAgent(config.get('strategy_agent', {}))
        self.execution_agent = ExecutionAgent(config.get('execution_agent', {}))

        # Track system state
        self.agents = {
            'market_analyst': self.market_analyst,
            'risk_manager': self.risk_manager,
            'strategy_agent': self.strategy_agent,
            'execution_agent': self.execution_agent
        }

    async def think(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate agents to analyze situation and formulate plan.

        Args:
            observation: Dict containing market_data, portfolio_state, external_signals

        Returns:
            Dict with orchestration_plan, agent_assignments, conflict_resolution
        """
        # Implement orchestration logic
        pass

    async def act(self, thought: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute coordinated actions across all agents.

        Args:
            thought: Orchestration plan from think()

        Returns:
            Dict with final_decision, actions_taken, agent_outputs
        """
        # Implement coordinated action execution
        pass

    async def _run_pipeline(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run full agent pipeline: Analysis → Risk → Strategy → Execution
        """
        # Get market analysis
        market_analysis = await self.market_analyst.think(observation)
        market_action = await self.market_analyst.act(market_analysis)

        # Assess risk
        risk_obs = {**observation, 'market_analysis': market_action}
        risk_assessment = await self.risk_manager.think(risk_obs)
        risk_action = await self.risk_manager.act(risk_assessment)

        # Formulate strategy
        strategy_obs = {**observation, 'market_analysis': market_action, 'risk_metrics': risk_action}
        strategy_thought = await self.strategy_agent.think(strategy_obs)
        strategy_action = await self.strategy_agent.act(strategy_thought)

        # Execute
        execution_obs = {**observation, 'trade_request': strategy_action}
        execution_plan = await self.execution_agent.think(execution_obs)
        execution_result = await self.execution_agent.act(execution_plan)

        return {
            'market_analysis': market_action,
            'risk_assessment': risk_action,
            'strategy': strategy_action,
            'execution': execution_result
        }

    async def _resolve_conflicts(self, agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve conflicts when agents have contradictory recommendations.
        E.g., Strategy wants to buy but Risk Manager says exposure too high.
        """
        pass

    async def _allocate_resources(self, agent_requests: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Allocate limited resources (compute, memory, API calls) across agents.
        """
        pass

    async def _make_final_decision(self, agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make final trading decision based on all agent inputs.
        """
        pass
```

### Update: `/Users/rajesh/athena/agents/__init__.py`
Add CoordinatorAgent to imports and __all__.

## Acceptance Criteria
- [ ] CoordinatorAgent class created and inherits from BaseAgent
- [ ] Initializes and manages all 4 specialized agents
- [ ] `think()` method implemented with orchestration logic
- [ ] `act()` method implemented with coordinated action execution
- [ ] `_run_pipeline()` method executes full agent workflow
- [ ] Helper methods for conflict resolution, resource allocation, final decision
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
