# TASK-015: Integrate Agents with AgeMem Memory Layer

## Status
- **State:** Queued
- **Priority:** 🔴 Critical
- **Depends on:** TASK-005
- **Created:** 2026-02-15

## Objective
Connect all five agents (Market Analyst, Risk Manager, Strategy, Execution, Coordinator) to the AgeMem memory layer for persistent context and learning.

## Context
This is the first major integration task in Sprint 3. Agents need to read from and write to AgeMem for episodic memory, long-term knowledge storage, and context retrieval.

Key integrations:
- Agents call AgeMem operations (ADD, RETRIEVE, SUMMARY, FILTER)
- Memory operations are async
- Graphiti backend handles persistence
- Coordinator manages memory allocation across agents

## Scope & Constraints

### Files to Modify
- `/Users/rajesh/athena/agents/market_analyst.py`
- `/Users/rajesh/athena/agents/risk_manager.py`
- `/Users/rajesh/athena/agents/strategy_agent.py`
- `/Users/rajesh/athena/agents/execution_agent.py`
- `/Users/rajesh/athena/agents/coordinator.py`

### Files to Reference (DO NOT MODIFY)
- `/Users/rajesh/athena/memory/agemem.py`
- `/Users/rajesh/athena/memory/operations.py`
- `/Users/rajesh/athena/memory/graphiti_backend.py`

### Constraints
- All memory operations must be async
- Use context.md conventions for memory usage
- Do not modify memory layer code
- Handle memory operation failures gracefully

## Input
- Agent implementations from TASK-001 through TASK-005
- AgeMem implementation from Sprint 1
- Memory operation interfaces

## Expected Output
Each agent class should:
- Initialize AgeMem connection in __init__()
- Use RETRIEVE for context in think() method
- Use ADD to store decisions/observations in act() method
- Use SUMMARY for context compression when needed
- Handle async memory operations properly

## Acceptance Criteria
- [ ] All 5 agents integrated with AgeMem
- [ ] Agents retrieve context before decision-making
- [ ] Agents store decisions and outcomes to memory
- [ ] Coordinator manages memory resource allocation
- [ ] All memory operations use async/await
- [ ] Error handling for memory operation failures
- [ ] Integration test: Agent workflow with memory persistence works end-to-end
- [ ] Documentation updated with memory usage patterns

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
