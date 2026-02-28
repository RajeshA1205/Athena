# TASK-016: Integrate Agents with LatentMAS Communication

## Status
- **State:** Queued
- **Priority:** 🔴 Critical
- **Depends on:** TASK-005, TASK-009
- **Created:** 2026-02-15

## Objective
Enable inter-agent communication via LatentMAS latent space messaging for all five agents.

## Context
Second major integration task in Sprint 3. Agents need to communicate via latent space rather than direct message passing, enabling emergent collaboration patterns.

Key integrations:
- Agents encode outputs via LatentEncoder
- Agents send messages via LatentSpace
- Agents receive and decode messages via LatentDecoder
- MessageRouter handles priority and routing
- Coordinator orchestrates communication patterns

## Scope & Constraints

### Files to Modify
- `/Users/rajesh/athena/agents/market_analyst.py`
- `/Users/rajesh/athena/agents/risk_manager.py`
- `/Users/rajesh/athena/agents/strategy_agent.py`
- `/Users/rajesh/athena/agents/execution_agent.py`
- `/Users/rajesh/athena/agents/coordinator.py`

### Files to Reference (DO NOT MODIFY)
- `/Users/rajesh/athena/communication/latent_space.py`
- `/Users/rajesh/athena/communication/encoder.py`
- `/Users/rajesh/athena/communication/decoder.py`
- `/Users/rajesh/athena/communication/router.py`

### Constraints
- All communication must go through LatentMAS
- Use async messaging
- Coordinator initializes MessageRouter
- Do not modify communication layer code

## Input
- Agent implementations from TASK-001 through TASK-005
- LatentMAS components from TASK-006 through TASK-009

## Expected Output
- Coordinator initializes LatentSpace, Encoder, Decoder, Router in __init__()
- Agents register with MessageRouter
- Agents encode outputs before sending
- Agents decode received messages
- Integration test: Multi-agent communication workflow works

## Acceptance Criteria
- [ ] All agents integrated with LatentMAS
- [ ] Coordinator initializes communication infrastructure
- [ ] Agents send/receive latent messages
- [ ] MessageRouter handles priority routing
- [ ] Integration test: Agent-to-agent communication via latent space works
- [ ] Documentation updated with communication patterns

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
