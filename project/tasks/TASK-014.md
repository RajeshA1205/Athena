# TASK-014: Implement RepExp Exploration Module

## Status
- **State:** Queued
- **Priority:** 🟢 Medium
- **Depends on:** None (Sprint 1 complete)
- **Created:** 2026-02-15

## Objective
Create the RepExp (Representation-based Exploration) module that implements representation-space exploration with diversity bonuses for improved agent learning.

## Context
RepExp is the second component of the Learning Layer. It encourages agents to explore diverse strategies by measuring novelty in representation space and providing exploration bonuses, leading to better generalization and capability discovery.

This component provides:
- Representation-space diversity measurement
- Exploration bonus computation
- Test-time exploration strategies
- Post-training capability expansion

Reference the RepExp paper in `/Users/rajesh/athena/architecture/base/` and the architecture plan at `/Users/rajesh/athena/agent-plan/atomic-wibbling-simon.md` (lines 83-87, 195-202, 314).

## Scope & Constraints

### Files to Create
- `/Users/rajesh/athena/learning/repexp.py`

### Files to Modify
- `/Users/rajesh/athena/learning/__init__.py` — Add RepExp import

### Files to Reference (DO NOT MODIFY)
- `/Users/rajesh/athena/models/embeddings.py` — Embedding models
- `/Users/rajesh/athena/core/config.py` — Configuration
- Research paper: `/Users/rajesh/athena/architecture/base/RepExp.pdf`

### Constraints
- Use PyTorch for representation computations
- Use async/await for all operations
- Implement basic novelty detection (advanced methods in Sprint 4)
- Focus on exploration bonus calculation, not full RL integration

## Input
- RepExp paper specification
- Embedding models interface
- PyTorch framework
- Project configuration system

## Expected Output

File: `/Users/rajesh/athena/learning/repexp.py` with RepExp class implementing:
- compute_novelty() — Measures representation novelty using k-NN
- compute_exploration_bonus() — Calculates exploration reward bonus
- compute_diversity() — Measures diversity within representation set
- select_diverse_subset() — Selects diverse representations
- get_exploration_strategy() — Suggests explore/exploit strategy
- Representation buffer management
- Save/load functionality

Update `/Users/rajesh/athena/learning/__init__.py` to add RepExp import.

## Acceptance Criteria
- [ ] RepresentationBuffer class created for representation storage
- [ ] RepExp class created with novelty computation
- [ ] compute_novelty() measures representation novelty using k-NN
- [ ] compute_exploration_bonus() calculates exploration reward bonus
- [ ] compute_diversity() measures diversity within representation set
- [ ] select_diverse_subset() selects diverse representations
- [ ] get_exploration_strategy() suggests explore/exploit strategy
- [ ] Representation buffer management
- [ ] Save/load functionality
- [ ] All async methods use async/await pattern
- [ ] get_stats() for monitoring
- [ ] Classes are importable and instantiable
- [ ] Docstrings present for all public methods

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
