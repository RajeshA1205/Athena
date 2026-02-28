# TASK-010: Implement AgentEvolver Workflow Discovery

## Status
- **State:** Queued
- **Priority:** 🟢 Medium
- **Depends on:** None (Sprint 1 complete)
- **Created:** 2026-02-15

## Objective
Create the workflow discovery component that extracts successful workflow patterns from agent interaction histories for the AgentEvolver evolution layer.

## Context
Workflow discovery is a key component of AgentEvolver. It analyzes historical agent interactions and execution traces to identify successful patterns that can be reused or evolved into new agent configurations.

This component provides:
- Analysis of agent interaction histories
- Pattern extraction from successful executions
- Workflow library management
- Pattern similarity and clustering

Reference the AgentEvolver paper in `/Users/rajesh/athena/architecture/base/` and the architecture plan at `/Users/rajesh/athena/agent-plan/atomic-wibbling-simon.md` (lines 78-82, 171-188, 313).

## Scope & Constraints

### Files to Create
- `/Users/rajesh/athena/evolution/workflow_discovery.py`
- `/Users/rajesh/athena/evolution/__init__.py`

### Files to Reference (DO NOT MODIFY)
- `/Users/rajesh/athena/memory/agemem.py` — Access agent histories
- `/Users/rajesh/athena/core/config.py` — Configuration
- Research paper: `/Users/rajesh/athena/architecture/base/AgentEvolver.pdf`

### Constraints
- Use async/await for all operations
- Store workflows in structured format (JSON or similar)
- No actual agent generation yet (that's TASK-011)
- Focus on pattern extraction, not execution

## Input
- AgentEvolver paper specification
- AgeMem interface for accessing histories
- Project configuration system

## Expected Output

### File: `/Users/rajesh/athena/evolution/workflow_discovery.py`
```python
from typing import Dict, Any, List, Optional, Tuple
import asyncio
import json
from datetime import datetime
from collections import defaultdict

class WorkflowPattern:
    """
    Represents a discovered workflow pattern.

    Attributes:
        pattern_id: Unique identifier
        agent_sequence: Ordered list of agent types in workflow
        interaction_pattern: Communication pattern between agents
        success_rate: Historical success rate
        use_count: Number of times pattern was observed
        metadata: Additional pattern metadata
    """

    def __init__(self, pattern_id: str, agent_sequence: List[str], interaction_pattern: Dict[str, Any]):
        self.pattern_id = pattern_id
        self.agent_sequence = agent_sequence
        self.interaction_pattern = interaction_pattern
        self.success_rate = 0.0
        self.use_count = 0
        self.metadata: Dict[str, Any] = {}
        self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize pattern to dict."""
        return {
            'pattern_id': self.pattern_id,
            'agent_sequence': self.agent_sequence,
            'interaction_pattern': self.interaction_pattern,
            'success_rate': self.success_rate,
            'use_count': self.use_count,
            'metadata': self.metadata,
            'created_at': self.created_at
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'WorkflowPattern':
        """Deserialize pattern from dict."""
        pattern = WorkflowPattern(
            data['pattern_id'],
            data['agent_sequence'],
            data['interaction_pattern']
        )
        pattern.success_rate = data.get('success_rate', 0.0)
        pattern.use_count = data.get('use_count', 0)
        pattern.metadata = data.get('metadata', {})
        pattern.created_at = data.get('created_at', datetime.now().isoformat())
        return pattern


class WorkflowDiscovery:
    """
    Discover and manage successful workflow patterns from agent histories.

    Based on: AgentEvolver paper
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize workflow discovery.

        Args:
            config: Configuration dict containing:
                - min_success_rate: Minimum success rate to keep pattern (default: 0.7)
                - min_use_count: Minimum occurrences to consider pattern (default: 3)
                - pattern_similarity_threshold: Threshold for pattern merging (default: 0.8)
        """
        self.min_success_rate = config.get('min_success_rate', 0.7)
        self.min_use_count = config.get('min_use_count', 3)
        self.similarity_threshold = config.get('pattern_similarity_threshold', 0.8)

        self.workflow_library: Dict[str, WorkflowPattern] = {}
        self.execution_history: List[Dict[str, Any]] = []

    async def analyze_execution(self, execution_trace: Dict[str, Any]) -> Optional[str]:
        """
        Analyze a single execution trace and extract workflow pattern.

        Args:
            execution_trace: Dict containing:
                - agent_sequence: List of agents involved
                - interactions: Agent communication log
                - outcome: Success/failure
                - metrics: Performance metrics

        Returns:
            Pattern ID if pattern extracted, None otherwise
        """
        # Store execution in history
        self.execution_history.append(execution_trace)

        # Extract agent sequence
        agent_sequence = execution_trace.get('agent_sequence', [])
        if len(agent_sequence) < 2:
            return None

        # Extract interaction pattern
        interactions = execution_trace.get('interactions', [])
        interaction_pattern = self._extract_interaction_pattern(interactions)

        # Generate pattern ID
        pattern_id = self._generate_pattern_id(agent_sequence, interaction_pattern)

        # Check if pattern exists
        if pattern_id in self.workflow_library:
            pattern = self.workflow_library[pattern_id]
            pattern.use_count += 1

            # Update success rate
            outcome = execution_trace.get('outcome', 'failure')
            if outcome == 'success':
                pattern.success_rate = (pattern.success_rate * (pattern.use_count - 1) + 1.0) / pattern.use_count
            else:
                pattern.success_rate = (pattern.success_rate * (pattern.use_count - 1)) / pattern.use_count
        else:
            # Create new pattern
            pattern = WorkflowPattern(pattern_id, agent_sequence, interaction_pattern)
            pattern.use_count = 1
            pattern.success_rate = 1.0 if execution_trace.get('outcome') == 'success' else 0.0
            pattern.metadata = execution_trace.get('metadata', {})
            self.workflow_library[pattern_id] = pattern

        return pattern_id

    def _extract_interaction_pattern(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract structured interaction pattern from interaction log.

        Args:
            interactions: List of interaction events

        Returns:
            Structured interaction pattern
        """
        pattern = {
            'communication_graph': defaultdict(list),
            'message_types': defaultdict(int),
            'timing': []
        }

        for interaction in interactions:
            sender = interaction.get('sender')
            receiver = interaction.get('receiver')
            msg_type = interaction.get('type', 'unknown')

            if sender and receiver:
                pattern['communication_graph'][sender].append(receiver)
                pattern['message_types'][msg_type] += 1

        # Convert defaultdicts to regular dicts for serialization
        pattern['communication_graph'] = dict(pattern['communication_graph'])
        pattern['message_types'] = dict(pattern['message_types'])

        return pattern

    def _generate_pattern_id(self, agent_sequence: List[str], interaction_pattern: Dict[str, Any]) -> str:
        """Generate unique ID for workflow pattern."""
        # Simple hash based on agent sequence
        seq_str = '->'.join(agent_sequence)
        return f"workflow_{hash(seq_str) % 100000}"

    async def get_successful_patterns(self) -> List[WorkflowPattern]:
        """
        Get all patterns meeting success criteria.

        Returns:
            List of successful workflow patterns
        """
        successful = []
        for pattern in self.workflow_library.values():
            if (pattern.success_rate >= self.min_success_rate and
                pattern.use_count >= self.min_use_count):
                successful.append(pattern)

        # Sort by success rate
        successful.sort(key=lambda p: p.success_rate, reverse=True)
        return successful

    async def find_similar_patterns(self, reference_pattern: WorkflowPattern) -> List[Tuple[WorkflowPattern, float]]:
        """
        Find patterns similar to a reference pattern.

        Args:
            reference_pattern: Pattern to compare against

        Returns:
            List of (pattern, similarity_score) tuples
        """
        similar = []
        for pattern in self.workflow_library.values():
            if pattern.pattern_id == reference_pattern.pattern_id:
                continue

            similarity = self._calculate_pattern_similarity(reference_pattern, pattern)
            if similarity >= self.similarity_threshold:
                similar.append((pattern, similarity))

        similar.sort(key=lambda x: x[1], reverse=True)
        return similar

    def _calculate_pattern_similarity(self, pattern1: WorkflowPattern, pattern2: WorkflowPattern) -> float:
        """
        Calculate similarity between two patterns.

        Returns:
            Similarity score [0, 1]
        """
        # Simple Jaccard similarity on agent sequences
        set1 = set(pattern1.agent_sequence)
        set2 = set(pattern2.agent_sequence)

        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union

    async def save_library(self, path: str):
        """Save workflow library to file."""
        data = {
            'patterns': [p.to_dict() for p in self.workflow_library.values()],
            'config': {
                'min_success_rate': self.min_success_rate,
                'min_use_count': self.min_use_count,
                'similarity_threshold': self.similarity_threshold
            }
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    async def load_library(self, path: str):
        """Load workflow library from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.workflow_library = {}
        for pattern_data in data.get('patterns', []):
            pattern = WorkflowPattern.from_dict(pattern_data)
            self.workflow_library[pattern.pattern_id] = pattern

    def get_stats(self) -> Dict[str, Any]:
        """Get workflow discovery statistics."""
        return {
            'total_patterns': len(self.workflow_library),
            'successful_patterns': len([p for p in self.workflow_library.values()
                                        if p.success_rate >= self.min_success_rate]),
            'executions_analyzed': len(self.execution_history),
            'avg_success_rate': sum(p.success_rate for p in self.workflow_library.values()) / max(len(self.workflow_library), 1)
        }
```

### File: `/Users/rajesh/athena/evolution/__init__.py`
```python
from .workflow_discovery import WorkflowDiscovery, WorkflowPattern

__all__ = ['WorkflowDiscovery', 'WorkflowPattern']
```

## Acceptance Criteria
- [ ] WorkflowPattern class created with serialization methods
- [ ] WorkflowDiscovery class created
- [ ] `analyze_execution()` method extracts patterns from execution traces
- [ ] `get_successful_patterns()` returns patterns meeting success criteria
- [ ] `find_similar_patterns()` implements pattern similarity search
- [ ] Workflow library management (save/load)
- [ ] Pattern tracking with success rate and use count
- [ ] All methods use async/await pattern
- [ ] `get_stats()` for monitoring
- [ ] Classes are importable and instantiable
- [ ] Docstrings present for all public methods

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
