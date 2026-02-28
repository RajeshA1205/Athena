# TASK-012: Implement AgentEvolver Cooperative Evolution

## Status
- **State:** Queued
- **Priority:** ⚪ Low
- **Depends on:** TASK-010, TASK-011
- **Created:** 2026-02-15

## Objective
Create the cooperative evolution component that enables multi-agent experience replay and cooperative improvement through shared learning.

## Context
Cooperative evolution is the third component of AgentEvolver. It enables agents to learn from each other's experiences through experience replay and knowledge distillation, improving the entire agent population collaboratively.

This component provides:
- Multi-agent experience replay
- Cross-agent knowledge distillation
- Population-based performance tracking
- Collaborative improvement mechanisms

Reference the AgentEvolver paper in `/Users/rajesh/athena/architecture/base/` and the architecture plan at `/Users/rajesh/athena/agent-plan/atomic-wibbling-simon.md` (lines 78-82, 182-188).

## Scope & Constraints

### Files to Create
- `/Users/rajesh/athena/evolution/cooperative_evolution.py`

### Files to Modify
- `/Users/rajesh/athena/evolution/__init__.py` — Add CooperativeEvolution import

### Files to Reference (DO NOT MODIFY)
- `/Users/rajesh/athena/evolution/workflow_discovery.py` — WorkflowDiscovery
- `/Users/rajesh/athena/evolution/agent_generator.py` — AgentGenerator
- `/Users/rajesh/athena/memory/agemem.py` — Memory access

### Constraints
- Use async/await for all operations
- Focus on experience collection and replay mechanisms
- Actual training happens in Sprint 4
- Simple knowledge sharing mechanisms for now

## Input
- WorkflowDiscovery and AgentGenerator implementations
- AgentEvolver paper specification
- AgeMem interface for experience storage

## Expected Output

### File: `/Users/rajesh/athena/evolution/cooperative_evolution.py`
```python
from typing import Dict, Any, List, Optional, Tuple
import asyncio
from datetime import datetime
from collections import deque
import json

class Experience:
    """
    Represents a single agent experience for replay.

    Attributes:
        experience_id: Unique identifier
        agent_id: Agent that generated this experience
        state: State/observation at time of experience
        action: Action taken
        outcome: Result of action
        reward: Reward received
        metadata: Additional context
    """

    def __init__(self, agent_id: str, state: Dict[str, Any], action: Dict[str, Any],
                 outcome: Dict[str, Any], reward: float):
        self.experience_id = f"{agent_id}_{datetime.now().timestamp()}"
        self.agent_id = agent_id
        self.state = state
        self.action = action
        self.outcome = outcome
        self.reward = reward
        self.timestamp = datetime.now().isoformat()
        self.metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize experience to dict."""
        return {
            'experience_id': self.experience_id,
            'agent_id': self.agent_id,
            'state': self.state,
            'action': self.action,
            'outcome': self.outcome,
            'reward': self.reward,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Experience':
        """Deserialize experience from dict."""
        exp = Experience(
            data['agent_id'],
            data['state'],
            data['action'],
            data['outcome'],
            data['reward']
        )
        exp.experience_id = data['experience_id']
        exp.timestamp = data['timestamp']
        exp.metadata = data.get('metadata', {})
        return exp


class CooperativeEvolution:
    """
    Enable cooperative evolution through multi-agent experience replay.

    Based on: AgentEvolver paper
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize cooperative evolution.

        Args:
            config: Configuration dict containing:
                - max_experience_buffer: Max experiences to store (default: 10000)
                - replay_batch_size: Batch size for experience replay (default: 32)
                - knowledge_sharing_rate: Rate of knowledge sharing (default: 0.1)
                - min_reward_threshold: Min reward to keep experience (default: 0.5)
        """
        self.max_buffer_size = config.get('max_experience_buffer', 10000)
        self.replay_batch_size = config.get('replay_batch_size', 32)
        self.sharing_rate = config.get('knowledge_sharing_rate', 0.1)
        self.min_reward_threshold = config.get('min_reward_threshold', 0.5)

        # Experience buffer per agent
        self.experience_buffers: Dict[str, deque] = {}

        # Shared experience pool (high-quality experiences)
        self.shared_pool: deque = deque(maxlen=self.max_buffer_size)

        # Performance tracking
        self.agent_performance: Dict[str, List[float]] = {}

    async def add_experience(self, agent_id: str, experience: Experience):
        """
        Add experience from an agent.

        Args:
            agent_id: Agent that generated experience
            experience: Experience object
        """
        # Initialize buffer if needed
        if agent_id not in self.experience_buffers:
            self.experience_buffers[agent_id] = deque(maxlen=self.max_buffer_size)
            self.agent_performance[agent_id] = []

        # Add to agent's buffer
        self.experience_buffers[agent_id].append(experience)

        # Track performance
        self.agent_performance[agent_id].append(experience.reward)
        if len(self.agent_performance[agent_id]) > 100:
            self.agent_performance[agent_id].pop(0)

        # Add high-quality experiences to shared pool
        if experience.reward >= self.min_reward_threshold:
            self.shared_pool.append(experience)

    async def replay_experiences(self, agent_id: str, batch_size: Optional[int] = None) -> List[Experience]:
        """
        Sample experiences for replay (own + shared).

        Args:
            agent_id: Agent requesting replay
            batch_size: Number of experiences to sample (default: from config)

        Returns:
            List of sampled experiences
        """
        if batch_size is None:
            batch_size = self.replay_batch_size

        experiences = []

        # Get agent's own experiences
        if agent_id in self.experience_buffers:
            own_buffer = list(self.experience_buffers[agent_id])
            num_own = int(batch_size * (1 - self.sharing_rate))

            if len(own_buffer) >= num_own:
                # Sample from own experiences
                import random
                experiences.extend(random.sample(own_buffer, num_own))

        # Get shared experiences from other agents
        num_shared = batch_size - len(experiences)
        if num_shared > 0 and len(self.shared_pool) > 0:
            shared_list = list(self.shared_pool)
            # Filter out own experiences
            other_experiences = [e for e in shared_list if e.agent_id != agent_id]

            if len(other_experiences) >= num_shared:
                import random
                experiences.extend(random.sample(other_experiences, num_shared))
            else:
                experiences.extend(other_experiences)

        return experiences

    async def share_knowledge(self, source_agent: str, target_agent: str) -> Dict[str, Any]:
        """
        Share knowledge from one agent to another.

        Args:
            source_agent: Agent sharing knowledge
            target_agent: Agent receiving knowledge

        Returns:
            Summary of shared knowledge
        """
        if source_agent not in self.experience_buffers:
            return {'status': 'no_experiences', 'shared_count': 0}

        # Get high-performing experiences from source
        source_experiences = list(self.experience_buffers[source_agent])
        high_quality = [e for e in source_experiences if e.reward >= self.min_reward_threshold]

        if not high_quality:
            return {'status': 'no_quality_experiences', 'shared_count': 0}

        # Add to shared pool (target will access via replay)
        for exp in high_quality:
            self.shared_pool.append(exp)

        return {
            'status': 'success',
            'shared_count': len(high_quality),
            'avg_reward': sum(e.reward for e in high_quality) / len(high_quality)
        }

    async def get_population_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the agent population.

        Returns:
            Population performance metrics
        """
        stats = {
            'total_agents': len(self.experience_buffers),
            'total_experiences': sum(len(buf) for buf in self.experience_buffers.values()),
            'shared_pool_size': len(self.shared_pool),
            'agent_performance': {}
        }

        for agent_id, rewards in self.agent_performance.items():
            if rewards:
                stats['agent_performance'][agent_id] = {
                    'avg_reward': sum(rewards) / len(rewards),
                    'max_reward': max(rewards),
                    'min_reward': min(rewards),
                    'experience_count': len(self.experience_buffers.get(agent_id, []))
                }

        return stats

    async def identify_top_performers(self, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Identify top-performing agents based on average reward.

        Args:
            top_k: Number of top performers to return

        Returns:
            List of (agent_id, avg_reward) tuples
        """
        agent_scores = []
        for agent_id, rewards in self.agent_performance.items():
            if rewards:
                avg_reward = sum(rewards) / len(rewards)
                agent_scores.append((agent_id, avg_reward))

        agent_scores.sort(key=lambda x: x[1], reverse=True)
        return agent_scores[:top_k]

    async def cross_pollinate(self, top_k: int = 3) -> Dict[str, Any]:
        """
        Share knowledge from top performers to all other agents.

        Args:
            top_k: Number of top performers to share from

        Returns:
            Summary of cross-pollination
        """
        top_performers = await self.identify_top_performers(top_k)

        if not top_performers:
            return {'status': 'no_performers', 'shared_count': 0}

        total_shared = 0
        for agent_id, avg_reward in top_performers:
            if agent_id not in self.experience_buffers:
                continue

            # Get high-quality experiences
            experiences = list(self.experience_buffers[agent_id])
            high_quality = [e for e in experiences if e.reward >= avg_reward * 0.9]

            # Add to shared pool
            for exp in high_quality:
                self.shared_pool.append(exp)
                total_shared += 1

        return {
            'status': 'success',
            'top_performers': [agent_id for agent_id, _ in top_performers],
            'shared_count': total_shared
        }

    async def save_experiences(self, path: str):
        """Save experience buffers to file."""
        data = {
            'buffers': {
                agent_id: [e.to_dict() for e in buf]
                for agent_id, buf in self.experience_buffers.items()
            },
            'shared_pool': [e.to_dict() for e in self.shared_pool],
            'performance': self.agent_performance
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    async def load_experiences(self, path: str):
        """Load experience buffers from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.experience_buffers = {}
        for agent_id, experiences in data.get('buffers', {}).items():
            self.experience_buffers[agent_id] = deque(maxlen=self.max_buffer_size)
            for exp_data in experiences:
                exp = Experience.from_dict(exp_data)
                self.experience_buffers[agent_id].append(exp)

        self.shared_pool = deque(maxlen=self.max_buffer_size)
        for exp_data in data.get('shared_pool', []):
            exp = Experience.from_dict(exp_data)
            self.shared_pool.append(exp)

        self.agent_performance = data.get('performance', {})
```

### Update: `/Users/rajesh/athena/evolution/__init__.py`
Add CooperativeEvolution and Experience to imports and __all__.

## Acceptance Criteria
- [ ] Experience class created with serialization methods
- [ ] CooperativeEvolution class created
- [ ] `add_experience()` method stores agent experiences
- [ ] `replay_experiences()` samples from own and shared experiences
- [ ] `share_knowledge()` enables agent-to-agent knowledge transfer
- [ ] `get_population_stats()` provides population-wide metrics
- [ ] `identify_top_performers()` finds best-performing agents
- [ ] `cross_pollinate()` shares knowledge from top performers
- [ ] Experience buffering with configurable limits
- [ ] Save/load functionality for persistence
- [ ] All methods use async/await pattern
- [ ] Classes are importable and instantiable
- [ ] Docstrings present for all public methods

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
