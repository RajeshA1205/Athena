# TASK-011: Implement AgentEvolver Agent Generator

## Status
- **State:** Queued
- **Priority:** 🟢 Medium
- **Depends on:** TASK-010
- **Created:** 2026-02-15

## Objective
Create the agent generator component that automatically generates new agent configurations from discovered workflow patterns.

## Context
The agent generator is the second part of AgentEvolver. It takes successful workflow patterns from the WorkflowDiscovery component and uses them to automatically generate new agent configurations, enabling the system to evolve specialized agents for specific tasks.

This component provides:
- Automatic agent configuration generation from patterns
- Task-specific agent instantiation
- Agent capability definition based on patterns
- Performance-based agent selection

Reference the AgentEvolver paper in `/Users/rajesh/athena/architecture/base/` and the architecture plan at `/Users/rajesh/athena/agent-plan/atomic-wibbling-simon.md` (lines 78-82, 178-181).

## Scope & Constraints

### Files to Create
- `/Users/rajesh/athena/evolution/agent_generator.py`

### Files to Modify
- `/Users/rajesh/athena/evolution/__init__.py` — Add AgentGenerator import

### Files to Reference (DO NOT MODIFY)
- `/Users/rajesh/athena/evolution/workflow_discovery.py` — WorkflowPattern, WorkflowDiscovery
- `/Users/rajesh/athena/core/base_agent.py` — BaseAgent
- `/Users/rajesh/athena/core/config.py` — Configuration

### Constraints
- Use async/await for all operations
- Generate agent configs, not actual agent code
- Store generated configs in structured format
- No actual execution yet (that comes in Sprint 3+)

## Input
- WorkflowDiscovery implementation from TASK-010
- BaseAgent specification
- AgentEvolver paper specification

## Expected Output

### File: `/Users/rajesh/athena/evolution/agent_generator.py`
```python
from typing import Dict, Any, List, Optional
import asyncio
import json
from datetime import datetime
from .workflow_discovery import WorkflowPattern, WorkflowDiscovery

class AgentConfiguration:
    """
    Generated agent configuration.

    Attributes:
        config_id: Unique identifier
        agent_type: Type/role of agent
        capabilities: List of agent capabilities
        parameters: Agent-specific parameters
        source_pattern: Workflow pattern this was generated from
        performance_score: Historical performance score
    """

    def __init__(self, config_id: str, agent_type: str, capabilities: List[str]):
        self.config_id = config_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.parameters: Dict[str, Any] = {}
        self.source_pattern: Optional[str] = None
        self.performance_score = 0.0
        self.created_at = datetime.now().isoformat()
        self.metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dict."""
        return {
            'config_id': self.config_id,
            'agent_type': self.agent_type,
            'capabilities': self.capabilities,
            'parameters': self.parameters,
            'source_pattern': self.source_pattern,
            'performance_score': self.performance_score,
            'created_at': self.created_at,
            'metadata': self.metadata
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'AgentConfiguration':
        """Deserialize configuration from dict."""
        config = AgentConfiguration(
            data['config_id'],
            data['agent_type'],
            data['capabilities']
        )
        config.parameters = data.get('parameters', {})
        config.source_pattern = data.get('source_pattern')
        config.performance_score = data.get('performance_score', 0.0)
        config.created_at = data.get('created_at', datetime.now().isoformat())
        config.metadata = data.get('metadata', {})
        return config


class AgentGenerator:
    """
    Generate new agent configurations from discovered workflow patterns.

    Based on: AgentEvolver paper
    """

    def __init__(self, workflow_discovery: WorkflowDiscovery, config: Dict[str, Any]):
        """
        Initialize agent generator.

        Args:
            workflow_discovery: WorkflowDiscovery instance
            config: Configuration dict containing:
                - min_pattern_success: Minimum success rate to generate from (default: 0.8)
                - max_generated_configs: Max configs to maintain (default: 50)
        """
        self.workflow_discovery = workflow_discovery
        self.min_pattern_success = config.get('min_pattern_success', 0.8)
        self.max_generated_configs = config.get('max_generated_configs', 50)

        self.generated_configs: Dict[str, AgentConfiguration] = {}

    async def generate_from_pattern(self, pattern: WorkflowPattern) -> AgentConfiguration:
        """
        Generate agent configuration from a workflow pattern.

        Args:
            pattern: Workflow pattern to generate from

        Returns:
            Generated agent configuration
        """
        # Determine agent type from pattern
        agent_type = self._infer_agent_type(pattern)

        # Extract capabilities from pattern
        capabilities = self._extract_capabilities(pattern)

        # Generate config ID
        config_id = f"generated_{agent_type}_{len(self.generated_configs)}"

        # Create configuration
        config = AgentConfiguration(config_id, agent_type, capabilities)
        config.source_pattern = pattern.pattern_id
        config.performance_score = pattern.success_rate

        # Set parameters based on pattern
        config.parameters = self._generate_parameters(pattern)
        config.metadata = {
            'source_pattern_use_count': pattern.use_count,
            'generated_from': 'workflow_pattern'
        }

        # Store configuration
        self.generated_configs[config_id] = config

        # Prune if too many configs
        await self._prune_configs()

        return config

    def _infer_agent_type(self, pattern: WorkflowPattern) -> str:
        """
        Infer agent type from workflow pattern.

        Args:
            pattern: Workflow pattern

        Returns:
            Inferred agent type
        """
        # Simple heuristic: use most frequent agent in sequence
        if not pattern.agent_sequence:
            return "generic_agent"

        # Count agent types
        type_counts = {}
        for agent in pattern.agent_sequence:
            type_counts[agent] = type_counts.get(agent, 0) + 1

        # Return most common
        most_common = max(type_counts.items(), key=lambda x: x[1])
        return f"specialized_{most_common[0]}"

    def _extract_capabilities(self, pattern: WorkflowPattern) -> List[str]:
        """
        Extract agent capabilities from workflow pattern.

        Args:
            pattern: Workflow pattern

        Returns:
            List of capabilities
        """
        capabilities = set()

        # Add capabilities based on agent sequence
        for agent in pattern.agent_sequence:
            if 'analyst' in agent.lower():
                capabilities.add('analysis')
            if 'risk' in agent.lower():
                capabilities.add('risk_assessment')
            if 'strategy' in agent.lower():
                capabilities.add('strategy_formulation')
            if 'execution' in agent.lower():
                capabilities.add('order_execution')

        # Add capabilities based on interaction pattern
        interaction = pattern.interaction_pattern
        if 'communication_graph' in interaction:
            if len(interaction['communication_graph']) > 3:
                capabilities.add('coordination')

        return list(capabilities)

    def _generate_parameters(self, pattern: WorkflowPattern) -> Dict[str, Any]:
        """
        Generate agent parameters from workflow pattern.

        Args:
            pattern: Workflow pattern

        Returns:
            Parameter dict
        """
        params = {
            'confidence_threshold': 0.7,
            'max_iterations': 10,
            'timeout_seconds': 30
        }

        # Adjust based on pattern metadata
        if pattern.success_rate > 0.9:
            params['confidence_threshold'] = 0.6  # Can be more aggressive

        return params

    async def generate_from_successful_patterns(self) -> List[AgentConfiguration]:
        """
        Generate agent configurations from all successful patterns.

        Returns:
            List of generated configurations
        """
        successful_patterns = await self.workflow_discovery.get_successful_patterns()

        generated = []
        for pattern in successful_patterns:
            if pattern.success_rate >= self.min_pattern_success:
                config = await self.generate_from_pattern(pattern)
                generated.append(config)

        return generated

    async def select_agent_for_task(self, task_requirements: Dict[str, Any]) -> Optional[AgentConfiguration]:
        """
        Select best agent configuration for a given task.

        Args:
            task_requirements: Dict specifying required capabilities, context, etc.

        Returns:
            Best matching agent configuration, or None
        """
        required_capabilities = set(task_requirements.get('capabilities', []))
        if not required_capabilities:
            return None

        # Score each configuration
        scored_configs = []
        for config in self.generated_configs.values():
            config_capabilities = set(config.capabilities)

            # Calculate capability match
            match_score = len(required_capabilities & config_capabilities) / len(required_capabilities)

            # Combine with performance score
            total_score = 0.6 * match_score + 0.4 * config.performance_score

            scored_configs.append((config, total_score))

        if not scored_configs:
            return None

        # Return best match
        scored_configs.sort(key=lambda x: x[1], reverse=True)
        best_config, score = scored_configs[0]

        if score > 0.5:  # Minimum threshold
            return best_config
        return None

    async def _prune_configs(self):
        """Prune low-performing configurations to stay under max limit."""
        if len(self.generated_configs) <= self.max_generated_configs:
            return

        # Sort by performance score
        configs_list = list(self.generated_configs.values())
        configs_list.sort(key=lambda c: c.performance_score)

        # Remove lowest performers
        num_to_remove = len(self.generated_configs) - self.max_generated_configs
        for i in range(num_to_remove):
            del self.generated_configs[configs_list[i].config_id]

    async def save_configs(self, path: str):
        """Save generated configurations to file."""
        data = {
            'configs': [c.to_dict() for c in self.generated_configs.values()],
            'metadata': {
                'total_generated': len(self.generated_configs),
                'min_pattern_success': self.min_pattern_success
            }
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    async def load_configs(self, path: str):
        """Load generated configurations from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.generated_configs = {}
        for config_data in data.get('configs', []):
            config = AgentConfiguration.from_dict(config_data)
            self.generated_configs[config.config_id] = config

    def get_stats(self) -> Dict[str, Any]:
        """Get agent generation statistics."""
        return {
            'total_configs': len(self.generated_configs),
            'avg_performance_score': sum(c.performance_score for c in self.generated_configs.values()) / max(len(self.generated_configs), 1),
            'capability_distribution': self._get_capability_distribution()
        }

    def _get_capability_distribution(self) -> Dict[str, int]:
        """Get distribution of capabilities across configs."""
        dist = {}
        for config in self.generated_configs.values():
            for capability in config.capabilities:
                dist[capability] = dist.get(capability, 0) + 1
        return dist
```

### Update: `/Users/rajesh/athena/evolution/__init__.py`
Add AgentGenerator and AgentConfiguration to imports and __all__.

## Acceptance Criteria
- [ ] AgentConfiguration class created with serialization methods
- [ ] AgentGenerator class created with WorkflowDiscovery integration
- [ ] `generate_from_pattern()` generates configs from workflow patterns
- [ ] `generate_from_successful_patterns()` batch generation from all successful patterns
- [ ] `select_agent_for_task()` selects best config for task requirements
- [ ] Capability extraction and parameter generation from patterns
- [ ] Configuration pruning to maintain max limit
- [ ] Save/load functionality for generated configs
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
