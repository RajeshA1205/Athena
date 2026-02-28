"""
Agent Generator Module
======================
Generates agent configurations from successful workflow patterns discovered
by the WorkflowDiscovery system.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .workflow_discovery import WorkflowDiscovery, WorkflowPattern


logger = logging.getLogger(__name__)


class AgentConfiguration:
    """
    Represents a generated agent configuration derived from a workflow pattern.

    Attributes:
        config_id: Unique identifier for this configuration
        agent_type: Type of agent this configuration describes
        capabilities: List of capability labels the agent possesses
        parameters: Operational parameters for the agent
        source_pattern: Pattern ID this configuration was generated from
        performance_score: Expected performance score (0.0 to 1.0)
        created_at: ISO-8601 creation timestamp (UTC)
        metadata: Additional metadata attached to this configuration
    """

    def __init__(
        self,
        config_id: str,
        agent_type: str,
        capabilities: List[str],
    ) -> None:
        """
        Initialize an agent configuration.

        Args:
            config_id: Unique identifier for this configuration
            agent_type: Type of agent this configuration describes
            capabilities: List of capability labels the agent possesses
        """
        self.config_id: str = config_id
        self.agent_type: str = agent_type
        self.capabilities: List[str] = capabilities
        self.parameters: Dict[str, Any] = {}
        self.source_pattern: Optional[str] = None
        self.performance_score: float = 0.0
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self.metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize agent configuration to dictionary.

        Returns:
            Dictionary representation of all configuration fields
        """
        return {
            "config_id": self.config_id,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "parameters": self.parameters,
            "source_pattern": self.source_pattern,
            "performance_score": self.performance_score,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AgentConfiguration":
        """
        Deserialize agent configuration from dictionary.

        Restores all fields from stored data, including config_id, timestamps,
        and scores, rather than generating new values for them.

        Args:
            data: Dictionary containing serialized configuration fields

        Returns:
            AgentConfiguration instance with all fields restored from data
        """
        config = AgentConfiguration(
            config_id=data["config_id"],
            agent_type=data["agent_type"],
            capabilities=data.get("capabilities", []),
        )
        config.parameters = data.get("parameters", {})
        config.source_pattern = data.get("source_pattern")
        config.performance_score = data.get("performance_score", 0.0)
        config.created_at = data.get(
            "created_at", datetime.now(timezone.utc).isoformat()
        )
        config.metadata = data.get("metadata", {})
        return config


class AgentGenerator:
    """
    Generates agent configurations from successful workflow patterns.

    Analyses WorkflowPattern instances to infer the agent type, capabilities,
    and operational parameters that best match a pattern's observed behaviour,
    then stores the resulting AgentConfiguration objects for later retrieval
    and task-matching.
    """

    def __init__(
        self,
        workflow_discovery: WorkflowDiscovery,
        config: Dict[str, Any],
    ) -> None:
        """
        Initialize the agent generator.

        Args:
            workflow_discovery: WorkflowDiscovery instance to source patterns from
            config: Configuration dictionary with optional keys:
                - min_pattern_success: Minimum pattern success rate to generate
                  from (default: 0.8)
                - max_generated_configs: Maximum number of configs to retain in
                  memory before pruning (default: 50)
        """
        config = config or {}

        self.workflow_discovery: WorkflowDiscovery = workflow_discovery
        self.min_pattern_success: float = config.get("min_pattern_success", 0.8)
        self.max_generated_configs: int = config.get("max_generated_configs", 50)

        self.generated_configs: Dict[str, AgentConfiguration] = {}
        self._next_config_id: int = 0

        logger.info(
            f"AgentGenerator initialized: min_pattern_success={self.min_pattern_success}, "
            f"max_generated_configs={self.max_generated_configs}"
        )

    async def generate_from_pattern(
        self, pattern: WorkflowPattern
    ) -> AgentConfiguration:
        """
        Generate an AgentConfiguration from a single workflow pattern.

        The agent type is inferred from the most frequent agent name in the
        pattern's sequence. Capabilities are extracted by scanning agent names
        and the size of the communication graph. Parameters are tuned based on
        the pattern's success rate.

        Args:
            pattern: WorkflowPattern to derive the configuration from

        Returns:
            The generated AgentConfiguration, already stored internally
        """
        agent_type = self._infer_agent_type(pattern)
        capabilities = self._extract_capabilities(pattern)

        config_id = f"generated_{agent_type}_{self._next_config_id}"
        self._next_config_id += 1

        config = AgentConfiguration(
            config_id=config_id,
            agent_type=agent_type,
            capabilities=capabilities,
        )
        config.source_pattern = pattern.pattern_id
        config.performance_score = pattern.success_rate
        config.parameters = self._generate_parameters(pattern)
        config.metadata = {
            "source_pattern_use_count": pattern.use_count,
            "generated_from": "workflow_pattern",
        }

        self.generated_configs[config_id] = config

        logger.debug(
            f"Generated config '{config_id}' from pattern '{pattern.pattern_id}' "
            f"(success_rate={pattern.success_rate:.3f})"
        )

        await self._prune_configs()
        return config

    async def generate_from_successful_patterns(self) -> List[AgentConfiguration]:
        """
        Generate configurations from all patterns that meet the success threshold.

        Fetches successful patterns from the WorkflowDiscovery instance and
        filters to those whose success_rate meets or exceeds min_pattern_success,
        then generates a configuration for each.

        Returns:
            List of generated AgentConfiguration objects
        """
        patterns = await self.workflow_discovery.get_successful_patterns()

        qualifying = [
            p for p in patterns if p.success_rate >= self.min_pattern_success
        ]

        logger.info(
            f"Generating configs from {len(qualifying)} qualifying patterns "
            f"(min_pattern_success={self.min_pattern_success})"
        )

        configs: List[AgentConfiguration] = []
        for pattern in qualifying:
            config = await self.generate_from_pattern(pattern)
            configs.append(config)

        return configs

    async def select_agent_for_task(
        self, task_requirements: Dict[str, Any]
    ) -> Optional[AgentConfiguration]:
        """
        Select the best-matching agent configuration for a set of task requirements.

        Scores each configuration using a weighted combination of capability match
        (60 %) and performance score (40 %). Returns the highest-scoring config if
        its combined score exceeds 0.5, otherwise returns None.

        Args:
            task_requirements: Dictionary with at least a 'capabilities' key
                containing a list of required capability strings

        Returns:
            Best-matching AgentConfiguration, or None if no adequate match exists
        """
        required_capabilities = set(task_requirements.get("capabilities", []))

        if not required_capabilities:
            logger.debug("select_agent_for_task: no required capabilities specified")
            return None

        best_config: Optional[AgentConfiguration] = None
        best_score: float = 0.0

        for config in self.generated_configs.values():
            config_caps = set(config.capabilities)
            intersection = required_capabilities & config_caps
            match_score = len(intersection) / len(required_capabilities)

            total_score = 0.6 * match_score + 0.4 * config.performance_score

            if total_score > best_score:
                best_score = total_score
                best_config = config

        if best_score > 0.5:
            logger.debug(
                f"select_agent_for_task: selected '{best_config.config_id}' "
                f"(score={best_score:.3f})"
            )
            return best_config

        logger.debug(
            f"select_agent_for_task: no adequate match found (best_score={best_score:.3f})"
        )
        return None

    async def save_configs(self, path: str) -> None:
        """
        Persist all generated configurations to a JSON file.

        Args:
            path: Filesystem path to write the JSON file to
        """
        try:
            data = {
                config_id: config.to_dict()
                for config_id, config in self.generated_configs.items()
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(
                f"Saved {len(self.generated_configs)} agent configs to '{path}'"
            )
        except Exception as e:
            logger.error(f"Failed to save agent configs to '{path}': {e}")

    async def load_configs(self, path: str) -> None:
        """
        Load generated configurations from a JSON file.

        Replaces all in-memory configurations with the data from the file.

        Args:
            path: Filesystem path of the JSON file to read
        """
        try:
            with open(path, "r") as f:
                data = json.load(f)

            self.generated_configs = {
                config_id: AgentConfiguration.from_dict(config_data)
                for config_id, config_data in data.items()
            }

            logger.info(
                f"Loaded {len(self.generated_configs)} agent configs from '{path}'"
            )
        except Exception as e:
            logger.error(f"Failed to load agent configs from '{path}': {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for all generated configurations.

        Returns:
            Dictionary with:
                - total_configs: Number of configs currently stored
                - avg_performance_score: Mean performance score across all configs
                - capability_distribution: Dict mapping each capability to the
                  number of configs that possess it
        """
        total_configs = len(self.generated_configs)

        if total_configs > 0:
            avg_performance_score = (
                sum(c.performance_score for c in self.generated_configs.values())
                / total_configs
            )
        else:
            avg_performance_score = 0.0

        capability_distribution: Dict[str, int] = {}
        for config in self.generated_configs.values():
            for cap in config.capabilities:
                capability_distribution[cap] = capability_distribution.get(cap, 0) + 1

        return {
            "total_configs": total_configs,
            "avg_performance_score": avg_performance_score,
            "capability_distribution": capability_distribution,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _infer_agent_type(self, pattern: WorkflowPattern) -> str:
        """
        Infer the agent type from the most frequent name in the pattern's sequence.

        Args:
            pattern: WorkflowPattern to analyse

        Returns:
            Agent type string of the form 'specialized_<name>', or
            'generic_agent' if the sequence is empty
        """
        if not pattern.agent_sequence:
            return "generic_agent"

        frequency: Dict[str, int] = {}
        for agent_name in pattern.agent_sequence:
            frequency[agent_name] = frequency.get(agent_name, 0) + 1

        most_common = max(frequency, key=lambda k: frequency[k])
        return f"specialized_{most_common}"

    def _extract_capabilities(self, pattern: WorkflowPattern) -> List[str]:
        """
        Extract capability labels from a workflow pattern.

        Checks each agent name in the sequence for known role keywords and
        inspects the size of the communication graph to determine if the pattern
        involves coordination across many nodes.

        Args:
            pattern: WorkflowPattern to extract capabilities from

        Returns:
            List of capability strings (may be empty)
        """
        capabilities: List[str] = []

        keyword_map = {
            "analyst": "analysis",
            "risk": "risk_assessment",
            "strategy": "strategy_formulation",
            "execution": "order_execution",
        }

        for agent_name in pattern.agent_sequence:
            agent_lower = agent_name.lower()
            for keyword, capability in keyword_map.items():
                if keyword in agent_lower and capability not in capabilities:
                    capabilities.append(capability)

        communication_graph = pattern.interaction_pattern.get(
            "communication_graph", {}
        )
        if len(communication_graph) > 3 and "coordination" not in capabilities:
            capabilities.append("coordination")

        return capabilities

    def _generate_parameters(self, pattern: WorkflowPattern) -> Dict[str, Any]:
        """
        Generate operational parameters tuned to the pattern's success rate.

        Args:
            pattern: WorkflowPattern whose success rate influences the parameters

        Returns:
            Dictionary of operational parameter key/value pairs
        """
        parameters: Dict[str, Any] = {
            "confidence_threshold": 0.7,
            "max_iterations": 10,
            "timeout_seconds": 30,
        }

        # High-performing patterns warrant a more permissive confidence threshold,
        # allowing the generated agent to act on slightly less certain signals.
        if pattern.success_rate > 0.9:
            parameters["confidence_threshold"] = 0.6

        return parameters

    async def _prune_configs(self) -> None:
        """
        Remove lowest-performing configurations when the storage limit is exceeded.

        Sorts all configs by performance_score ascending and deletes enough of
        them to bring the total count back to max_generated_configs.
        """
        if len(self.generated_configs) <= self.max_generated_configs:
            return

        sorted_ids = sorted(
            self.generated_configs,
            key=lambda cid: self.generated_configs[cid].performance_score,
        )

        excess = len(self.generated_configs) - self.max_generated_configs
        ids_to_remove = sorted_ids[:excess]

        for config_id in ids_to_remove:
            del self.generated_configs[config_id]

        logger.debug(
            f"Pruned {len(ids_to_remove)} low-performing configs; "
            f"{len(self.generated_configs)} remaining"
        )
