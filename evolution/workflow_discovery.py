"""
Workflow Discovery Module
==========================
Discovers and manages successful workflow patterns from agent interaction histories.
Pure Python implementation without PyTorch dependencies.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging


@dataclass
class WorkflowPattern:
    """
    Represents a discovered workflow pattern from agent interactions.

    Attributes:
        pattern_id: Unique identifier for the pattern
        agent_sequence: Ordered list of agent names involved
        interaction_pattern: Communication graph and message types
        success_rate: Success rate of this pattern (0.0 to 1.0)
        use_count: Number of times this pattern has been observed
        metadata: Additional pattern metadata
        created_at: Timestamp when pattern was first discovered
    """
    pattern_id: str
    agent_sequence: List[str]
    interaction_pattern: Dict[str, Any]
    success_rate: float = 0.0
    use_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize workflow pattern to dictionary.

        Returns:
            Dictionary representation of the pattern
        """
        return {
            "pattern_id": self.pattern_id,
            "agent_sequence": self.agent_sequence,
            "interaction_pattern": self.interaction_pattern,
            "success_rate": self.success_rate,
            "use_count": self.use_count,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "WorkflowPattern":
        """
        Deserialize workflow pattern from dictionary.

        Args:
            data: Dictionary containing pattern data

        Returns:
            WorkflowPattern instance
        """
        return WorkflowPattern(
            pattern_id=data["pattern_id"],
            agent_sequence=data["agent_sequence"],
            interaction_pattern=data["interaction_pattern"],
            success_rate=data.get("success_rate", 0.0),
            use_count=data.get("use_count", 0),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )


class WorkflowDiscovery:
    """
    Discovers and manages successful workflow patterns from agent execution traces.

    This class analyzes agent interaction histories to identify recurring successful
    patterns, which can be used to optimize future multi-agent collaborations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize workflow discovery system.

        Args:
            config: Configuration dictionary with optional keys:
                - min_success_rate: Minimum success rate for pattern (default: 0.7)
                - min_use_count: Minimum usage count for pattern (default: 3)
                - similarity_threshold: Minimum similarity for pattern matching (default: 0.8)
        """
        config = config or {}

        self.min_success_rate: float = config.get("min_success_rate", 0.7)
        self.min_use_count: int = config.get("min_use_count", 3)
        self.similarity_threshold: float = config.get("similarity_threshold", 0.8)

        self.workflow_library: Dict[str, WorkflowPattern] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("athena.evolution.workflow_discovery")

        self.logger.info(
            f"WorkflowDiscovery initialized: min_success_rate={self.min_success_rate}, "
            f"min_use_count={self.min_use_count}, similarity_threshold={self.similarity_threshold}"
        )

    async def analyze_execution(self, execution_trace: Dict[str, Any]) -> Optional[str]:
        """
        Analyze an execution trace and extract or update workflow pattern.

        Args:
            execution_trace: Dictionary containing:
                - agents: List of agent names involved
                - interactions: List of interaction records
                - outcome: Success/failure indicator
                - metadata: Additional trace metadata

        Returns:
            Pattern ID if pattern was extracted/updated, None otherwise
        """
        self.logger.debug(f"Analyzing execution trace with {len(execution_trace.get('interactions', []))} interactions")

        # Store execution trace in history
        self.execution_history.append(execution_trace)

        # Extract agent sequence
        agent_sequence = execution_trace.get("agents", [])
        if not agent_sequence:
            self.logger.warning("No agents found in execution trace")
            return None

        # Extract interaction pattern
        interactions = execution_trace.get("interactions", [])
        interaction_pattern = self._extract_interaction_pattern(interactions)

        # Generate pattern ID
        pattern_id = self._generate_pattern_id(agent_sequence, interaction_pattern)

        # Get or create pattern
        if pattern_id in self.workflow_library:
            pattern = self.workflow_library[pattern_id]

            # Update pattern statistics
            outcome = execution_trace.get("outcome", {})
            success = outcome.get("success", False)

            # Update success rate using incremental average
            pattern.use_count += 1
            old_success_rate = pattern.success_rate
            if success:
                pattern.success_rate = old_success_rate + (1.0 - old_success_rate) / pattern.use_count
            else:
                pattern.success_rate = old_success_rate * (pattern.use_count - 1) / pattern.use_count

            self.logger.info(
                f"Updated pattern {pattern_id}: use_count={pattern.use_count}, "
                f"success_rate={pattern.success_rate:.3f}"
            )
        else:
            # Create new pattern
            outcome = execution_trace.get("outcome", {})
            success = outcome.get("success", False)

            pattern = WorkflowPattern(
                pattern_id=pattern_id,
                agent_sequence=agent_sequence,
                interaction_pattern=interaction_pattern,
                success_rate=1.0 if success else 0.0,
                use_count=1,
                metadata=execution_trace.get("metadata", {}),
            )

            self.workflow_library[pattern_id] = pattern
            self.logger.info(f"Created new pattern {pattern_id}")

        return pattern_id

    async def get_successful_patterns(self) -> List[WorkflowPattern]:
        """
        Get workflow patterns that meet success criteria.

        Returns:
            List of successful patterns (meeting min_success_rate and min_use_count)
        """
        successful_patterns = [
            pattern
            for pattern in self.workflow_library.values()
            if pattern.success_rate >= self.min_success_rate
            and pattern.use_count >= self.min_use_count
        ]

        self.logger.debug(f"Found {len(successful_patterns)} successful patterns")
        return successful_patterns

    async def find_similar_patterns(
        self, reference: WorkflowPattern
    ) -> List[Tuple[WorkflowPattern, float]]:
        """
        Find patterns similar to a reference pattern.

        Args:
            reference: Reference workflow pattern

        Returns:
            List of tuples (pattern, similarity_score) for patterns above similarity_threshold
        """
        similar_patterns = []

        for pattern in self.workflow_library.values():
            if pattern.pattern_id == reference.pattern_id:
                continue

            similarity = self._calculate_pattern_similarity(reference, pattern)

            if similarity >= self.similarity_threshold:
                similar_patterns.append((pattern, similarity))

        # Sort by similarity descending
        similar_patterns.sort(key=lambda x: x[1], reverse=True)

        self.logger.debug(f"Found {len(similar_patterns)} similar patterns")
        return similar_patterns

    def _extract_interaction_pattern(self, interactions: List[Dict]) -> Dict[str, Any]:
        """
        Extract communication graph and message types from interactions.

        Args:
            interactions: List of interaction records

        Returns:
            Dictionary containing communication_graph and message_types
        """
        communication_graph = {}
        message_types = {}

        for interaction in interactions:
            sender = interaction.get("sender")
            recipient = interaction.get("recipient")
            message_type = interaction.get("message_type", "default")

            if not sender or not recipient:
                continue

            # Build communication graph
            if sender not in communication_graph:
                communication_graph[sender] = []
            if recipient not in communication_graph[sender]:
                communication_graph[sender].append(recipient)

            # Track message types
            edge_key = f"{sender}->{recipient}"
            if edge_key not in message_types:
                message_types[edge_key] = []
            if message_type not in message_types[edge_key]:
                message_types[edge_key].append(message_type)

        return {
            "communication_graph": communication_graph,
            "message_types": message_types,
        }

    def _generate_pattern_id(
        self, agent_sequence: List[str], interaction_pattern: Dict
    ) -> str:
        """
        Generate stable pattern ID using hash of agent sequence and interaction pattern.

        Args:
            agent_sequence: Ordered list of agent names
            interaction_pattern: Communication pattern dictionary

        Returns:
            Hexadecimal pattern ID string
        """
        # Create deterministic representation
        pattern_repr = {
            "agents": sorted(agent_sequence),  # Sort for consistency
            "graph": {
                k: sorted(v) for k, v in interaction_pattern.get("communication_graph", {}).items()
            },
            "types": {
                k: sorted(v) for k, v in interaction_pattern.get("message_types", {}).items()
            },
        }

        # Generate stable hash
        pattern_str = json.dumps(pattern_repr, sort_keys=True)
        pattern_hash = hashlib.sha256(pattern_str.encode()).hexdigest()

        return f"pattern_{pattern_hash[:16]}"

    def _calculate_pattern_similarity(
        self, p1: WorkflowPattern, p2: WorkflowPattern
    ) -> float:
        """
        Calculate similarity between two patterns using Jaccard similarity.

        Args:
            p1: First workflow pattern
            p2: Second workflow pattern

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Agent sequence similarity (Jaccard)
        agents1 = set(p1.agent_sequence)
        agents2 = set(p2.agent_sequence)

        if not agents1 or not agents2:
            agent_similarity = 0.0
        else:
            agent_intersection = len(agents1 & agents2)
            agent_union = len(agents1 | agents2)
            agent_similarity = agent_intersection / agent_union if agent_union > 0 else 0.0

        # Communication graph similarity
        graph1 = p1.interaction_pattern.get("communication_graph", {})
        graph2 = p2.interaction_pattern.get("communication_graph", {})

        # Convert graphs to edge sets
        edges1 = set()
        for sender, recipients in graph1.items():
            for recipient in recipients:
                edges1.add((sender, recipient))

        edges2 = set()
        for sender, recipients in graph2.items():
            for recipient in recipients:
                edges2.add((sender, recipient))

        if not edges1 or not edges2:
            graph_similarity = 0.0
        else:
            edge_intersection = len(edges1 & edges2)
            edge_union = len(edges1 | edges2)
            graph_similarity = edge_intersection / edge_union if edge_union > 0 else 0.0

        # Combine similarities (equal weight)
        overall_similarity = (agent_similarity + graph_similarity) / 2.0

        return overall_similarity

    async def save_library(self, path: str) -> None:
        """
        Save workflow library to JSON file.

        Args:
            path: File path to save to
        """
        data = {
            "patterns": {
                pattern_id: pattern.to_dict()
                for pattern_id, pattern in self.workflow_library.items()
            },
            "config": {
                "min_success_rate": self.min_success_rate,
                "min_use_count": self.min_use_count,
                "similarity_threshold": self.similarity_threshold,
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Saved workflow library to {path}")

    async def load_library(self, path: str) -> None:
        """
        Load workflow library from JSON file.

        Args:
            path: File path to load from
        """
        with open(path, "r") as f:
            data = json.load(f)

        # Load patterns
        self.workflow_library = {
            pattern_id: WorkflowPattern.from_dict(pattern_data)
            for pattern_id, pattern_data in data.get("patterns", {}).items()
        }

        # Load config if present
        config = data.get("config", {})
        if config:
            self.min_success_rate = config.get("min_success_rate", self.min_success_rate)
            self.min_use_count = config.get("min_use_count", self.min_use_count)
            self.similarity_threshold = config.get("similarity_threshold", self.similarity_threshold)

        self.logger.info(f"Loaded {len(self.workflow_library)} patterns from {path}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get workflow discovery statistics.

        Returns:
            Dictionary with statistics including total_patterns, successful_patterns,
            executions_analyzed, and avg_success_rate
        """
        total_patterns = len(self.workflow_library)

        successful_patterns = sum(
            1
            for pattern in self.workflow_library.values()
            if pattern.success_rate >= self.min_success_rate
            and pattern.use_count >= self.min_use_count
        )

        executions_analyzed = len(self.execution_history)

        if total_patterns > 0:
            avg_success_rate = sum(
                pattern.success_rate for pattern in self.workflow_library.values()
            ) / total_patterns
        else:
            avg_success_rate = 0.0

        return {
            "total_patterns": total_patterns,
            "successful_patterns": successful_patterns,
            "executions_analyzed": executions_analyzed,
            "avg_success_rate": avg_success_rate,
        }
