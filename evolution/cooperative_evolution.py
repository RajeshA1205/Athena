"""
Cooperative Evolution Module
============================
Enables multi-agent cooperative improvement through experience replay,
knowledge sharing, and population-based performance tracking.
"""

import json
import logging
import random
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


class Experience:
    """
    Represents a single agent experience for replay.

    Attributes:
        experience_id: Unique identifier derived from agent_id and creation timestamp
        agent_id: Agent that generated this experience
        state: State/observation at the time of the experience
        action: Action taken by the agent
        outcome: Result of the action
        reward: Reward signal received
        timestamp: ISO-8601 creation timestamp (UTC)
        metadata: Additional context for the experience
    """

    def __init__(
        self,
        agent_id: str,
        state: Dict[str, Any],
        action: Dict[str, Any],
        outcome: Dict[str, Any],
        reward: float,
    ) -> None:
        """
        Initialize an experience record.

        Args:
            agent_id: Identifier of the agent that generated this experience
            state: State/observation at the time of the experience
            action: Action taken by the agent
            outcome: Result of the action
            reward: Reward signal received (higher is better)
        """
        ts = datetime.now(timezone.utc)
        self.experience_id: str = f"{agent_id}_{ts.timestamp()}"
        self.agent_id: str = agent_id
        self.state: Dict[str, Any] = state
        self.action: Dict[str, Any] = action
        self.outcome: Dict[str, Any] = outcome
        self.reward: float = reward
        self.timestamp: str = ts.isoformat()
        self.metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize experience to dictionary.

        Returns:
            Dictionary representation of all experience fields
        """
        return {
            "experience_id": self.experience_id,
            "agent_id": self.agent_id,
            "state": self.state,
            "action": self.action,
            "outcome": self.outcome,
            "reward": self.reward,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Experience":
        """
        Deserialize experience from dictionary.

        Restores experience_id and timestamp exactly from the stored data rather
        than generating new values, preserving the original record identity.

        Args:
            data: Dictionary containing serialized experience fields

        Returns:
            Experience instance with all fields restored from data
        """
        exp = Experience(
            agent_id=data["agent_id"],
            state=data["state"],
            action=data["action"],
            outcome=data["outcome"],
            reward=data["reward"],
        )
        exp.experience_id = data["experience_id"]
        exp.timestamp = data["timestamp"]
        exp.metadata = data.get("metadata", {})
        return exp


class CooperativeEvolution:
    """
    Enable cooperative evolution through multi-agent experience replay.

    Agents contribute experiences to per-agent buffers and a shared high-quality
    pool. Replay mixes an agent's own experiences with experiences from the shared
    pool, so agents implicitly learn from each other. Top-performing agents can
    cross-pollinate the rest of the population by seeding the shared pool with
    their best experiences.

    Based on: AgentEvolver — Towards Efficient Self-Evolving Agent Systems
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize cooperative evolution.

        Args:
            config: Configuration dictionary with optional keys:
                - max_experience_buffer: Maximum experiences to store per agent
                  and for the shared pool (default: 10000)
                - replay_batch_size: Number of experiences per replay batch
                  (default: 32)
                - knowledge_sharing_rate: Fraction of each replay batch drawn
                  from the shared pool (default: 0.1)
                - min_reward_threshold: Minimum reward required for an experience
                  to enter the shared pool (default: 0.5)
        """
        self.max_buffer_size: int = config.get("max_experience_buffer", 10000)
        self.replay_batch_size: int = config.get("replay_batch_size", 32)
        self.sharing_rate: float = config.get("knowledge_sharing_rate", 0.1)
        self.min_reward_threshold: float = config.get("min_reward_threshold", 0.5)

        # Per-agent experience buffers (bounded by max_buffer_size)
        self.experience_buffers: Dict[str, deque] = {}

        # Shared pool of high-quality experiences accessible to all agents
        self.shared_pool: deque = deque(maxlen=self.max_buffer_size)

        # Recent reward history per agent (capped at 100 entries)
        self.agent_performance: Dict[str, List[float]] = {}

        logger.info(
            f"CooperativeEvolution initialized: max_buffer_size={self.max_buffer_size}, "
            f"replay_batch_size={self.replay_batch_size}, "
            f"sharing_rate={self.sharing_rate}, "
            f"min_reward_threshold={self.min_reward_threshold}"
        )

    async def add_experience(self, agent_id: str, experience: Experience) -> None:
        """
        Add an experience from an agent to its buffer and optionally the shared pool.

        If the experience reward meets min_reward_threshold it is also added to
        the shared pool so other agents can benefit from it during replay.

        Args:
            agent_id: Identifier of the agent that generated the experience
            experience: Experience object to record
        """
        # Initialize per-agent structures on first encounter
        if agent_id not in self.experience_buffers:
            self.experience_buffers[agent_id] = deque(maxlen=self.max_buffer_size)
            self.agent_performance[agent_id] = []
            logger.debug(f"Initialized experience buffer for agent '{agent_id}'")

        # Add to the agent's own buffer
        self.experience_buffers[agent_id].append(experience)

        # Track recent rewards, capping at 100 to keep memory bounded
        self.agent_performance[agent_id].append(experience.reward)
        if len(self.agent_performance[agent_id]) > 100:
            self.agent_performance[agent_id].pop(0)

        # Promote high-quality experiences to the shared pool
        if experience.reward >= self.min_reward_threshold:
            self.shared_pool.append(experience)
            logger.debug(
                f"Experience {experience.experience_id} added to shared pool "
                f"(reward={experience.reward:.4f})"
            )

    async def replay_experiences(
        self, agent_id: str, batch_size: Optional[int] = None
    ) -> List[Experience]:
        """
        Sample a mixed batch of experiences for replay.

        The batch is composed of:
        - ``(1 - sharing_rate)`` fraction drawn from the agent's own buffer
        - ``sharing_rate`` fraction drawn from other agents in the shared pool

        If either source has fewer experiences than the target fraction,
        all available experiences from that source are included and the remainder
        is filled from the other source.

        Args:
            agent_id: Identifier of the agent requesting the replay batch
            batch_size: Number of experiences to return; defaults to
                replay_batch_size from config

        Returns:
            List of sampled Experience objects (may be smaller than batch_size
            if insufficient experiences are available)
        """
        if batch_size is None:
            batch_size = self.replay_batch_size

        experiences: List[Experience] = []

        # --- Own experiences ---
        own_buffer = list(self.experience_buffers.get(agent_id, []))
        num_own = int(batch_size * (1.0 - self.sharing_rate))

        if own_buffer:
            if len(own_buffer) >= num_own:
                experiences.extend(random.sample(own_buffer, num_own))
            else:
                experiences.extend(own_buffer)

        # --- Shared experiences from other agents ---
        num_shared = batch_size - len(experiences)
        if num_shared > 0 and self.shared_pool:
            other_experiences = [
                e for e in self.shared_pool if e.agent_id != agent_id
            ]
            if len(other_experiences) >= num_shared:
                experiences.extend(random.sample(other_experiences, num_shared))
            else:
                experiences.extend(other_experiences)

        logger.debug(
            f"Replay for agent '{agent_id}': {len(experiences)} experiences "
            f"(requested batch_size={batch_size})"
        )
        return experiences

    async def share_knowledge(
        self, source_agent: str, target_agent: str
    ) -> Dict[str, Any]:
        """
        Share high-quality experiences from a source agent into the shared pool.

        The target agent does not receive experiences directly; it gains access
        to them on the next replay_experiences call. This preserves the
        decoupled, pool-based knowledge sharing model.

        Args:
            source_agent: Agent whose experiences are being shared
            target_agent: Agent that will benefit from the shared experiences

        Returns:
            Status dictionary with keys:
                - status: "success", "no_experiences", or "no_quality_experiences"
                - shared_count: Number of experiences added to the shared pool
                - avg_reward: Average reward of shared experiences (omitted on failure)
        """
        if source_agent not in self.experience_buffers:
            logger.warning(
                f"share_knowledge: source agent '{source_agent}' has no experience buffer"
            )
            return {"status": "no_experiences", "shared_count": 0}

        source_experiences = list(self.experience_buffers[source_agent])
        high_quality = [
            e for e in source_experiences if e.reward >= self.min_reward_threshold
        ]

        if not high_quality:
            logger.info(
                f"share_knowledge: no high-quality experiences from '{source_agent}' "
                f"(threshold={self.min_reward_threshold})"
            )
            return {"status": "no_quality_experiences", "shared_count": 0}

        for exp in high_quality:
            self.shared_pool.append(exp)

        avg_reward = sum(e.reward for e in high_quality) / len(high_quality)
        logger.info(
            f"share_knowledge: '{source_agent}' -> '{target_agent}': "
            f"shared {len(high_quality)} experiences (avg_reward={avg_reward:.4f})"
        )

        return {
            "status": "success",
            "shared_count": len(high_quality),
            "avg_reward": avg_reward,
        }

    async def get_population_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics across the entire agent population.

        Returns:
            Dictionary with keys:
                - total_agents: Number of registered agents
                - total_experiences: Total experiences across all agent buffers
                - shared_pool_size: Current size of the shared experience pool
                - agent_performance: Per-agent dict with avg_reward, max_reward,
                  min_reward, and experience_count
        """
        agent_performance: Dict[str, Dict[str, Any]] = {}

        for agent_id, rewards in self.agent_performance.items():
            if rewards:
                agent_performance[agent_id] = {
                    "avg_reward": sum(rewards) / len(rewards),
                    "max_reward": max(rewards),
                    "min_reward": min(rewards),
                    "experience_count": len(self.experience_buffers.get(agent_id, [])),
                }

        return {
            "total_agents": len(self.experience_buffers),
            "total_experiences": sum(
                len(buf) for buf in self.experience_buffers.values()
            ),
            "shared_pool_size": len(self.shared_pool),
            "agent_performance": agent_performance,
        }

    async def identify_top_performers(
        self, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Identify the top-performing agents by average reward.

        Only agents that have at least one recorded reward are considered.

        Args:
            top_k: Number of top performers to return

        Returns:
            List of (agent_id, avg_reward) tuples sorted by avg_reward descending,
            truncated to top_k entries
        """
        agent_scores: List[Tuple[str, float]] = []

        for agent_id, rewards in self.agent_performance.items():
            if rewards:
                avg_reward = sum(rewards) / len(rewards)
                agent_scores.append((agent_id, avg_reward))

        agent_scores.sort(key=lambda x: x[1], reverse=True)
        top_performers = agent_scores[:top_k]

        logger.debug(
            f"Top {top_k} performers: "
            + ", ".join(f"{aid}={r:.4f}" for aid, r in top_performers)
        )
        return top_performers

    async def cross_pollinate(self, top_k: int = 3) -> Dict[str, Any]:
        """
        Seed the shared pool with high-quality experiences from top performers.

        For each top performer, experiences with reward >= (agent_avg * 0.9) are
        added to the shared pool. This lets the rest of the population learn from
        the best strategies discovered so far.

        Args:
            top_k: Number of top-performing agents to draw from

        Returns:
            Status dictionary with keys:
                - status: "success" or "no_performers"
                - top_performers: List of agent IDs contributing experiences
                - shared_count: Total number of experiences added to the shared pool
        """
        top_performers = await self.identify_top_performers(top_k)

        if not top_performers:
            logger.info("cross_pollinate: no performers available")
            return {"status": "no_performers", "shared_count": 0}

        total_shared = 0

        for agent_id, avg_reward in top_performers:
            if agent_id not in self.experience_buffers:
                continue

            reward_cutoff = avg_reward * 0.9
            high_quality = [
                e
                for e in self.experience_buffers[agent_id]
                if e.reward >= reward_cutoff
            ]

            for exp in high_quality:
                self.shared_pool.append(exp)
                total_shared += 1

        logger.info(
            f"cross_pollinate: seeded shared pool with {total_shared} experiences "
            f"from {[aid for aid, _ in top_performers]}"
        )

        return {
            "status": "success",
            "top_performers": [agent_id for agent_id, _ in top_performers],
            "shared_count": total_shared,
        }

    async def save_experiences(self, path: str) -> None:
        """
        Persist all experience buffers and the shared pool to a JSON file.

        Args:
            path: Filesystem path to write the JSON file to
        """
        try:
            data = {
                "buffers": {
                    agent_id: [e.to_dict() for e in buf]
                    for agent_id, buf in self.experience_buffers.items()
                },
                "shared_pool": [e.to_dict() for e in self.shared_pool],
                "performance": self.agent_performance,
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(
                f"Saved experiences to '{path}': "
                f"{sum(len(b) for b in self.experience_buffers.values())} agent experiences, "
                f"{len(self.shared_pool)} shared"
            )
        except Exception as e:
            logger.error(f"Failed to save experiences to '{path}': {e}")

    async def load_experiences(self, path: str) -> None:
        """
        Load experience buffers and shared pool from a JSON file.

        Replaces all in-memory buffers with the data from the file.

        Args:
            path: Filesystem path of the JSON file to read
        """
        try:
            with open(path, "r") as f:
                data = json.load(f)

            self.experience_buffers = {}
            for agent_id, raw_experiences in data.get("buffers", {}).items():
                self.experience_buffers[agent_id] = deque(maxlen=self.max_buffer_size)
                for exp_data in raw_experiences:
                    self.experience_buffers[agent_id].append(
                        Experience.from_dict(exp_data)
                    )

            self.shared_pool = deque(maxlen=self.max_buffer_size)
            for exp_data in data.get("shared_pool", []):
                self.shared_pool.append(Experience.from_dict(exp_data))

            self.agent_performance = data.get("performance", {})

            logger.info(
                f"Loaded experiences from '{path}': "
                f"{len(self.experience_buffers)} agents, "
                f"{len(self.shared_pool)} shared pool entries"
            )
        except Exception as e:
            logger.error(f"Failed to load experiences from '{path}': {e}")
