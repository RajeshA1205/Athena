"""
RepExp: Representation-Based Exploration Module
===============================================
Encourages agents to explore diverse strategies by measuring novelty in
representation space and providing exploration bonuses. High novelty →
explore more; low novelty (familiar territory) → exploit.

Pure Python implementation — no numpy or torch dependency.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from datetime import datetime, timezone
from math import sqrt
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.config import LearningConfig

logger = logging.getLogger(__name__)


def _normalize(v: List[float]) -> List[float]:
    """Return a unit-length copy of vector v. Returns v unchanged if all zeros."""
    norm = sqrt(sum(x * x for x in v))
    if norm == 0.0:
        return list(v)
    return [x / norm for x in v]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two pre-normalized unit vectors."""
    return sum(x * y for x, y in zip(a, b))


class RepresentationBuffer:
    """
    Fixed-size buffer for storing agent representation vectors.

    Each entry is a dict with keys:
        - ``representation``: unit-normalized List[float]
        - ``agent_id``: str
        - ``task_id``: str
        - ``timestamp``: UTC ISO str

    Args:
        max_size: Maximum number of entries before oldest are evicted.
        representation_dim: Expected dimensionality (informational only).
    """

    def __init__(self, max_size: int = 10_000, representation_dim: int = 256) -> None:
        self.max_size = max_size
        self.representation_dim = representation_dim
        self._buffer: deque = deque(maxlen=max_size)

    def add(self, representation: List[float], agent_id: str, task_id: str) -> None:
        """
        Normalize and store a representation vector.

        Args:
            representation: Raw representation vector.
            agent_id: Agent that produced this representation.
            task_id: Task context for this representation.
        """
        normed = _normalize(representation)
        self._buffer.append(
            {
                "representation": normed,
                "agent_id": agent_id,
                "task_id": task_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def get_all(self) -> List[List[float]]:
        """Return all stored representation vectors."""
        return [entry["representation"] for entry in self._buffer]

    def get_by_agent(self, agent_id: str) -> List[List[float]]:
        """Return all stored representations for a specific agent."""
        return [
            entry["representation"]
            for entry in self._buffer
            if entry["agent_id"] == agent_id
        ]

    def __len__(self) -> int:
        return len(self._buffer)


class RepExp:
    """
    Representation-Based Exploration (RepExp).

    Computes novelty scores and exploration bonuses by comparing a new
    representation against the k nearest neighbours in the stored buffer.
    A high novelty score means the agent is exploring new territory and
    earns a larger bonus; a low score means the agent is in familiar
    territory and should exploit.

    Args:
        config: LearningConfig with representation_dim, diversity_threshold,
                exploration_coefficient fields.
    """

    def __init__(self, config: "LearningConfig") -> None:
        self.representation_dim: int = getattr(config, "representation_dim", 256)
        self.diversity_threshold: float = getattr(config, "diversity_threshold", 0.5)
        self.exploration_coefficient: float = getattr(
            config, "exploration_coefficient", 0.1
        )
        self.buffer = RepresentationBuffer(
            max_size=10_000, representation_dim=self.representation_dim
        )
        self._exploration_history: deque = deque(maxlen=1_000)
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    async def compute_novelty(self, representation: List[float], k: int = 5) -> float:
        """
        Measure how novel a representation is relative to the buffer.

        Uses k-nearest-neighbour cosine similarity. Novelty = 1 − mean of the
        k highest similarities. Returns 1.0 (fully novel) when the buffer
        has fewer than k entries.

        Args:
            representation: Raw representation vector.
            k: Number of nearest neighbours to consider.

        Returns:
            Novelty score in [0.0, 1.0].
        """
        if len(self.buffer) < k:
            return 1.0

        normed = _normalize(representation)
        all_vecs = self.buffer.get_all()

        similarities = [_cosine_similarity(normed, v) for v in all_vecs]
        similarities.sort(reverse=True)
        top_k = similarities[:k]

        novelty = 1.0 - (sum(top_k) / len(top_k))
        return max(0.0, min(1.0, novelty))

    async def compute_exploration_bonus(
        self, representation: List[float], agent_id: str, task_id: str
    ) -> float:
        """
        Compute exploration bonus for a representation, then store it.

        Novelty is computed first, the representation is added to the buffer,
        and the bonus is recorded in the exploration history.

        Args:
            representation: Raw representation vector.
            agent_id: Agent generating the representation.
            task_id: Task context.

        Returns:
            Exploration bonus = exploration_coefficient × novelty ∈ [0, exploration_coefficient].
        """
        novelty = await self.compute_novelty(representation)
        self.buffer.add(representation, agent_id=agent_id, task_id=task_id)

        bonus = self.exploration_coefficient * novelty
        self._exploration_history.append(
            {
                "agent_id": agent_id,
                "task_id": task_id,
                "novelty": novelty,
                "bonus": bonus,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.logger.debug(
            "Exploration bonus: agent=%s novelty=%.4f bonus=%.4f", agent_id, novelty, bonus
        )
        return bonus

    async def compute_diversity(self, representations: List[List[float]]) -> float:
        """
        Measure diversity within a set of representations.

        Computes mean pairwise cosine distance (1 − cosine_similarity).
        Samples up to 100 representations when the list is large for efficiency.

        Args:
            representations: List of raw representation vectors.

        Returns:
            Mean pairwise diversity in [0.0, 1.0]. Returns 0.0 for < 2 inputs.
        """
        if len(representations) < 2:
            return 0.0

        # Sample for efficiency
        sample = representations if len(representations) <= 100 else representations[:100]
        normed = [_normalize(v) for v in sample]

        distances: List[float] = []
        n = len(normed)
        for i in range(n):
            for j in range(i + 1, n):
                sim = _cosine_similarity(normed[i], normed[j])
                distances.append(1.0 - sim)

        diversity = sum(distances) / len(distances) if distances else 0.0
        return max(0.0, min(1.0, diversity))

    async def select_diverse_subset(
        self, representations: List[List[float]], k: int
    ) -> List[int]:
        """
        Select a diverse subset of k representations using greedy k-medoids.

        Starts with the representation most different from all others (lowest
        mean similarity), then iteratively selects the representation most
        different from the already-selected set.

        Args:
            representations: List of raw representation vectors.
            k: Number of representations to select.

        Returns:
            List of indices into ``representations``.
        """
        n = len(representations)
        if n <= k:
            return list(range(n))

        normed = [_normalize(v) for v in representations]

        # Mean similarity of each representation to all others
        mean_sims = []
        for i in range(n):
            sims = [_cosine_similarity(normed[i], normed[j]) for j in range(n) if j != i]
            mean_sims.append(sum(sims) / len(sims) if sims else 0.0)

        # Start with the most unique (lowest mean similarity)
        selected = [int(min(range(n), key=lambda i: mean_sims[i]))]

        while len(selected) < k:
            best_idx = -1
            best_min_sim = float("inf")
            for i in range(n):
                if i in selected:
                    continue
                # Minimum similarity to any already-selected vector
                min_sim = min(_cosine_similarity(normed[i], normed[s]) for s in selected)
                if min_sim < best_min_sim:
                    best_min_sim = min_sim
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)

        return selected

    async def get_exploration_strategy(
        self, agent_id: str, task_id: str, current_performance: float
    ) -> Dict[str, Any]:
        """
        Recommend an explore-vs-exploit strategy based on recent novelty history.

        Args:
            agent_id: Agent requesting the strategy recommendation.
            task_id: Current task identifier.
            current_performance: Agent's recent performance score (unused in
                                 this basic version; available for extensions).

        Returns:
            Dict with keys:
                - strategy: "explore" | "exploit"
                - exploration_weight: float
                - mean_novelty: float
                - reasoning: str
        """
        # Collect recent novelty scores for this agent
        recent = [
            e["novelty"]
            for e in self._exploration_history
            if e["agent_id"] == agent_id
        ][-20:]

        mean_novelty = sum(recent) / len(recent) if recent else 1.0

        if mean_novelty < self.diversity_threshold:
            strategy = "exploit"
            weight = 0.1
            reasoning = (
                f"Mean novelty {mean_novelty:.3f} below threshold "
                f"{self.diversity_threshold:.3f} — agent is in familiar territory."
            )
        else:
            strategy = "explore"
            weight = min(1.0, self.exploration_coefficient * 2)
            reasoning = (
                f"Mean novelty {mean_novelty:.3f} above threshold "
                f"{self.diversity_threshold:.3f} — agent is exploring new territory."
            )

        return {
            "strategy": strategy,
            "exploration_weight": weight,
            "mean_novelty": mean_novelty,
            "reasoning": reasoning,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def save(self, path: str) -> None:
        """
        Serialize buffer and exploration history to JSON.

        Args:
            path: File path to write JSON data to.
        """
        try:
            data = {
                "buffer": list(self.buffer._buffer),
                "exploration_history": list(self._exploration_history),
            }

            def _write() -> None:
                with open(path, "w") as f:
                    json.dump(data, f, indent=2)

            await asyncio.to_thread(_write)
            self.logger.info("RepExp state saved to %s", path)
        except Exception as e:
            self.logger.error("Failed to save RepExp state: %s", e)

    async def load(self, path: str) -> None:
        """
        Restore buffer and exploration history from JSON.

        Args:
            path: File path to read JSON data from.
        """
        try:
            def _read() -> Dict[str, Any]:
                with open(path, "r") as f:
                    return json.load(f)

            data = await asyncio.to_thread(_read)
            self.buffer._buffer = deque(
                data.get("buffer", []), maxlen=self.buffer.max_size
            )
            self._exploration_history = deque(
                data.get("exploration_history", []), maxlen=1_000
            )
            self.logger.info("RepExp state loaded from %s", path)
        except Exception as e:
            self.logger.error("Failed to load RepExp state: %s", e)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return monitoring statistics for this RepExp instance."""
        recent_novelties = [e["novelty"] for e in self._exploration_history][-100:]
        mean_novelty = (
            sum(recent_novelties) / len(recent_novelties) if recent_novelties else 0.0
        )
        return {
            "buffer_size": len(self.buffer),
            "mean_novelty_recent": mean_novelty,
            "diversity_threshold": self.diversity_threshold,
            "exploration_coefficient": self.exploration_coefficient,
            "exploration_history_size": len(self._exploration_history),
        }
