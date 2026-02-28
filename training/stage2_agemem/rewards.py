"""
Composite Reward Function
=========================
Reward computation for AgeMem training based on the paper's formulation:
R = α * R_task + β * R_efficiency + γ * R_quality
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import math


@dataclass
class RewardConfig:
    """Configuration for reward computation."""
    # Weights for composite reward
    alpha: float = 0.5  # Task completion weight
    beta: float = 0.3   # Efficiency weight
    gamma: float = 0.2  # Quality weight

    # Task reward parameters
    success_reward: float = 1.0
    failure_penalty: float = -0.5

    # Efficiency parameters
    max_latency_ms: float = 1000.0
    optimal_latency_ms: float = 100.0

    # Quality parameters
    retrieve_optimal_count: int = 5
    summary_compression_target: float = 0.3
    filter_reduction_target: float = 0.5


@dataclass
class OperationOutcome:
    """Outcome of a memory operation for reward calculation."""
    operation: str
    success: bool
    latency_ms: float
    retrieved_count: Optional[int] = None
    relevant_count: Optional[int] = None
    input_length: Optional[int] = None
    output_length: Optional[int] = None
    filtered_count: Optional[int] = None
    original_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CompositeReward:
    """
    Composite Reward Function for AgeMem GRPO Training.

    R = α·R_task + β·R_efficiency + γ·R_quality

    Components:
    - R_task: Binary success/failure
    - R_efficiency: Latency-based (exponential decay)
    - R_quality: Operation-specific quality metrics
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

    def compute(self, outcome: OperationOutcome) -> float:
        """Compute composite reward."""
        r_task = self._task_reward(outcome)
        r_efficiency = self._efficiency_reward(outcome)
        r_quality = self._quality_reward(outcome)

        return (
            self.config.alpha * r_task +
            self.config.beta * r_efficiency +
            self.config.gamma * r_quality
        )

    def compute_detailed(self, outcome: OperationOutcome) -> Dict[str, float]:
        """Compute detailed reward breakdown."""
        r_task = self._task_reward(outcome)
        r_efficiency = self._efficiency_reward(outcome)
        r_quality = self._quality_reward(outcome)

        return {
            "task": r_task,
            "efficiency": r_efficiency,
            "quality": r_quality,
            "composite": (
                self.config.alpha * r_task +
                self.config.beta * r_efficiency +
                self.config.gamma * r_quality
            ),
        }

    def _task_reward(self, outcome: OperationOutcome) -> float:
        """Task completion reward."""
        return self.config.success_reward if outcome.success else self.config.failure_penalty

    def _efficiency_reward(self, outcome: OperationOutcome) -> float:
        """Efficiency reward based on latency."""
        if outcome.latency_ms >= self.config.max_latency_ms:
            return 0.0
        normalized = outcome.latency_ms / self.config.optimal_latency_ms
        return math.exp(-normalized + 1)

    def _quality_reward(self, outcome: OperationOutcome) -> float:
        """Quality reward based on operation type."""
        op = outcome.operation.lower()

        if op == "retrieve":
            count = outcome.relevant_count or outcome.retrieved_count or 0
            return min(1.0, count / self.config.retrieve_optimal_count)

        elif op == "summary":
            if outcome.input_length and outcome.output_length:
                compression = outcome.output_length / outcome.input_length
                target = self.config.summary_compression_target
                return 1.0 - abs(compression - target)
            return 0.5

        elif op == "filter":
            if outcome.original_count and outcome.filtered_count:
                reduction = 1.0 - (outcome.filtered_count / outcome.original_count)
                target = self.config.filter_reduction_target
                if reduction < target:
                    return reduction / target
                return 1.0
            return 0.5

        return 1.0 if outcome.success else 0.0


class TrajectoryReward:
    """Reward computation for operation trajectories."""

    def __init__(self, reward_fn: CompositeReward):
        self.reward_fn = reward_fn

    def compute_trajectory_reward(
        self,
        outcomes: List[OperationOutcome],
        discount: float = 0.99,
    ) -> float:
        """Compute discounted cumulative reward."""
        total = 0.0
        for i, outcome in enumerate(outcomes):
            total += (discount ** i) * self.reward_fn.compute(outcome)
        return total

    def compute_advantages(
        self,
        outcomes: List[OperationOutcome],
        baseline: Optional[float] = None,
    ) -> List[float]:
        """Compute advantages for GRPO."""
        rewards = [self.reward_fn.compute(o) for o in outcomes]
        if baseline is None:
            baseline = sum(rewards) / len(rewards) if rewards else 0.0
        return [r - baseline for r in rewards]
