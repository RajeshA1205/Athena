"""
Nested Learning Framework
=========================
Bilevel meta-learning framework enabling agents to adapt rapidly to specific
tasks (inner loop) while updating meta-parameters that guide learning across
all tasks (outer loop).

Pure Python implementation — no PyTorch dependency in this module.
Full gradient-based training integrates with the GRPO pipeline in Sprint 4.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.config import LearningConfig

logger = logging.getLogger(__name__)


@dataclass
class TaskTrajectory:
    """
    A sequence of states, actions, and rewards from a single task episode.

    Attributes:
        task_id: Identifier for the task type (e.g. "market_analysis", "risk_assess")
        agent_id: The agent that generated this trajectory
        states: Sequence of state observations
        actions: Sequence of actions taken
        rewards: Sequence of scalar rewards received
        metadata: Additional context (market regime, portfolio state, etc.)
        created_at: UTC ISO timestamp of creation
    """

    task_id: str
    agent_id: str
    states: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    rewards: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize trajectory to dictionary."""
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TaskTrajectory":
        """Deserialize trajectory from dictionary."""
        return TaskTrajectory(
            task_id=data["task_id"],
            agent_id=data["agent_id"],
            states=data.get("states", []),
            actions=data.get("actions", []),
            rewards=data.get("rewards", []),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class MetaParameters:
    """
    Learned meta-parameters that guide inner-loop adaptation.

    Attributes:
        param_id: Unique identifier
        values: Named parameter values (lr_scale, exploration_weight, etc.)
        performance_score: Latest outer-loop performance estimate
        update_count: Number of outer-loop updates applied
        created_at: UTC ISO timestamp of creation
    """

    param_id: str
    values: Dict[str, float]
    performance_score: float = 0.0
    update_count: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize meta-parameters to dictionary."""
        return {
            "param_id": self.param_id,
            "values": self.values,
            "performance_score": self.performance_score,
            "update_count": self.update_count,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "MetaParameters":
        """Deserialize meta-parameters from dictionary."""
        mp = MetaParameters(
            param_id=data["param_id"],
            values=data.get("values", {}),
        )
        mp.performance_score = data.get("performance_score", 0.0)
        mp.update_count = data.get("update_count", 0)
        mp.created_at = data.get("created_at", datetime.now(timezone.utc).isoformat())
        return mp


class NestedLearning:
    """
    Nested (bilevel) meta-learning framework.

    Inner loop: Rapid task-specific adaptation using stored trajectories to
    estimate local gradient direction and adjust adaptation parameters.

    Outer loop: Slower meta-parameter update across all tasks, improving the
    initialization point for future inner-loop adaptations.

    Note: This module simulates meta-learning arithmetic without actual
    gradient computation. Full gradient-based MAML training is wired in
    Sprint 4 via the GRPO training pipeline.

    Args:
        config: LearningConfig with inner_lr, outer_lr, inner_steps,
                exploration_coefficient fields.
        agent_id: Identifier of the agent this instance belongs to.
    """

    def __init__(self, config: "LearningConfig", agent_id: str) -> None:
        self.agent_id = agent_id
        self._inner_lr: float = getattr(config, "inner_lr", 1e-4)
        self._outer_lr: float = getattr(config, "outer_lr", 1e-5)
        self._inner_steps: int = getattr(config, "inner_steps", 5)
        self._exploration_coeff: float = getattr(config, "exploration_coefficient", 0.1)

        self.meta_params = MetaParameters(
            param_id=f"{agent_id}_meta",
            values={
                "lr_scale": 1.0,
                "exploration_weight": self._exploration_coeff,
                "adaptation_steps": float(self._inner_steps),
                "baseline_performance": 0.0,
            },
        )

        # task_id → list of TaskTrajectory
        self.task_trajectories: Dict[str, List[TaskTrajectory]] = {}
        from collections import deque
        self.adaptation_history: deque = deque(maxlen=1000)
        self.logger = logging.getLogger(f"athena.learning.{agent_id}")

    # ------------------------------------------------------------------
    # Inner loop
    # ------------------------------------------------------------------

    async def adapt_to_task(
        self, task_id: str, trajectory: TaskTrajectory
    ) -> Dict[str, Any]:
        """
        Inner loop: adapt to a specific task using a single trajectory.

        Stores the trajectory, estimates task performance from rewards, and
        computes an adaptation gain relative to the current baseline.

        Args:
            task_id: Identifier for this task type.
            trajectory: The trajectory to learn from.

        Returns:
            Dict with task_id, task_performance, adaptation_gain,
            inner_steps_run, and a snapshot of meta_params.
        """
        if task_id not in self.task_trajectories:
            self.task_trajectories[task_id] = []
        self.task_trajectories[task_id].append(trajectory)

        rewards = trajectory.rewards
        window = rewards[-min(10, len(rewards)):] if rewards else []
        task_performance = sum(window) / len(window) if window else 0.0

        baseline = self.meta_params.values.get("baseline_performance", 0.0)
        adaptation_gain = task_performance - baseline

        # Simulated inner-loop: adjust lr_scale based on performance delta
        if adaptation_gain > 0:
            # Performing above baseline → can be slightly more aggressive
            self.meta_params.values["lr_scale"] = min(
                2.0,
                self.meta_params.values["lr_scale"] * (1.0 + self._inner_lr * adaptation_gain),
            )
        else:
            # Below baseline → be more conservative
            self.meta_params.values["lr_scale"] = max(
                0.1,
                self.meta_params.values["lr_scale"] * (1.0 + self._inner_lr * adaptation_gain),
            )

        result = {
            "task_id": task_id,
            "task_performance": task_performance,
            "adaptation_gain": adaptation_gain,
            "inner_steps_run": self._inner_steps,
            "meta_params_snapshot": dict(self.meta_params.values),
        }
        self.adaptation_history.append(
            {**result, "timestamp": datetime.now(timezone.utc).isoformat()}
        )
        self.logger.debug(
            "Inner loop adapt: task=%s perf=%.4f gain=%.4f",
            task_id,
            task_performance,
            adaptation_gain,
        )
        return result

    # ------------------------------------------------------------------
    # Outer loop
    # ------------------------------------------------------------------

    async def update_meta_parameters(
        self, task_trajectories: List[TaskTrajectory]
    ) -> Dict[str, Any]:
        """
        Outer loop: update meta-parameters across multiple task trajectories.

        Computes mean performance and reward variance across all provided
        trajectories and applies an EMA update to baseline_performance and
        lr_scale.

        Args:
            task_trajectories: Trajectories from multiple tasks/episodes.

        Returns:
            Dict with updated_params, mean_performance, num_tasks,
            update_count.
        """
        if not task_trajectories:
            return {
                "updated_params": dict(self.meta_params.values),
                "mean_performance": 0.0,
                "num_tasks": 0,
                "update_count": self.meta_params.update_count,
            }

        all_rewards: List[float] = []
        for traj in task_trajectories:
            all_rewards.extend(traj.rewards)

        mean_performance = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0

        # Variance for lr_scale adjustment
        if len(all_rewards) > 1:
            mean_sq = sum(r * r for r in all_rewards) / len(all_rewards)
            variance = mean_sq - mean_performance ** 2
        else:
            variance = 0.0

        # EMA update of baseline
        old_baseline = self.meta_params.values.get("baseline_performance", 0.0)
        new_baseline = (
            mean_performance * self._outer_lr + old_baseline * (1.0 - self._outer_lr)
        )
        self.meta_params.values["baseline_performance"] = new_baseline

        # High variance → slightly increase lr_scale to explore more
        if variance > 0.01:
            self.meta_params.values["lr_scale"] = min(
                2.0, self.meta_params.values["lr_scale"] * 1.01
            )

        self.meta_params.update_count += 1
        self.meta_params.performance_score = mean_performance

        self.logger.info(
            "Outer loop update #%d: mean_perf=%.4f variance=%.4f",
            self.meta_params.update_count,
            mean_performance,
            variance,
        )
        return {
            "updated_params": dict(self.meta_params.values),
            "mean_performance": mean_performance,
            "num_tasks": len(task_trajectories),
            "update_count": self.meta_params.update_count,
        }

    # ------------------------------------------------------------------
    # Knowledge consolidation
    # ------------------------------------------------------------------

    async def consolidate_knowledge(self, recent_window: int = 100) -> Dict[str, Any]:
        """
        Consolidate stored trajectories to prevent unbounded growth and
        identify top-performing task configurations.

        Args:
            recent_window: Number of most-recent trajectories to retain
                           when computing overall statistics.

        Returns:
            Dict with consolidated_tasks, mean_reward, top_tasks,
            pruned_entries.
        """
        # Flatten all trajectories, sorted by created_at
        all_trajs: List[TaskTrajectory] = []
        for trajs in self.task_trajectories.values():
            all_trajs.extend(trajs)

        all_trajs.sort(key=lambda t: t.created_at)
        recent = all_trajs[-recent_window:] if len(all_trajs) > recent_window else all_trajs

        all_rewards: List[float] = []
        for t in recent:
            all_rewards.extend(t.rewards)

        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0

        # Per-task mean rewards
        task_means: Dict[str, float] = {}
        for task_id, trajs in self.task_trajectories.items():
            t_rewards: List[float] = []
            for t in trajs:
                t_rewards.extend(t.rewards)
            task_means[task_id] = sum(t_rewards) / len(t_rewards) if t_rewards else 0.0

        top_tasks = [tid for tid, m in task_means.items() if m >= mean_reward]

        # Prune each task's trajectory list to last 50
        pruned_count = 0
        for task_id in self.task_trajectories:
            current = self.task_trajectories[task_id]
            if len(current) > 50:
                pruned_count += len(current) - 50
                self.task_trajectories[task_id] = current[-50:]

        self.logger.info(
            "Consolidation: tasks=%d mean_reward=%.4f pruned=%d",
            len(self.task_trajectories),
            mean_reward,
            pruned_count,
        )
        return {
            "consolidated_tasks": len(self.task_trajectories),
            "mean_reward": mean_reward,
            "top_tasks": top_tasks,
            "pruned_entries": pruned_count,
        }

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    async def get_task_trajectory(self, task_id: str) -> List[TaskTrajectory]:
        """Return all stored trajectories for a given task_id."""
        return self.task_trajectories.get(task_id, [])

    async def get_exploration_weight(self, task_id: str) -> float:
        """
        Return exploration weight for a task, reduced as familiarity grows.

        New tasks receive the full exploration_weight; tasks with many stored
        trajectories approach a minimum of 0.05.

        Args:
            task_id: The task to compute exploration weight for.

        Returns:
            Exploration weight in [0.05, exploration_weight].
        """
        base_weight = self.meta_params.values.get(
            "exploration_weight", self._exploration_coeff
        )
        n = len(self.task_trajectories.get(task_id, []))
        if n == 0:
            return base_weight
        # Decay toward 0.05 as n grows; halved every 20 trajectories
        decayed = base_weight * (0.5 ** (n / 20.0))
        return max(0.05, decayed)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def save_state(self, path: str) -> None:
        """
        Persist meta-parameters and trajectory summaries to JSON.

        Full trajectory state arrays are not saved to keep files compact;
        only reward statistics and metadata are written.

        Args:
            path: File path to write JSON state to.
        """
        try:
            summaries: Dict[str, Any] = {}
            for task_id, trajs in self.task_trajectories.items():
                summaries[task_id] = [
                    {
                        "task_id": t.task_id,
                        "agent_id": t.agent_id,
                        "reward_mean": sum(t.rewards) / len(t.rewards) if t.rewards else 0.0,
                        "reward_count": len(t.rewards),
                        "metadata": t.metadata,
                        "created_at": t.created_at,
                    }
                    for t in trajs
                ]
            data = {
                "meta_params": self.meta_params.to_dict(),
                "trajectory_summaries": summaries,
                "adaptation_history_count": len(self.adaptation_history),
            }

            def _write() -> None:
                with open(path, "w") as f:
                    json.dump(data, f, indent=2)

            await asyncio.to_thread(_write)
            self.logger.info("NestedLearning state saved to %s", path)
        except Exception as e:
            self.logger.error("Failed to save state: %s", e)

    async def load_state(self, path: str) -> None:
        """
        Restore meta-parameters from a previously saved JSON state file.

        Args:
            path: File path to read JSON state from.
        """
        try:
            def _read() -> Dict[str, Any]:
                with open(path, "r") as f:
                    return json.load(f)

            data = await asyncio.to_thread(_read)
            self.meta_params = MetaParameters.from_dict(data["meta_params"])
            self.logger.info("NestedLearning state loaded from %s", path)
        except Exception as e:
            self.logger.error("Failed to load state: %s", e)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return monitoring statistics for this NestedLearning instance."""
        total_trajs = sum(len(v) for v in self.task_trajectories.values())
        return {
            "agent_id": self.agent_id,
            "total_tasks": len(self.task_trajectories),
            "total_trajectories": total_trajs,
            "meta_params": dict(self.meta_params.values),
            "meta_update_count": self.meta_params.update_count,
            "performance_score": self.meta_params.performance_score,
            "adaptation_history_length": len(self.adaptation_history),
        }
