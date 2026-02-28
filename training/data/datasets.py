"""
Training Datasets
=================
PyTorch Dataset implementations for the ATHENA training pipeline.

Provides:
  - FinanceDataset     — general-purpose dataset for market/news/social records
  - AgentTrajectoryDataset — stores agent experience trajectories for AgeMem training

Both implement the standard __getitem__ / __len__ interface and support
optional data augmentation via a configurable transform callable.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset

    class _BaseDataset(TorchDataset):  # type: ignore[misc]
        """Base class when PyTorch is available."""
        pass

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

    class _BaseDataset:  # type: ignore[misc]  # noqa: N801
        """Minimal fallback base class when PyTorch is not installed."""

        def __getitem__(self, index: int) -> Any:
            raise NotImplementedError

        def __len__(self) -> int:
            raise NotImplementedError


class FinanceDataset(_BaseDataset):
    """
    Dataset for formatted financial training records.

    Wraps a list of formatted records produced by DataFormatter and exposes
    them as index-addressable samples. Supports an optional transform for
    data augmentation (e.g., feature noise, text masking).

    Args:
        records: List of formatted training record dicts.
        transform: Optional callable applied to each record on __getitem__.
        augment_prob: Probability of applying noise augmentation (default 0.0).
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        transform: Optional[Callable[[Dict[str, Any]], Any]] = None,
        augment_prob: float = 0.0,
    ) -> None:
        self._records = list(records)
        self._transform = transform
        self._augment_prob = float(augment_prob)
        logger.info("FinanceDataset: %d records loaded", len(self._records))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Any:
        record = dict(self._records[index])

        # Optional feature-level noise augmentation for market records
        if self._augment_prob > 0.0 and random.random() < self._augment_prob:
            record = self._augment(record)

        if self._transform is not None:
            return self._transform(record)
        return record

    def _augment(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Add small Gaussian noise to numeric feature vectors."""
        features = record.get("features")
        if features is not None:
            scale = 0.01
            record["features"] = [
                f + random.gauss(0.0, scale * (abs(f) + 1e-6)) for f in features
            ]
        return record

    def add_records(self, records: List[Dict[str, Any]]) -> None:
        """Append additional records to the dataset."""
        self._records.extend(records)
        logger.debug("FinanceDataset: added %d records (total=%d)", len(records), len(self._records))

    def get_stats(self) -> Dict[str, Any]:
        """Return dataset statistics broken down by content_type."""
        type_counts: Dict[str, int] = {}
        for r in self._records:
            ct = r.get("content_type", "unknown")
            type_counts[ct] = type_counts.get(ct, 0) + 1
        return {"total": len(self._records), "by_content_type": type_counts}


class AgentTrajectoryDataset(_BaseDataset):
    """
    Dataset for agent experience trajectories used in AgeMem training.

    Each sample represents a single (state, action, reward, next_state) tuple
    extracted from stored agent trajectories.

    Args:
        trajectories: List of trajectory dicts, each containing:
            - agent_id (str)
            - task_id (str)
            - states (List[Any])
            - actions (List[Any])
            - rewards (List[float])
            - metadata (Dict)
        transform: Optional callable applied to each sample on __getitem__.
        augment_prob: Probability of reward noise augmentation (default 0.0).
    """

    def __init__(
        self,
        trajectories: Optional[List[Dict[str, Any]]] = None,
        transform: Optional[Callable[[Dict[str, Any]], Any]] = None,
        augment_prob: float = 0.0,
    ) -> None:
        self._samples: List[Dict[str, Any]] = []
        self._transform = transform
        self._augment_prob = float(augment_prob)

        if trajectories:
            for traj in trajectories:
                self._expand_trajectory(traj)

        logger.info(
            "AgentTrajectoryDataset: %d samples from %d trajectories",
            len(self._samples),
            len(trajectories) if trajectories else 0,
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Any:
        sample = dict(self._samples[index])

        if self._augment_prob > 0.0 and random.random() < self._augment_prob:
            sample = self._augment(sample)

        if self._transform is not None:
            return self._transform(sample)
        return sample

    def add_trajectory(self, trajectory: Dict[str, Any]) -> None:
        """
        Expand and add a single trajectory to the dataset.

        Args:
            trajectory: Trajectory dict with states, actions, rewards lists.
        """
        before = len(self._samples)
        self._expand_trajectory(trajectory)
        added = len(self._samples) - before
        logger.debug("AgentTrajectoryDataset: added %d samples from trajectory", added)

    def _expand_trajectory(self, traj: Dict[str, Any]) -> None:
        """Flatten a trajectory into individual (s, a, r, s') samples."""
        states = traj.get("states", [])
        actions = traj.get("actions", [])
        rewards = traj.get("rewards", [])
        agent_id = traj.get("agent_id", "")
        task_id = traj.get("task_id", "")
        metadata = traj.get("metadata", {})

        n = min(len(states), len(actions), len(rewards))
        for i in range(n):
            next_state = states[i + 1] if i + 1 < len(states) else None
            self._samples.append(
                {
                    "agent_id": agent_id,
                    "task_id": task_id,
                    "state": states[i],
                    "action": actions[i],
                    "reward": float(rewards[i]),
                    "next_state": next_state,
                    "done": next_state is None,
                    "metadata": metadata,
                }
            )

    def _augment(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Add small noise to reward for augmentation."""
        sample["reward"] = sample["reward"] + random.gauss(0.0, 0.01)
        return sample

    def get_stats(self) -> Dict[str, Any]:
        """Return dataset statistics."""
        agents: Dict[str, int] = {}
        tasks: Dict[str, int] = {}
        for s in self._samples:
            aid = s.get("agent_id", "unknown")
            tid = s.get("task_id", "unknown")
            agents[aid] = agents.get(aid, 0) + 1
            tasks[tid] = tasks.get(tid, 0) + 1
        return {
            "total_samples": len(self._samples),
            "by_agent": agents,
            "by_task": tasks,
        }
