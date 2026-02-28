"""
Step-wise GRPO Implementation
=============================
Group Relative Policy Optimization for AgeMem memory management.
Based on the AgeMem paper's Step-wise GRPO approach.
"""

import copy
import dataclasses
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    _torch_ver_parts = torch.__version__.split(".")
    _torch_version = (int(_torch_ver_parts[0]), int(_torch_ver_parts[1].split("+")[0].split("a")[0].split("b")[0].split("rc")[0]))
    if _torch_version < (2, 0):
        import warnings
        warnings.warn(
            "ATHENA GRPO requires PyTorch >= 2.0 for safe checkpoint loading. "
            f"Found {torch.__version__}. Training pipeline may not function correctly.",
            UserWarning,
            stacklevel=1,
        )

try:
    from memory.agemem import MemoryOperation
    _ACTION_TO_IDX: Dict[str, int] = {op.value: i for i, op in enumerate(MemoryOperation)}
    _MEMORY_OP_AVAILABLE = True
except ImportError:
    _ACTION_TO_IDX = {}
    _MEMORY_OP_AVAILABLE = False

# Discrete action dimension: ADD, UPDATE, DELETE, RETRIEVE, SUMMARY, FILTER
ACTION_DIM = 6


@dataclass
class GRPOConfig:
    """Configuration for Step-wise GRPO."""
    # Group size for relative comparison
    group_size: int = 8

    # Learning parameters
    learning_rate: float = 1e-5
    beta: float = 0.1  # KL penalty coefficient
    clip_range: float = 0.2  # PPO-style clipping

    # Advantage computation
    use_gae: bool = True
    gae_lambda: float = 0.95
    discount: float = 0.99

    # Training
    num_epochs: int = 4
    batch_size: int = 32
    max_grad_norm: float = 1.0

    # Reference model
    use_reference_model: bool = True
    reference_update_freq: int = 100


@dataclass
class Trajectory:
    """A trajectory of memory operations."""
    states: List[Dict[str, Any]]  # Agent state at each step
    actions: List[str]  # Memory operations (ADD, RETRIEVE, etc.)
    action_logprobs: List[float]  # Log probabilities of actions
    rewards: List[float]  # Rewards at each step
    values: Optional[List[float]] = None  # Value estimates


class StepwiseGRPO:
    """
    Step-wise Group Relative Policy Optimization.

    Key ideas from AgeMem paper:
    1. Group trajectories and compute relative advantages
    2. Optimize policy at each step (not just trajectory-level)
    3. Use composite rewards (task + efficiency + quality)

    Training procedure:
    1. Collect group of G trajectories
    2. Compute step-wise rewards using CompositeReward
    3. Compute relative advantages within group
    4. Update policy using clipped surrogate objective
    """

    def __init__(
        self,
        policy_model: Any,
        config: Optional[GRPOConfig] = None,
    ):
        """
        Initialize GRPO trainer.

        Args:
            policy_model: The model to train (OLMoE with memory action head)
            config: GRPO configuration
        """
        self.config = config or GRPOConfig()
        self.logger = logging.getLogger("athena.training.grpo")

        self.policy_model = policy_model
        self.reference_model = None
        self.optimizer = None

        self._step_count = 0

    def setup(self) -> bool:
        """Set up optimizer and reference model."""
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch not available")
            return False

        try:
            # Optimizer
            self.optimizer = torch.optim.AdamW(
                self.policy_model.parameters(),
                lr=self.config.learning_rate,
            )

            # Reference model (frozen copy)
            if self.config.use_reference_model:
                self._update_reference_model()

            self.logger.info("GRPO setup complete")
            return True

        except Exception as e:
            self.logger.error("GRPO setup failed: %s", e)
            return False

    def compute_group_advantages(
        self,
        trajectories: List[Trajectory],
    ) -> List[List[float]]:
        """
        Compute step-wise advantages relative to group.

        For each step t, advantage is computed relative to
        the mean reward of all trajectories at step t.

        Args:
            trajectories: Group of trajectories

        Returns:
            List of advantage lists (one per trajectory)
        """
        if not trajectories:
            return []

        # Find max length
        max_len = max(len(t.rewards) for t in trajectories)

        # Compute mean reward at each step
        step_means = []
        for t in range(max_len):
            rewards_at_t = [
                traj.rewards[t] for traj in trajectories
                if t < len(traj.rewards)
            ]
            step_means.append(sum(rewards_at_t) / len(rewards_at_t) if rewards_at_t else 0.0)

        # Compute relative advantages
        all_advantages = []
        for traj in trajectories:
            advantages = []
            for t, reward in enumerate(traj.rewards):
                # Advantage = reward - group_mean
                adv = reward - step_means[t]

                # Apply GAE if enabled and values available
                if self.config.use_gae and traj.values:
                    adv = self._compute_gae_advantage(
                        traj.rewards[t:],
                        traj.values[t:],
                    )

                advantages.append(adv)
            all_advantages.append(advantages)

        return all_advantages

    def _compute_gae_advantage(
        self,
        rewards: List[float],
        values: List[float],
    ) -> float:
        """Compute GAE advantage for first step."""
        if not rewards or not values:
            return 0.0

        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.config.discount * next_value - values[t]
            gae = delta + self.config.discount * self.config.gae_lambda * gae

        return gae

    def compute_loss(
        self,
        trajectories: List[Trajectory],
        advantages: List[List[float]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO loss.

        L = -E[min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)] + β·KL(π||π_ref)

        Args:
            trajectories: Group of trajectories
            advantages: Pre-computed advantages

        Returns:
            Tuple of (loss tensor, metrics dict)
        """
        policy_losses = []
        kl_penalties = []

        for traj, advs in zip(trajectories, advantages):
            for t, (state, action, old_logprob, adv) in enumerate(
                zip(traj.states, traj.actions, traj.action_logprobs, advs)
            ):
                # Get new log probability
                new_logprob = self._get_action_logprob(state, action)

                # Probability ratio
                ratio = torch.exp(new_logprob - old_logprob)

                # Clipped surrogate loss
                adv_tensor = torch.tensor(adv, dtype=torch.float32)
                surr1 = ratio * adv_tensor
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_range,
                    1.0 + self.config.clip_range,
                ) * adv_tensor
                policy_loss = -torch.min(surr1, surr2)
                policy_losses.append(policy_loss)

                # KL penalty (if using reference model)
                if self.config.use_reference_model and self.reference_model is not None:
                    ref_logprob = self._get_reference_logprob(state, action)
                    kl = old_logprob - ref_logprob  # Approximate KL
                    kl_penalties.append(kl)

        # Aggregate losses
        total_policy_loss = torch.stack(policy_losses).mean()
        total_kl = torch.stack(kl_penalties).mean() if kl_penalties else torch.tensor(0.0)

        total_loss = total_policy_loss + self.config.beta * total_kl

        metrics = {
            "policy_loss": total_policy_loss.item(),
            "kl_penalty": total_kl.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, metrics

    def train_step(
        self,
        trajectories: List[Trajectory],
    ) -> Dict[str, float]:
        """
        Perform one GRPO training step.

        Args:
            trajectories: Group of trajectories

        Returns:
            Training metrics
        """
        if len(trajectories) < self.config.group_size:
            self.logger.warning(
                f"Got {len(trajectories)} trajectories, expected {self.config.group_size}"
            )

        # Compute advantages
        advantages = self.compute_group_advantages(trajectories)

        # Compute and backprop loss
        self.optimizer.zero_grad()
        loss, metrics = self.compute_loss(trajectories, advantages)
        loss.backward()

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                self.config.max_grad_norm,
            )

        self.optimizer.step()
        self._step_count += 1

        # Update reference model periodically
        if (
            self.config.use_reference_model and
            self._step_count % self.config.reference_update_freq == 0
        ):
            self._update_reference_model()

        return metrics

    def _format_state(self, state: Dict[str, Any]) -> str:
        """Convert a state dict to a text prompt for the OLMoE encoder."""
        parts = []
        for key, value in state.items():
            parts.append(f"{key}: {value}")
        return "; ".join(parts)

    def _compute_action_logprob(self, model: Any, state: Dict[str, Any], action: str) -> "torch.Tensor":
        """
        Shared logic: encode state with model backbone, apply action head, return log-prob.

        Architecture: OLMoE encodes state text into a hidden-state embedding, then the
        MLP action head maps it to a 6-dim logit vector over MemoryOperation actions.
        Note: model must expose .encode(text) and .action_head attribute (MemoryActionHead).
        """
        state_text = self._format_state(state)

        # 1. Encode state with OLMoE backbone
        embedding = model.encode(state_text)

        # 2. Pass through MLP action head → logits of shape (ACTION_DIM,)
        logits = model.action_head(embedding)

        # 3. log_softmax over action dimension
        log_probs = F.log_softmax(logits, dim=-1)

        # 4. Index by action name → scalar log-probability
        action_idx = _ACTION_TO_IDX.get(action, 0)
        return log_probs[action_idx]

    def _get_action_logprob(self, state: Dict[str, Any], action: str) -> "torch.Tensor":
        """
        Get log probability of action given state from the policy model.

        Uses the OLMoE backbone + MLP action head. Returns a scalar tensor with
        requires_grad=True so GRPO gradients can flow through the policy.
        Falls back to a zero tensor if the model does not support encode/action_head.
        """
        if not TORCH_AVAILABLE:
            return torch.tensor(0.0, requires_grad=True)
        if not hasattr(self.policy_model, "encode") or not hasattr(self.policy_model, "action_head"):
            return torch.tensor(0.0, requires_grad=True)
        return self._compute_action_logprob(self.policy_model, state, action)

    def _get_reference_logprob(self, state: Dict[str, Any], action: str) -> "torch.Tensor":
        """
        Get log probability from the reference model (same architecture, frozen weights).

        No gradients are computed — the reference model is used only for KL penalty.
        Falls back to a zero tensor if reference model is unavailable.
        """
        if not TORCH_AVAILABLE:
            return torch.tensor(0.0)
        if self.reference_model is None:
            return torch.tensor(0.0)
        if not hasattr(self.reference_model, "encode") or not hasattr(self.reference_model, "action_head"):
            return torch.tensor(0.0)
        with torch.no_grad():
            return self._compute_action_logprob(self.reference_model, state, action)

    def _update_reference_model(self) -> None:
        """
        Snapshot current policy weights into the reference model.

        On first call creates a deep copy of policy_model (doubles GPU memory).
        On subsequent calls uses load_state_dict for an in-place weight update.
        The reference model is kept frozen — no gradient updates flow through it.
        """
        if self.reference_model is None:
            self.reference_model = copy.deepcopy(self.policy_model)
        elif hasattr(self.policy_model, "state_dict"):
            self.reference_model.load_state_dict(self.policy_model.state_dict())
        self.logger.debug("Reference model updated at step %d", self._step_count)

    def save(self, path: str) -> None:
        """Save GRPO state."""
        torch.save({
            "step_count": self._step_count,
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
            "config": dataclasses.asdict(self.config),
        }, path)

    def load(self, path: str) -> None:
        """Load GRPO state."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for checkpoint loading")
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        self._step_count = checkpoint["step_count"]
        if self.optimizer and checkpoint["optimizer_state"]:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
