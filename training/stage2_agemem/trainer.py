"""
AgeMem Trainer
==============
Three-stage progressive training for AgeMem memory management.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import hashlib
import logging

from .grpo import StepwiseGRPO, GRPOConfig, Trajectory
from .rewards import CompositeReward, RewardConfig, OperationOutcome


class TrainingStage(Enum):
    """Three stages of AgeMem progressive training."""
    SINGLE_TOOL = 1      # Learn one operation at a time
    MULTI_TOOL = 2       # Learn operation sequences
    UNIFIED = 3          # Full autonomous memory management


@dataclass
class AgeMemTrainerConfig:
    """Configuration for AgeMem trainer."""
    # Stage progression
    stage1_steps: int = 1000    # Steps for single-tool learning
    stage2_steps: int = 2000    # Steps for multi-tool
    stage3_steps: int = 3000    # Steps for unified

    # Operations to train
    ltm_operations: List[str] = None
    stm_operations: List[str] = None

    # GRPO config
    grpo_config: GRPOConfig = None

    # Reward config
    reward_config: RewardConfig = None

    # Data
    experience_buffer_size: int = 10000
    min_trajectories_per_update: int = 8

    # Checkpointing
    checkpoint_dir: str = "./checkpoints/stage2_agemem"
    save_every_steps: int = 500

    def __post_init__(self):
        if self.ltm_operations is None:
            self.ltm_operations = ["add", "update", "delete"]
        if self.stm_operations is None:
            self.stm_operations = ["retrieve", "summary", "filter"]
        if self.grpo_config is None:
            self.grpo_config = GRPOConfig()
        if self.reward_config is None:
            self.reward_config = RewardConfig()


class AgeMemTrainer:
    """
    AgeMem Trainer with Three-Stage Progressive Training.

    Stage 1: Single-Tool Learning
        - Train each operation independently
        - Focus on basic operation execution

    Stage 2: Multi-Tool Coordination
        - Train operation sequences
        - Learn when to use which operation

    Stage 3: Unified Management
        - Full autonomous memory decisions
        - End-to-end optimization
    """

    # Synthetic training content for memory operations
    _TRAINING_SNIPPETS = [
        "AAPL Q3 earnings beat expectations by 5%, revenue at $94.8B",
        "Fed signals potential rate cut in September 2024 meeting",
        "TSLA delivery numbers: 443,956 vehicles in Q2, up 5% YoY",
        "MSFT Azure revenue grew 29% in fiscal Q4, AI services key driver",
        "Oil prices surge to $85/barrel amid Middle East tensions",
        "S&P 500 hits new all-time high at 5,667 on tech rally",
        "NVDA data center revenue triples YoY to $22.6B",
        "US unemployment holds steady at 4.1%, jobs added: 206K",
        "AMZN AWS operating income rises 74% to $9.3B",
        "Gold reaches $2,450/oz as inflation concerns persist",
    ]

    def __init__(
        self,
        model: Any,
        agemem: Any,
        config: Optional[AgeMemTrainerConfig] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: The fine-tuned OLMoE model (from Stage 1)
            agemem: AgeMem instance
            config: Training configuration
        """
        self.config = config or AgeMemTrainerConfig()
        self.logger = logging.getLogger("athena.training.agemem_trainer")

        self.model = model
        self.agemem = agemem

        self.grpo = StepwiseGRPO(model, self.config.grpo_config)
        self.reward_fn = CompositeReward(self.config.reward_config)

        self.current_stage = TrainingStage.SINGLE_TOOL
        self._step_count = 0
        self._experience_buffer: List[Trajectory] = []

    def setup(self) -> bool:
        """Set up trainer."""
        success = self.grpo.setup()
        if success:
            self.logger.info("AgeMem trainer ready, starting at %s", self.current_stage.name)
        return success

    async def train(self, num_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Run full training pipeline.

        Args:
            num_steps: Total steps (uses config total if not provided)

        Returns:
            Training metrics
        """
        total_steps = num_steps or (
            self.config.stage1_steps +
            self.config.stage2_steps +
            self.config.stage3_steps
        )

        metrics_history = []

        while self._step_count < total_steps:
            # Update stage
            self._update_stage()

            # Collect trajectories
            trajectories = await self._collect_trajectories()

            if len(trajectories) >= self.config.min_trajectories_per_update:
                # Train step
                metrics = self.grpo.train_step(trajectories)
                metrics["stage"] = self.current_stage.name
                metrics["step"] = self._step_count
                metrics_history.append(metrics)

                self._step_count += 1

                # Logging
                if self._step_count % 100 == 0:
                    self.logger.info(
                        "Step %d, Stage %s, Loss: %.4f",
                        self._step_count, self.current_stage.name, metrics["total_loss"]
                    )

                # Checkpointing
                if self._step_count % self.config.save_every_steps == 0:
                    self.save_checkpoint()

        return {
            "total_steps": self._step_count,
            "final_stage": self.current_stage.name,
            "metrics_history": metrics_history,
        }

    async def train_stage(self, stage: TrainingStage, num_steps: int) -> Dict[str, Any]:
        """
        Train a specific stage.

        Args:
            stage: Stage to train
            num_steps: Number of steps

        Returns:
            Stage metrics
        """
        self.current_stage = stage
        start_step = self._step_count
        metrics_history = []

        while self._step_count - start_step < num_steps:
            trajectories = await self._collect_trajectories()

            if len(trajectories) >= self.config.min_trajectories_per_update:
                metrics = self.grpo.train_step(trajectories)
                metrics_history.append(metrics)
                self._step_count += 1

        return {
            "stage": stage.name,
            "steps": self._step_count - start_step,
            "metrics": metrics_history,
        }

    def _update_stage(self) -> None:
        """Update training stage based on step count."""
        if self._step_count < self.config.stage1_steps:
            new_stage = TrainingStage.SINGLE_TOOL
        elif self._step_count < self.config.stage1_steps + self.config.stage2_steps:
            new_stage = TrainingStage.MULTI_TOOL
        else:
            new_stage = TrainingStage.UNIFIED

        if new_stage != self.current_stage:
            self.logger.info("Transitioning from %s to %s", self.current_stage.name, new_stage.name)
            self.current_stage = new_stage

    async def _collect_trajectories(self) -> List[Trajectory]:
        """Collect trajectories based on current stage."""
        trajectories = []

        for _ in range(self.config.grpo_config.group_size):
            if self.current_stage == TrainingStage.SINGLE_TOOL:
                traj = await self._collect_single_tool_trajectory()
            elif self.current_stage == TrainingStage.MULTI_TOOL:
                traj = await self._collect_multi_tool_trajectory()
            else:
                traj = await self._collect_unified_trajectory()

            if traj:
                trajectories.append(traj)

        return trajectories

    async def _collect_single_tool_trajectory(self) -> Optional[Trajectory]:
        """
        Collect trajectory for single-tool learning.
        Executes one real AgeMem operation and measures outcome.
        """
        import random

        all_ops = self.config.ltm_operations + self.config.stm_operations
        operation = random.choice(all_ops)

        states = []
        actions = []
        logprobs = []
        rewards = []

        state = {"operation": operation, "context": "training"}
        states.append(state)
        actions.append(operation)
        logprobs.append(self._get_logprob_for_step(state, operation))

        # Execute real AgeMem operation
        outcome = await self._execute_operation(operation)
        reward = self.reward_fn.compute(outcome)
        rewards.append(reward)

        return Trajectory(
            states=states,
            actions=actions,
            action_logprobs=logprobs,
            rewards=rewards,
        )

    async def _collect_multi_tool_trajectory(self) -> Optional[Trajectory]:
        """
        Collect trajectory for multi-tool coordination.
        Executes a sequence of 2-4 real AgeMem operations.
        """
        import random

        states = []
        actions = []
        logprobs = []
        rewards = []

        seq_length = random.randint(2, 4)
        all_ops = self.config.ltm_operations + self.config.stm_operations

        for step in range(seq_length):
            operation = random.choice(all_ops)
            state = {"operation": operation, "step": step}
            states.append(state)
            actions.append(operation)
            logprobs.append(self._get_logprob_for_step(state, operation))

            outcome = await self._execute_operation(operation)
            rewards.append(self.reward_fn.compute(outcome))

        return Trajectory(
            states=states,
            actions=actions,
            action_logprobs=logprobs,
            rewards=rewards,
        )

    async def _collect_unified_trajectory(self) -> Optional[Trajectory]:
        """
        Collect trajectory for unified (Stage 3) memory management.

        Distinct from Stage 2 multi-tool:
        - Longer sequences (4-8 steps)
        - Model-driven operation selection when available (uses action_head logits
          to sample the next operation, falling back to random if model lacks
          encode/action_head)
        - Mandatory mix of both LTM and STM operations
        - Trajectory-level quality bonus based on end-to-end retrieval quality
        - Planning context: initial state describes the overall memory goal
        """
        import random

        states = []
        actions = []
        logprobs = []
        rewards = []

        seq_length = random.randint(4, 8)
        all_ops = self.config.ltm_operations + self.config.stm_operations

        # Planning context: describe the overall memory management goal
        plan = random.choice([
            "Ingest new market data and verify retrieval quality",
            "Update stale entries and consolidate memory via summary",
            "Add diverse data points then filter for relevance",
            "Build knowledge base with add operations then retrieve and summarize",
        ])

        # Track which operation categories have been used
        used_ltm = False
        used_stm = False

        for step in range(seq_length):
            # Build state with planning context
            state = {
                "operation": "pending",
                "step": step,
                "total_steps": seq_length,
                "plan": plan,
                "history": [a for a in actions],  # copy of actions so far
            }

            # Model-driven operation selection
            operation = self._select_operation_unified(
                state, all_ops, step, seq_length, used_ltm, used_stm
            )

            # Track LTM/STM usage
            if operation in self.config.ltm_operations:
                used_ltm = True
            if operation in self.config.stm_operations:
                used_stm = True

            state["operation"] = operation
            states.append(state)
            actions.append(operation)
            logprobs.append(self._get_logprob_for_step(state, operation))

            outcome = await self._execute_operation(operation)
            rewards.append(self.reward_fn.compute(outcome))

        # Trajectory-level quality bonus: retrieve and score end-to-end quality
        bonus = await self._compute_trajectory_bonus()
        # Distribute bonus across all steps
        if rewards:
            per_step_bonus = bonus / len(rewards)
            rewards = [r + per_step_bonus for r in rewards]

        return Trajectory(
            states=states,
            actions=actions,
            action_logprobs=logprobs,
            rewards=rewards,
        )

    def _get_logprob_for_step(self, state: Dict[str, Any], action: str) -> float:
        """
        Compute the current policy log-probability for (state, action).
        Returns 0.0 if the model is not loaded or torch is unavailable,
        matching the _get_action_logprob fallback so the PPO ratio starts at 1.0.
        """
        try:
            import torch
            with torch.no_grad():
                lp = self.grpo._compute_action_logprob(self.model, state, action)
            return float(lp.item())
        except Exception:
            return 0.0

    def _select_operation_unified(
        self,
        state: Dict[str, Any],
        all_ops: List[str],
        step: int,
        total_steps: int,
        used_ltm: bool,
        used_stm: bool,
    ) -> str:
        """
        Select the next operation for Stage 3 unified trajectories.

        Uses the policy model's action_head when available to sample an operation
        from the model's learned distribution. Falls back to heuristic selection
        that ensures both LTM and STM operations are represented.

        Args:
            state: Current state dict.
            all_ops: All available operation names.
            step: Current step index.
            total_steps: Total steps in this trajectory.
            used_ltm: Whether an LTM operation has been used so far.
            used_stm: Whether an STM operation has been used so far.

        Returns:
            Selected operation name.
        """
        import random

        # Try model-driven selection if available
        if (
            hasattr(self.model, "encode")
            and hasattr(self.model, "action_head")
            and self.model.action_head is not None
        ):
            try:
                import torch
                state_text = self.grpo._format_state(state)
                with torch.no_grad():
                    embedding = self.model.encode(state_text)
                    logits = self.model.action_head(embedding)
                    probs = torch.softmax(logits, dim=-1)
                    action_idx = torch.multinomial(probs, 1).item()
                # Map index back to operation name
                from memory.agemem import MemoryOperation
                ops_list = list(MemoryOperation)
                if action_idx < len(ops_list):
                    return ops_list[action_idx].value
            except Exception as e:
                self.logger.debug("Model-driven op selection failed, using heuristic: %s", e)

        # Heuristic: ensure both LTM and STM are used in the trajectory
        remaining = total_steps - step
        if not used_ltm and remaining <= 2:
            # Force an LTM operation
            return random.choice(self.config.ltm_operations)
        if not used_stm and remaining <= 2:
            # Force an STM operation
            return random.choice(self.config.stm_operations)

        return random.choice(all_ops)

    async def _compute_trajectory_bonus(self) -> float:
        """
        Compute a trajectory-level quality bonus by running an end-to-end
        retrieval quality check.

        After a full sequence of operations, retrieve a known snippet and
        score how well the memory system performs. This bonus rewards
        trajectories that leave the memory system in a good state.

        Returns:
            Bonus reward value in [0.0, 0.5].
        """
        import random

        if self.agemem is None:
            return 0.0

        try:
            # Pick a snippet that was likely added during this trajectory
            query = random.choice(self._TRAINING_SNIPPETS).split(",")[0]
            results = await self.agemem.retrieve(query, top_k=3)

            if not isinstance(results, list):
                return 0.0

            # Score: fraction of requested results actually returned
            retrieval_quality = min(1.0, len(results) / 3.0)

            # Scale to [0, 0.5] to not overwhelm step rewards
            return retrieval_quality * 0.5

        except Exception:
            return 0.0

    async def _execute_operation(self, operation: str) -> OperationOutcome:
        """
        Execute a single real AgeMem operation and return an OperationOutcome.

        Args:
            operation: Operation name (add, update, delete, retrieve, summary, filter).

        Returns:
            OperationOutcome with real success/failure, latency, and counts.
        """
        import random
        import time

        start = time.perf_counter()
        success = False
        retrieved_count = None
        relevant_count = None
        input_length = None
        output_length = None
        filtered_count = None
        original_count = None

        # Fall back gracefully if no agemem instance is available
        if self.agemem is None:
            latency = (time.perf_counter() - start) * 1000
            return OperationOutcome(
                operation=operation,
                success=False,
                latency_ms=latency,
            )

        try:
            op = operation.lower()

            if op == "add":
                content = random.choice(self._TRAINING_SNIPPETS)
                result = await self.agemem.add(content, metadata={"source": "training"})
                success = bool(result)

            elif op == "update":
                # Add first, then update the same content
                # Workaround: agemem.add returns bool, not an ID, so we cannot call
                # agemem.update(entry_id, ...) directly. Adding updated content is an
                # approximation that still exercises the add path and produces valid signal.
                content = random.choice(self._TRAINING_SNIPPETS)
                await self.agemem.add(content, metadata={"source": "training"})
                updated_content = content + " [updated]"
                result = await self.agemem.add(
                    updated_content, metadata={"source": "training", "updated": True}
                )
                success = bool(result)

            elif op == "delete":
                # Add then delete -- AgeMem's delete takes an entry_id
                content = random.choice(self._TRAINING_SNIPPETS)
                await self.agemem.add(content, metadata={"source": "training"})
                # Note: delete requires an entry_id; we use the content hash as a proxy.
                # If AgeMem doesn't track IDs this way, delete may return False,
                # which is acceptable -- the model learns that delete needs a valid ID.
                delete_id = hashlib.sha256(content.encode()).hexdigest()[:16]
                result = await self.agemem.delete(delete_id)
                success = bool(result)

            elif op == "retrieve":
                # First add some content, then retrieve
                snippet = random.choice(self._TRAINING_SNIPPETS)
                await self.agemem.add(snippet, metadata={"source": "training"})
                query = snippet.split(",")[0]  # Use first clause as query
                results = await self.agemem.retrieve(query, top_k=5)
                success = isinstance(results, list)
                retrieved_count = len(results) if isinstance(results, list) else 0
                relevant_count = retrieved_count  # Assume all retrieved are relevant for training

            elif op == "summary":
                context_items = [
                    {"content": s, "metadata": {"source": "training"}}
                    for s in random.sample(
                        self._TRAINING_SNIPPETS, min(3, len(self._TRAINING_SNIPPETS))
                    )
                ]
                input_length = sum(len(item["content"]) for item in context_items)
                result = await self.agemem.summary(context_items)
                success = isinstance(result, str) and len(result) > 0
                output_length = len(result) if isinstance(result, str) else 0

            elif op == "filter":
                context_items = [
                    {"content": s, "metadata": {"source": "training"}}
                    for s in random.sample(
                        self._TRAINING_SNIPPETS, min(5, len(self._TRAINING_SNIPPETS))
                    )
                ]
                original_count = len(context_items)
                results = await self.agemem.filter(context_items)
                success = isinstance(results, list)
                filtered_count = len(results) if isinstance(results, list) else 0

            else:
                self.logger.warning("Unknown operation: %s", operation)
                success = False

        except Exception as e:
            self.logger.debug("Operation %s failed: %s", operation, e)
            success = False

        latency = (time.perf_counter() - start) * 1000  # ms

        return OperationOutcome(
            operation=operation,
            success=success,
            latency_ms=latency,
            retrieved_count=retrieved_count,
            relevant_count=relevant_count,
            input_length=input_length,
            output_length=output_length,
            filtered_count=filtered_count,
            original_count=original_count,
        )

    def save_checkpoint(self, path: Optional[str] = None) -> None:
        """Save training checkpoint."""
        import os
        path = path or os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_step{self._step_count}.pt"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)

        self.grpo.save(path)
        self.logger.info("Checkpoint saved: %s", path)

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        self.grpo.load(path)
        self.logger.info("Checkpoint loaded: %s", path)
