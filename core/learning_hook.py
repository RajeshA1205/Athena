"""
LearningHook
============
Post-cycle hook that wires NestedLearning (inner/outer loops), WorkflowDiscovery,
and CooperativeEvolution into the coordinator's act() loop.

Instantiate once, pass to ``CoordinatorAgent(post_cycle_hooks=[hook])``, and
every completed coordination cycle will automatically trigger learning updates.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from learning.nested_learning import NestedLearning
    from evolution.workflow_discovery import WorkflowDiscovery
    from evolution.cooperative_evolution import CooperativeEvolution


class LearningHook:
    """
    Encapsulates the learning + evolution logic executed after each agent cycle.

    Responsibilities:
    - Build a ``TaskTrajectory`` for every agent from their ``AgentAction`` and
      call the inner-loop ``adapt_to_task``.
    - Every ``meta_update_interval`` cycles: run the outer-loop
      ``update_meta_parameters`` for each agent.
    - Every ``consolidation_interval`` cycles: call ``consolidate_knowledge``
      for each agent.
    - Build an execution trace and call ``WorkflowDiscovery.analyze_execution``.
    - If ``cooperative_evolution`` is provided: add an ``Experience`` for each
      agent so cross-agent learning can occur.

    Args:
        learners: Mapping of agent name → ``NestedLearning`` instance.
        workflow_discovery: Shared ``WorkflowDiscovery`` instance.
        cooperative_evolution: Optional ``CooperativeEvolution`` instance.
        meta_update_interval: How often (in cycles) to run the outer loop
            (default: 10).
        consolidation_interval: How often (in cycles) to consolidate knowledge
            (default: 50).
    """

    def __init__(
        self,
        learners: Dict[str, "NestedLearning"],
        workflow_discovery: "WorkflowDiscovery",
        cooperative_evolution: Optional["CooperativeEvolution"] = None,
        meta_update_interval: int = 10,
        consolidation_interval: int = 50,
    ) -> None:
        self.learners = learners
        self.workflow_discovery = workflow_discovery
        self.cooperative_evolution = cooperative_evolution
        self.meta_update_interval = meta_update_interval
        self.consolidation_interval = consolidation_interval
        self.cycle_count: int = 0
        self.logger = logging.getLogger("athena.core.learning_hook")

    async def on_cycle_complete(
        self, agent_actions: dict, context: dict
    ) -> None:
        """
        Execute all learning and evolution updates for a completed cycle.

        Args:
            agent_actions: Mapping of agent name → ``AgentAction`` (or any
                object with ``action_type``, ``success``, ``result``, and
                ``duration`` attributes).
            context: Cycle context dict; expected keys: ``"task"`` (str) and
                ``"cycle"`` (int).
        """
        from learning.nested_learning import TaskTrajectory

        self.cycle_count += 1
        task_label = context.get("task", "unknown")
        self.logger.debug("LearningHook cycle %d: task=%s", self.cycle_count, task_label)

        # ------------------------------------------------------------------
        # Inner loop: per-agent trajectory + adapt
        # ------------------------------------------------------------------
        trajectories: Dict[str, "TaskTrajectory"] = {}

        for agent_name, action in agent_actions.items():
            learner = self.learners.get(agent_name)
            if learner is None:
                continue

            # Compute reward; use confidence weighting when available
            success = getattr(action, "success", True)
            reward = 1.0 if success else -0.5
            result = getattr(action, "result", None)
            if isinstance(result, dict):
                conf = result.get("confidence", None)
                if conf is not None:
                    reward *= float(conf)

            action_type = getattr(action, "action_type", "unknown")
            duration = getattr(action, "duration", 0.0)

            trajectory = TaskTrajectory(
                task_id=task_label,
                agent_id=agent_name,
                states=[{"task": task_label, "cycle": self.cycle_count}],
                actions=[{"type": action_type, "success": success}],
                rewards=[reward],
                metadata={"task": task_label, "duration": duration},
            )
            trajectories[agent_name] = trajectory

            try:
                await learner.adapt_to_task(task_label, trajectory)
            except Exception as exc:
                self.logger.warning(
                    "adapt_to_task failed for agent %s: %s", agent_name, exc
                )

        # ------------------------------------------------------------------
        # Outer loop: meta-parameter update every meta_update_interval cycles
        # ------------------------------------------------------------------
        if self.cycle_count % self.meta_update_interval == 0:
            self.logger.debug(
                "Outer-loop meta-update at cycle %d", self.cycle_count
            )
            for agent_name, learner in self.learners.items():
                all_trajs = []
                for trajs in learner.task_trajectories.values():
                    all_trajs.extend(trajs[-10:])  # recent 10 per task
                if all_trajs:
                    try:
                        await learner.update_meta_parameters(all_trajs)
                    except Exception as exc:
                        self.logger.warning(
                            "update_meta_parameters failed for agent %s: %s",
                            agent_name,
                            exc,
                        )

        # ------------------------------------------------------------------
        # Knowledge consolidation every consolidation_interval cycles
        # ------------------------------------------------------------------
        if self.cycle_count % self.consolidation_interval == 0:
            self.logger.debug(
                "Knowledge consolidation at cycle %d", self.cycle_count
            )
            for agent_name, learner in self.learners.items():
                try:
                    await learner.consolidate_knowledge()
                except Exception as exc:
                    self.logger.warning(
                        "consolidate_knowledge failed for agent %s: %s",
                        agent_name,
                        exc,
                    )

        # ------------------------------------------------------------------
        # Workflow discovery: analyze execution trace
        # ------------------------------------------------------------------
        outcome_success = all(
            getattr(a, "success", True)
            for a in agent_actions.values()
        )
        execution_trace = {
            "agents": list(agent_actions.keys()),
            "interactions": [
                {
                    "sender": agent_name,
                    "recipient": "coordinator",
                    "message_type": "recommendation",
                }
                for agent_name in agent_actions
            ],
            "outcome": {"success": outcome_success},
            "metadata": {"task": task_label, "cycle": self.cycle_count},
        }
        try:
            await self.workflow_discovery.analyze_execution(execution_trace)
        except Exception as exc:
            self.logger.warning(
                "workflow_discovery.analyze_execution failed: %s", exc
            )

        # ------------------------------------------------------------------
        # Cooperative evolution: add per-agent experience
        # ------------------------------------------------------------------
        if self.cooperative_evolution is not None:
            from evolution.cooperative_evolution import Experience

            for agent_name, action in agent_actions.items():
                success = getattr(action, "success", True)
                reward = 1.0 if success else -0.5
                result = getattr(action, "result", None)
                if isinstance(result, dict):
                    conf = result.get("confidence", None)
                    if conf is not None:
                        reward *= float(conf)

                experience = Experience(
                    agent_id=agent_name,
                    state={"task": task_label, "cycle": self.cycle_count},
                    action={
                        "type": getattr(action, "action_type", "unknown"),
                        "success": success,
                    },
                    outcome={"result": result} if result is not None else {},
                    reward=reward,
                )
                try:
                    await self.cooperative_evolution.add_experience(
                        agent_name, experience
                    )
                except Exception as exc:
                    self.logger.warning(
                        "cooperative_evolution.add_experience failed for agent %s: %s",
                        agent_name,
                        exc,
                    )
