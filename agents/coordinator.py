"""
Coordinator Agent
=================
Orchestrates all ATHENA agents, resolves conflicts, allocates resources,
and makes final trading decisions.
"""

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from dataclasses import dataclass
import logging

from core.base_agent import BaseAgent, AgentContext, AgentAction
from core.config import get_default_agent_configs

if TYPE_CHECKING:
    from memory.agemem import AgeMem
    from communication.latent_space import LatentSpace
    from communication.router import MessageRouter


@dataclass
class ConflictResolution:
    """Result of conflict resolution between agent recommendations."""
    decision: str  # buy, sell, hold
    confidence: float
    conflicts_detected: List[str]
    resolution_method: str
    winning_agent: Optional[str] = None


@dataclass
class ResourceAllocation:
    """Resource allocation decision."""
    agent_name: str
    allocated_resources: Dict[str, Any]
    priority: int
    allocation_method: str


class CoordinatorAgent(BaseAgent):
    """
    Coordinator Agent for ATHENA multi-agent system.

    Responsibilities:
    - Orchestrate all other agents in the system
    - Collect and aggregate agent recommendations
    - Resolve conflicts between agent decisions
    - Allocate resources across agents
    - Make final trading decisions based on aggregated intelligence
    """

    def __init__(
        self,
        name: str = "coordinator",
        role: str = "coordinator",
        system_prompt: Optional[str] = None,
        model: Optional[Any] = None,
        memory: Optional["AgeMem"] = None,
        communication: Optional["LatentSpace"] = None,
        router: Optional["MessageRouter"] = None,
        tools: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Coordinator Agent.

        Args:
            name: Agent identifier
            role: Agent role
            system_prompt: System prompt defining behavior
            model: Language model instance
            memory: AgeMem memory instance
            communication: LatentSpace communication instance
            router: MessageRouter for LatentMAS routing (optional)
            tools: List of available tool names
            config: Additional configuration
        """
        if system_prompt is None:
            default_configs = get_default_agent_configs()
            system_prompt = default_configs["coordinator"].system_prompt

        super().__init__(
            name=name,
            role=role,
            system_prompt=system_prompt,
            model=model,
            memory=memory,
            communication=communication,
            tools=tools or ["delegate_task", "resolve_conflict", "allocate_resources"],
            config=config,
        )

        self.memory: Optional["AgeMem"] = memory
        self.router: Optional["MessageRouter"] = router
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_priority = {
            "risk": 3,
            "strategy": 2,
            "analyst": 1,
            "execution": 1,
        }

    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """
        Register an agent for coordination.

        Args:
            name: Unique agent identifier
            agent: BaseAgent instance to register
        """
        self.agents[name] = agent
        self.logger.info("Registered agent: %s (%s)", name, agent.role)

    async def initialize_communication(self, latent_space: "LatentSpace") -> None:
        """Initialize LatentMAS communication infrastructure."""
        from communication.router import MessageRouter, MessagePriority
        from communication.encoder import AgentStateEncoder
        from communication.decoder import AgentStateDecoder

        self.communication = latent_space
        # Create encoder/decoder with latent_dim derived from the latent space
        latent_dim = latent_space.latent_dim
        encoder = AgentStateEncoder(latent_dim=latent_dim, input_dim=512)
        decoder = AgentStateDecoder(latent_dim=latent_dim, output_dim=512)
        # Create router
        self.router = MessageRouter(
            latent_space=latent_space,
            encoder=encoder,
            decoder=decoder,
            config={"enable_priority": True, "enable_attention": False},
        )
        # Register all known agents
        for agent_name in self.agents:
            self.router.register_agent(agent_name, {"role": self.agents[agent_name].role})
        # Pre-register agents in LatentSpace so broadcasts reach them
        for agent_name in self.agents:
            latent_space.register_agent(agent_name)
        self.logger.info(
            f"Communication initialized with {len(self.agents)} registered agents"
        )

    async def think(self, context: AgentContext) -> Dict[str, Any]:
        """
        Orchestrate agents and build coordination plan.

        Args:
            context: Current context including task, history, memory, messages

        Returns:
            Dictionary with orchestration plan, agent recommendations, conflicts
        """
        self.logger.info("Coordinating agents for task: %s", context.task)

        # Retrieve relevant memory context
        memory_context = []
        if self.memory:
            try:
                task_info = context.metadata.get("task", {})
                task_type = task_info.get("type", "") if isinstance(task_info, dict) else ""
                memory_context = await self.memory.retrieve(
                    query=f"coordination task {task_type}",
                    top_k=5
                )
            except Exception as e:
                self.logger.warning("Memory retrieve failed: %s", e)

        # Receive any LatentMAS messages from specialist agents
        latent_messages = []
        if self.router:
            try:
                latent_messages = await self.router.receive(
                    receiver_id=self.name,
                    decode_mode="structured",
                )
            except Exception as e:
                self.logger.warning("LatentMAS receive failed: %s", e)

        orchestration_plan = {
            "task": context.task,
            "timestamp": time.time(),
            "agent_recommendations": {},
            "conflicts": [],
            "resource_requests": {},
            "coordination_strategy": "priority_weighted",
            "done": False,
        }

        recommendations = {}
        for message in context.messages:
            if message.message_type == "recommendation":
                recommendations[message.sender] = message.content
                self.logger.info(
                    f"Received recommendation from {message.sender}: "
                    f"{message.content.get('action', 'unknown')}"
                )

        orchestration_plan["agent_recommendations"] = recommendations

        if len(recommendations) >= 2:
            conflicts = self._detect_conflicts(recommendations)
            orchestration_plan["conflicts"] = conflicts

            if conflicts:
                self.logger.warning(
                    f"Detected {len(conflicts)} conflicts between agents"
                )

        resource_requests = {}
        for message in context.messages:
            if message.message_type == "resource_request":
                resource_requests[message.sender] = message.content

        orchestration_plan["resource_requests"] = resource_requests

        if recommendations or resource_requests:
            orchestration_plan["done"] = True

        # Optional LLM synthesis for decision reasoning
        if recommendations:
            llm_synthesis = await self._llm_reason(
                f"Given these agent recommendations: {recommendations}, "
                f"what is the best trading decision?"
            )
            if llm_synthesis:
                orchestration_plan["llm_synthesis"] = llm_synthesis

        orchestration_plan["memory_context"] = memory_context
        orchestration_plan["latent_messages"] = latent_messages
        return orchestration_plan

    async def act(self, thought: Dict[str, Any]) -> AgentAction:
        """
        Execute orchestration plan and make final decision.

        Args:
            thought: Orchestration plan from think()

        Returns:
            AgentAction with final trading decision and coordination results
        """
        start_time = time.time()

        try:
            recommendations = thought.get("agent_recommendations", {})
            conflicts = thought.get("conflicts", [])
            resource_requests = thought.get("resource_requests", {})

            result = {
                "final_decision": None,
                "conflict_resolution": None,
                "resource_allocation": None,
                "coordination_status": "success",
            }

            if resource_requests:
                allocations = await self._allocate_resources(resource_requests)
                result["resource_allocation"] = allocations
                self.logger.info(
                    f"Allocated resources to {len(allocations)} agents"
                )

            if recommendations:
                if conflicts:
                    resolved = await self._resolve_conflicts(recommendations)
                    result["conflict_resolution"] = {
                        "decision": resolved["decision"],
                        "confidence": resolved["confidence"],
                        "conflicts": conflicts,
                        "method": resolved["method"],
                        "winning_agent": resolved.get("winning_agent"),
                    }
                else:
                    actions: Dict[str, int] = {}
                    total_confidence = 0.0
                    count = 0
                    for rec in recommendations.values():
                        if isinstance(rec, dict) and "action" in rec:
                            action = rec["action"]
                            actions[action] = actions.get(action, 0) + 1
                            total_confidence += rec.get("confidence", 0.5)
                            count += 1

                    if actions:
                        best_action = max(actions, key=actions.get)
                        avg_confidence = total_confidence / count if count > 0 else 0.5
                        resolved = {
                            "decision": best_action,
                            "confidence": avg_confidence,
                            "method": "unanimous" if len(actions) == 1 else "majority",
                        }
                    else:
                        resolved = {"decision": "hold", "confidence": 0.0, "method": "no_input"}

                risk_assessment = recommendations.get("risk_manager", {})
                final_decision = await self._make_final_decision(
                    resolved, risk_assessment
                )
                result["final_decision"] = final_decision

                # Broadcast final decision via LatentMAS
                if self.router and result.get("final_decision"):
                    try:
                        from communication.router import MessagePriority
                        for agent_name in self.agents:
                            await self.router.send(
                                sender_id=self.name,
                                receiver_id=agent_name,
                                message=result["final_decision"],
                                priority=MessagePriority.HIGH,
                            )
                        self.logger.info(
                            f"Broadcast final decision to {len(self.agents)} agents via LatentMAS"
                        )
                    except Exception as e:
                        self.logger.warning("LatentMAS broadcast failed: %s", e)

                if self.communication:
                    await self.send_message(
                        recipient="*",
                        content=final_decision,
                        message_type="final_decision",
                        priority=3,
                    )
                    self.logger.info("Broadcast final decision to all agents")

            duration = time.time() - start_time

            # Store coordination result and summary to memory in a single write
            if self.memory:
                try:
                    await self.memory.add(
                        content={
                            "coordination": result,
                            "thought": thought,
                            "final_decision": result.get("final_decision"),
                            "agents_queried": list(self.agents.keys()),
                        },
                        metadata={
                            "agent": self.name,
                            "role": self.role,
                            "success": True,
                            "operation": "coordination_summary",
                        }
                    )
                except Exception as e:
                    self.logger.warning("Memory store failed: %s", e)

            return AgentAction(
                action_type="coordination",
                parameters={"task": thought.get("task")},
                result=result,
                success=True,
                error=None,
                duration=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error("Error in act(): %s", e)

            # Store failure to memory
            if self.memory:
                try:
                    await self.memory.add(
                        content={"coordination": None, "thought": thought},
                        metadata={
                            "agent": self.name,
                            "role": self.role,
                            "success": False,
                        }
                    )
                except Exception as mem_e:
                    self.logger.warning("Memory store failed: %s", mem_e)

            return AgentAction(
                action_type="coordination",
                parameters={"task": thought.get("task")},
                result=None,
                success=False,
                error=str(e),
                duration=duration,
            )

    def _detect_conflicts(self, recommendations: Dict[str, Dict]) -> List[str]:
        """
        Detect conflicts between agent recommendations.

        Args:
            recommendations: Dictionary of agent recommendations

        Returns:
            List of conflict descriptions
        """
        conflicts = []
        actions = {}

        for agent_name, rec in recommendations.items():
            if isinstance(rec, dict) and "action" in rec:
                action = rec["action"]
                if action not in actions:
                    actions[action] = []
                actions[action].append(agent_name)

        if len(actions) > 1:
            action_list = list(actions.keys())
            for i, action1 in enumerate(action_list):
                for action2 in action_list[i + 1 :]:
                    if action1 != action2:
                        if (action1 == "buy" and action2 == "sell") or (
                            action1 == "sell" and action2 == "buy"
                        ):
                            conflicts.append(
                                f"Direct conflict: {', '.join(actions[action1])} "
                                f"recommends {action1}, but "
                                f"{', '.join(actions[action2])} recommends {action2}"
                            )

        return conflicts

    async def _resolve_conflicts(
        self, recommendations: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """
        Resolve conflicting recommendations using priority weighting.

        Priority order: risk > strategy > analyst > execution

        Args:
            recommendations: Dictionary of agent recommendations

        Returns:
            Dictionary with resolved decision, confidence, and method
        """
        self.logger.info("Resolving conflicts using priority weighting")

        weighted_votes = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        total_weight = 0.0

        for agent_name, rec in recommendations.items():
            if not isinstance(rec, dict) or "action" not in rec:
                continue

            agent_role = None
            if agent_name in self.agents:
                agent_role = self.agents[agent_name].role
            else:
                for role in self.agent_priority.keys():
                    if role in agent_name:
                        agent_role = role
                        break

            priority = self.agent_priority.get(agent_role, 1)
            action = rec["action"]
            confidence = rec.get("confidence", 0.5)
            weight = priority * confidence

            if action in weighted_votes:
                weighted_votes[action] += weight
                total_weight += weight

        if total_weight == 0:
            return {
                "decision": "hold",
                "confidence": 0.0,
                "method": "priority_weighted",
            }

        decision = max(weighted_votes, key=weighted_votes.get)

        # Find the agent that contributed most to the winning action
        winning_agent = None
        max_weight = -1
        for agent_name, rec in recommendations.items():
            if not isinstance(rec, dict) or "action" not in rec:
                continue
            if rec.get("action") == decision:
                agent_role = None
                if agent_name in self.agents:
                    agent_role = self.agents[agent_name].role
                else:
                    for role in self.agent_priority.keys():
                        if role in agent_name:
                            agent_role = role
                            break
                priority = self.agent_priority.get(agent_role, 1)
                weight = priority * rec.get("confidence", 0.5)
                if weight > max_weight:
                    max_weight = weight
                    winning_agent = agent_name

        confidence = weighted_votes[decision] / total_weight

        return {
            "decision": decision,
            "confidence": confidence,
            "method": "priority_weighted",
            "winning_agent": winning_agent,
            "vote_distribution": weighted_votes,
        }

    async def _allocate_resources(
        self, requests: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        Allocate resources to agents using demand-proportional allocation.

        Args:
            requests: Dictionary of resource requests from agents

        Returns:
            Dictionary mapping agent names to allocated resources
        """
        self.logger.info("Allocating resources for %d requests", len(requests))

        allocations = {}
        total_requests = {}

        for agent_name, request in requests.items():
            if isinstance(request, dict):
                for resource_type, amount in request.items():
                    if resource_type not in total_requests:
                        total_requests[resource_type] = 0.0
                    total_requests[resource_type] += amount

        for agent_name, request in requests.items():
            if not isinstance(request, dict):
                continue

            allocated = {}
            for resource_type, requested_amount in request.items():
                if total_requests[resource_type] > 0:
                    share = requested_amount / total_requests[resource_type]
                    allocated[resource_type] = share
                else:
                    allocated[resource_type] = 0.0

            allocations[agent_name] = {
                "allocated_resources": allocated,
                "method": "proportional",
            }

        return allocations

    async def _make_final_decision(
        self, resolved: Dict, risk_assessment: Dict
    ) -> Dict[str, Any]:
        """
        Combine resolved recommendations with risk assessment into final decision.

        Args:
            resolved: Resolved recommendation from conflict resolution
            risk_assessment: Risk manager's assessment

        Returns:
            Final trading decision with reasoning
        """
        decision = resolved.get("decision", "hold")
        confidence = resolved.get("confidence", 0.0)

        risk_level = "unknown"
        if isinstance(risk_assessment, dict):
            risk_level = risk_assessment.get("risk_level", "unknown")
            compliance = risk_assessment.get("compliance_violations", [])

            if compliance:
                self.logger.warning(
                    f"Compliance violations detected: {compliance}"
                )
                decision = "hold"
                confidence = 0.0

            if risk_level == "high" and decision == "buy":
                self.logger.warning(
                    "High risk detected - downgrading buy to hold"
                )
                decision = "hold"
                confidence *= 0.5

        final_decision = {
            "action": decision,
            "confidence": confidence,
            "risk_level": risk_level,
            "timestamp": time.time(),
            "reasoning": self._build_reasoning(resolved, risk_assessment),
        }

        self.logger.info(
            f"Final decision: {decision} (confidence: {confidence:.2f}, "
            f"risk: {risk_level})"
        )

        return final_decision

    def _build_reasoning(self, resolved: Dict, risk_assessment: Dict) -> str:
        """
        Build human-readable reasoning for the final decision.

        Args:
            resolved: Resolved recommendation
            risk_assessment: Risk assessment data

        Returns:
            Reasoning string
        """
        parts = []

        method = resolved.get("method", "unknown")
        decision = resolved.get("decision", "hold")
        parts.append(f"Decision '{decision}' reached via {method}")

        if "vote_distribution" in resolved:
            votes = resolved["vote_distribution"]
            parts.append(
                f"Vote distribution: buy={votes.get('buy', 0):.2f}, "
                f"sell={votes.get('sell', 0):.2f}, hold={votes.get('hold', 0):.2f}"
            )

        if isinstance(risk_assessment, dict):
            risk_level = risk_assessment.get("risk_level", "unknown")
            parts.append(f"Risk level: {risk_level}")

            if "var_95" in risk_assessment:
                parts.append(f"VaR (95%): {risk_assessment['var_95']:.4f}")

        return "; ".join(parts)

    def _get_tool_definition(self, tool_name: str) -> Dict[str, Any]:
        """Get tool definition for coordinator tools."""
        tools = {
            "delegate_task": {
                "name": "delegate_task",
                "description": "Delegate a task to a specific agent",
                "parameters": {
                    "agent_name": {"type": "string", "description": "Target agent"},
                    "task": {"type": "string", "description": "Task description"},
                },
            },
            "resolve_conflict": {
                "name": "resolve_conflict",
                "description": "Resolve conflicts between agent recommendations",
                "parameters": {
                    "recommendations": {
                        "type": "object",
                        "description": "Agent recommendations",
                    },
                },
            },
            "allocate_resources": {
                "name": "allocate_resources",
                "description": "Allocate resources across agents",
                "parameters": {
                    "requests": {
                        "type": "object",
                        "description": "Resource requests",
                    },
                },
            },
        }
        return tools.get(tool_name, super()._get_tool_definition(tool_name))
