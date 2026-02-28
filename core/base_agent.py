"""
ATHENA Base Agent
=================
Abstract base class for all ATHENA agents.
"""

import asyncio
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Deque, Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
import logging

if TYPE_CHECKING:
    from athena.memory.agemem import AgeMem
    from athena.communication.latent_space import LatentSpace
    from models.olmoe import OLMoEModel


class AgentState(Enum):
    """Agent lifecycle states."""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING = "waiting"
    ERROR = "error"


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication."""
    sender: str
    recipient: str  # Use "*" for broadcast
    content: Any
    message_type: str = "default"
    priority: int = 1
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentAction:
    """Represents an action taken by an agent."""
    action_type: str
    parameters: Dict[str, Any]
    result: Optional[Any] = None
    success: bool = True
    error: Optional[str] = None
    duration: float = 0.0


@dataclass
class AgentContext:
    """Context information for agent decision making."""
    task: str
    history: List[Dict[str, Any]] = field(default_factory=list)
    memory_context: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[AgentMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all ATHENA agents.

    All specialized agents (Market Analyst, Risk Manager, Strategy,
    Execution, Coordinator) inherit from this class.
    """

    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        model: Optional[Any] = None,
        memory: Optional["AgeMem"] = None,
        communication: Optional["LatentSpace"] = None,
        tools: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        llm: Optional["OLMoEModel"] = None,
    ):
        """
        Initialize the base agent.

        Args:
            name: Unique agent identifier
            role: Agent role (analyst, risk, strategy, execution, coordinator)
            system_prompt: System prompt defining agent behavior
            model: Language model instance (OLMo 3)
            memory: AgeMem memory instance
            communication: LatentMAS communication instance
            tools: List of available tool names
            config: Additional configuration
        """
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.model = model
        self.memory = memory
        self.communication = communication
        self.tools = tools or []
        self.config = config or {}

        self.llm = llm  # OLMoEModel instance or None
        self.state = AgentState.IDLE
        self.action_history: Deque[AgentAction] = deque(maxlen=100)
        self.logger = logging.getLogger(f"athena.agent.{name}")

    @abstractmethod
    async def think(self, context: AgentContext) -> Dict[str, Any]:
        """
        Process context and decide on action.

        Args:
            context: Current context including task, history, memory, messages

        Returns:
            Dictionary containing thought process and planned action
        """
        pass

    @abstractmethod
    async def act(self, thought: Dict[str, Any]) -> AgentAction:
        """
        Execute the planned action.

        Args:
            thought: Output from think() method

        Returns:
            AgentAction describing the action taken and its result
        """
        pass

    async def run(self, task: str, max_iterations: int = 10) -> List[AgentAction]:
        """
        Main agent loop: observe -> think -> act.

        Args:
            task: The task to accomplish
            max_iterations: Maximum number of think-act cycles

        Returns:
            List of actions taken
        """
        self.logger.info(f"Starting task: {task}")
        actions = []

        for i in range(max_iterations):
            self.state = AgentState.THINKING

            # Build context
            context = await self._build_context(task)

            # Think
            thought = await self.think(context)

            # Check if done
            if thought.get("done", False):
                self.logger.info(f"Task completed after {i + 1} iterations")
                break

            # Act
            self.state = AgentState.ACTING
            action = await self.act(thought)
            actions.append(action)
            self.action_history.append(action)

            if not action.success:
                self.logger.warning(f"Action failed: {action.error}")
                self.state = AgentState.ERROR
                break

        self.state = AgentState.IDLE
        return actions

    async def _llm_reason(self, prompt: str) -> Optional[str]:
        """Route a reasoning prompt through OLMoE if available, else return None."""
        if self.llm is None:
            return None
        try:
            return await self.llm.generate(prompt)
        except Exception as e:
            self.logger.warning("LLM reasoning failed: %s", e)
            return None

    async def _build_context(self, task: str) -> AgentContext:
        """
        Build context for decision making.

        Args:
            task: Current task

        Returns:
            AgentContext with relevant information
        """
        context = AgentContext(task=task)

        # Get recent action history
        context.history = [
            {
                "action": a.action_type,
                "parameters": a.parameters,
                "result": a.result,
                "success": a.success,
            }
            for a in self.action_history[-10:]
        ]

        # Retrieve memory context if available
        if self.memory is not None:
            context.memory_context = await self.memory.retrieve(task)

        # Get pending messages if communication is available
        if self.communication is not None:
            context.messages = await self.communication.receive(self.name)

        return context

    async def send_message(
        self,
        recipient: str,
        content: Any,
        message_type: str = "default",
        priority: int = 1,
    ) -> bool:
        """
        Send message to another agent.

        Args:
            recipient: Target agent name or "*" for broadcast
            content: Message content
            message_type: Type of message
            priority: Message priority (1-3)

        Returns:
            True if message was sent successfully
        """
        if self.communication is None:
            self.logger.warning("Communication not available")
            return False

        message = AgentMessage(
            sender=self.name,
            recipient=recipient,
            content=content,
            message_type=message_type,
            priority=priority,
        )

        return await self.communication.send(message)

    async def remember(self, content: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store information in long-term memory.

        Args:
            content: Content to remember
            metadata: Optional metadata

        Returns:
            True if stored successfully
        """
        if self.memory is None:
            self.logger.warning("Memory not available")
            return False

        return await self.memory.add(content, metadata)

    async def recall(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve information from memory.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant memory entries
        """
        if self.memory is None:
            self.logger.warning("Memory not available")
            return []

        return await self.memory.retrieve(query, top_k)

    def get_tools(self) -> List[Dict[str, Any]]:
        """
        Get available tools for this agent.

        Returns:
            List of tool definitions
        """
        return [self._get_tool_definition(tool) for tool in self.tools]

    def _get_tool_definition(self, tool_name: str) -> Dict[str, Any]:
        """
        Get tool definition by name.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool definition dictionary
        """
        # This will be overridden by specialized agents
        return {
            "name": tool_name,
            "description": f"Tool: {tool_name}",
            "parameters": {},
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Get current agent state.

        Returns:
            Dictionary with agent state information
        """
        return {
            "name": self.name,
            "role": self.role,
            "state": self.state.value,
            "action_count": len(self.action_history),
            "tools": self.tools,
        }

    def reset(self) -> None:
        """Reset agent state."""
        self.state = AgentState.IDLE
        self.action_history = deque(maxlen=100)
        self.logger.info(f"Agent {self.name} reset")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, role={self.role}, state={self.state.value})>"
