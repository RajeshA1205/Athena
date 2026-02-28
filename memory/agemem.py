"""
AgeMem: Unified Memory Management
=================================
Main controller for AgeMem memory layer implementing:
- LTM operations: ADD, UPDATE, DELETE
- STM operations: RETRIEVE, SUMMARY, FILTER

Uses Graphiti (Zep) as the storage backend.
Supports Step-wise GRPO training for memory management optimization.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

from .graphiti_backend import GraphitiBackend
from .operations import LTMOperations, STMOperations


class MemoryOperation(Enum):
    """Available memory operations."""
    # LTM operations
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    # STM operations
    RETRIEVE = "retrieve"
    SUMMARY = "summary"
    FILTER = "filter"


@dataclass
class MemoryOperationResult:
    """Result of a memory operation."""
    success: bool
    operation: MemoryOperation
    data: Any = None
    error: Optional[str] = None
    duration: float = 0.0


class MemoryInterface(ABC):
    """Abstract interface for memory operations."""

    @abstractmethod
    async def add(self, content: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add new memory entry to LTM."""
        pass

    @abstractmethod
    async def update(self, entry_id: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update existing memory entry in LTM."""
        pass

    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """Delete memory entry from LTM."""
        pass

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant memories from LTM to STM."""
        pass

    @abstractmethod
    async def summary(self, context: List[Dict[str, Any]]) -> str:
        """Summarize conversation history in STM."""
        pass

    @abstractmethod
    async def filter(self, context: List[Dict[str, Any]], relevance_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Filter irrelevant information from STM context."""
        pass


class AgeMem(MemoryInterface):
    """
    AgeMem: Unified Memory Management System.

    Architecture:
    ┌─────────────────────────────────────┐
    │           AgeMem (this class)       │  ← Logical operations
    ├─────────────────────────────────────┤
    │    LTMOperations  │  STMOperations  │  ← Operation implementations
    ├─────────────────────────────────────┤
    │          GraphitiBackend            │  ← Storage layer
    └─────────────────────────────────────┘

    Operations:
    - LTM: ADD, UPDATE, DELETE (persistent storage via Graphiti)
    - STM: RETRIEVE, SUMMARY, FILTER (working memory management)
    """

    def __init__(
        self,
        backend: Optional[GraphitiBackend] = None,
        model: Optional[Any] = None,
        buffer_size: int = 20,
        context_window: int = 4096,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize AgeMem.

        Args:
            backend: Graphiti backend instance (created if not provided)
            model: Language model for summary/filtering operations
            buffer_size: STM buffer size
            context_window: Maximum context window size
            config: Additional configuration
        """
        self.config = config or {}
        self.model = model
        self.logger = logging.getLogger("athena.memory.agemem")

        # Initialize backend
        self.backend = backend or GraphitiBackend(
            neo4j_uri=self.config.get("neo4j_uri", "bolt://localhost:7687"),
            neo4j_user=self.config.get("neo4j_user", "neo4j"),
            neo4j_password=self.config.get("neo4j_password", "password"),
        )

        # Initialize operations
        self.ltm = LTMOperations(backend=self.backend, config=config)
        self.stm = STMOperations(
            backend=self.backend,
            buffer_size=buffer_size,
            context_window=context_window,
            model=model,
            config=config,
        )

        # Operation statistics for reward calculation
        self.operation_stats: Dict[str, Dict[str, Any]] = {
            op.value: {"count": 0, "success": 0, "total_time": 0.0}
            for op in MemoryOperation
        }

        # Configurable placeholder quality rewards (no empirical basis yet)
        self._quality_rewards = {
            "summary": float(self.config.get("summary_quality_reward", 0.8)),
            "filter": float(self.config.get("filter_quality_reward", 0.9)),
        }

        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the memory system."""
        if self._initialized:
            return True

        success = await self.backend.initialize()
        self._initialized = success
        self.logger.info("AgeMem initialized")
        return success

    # ==================== LTM Operations ====================

    async def add(self, content: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        ADD: Store new memory in long-term memory.

        Args:
            content: Content to store
            metadata: Optional metadata (source, importance, tags, etc.)

        Returns:
            True if stored successfully
        """
        import time
        start = time.perf_counter()

        if not self._initialized:
            await self.initialize()

        try:
            result = await self.ltm.add(content, metadata)
            success = result is not None
            self._update_stats(MemoryOperation.ADD, success, time.perf_counter() - start)
            self.logger.debug(f"ADD: {str(content)[:50]}... -> {result}")
            return success
        except Exception as e:
            self._update_stats(MemoryOperation.ADD, False, time.perf_counter() - start)
            self.logger.error(f"ADD failed: {e}")
            return False

    async def update(self, entry_id: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        UPDATE: Modify existing memory in long-term memory.

        Args:
            entry_id: ID of the entry to update
            content: New content
            metadata: Optional updated metadata

        Returns:
            True if updated successfully
        """
        import time
        start = time.perf_counter()

        if not self._initialized:
            await self.initialize()

        try:
            result = await self.ltm.update(entry_id, content, metadata)
            self._update_stats(MemoryOperation.UPDATE, result, time.perf_counter() - start)
            self.logger.debug(f"UPDATE: {entry_id}")
            return result
        except Exception as e:
            self._update_stats(MemoryOperation.UPDATE, False, time.perf_counter() - start)
            self.logger.error(f"UPDATE failed: {e}")
            return False

    async def delete(self, entry_id: str) -> bool:
        """
        DELETE: Remove memory from long-term memory.

        Args:
            entry_id: ID of the entry to delete

        Returns:
            True if deleted successfully
        """
        import time
        start = time.perf_counter()

        if not self._initialized:
            await self.initialize()

        try:
            result = await self.ltm.delete(entry_id)
            self._update_stats(MemoryOperation.DELETE, result, time.perf_counter() - start)
            self.logger.debug(f"DELETE: {entry_id}")
            return result
        except Exception as e:
            self._update_stats(MemoryOperation.DELETE, False, time.perf_counter() - start)
            self.logger.error(f"DELETE failed: {e}")
            return False

    # ==================== STM Operations ====================

    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        RETRIEVE: Fetch relevant context from LTM to STM.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant memory entries
        """
        import time
        start = time.perf_counter()

        if not self._initialized:
            await self.initialize()

        try:
            results = await self.stm.retrieve(query, top_k)
            self._update_stats(MemoryOperation.RETRIEVE, True, time.perf_counter() - start)
            self.logger.debug(f"RETRIEVE: {len(results)} items for '{query[:30]}...'")
            return results
        except Exception as e:
            self._update_stats(MemoryOperation.RETRIEVE, False, time.perf_counter() - start)
            self.logger.error(f"RETRIEVE failed: {e}")
            return []

    async def summary(self, context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        SUMMARY: Compress conversation history in STM.

        Args:
            context: Optional context to summarize (uses STM buffer if not provided)

        Returns:
            Summarized text
        """
        import time
        start = time.perf_counter()

        if not self._initialized:
            await self.initialize()

        try:
            result = await self.stm.summary(context)
            self._update_stats(MemoryOperation.SUMMARY, True, time.perf_counter() - start)
            return result
        except Exception as e:
            self._update_stats(MemoryOperation.SUMMARY, False, time.perf_counter() - start)
            self.logger.error(f"SUMMARY failed: {e}")
            return ""

    async def filter(
        self,
        context: Optional[List[Dict[str, Any]]] = None,
        relevance_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        FILTER: Remove irrelevant information from STM context.

        Args:
            context: Optional context to filter (uses STM buffer if not provided)
            relevance_threshold: Minimum relevance score to keep

        Returns:
            Filtered context
        """
        import time
        start = time.perf_counter()

        if not self._initialized:
            await self.initialize()

        try:
            result = await self.stm.filter(context, relevance_threshold)
            self._update_stats(MemoryOperation.FILTER, True, time.perf_counter() - start)
            return result
        except Exception as e:
            self._update_stats(MemoryOperation.FILTER, False, time.perf_counter() - start)
            self.logger.error(f"FILTER failed: {e}")
            return context or []

    # ==================== Combined Operations ====================

    async def process_query(self, query: str, summarize: bool = True) -> Tuple[List[Dict[str, Any]], str]:
        """
        Process a query with retrieve, filter, and optional summary.

        Args:
            query: Search query
            summarize: Whether to summarize the context

        Returns:
            Tuple of (filtered_context, summary)
        """
        # Retrieve from LTM via STM
        context = await self.retrieve(query)

        # Filter irrelevant
        filtered = await self.filter(context)

        # Optionally summarize
        summary_text = ""
        if summarize and filtered:
            summary_text = await self.summary(filtered)

        return filtered, summary_text

    # ==================== Training Support (Step-wise GRPO) ====================

    def get_operation_reward(self, operation: MemoryOperation, result: MemoryOperationResult) -> float:
        """
        Calculate reward for a memory operation (for RL training).

        Based on AgeMem's composite reward function:
        R = α * R_task + β * R_efficiency + γ * R_quality

        Args:
            operation: The operation performed
            result: The operation result

        Returns:
            Reward value
        """
        alpha = self.config.get("reward_alpha", 0.5)
        beta = self.config.get("reward_beta", 0.3)
        gamma = self.config.get("reward_gamma", 0.2)

        # Task reward: operation success
        r_task = 1.0 if result.success else -0.5

        # Efficiency reward: operation speed
        avg_time = self._get_avg_operation_time(operation)
        if avg_time > 0:
            r_efficiency = min(1.0, avg_time / max(result.duration, 0.001))
        else:
            r_efficiency = 1.0

        # Quality reward: based on operation type
        r_quality = self._calculate_quality_reward(operation, result)

        return alpha * r_task + beta * r_efficiency + gamma * r_quality

    def get_trajectory_for_training(self) -> List[Dict[str, Any]]:
        """
        Get operation trajectory for GRPO training.

        Returns:
            List of operation records with states, actions, rewards
        """
        # This will be populated by the training module
        return []

    def _calculate_quality_reward(self, operation: MemoryOperation, result: MemoryOperationResult) -> float:
        """
        Calculate quality component of the GRPO reward signal.

        NOTE: SUMMARY and FILTER rewards are configurable placeholder values.
        These should be replaced with learned metrics once training data is available:
        - SUMMARY: should reflect compression quality (e.g. ROUGE score vs. original)
        - FILTER: should reflect noise reduction (e.g. relevance improvement post-filter)
        Configure via AgeMem config keys 'summary_quality_reward' and 'filter_quality_reward'.
        See training/stage2_agemem/ for the reward model training infrastructure.
        """
        if operation == MemoryOperation.RETRIEVE:
            if result.data and isinstance(result.data, list):
                return min(1.0, len(result.data) / 5)
        elif operation == MemoryOperation.SUMMARY:
            return self._quality_rewards.get("summary", 0.8)  # placeholder
        elif operation == MemoryOperation.FILTER:
            return self._quality_rewards.get("filter", 0.9)  # placeholder
        return 1.0 if result.success else 0.0

    def _get_avg_operation_time(self, operation: MemoryOperation) -> float:
        """Get average time for an operation type."""
        stats = self.operation_stats[operation.value]
        if stats["count"] > 0:
            return stats["total_time"] / stats["count"]
        return 0.0

    def _update_stats(self, operation: MemoryOperation, success: bool, duration: float) -> None:
        """Update operation statistics."""
        stats = self.operation_stats[operation.value]
        stats["count"] += 1
        stats["total_time"] += duration
        if success:
            stats["success"] += 1

    # ==================== State Management ====================

    async def clear_stm(self) -> None:
        """Clear short-term memory buffer."""
        await self.stm.clear()
        self.logger.info("STM cleared")

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        backend_stats = await self.backend.get_stats()
        stm_stats = await self.stm.get_stats()

        return {
            "backend": backend_stats,
            "stm": stm_stats,
            "operations": self.operation_stats,
            "initialized": self._initialized,
        }

    def reset_stats(self) -> None:
        """Reset operation statistics."""
        for op in MemoryOperation:
            self.operation_stats[op.value] = {"count": 0, "success": 0, "total_time": 0.0}

    async def close(self) -> None:
        """Close memory system."""
        await self.backend.close()
        self._initialized = False
        self.logger.info("AgeMem closed")
