"""
AgeMem Operations
=================
LTM and STM operation implementations for AgeMem.
- LTM: ADD, UPDATE, DELETE
- STM: RETRIEVE, SUMMARY, FILTER
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import deque
import logging

if TYPE_CHECKING:
    from .graphiti_backend import GraphitiBackend


@dataclass
class ContextItem:
    """Item in the STM context buffer."""
    id: str
    content: Any
    source: str  # "ltm_retrieval", "conversation", "external"
    relevance_score: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class LTMOperations:
    """
    Long-Term Memory Operations.

    Wraps Graphiti backend to provide:
    - ADD: Store new memories
    - UPDATE: Modify existing memories
    - DELETE: Remove memories
    """

    def __init__(
        self,
        backend: "GraphitiBackend",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LTM operations.

        Args:
            backend: Graphiti backend instance
            config: Configuration options
        """
        self.backend = backend
        self.config = config or {}
        self.logger = logging.getLogger("athena.memory.ltm")

    async def add(
        self,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "agent",
    ) -> Optional[str]:
        """
        ADD: Store new memory in LTM.

        Args:
            content: Content to store (will be converted to string)
            metadata: Optional metadata (source, importance, tags, etc.)
            source: Source identifier

        Returns:
            Memory ID if successful, None otherwise
        """
        # Convert content to string if needed
        content_str = str(content) if not isinstance(content, str) else content

        # Add to Graphiti
        episode_id = await self.backend.add_episode(
            content=content_str,
            source=source,
            episode_type="text",
            metadata=metadata,
        )

        if episode_id:
            self.logger.debug(f"LTM ADD: {episode_id} - {content_str[:50]}...")

        return episode_id

    async def update(
        self,
        memory_id: str,
        content: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        UPDATE: Modify existing memory in LTM.

        Args:
            memory_id: ID of memory to update
            content: New content (optional)
            metadata: Updated metadata (merged with existing)

        Returns:
            True if successful
        """
        content_str = None
        if content is not None:
            content_str = str(content) if not isinstance(content, str) else content

        success = await self.backend.update_episode(
            episode_id=memory_id,
            content=content_str,
            metadata=metadata,
        )

        if success:
            self.logger.debug(f"LTM UPDATE: {memory_id}")

        return success

    async def delete(self, memory_id: str) -> bool:
        """
        DELETE: Remove memory from LTM.

        Args:
            memory_id: ID of memory to delete

        Returns:
            True if successful
        """
        success = await self.backend.delete_episode(memory_id)

        if success:
            self.logger.debug(f"LTM DELETE: {memory_id}")

        return success

    async def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get memory by ID.

        Args:
            memory_id: Memory ID

        Returns:
            Memory as dictionary or None
        """
        episode = await self.backend.get_episode(memory_id)
        if episode:
            return {
                "id": episode.id,
                "content": episode.content,
                "metadata": episode.metadata,
                "timestamp": episode.timestamp,
            }
        return None


class STMOperations:
    """
    Short-Term Memory Operations.

    Provides working memory management:
    - RETRIEVE: Fetch relevant context from LTM
    - SUMMARY: Compress context
    - FILTER: Remove irrelevant context
    """

    def __init__(
        self,
        backend: "GraphitiBackend",
        buffer_size: int = 20,
        context_window: int = 4096,
        model: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize STM operations.

        Args:
            backend: Graphiti backend instance
            buffer_size: Maximum items in context buffer
            context_window: Maximum tokens in context
            model: LLM for summary/filter operations
            config: Configuration options
        """
        self.backend = backend
        self.buffer_size = buffer_size
        self.context_window = context_window
        self.model = model
        self.config = config or {}

        self.logger = logging.getLogger("athena.memory.stm")

        # Context buffer (recent items at the end)
        self._buffer: deque[ContextItem] = deque(maxlen=buffer_size)

        # Current query context (for relevance scoring)
        self._current_query: Optional[str] = None

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        RETRIEVE: Fetch relevant context from LTM.

        Searches LTM via Graphiti and adds results to STM buffer.

        Args:
            query: Search query
            top_k: Number of results to retrieve

        Returns:
            List of retrieved memories
        """
        self._current_query = query

        # Search Graphiti
        results = await self.backend.search(query=query, top_k=top_k)

        retrieved = []
        for result in results:
            # Create context item
            item = ContextItem(
                id=result.id,
                content=result.content,
                source="ltm_retrieval",
                relevance_score=result.score,
                timestamp=result.timestamp,
                metadata=result.metadata,
            )

            # Add to buffer
            self._buffer.append(item)

            retrieved.append({
                "id": result.id,
                "content": result.content,
                "score": result.score,
                "timestamp": result.timestamp,
                "metadata": result.metadata,
            })

        self.logger.debug(f"STM RETRIEVE: {len(retrieved)} items for '{query[:30]}...'")
        return retrieved

    async def summary(
        self,
        context: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        SUMMARY: Compress context into concise summary.

        Args:
            context: Context to summarize (uses buffer if not provided)

        Returns:
            Summarized text
        """
        if context is None:
            context = await self.get_context()

        if not context:
            return ""

        # If no model, return simple concatenation
        if self.model is None:
            contents = [item.get("content", str(item)) for item in context[:5]]
            summary = " | ".join(contents)
            self.logger.debug(f"STM SUMMARY (simple): {len(context)} items")
            return summary

        # Use LLM for summarization
        try:
            prompt = self._build_summary_prompt(context)
            summary = await self.model.generate(prompt)
            self.logger.debug(f"STM SUMMARY (LLM): {len(context)} items -> {len(summary)} chars")
            return summary
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            # Fallback to simple concatenation
            contents = [item.get("content", str(item)) for item in context[:5]]
            return " | ".join(contents)

    async def filter(
        self,
        context: Optional[List[Dict[str, Any]]] = None,
        relevance_threshold: float = 0.3,
        query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        FILTER: Remove irrelevant information from context.

        Args:
            context: Context to filter (uses buffer if not provided)
            relevance_threshold: Minimum relevance score to keep
            query: Query for relevance scoring (uses current query if not provided)

        Returns:
            Filtered context
        """
        if context is None:
            context = await self.get_context()

        if not context:
            return []

        query = query or self._current_query

        filtered = []
        for item in context:
            score = item.get("score", item.get("relevance_score", 1.0))

            # If we have a query and model, compute relevance
            if query and self.model and score == 1.0:
                score = await self._compute_relevance(item, query)

            if score >= relevance_threshold:
                filtered.append(item)

        self.logger.debug(f"STM FILTER: {len(context)} -> {len(filtered)} items (threshold={relevance_threshold})")
        return filtered

    async def get_context(self) -> List[Dict[str, Any]]:
        """
        Get current context buffer.

        Returns:
            List of context items as dictionaries
        """
        return [
            {
                "id": item.id,
                "content": item.content,
                "source": item.source,
                "relevance_score": item.relevance_score,
                "timestamp": item.timestamp,
                "metadata": item.metadata,
            }
            for item in self._buffer
        ]

    async def add_to_context(
        self,
        content: Any,
        source: str = "external",
        relevance_score: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add item to context buffer.

        Args:
            content: Content to add
            source: Source identifier
            relevance_score: Relevance score
            metadata: Additional metadata
        """
        item = ContextItem(
            id=self._generate_id(),
            content=content,
            source=source,
            relevance_score=relevance_score,
            metadata=metadata or {},
        )
        self._buffer.append(item)

    async def clear(self) -> None:
        """Clear the context buffer."""
        self._buffer.clear()
        self._current_query = None
        self.logger.debug("STM cleared")

    async def get_stats(self) -> Dict[str, Any]:
        """Get STM statistics."""
        return {
            "buffer_size": len(self._buffer),
            "max_buffer_size": self.buffer_size,
            "context_window": self.context_window,
            "current_query": self._current_query,
        }

    def _build_summary_prompt(self, context: List[Dict[str, Any]]) -> str:
        """Build prompt for LLM summarization."""
        context_text = "\n\n".join([
            f"[{i+1}] {item.get('content', str(item))}"
            for i, item in enumerate(context)
        ])

        return f"""Summarize the following context items into a concise summary that captures the key information:

{context_text}

Summary:"""

    async def _compute_relevance(self, item: Dict[str, Any], query: str) -> float:
        """Compute relevance score for an item given a query."""
        # Simple keyword overlap for now
        content = str(item.get("content", "")).lower()
        query_words = set(query.lower().split())
        content_words = set(content.split())

        if not query_words:
            return 1.0

        overlap = len(query_words & content_words)
        return overlap / len(query_words)

    def _generate_id(self) -> str:
        """Generate unique ID."""
        import secrets
        return secrets.token_hex(6)
