"""
Graphiti Backend Adapter
========================
Adapter for Graphiti (Zep) temporal knowledge graph storage.
Provides the storage layer for AgeMem operations.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import asyncio
import os

try:
    from graphiti_core import Graphiti
    from graphiti_core.nodes import EpisodeType
    from graphiti_core.llm_client.groq_client import GroqClient
    from graphiti_core.llm_client.config import LLMConfig
    # Groq entity extraction may produce validation warnings — episodes are
    # still stored and searchable via keyword fallback. Full graph search
    # requires a structured-output-capable LLM (e.g. Anthropic/OpenAI).
    GRAPHITI_AVAILABLE = True
except ImportError:
    GRAPHITI_AVAILABLE = False


@dataclass
class Episode:
    """Represents a memory episode in Graphiti."""
    id: str
    content: str
    source: str = "athena"
    episode_type: str = "text"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SearchResult:
    """Result from Graphiti search."""
    id: str
    content: str
    score: float
    episode_type: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    entities: List[str] = field(default_factory=list)


class GraphitiBackend:
    """
    Graphiti Backend for AgeMem.

    Wraps Graphiti's temporal knowledge graph to provide:
    - Episode storage (memories as episodes with entities/relations)
    - Semantic search with temporal awareness
    - Entity and relationship extraction
    """

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        llm_client: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        groq_api_key: Optional[str] = None,
        groq_model: str = "llama-3.3-70b-versatile",
        voyage_api_key: Optional[str] = None,
    ):
        """
        Initialize Graphiti backend.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            llm_client: LLM client for entity extraction
            embedding_model: Model for embeddings
            config: Additional configuration
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.config = config or {}

        # Build LLM client — use Groq if available, else caller-supplied
        _groq_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if llm_client is not None:
            self.llm_client = llm_client
        elif GRAPHITI_AVAILABLE and _groq_key:
            self.llm_client = GroqClient(
                config=LLMConfig(api_key=_groq_key, model=groq_model)
            )
        else:
            self.llm_client = None

        # Build embedder — use Voyage if available, else caller-supplied
        # VoyageAIEmbedder imported lazily to avoid triggering transformers
        # torch-check at module import time (causes ValueError when torch
        # is partially initialized earlier in the test suite).
        _voyage_key = voyage_api_key or os.getenv("VOYAGE_API_KEY")
        if embedding_model is not None:
            self.embedding_model = embedding_model
        elif GRAPHITI_AVAILABLE and _voyage_key:
            try:
                from graphiti_core.embedder.voyage import VoyageAIEmbedder, VoyageAIEmbedderConfig
                self.embedding_model = VoyageAIEmbedder(
                    VoyageAIEmbedderConfig(api_key=_voyage_key)
                )
            except Exception:
                self.embedding_model = None
        else:
            self.embedding_model = None

        self.logger = logging.getLogger("athena.memory.graphiti")
        self._client: Optional[Any] = None
        self._initialized = False

        # In-memory fallback when Graphiti is not available
        self._fallback_store: Dict[str, Episode] = {}
        self._use_fallback = not GRAPHITI_AVAILABLE

    async def initialize(self) -> bool:
        """
        Initialize connection to Graphiti/Neo4j.

        Returns:
            True if initialized successfully
        """
        if self._initialized:
            return True

        if self._use_fallback:
            self.logger.warning("Graphiti not available, using in-memory fallback")
            self._initialized = True
            return True

        try:
            self._client = Graphiti(
                uri=self.neo4j_uri,
                user=self.neo4j_user,
                password=self.neo4j_password,
                llm_client=self.llm_client,
                embedder=self.embedding_model,
            )

            # Build indices
            await self._client.build_indices_and_constraints()

            self._initialized = True
            self.logger.info("Graphiti backend initialized")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Graphiti: {e}")
            self.logger.warning("Falling back to in-memory storage")
            self._use_fallback = True
            self._initialized = True
            return True

    async def add_episode(
        self,
        content: str,
        source: str = "athena",
        episode_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
        reference_time: Optional[datetime] = None,
    ) -> Optional[str]:
        """
        Add a new episode to the knowledge graph.

        Args:
            content: Episode content
            source: Source identifier
            episode_type: Type of episode (text, json, message)
            metadata: Additional metadata
            reference_time: Timestamp for the episode

        Returns:
            Episode ID if successful, None otherwise
        """
        if not self._initialized:
            await self.initialize()

        episode_id = self._generate_id()
        timestamp = (reference_time or datetime.now()).isoformat()

        episode = Episode(
            id=episode_id,
            content=content,
            source=source,
            episode_type=episode_type,
            timestamp=timestamp,
            metadata=metadata or {},
        )

        if self._use_fallback:
            self._fallback_store[episode_id] = episode
            self.logger.debug(f"Added episode to fallback store: {episode_id}")
            return episode_id

        try:
            # Map to Graphiti episode type
            graphiti_type = self._map_episode_type(episode_type)

            await self._client.add_episode(
                name=f"episode_{episode_id}",
                episode_body=content,
                source_description=source,
                source=EpisodeType(graphiti_type),
                reference_time=reference_time or datetime.now(timezone.utc),
                group_id=source,
            )

            # Store metadata separately (Graphiti may not support arbitrary metadata)
            self._fallback_store[episode_id] = episode  # Keep local copy with metadata

            self.logger.debug(f"Added episode to Graphiti: {episode_id}")
            return episode_id

        except Exception as e:
            self.logger.error(f"Failed to add episode: {e}")
            # Fallback to local storage
            self._fallback_store[episode_id] = episode
            return episode_id

    async def update_episode(
        self,
        episode_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update an existing episode.

        Args:
            episode_id: ID of episode to update
            content: New content (if any)
            metadata: Updated metadata (merged with existing)

        Returns:
            True if successful
        """
        if not self._initialized:
            await self.initialize()

        if episode_id not in self._fallback_store:
            self.logger.warning(f"Episode not found: {episode_id}")
            return False

        episode = self._fallback_store[episode_id]

        if content is not None:
            episode.content = content

        if metadata is not None:
            episode.metadata.update(metadata)

        episode.timestamp = datetime.now().isoformat()

        # If using Graphiti, we'd need to delete and re-add since
        # Graphiti doesn't have direct update functionality
        if not self._use_fallback and self._client:
            try:
                # For now, just update local copy
                # TODO: Implement proper Graphiti update via delete+add
                pass
            except Exception as e:
                self.logger.error(f"Failed to update in Graphiti: {e}")

        self.logger.debug(f"Updated episode: {episode_id}")
        return True

    async def delete_episode(self, episode_id: str) -> bool:
        """
        Delete an episode from the knowledge graph.

        Args:
            episode_id: ID of episode to delete

        Returns:
            True if successful
        """
        if not self._initialized:
            await self.initialize()

        if episode_id not in self._fallback_store:
            self.logger.warning(f"Episode not found: {episode_id}")
            return False

        del self._fallback_store[episode_id]

        # TODO: Delete from Graphiti if using it
        # Graphiti deletion would require finding and removing
        # the episode node and its relationships

        self.logger.debug(f"Deleted episode: {episode_id}")
        return True

    async def search(
        self,
        query: str,
        top_k: int = 5,
        center_date: Optional[datetime] = None,
        group_ids: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Search for relevant episodes.

        Args:
            query: Search query
            top_k: Number of results to return
            center_date: Center date for temporal search
            group_ids: Optional group IDs to filter

        Returns:
            List of search results
        """
        if not self._initialized:
            await self.initialize()

        if self._use_fallback:
            return await self._fallback_search(query, top_k)

        try:
            results = await self._client.search(
                query=query,
                num_results=top_k,
                group_ids=group_ids,
            )

            return [
                SearchResult(
                    id=getattr(r, 'uuid', str(i)),
                    content=r.fact if hasattr(r, 'fact') else str(r),
                    score=getattr(r, 'score', 1.0),
                    episode_type="text",
                    timestamp=datetime.now().isoformat(),
                )
                for i, r in enumerate(results)
            ]

        except Exception as e:
            self.logger.error(f"Graphiti search failed: {e}")
            return await self._fallback_search(query, top_k)

    async def _fallback_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Simple keyword-based fallback search."""
        query_lower = query.lower()
        scored_results = []

        for episode_id, episode in self._fallback_store.items():
            content_lower = episode.content.lower()

            # Simple relevance scoring
            score = 0.0
            query_words = query_lower.split()
            for word in query_words:
                if word in content_lower:
                    score += 1.0

            if score > 0:
                score = score / len(query_words)  # Normalize
                scored_results.append((episode, score))

        # Sort by score descending
        scored_results.sort(key=lambda x: x[1], reverse=True)

        return [
            SearchResult(
                id=episode.id,
                content=episode.content,
                score=score,
                episode_type=episode.episode_type,
                timestamp=episode.timestamp,
                metadata=episode.metadata,
            )
            for episode, score in scored_results[:top_k]
        ]

    async def get_episode(self, episode_id: str) -> Optional[Episode]:
        """
        Get episode by ID.

        Args:
            episode_id: Episode ID

        Returns:
            Episode or None
        """
        return self._fallback_store.get(episode_id)

    async def get_all_episodes(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Episode]:
        """
        Get all episodes with pagination.

        Args:
            limit: Maximum number to return
            offset: Offset for pagination

        Returns:
            List of episodes
        """
        episodes = list(self._fallback_store.values())
        return episodes[offset:offset + limit]

    async def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics."""
        return {
            "episode_count": len(self._fallback_store),
            "using_graphiti": not self._use_fallback,
            "initialized": self._initialized,
            "neo4j_uri": self.neo4j_uri,
        }

    async def close(self) -> None:
        """Close backend connections."""
        if self._client and hasattr(self._client, 'close'):
            await self._client.close()
        self._initialized = False
        self.logger.info("Graphiti backend closed")

    def _generate_id(self) -> str:
        """Generate unique episode ID."""
        import hashlib
        import random
        timestamp = datetime.now().isoformat()
        random_part = str(random.random())
        return hashlib.sha256(f"{timestamp}:{random_part}".encode()).hexdigest()[:16]

    def _map_episode_type(self, episode_type: str) -> str:
        """Map internal episode type to Graphiti type."""
        mapping = {
            "text": "text",
            "json": "json",
            "message": "message",
        }
        return mapping.get(episode_type, "text")
