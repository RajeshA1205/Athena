"""
LatentMAS Message Router
========================
Priority-based and attention-based message routing across the shared latent
space for the ATHENA multi-agent trading system.

Routes messages between agents using:
- Per-agent priority queues (HIGH / MEDIUM / LOW) backed by asyncio.Queue
- Fallback routing through the shared LatentSpace
- Optional attention-based selective broadcast
"""

import asyncio
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from communication.encoder import AgentStateEncoder
from communication.decoder import AgentStateDecoder
from communication.latent_space import LatentSpace

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Priority levels for routed messages, highest value processed first."""

    HIGH = 3
    MEDIUM = 2
    LOW = 1


class MessageRouter:
    """
    Routes messages between agents via priority queues and the shared latent
    space.

    Supports three routing strategies:
    1. Priority-queue routing — direct, in-process queues per agent per level.
    2. LatentSpace routing — encodes messages into the shared latent space,
       useful for inter-process or broadcast communication.
    3. Attention-based broadcast — selective broadcast to the top-N most
       relevant agents, scored by dot-product attention.
    """

    def __init__(
        self,
        latent_space: LatentSpace,
        encoder: AgentStateEncoder,
        decoder: AgentStateDecoder,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize MessageRouter.

        Args:
            latent_space: Shared LatentSpace for fallback/broadcast routing.
            encoder: AgentStateEncoder for encoding messages to latent vectors.
            decoder: AgentStateDecoder for decoding latent vectors back to
                     agent-readable outputs.
            config: Optional configuration dict. Supported keys:
                    - enable_priority (bool, default True): Use priority queues
                      for unicast routing.
                    - enable_attention (bool, default False): Use dot-product
                      attention to select broadcast recipients.
                    - max_attention_recipients (int, default 5): Maximum number
                      of agents to target in an attention-based broadcast.
        """
        self.latent_space = latent_space
        self.encoder = encoder
        self.decoder = decoder

        config = config or {}
        self.enable_priority: bool = config.get("enable_priority", True)
        self.enable_attention: bool = config.get("enable_attention", False)
        self.max_attention_recipients: int = config.get("max_attention_recipients", 5)

        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        self.priority_queues: Dict[str, Dict[MessagePriority, asyncio.Queue]] = {}

        logger.info(
            "MessageRouter initialized: enable_priority=%s, "
            "enable_attention=%s, max_attention_recipients=%d",
            self.enable_priority,
            self.enable_attention,
            self.max_attention_recipients,
        )

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def register_agent(self, agent_id: str, agent_info: Dict[str, Any]) -> None:
        """
        Register an agent with the router.

        Creates per-priority asyncio queues for the agent if they do not
        already exist.

        Args:
            agent_id: Unique identifier for the agent.
            agent_info: Arbitrary metadata dict associated with the agent.
        """
        self.agent_registry[agent_id] = agent_info

        if agent_id not in self.priority_queues:
            self.priority_queues[agent_id] = {
                MessagePriority.HIGH: asyncio.Queue(),
                MessagePriority.MEDIUM: asyncio.Queue(),
                MessagePriority.LOW: asyncio.Queue(),
            }

        logger.debug("Agent registered: %s (info=%s)", agent_id, agent_info)

    # ------------------------------------------------------------------
    # Send
    # ------------------------------------------------------------------

    async def send(
        self,
        sender_id: str,
        receiver_id: str,
        message: Any,
        priority: MessagePriority = MessagePriority.MEDIUM,
    ) -> bool:
        """
        Send a message from one agent to another (or broadcast).

        Encodes the message to a latent vector, then routes it:
        - If priority routing is enabled and receiver is not 'broadcast':
          puts the latent vector directly into the receiver's priority queue.
        - Otherwise: wraps the raw message in an AgentMessage and routes it
          through the shared LatentSpace.

        Args:
            sender_id: ID of the sending agent.
            receiver_id: ID of the receiving agent, or 'broadcast'.
            message: Message payload (torch.Tensor, dict, or str).
            priority: Routing priority (default MEDIUM).

        Returns:
            True if the message was queued / sent successfully, False otherwise.
        """
        try:
            latent_message = await self.encoder.encode_agent_state(message)
        except Exception as e:
            logger.error(
                "Failed to encode message from %s to %s: %s",
                sender_id,
                receiver_id,
                e,
            )
            return False

        if self.enable_priority and receiver_id != "broadcast":
            if receiver_id not in self.agent_registry:
                self.register_agent(receiver_id, {})

            await self.priority_queues[receiver_id][priority].put(latent_message)
            logger.debug(
                "Message queued: %s -> %s (priority=%s)",
                sender_id,
                receiver_id,
                priority.name,
            )
            return True

        # Fallback: route through the shared LatentSpace.
        try:
            from core.base_agent import AgentMessage

            recipient = "*" if receiver_id == "broadcast" else receiver_id
            agent_msg = AgentMessage(
                sender=sender_id,
                recipient=recipient,
                content=message,
                priority=priority.value,
            )
            return await self.latent_space.send(agent_msg)
        except Exception as e:
            logger.error(
                "Failed to route message via LatentSpace (%s -> %s): %s",
                sender_id,
                receiver_id,
                e,
            )
            return False

    # ------------------------------------------------------------------
    # Receive
    # ------------------------------------------------------------------

    async def receive(
        self, receiver_id: str, decode_mode: str = "numeric"
    ) -> List[Any]:
        """
        Collect and decode all pending messages for an agent.

        Drains the agent's priority queues from HIGH down to LOW, then
        collects any messages buffered in the shared LatentSpace. All
        gathered latent vectors are batch-decoded in one pass.

        Args:
            receiver_id: ID of the receiving agent.
            decode_mode: Decoder output mode — 'numeric', 'text_embedding',
                         or 'structured' (passed to AgentStateDecoder).

        Returns:
            List of decoded message outputs. Returns an empty list when no
            messages are pending.
        """
        latent_messages: List[Any] = []

        # Drain priority queues HIGH -> MEDIUM -> LOW.
        if receiver_id in self.priority_queues:
            for level in (MessagePriority.HIGH, MessagePriority.MEDIUM, MessagePriority.LOW):
                queue = self.priority_queues[receiver_id][level]
                while True:
                    try:
                        latent_messages.append(queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

        # Collect from the shared LatentSpace.  LatentSpace.receive returns
        # AgentMessage objects (already decoded); we re-encode their content
        # to obtain latent tensors so we can run them through our decoder
        # for a consistent output format.
        try:
            agent_messages = await self.latent_space.receive(receiver_id)
            for agent_msg in agent_messages:
                try:
                    latent = await self.encoder.encode_agent_state(agent_msg.content)
                    latent_messages.append(latent)
                except Exception as enc_err:
                    logger.warning(
                        "Could not re-encode LatentSpace message for %s: %s",
                        receiver_id,
                        enc_err,
                    )
        except Exception as e:
            logger.error(
                "Failed to receive messages from LatentSpace for %s: %s",
                receiver_id,
                e,
            )

        if not latent_messages:
            return []

        try:
            decoded = await self.decoder.decode_messages(latent_messages, mode=decode_mode)
        except Exception as e:
            logger.error(
                "Failed to decode messages for %s: %s", receiver_id, e
            )
            return []

        logger.debug(
            "Received %d message(s) for %s (mode=%s)",
            len(decoded),
            receiver_id,
            decode_mode,
        )
        return decoded

    # ------------------------------------------------------------------
    # Attention-based broadcast
    # ------------------------------------------------------------------

    async def broadcast_with_attention(
        self,
        sender_id: str,
        message: Any,
        agent_embeddings: Dict[str, Any],
    ) -> int:
        """
        Broadcast a message to the most relevant agents using attention scoring.

        When attention is enabled and agent embeddings are provided, computes
        a dot-product score between the encoded message latent and each agent's
        embedding, then sends the message only to the top-N highest-scoring
        agents (where N = max_attention_recipients).

        Falls back to a plain broadcast through the LatentSpace when attention
        is disabled or no agent embeddings are provided.

        Args:
            sender_id: ID of the sending agent.
            message: Message payload to broadcast.
            agent_embeddings: Dict mapping agent_id -> embedding (torch.Tensor
                              or compatible numeric type). Used to compute
                              relevance scores.

        Returns:
            Number of agents that received the message.
        """
        try:
            latent_message = await self.encoder.encode_agent_state(message)
        except Exception as e:
            logger.error(
                "Failed to encode broadcast message from %s: %s", sender_id, e
            )
            return 0

        if not self.enable_attention or not agent_embeddings:
            # Fallback: standard LatentSpace broadcast.
            try:
                from core.base_agent import AgentMessage

                agent_msg = AgentMessage(
                    sender=sender_id,
                    recipient="*",
                    content=message,
                )
                await self.latent_space.send(agent_msg)
                recipient_count = len(self.agent_registry)
                logger.debug(
                    "Broadcast (no attention) from %s to %d agents",
                    sender_id,
                    recipient_count,
                )
                return recipient_count
            except Exception as e:
                logger.error(
                    "Fallback broadcast failed for %s: %s", sender_id, e
                )
                return 0

        # Attention-based selection: score each agent's embedding against the
        # encoded message latent via dot product, then send to the top-N.
        scores: List[Tuple[str, float]] = []

        for agent_id, embedding in agent_embeddings.items():
            if agent_id == sender_id:
                continue
            try:
                if HAS_TORCH and isinstance(latent_message, torch.Tensor):
                    # Both operands are (or can be cast to) tensors.
                    if not isinstance(embedding, torch.Tensor):
                        emb_vec = torch.tensor(embedding, dtype=torch.float32)
                    else:
                        emb_vec = embedding.float()

                    msg_vec = latent_message.float()

                    # Flatten to 1-D and align lengths before dot product.
                    msg_flat = msg_vec.flatten()
                    emb_flat = emb_vec.flatten()
                    min_len = min(len(msg_flat), len(emb_flat))
                    score = float(
                        torch.dot(msg_flat[:min_len], emb_flat[:min_len])
                    )
                else:
                    # Numeric fallback: works for Python lists, tuples, and
                    # any iterable of numbers.
                    def _dot(a: Any, b: Any) -> float:
                        try:
                            return sum(float(x) * float(y) for x, y in zip(a, b))
                        except TypeError:
                            return float(a) * float(b)

                    score = _dot(latent_message, embedding)

                scores.append((agent_id, score))
            except Exception as score_err:
                logger.warning(
                    "Could not compute attention score for agent %s: %s",
                    agent_id,
                    score_err,
                )

        # Select top-N recipients by descending score.
        scores.sort(key=lambda pair: pair[1], reverse=True)
        selected = scores[: self.max_attention_recipients]

        sent_count = 0
        for target_id, score in selected:
            success = await self.send(
                sender_id=sender_id,
                receiver_id=target_id,
                message=message,
                priority=MessagePriority.HIGH,
            )
            if success:
                sent_count += 1
                logger.debug(
                    "Attention broadcast: %s -> %s (score=%.4f)",
                    sender_id,
                    target_id,
                    score,
                )
            else:
                logger.warning(
                    "Attention broadcast failed for target %s", target_id
                )

        logger.info(
            "Attention broadcast from %s: sent to %d/%d candidate agents",
            sender_id,
            sent_count,
            len(selected),
        )
        return sent_count

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Return routing statistics.

        Returns:
            Dict with:
            - registered_agents (int): Number of registered agents.
            - queue_depths (Dict[str, Dict[str, int]]): Per-agent queue depth
              for each priority level.
        """
        queue_depths: Dict[str, Dict[str, int]] = {}
        for agent_id, queues in self.priority_queues.items():
            queue_depths[agent_id] = {
                level.name: queues[level].qsize()
                for level in (MessagePriority.HIGH, MessagePriority.MEDIUM, MessagePriority.LOW)
            }

        return {
            "registered_agents": len(self.agent_registry),
            "queue_depths": queue_depths,
        }
