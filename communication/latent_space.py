"""
LatentMAS Communication Layer
==============================
Shared latent space infrastructure for inter-agent communication.

Based on LatentMAS research: agents communicate via learned latent
representations instead of discrete messages, enabling:
- Efficient encoding/decoding of complex agent states
- Attention-based message routing
- Priority-based message channels
- Broadcast communication
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timezone
import asyncio
import hashlib
import logging
import random

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    import types as _types
    torch = _types.SimpleNamespace(  # type: ignore[assignment]
        Tensor=object,
        device=lambda x: x,
        cuda=_types.SimpleNamespace(is_available=lambda: False),
    )
    nn = _types.SimpleNamespace(Module=object)  # type: ignore[assignment]

from core.base_agent import AgentMessage


@dataclass
class LatentMessage:
    """Internal latent representation of an agent message."""
    message_id: str
    sender: str
    recipient: str
    latent_vector: Any
    priority: int = 1
    message_type: str = "default"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class LatentEncoder(nn.Module if HAS_TORCH else object):  # type: ignore[misc]
    """Encodes agent messages to latent representations."""

    def __init__(
        self, latent_dim: int = 512, num_heads: int = 8,
        hidden_dim: int = 2048, dropout: float = 0.1,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for LatentEncoder")
        super().__init__()
        self.latent_dim = latent_dim

        self.attention = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.

        Args:
            x: Input tensor [batch, seq_len, latent_dim]

        Returns:
            Encoded latent tensor [batch, latent_dim]
        """
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x.mean(dim=1)


class LatentDecoder(nn.Module if HAS_TORCH else object):  # type: ignore[misc]
    """Decodes latent representations back to agent messages."""

    def __init__(
        self, latent_dim: int = 512, num_heads: int = 8,
        hidden_dim: int = 2048, dropout: float = 0.1,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for LatentDecoder")
        super().__init__()
        self.latent_dim = latent_dim

        self.attention = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, latent: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode latent representation.

        Args:
            latent: Latent tensor [batch, latent_dim]
            context: Optional context for cross-attention

        Returns:
            Decoded tensor [batch, latent_dim]
        """
        x = latent.unsqueeze(1) if latent.dim() == 2 else latent

        if context is not None:
            attn_out, _ = self.attention(x, context, context)
        else:
            attn_out, _ = self.attention(x, x, x)

        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x.squeeze(1)


class LatentSpace:
    """
    Shared latent space for inter-agent communication.

    Implements LatentMAS architecture:
    - Encodes agent messages to latent vectors
    - Maintains per-agent message queues
    - Supports broadcast and priority channels
    - Attention-based message retrieval
    """

    def __init__(
        self,
        latent_dim: int = 512,
        num_attention_heads: int = 8,
        message_queue_size: int = 100,
        broadcast_enabled: bool = True,
        priority_channels: int = 3,
        encoding_method: str = "transformer",
        device: Optional[str] = None,
    ):
        """
        Initialize latent space.

        Args:
            latent_dim: Dimension of latent representations
            num_attention_heads: Number of attention heads
            message_queue_size: Maximum messages per agent queue
            broadcast_enabled: Enable broadcast messages
            priority_channels: Number of priority levels
            encoding_method: Encoding method ("transformer")
            device: Device for PyTorch tensors
        """
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for LatentSpace communication. "
                "Install with: pip install torch"
            )

        self.latent_dim = latent_dim
        self.num_attention_heads = num_attention_heads
        self.message_queue_size = message_queue_size
        self.broadcast_enabled = broadcast_enabled
        self.priority_channels = priority_channels
        self.encoding_method = encoding_method

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.encoder = LatentEncoder(
            latent_dim=latent_dim, num_heads=num_attention_heads,
        ).to(self.device)
        self.decoder = LatentDecoder(
            latent_dim=latent_dim, num_heads=num_attention_heads,
        ).to(self.device)

        self._queues: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=message_queue_size)
        )
        self._broadcast_queue: deque = deque(maxlen=message_queue_size)
        self._lock = asyncio.Lock()

        self._stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "broadcasts": 0,
            "encoding_errors": 0,
            "decoding_errors": 0,
        }

        self.logger = logging.getLogger("athena.communication.latent_space")
        self.logger.info(
            f"LatentSpace initialized: dim={latent_dim}, "
            f"heads={num_attention_heads}, device={self.device}"
        )

        self.encoder.eval()
        self.decoder.eval()

    async def send(self, message: AgentMessage) -> bool:
        """
        Send a message through the latent space.

        Encodes the message content to latent representation and buffers it
        for the recipient(s).

        Args:
            message: AgentMessage to send

        Returns:
            True if message was sent successfully
        """
        async with self._lock:
            try:
                latent_vector = await self.encode_to_latent(message.content)

                latent_msg = LatentMessage(
                    message_id=self._generate_message_id(),
                    sender=message.sender,
                    recipient=message.recipient,
                    latent_vector=latent_vector,
                    priority=message.priority,
                    message_type=message.message_type,
                    timestamp=message.timestamp or datetime.now(timezone.utc).isoformat(),
                    metadata={**message.metadata, "_original_content": message.content},
                )

                if message.recipient == "*":
                    if self.broadcast_enabled:
                        self._broadcast_queue.append(latent_msg)
                        for agent_name in list(self._queues.keys()):
                            if agent_name != message.sender:
                                self._queues[agent_name].append(latent_msg)
                        self._stats["broadcasts"] += 1
                    else:
                        self.logger.warning("Broadcast attempted but not enabled")
                        return False
                else:
                    self._queues[message.recipient].append(latent_msg)

                self._stats["messages_sent"] += 1
                self.logger.debug(
                    f"Message sent: {message.sender} -> {message.recipient} "
                    f"(priority={message.priority})"
                )
                return True

            except Exception as e:
                self._stats["encoding_errors"] += 1
                self.logger.error(f"Failed to send message: {e}")
                return False

    async def receive(self, agent_name: str) -> List[AgentMessage]:
        """
        Receive pending messages for an agent.

        Args:
            agent_name: Name of the receiving agent

        Returns:
            List of AgentMessage objects
        """
        async with self._lock:
            try:
                messages = []

                queue = self._queues[agent_name]
                while queue:
                    latent_msg = queue.popleft()
                    agent_msg = await self._decode_latent_message(latent_msg)
                    if agent_msg:
                        messages.append(agent_msg)

                messages.sort(key=lambda m: m.priority, reverse=True)

                if messages:
                    self._stats["messages_received"] += len(messages)

                return messages

            except Exception as e:
                self.logger.error(f"Failed to receive messages for {agent_name}: {e}")
                return []

    async def broadcast(self, message: AgentMessage) -> bool:
        """
        Broadcast a message to all agents.

        Args:
            message: AgentMessage to broadcast

        Returns:
            True if broadcast was successful
        """
        if not self.broadcast_enabled:
            return False
        message.recipient = "*"
        return await self.send(message)

    async def encode_to_latent(self, content: Any) -> torch.Tensor:
        """
        Encode content to latent vector representation.

        Args:
            content: Content to encode

        Returns:
            Latent vector tensor [latent_dim]
        """
        content_str = str(content)

        tokens = [ord(c) % 256 for c in content_str[:self.latent_dim]]
        if len(tokens) < self.latent_dim:
            tokens.extend([0] * (self.latent_dim - len(tokens)))
        else:
            tokens = tokens[:self.latent_dim]

        embedding = torch.tensor(tokens, dtype=torch.float32, device=self.device)
        embedding = embedding / 256.0

        with torch.no_grad():
            x = embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, latent_dim]
            latent = self.encoder(x)  # [1, latent_dim]
            latent = latent.squeeze(0)  # [latent_dim]

        return latent

    async def decode_from_latent(self, latent: torch.Tensor) -> Any:
        """
        Decode latent vector back to content.

        Args:
            latent: Latent vector tensor [latent_dim]

        Returns:
            Decoded content
        """
        with torch.no_grad():
            x = latent.unsqueeze(0)  # [1, latent_dim]
            decoded = self.decoder(x)  # [1, latent_dim]
            decoded = decoded.squeeze(0)  # [latent_dim]

        decoded_np = decoded.cpu().numpy()
        chars = []
        for val in decoded_np:
            char_code = int(val * 256)
            if 32 <= char_code < 127:
                chars.append(chr(char_code))

        return "".join(chars).strip()

    async def get_message_summary(self, messages: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate multiple latent messages into a summary vector.

        Args:
            messages: List of latent message tensors

        Returns:
            Summary latent vector [latent_dim]
        """
        if not messages:
            return torch.zeros(self.latent_dim, device=self.device)
        stacked = torch.stack(messages)
        return stacked.mean(dim=0)

    async def compute_attention(
        self, query: torch.Tensor, keys: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute attention-based retrieval over messages.

        Args:
            query: Query latent vector [latent_dim]
            keys: List of key latent vectors

        Returns:
            Attention-weighted summary [latent_dim]
        """
        if not keys:
            return torch.zeros(self.latent_dim, device=self.device)

        K = torch.stack(keys)  # [num_keys, latent_dim]
        Q = query.unsqueeze(0)  # [1, latent_dim]

        scores = torch.matmul(Q, K.T) / (self.latent_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attention_weights, K)

        return attended.squeeze(0)

    async def _decode_latent_message(
        self, latent_msg: LatentMessage
    ) -> Optional[AgentMessage]:
        """Decode a latent message back to AgentMessage."""
        try:
            metadata = dict(latent_msg.metadata)
            content = metadata.pop("_original_content", None)
            if content is None:
                content = await self.decode_from_latent(latent_msg.latent_vector)
            return AgentMessage(
                sender=latent_msg.sender,
                recipient=latent_msg.recipient,
                content=content,
                message_type=latent_msg.message_type,
                priority=latent_msg.priority,
                timestamp=latent_msg.timestamp,
                metadata=metadata,
            )
        except Exception as e:
            self._stats["decoding_errors"] += 1
            self.logger.error(f"Failed to decode latent message: {e}")
            return None

    def _generate_message_id(self) -> str:
        """Generate unique message ID."""
        import secrets
        return secrets.token_hex(8)

    def _get_buffer_status_unlocked(self) -> Dict[str, Any]:
        """Caller must hold self._lock."""
        status = {
            "agents": {},
            "broadcast": {
                "size": len(self._broadcast_queue),
                "max_size": self.message_queue_size,
            },
        }
        for agent_name, queue in self._queues.items():
            status["agents"][agent_name] = {
                "size": len(queue),
                "max_size": self.message_queue_size,
            }
        return status

    async def get_buffer_status(self) -> Dict[str, Any]:
        """Get status of message buffers."""
        async with self._lock:
            return self._get_buffer_status_unlocked()

    async def clear_queue(self, agent_name: str) -> None:
        """Clear message queue for a specific agent."""
        async with self._lock:
            if agent_name in self._queues:
                self._queues[agent_name].clear()

    async def clear_broadcast_queue(self) -> None:
        """Clear the broadcast queue."""
        async with self._lock:
            self._broadcast_queue.clear()

    async def get_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        async with self._lock:
            buffer_status = self._get_buffer_status_unlocked()
            return {
                **self._stats,
                "buffers": buffer_status,
                "latent_dim": self.latent_dim,
                "device": str(self.device),
            }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "broadcasts": 0,
            "encoding_errors": 0,
            "decoding_errors": 0,
        }

    def register_agent(self, agent_name: str) -> None:
        """
        Pre-register an agent so it receives broadcasts even before
        receiving a direct message.

        Creates the agent's message queue eagerly, ensuring the agent
        appears in _queues.keys() during broadcast delivery.

        Args:
            agent_name: Unique agent identifier.
        """
        if agent_name not in self._queues:
            self._queues[agent_name] = deque(maxlen=self.message_queue_size)
            self.logger.debug("Registered agent queue: %s", agent_name)

    def train_mode(self) -> None:
        """Set encoder/decoder to training mode."""
        self.encoder.train()
        self.decoder.train()

    def eval_mode(self) -> None:
        """Set encoder/decoder to evaluation mode."""
        self.encoder.eval()
        self.decoder.eval()
