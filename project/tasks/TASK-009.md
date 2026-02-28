# TASK-009: Implement LatentMAS Router

## Status
- **State:** Queued
- **Priority:** 🟢 Medium
- **Depends on:** TASK-006, TASK-007, TASK-008
- **Created:** 2026-02-15

## Objective
Create the router component that manages message routing, prioritization, and attention-based delivery in the LatentMAS communication system.

## Context
The router handles intelligent message routing between agents. It determines which messages should be delivered to which agents, with what priority, and enables attention-based selective communication for efficiency.

This component provides:
- Message routing logic (point-to-point, broadcast, selective)
- Priority-based message queuing
- Attention-based message filtering (basic version, enhanced in Sprint 4)
- Communication channel management

Reference the LatentMAS paper in `/Users/rajesh/athena/architecture/base/` and the architecture plan at `/Users/rajesh/athena/agent-plan/atomic-wibbling-simon.md` (lines 73-77, 165-170).

## Scope & Constraints

### Files to Create
- `/Users/rajesh/athena/communication/router.py`

### Files to Modify
- `/Users/rajesh/athena/communication/__init__.py` — Add MessageRouter import

### Files to Reference (DO NOT MODIFY)
- `/Users/rajesh/athena/communication/latent_space.py` — LatentSpace
- `/Users/rajesh/athena/communication/encoder.py` — LatentEncoder
- `/Users/rajesh/athena/communication/decoder.py` — LatentDecoder

### Constraints
- Must work with LatentSpace, Encoder, and Decoder
- Support priority queuing (high, medium, low)
- Use **async/await** for all operations
- Thread-safe for concurrent routing
- Basic attention mechanism (advanced version in Sprint 4)

## Input
- LatentSpace, LatentEncoder, LatentDecoder implementations
- LatentMAS paper specification
- PyTorch framework

## Expected Output

### File: `/Users/rajesh/athena/communication/router.py`
```python
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
import asyncio
from enum import Enum
from .latent_space import LatentSpace
from .encoder import LatentEncoder
from .decoder import LatentDecoder

class MessagePriority(Enum):
    HIGH = 3
    MEDIUM = 2
    LOW = 1

class MessageRouter:
    """
    Router for managing LatentMAS message routing and prioritization.

    Handles:
    - Point-to-point and broadcast routing
    - Priority-based message queuing
    - Attention-based message filtering (basic)
    - Communication channel management

    Based on: Latent Collaboration for Multi-Agent Systems
    """

    def __init__(self, latent_space: LatentSpace, encoder: LatentEncoder, decoder: LatentDecoder, config: Dict[str, Any]):
        """
        Initialize message router.

        Args:
            latent_space: Shared LatentSpace instance
            encoder: LatentEncoder instance
            decoder: LatentDecoder instance
            config: Configuration dict containing:
                - enable_priority: Enable priority routing (default: True)
                - enable_attention: Enable attention filtering (default: False, Sprint 4)
                - max_attention_recipients: Max recipients for broadcast (default: 5)
        """
        self.latent_space = latent_space
        self.encoder = encoder
        self.decoder = decoder

        self.enable_priority = config.get('enable_priority', True)
        self.enable_attention = config.get('enable_attention', False)
        self.max_attention_recipients = config.get('max_attention_recipients', 5)

        # Routing tables
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        self.priority_queues: Dict[str, Dict[MessagePriority, List[torch.Tensor]]] = {}

    def register_agent(self, agent_id: str, agent_info: Dict[str, Any]):
        """
        Register an agent with the router.

        Args:
            agent_id: Unique agent identifier
            agent_info: Agent metadata (role, capabilities, etc.)
        """
        self.agent_registry[agent_id] = agent_info
        self.priority_queues[agent_id] = {
            MessagePriority.HIGH: [],
            MessagePriority.MEDIUM: [],
            MessagePriority.LOW: []
        }

    async def send(self, sender_id: str, receiver_id: str, message: Any, priority: MessagePriority = MessagePriority.MEDIUM) -> bool:
        """
        Send message from sender to receiver via latent space.

        Args:
            sender_id: Sending agent ID
            receiver_id: Receiving agent ID (or 'broadcast')
            message: Message to send (will be encoded)
            priority: Message priority level

        Returns:
            True if successful
        """
        # Encode message to latent representation
        latent_message = await self.encoder.encode_agent_state(message)

        # Route based on priority if enabled
        if self.enable_priority and receiver_id != 'broadcast':
            if receiver_id not in self.priority_queues:
                self.register_agent(receiver_id, {})

            self.priority_queues[receiver_id][priority].append(latent_message)
            return True
        else:
            # Direct send to latent space
            return await self.latent_space.send_message(sender_id, receiver_id, latent_message)

    async def receive(self, receiver_id: str, decode_mode: str = 'numeric') -> List[Any]:
        """
        Receive messages for an agent, respecting priority order.

        Args:
            receiver_id: Receiving agent ID
            decode_mode: How to decode messages ('numeric', 'text', 'structured')

        Returns:
            List of decoded messages, ordered by priority
        """
        messages = []

        # If priority routing enabled, collect from priority queues
        if self.enable_priority and receiver_id in self.priority_queues:
            for priority in [MessagePriority.HIGH, MessagePriority.MEDIUM, MessagePriority.LOW]:
                priority_messages = self.priority_queues[receiver_id][priority]
                messages.extend(priority_messages)
                self.priority_queues[receiver_id][priority].clear()

        # Also get messages from latent space (non-priority)
        latent_messages = await self.latent_space.receive_messages(receiver_id, clear=True)
        messages.extend(latent_messages)

        # Decode all messages
        if not messages:
            return []

        decoded = await self.decoder.decode_messages(messages, mode=decode_mode)
        return decoded

    async def broadcast_with_attention(self, sender_id: str, message: Any, agent_embeddings: Dict[str, torch.Tensor]) -> int:
        """
        Broadcast message to subset of agents based on attention scores.

        NOTE: Basic version. Advanced attention mechanism comes in Sprint 4.

        Args:
            sender_id: Sending agent ID
            message: Message to broadcast
            agent_embeddings: Agent ID -> embedding tensor for attention calculation

        Returns:
            Number of agents message was sent to
        """
        # Encode message
        latent_message = await self.encoder.encode_agent_state(message)

        if not self.enable_attention or not agent_embeddings:
            # Fallback to full broadcast
            await self.latent_space.send_message(sender_id, 'broadcast', latent_message)
            return len(self.agent_registry)

        # Calculate attention scores (simple dot product for now)
        attention_scores = {}
        for agent_id, agent_emb in agent_embeddings.items():
            if agent_id == sender_id:
                continue
            score = torch.dot(latent_message, agent_emb).item()
            attention_scores[agent_id] = score

        # Select top-k agents
        sorted_agents = sorted(attention_scores.items(), key=lambda x: x[1], reverse=True)
        selected_agents = sorted_agents[:self.max_attention_recipients]

        # Send to selected agents
        sent_count = 0
        for agent_id, score in selected_agents:
            success = await self.latent_space.send_message(sender_id, agent_id, latent_message)
            if success:
                sent_count += 1

        return sent_count

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics for monitoring."""
        stats = {
            'registered_agents': len(self.agent_registry),
            'priority_queue_depths': {}
        }

        for agent_id, queues in self.priority_queues.items():
            stats['priority_queue_depths'][agent_id] = {
                'high': len(queues[MessagePriority.HIGH]),
                'medium': len(queues[MessagePriority.MEDIUM]),
                'low': len(queues[MessagePriority.LOW])
            }

        return stats
```

### Update: `/Users/rajesh/athena/communication/__init__.py`
Add MessageRouter and MessagePriority to imports and __all__.

## Acceptance Criteria
- [ ] MessageRouter class created with LatentSpace, Encoder, Decoder integration
- [ ] `register_agent()` method for agent registration
- [ ] `send()` method with priority-based routing
- [ ] `receive()` method that respects message priority order
- [ ] `broadcast_with_attention()` method for selective broadcast (basic version)
- [ ] MessagePriority enum with HIGH, MEDIUM, LOW levels
- [ ] Priority queuing per agent
- [ ] All methods use async/await pattern
- [ ] `get_routing_stats()` for monitoring
- [ ] Class is importable and instantiable
- [ ] Docstrings present for all public methods

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
