# TASK-006: Implement LatentMAS Shared Latent Space

## Status
- **State:** Queued
- **Priority:** 🟡 High
- **Depends on:** None (Sprint 1 complete)
- **Created:** 2026-02-15

## Objective
Create the shared latent space infrastructure for LatentMAS-based inter-agent communication, enabling agents to communicate via learned latent representations.

## Context
LatentMAS (from the Latent Collaboration paper) enables agents to communicate in a shared embedding space rather than through discrete messages. This reduces communication overhead and enables emergent collaboration patterns.

This component provides:
- Shared latent space configuration and initialization
- Dimension alignment across agents
- Latent representation management
- Foundation for encoder/decoder components

Reference the LatentMAS paper in `/Users/rajesh/athena/architecture/base/` and the architecture plan at `/Users/rajesh/athena/agent-plan/atomic-wibbling-simon.md` (lines 73-77, 154-170, 312).

## Scope & Constraints

### Files to Create
- `/Users/rajesh/athena/communication/latent_space.py`
- `/Users/rajesh/athena/communication/__init__.py`

### Files to Reference (DO NOT MODIFY)
- `/Users/rajesh/athena/models/embeddings.py` — Embedding models
- `/Users/rajesh/athena/core/config.py` — Configuration
- Research paper: `/Users/rajesh/athena/architecture/base/LatentMAS.pdf`

### Constraints
- Use PyTorch tensors for latent representations
- Support configurable latent space dimensionality
- Use **async/await** for all operations
- Thread-safe for concurrent agent access
- No training logic (that comes in Sprint 4)

## Input
- LatentMAS paper specification
- Embedding models interface
- Project configuration system
- PyTorch framework

## Expected Output

### File: `/Users/rajesh/athena/communication/latent_space.py`
```python
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import asyncio
from threading import Lock

class LatentSpace:
    """
    Shared latent space for LatentMAS inter-agent communication.

    Provides a unified embedding space where agents can communicate via
    learned latent representations. Supports:
    - Shared latent dimension across all agents
    - Thread-safe concurrent access
    - Message buffering and retrieval
    - Attention-based message routing (basic version)

    Based on: Latent Collaboration for Multi-Agent Systems
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize shared latent space.

        Args:
            config: Configuration dict containing:
                - latent_dim: Dimension of latent space (default: 256)
                - max_messages: Max messages to buffer per agent (default: 100)
                - device: PyTorch device (default: 'cpu')
        """
        self.latent_dim = config.get('latent_dim', 256)
        self.max_messages = config.get('max_messages', 100)
        self.device = config.get('device', 'cpu')

        # Message buffer: agent_id -> List[torch.Tensor]
        self.message_buffer: Dict[str, List[torch.Tensor]] = {}
        self.buffer_lock = Lock()

        # Broadcast channel for system-wide messages
        self.broadcast_buffer: List[torch.Tensor] = []

    async def send_message(self, sender_id: str, receiver_id: str, latent_message: torch.Tensor) -> bool:
        """
        Send latent message from one agent to another.

        Args:
            sender_id: ID of sending agent
            receiver_id: ID of receiving agent (or 'broadcast' for all)
            latent_message: Latent representation tensor [latent_dim]

        Returns:
            True if successful, False if buffer full
        """
        if latent_message.shape[0] != self.latent_dim:
            raise ValueError(f"Message dimension {latent_message.shape[0]} != latent_dim {self.latent_dim}")

        with self.buffer_lock:
            if receiver_id == 'broadcast':
                self.broadcast_buffer.append(latent_message.to(self.device))
                if len(self.broadcast_buffer) > self.max_messages:
                    self.broadcast_buffer.pop(0)
            else:
                if receiver_id not in self.message_buffer:
                    self.message_buffer[receiver_id] = []

                if len(self.message_buffer[receiver_id]) >= self.max_messages:
                    return False

                self.message_buffer[receiver_id].append(latent_message.to(self.device))

        return True

    async def receive_messages(self, agent_id: str, clear: bool = True) -> List[torch.Tensor]:
        """
        Receive all pending messages for an agent.

        Args:
            agent_id: ID of receiving agent
            clear: Whether to clear messages after retrieval

        Returns:
            List of latent message tensors
        """
        with self.buffer_lock:
            messages = self.message_buffer.get(agent_id, []).copy()
            broadcast_messages = self.broadcast_buffer.copy()

            if clear:
                self.message_buffer[agent_id] = []

        # Combine direct and broadcast messages
        all_messages = messages + broadcast_messages
        return all_messages

    async def get_message_summary(self, messages: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate multiple messages into a single summary representation.

        Args:
            messages: List of latent message tensors

        Returns:
            Summary tensor [latent_dim]
        """
        if not messages:
            return torch.zeros(self.latent_dim, device=self.device)

        # Simple mean aggregation (can be enhanced with attention in Sprint 4)
        stacked = torch.stack(messages, dim=0)  # [num_messages, latent_dim]
        summary = torch.mean(stacked, dim=0)  # [latent_dim]
        return summary

    def clear_all_messages(self):
        """Clear all message buffers (useful for resets)."""
        with self.buffer_lock:
            self.message_buffer.clear()
            self.broadcast_buffer.clear()

    def get_buffer_status(self) -> Dict[str, int]:
        """Get current buffer fill levels for monitoring."""
        with self.buffer_lock:
            status = {agent_id: len(msgs) for agent_id, msgs in self.message_buffer.items()}
            status['broadcast'] = len(self.broadcast_buffer)
        return status
```

### File: `/Users/rajesh/athena/communication/__init__.py`
```python
from .latent_space import LatentSpace

__all__ = ['LatentSpace']
```

## Acceptance Criteria
- [ ] LatentSpace class created with configurable latent dimension
- [ ] `send_message()` method supports direct and broadcast messaging
- [ ] `receive_messages()` method retrieves pending messages
- [ ] `get_message_summary()` aggregates multiple messages
- [ ] Message buffering with configurable max size
- [ ] Thread-safe operations with proper locking
- [ ] All async methods use async/await pattern
- [ ] Uses PyTorch tensors on configurable device
- [ ] Class is importable and instantiable
- [ ] Docstrings present for all public methods

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
