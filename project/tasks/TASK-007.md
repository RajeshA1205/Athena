# TASK-007: Implement LatentMAS Encoder

## Status
- **State:** Queued
- **Priority:** 🟡 High
- **Depends on:** TASK-006
- **Created:** 2026-02-15

## Objective
Create the encoder component that transforms agent outputs and decisions into latent representations for communication in the shared latent space.

## Context
The encoder is a critical component of LatentMAS. It takes high-dimensional agent outputs (observations, decisions, internal state) and projects them into the shared latent space for efficient communication with other agents.

This component provides:
- Agent output encoding to latent vectors
- Support for different input modalities (text, numeric, structured)
- Dimension reduction and alignment
- Encoding of agent internal state

Reference the LatentMAS paper in `/Users/rajesh/athena/architecture/base/` and the architecture plan at `/Users/rajesh/athena/agent-plan/atomic-wibbling-simon.md` (lines 73-77, 161-164).

## Scope & Constraints

### Files to Create
- `/Users/rajesh/athena/communication/encoder.py`

### Files to Modify
- `/Users/rajesh/athena/communication/__init__.py` — Add Encoder import

### Files to Reference (DO NOT MODIFY)
- `/Users/rajesh/athena/communication/latent_space.py` — LatentSpace
- `/Users/rajesh/athena/models/embeddings.py` — Embedding models
- `/Users/rajesh/athena/core/config.py` — Configuration

### Constraints
- Must work with LatentSpace from TASK-006
- Use PyTorch for encoding networks
- Support multiple input types (text, numeric, mixed)
- Use **async/await** for all operations
- No training logic yet (that comes in Sprint 4)

## Input
- LatentSpace implementation from TASK-006
- Embedding models interface
- LatentMAS paper specification
- PyTorch framework

## Expected Output

### File: `/Users/rajesh/athena/communication/encoder.py`
```python
import torch
import torch.nn as nn
from typing import Dict, Any, Union, List
import asyncio

class LatentEncoder(nn.Module):
    """
    Encoder for transforming agent outputs into latent representations.

    Supports encoding of:
    - Text outputs (via embedding models)
    - Numeric outputs (via learned projection)
    - Structured outputs (via learned aggregation)

    Based on: Latent Collaboration for Multi-Agent Systems
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize encoder.

        Args:
            config: Configuration dict containing:
                - latent_dim: Target latent dimension (default: 256)
                - input_dim: Dimension of numeric inputs (default: 512)
                - hidden_dims: Hidden layer dimensions (default: [512, 256])
                - dropout: Dropout rate (default: 0.1)
                - device: PyTorch device (default: 'cpu')
        """
        super().__init__()

        self.latent_dim = config.get('latent_dim', 256)
        self.input_dim = config.get('input_dim', 512)
        self.hidden_dims = config.get('hidden_dims', [512, 256])
        self.dropout = config.get('dropout', 0.1)
        self.device = config.get('device', 'cpu')

        # Build encoder network for numeric inputs
        layers = []
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.latent_dim))
        self.encoder_net = nn.Sequential(*layers).to(self.device)

        # Text embedding projection (if needed)
        self.text_projection = nn.Linear(768, self.latent_dim).to(self.device)  # Assume 768-dim embeddings

    async def encode_numeric(self, numeric_input: torch.Tensor) -> torch.Tensor:
        """
        Encode numeric agent output to latent representation.

        Args:
            numeric_input: Tensor of shape [input_dim] or [batch, input_dim]

        Returns:
            Latent representation tensor [latent_dim] or [batch, latent_dim]
        """
        if numeric_input.dim() == 1:
            numeric_input = numeric_input.unsqueeze(0)

        latent = self.encoder_net(numeric_input.to(self.device))

        if numeric_input.shape[0] == 1:
            latent = latent.squeeze(0)

        return latent

    async def encode_text(self, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Encode text embedding to latent representation.

        Args:
            text_embedding: Pre-computed text embedding [embed_dim]

        Returns:
            Latent representation tensor [latent_dim]
        """
        latent = self.text_projection(text_embedding.to(self.device))
        return latent

    async def encode_structured(self, structured_data: Dict[str, Any]) -> torch.Tensor:
        """
        Encode structured agent output (dict with mixed types) to latent.

        Args:
            structured_data: Dict containing 'numeric', 'text_embedding', etc.

        Returns:
            Latent representation tensor [latent_dim]
        """
        components = []

        if 'numeric' in structured_data:
            numeric_latent = await self.encode_numeric(structured_data['numeric'])
            components.append(numeric_latent)

        if 'text_embedding' in structured_data:
            text_latent = await self.encode_text(structured_data['text_embedding'])
            components.append(text_latent)

        if not components:
            return torch.zeros(self.latent_dim, device=self.device)

        # Average multiple components
        stacked = torch.stack(components, dim=0)
        latent = torch.mean(stacked, dim=0)
        return latent

    async def encode_agent_state(self, agent_output: Dict[str, Any]) -> torch.Tensor:
        """
        Main entry point: Encode agent output/state to latent representation.

        Automatically detects input type and applies appropriate encoding.

        Args:
            agent_output: Agent output dict from agent.act()

        Returns:
            Latent representation tensor [latent_dim]
        """
        # Determine input type and encode appropriately
        if isinstance(agent_output, torch.Tensor):
            return await self.encode_numeric(agent_output)
        elif isinstance(agent_output, dict):
            return await self.encode_structured(agent_output)
        else:
            raise ValueError(f"Unsupported agent output type: {type(agent_output)}")

    def save(self, path: str):
        """Save encoder weights."""
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Load encoder weights."""
        self.load_state_dict(torch.load(path, map_location=self.device))
```

### Update: `/Users/rajesh/athena/communication/__init__.py`
Add LatentEncoder to imports and __all__.

## Acceptance Criteria
- [ ] LatentEncoder class created as nn.Module
- [ ] `encode_numeric()` method encodes numeric inputs to latent vectors
- [ ] `encode_text()` method encodes text embeddings to latent vectors
- [ ] `encode_structured()` method handles mixed-type inputs
- [ ] `encode_agent_state()` main method auto-detects input type
- [ ] Configurable network architecture (hidden dims, dropout)
- [ ] All methods use async/await pattern
- [ ] Outputs match latent_dim from configuration
- [ ] Save/load methods for model persistence
- [ ] Class is importable and instantiable
- [ ] Docstrings present for all public methods

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
