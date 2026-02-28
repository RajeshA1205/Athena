# TASK-008: Implement LatentMAS Decoder

## Status
- **State:** Queued
- **Priority:** 🟡 High
- **Depends on:** TASK-006
- **Created:** 2026-02-15

## Objective
Create the decoder component that transforms latent representations received from the shared latent space back into interpretable agent inputs.

## Context
The decoder is the complement to the encoder in LatentMAS. It takes latent vectors from other agents and projects them back into a form that the receiving agent can understand and use for decision-making.

This component provides:
- Latent vector decoding to agent-readable format
- Support for different output modalities
- Integration with agent observation space
- Message interpretation for decision-making

Reference the LatentMAS paper in `/Users/rajesh/athena/architecture/base/` and the architecture plan at `/Users/rajesh/athena/agent-plan/atomic-wibbling-simon.md` (lines 73-77, 161-164).

## Scope & Constraints

### Files to Create
- `/Users/rajesh/athena/communication/decoder.py`

### Files to Modify
- `/Users/rajesh/athena/communication/__init__.py` — Add Decoder import

### Files to Reference (DO NOT MODIFY)
- `/Users/rajesh/athena/communication/latent_space.py` — LatentSpace
- `/Users/rajesh/athena/communication/encoder.py` — LatentEncoder
- `/Users/rajesh/athena/models/embeddings.py` — Embedding models

### Constraints
- Must work with LatentSpace from TASK-006
- Use PyTorch for decoding networks
- Support multiple output types (numeric, text, structured)
- Use **async/await** for all operations
- No training logic yet (that comes in Sprint 4)

## Input
- LatentSpace implementation from TASK-006
- LatentEncoder implementation from TASK-007
- LatentMAS paper specification
- PyTorch framework

## Expected Output

### File: `/Users/rajesh/athena/communication/decoder.py`
```python
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import asyncio

class LatentDecoder(nn.Module):
    """
    Decoder for transforming latent representations into agent-readable outputs.

    Supports decoding to:
    - Numeric outputs (via learned projection)
    - Text outputs (via embedding projection)
    - Structured outputs (via multiple heads)

    Based on: Latent Collaboration for Multi-Agent Systems
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize decoder.

        Args:
            config: Configuration dict containing:
                - latent_dim: Input latent dimension (default: 256)
                - output_dim: Dimension of numeric outputs (default: 512)
                - hidden_dims: Hidden layer dimensions (default: [256, 512])
                - dropout: Dropout rate (default: 0.1)
                - device: PyTorch device (default: 'cpu')
        """
        super().__init__()

        self.latent_dim = config.get('latent_dim', 256)
        self.output_dim = config.get('output_dim', 512)
        self.hidden_dims = config.get('hidden_dims', [256, 512])
        self.dropout = config.get('dropout', 0.1)
        self.device = config.get('device', 'cpu')

        # Build decoder network for numeric outputs
        layers = []
        prev_dim = self.latent_dim
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.decoder_net = nn.Sequential(*layers).to(self.device)

        # Text embedding reconstruction (if needed)
        self.text_reconstruction = nn.Linear(self.latent_dim, 768).to(self.device)

    async def decode_to_numeric(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to numeric output.

        Args:
            latent: Latent tensor [latent_dim] or [batch, latent_dim]

        Returns:
            Numeric output tensor [output_dim] or [batch, output_dim]
        """
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)

        numeric = self.decoder_net(latent.to(self.device))

        if latent.shape[0] == 1:
            numeric = numeric.squeeze(0)

        return numeric

    async def decode_to_text_embedding(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to text embedding space.

        Args:
            latent: Latent tensor [latent_dim]

        Returns:
            Text embedding tensor [768]
        """
        text_embedding = self.text_reconstruction(latent.to(self.device))
        return text_embedding

    async def decode_to_structured(self, latent: torch.Tensor, output_spec: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """
        Decode latent to structured output with multiple components.

        Args:
            latent: Latent tensor [latent_dim]
            output_spec: Dict specifying desired outputs, e.g., {'numeric': True, 'text': True}

        Returns:
            Dict with decoded components
        """
        result = {}

        if output_spec.get('numeric', False):
            result['numeric'] = await self.decode_to_numeric(latent)

        if output_spec.get('text', False):
            result['text_embedding'] = await self.decode_to_text_embedding(latent)

        return result

    async def interpret_message(self, latent: torch.Tensor, interpretation_mode: str = 'numeric') -> Any:
        """
        Main entry point: Interpret latent message from another agent.

        Args:
            latent: Latent representation from another agent [latent_dim]
            interpretation_mode: How to interpret ('numeric', 'text', 'structured')

        Returns:
            Decoded message in requested format
        """
        if interpretation_mode == 'numeric':
            return await self.decode_to_numeric(latent)
        elif interpretation_mode == 'text':
            return await self.decode_to_text_embedding(latent)
        elif interpretation_mode == 'structured':
            return await self.decode_to_structured(latent, {'numeric': True, 'text': True})
        else:
            raise ValueError(f"Unknown interpretation mode: {interpretation_mode}")

    async def decode_messages(self, latent_messages: List[torch.Tensor], mode: str = 'numeric') -> List[Any]:
        """
        Decode multiple latent messages.

        Args:
            latent_messages: List of latent tensors
            mode: Interpretation mode

        Returns:
            List of decoded messages
        """
        decoded = []
        for latent in latent_messages:
            decoded_msg = await self.interpret_message(latent, mode)
            decoded.append(decoded_msg)
        return decoded

    def save(self, path: str):
        """Save decoder weights."""
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Load decoder weights."""
        self.load_state_dict(torch.load(path, map_location=self.device))
```

### Update: `/Users/rajesh/athena/communication/__init__.py`
Add LatentDecoder to imports and __all__.

## Acceptance Criteria
- [ ] LatentDecoder class created as nn.Module
- [ ] `decode_to_numeric()` method decodes latent vectors to numeric outputs
- [ ] `decode_to_text_embedding()` method decodes to text embedding space
- [ ] `decode_to_structured()` method handles multi-component outputs
- [ ] `interpret_message()` main method with configurable interpretation mode
- [ ] `decode_messages()` batch processing for multiple messages
- [ ] Configurable network architecture (hidden dims, dropout)
- [ ] All methods use async/await pattern
- [ ] Save/load methods for model persistence
- [ ] Class is importable and instantiable
- [ ] Docstrings present for all public methods

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)
