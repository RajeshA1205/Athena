"""
LatentMAS Agent State Encoder
==============================
Encoder component that transforms agent outputs and states into latent
representations for communication in the shared latent space.

Based on: Latent Collaboration for Multi-Agent Systems
"""

from typing import Any, Dict, Optional, TYPE_CHECKING
import logging

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    if not TYPE_CHECKING:
        # Define dummy classes when torch is not available
        class torch:  # type: ignore
            Tensor = Any
            device = Any
            @staticmethod
            def zeros(*args, **kwargs):
                pass
        class nn:  # type: ignore
            class Module:
                pass


if TYPE_CHECKING:
    import torch
    import torch.nn as nn


class AgentStateEncoder(nn.Module):
    """
    Encoder for transforming agent outputs into latent representations.

    Supports encoding of:
    - Numeric outputs (via learned MLP projection)
    - Text outputs (via embedding projection)
    - Structured outputs (via learned aggregation)

    This encoder is distinct from the internal LatentEncoder used in
    LatentSpace for message encoding. AgentStateEncoder transforms agent
    outputs/states into latent representations before communication.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        input_dim: int = 512,
        hidden_dims: Optional[list] = None,
        text_embed_dim: int = 768,
        dropout: float = 0.1,
        device: Optional[str] = None,
    ):
        """
        Initialize agent state encoder.

        Args:
            latent_dim: Target latent dimension
            input_dim: Dimension of numeric inputs
            hidden_dims: Hidden layer dimensions for MLP
            text_embed_dim: Dimension of text embeddings
            dropout: Dropout rate
            device: PyTorch device (auto-detected if None)
        """
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for AgentStateEncoder. "
                "Install with: pip install torch"
            )

        super().__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [512, 256]
        self.text_embed_dim = text_embed_dim
        self.dropout = dropout

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._build_encoder_net()
        self._build_projections()

        self.logger = logging.getLogger("athena.communication.encoder")
        self.logger.info(
            f"AgentStateEncoder initialized: latent_dim={latent_dim}, "
            f"input_dim={input_dim}, device={self.device}"
        )

        self.eval()

    def _build_encoder_net(self) -> None:
        """Build MLP encoder network for numeric inputs."""
        layers = []
        prev_dim = self.input_dim

        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.latent_dim))
        self.encoder_net = nn.Sequential(*layers).to(self.device)

    def _build_projections(self) -> None:
        """Build projection layers for different modalities."""
        self.text_projection = nn.Linear(
            self.text_embed_dim, self.latent_dim
        ).to(self.device)

        self.output_norm = nn.LayerNorm(self.latent_dim).to(self.device)

    async def encode_numeric(self, numeric_input: torch.Tensor) -> torch.Tensor:
        """
        Encode numeric agent output to latent representation.

        Args:
            numeric_input: Tensor of shape [input_dim] or [batch, input_dim]

        Returns:
            Latent representation tensor [latent_dim] or [batch, latent_dim]
        """
        with torch.no_grad():
            is_single = numeric_input.dim() == 1
            if is_single:
                numeric_input = numeric_input.unsqueeze(0)

            latent = self.encoder_net(numeric_input.to(self.device))
            latent = self.output_norm(latent)

            if is_single:
                latent = latent.squeeze(0)

            return latent

    async def encode_text(self, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Encode text embedding to latent representation.

        Args:
            text_embedding: Pre-computed text embedding [text_embed_dim] or [batch, text_embed_dim]

        Returns:
            Latent representation tensor [latent_dim] or [batch, latent_dim]
        """
        with torch.no_grad():
            is_single = text_embedding.dim() == 1
            if is_single:
                text_embedding = text_embedding.unsqueeze(0)

            latent = self.text_projection(text_embedding.to(self.device))
            latent = self.output_norm(latent)

            if is_single:
                latent = latent.squeeze(0)

            return latent

    async def encode_structured(self, structured_data: Dict[str, Any]) -> torch.Tensor:
        """
        Encode structured agent output (dict with mixed types) to latent.

        Args:
            structured_data: Dict containing keys like:
                - 'numeric': torch.Tensor with numeric data
                - 'text_embedding': torch.Tensor with text embedding

        Returns:
            Latent representation tensor [latent_dim]
        """
        with torch.no_grad():
            components = []

            if "numeric" in structured_data:
                numeric_latent = await self.encode_numeric(structured_data["numeric"])
                components.append(numeric_latent)

            if "text_embedding" in structured_data:
                text_latent = await self.encode_text(structured_data["text_embedding"])
                components.append(text_latent)

            if not components:
                self.logger.warning("No valid components in structured data, returning zero vector")
                return torch.zeros(self.latent_dim, device=self.device)

            if len(components) == 1:
                return components[0]

            stacked = torch.stack(components, dim=0)
            latent = torch.mean(stacked, dim=0)

            return latent

    async def encode_agent_state(self, agent_output: Any) -> torch.Tensor:
        """
        Main entry point: Encode agent output/state to latent representation.

        Automatically detects input type and applies appropriate encoding.

        Args:
            agent_output: Agent output which can be:
                - torch.Tensor: treated as numeric input
                - Dict[str, Any]: treated as structured input
                - str: placeholder for text (requires embedding first)

        Returns:
            Latent representation tensor [latent_dim]
        """
        if isinstance(agent_output, torch.Tensor):
            return await self.encode_numeric(agent_output)
        elif isinstance(agent_output, dict):
            # Check if dict has known structured keys; otherwise serialize to string
            if "numeric" in agent_output or "text_embedding" in agent_output:
                return await self.encode_structured(agent_output)
            else:
                import json
                serialized = json.dumps(agent_output, default=str)
                char_codes = [float(ord(c)) / 255.0 for c in serialized[: self.input_dim]]
                while len(char_codes) < self.input_dim:
                    char_codes.append(0.0)
                text_tensor = torch.tensor(char_codes, dtype=torch.float32, device=self.device)
                return await self.encode_numeric(text_tensor)
        elif isinstance(agent_output, str):
            # Convert string to normalized character codes, pad/truncate to input_dim
            char_codes = [float(ord(c)) / 255.0 for c in agent_output[: self.input_dim]]
            while len(char_codes) < self.input_dim:
                char_codes.append(0.0)
            text_tensor = torch.tensor(char_codes, dtype=torch.float32, device=self.device)
            return await self.encode_numeric(text_tensor)
        else:
            raise ValueError(
                f"Unsupported agent output type: {type(agent_output)}. "
                f"Expected torch.Tensor, dict, or str."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard nn.Module forward pass through encoder network.

        Args:
            x: Input tensor [batch, input_dim] or [input_dim]

        Returns:
            Encoded latent tensor [batch, latent_dim] or [latent_dim]
        """
        is_single = x.dim() == 1
        if is_single:
            x = x.unsqueeze(0)

        latent = self.encoder_net(x.to(self.device))
        latent = self.output_norm(latent)

        if is_single:
            latent = latent.squeeze(0)

        return latent

    def save(self, path: str) -> None:
        """
        Save encoder weights to file.

        Args:
            path: Path to save model weights
        """
        torch.save(self.state_dict(), path)
        self.logger.info(f"Encoder saved to {path}")

    def load(self, path: str) -> None:
        """
        Load encoder weights from file.

        Args:
            path: Path to load model weights from
        """
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.eval()
        self.logger.info(f"Encoder loaded from {path}")
