"""
AgentStateDecoder for LatentMAS Communication
==============================================
Transforms latent representations back into agent-readable formats.

This decoder is distinct from the internal LatentDecoder in latent_space.py.
While LatentDecoder handles transformer-based reconstruction within the
latent space, AgentStateDecoder provides multi-modal decoding capabilities:
- Numeric output via MLP
- Text embedding reconstruction
- Structured multi-component output
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
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
            def no_grad():
                class DummyContext:
                    def __enter__(self):
                        return self
                    def __exit__(self, *args):
                        pass
                return DummyContext()
            @staticmethod
            def save(*args, **kwargs):
                pass
            @staticmethod
            def load(*args, **kwargs):
                pass
        class nn:  # type: ignore
            class Module:
                def __init__(self):
                    pass
                def to(self, *args, **kwargs):
                    return self
                def eval(self):
                    return self
                def state_dict(self):
                    return {}
                def load_state_dict(self, *args, **kwargs):
                    pass


if TYPE_CHECKING:
    import torch
    import torch.nn as nn


class AgentStateDecoder(nn.Module):
    """
    Decodes latent representations into agent-readable formats.

    Supports multiple decoding modes:
    - Numeric: MLP-based projection to numeric output space
    - Text embedding: Reconstruction back to text embedding space
    - Structured: Multi-component output based on specification
    """

    def __init__(
        self,
        latent_dim: int = 256,
        output_dim: int = 512,
        hidden_dims: Optional[List[int]] = None,
        text_embed_dim: int = 768,
        dropout: float = 0.1,
        device: Optional[str] = None,
    ):
        """
        Initialize AgentStateDecoder.

        Args:
            latent_dim: Dimension of input latent vectors
            output_dim: Dimension of numeric output
            hidden_dims: List of hidden layer dimensions for MLP
            text_embed_dim: Dimension of text embedding space
            dropout: Dropout probability
            device: Device for PyTorch tensors

        Raises:
            ImportError: If PyTorch is not installed
        """
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for AgentStateDecoder. "
                "Install with: pip install torch"
            )

        super().__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.text_embed_dim = text_embed_dim
        self.dropout = dropout

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if hidden_dims is None:
            hidden_dims = [256, 512]

        # Input normalization
        self.input_norm = nn.LayerNorm(latent_dim)

        # MLP decoder network: latent_dim -> hidden_dims -> output_dim
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.decoder_net = nn.Sequential(*layers)

        # Text embedding reconstruction
        self.text_reconstruction = nn.Linear(latent_dim, text_embed_dim)

        self.to(self.device)
        self.eval()

        self.logger = logging.getLogger("athena.communication.decoder")
        self.logger.info(
            f"AgentStateDecoder initialized: latent_dim={latent_dim}, "
            f"output_dim={output_dim}, text_embed_dim={text_embed_dim}, "
            f"device={self.device}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard nn.Module forward pass through MLP decoder.

        Args:
            x: Input latent tensor [batch, latent_dim] or [latent_dim]

        Returns:
            Decoded tensor [batch, output_dim] or [output_dim]
        """
        x = self.input_norm(x)
        return self.decoder_net(x)

    async def decode_to_numeric(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to numeric output via MLP.

        Args:
            latent: Latent tensor [latent_dim] or [batch, latent_dim]

        Returns:
            Numeric output tensor [output_dim] or [batch, output_dim]
        """
        with torch.no_grad():
            latent = latent.to(self.device)
            normalized = self.input_norm(latent)
            output = self.decoder_net(normalized)
            return output

    async def decode_to_text_embedding(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Project latent representation back to text embedding space.

        Args:
            latent: Latent tensor [latent_dim] or [batch, latent_dim]

        Returns:
            Text embedding tensor [text_embed_dim] or [batch, text_embed_dim]
        """
        with torch.no_grad():
            latent = latent.to(self.device)
            normalized = self.input_norm(latent)
            text_embedding = self.text_reconstruction(normalized)
            return text_embedding

    async def decode_to_structured(
        self, latent: torch.Tensor, output_spec: Dict[str, bool]
    ) -> Dict[str, torch.Tensor]:
        """
        Decode latent to multiple output components based on specification.

        Args:
            latent: Latent tensor [latent_dim] or [batch, latent_dim]
            output_spec: Dictionary specifying which outputs to generate.
                        Keys: "numeric", "text_embedding"
                        Values: True to include that output

        Returns:
            Dictionary mapping output names to decoded tensors
        """
        result = {}

        if output_spec.get("numeric", False):
            result["numeric"] = await self.decode_to_numeric(latent)

        if output_spec.get("text_embedding", False):
            result["text_embedding"] = await self.decode_to_text_embedding(latent)

        return result

    async def interpret_message(
        self, latent: torch.Tensor, mode: str = "numeric"
    ) -> Any:
        """
        Main entry point for decoding latent messages.

        Dispatches to appropriate decoding method based on mode.

        Args:
            latent: Latent tensor [latent_dim] or [batch, latent_dim]
            mode: Decoding mode - "numeric", "text_embedding", or "structured"

        Returns:
            Decoded output in requested format

        Raises:
            ValueError: If mode is not recognized
        """
        if mode == "numeric":
            return await self.decode_to_numeric(latent)
        elif mode == "text_embedding":
            return await self.decode_to_text_embedding(latent)
        elif mode == "structured":
            # Default structured output includes both numeric and text
            output_spec = {"numeric": True, "text_embedding": True}
            return await self.decode_to_structured(latent, output_spec)
        else:
            raise ValueError(
                f"Unknown decoding mode: {mode}. "
                f"Valid modes: 'numeric', 'text_embedding', 'structured'"
            )

    async def decode_messages(
        self, latent_messages: List[torch.Tensor], mode: str = "numeric"
    ) -> List[Any]:
        """
        Batch decode multiple latent messages.

        Args:
            latent_messages: List of latent tensors, each [latent_dim]
            mode: Decoding mode - "numeric", "text_embedding", or "structured"

        Returns:
            List of decoded outputs in requested format
        """
        results = []
        for latent in latent_messages:
            decoded = await self.interpret_message(latent, mode=mode)
            results.append(decoded)
        return results

    def save(self, path: str) -> None:
        """
        Save model state to disk.

        Args:
            path: Path to save model state
        """
        torch.save({
            "state_dict": self.state_dict(),
            "latent_dim": self.latent_dim,
            "output_dim": self.output_dim,
            "text_embed_dim": self.text_embed_dim,
            "dropout": self.dropout,
        }, path)
        self.logger.info(f"AgentStateDecoder saved to {path}")

    def load(self, path: str) -> None:
        """
        Load model state from disk.

        Args:
            path: Path to load model state from
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.load_state_dict(checkpoint["state_dict"])
        self.logger.info(f"AgentStateDecoder loaded from {path}")
