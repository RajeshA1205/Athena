"""
ATHENA Communication Layer
===========================
LatentMAS-based inter-agent communication.
"""

from .latent_space import LatentSpace
from .encoder import AgentStateEncoder
from .decoder import AgentStateDecoder
from .router import MessageRouter, MessagePriority

__all__ = ['LatentSpace', 'AgentStateEncoder', 'AgentStateDecoder', 'MessageRouter', 'MessagePriority']
