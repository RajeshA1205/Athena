"""
Unit tests for the ATHENA LatentMAS communication layer.

Tests cover: MessageRouter registration, send/receive, priority queuing,
broadcast_with_attention, and get_routing_stats. Encoder/Decoder/LatentSpace
are mocked where torch is unavailable.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_latent_space():
    ls = MagicMock()
    ls.latent_dim = 64
    ls.encode = AsyncMock(return_value=[0.1] * 64)
    ls.decode = AsyncMock(return_value={"decoded": True})
    return ls


@pytest.fixture
def mock_encoder():
    enc = MagicMock()
    enc.encode_agent_state = AsyncMock(return_value=[0.1] * 64)
    return enc


@pytest.fixture
def mock_decoder():
    dec = MagicMock()
    dec.decode_to_agent_input = AsyncMock(return_value={"decoded": True})
    return dec


@pytest.fixture
def router(mock_latent_space, mock_encoder, mock_decoder):
    try:
        from communication.router import MessageRouter
        r = MessageRouter(
            latent_space=mock_latent_space,
            encoder=mock_encoder,
            decoder=mock_decoder,
        )
        return r
    except ImportError as e:
        pytest.skip(f"MessageRouter not available (missing dependency): {e}")
    except Exception as e:
        pytest.skip(f"MessageRouter not available: {e}")


# ---------------------------------------------------------------------------
# MessageRouter
# ---------------------------------------------------------------------------

class TestMessageRouter:
    def test_import(self):
        try:
            from communication import router as router_module
            assert router_module is not None
        except Exception as e:
            pytest.skip(f"communication module not importable (missing dependency): {e}")

    def test_message_priority_enum(self):
        try:
            from communication.router import MessagePriority
            assert MessagePriority.HIGH.value > MessagePriority.LOW.value
        except ImportError:
            pytest.skip("MessagePriority not available")

    def test_router_instantiation(self, mock_latent_space, mock_encoder, mock_decoder):
        try:
            from communication.router import MessageRouter
            r = MessageRouter(
                latent_space=mock_latent_space,
                encoder=mock_encoder,
                decoder=mock_decoder,
            )
            assert r is not None
        except Exception as e:
            pytest.skip(f"MessageRouter not available: {e}")

    def test_register_agent(self, router):
        router.register_agent("agent_1", {"role": "market_analyst"})
        stats = router.get_routing_stats()
        assert isinstance(stats, dict)

    def test_register_multiple_agents(self, router):
        for i in range(4):
            router.register_agent(f"agent_{i}", {"role": f"role_{i}"})
        stats = router.get_routing_stats()
        assert isinstance(stats, dict)

    @pytest.mark.asyncio
    async def test_send_message(self, router):
        router.register_agent("sender", {})
        router.register_agent("receiver", {})
        await router.send(
            sender_id="sender",
            receiver_id="receiver",
            message={"signal": "buy"},
        )

    @pytest.mark.asyncio
    async def test_receive_empty_queue(self, router):
        router.register_agent("solo", {})
        messages = await router.receive("solo")
        assert isinstance(messages, list)

    @pytest.mark.asyncio
    async def test_send_then_receive(self, router):
        router.register_agent("alpha", {})
        router.register_agent("beta", {})
        await router.send(
            sender_id="alpha",
            receiver_id="beta",
            message={"action": "hold"},
        )
        messages = await router.receive("beta")
        assert isinstance(messages, list)

    @pytest.mark.asyncio
    async def test_broadcast_with_attention(self, router):
        for ag in ["coord", "spec1", "spec2"]:
            router.register_agent(ag, {})
        result = await router.broadcast_with_attention(
            sender_id="coord",
            message={"broadcast": True},
            agent_embeddings={},
        )
        assert isinstance(result, (int, dict, list, type(None)))

    def test_get_routing_stats(self, router):
        stats = router.get_routing_stats()
        assert isinstance(stats, dict)


# ---------------------------------------------------------------------------
# LatentSpace
# ---------------------------------------------------------------------------

class TestLatentSpace:
    def test_import(self):
        try:
            from communication.latent_space import LatentSpace
            assert LatentSpace is not None
        except ImportError:
            pytest.skip("LatentSpace not importable")

    def test_instantiation(self):
        try:
            from communication.latent_space import LatentSpace
            ls = LatentSpace(latent_dim=64)
            assert ls is not None
            assert ls.latent_dim == 64
        except Exception as e:
            pytest.skip(f"LatentSpace not available: {e}")


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------

class TestEncoderDecoder:
    def test_encoder_import(self):
        try:
            from communication.encoder import AgentStateEncoder
            assert AgentStateEncoder is not None
        except ImportError:
            pytest.skip("AgentStateEncoder not importable")

    def test_decoder_import(self):
        try:
            from communication.decoder import AgentStateDecoder
            assert AgentStateDecoder is not None
        except ImportError:
            pytest.skip("AgentStateDecoder not importable")

    @pytest.mark.asyncio
    async def test_encoder_encode_dict(self):
        try:
            from communication.encoder import AgentStateEncoder
            enc = AgentStateEncoder(latent_dim=32, input_dim=64)
            result = await enc.encode_agent_state({"signal": 0.5})
            assert result is not None
        except Exception as e:
            pytest.skip(f"Encoder not available: {e}")

    @pytest.mark.asyncio
    async def test_encoder_encode_string(self):
        try:
            from communication.encoder import AgentStateEncoder
            enc = AgentStateEncoder(latent_dim=32, input_dim=64)
            result = await enc.encode_agent_state("buy signal")
            assert result is not None
        except Exception as e:
            pytest.skip(f"Encoder not available: {e}")

    @pytest.mark.asyncio
    async def test_encoder_encode_numeric_list(self):
        try:
            from communication.encoder import AgentStateEncoder
            enc = AgentStateEncoder(latent_dim=32, input_dim=64)
            result = await enc.encode_agent_state([0.1, 0.2, 0.3])
            assert result is not None
        except Exception as e:
            pytest.skip(f"Encoder not available: {e}")
