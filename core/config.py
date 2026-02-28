"""
ATHENA Configuration System
===========================
Unified configuration for all ATHENA layers and components.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import yaml


@dataclass
class ModelConfig:
    """Configuration for the foundation model (OLMo 3)."""
    model_name: str = "allenai/OLMo-1B"
    model_path: Optional[str] = None
    device: str = "auto"
    dtype: str = "float16"
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50


@dataclass
class MemoryConfig:
    """Configuration for AgeMem memory layer."""
    # Long-term memory settings
    ltm_vector_dim: int = 768
    ltm_max_entries: int = 100000
    ltm_index_type: str = "faiss"  # faiss or chroma
    ltm_similarity_threshold: float = 0.7

    # Short-term memory settings
    stm_buffer_size: int = 10
    stm_context_window: int = 4096
    stm_summary_ratio: float = 0.3

    # Memory operations
    retrieval_top_k: int = 5
    update_strategy: str = "merge"  # merge, replace, append


@dataclass
class CommunicationConfig:
    """Configuration for LatentMAS communication layer."""
    latent_dim: int = 512
    num_attention_heads: int = 8
    message_queue_size: int = 100
    broadcast_enabled: bool = True
    priority_channels: int = 3
    encoding_method: str = "transformer"  # transformer, mlp


@dataclass
class EvolutionConfig:
    """Configuration for AgentEvolver evolution layer."""
    workflow_library_size: int = 1000
    discovery_interval: int = 100  # steps between workflow discovery
    population_size: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.3
    experience_replay_size: int = 10000


@dataclass
class LearningConfig:
    """Configuration for Nested Learning + RepExp layer."""
    # Nested learning settings
    inner_lr: float = 1e-4
    outer_lr: float = 1e-5
    inner_steps: int = 5

    # RepExp settings
    exploration_coefficient: float = 0.1
    diversity_threshold: float = 0.5
    representation_dim: int = 256


@dataclass
class TradingConfig:
    """Configuration for trading domain."""
    markets: List[str] = field(default_factory=lambda: ["stocks"])
    data_source: str = "mock"  # mock, yahoo, alpaca
    position_limit: float = 100000.0
    max_drawdown: float = 0.1
    risk_free_rate: float = 0.02


@dataclass
class OLMoEIntegrationConfig:
    """Controls OLMoE LLM backend wiring into agents."""
    enabled: bool = False                          # off by default (no transformers needed)
    model_name: str = "allenai/OLMoE-1B-7B-0924"
    device: str = "auto"
    dtype: str = "float16"
    load_in_4bit: bool = False
    max_length: int = 2048
    lora_adapter_path: Optional[str] = None        # path to fine-tuned LoRA weights

    # MLX backend settings (Apple Silicon)
    use_mlx: bool = True           # prefer mlx-lm when available
    mlx_model_path: Optional[str] = None  # local path or HF repo for mlx weights


@dataclass
class AgentConfig:
    """Configuration for individual agents."""
    name: str = "agent"
    role: str = "general"
    system_prompt: str = ""
    tools: List[str] = field(default_factory=list)
    memory_enabled: bool = True
    communication_enabled: bool = True


@dataclass
class AthenaConfig:
    """Master configuration for ATHENA system."""
    # Component configs
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    olmoe: OLMoEIntegrationConfig = field(default_factory=OLMoEIntegrationConfig)

    # Agent configs
    agents: Dict[str, AgentConfig] = field(default_factory=dict)

    # System settings
    log_level: str = "INFO"
    checkpoint_dir: str = "./checkpoints"
    data_dir: str = "./data"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "AthenaConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def from_json(cls, path: str) -> "AthenaConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "AthenaConfig":
        """Create config from dictionary."""
        config = cls()

        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "memory" in data:
            config.memory = MemoryConfig(**data["memory"])
        if "communication" in data:
            config.communication = CommunicationConfig(**data["communication"])
        if "evolution" in data:
            config.evolution = EvolutionConfig(**data["evolution"])
        if "learning" in data:
            config.learning = LearningConfig(**data["learning"])
        if "trading" in data:
            config.trading = TradingConfig(**data["trading"])
        if "olmoe" in data:
            config.olmoe = OLMoEIntegrationConfig(**data["olmoe"])
        if "agents" in data:
            config.agents = {
                name: AgentConfig(**cfg)
                for name, cfg in data["agents"].items()
            }

        # System settings
        for key in ["log_level", "checkpoint_dir", "data_dir", "seed"]:
            if key in data:
                setattr(config, key, data[key])

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)

    def save_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def save_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def get_default_agent_configs() -> Dict[str, AgentConfig]:
    """Get default configurations for all ATHENA agents."""
    return {
        "market_analyst": AgentConfig(
            name="market_analyst",
            role="analyst",
            system_prompt="""You are a Market Analyst agent specializing in:
- Real-time market data analysis
- Pattern recognition and technical analysis
- Sentiment analysis from news and social media
- Market trend identification and forecasting

Provide detailed, data-driven analysis to support trading decisions.""",
            tools=["analyze_price", "detect_patterns", "analyze_sentiment"],
        ),
        "risk_manager": AgentConfig(
            name="risk_manager",
            role="risk",
            system_prompt="""You are a Risk Manager agent responsible for:
- Portfolio risk assessment and monitoring
- Exposure analysis across positions
- Compliance verification
- Risk limit enforcement

Ensure all trading activities stay within acceptable risk parameters.""",
            tools=["calculate_var", "check_exposure", "verify_compliance"],
        ),
        "strategy_agent": AgentConfig(
            name="strategy_agent",
            role="strategy",
            system_prompt="""You are a Strategy Agent focused on:
- Trading strategy formulation
- Backtesting and optimization
- Signal generation
- Strategy performance analysis

Develop and refine trading strategies based on market conditions.""",
            tools=["generate_signals", "backtest", "optimize_params"],
        ),
        "execution_agent": AgentConfig(
            name="execution_agent",
            role="execution",
            system_prompt="""You are an Execution Agent handling:
- Order execution and management
- Timing optimization
- Slippage minimization
- Transaction cost analysis

Execute trades efficiently while minimizing market impact.""",
            tools=["place_order", "cancel_order", "get_execution_stats"],
        ),
        "coordinator": AgentConfig(
            name="coordinator",
            role="coordinator",
            system_prompt="""You are the Coordinator agent responsible for:
- Orchestrating all other agents
- Conflict resolution between agents
- Resource allocation
- Final decision making

Ensure smooth collaboration between all agents for optimal outcomes.""",
            tools=["delegate_task", "resolve_conflict", "allocate_resources"],
        ),
    }
