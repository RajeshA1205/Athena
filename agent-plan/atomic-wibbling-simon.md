# ATHENA Multi-Agent System Implementation Plan

## Overview

ATHENA is a multi-agent financial trading system built on **OLMoE 1B** (Mixture of Experts) with 5 specialized agents and 5 architectural layers.

**Key modifications from original architecture:**
- Using **AgeMem + Graphiti** instead of GAM for the memory layer
- Using **OLMoE 1B** (MoE) instead of OLMo 3 as foundation model
- **Staged training**: Fine-tune on finance data first, then AgeMem GRPO

---

## Architecture Summary

### Specialized Agents
1. **Market Analyst Agent** - Real-time market analysis, pattern recognition, sentiment analysis
2. **Risk Manager Agent** - Portfolio risk assessment, exposure monitoring, compliance
3. **Strategy Agent** - Trading strategy formulation, backtesting, optimization
4. **Execution Agent** - Order execution, timing optimization, slippage minimization
5. **Coordinator Agent** - Orchestration, conflict resolution, resource allocation

### 5 Architectural Layers
| Layer | Component | Research Paper |
|-------|-----------|----------------|
| Agent Layer | OLMo 3 based specialized agents | ATHENA Architecture |
| Memory Layer | **AgeMem + Graphiti (Zep)** | AgeMem paper |
| Communication Layer | LatentMAS | Latent Collaboration paper |
| Evolution Layer | AgentEvolver | AgentEvolver paper |
| Learning Layer | Nested Learning + RepExp | RepExp paper |

### Memory Layer Architecture (AgeMem + Graphiti)
```
┌─────────────────────────────────────────────────────────┐
│                    AgeMem Layer                         │
│  Logical operations: ADD, UPDATE, DELETE, RETRIEVE,     │
│                      SUMMARY, FILTER                    │
│  + Step-wise GRPO RL Training                          │
├─────────────────────────────────────────────────────────┤
│                  Graphiti (Zep)                         │
│  Storage layer: Temporal knowledge graph, entity        │
│  extraction, episodic memory, semantic search           │
└─────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Foundation Setup
**Goal**: Set up the base infrastructure and OLMo 3 integration

1. **Project Structure**
   ```
   athena/
   ├── core/                    # Core framework
   │   ├── base_agent.py        # Abstract base agent class
   │   ├── config.py            # System configuration
   │   └── utils.py             # Shared utilities
   ├── models/                  # Model management
   │   ├── olmoe.py             # OLMoE 1B loader and adapter
   │   └── embeddings.py        # Embedding models
   ├── agents/                  # Specialized agents
   │   ├── market_analyst.py
   │   ├── risk_manager.py
   │   ├── strategy_agent.py
   │   ├── execution_agent.py
   │   └── coordinator.py
   ├── memory/                  # AgeMem + Graphiti (NO training code)
   │   ├── agemem.py            # Main AgeMem controller (logical layer)
   │   ├── graphiti_backend.py  # Graphiti adapter for storage
   │   └── operations.py        # LTM/STM operation implementations
   ├── communication/           # LatentMAS implementation
   │   ├── latent_space.py      # Shared latent space
   │   ├── encoder.py           # Message encoding
   │   └── decoder.py           # Message decoding
   ├── evolution/               # AgentEvolver implementation
   │   ├── workflow_discovery.py
   │   ├── agent_generator.py
   │   └── cooperative_evolution.py
   ├── learning/                # Nested Learning + RepExp
   │   ├── nested_learning.py
   │   └── repexp.py            # Representation-based exploration
   ├── training/                # CENTRALIZED TRAINING (all training code here)
   │   ├── stage1_finetune/     # Stage 1: OLMoE finance fine-tuning
   │   │   ├── finetune.py      # Fine-tuning script
   │   │   ├── lora.py          # LoRA adapter training
   │   │   └── config.py        # Training configs
   │   ├── stage2_agemem/       # Stage 2: AgeMem GRPO training
   │   │   ├── grpo.py          # Step-wise GRPO implementation
   │   │   ├── rewards.py       # Composite reward function
   │   │   └── trainer.py       # AgeMem trainer
   │   └── data/                # Data collection & processing
   │       ├── scrapers/        # Web scrapers for finance data
   │       │   ├── news.py      # News & SEC filings
   │       │   ├── market.py    # Market data (prices, indicators)
   │       │   └── social.py    # Reddit, Twitter, sentiment
   │       ├── processors/      # Data preprocessing
   │       │   ├── cleaner.py   # Text cleaning
   │       │   └── formatter.py # Format for training
   │       └── datasets.py      # PyTorch datasets
   ├── trading/                 # Trading-specific components
   │   ├── market_data.py
   │   ├── order_management.py
   │   └── portfolio.py
   └── tests/
   ```

2. **OLMoE 1B Integration**
   - Set up OLMoE 1B model loading (Mixture of Experts)
   - Configure inference pipeline with expert routing
   - Implement model adapter for agent specialization
   - Support for LoRA fine-tuning on finance data

### Phase 2: AgeMem Memory Layer (with Graphiti Backend)
**Goal**: Implement unified LTM/STM management based on AgeMem paper, using Graphiti as storage

#### 2.1 Graphiti Setup
```bash
pip install graphiti-core
# Requires Neo4j database
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

#### 2.2 AgeMem Operations (wrapping Graphiti)
1. **LTM Operations** (mapped to Graphiti)
   - `ADD`: → `graphiti.add_episode()` - Store memories as episodes with entities/relations
   - `UPDATE`: → `graphiti.update_episode()` - Modify existing episodes
   - `DELETE`: → `graphiti.delete_episode()` - Remove from knowledge graph

2. **STM Operations** (AgeMem logic layer)
   - `RETRIEVE`: → `graphiti.search()` - Semantic search with temporal awareness
   - `SUMMARY`: → LLM-based compression of retrieved context
   - `FILTER`: → Relevance scoring and pruning

#### 2.3 Step-wise GRPO RL Training (Full AgeMem)
1. **Three-stage progressive training**:
   - Stage 1: Single-tool learning (one operation at a time)
   - Stage 2: Multi-tool coordination (operation sequences)
   - Stage 3: Full unified management (autonomous memory decisions)

2. **Step-wise GRPO Implementation**:
   - Group Relative Policy Optimization at each decision step
   - Reward: R = α·R_task + β·R_efficiency + γ·R_quality
   - Training loop with experience replay from Graphiti episodes

3. **Composite Reward Function**:
   - R_task: Task completion success
   - R_efficiency: Memory operation speed/cost
   - R_quality: Relevance of retrieved context

### Phase 3: LatentMAS Communication Layer
**Goal**: Implement embedding-based inter-agent communication

1. **Latent Space Setup**
   - Shared embedding space for all agents
   - Dimension alignment across agent representations

2. **Communication Protocol**
   - Encode agent outputs to latent representations
   - Broadcast/receive in shared latent space
   - Decode latent representations for receiving agents

3. **Collaboration Mechanisms**
   - Attention-based message routing
   - Priority-based communication channels
   - Conflict detection in latent space

### Phase 4: AgentEvolver Evolution Layer
**Goal**: Enable self-evolving agent capabilities

1. **Workflow Discovery**
   - Analyze historical trading data/decisions
   - Extract successful workflow patterns
   - Build workflow library

2. **Agent Generation**
   - Automatic generation of new agent configurations
   - Task-specific agent instantiation
   - Performance-based agent selection

3. **Cooperative Evolution**
   - Multi-agent experience replay
   - Cross-agent knowledge distillation
   - Population-based training

### Phase 5: Nested Learning + RepExp Learning Layer
**Goal**: Implement continual learning with diversity

1. **Nested Learning Framework**
   - Inner loop: Task-specific adaptation
   - Outer loop: Meta-learning across tasks
   - Knowledge consolidation mechanisms

2. **RepExp Integration**
   - Representation-space diversity measurement
   - Exploration bonus computation
   - Test-time exploration strategies
   - Post-training exploration for capability expansion

### Phase 6: Trading Domain Integration
**Goal**: Connect agents to financial trading infrastructure

1. **Market Data Pipeline**
   - Real-time data feeds integration
   - Historical data storage
   - Feature engineering pipeline

2. **Order Management**
   - Order execution interface
   - Position tracking
   - P&L calculation

3. **Risk Controls**
   - Position limits
   - Drawdown monitoring
   - Circuit breakers

### Phase 7: Integration & Testing
**Goal**: End-to-end system integration

1. **Agent Orchestration**
   - Coordinator-driven workflow
   - Agent lifecycle management
   - Resource allocation

2. **Testing Strategy**
   - Unit tests per component
   - Integration tests for layer interactions
   - Backtesting on historical data
   - Paper trading validation

---

## Key Technical Decisions

### AgeMem vs GAM Comparison
| Aspect | GAM (Original) | AgeMem (New) |
|--------|----------------|--------------|
| Architecture | Memorizer + Page-Store + Researcher | Unified LTM/STM tools |
| Operations | JIT context generation | ADD/UPDATE/DELETE + RETRIEVE/SUMMARY/FILTER |
| Training | Supervised | Three-stage progressive RL |
| Optimization | - | Step-wise GRPO |

### Technology Stack (Recommended)
- **Foundation Model**: OLMoE 1B (Mixture of Experts) via HuggingFace Transformers
- **Fine-tuning**: LoRA/QLoRA for parameter-efficient training
- **Memory Backend**: Graphiti (Zep) + Neo4j for knowledge graph storage
- **Memory Logic**: AgeMem operations (training in centralized module)
- **Communication**: PyTorch tensors for latent space
- **Training**: PyTorch + DeepSpeed/Accelerate for distributed training
- **Data Collection**: BeautifulSoup, Scrapy, yfinance, praw (Reddit API)
- **Data Processing**: Pandas + NumPy for market data
- **Testing**: pytest + unittest

### Staged Training Pipeline
```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: Finance Fine-tuning                 │
├─────────────────────────────────────────────────────────────────┤
│  Data Sources:                                                  │
│  • News & Articles (Reuters, Bloomberg, SEC filings)            │
│  • Market Data (prices, volume, technical indicators)           │
│  • Social/Sentiment (Reddit r/wallstreetbets, Twitter, reports) │
│                                                                 │
│  Training:                                                      │
│  • OLMoE 1B base model                                          │
│  • LoRA fine-tuning on finance corpus                           │
│  • Causal LM objective + domain-specific tasks                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 2: AgeMem GRPO Training                │
├─────────────────────────────────────────────────────────────────┤
│  Uses fine-tuned OLMoE as base                                  │
│                                                                 │
│  Three-stage progressive training:                              │
│  1. Single-tool learning (one memory op at a time)              │
│  2. Multi-tool coordination (operation sequences)               │
│  3. Full unified management (autonomous memory decisions)       │
│                                                                 │
│  Reward: R = α·R_task + β·R_efficiency + γ·R_quality            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Approach: Full Architecture (Parallel Development)

### Sprint 1: Foundation & Core Abstractions
**Goal**: Establish all layer interfaces simultaneously

1. **Project scaffolding** - Create full directory structure
2. **Abstract base classes** for all 5 layers:
   - `BaseAgent` - Agent layer interface
   - `MemoryInterface` - AgeMem LTM/STM interface
   - `CommunicationBus` - LatentMAS interface
   - `EvolutionEngine` - AgentEvolver interface
   - `LearningModule` - Nested Learning + RepExp interface
3. **OLMo 3 integration** - Model loading and inference
4. **Configuration system** - Unified config for all layers

### Sprint 2: Parallel Layer Implementation
**Develop all 5 layers in parallel with minimal viable features**

| Layer | Core Implementation |
|-------|---------------------|
| Agent | 5 specialized agents with basic prompts |
| Memory (AgeMem) | Graphiti backend + AgeMem operations (ADD/RETRIEVE/SUMMARY/FILTER) |
| Communication (LatentMAS) | Basic latent encoding/decoding |
| Evolution (AgentEvolver) | Workflow logging (discovery later) |
| Learning (RepExp) | Exploration bonus calculation |

### Sprint 3: Layer Integration
**Connect all layers through the Coordinator**

1. Agent ↔ Memory: Agents read/write to AgeMem
2. Agent ↔ Communication: Latent space message passing
3. Coordinator orchestration workflow
4. Basic trading simulation interface

### Sprint 4: Advanced Features
**Enhance each layer with full paper implementations**

1. **AgeMem RL Training**:
   - Three-stage progressive training (single-tool → multi-tool → unified)
   - Step-wise GRPO implementation
   - Composite reward function (task + efficiency + quality)
   - Experience replay from Graphiti episodes
2. LatentMAS: Attention-based routing, priority channels
3. AgentEvolver: Workflow discovery, agent generation
4. RepExp: Post-training exploration, diversity measurement

### Sprint 5: Trading Domain & Testing
**Domain integration and validation**

1. Market data pipeline (mock + real feeds)
2. Order management system
3. Backtesting framework
4. End-to-end testing

---

## Immediate Next Steps (Start Here)

```
Step 1: Create project structure
Step 2: Define all abstract interfaces
Step 3: Set up OLMo 3 model loading
Step 4: Implement stub classes for all components
Step 5: Create basic Coordinator workflow
```

---

## Verification Strategy

1. **Unit Tests**: Each component tested in isolation
2. **Integration Tests**: Layer interactions verified
3. **Memory Tests**: AgeMem operations validated with synthetic scenarios
4. **Communication Tests**: LatentMAS message passing verified
5. **Backtesting**: Historical trading data simulation
6. **Paper Trading**: Real-time testing without capital risk

---

## Confirmed Requirements

- **Language**: Python
- **Deployment**: Local development (research/prototyping)
- **Approach**: Full architecture - all 5 layers developed in parallel
- **Starting Point**: Fresh codebase, no existing code
- **Foundation Model**: OLMoE 1B (Mixture of Experts)
- **Memory Layer**: AgeMem (logical) + Graphiti/Zep (storage backend)
- **Training Structure**:
  - Centralized `training/` folder (not per-component)
  - Stage 1: Fine-tune OLMoE on finance data
  - Stage 2: AgeMem GRPO training
- **Finance Data Sources**: News/SEC filings, Market data, Social/sentiment (all)
- **Graphiti**: Needs setup (Neo4j + graphiti-core)
