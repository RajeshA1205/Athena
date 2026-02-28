# ATHENA Project Context

## Project Overview

ATHENA is a multi-agent financial trading framework built on OLMoE 1B (Mixture of Experts) with 5 specialized agents and 5 architectural layers. The system implements cutting-edge research in agent memory, communication, evolution, and learning.

---

## Architecture

### Foundation Model
- **OLMoE 1B** — Mixture of Experts architecture for efficient specialized routing
- **LoRA Adapters** — Parameter-efficient fine-tuning for domain specialization
- **Staged Training** — Finance fine-tuning → AgeMem GRPO training

### Five Specialized Agents
1. **Market Analyst Agent** — Real-time market analysis, pattern recognition, technical indicators, sentiment analysis
2. **Risk Manager Agent** — Portfolio risk assessment, exposure monitoring, VaR calculations, compliance checks
3. **Strategy Agent** — Strategy formulation, backtesting logic, portfolio optimization
4. **Execution Agent** — Order execution, timing optimization, slippage minimization
5. **Coordinator Agent** — Orchestration, conflict resolution, resource allocation

### Five Architectural Layers

| Layer | Component | Purpose | Research Paper |
|-------|-----------|---------|----------------|
| Agent | 5 Specialized Agents | Domain-specific decision making | ATHENA Architecture |
| Memory | AgeMem + Graphiti | Unified LTM/STM with RL-trained operations | AgeMem |
| Communication | LatentMAS | Latent space message passing | LatentMAS |
| Evolution | AgentEvolver | Workflow discovery & agent generation | AgentEvolver |
| Learning | Nested Learning + RepExp | Meta-learning with exploration bonuses | RepExp |

---

## Technical Stack

- **Language:** Python 3.10+
- **ML Framework:** PyTorch (for GRPO training + MemoryActionHead); mlx / mlx-lm (for inference on Apple Silicon)
- **Foundation Model:** OLMoE 1B — primary inference via mlx-lm; HuggingFace Transformers kept as fallback
- **Inference Backend Decision (Sprint 10):** mlx-lm is primary on Apple Silicon (Darwin). bitsandbytes/CUDA path is kept as fallback but will not work on Mac. MemoryActionHead and GRPO training remain PyTorch-only (transformers backend required for training).
- **Fine-tuning:** LoRA/QLoRA (parameter-efficient)
- **Memory Backend:** Graphiti (Zep) + Neo4j knowledge graph
- **Training:** DeepSpeed/Accelerate for distributed training
- **Data Collection:** BeautifulSoup, Scrapy, yfinance, praw (Reddit API)
- **Testing:** pytest + unittest
- **Async:** All components use async/await patterns throughout

---

## File Structure

```
/Users/rajesh/athena/
├── core/                    # ✅ Base agent, config, utilities
├── models/                  # ✅ OLMoE, embeddings
├── memory/                  # ✅ AgeMem, Graphiti backend, operations
├── training/                # ✅ Stage 1 fine-tuning, Stage 2 GRPO
├── agents/                  # 🔄 5 specialized agents (TO BUILD)
├── communication/           # 🔄 LatentMAS implementation (TO BUILD)
├── evolution/               # 🔄 AgentEvolver implementation (TO BUILD)
├── learning/                # 🔄 Nested learning + RepExp (TO BUILD)
├── trading/                 # 🔄 Trading infrastructure (TO BUILD)
├── tests/                   # 🔄 Test suite (TO BUILD)
└── project/                 # Project management
```

---

## Memory Layer (AgeMem + Graphiti)

### Architecture
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

### Operations
- **LTM Operations:** ADD, UPDATE, DELETE (write to knowledge graph)
- **STM Operations:** RETRIEVE, SUMMARY, FILTER (read/process from graph)
- **Training:** Three-stage progressive GRPO (single-tool → multi-tool → unified)

---

## Training Pipeline

### Stage 1: Finance Fine-tuning
- **Data Sources:** News/SEC filings, Market data, Social sentiment
- **Method:** LoRA fine-tuning on OLMoE 1B
- **Objective:** Causal LM + domain-specific tasks
- **Location:** `/Users/rajesh/athena/training/stage1_finetune/`

### Stage 2: AgeMem GRPO Training
- **Base Model:** Fine-tuned OLMoE from Stage 1
- **Method:** Step-wise Group Relative Policy Optimization
- **Reward:** R = α·R_task + β·R_efficiency + γ·R_quality
- **Location:** `/Users/rajesh/athena/training/stage2_agemem/`

---

## Coding Conventions

### Agent Development
- All agents **inherit from BaseAgent** (`/Users/rajesh/athena/core/base_agent.py`)
- Implement required methods: `think()` and `act()`
- Use **async/await** for all I/O operations
- Access memory via AgeMem interface
- Use LatentMAS for inter-agent communication

### Memory Operations
- AgeMem provides unified interface to Graphiti backend
- Use logical operations (ADD, RETRIEVE, etc.) not direct Graphiti calls
- All memory operations are async
- Memory training code lives in `/Users/rajesh/athena/training/stage2_agemem/`

### Communication
- Agents communicate through LatentMAS latent space
- Encode agent outputs to latent representations before sending
- Decode latent messages when receiving
- Router handles message prioritization and routing

### Configuration
- All configuration via `/Users/rajesh/athena/core/config.py`
- Environment-specific settings in config files
- No hardcoded paths or credentials

### Testing
- Unit tests for each component in isolation
- Integration tests for layer interactions
- Use pytest fixtures for shared setup
- Mock external dependencies (market data, Neo4j)

---

## Current Status (Sprint 1 Complete, Sprint 2 Starting)

### ✅ Completed (DO NOT MODIFY)
- Core infrastructure: base_agent.py, config.py, utils.py
- Model integration: olmoe.py, embeddings.py
- Memory layer: agemem.py, graphiti_backend.py, operations.py
- Training pipeline: stage1_finetune/, stage2_agemem/ infrastructure

### 🔄 In Progress (SPRINT 2)
- Agent layer: 5 specialized agents
- Communication layer: LatentMAS implementation
- Evolution layer: AgentEvolver implementation
- Learning layer: Nested learning + RepExp

### ⏳ Pending (SPRINTS 3-5)
- Layer integration and orchestration
- Advanced features (RL training, workflow discovery)
- Trading domain integration
- Comprehensive testing

---

## Key Design Decisions

### 1. AgeMem over GAM
- Unified LTM/STM operations vs. separate components
- RL-trained operations vs. supervised learning
- Graphiti provides temporal knowledge graph backend

### 2. OLMoE 1B over OLMo 3
- Mixture of Experts enables efficient agent specialization
- Smaller, faster model suitable for multi-agent deployment
- Expert routing aligns naturally with agent roles

### 3. Staged Training
- Stage 1 (finance fine-tuning) provides domain knowledge
- Stage 2 (AgeMem GRPO) trains memory management skills
- Separation enables independent optimization

### 4. Async-First Design
- All agent operations are async for parallel execution
- Enables efficient multi-agent coordination
- Better resource utilization during I/O (market data, memory)

### 5. Centralized Training
- All training code in `/Users/rajesh/athena/training/`
- Logical separation from inference components
- Easier to manage datasets and training runs

### 6. mlx-lm as Primary Inference Backend (Sprint 10 decision)
- Apple Silicon Mac cannot run bitsandbytes/CUDA; mlx-lm is the correct runtime
- OLMoEModel.load() tries mlx_lm.load() first; falls back to AutoModelForCausalLM
- OLMoEModel.generate() wraps mlx_lm.generate() in asyncio.to_thread() to stay async
- OLMoEModel.encode() on mlx path: tokenize → forward → mean-pool → numpy → torch.Tensor (CPU, no grad)
- MemoryActionHead (torch.nn.Module) and GRPO training are ONLY supported on the transformers backend
- Backend indicator: OLMoEModel._backend is "mlx" or "transformers" after load()

---

## External Dependencies

### Required Services
- **Neo4j** — Knowledge graph backend for Graphiti (docker container)
- **Graphiti-core** — Python package for temporal graph operations

### Python Packages
- torch, transformers, accelerate, deepspeed
- graphiti-core, neo4j
- yfinance, beautifulsoup4, scrapy, praw
- pytest, unittest

---

## Research Papers Location
`/Users/rajesh/athena/architecture/base/`
- AgeMem.pdf
- AgentEvolver.pdf
- LatentMAS.pdf
- RepExp.pdf

---

## Contact & References
- Full implementation plan: `/Users/rajesh/athena/agent-plan/atomic-wibbling-simon.md`
- Project board: `/Users/rajesh/athena/project/board.md`
- Roadmap: `/Users/rajesh/athena/project/roadmap.md`
