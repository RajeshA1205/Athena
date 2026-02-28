# ATHENA System Design Document

**Version:** 1.0
**Date:** 2026-02-22
**Author:** Senior-Dev Agent (Technical Owner)
**Status:** Post-Sprint 2, Pre-Sprint 3

---

## 1. Executive Summary

ATHENA is a multi-agent financial trading system in which five specialized agents -- Market Analyst, Risk Manager, Strategy, Execution, and Coordinator -- collaborate to analyse markets, assess risk, formulate strategies, execute trades, and resolve conflicts. The system is built on two key architectural bets:

1. **LatentMAS** -- Agents communicate through a shared latent space rather than passing discrete messages. Messages are encoded to fixed-dimension latent vectors via transformer-based encoder/decoder networks, routed through priority queues, and optionally broadcast to relevant agents via attention-based selection. This enables learned, compressed communication that can evolve with training.

2. **AgentEvolver** -- The system can discover successful workflow patterns from execution traces, generate new agent configurations from those patterns, and improve over time through cooperative experience replay and cross-pollination of high-performing strategies.

The foundation model is OLMoE 1B (Mixture of Experts), intended to be fine-tuned on financial data and then trained with AgeMem's step-wise GRPO for memory management. The memory layer (AgeMem) uses Graphiti/Zep as a temporal knowledge graph backend with LTM/STM operations.

**Current implementation status:** Sprint 2 is complete. All 12 tasks (TASK-001 through TASK-012) have been implemented, reviewed (3 batches), and accepted. The Agent layer, Communication layer, and Evolution layer are implemented as standalone components. Sprint 3 (integration) is next -- agents do not yet wire to AgeMem or LatentMAS at runtime. No training infrastructure exists yet. No tests exist yet.

---

## 2. System Architecture

### 2.1 Layer Diagram

```
+===================================================================+
|                        ATHENA SYSTEM                               |
+===================================================================+
|                                                                     |
|  +-------------------------------------------------------------+  |
|  |                    AGENT LAYER                                |  |
|  |  +-------------+ +-------------+ +-----------+ +-----------+ |  |
|  |  | Market      | | Risk        | | Strategy  | | Execution | |  |
|  |  | Analyst     | | Manager     | | Agent     | | Agent     | |  |
|  |  +------+------+ +------+------+ +-----+-----+ +-----+-----+ |  |
|  |         |               |              |              |        |  |
|  |         +-------+-------+------+-------+------+-------+       |  |
|  |                 |              |               |               |  |
|  |          +------+------+      |        +------+------+        |  |
|  |          | Coordinator |------+--------| BaseAgent   |        |  |
|  |          +-------------+               | (ABC)       |        |  |
|  |                                        +-------------+        |  |
|  +-------------------------------------------------------------+  |
|         |                    |                    |                  |
|         v                    v                    v                  |
|  +--------------+  +------------------+  +------------------+       |
|  | MEMORY LAYER |  | COMMUNICATION    |  | EVOLUTION LAYER  |       |
|  |              |  | LAYER (LatentMAS)|  | (AgentEvolver)   |       |
|  | +----------+ |  |                  |  |                  |       |
|  | | AgeMem   | |  | +------------+  |  | +-------------+  |       |
|  | |  (LTM/   | |  | |LatentSpace |  |  | | Workflow    |  |       |
|  | |   STM)   | |  | +-----+------+  |  | | Discovery   |  |       |
|  | +----+-----+ |  |       |         |  | +------+------+  |       |
|  |      |       |  | +-----+------+  |  |        |         |       |
|  | +----+-----+ |  | | Encoder/   |  |  | +------+------+  |       |
|  | | Graphiti | |  | | Decoder    |  |  | | Agent       |  |       |
|  | | Backend  | |  | +-----+------+  |  | | Generator   |  |       |
|  | +----------+ |  |       |         |  | +------+------+  |       |
|  |              |  | +-----+------+  |  |        |         |       |
|  |              |  | | Message    |  |  | +------+------+  |       |
|  |              |  | | Router     |  |  | | Cooperative |  |       |
|  |              |  | +------------+  |  | | Evolution   |  |       |
|  +--------------+  +------------------+  | +-------------+  |       |
|                                          +------------------+       |
|                                                                     |
|  +-------------------------------------------------------------+  |
|  |                    LEARNING LAYER                             |  |
|  |          (Sprint 4 -- Not Yet Implemented)                    |  |
|  |  +-------------------+  +--------------------+                |  |
|  |  | Nested Learning   |  | RepExp Exploration |                |  |
|  |  +-------------------+  +--------------------+                |  |
|  +-------------------------------------------------------------+  |
|                                                                     |
+===================================================================+
```

### 2.2 Layer Responsibilities

| Layer | Responsibility | Implementation Status |
|-------|---------------|----------------------|
| **Agent** | Domain-specific analysis, decision-making, and action execution. Each agent implements `think()`/`act()` lifecycle. | Implemented (Sprint 2) |
| **Memory** | Long-term and short-term memory management. LTM: ADD/UPDATE/DELETE via Graphiti. STM: RETRIEVE/SUMMARY/FILTER. | Interface + AgeMem implemented (Sprint 1); backend integration pending |
| **Communication** | Inter-agent message passing via latent space. Encode/decode, priority routing, broadcast. | Implemented (Sprint 2) |
| **Evolution** | Workflow pattern discovery, agent config generation, cooperative experience replay. | Implemented (Sprint 2) |
| **Learning** | Nested meta-learning (inner/outer loop) and RepExp exploration for diversity. | Not implemented (Sprint 4) |

### 2.3 Component Inventory

| Component | File | Sprint | Status |
|-----------|------|--------|--------|
| BaseAgent | `/Users/rajesh/athena/core/base_agent.py` | 1 | Complete |
| AthenaConfig | `/Users/rajesh/athena/core/config.py` | 1 | Complete |
| MarketAnalystAgent | `/Users/rajesh/athena/agents/market_analyst.py` | 2 | Complete |
| RiskManagerAgent | `/Users/rajesh/athena/agents/risk_manager.py` | 2 | Complete |
| StrategyAgent | `/Users/rajesh/athena/agents/strategy_agent.py` | 2 | Complete |
| ExecutionAgent | `/Users/rajesh/athena/agents/execution_agent.py` | 2 | Complete |
| CoordinatorAgent | `/Users/rajesh/athena/agents/coordinator.py` | 2 | Complete |
| LatentSpace | `/Users/rajesh/athena/communication/latent_space.py` | 2 | Complete |
| LatentEncoder/Decoder | `/Users/rajesh/athena/communication/latent_space.py` (lines 47-134) | 2 | Complete |
| AgentStateEncoder | `/Users/rajesh/athena/communication/encoder.py` | 2 | Complete |
| AgentStateDecoder | `/Users/rajesh/athena/communication/decoder.py` | 2 | Complete |
| MessageRouter | `/Users/rajesh/athena/communication/router.py` | 2 | Complete |
| WorkflowDiscovery | `/Users/rajesh/athena/evolution/workflow_discovery.py` | 2 | Complete |
| AgentGenerator | `/Users/rajesh/athena/evolution/agent_generator.py` | 2 | Complete |
| CooperativeEvolution | `/Users/rajesh/athena/evolution/cooperative_evolution.py` | 2 | Complete |
| AgeMem | `/Users/rajesh/athena/memory/agemem.py` | 1 | Complete (needs backend) |
| Nested Learning | `learning/nested_learning.py` | 4 | Not started |
| RepExp | `learning/repexp.py` | 4 | Not started |
| Trading modules | `trading/` | 5 | Not started |

---

## 3. Core Abstractions

### 3.1 BaseAgent Contract

**File:** `/Users/rajesh/athena/core/base_agent.py`

`BaseAgent` is the abstract base class from which all five agents inherit. It defines the think-act lifecycle, context building, and communication/memory integration points.

**Constructor signature (line 68):**

```python
def __init__(
    self,
    name: str,              # Unique agent identifier
    role: str,              # Agent role (analyst, risk, strategy, execution, coordinator)
    system_prompt: str,     # System prompt defining agent behavior
    model: Optional[Any],   # Language model instance (OLMo 3 / OLMoE)
    memory: Optional["AgeMem"],         # AgeMem memory instance
    communication: Optional["LatentSpace"],  # LatentMAS communication
    tools: Optional[List[str]],         # Available tool names
    config: Optional[Dict[str, Any]],   # Additional configuration
)
```

Note: `memory` and `communication` use `TYPE_CHECKING` forward references to `athena.memory.agemem.AgeMem` and `athena.communication.latent_space.LatentSpace` (lines 13-15), but runtime imports use bare paths (`from core.base_agent import ...`).

**Lifecycle states** (line 18, `AgentState` enum): `IDLE`, `THINKING`, `ACTING`, `WAITING`, `ERROR`.

**Abstract methods:**

- `async think(context: AgentContext) -> Dict[str, Any]` -- Process context, decide on action. Returns a dict that must include a `"done"` key; when `True`, the `run()` loop terminates.
- `async act(thought: Dict[str, Any]) -> AgentAction` -- Execute the planned action. Returns an `AgentAction` dataclass.

**Run loop** (line 131):

```
for i in range(max_iterations):
    state = THINKING
    context = await _build_context(task)
    thought = await think(context)
    if thought.get("done", False): break   # <-- termination check
    state = ACTING
    action = await act(thought)
    if not action.success: break           # <-- error bail-out
state = IDLE
```

The `done` flag semantics vary by agent -- this is a known inconsistency (see Section 10).

**Context building** (line 173, `_build_context`):

1. Creates `AgentContext(task=task)`
2. Populates `history` from last 10 `action_history` entries
3. If `self.memory` is not None: `await self.memory.retrieve(task)` into `memory_context`
4. If `self.communication` is not None: `await self.communication.receive(self.name)` into `messages`

**Key dataclasses:**

| Dataclass | Fields | Purpose |
|-----------|--------|---------|
| `AgentMessage` (line 28) | sender, recipient (`"*"` for broadcast), content (Any), message_type, priority (1-3), timestamp, metadata | Inter-agent communication unit |
| `AgentAction` (line 40) | action_type, parameters, result, success, error, duration | Action execution result |
| `AgentContext` (line 51) | task, history, memory_context, messages, metadata | Decision-making input |

### 3.2 Configuration System

**File:** `/Users/rajesh/athena/core/config.py`

The configuration is a hierarchy of `@dataclass` objects under a single `AthenaConfig` master:

| Config Class | Key Fields | Default Values |
|-------------|------------|----------------|
| `ModelConfig` | model_name, device, dtype, max_length, temperature | `"allenai/OLMo-1B"`, `"auto"`, `"float16"`, 2048, 0.7 |
| `MemoryConfig` | ltm_vector_dim, stm_buffer_size, retrieval_top_k, update_strategy | 768, 10, 5, `"merge"` |
| `CommunicationConfig` | latent_dim, num_attention_heads, message_queue_size, encoding_method | 512, 8, 100, `"transformer"` |
| `EvolutionConfig` | workflow_library_size, population_size, mutation_rate, experience_replay_size | 1000, 10, 0.1, 10000 |
| `LearningConfig` | inner_lr, outer_lr, inner_steps, exploration_coefficient | 1e-4, 1e-5, 5, 0.1 |
| `TradingConfig` | markets, data_source, position_limit, max_drawdown | `["stocks"]`, `"mock"`, 100000, 0.1 |
| `AgentConfig` | name, role, system_prompt, tools, memory_enabled, communication_enabled | (varies per agent) |

`AthenaConfig` supports YAML and JSON serialization via `from_yaml()`, `from_json()`, `save_yaml()`, `save_json()`.

**Default agent configs** (`get_default_agent_configs()`, line 185): Returns a dict of 5 `AgentConfig` instances keyed by name. Each agent's constructor falls back to this when no `system_prompt` is provided. All five agents consistently source their prompts from this function (verified in Sprint 2 review).

### 3.3 Agent Registration

Agents register with the system in two ways:

1. **Coordinator registration** -- `CoordinatorAgent.register_agent(name, agent)` (coordinator.py line 95) stores agents in `self.agents: Dict[str, BaseAgent]` for orchestration and priority resolution.

2. **Router registration** -- `MessageRouter.register_agent(agent_id, agent_info)` (router.py line 99) creates per-priority `asyncio.Queue` instances for the agent. Note: `LatentSpace` uses a `defaultdict` for queues, which creates queues lazily; agents not yet known miss broadcast fan-out (see Section 10, N2).

There is no unified agent registry. The Coordinator and Router maintain independent registrations.

---

## 4. Agent Layer

### 4.1 MarketAnalystAgent

**File:** `/Users/rajesh/athena/agents/market_analyst.py`
**Role:** `analyst` | **Priority:** 1 (lowest in coordinator weighting)

**Responsibility:** Analyse price data to produce technical indicators, detect chart patterns, classify market regime, and assess sentiment.

**`think()` (line 100):**
- Reads `market_data` from `context.metadata` (expects `prices: List[float]` and `news: List[str]`)
- Requires at least `min_data_points` (default 50) price points
- Computes: technical indicators, chart patterns, market regime, confidence score
- Analyses sentiment from text data via keyword matching
- Sets `done: True` if indicators OR sentiment were computed

**`act()` (line 155):**
- Interprets indicators, patterns, and sentiment into a recommendation: `{action: "buy"|"sell"|"hold", confidence, reasons, supporting_evidence}`
- Returns `AgentAction(action_type="market_analysis", result=recommendation)`

**Key algorithms:**

| Algorithm | Method | Details |
|-----------|--------|---------|
| **SMA** | `_calculate_technical_indicators` (line 256) | Simple arithmetic mean over last N prices |
| **EMA** | `_calculate_ema` (line 407) | Standard EMA: multiplier = 2/(period+1), seeded with SMA of first `period` values |
| **MACD** | `_calculate_technical_indicators` (line 268-280) | Builds full MACD series by computing EMA-12 minus EMA-26 at each index from 26 onward. Signal line = 9-period EMA of MACD series. Falls back to SMA when <9 MACD values. **Known O(n^2)** -- recalculates EMA from scratch per index (review finding N1). |
| **RSI** | `_calculate_rsi` (line 420) | Wilder's smoothing: SMA seed over first `period` gains/losses, then `(avg * (period-1) + current) / period` for subsequent values |
| **Bollinger Bands** | `_calculate_technical_indicators` (line 285-289) | SMA-20 +/- 2 * population std dev |
| **Pattern detection** | `_is_double_top`, `_is_double_bottom`, `_is_head_and_shoulders`, `_is_ascending_triangle`, `_is_descending_triangle` (lines 458-536) | Peak/trough detection with 5-bar lookback windows; 2% tolerance for double top/bottom, 5% for shoulder symmetry. Triangle detection uses index parity as proxy for highs/lows (review finding P2). |
| **Sentiment** | `_analyze_sentiment` (line 324) | Keyword bag-of-words with 13 positive and 12 negative keywords. Per-text normalized score, averaged across all texts, clamped to [-1, 1]. |
| **Market regime** | `_calculate_market_regime` (line 357) | Classifies as trending/ranging/volatile based on SMA-20 vs SMA-50 divergence (2% threshold) and volatility percentage (low <1%, normal <3%, high >=3%). |
| **Confidence** | `_calculate_confidence` (line 538) | Additive: data points (0.1-0.4) + indicator count (0-0.4) + pattern count (0-0.2), capped at 1.0 |

**Notable design decisions:**
- `done` is set to `True` if any analysis succeeds (indicators or sentiment), meaning MarketAnalyst is a single-pass agent.
- Sentiment analysis is keyword-based (no LLM call); intended as placeholder until model integration.
- All trading signals use hardcoded asset name `"ASSET"` (review finding P3).

### 4.2 RiskManagerAgent

**File:** `/Users/rajesh/athena/agents/risk_manager.py`
**Role:** `risk` | **Priority:** 3 (highest in coordinator weighting)

**Responsibility:** Assess portfolio risk, check compliance limits, compute VaR and Expected Shortfall, generate alerts.

**`think()` (line 82):**
- Reads `positions` (list of dicts with `symbol`, `value`, `sector`) and `returns` (dict mapping symbol to return series) from `context.metadata`
- Empty positions: returns immediately with `done: True`, all metrics zeroed
- Computes: VaR at 95% and 99%, Expected Shortfall, portfolio metrics (Sharpe, volatility, max drawdown), position limit compliance, correlation risk
- **Always returns `done: False`** on the main path (review finding N3), so `BaseAgent.run()` will always hit `max_iterations`

**`act()` (line 149):**
- Generates recommendations from compliance issues: `reduce_position` (high urgency), `diversify` (medium), `reduce_exposure` (critical for drawdown)
- Broadcasts risk alerts via `send_message(recipient="*", message_type="risk_alert")` when communication is available
- Returns `AgentAction(action_type="risk_assessment")`

**Key algorithms:**

| Algorithm | Method | Details |
|-----------|--------|---------|
| **Historical VaR** | `_calculate_var` (line 229) | Sorts returns, picks percentile index: `var = -sorted_returns[index]`. Portfolio returns are equal-weighted average across assets. |
| **Parametric VaR** | `_calculate_var` (line 272-278) | Uses z-score lookup table (1.28/1.645/2.326 for 90/95/99%), `var = abs(mean - z * std)` |
| **Expected Shortfall** | `_calculate_expected_shortfall` (line 285) | Average of tail losses below VaR cutoff: `abs(mean(sorted_returns[:cutoff]))` |
| **Sharpe Ratio** | `_calculate_portfolio_metrics` (line 416-418) | `(mean_return - risk_free_rate/252) / daily_vol * sqrt(252)` |
| **Max Drawdown** | `_calculate_portfolio_metrics` (line 423-434) | Multiplicative compounding: `cumulative *= (1+r)`, `dd = (peak - cumulative) / peak` |
| **Correlation Matrix** | `_calculate_correlation_matrix` (line 438) | Pearson correlation from manual covariance/std computation. O(n^2 * m) where n = assets, m = return length. |
| **Risk Level** | `_determine_risk_level` (line 557) | `critical` if >=3 issues; `high` if VaR or drawdown exceed limits; `medium` if any issues or VaR > 70% of limit; else `low` |

**Risk limits** (`RiskLimits` dataclass, line 18): max_position_pct=20%, max_sector_pct=30%, max_drawdown=10%, max_var_pct=5%, max_leverage=2.0, max_correlation=0.70.

**Notable design decisions:**
- Portfolio return aggregation is duplicated 3 times across VaR, ES, and metrics methods (review finding W1).
- Risk Manager has the highest coordinator priority (3), meaning its votes carry 3x weight in conflict resolution.
- The `done: False` issue means the risk manager will repeatedly re-assess until `max_iterations`.

### 4.3 StrategyAgent

**File:** `/Users/rajesh/athena/agents/strategy_agent.py`
**Role:** `strategy` | **Priority:** 2

**Responsibility:** Select trading strategy, generate signals, backtest, optimize portfolio.

**`think()` (line 104):**
- Reads `market_data` from `context.metadata` (prices, volatility, trend_strength)
- Classifies market regime: `high_volatility`, `trending`, `downtrending`, `mean_reverting`, `neutral`
- Selects strategy type: `MOMENTUM`, `MEAN_REVERSION`, `BREAKOUT`, `PAIRS_TRADING`, `ARBITRAGE`
- Returns `done: False` -- delegates signal generation to `act()`

**`act()` (line 155):**
- Dispatches on `thought["action"]`: `"generate_signals"` or `"backtest"`
- Signal generation produces `TradingSignal` dataclasses with entry/exit/stop/take-profit/position-size
- Returns `AgentAction(action_type=action_type)`

**Key algorithms:**

| Algorithm | Method | Details |
|-----------|--------|---------|
| **Strategy selection** | `_select_strategy` (line 248) | Rule-based: trending + strong trend -> MOMENTUM; trending + moderate -> BREAKOUT; mean_reverting -> MEAN_REVERSION; high_volatility -> PAIRS_TRADING; default -> MOMENTUM |
| **Momentum signals** | `_generate_momentum_signals` (line 291) | `momentum = (current - lookback_price) / lookback_price`. Buy if >2%, sell if <-2%. Strength = `min(abs(momentum)*10, 1.0)`. |
| **Mean reversion signals** | `_generate_mean_reversion_signals` (line 350) | Z-score against rolling window. Buy if z < -num_std, sell if z > +num_std. Exit target = mean price. |
| **Position sizing** | `_calculate_position_size` (line 470) | Volatility-based: `base = risk_budget / volatility`, `size = base * signal_strength`, capped at 20% of portfolio. |
| **Portfolio optimization** | `_optimize_portfolio` (line 413) | Three methods: equal_weight, risk_parity (inverse volatility), mean_variance (simplified return/risk ratio). |
| **Backtesting** | `_backtest_strategy` (line 492) | Walk-forward: for each bar from `lookback` to end, generates signals on trailing window, applies position * period_return. Computes total_return (additive sum -- review finding N4), Sharpe, max drawdown (multiplicative), win_rate, profit_factor. |

**Strategy types** (`StrategyType` enum, line 18): MOMENTUM, MEAN_REVERSION, ARBITRAGE, PAIRS_TRADING, BREAKOUT. Only MOMENTUM and MEAN_REVERSION have signal generators implemented.

**Notable design decisions:**
- `total_return` in backtest is additive `sum(returns)` while drawdown uses multiplicative compounding -- inconsistent (review finding N4).
- ARBITRAGE, PAIRS_TRADING, and BREAKOUT strategies have no signal generation implementation yet.
- All signals use hardcoded `asset="ASSET"`.

### 4.4 ExecutionAgent

**File:** `/Users/rajesh/athena/agents/execution_agent.py`
**Role:** `execution` | **Priority:** 1

**Responsibility:** Execute orders with optimal timing, minimize slippage and market impact.

**`think()` (line 129):**
- Parses trade request from `context.metadata["trade_request"]`
- Reads market conditions (current_price, avg_volume, volatility)
- Estimates market impact via square-root model
- Selects order type based on urgency, size relative to volume, and volatility
- For TWAP/VWAP: computes execution schedule
- For LIMIT: sets price 0.1% better than market
- Returns `done: False`

**`act()` (line 206):**
- Creates `Order` from execution plan
- Simulates execution with realistic slippage (random factor 0.5-1.5x of estimated slippage)
- Computes execution quality metrics (slippage, fees, implementation shortfall, fill rate)
- Moves filled orders from `active_orders` to `completed_orders`
- Returns `AgentAction(action_type="execute_order")`

**Key algorithms:**

| Algorithm | Method | Details |
|-----------|--------|---------|
| **Market impact** | `_estimate_market_impact` (line 302) | Square-root model: `impact = volatility * sqrt(order_size / avg_volume)`, capped at 5% |
| **Order type selection** | `_select_order_type` (line 324) | High urgency or <1% of volume -> MARKET; >10% of volume -> VWAP; >5% -> TWAP; low volatility -> LIMIT; else MARKET |
| **TWAP schedule** | `_calculate_twap_schedule` (line 348) | Equal-size slices at equal time intervals |
| **VWAP schedule** | `_calculate_vwap_schedule` (line 364) | Proportional to volume profile; falls back to TWAP if profile sums to zero |
| **Slippage simulation** | `_simulate_fill` (line 400) | `actual_slippage = estimated * (0.5 + random())`. Applies directionally (buy: price up, sell: price down). Respects limit price constraints. |
| **Implementation shortfall** | `_calculate_execution_metrics` (line 434) | `(avg_fill_price - arrival_price) / arrival_price * 10000` in bps |

**Order types** (`OrderType` enum, line 19): MARKET, LIMIT, STOP, STOP_LIMIT, TWAP, VWAP.

**Notable design decisions:**
- Execution is fully simulated (no real broker integration). The `random.random()` call is non-deterministic and not seeded (review finding W2).
- Order IDs use a monotonic counter: `ORD-{counter:06d}`.
- Fee rate and slippage defaults are configurable via `config["fee_bps"]` (default 1.0) and `config["default_slippage_bps"]` (default 5.0).

### 4.5 CoordinatorAgent

**File:** `/Users/rajesh/athena/agents/coordinator.py`
**Role:** `coordinator` | **Priority:** N/A (orchestrator, not a voter)

**Responsibility:** Orchestrate other agents, aggregate recommendations, resolve conflicts, allocate resources, make final trading decisions.

**`think()` (line 106):**
- Collects `"recommendation"` messages from `context.messages`
- Collects `"resource_request"` messages
- Detects buy-vs-sell conflicts via `_detect_conflicts()`
- **Always sets `done: True`** when recommendations or resource requests exist (line 156) -- single-pass design
- Returns orchestration plan with agent_recommendations, conflicts, resource_requests

**`act()` (line 160):**
- If resource requests exist: allocates resources proportionally (not round-robin despite the docstring -- review finding #12)
- If recommendations exist with conflicts: calls `_resolve_conflicts()` (priority-weighted voting)
- If no conflicts: aggregates via majority vote with averaged confidence
- Applies risk veto: high risk downgrades buy to hold; compliance violations force hold
- Broadcasts final decision via `send_message(recipient="*", message_type="final_decision", priority=3)`
- Returns `AgentAction(action_type="coordination")`

**Key algorithms:**

| Algorithm | Method | Details |
|-----------|--------|---------|
| **Conflict detection** | `_detect_conflicts` (line 261) | Scans for buy/sell contradictions only. buy-vs-hold is not considered a conflict. |
| **Priority-weighted voting** | `_resolve_conflicts` (line 297) | `weight = priority * confidence`. Priorities: risk=3, strategy=2, analyst=1, execution=1. `decision = max(weighted_votes)`. `winning_agent` = agent with highest weight voting for the winning action. |
| **Risk veto** | `_make_final_decision` (line 421) | Compliance violations -> hold (confidence 0). High risk + buy -> hold (confidence halved). |
| **Resource allocation** | `_allocate_resources` (line 378) | Proportional share: `allocated = requested / total_requested` per resource type. |

**Agent priority table:**

```
risk       -> priority 3  (highest influence)
strategy   -> priority 2
analyst    -> priority 1
execution  -> priority 1  (lowest influence)
```

**Role resolution:** When an agent is not in `self.agents` (the registered dict), the coordinator falls back to substring matching on the agent name against known role keywords (lines 323-327). This logic is duplicated in the `winning_agent` resolution pass (review nit).

**Notable design decisions:**
- Single-pass design: `done: True` always. The coordinator never iterates.
- Only buy-vs-sell is flagged as a conflict. buy-vs-hold or sell-vs-hold silently takes majority.
- The risk manager has veto power over buy decisions when risk is `"high"`, independent of voting.

---

## 5. Communication Layer (LatentMAS)

The communication layer implements a shared latent space for inter-agent communication, inspired by the Latent Collaboration for Multi-Agent Systems research.

### 5.1 LatentSpace

**File:** `/Users/rajesh/athena/communication/latent_space.py`
**Class:** `LatentSpace` (line 137)

The central message bus. Manages per-agent message queues, handles broadcast, and provides encode/decode capabilities.

**Initialization (line 148):**
- Requires PyTorch (raises `ImportError` with install instructions if missing)
- Creates `LatentEncoder` and `LatentDecoder` (transformer-based, see below)
- Initializes per-agent queues as `defaultdict(lambda: deque(maxlen=message_queue_size))`
- Separate `_broadcast_queue` (bounded audit log only -- broadcasts fan out to per-agent queues)
- Single `asyncio.Lock` for all queue mutations
- Encoder/decoder start in `eval()` mode

**`send(message: AgentMessage) -> bool` (line 218):**
1. Acquires `self._lock`
2. Encodes `message.content` to latent vector via `encode_to_latent()`
3. Creates `LatentMessage` with the latent vector
4. Stores `message.content` as `metadata["_original_content"]` for lossless passthrough
5. If recipient is `"*"` (broadcast):
   - Appends to `_broadcast_queue` (audit)
   - Appends to every existing per-agent queue **except sender**
   - Caveat: only reaches agents that already have a queue entry (review finding N2)
6. Otherwise: appends to `self._queues[recipient]`

**`receive(agent_name: str) -> List[AgentMessage]` (line 271):**
1. Acquires `self._lock`
2. Drains the agent's queue via `popleft()` (destructive read)
3. Decodes each `LatentMessage` via `_decode_latent_message()`:
   - First checks for `_original_content` in metadata (lossless passthrough)
   - Falls back to neural decode only if `_original_content` is missing
   - Pops `_original_content` from metadata before returning (does not leak to consumers)
4. Sorts messages by priority (descending)
5. Returns list of `AgentMessage`

**Locking design:** A single `asyncio.Lock` serializes all `send()` and `receive()` operations. The `get_stats()` method calls `_get_buffer_status_unlocked()` inside a single lock acquisition to avoid the deadlock that existed in the original implementation (batch 1 fix #1).

**Encoding (line 318, `encode_to_latent`):**
- Converts content to string via `str(content)`
- Character-code tokenization: `ord(c) % 256` for each char, padded/truncated to `latent_dim`
- Normalizes to [0, 1] by dividing by 256
- Passes through `LatentEncoder` (transformer) with `torch.no_grad()`
- Returns latent vector of shape `[latent_dim]`

**Decoding (line 346, `decode_from_latent`):**
- Passes through `LatentDecoder` with `torch.no_grad()`
- Converts output to chars: `chr(int(val * 256))` for printable ASCII (32-126)
- Note: In practice, this path is rarely used because `_original_content` passthrough handles all messages sent through `send()`

### 5.2 LatentEncoder / LatentDecoder

**File:** `/Users/rajesh/athena/communication/latent_space.py` (lines 47-134)

These are transformer-based `nn.Module` subclasses used internally by `LatentSpace` for latent encoding/decoding.

**LatentEncoder (line 47):**
- Architecture: MultiheadAttention(self-attention) -> LayerNorm -> FFN(Linear-ReLU-Dropout-Linear) -> LayerNorm
- Input: `[batch, seq_len, latent_dim]`, output: `[batch, latent_dim]` (mean pooling over seq_len)
- Default: latent_dim=512, num_heads=8, hidden_dim=2048, dropout=0.1

**LatentDecoder (line 88):**
- Same architecture as encoder
- Supports optional cross-attention context: if provided, attention query=x, key/value=context
- Input: `[batch, latent_dim]` (unsqueezed to `[batch, 1, latent_dim]`), output: `[batch, latent_dim]`

**Important:** These classes inherit `nn.Module` at class definition time (line 47, 88). If torch is not installed, the module fails to import with `NameError` before reaching the `HAS_TORCH` guard in `LatentSpace.__init__()` (review finding N5). This means the `communication/latent_space.py` module requires torch at import time.

### 5.3 AgentStateEncoder

**File:** `/Users/rajesh/athena/communication/encoder.py`
**Class:** `AgentStateEncoder(nn.Module)` (line 37)

MLP-based encoder that transforms agent outputs/states into latent representations. Distinct from `LatentEncoder` -- this operates on agent states before communication, while `LatentEncoder` operates within the latent space message pipeline.

**Architecture:**
- MLP encoder network: `input_dim` (default 512) -> hidden_dims (default [512, 256]) -> `latent_dim` (default 256), with ReLU and Dropout
- Text projection: Linear(text_embed_dim=768 -> latent_dim)
- Output normalization: LayerNorm(latent_dim)

**Encoding dispatch** (`encode_agent_state`, line 206):

| Input Type | Path | Details |
|-----------|------|---------|
| `torch.Tensor` | `encode_numeric()` | Direct MLP forward pass |
| `dict` | `encode_structured()` | Encodes `"numeric"` and `"text_embedding"` sub-keys separately, mean-pools if both present |
| `str` | `encode_numeric()` | Character-code tokenization: `ord(c)/255.0`, padded/truncated to `input_dim`, then MLP. Fixed in Sprint 2 batch 3 review -- originally returned zero vector. |

**Persistence:** `save(path)` uses `torch.save(self.state_dict(), path)`. `load(path)` uses `torch.load(path, map_location=self.device, weights_only=True)` -- the `weights_only=True` parameter was added as a security fix in Sprint 2 batch 2 review.

**HAS_TORCH guard:** The encoder defines dummy `torch` and `nn` classes when torch is not available (lines 21-29), allowing the module to be imported without torch. The actual `__init__` raises `ImportError` if `HAS_TORCH` is False.

### 5.4 AgentStateDecoder

**File:** `/Users/rajesh/athena/communication/decoder.py`
**Class:** `AgentStateDecoder(nn.Module)` (line 61)

MLP-based decoder that transforms latent representations back to agent-readable formats.

**Architecture:**
- Input normalization: LayerNorm(latent_dim)
- MLP decoder network: `latent_dim` (default 256) -> hidden_dims (default [256, 512]) -> `output_dim` (default 512)
- Text reconstruction: Linear(latent_dim -> text_embed_dim=768)

**Decoding dispatch** (`interpret_message`, line 215):

| Mode | Method | Output |
|------|--------|--------|
| `"numeric"` | `decode_to_numeric()` | Tensor `[output_dim]` |
| `"text_embedding"` | `decode_to_text_embedding()` | Tensor `[text_embed_dim]` |
| `"structured"` | `decode_to_structured()` | Dict with both `"numeric"` and `"text_embedding"` keys |

**Batch decoding** (`decode_messages`, line 247): Processes messages sequentially with individual `await` calls. Could be batched via `torch.stack()` for performance (review finding #9).

**Persistence:** Uses `torch.save()` with a checkpoint dict containing `state_dict`, `latent_dim`, `output_dim`, `text_embed_dim`, `dropout`. `load()` uses `weights_only=True`.

### 5.5 MessageRouter

**File:** `/Users/rajesh/athena/communication/router.py`
**Class:** `MessageRouter` (line 39)

Provides priority-queue routing and attention-based selective broadcast on top of the `LatentSpace`.

**Initialization (line 52):**
- Takes `LatentSpace`, `AgentStateEncoder`, `AgentStateDecoder` instances
- Config options: `enable_priority` (default True), `enable_attention` (default False), `max_attention_recipients` (default 5)
- Maintains `agent_registry: Dict[str, Dict]` and `priority_queues: Dict[str, Dict[MessagePriority, asyncio.Queue]]`

**Three routing strategies:**

| Strategy | When Used | Path |
|----------|-----------|------|
| **Priority-queue** | `enable_priority=True` and recipient is not `"broadcast"` | Encodes message via `AgentStateEncoder`, puts latent vector directly into receiver's priority queue (HIGH/MEDIUM/LOW) |
| **LatentSpace** | `enable_priority=False` or recipient is `"broadcast"` | Wraps in `AgentMessage`, delegates to `LatentSpace.send()` |
| **Attention broadcast** | Explicit call to `broadcast_with_attention()` with `enable_attention=True` | Encodes message, computes dot-product attention between message and agent embeddings, sends to top-N agents |

**Priority levels** (`MessagePriority` enum, line 31): HIGH=3, MEDIUM=2, LOW=1.

**`receive()` (line 199):**
1. Drains priority queues HIGH -> MEDIUM -> LOW (non-blocking `get_nowait()`)
2. Collects messages from `LatentSpace.receive()` -- these are already decoded `AgentMessage` objects, so they are re-encoded to latent vectors for uniform output format
3. Batch-decodes all latent vectors via `AgentStateDecoder.decode_messages()`

Note: There is a non-atomicity gap -- the `await` on LatentSpace receive yields control, so a HIGH-priority message arriving between the queue drain and the LatentSpace call will be picked up on the next `receive()`, after lower-priority LatentSpace messages (review finding #3).

**Attention broadcast** (`broadcast_with_attention`, line 276):
- Encodes message once
- Computes dot-product score between message latent and each agent embedding
- Sends to top-N agents by score (currently re-encodes per send -- N+1 total encodes, review finding #4)
- Falls back to standard LatentSpace broadcast if attention is disabled or no embeddings provided

---

## 6. Evolution Layer (AgentEvolver)

The evolution layer enables the system to discover successful interaction patterns, generate new agent configurations, and improve through cooperative experience replay.

### 6.1 WorkflowDiscovery

**File:** `/Users/rajesh/athena/evolution/workflow_discovery.py`
**Class:** `WorkflowDiscovery` (line 77)

Analyses agent execution traces to identify recurring successful patterns.

**Configuration:**
- `min_success_rate`: 0.7 (minimum to qualify as "successful")
- `min_use_count`: 3 (minimum observations required)
- `similarity_threshold`: 0.8 (for pattern matching)

**Core data structure** (`WorkflowPattern`, line 17):
- `pattern_id`: SHA-256 hash of sorted agent sequence + sorted communication graph + sorted message types, truncated to 16 hex chars, prefixed with `"pattern_"`
- `agent_sequence`: Ordered list of agent names
- `interaction_pattern`: Dict with `communication_graph` (sender -> [recipients]) and `message_types` (edge -> [types])
- `success_rate`: Incremental moving average
- `use_count`: Observation count

**`analyze_execution(execution_trace)` (line 110):**
1. Appends trace to `execution_history` (unbounded list -- review finding #10)
2. Extracts agent sequence and interaction pattern
3. Generates deterministic pattern ID from hash
4. If pattern exists: updates success_rate using incremental average formula
5. If new: creates pattern with initial success 1.0 or 0.0 based on outcome

**Pattern similarity** (`_calculate_pattern_similarity`, line 296):
- Jaccard similarity of agent sets (50% weight)
- Jaccard similarity of communication graph edge sets (50% weight)
- Combined as arithmetic mean

**Persistence:** `save_library(path)` and `load_library(path)` serialize to/from JSON. No error handling on file I/O (review finding #11).

### 6.2 AgentGenerator

**File:** `/Users/rajesh/athena/evolution/agent_generator.py`
**Class:** `AgentGenerator` (line 104)

Generates agent configurations from successful workflow patterns.

**Configuration:**
- `min_pattern_success`: 0.8 (minimum pattern success rate to generate from)
- `max_generated_configs`: 50 (triggers pruning when exceeded)

**Core data structure** (`AgentConfiguration`, line 19):
- `config_id`: `"generated_{agent_type}_{counter}"` using monotonic `_next_config_id` counter (fixed in Sprint 2 batch 3 -- originally used `len(generated_configs)` which was non-unique after pruning)
- `agent_type`: Inferred from most frequent agent name in pattern
- `capabilities`: Extracted from keyword matching on agent names
- `parameters`: Tuned based on pattern success rate
- `performance_score`: Set to pattern's success_rate

**`generate_from_pattern(pattern)` (line 144):**
1. Infers agent type: `"specialized_{most_frequent_agent}"` or `"generic_agent"`
2. Extracts capabilities via keyword map: `analyst->analysis`, `risk->risk_assessment`, `strategy->strategy_formulation`, `execution->order_execution`. Adds `"coordination"` if communication graph has >3 nodes.
3. Generates parameters: `confidence_threshold=0.7` (lowered to 0.6 for patterns with >90% success), `max_iterations=10`, `timeout_seconds=30`
4. Stores in `generated_configs` dict
5. Calls `_prune_configs()` if over limit

**Task matching** (`select_agent_for_task`, line 219):
- Scores configs as `0.6 * capability_match + 0.4 * performance_score`
- Returns best config if score > 0.5, else None

**Pruning** (`_prune_configs`, line 429):
- Sorts by `performance_score` ascending
- Removes lowest performers until count <= `max_generated_configs`

### 6.3 CooperativeEvolution

**File:** `/Users/rajesh/athena/evolution/cooperative_evolution.py`
**Class:** `CooperativeEvolution` (line 107)

Enables cooperative improvement through experience replay and knowledge sharing.

**Configuration:**
- `max_experience_buffer`: 10000 per agent and for shared pool
- `replay_batch_size`: 32
- `knowledge_sharing_rate`: 0.1 (10% of replay batch from shared pool)
- `min_reward_threshold`: 0.5 (minimum reward for shared pool entry)

**Core data structure** (`Experience`, line 19):
- `experience_id`: `"{agent_id}_{unix_timestamp}"`
- Fields: agent_id, state, action, outcome, reward, timestamp, metadata
- Serializable via `to_dict()`/`from_dict()`

**Per-agent experience buffers** (`experience_buffers`, line 141):
- `Dict[str, deque]` with each deque bounded by `max_buffer_size`
- Created lazily on first `add_experience()` call

**Shared pool** (`shared_pool`, line 144):
- Single `deque(maxlen=max_buffer_size)` accessible to all agents
- Populated when individual experiences have `reward >= min_reward_threshold`

**`replay_experiences(agent_id, batch_size)` (line 189):**
1. Draws `(1 - sharing_rate)` fraction from agent's own buffer (random sample)
2. Fills remainder from shared pool (excluding agent's own experiences)
3. Returns mixed list

**`cross_pollinate(top_k=3)` (line 360):**
1. Identifies top-K performers by average reward
2. For each top performer: adds experiences with `reward >= avg_reward * 0.9` to shared pool
3. Note: For negative rewards, `avg * 0.9` produces a cutoff that is less negative (more restrictive), which is the opposite of intent (review finding #6)

**Performance tracking** (`agent_performance`, line 147):
- `Dict[str, List[float]]` of recent rewards, capped at 100 entries via `pop(0)` (O(n) -- could be `deque(maxlen=100)`, review finding #11)

---

## 7. Memory Layer

### 7.1 AgeMem

**File:** `/Users/rajesh/athena/memory/agemem.py`
**Class:** `AgeMem(MemoryInterface)` (line 79)

AgeMem is the unified memory management system implementing both long-term and short-term memory operations, backed by Graphiti (Zep) as the storage layer.

**Architecture:**
```
+-----------------------------------+
|         AgeMem (controller)       |  <-- Logical operations + stats
+-----------------------------------+
|  LTMOperations  |  STMOperations  |  <-- Operation implementations
+-----------------------------------+
|         GraphitiBackend           |  <-- Temporal knowledge graph (Neo4j)
+-----------------------------------+
```

**Operations:**

| Category | Operation | Method | Description |
|----------|-----------|--------|-------------|
| LTM | ADD | `add(content, metadata)` | Store new memory in long-term storage |
| LTM | UPDATE | `update(entry_id, content, metadata)` | Modify existing memory entry |
| LTM | DELETE | `delete(entry_id)` | Remove memory entry |
| STM | RETRIEVE | `retrieve(query, top_k=5)` | Fetch relevant context from LTM to STM |
| STM | SUMMARY | `summary(context)` | Compress conversation history |
| STM | FILTER | `filter(context, relevance_threshold=0.5)` | Remove irrelevant information |

**Combined operation** (`process_query`, line 324): RETRIEVE -> FILTER -> optional SUMMARY pipeline.

**Training support** (`get_operation_reward`, line 350):
- Composite reward: `R = alpha * R_task + beta * R_efficiency + gamma * R_quality`
- Default weights: alpha=0.5, beta=0.3, gamma=0.2
- `R_task`: 1.0 for success, -0.5 for failure
- `R_efficiency`: ratio of average time to actual time
- `R_quality`: operation-specific (RETRIEVE: count/5, SUMMARY: 0.8, FILTER: 0.9 -- placeholders)

**Operation statistics:** Tracked per operation type (count, success count, total time).

### 7.2 Integration Points

Agents interact with AgeMem through `BaseAgent`:
- `_build_context()` (base_agent.py line 198): `await self.memory.retrieve(task)` to populate `memory_context`
- `remember()` (base_agent.py line 239): `await self.memory.add(content, metadata)`
- `recall()` (base_agent.py line 256): `await self.memory.retrieve(query, top_k)`

**Current status:** The AgeMem class is implemented but depends on `GraphitiBackend` and `LTMOperations`/`STMOperations` from sibling modules (`memory/graphiti_backend.py`, `memory/operations.py`). These were part of Sprint 1 but their integration with live agents is a Sprint 3 task (TASK-015). Currently, all agents are instantiated with `memory=None`.

---

## 8. Data Flow Diagrams

### 8.1 Single Coordination Cycle

```
Market Data Input
       |
       v
+------------------+
| MarketAnalyst    |  think(): compute indicators, patterns, regime, sentiment
| .think() / .act()|  act():   interpret -> recommendation {action, confidence, reasons}
+--------+---------+
         |
         | AgentMessage(type="recommendation")
         v
+------------------+      +------------------+
| RiskManager      |      | StrategyAgent    |
| .think() / .act()|      | .think() / .act()|
| VaR, ES, limits  |      | select strategy, |
| compliance check |      | generate signals |
+--------+---------+      +--------+---------+
         |                          |
         | AgentMessage             | AgentMessage
         | (type="recommendation")  | (type="recommendation")
         |                          |
         +----------+   +-----------+
                    |   |
                    v   v
            +------------------+
            | Coordinator      |  think(): collect recommendations, detect conflicts
            | .think() / .act()|  act():   priority-weighted vote OR majority
            |                  |         -> risk veto check
            |                  |         -> final_decision {action, confidence, risk_level}
            +--------+---------+
                     |
                     | AgentMessage(type="final_decision", priority=3)
                     | (broadcast to all agents)
                     v
            +------------------+
            | ExecutionAgent   |  think(): parse trade request, estimate impact,
            | .think() / .act()|          select order type, compute schedule
            |                  |  act():  create order, simulate execution,
            |                  |          compute metrics (slippage, fees, shortfall)
            +------------------+
                     |
                     v
              Order Execution Result
              {order_id, status, fills, metrics}
```

**Note:** In the current implementation, this cycle does not run end-to-end. Agents are standalone -- they do not wire to each other via LatentSpace at runtime. Sprint 3 (TASK-015, TASK-016) will connect them.

### 8.2 Message Flow Through LatentMAS

```
Sending Agent
     |
     | agent.send_message(recipient, content, message_type, priority)
     v
BaseAgent.send_message()
     |
     | Creates AgentMessage(sender, recipient, content, message_type, priority)
     v
LatentSpace.send(message)
     |
     | 1. Acquire asyncio.Lock
     | 2. encode_to_latent(content)
     |    |
     |    | str(content) -> char codes -> [ord(c)%256] -> /256 -> [latent_dim]
     |    | -> LatentEncoder (self-attention + FFN) -> latent vector [latent_dim]
     |    v
     | 3. Create LatentMessage(latent_vector, metadata={_original_content: content})
     | 4. Route:
     |    if recipient == "*":
     |        append to _broadcast_queue (audit)
     |        for each known agent queue (except sender):
     |            append LatentMessage
     |    else:
     |        append to _queues[recipient]
     | 5. Release lock
     v

     ... time passes ...

Receiving Agent
     |
     | context = await agent._build_context(task)  # or explicit receive
     v
LatentSpace.receive(agent_name)
     |
     | 1. Acquire asyncio.Lock
     | 2. Drain _queues[agent_name] via popleft()
     | 3. For each LatentMessage:
     |    |
     |    | _decode_latent_message(latent_msg):
     |    |   if "_original_content" in metadata:
     |    |       content = metadata.pop("_original_content")  # LOSSLESS
     |    |   else:
     |    |       content = decode_from_latent(latent_vector)  # LOSSY fallback
     |    |   -> AgentMessage(sender, recipient, content, ...)
     |    v
     | 4. Sort by priority (descending)
     | 5. Release lock
     v
List[AgentMessage] -> context.messages
```

### 8.3 Alternative Path: MessageRouter

```
Sending Agent
     |
     v
MessageRouter.send(sender_id, receiver_id, message, priority)
     |
     | 1. encoder.encode_agent_state(message) -> latent tensor
     |
     +-- [enable_priority AND not broadcast] ----+
     |                                            |
     |   register_agent(receiver_id) if new       |
     |   priority_queues[receiver_id][priority]   |
     |       .put(latent_tensor)                  |
     |                                            |
     +-- [else: fallback to LatentSpace] ---------+
     |                                            |
     |   Wrap in AgentMessage, LatentSpace.send() |
     +--------------------------------------------+

MessageRouter.receive(receiver_id, decode_mode)
     |
     | 1. Drain priority queues (HIGH -> MEDIUM -> LOW)
     | 2. LatentSpace.receive() -> re-encode content to latent
     | 3. decoder.decode_messages(all_latents, mode) -> outputs
     v
List[decoded outputs]  (tensors, not AgentMessages)
```

### 8.4 Evolution Loop

```
Execution Trace
{agents, interactions, outcome, metadata}
     |
     v
WorkflowDiscovery.analyze_execution(trace)
     |
     | 1. Extract agent_sequence
     | 2. Extract interaction_pattern (communication graph + message types)
     | 3. Generate pattern_id (SHA-256 hash)
     | 4. Create or update WorkflowPattern
     |    - Incremental success rate update
     |    - Increment use_count
     v
WorkflowPattern stored in workflow_library
     |
     | (periodically)
     v
AgentGenerator.generate_from_successful_patterns()
     |
     | 1. WorkflowDiscovery.get_successful_patterns()
     |    (success_rate >= 0.7, use_count >= 3)
     | 2. Filter patterns with success_rate >= min_pattern_success (0.8)
     | 3. For each qualifying pattern:
     |    |
     |    | generate_from_pattern(pattern):
     |    |   infer agent_type (most frequent agent name)
     |    |   extract capabilities (keyword matching)
     |    |   generate parameters (tuned by success rate)
     |    |   -> AgentConfiguration
     |    v
     | 4. Prune lowest performers if > max_generated_configs
     v
AgentConfiguration stored in generated_configs
     |
     | (in parallel)
     v
CooperativeEvolution
     |
     | add_experience(agent_id, experience):
     |   -> per-agent buffer
     |   -> shared_pool (if reward >= threshold)
     |
     | replay_experiences(agent_id):
     |   -> 90% from own buffer
     |   -> 10% from shared pool (others' experiences)
     |
     | cross_pollinate(top_k=3):
     |   -> identify top performers
     |   -> seed shared pool with their best experiences
     v
Agents learn from shared experiences
```

---

## 9. Key Design Decisions and Rationale

### 9.1 LatentMAS Instead of Direct Message Passing

**Decision:** Agents communicate through a shared latent space with transformer-based encoding rather than passing structured messages directly.

**Rationale:** Latent space communication enables:
- Learned, compressed representations that can evolve with training
- Attention-based message routing for selective broadcast
- A shared representation space that agents can jointly optimize
- Future compatibility with gradient-based communication optimization

**Trade-off:** Currently, the `_original_content` passthrough means messages are not actually transformed by the neural encoder/decoder -- the latent path is bypassed in favor of lossless content preservation. The neural path will become meaningful once the encoder/decoder are trained.

### 9.2 Single-Pass Coordinator (`done: True` Always)

**Decision:** The CoordinatorAgent sets `done: True` whenever it receives any recommendations or resource requests (coordinator.py line 156), ensuring it never iterates.

**Rationale:** The coordinator's role is aggregation and decision-making, not iterative analysis. It collects inputs from other agents, resolves conflicts, and produces a final decision in one pass. Multi-pass coordination would require inter-round communication with other agents, which is deferred to Sprint 3 integration.

**Trade-off:** The coordinator cannot request additional information from agents. If it receives incomplete inputs, it decides with what it has.

### 9.3 Priority-Weighted Conflict Resolution

**Decision:** Conflicts are resolved by weighted voting where `weight = priority * confidence`, with risk_manager having priority 3 (highest).

**Rationale:** In financial trading, risk management should have the strongest voice. A risk-aware buy recommendation from the risk manager outweighs a bullish signal from the analyst. The multiplicative weighting ensures both expertise (priority) and certainty (confidence) contribute to the decision.

**Trade-off:** The simple priority scheme does not adapt to context. In a low-risk environment, the risk manager's conservative bias may dominate unnecessarily. More sophisticated weighting (e.g., regime-dependent) is deferred.

### 9.4 `_original_content` Passthrough

**Decision:** When `LatentSpace.send()` encodes a message, it stores the original content in `metadata["_original_content"]`. On `receive()`, this is popped and used directly, bypassing neural decoding.

**Rationale:** The transformer encoder/decoder are initialized with random weights and are not yet trained. Neural encoding followed by decoding with untrained weights would destroy message content. The passthrough ensures functional correctness while the neural path exists for future training.

**Trade-off:** The latent vector in the `LatentMessage` is computed but never used for decoding. It exists primarily so the encode/decode infrastructure is exercised and ready for training.

### 9.5 Per-Agent Async Queues vs. Shared Broadcast Queue

**Decision:** Broadcasts fan out to per-agent queues at send time. The `_broadcast_queue` is retained only as a bounded audit log.

**Rationale:** The original design used a separate broadcast queue that `receive()` was supposed to drain. This meant each agent had to scan and filter the broadcast queue on every receive, and messages could be consumed multiple times or missed entirely. The fan-out design guarantees each agent gets exactly one copy and drains it from its own queue, consistent with unicast behavior.

**Trade-off:** Fan-out only reaches agents with existing queues. An agent registered after a broadcast misses the message. Explicit agent registration (review finding N2) would fix this.

### 9.6 `HAS_TORCH` Optional Dependency Pattern

**Decision:** PyTorch is imported with `try/except ImportError` and a `HAS_TORCH` flag. Components that require torch raise `ImportError` with install instructions in their `__init__`.

**Rationale:** Allows non-communication code (agents, evolution, config) to be imported and used without torch installed. This is useful for testing, CI, and environments where only specific layers are needed.

**Trade-off:** `LatentEncoder` and `LatentDecoder` in `latent_space.py` inherit from `nn.Module` at class definition time (not behind an `if HAS_TORCH:` guard), so importing `communication.latent_space` still fails without torch. The encoder and decoder modules handle this with dummy class stubs.

### 9.7 `weights_only=True` on All `torch.load()` Calls

**Decision:** All `torch.load()` calls explicitly pass `weights_only=True`.

**Rationale:** Without this flag (on PyTorch < 2.6), `torch.load()` uses `pickle.load()` which allows arbitrary code execution from a malicious model file. This is a critical security hardening measure.

### 9.8 Character-Code Tokenization for String Inputs

**Decision:** Both `LatentSpace.encode_to_latent()` and `AgentStateEncoder.encode_agent_state()` convert strings to normalized character codes (`ord(c)/255.0` or `ord(c)%256/256.0`).

**Rationale:** A simple, deterministic mapping that works without a trained tokenizer or embedding model. It provides a minimal representation for the MLP/transformer to operate on.

**Trade-off:** Character-code tokenization is semantically meaningless -- "buy" and "yub" produce similar magnitude vectors. This is acceptable because the `_original_content` passthrough handles actual message content, and trained embeddings will replace this in the future.

### 9.9 Monotonic Counter for AgentGenerator Config IDs

**Decision:** Config IDs use `self._next_config_id` (monotonically increasing) instead of `len(generated_configs)`.

**Rationale:** After pruning, `len(generated_configs)` can return a previously-used value, causing silent overwrites. A monotonic counter guarantees uniqueness across the generator's lifetime.

---

## 10. Known Limitations and Technical Debt

### 10.1 Blocking Issues for Production

| ID | Component | Issue | Impact |
|----|-----------|-------|--------|
| B1 | All agents | No LLM integration -- agents use hardcoded logic, not model inference | Agents cannot learn or adapt; all behavior is rule-based |
| B2 | Memory | AgeMem requires Graphiti backend (Neo4j) -- not yet integrated with agents | No persistent memory across sessions |
| B3 | Communication | Encoder/decoder weights are random (untrained) | Neural communication path is nonfunctional; only passthrough works |
| B4 | Learning | Nested Learning and RepExp not implemented | No meta-learning or exploration |
| B5 | Trading | No real market data, no broker integration, no order management system | System can only operate on synthetic data |
| B6 | Testing | No tests exist for any Sprint 2 code | No automated verification of correctness |

### 10.2 Open Review Findings (Should-Fix)

**From Sprint 2 Batch 1 review:**

| ID | File | Finding |
|----|------|---------|
| N1 | `agents/market_analyst.py:270-273` | MACD series computation is O(n^2) -- recalculates EMA from scratch per index |
| N2 | `communication/latent_space.py:249` | Broadcast fan-out only reaches agents with existing queues |
| N3 | `agents/risk_manager.py:146` | `think()` always returns `done: False` on main path -- will hit max_iterations |
| N4 | `agents/strategy_agent.py:540` | Backtest `total_return` is additive sum while drawdown uses multiplicative compounding |
| N5 | `communication/latent_space.py:47,88` | `LatentEncoder`/`LatentDecoder` fail at import time without torch |
| W1 | `agents/risk_manager.py` | Portfolio return aggregation duplicated 3 times |
| W2 | `agents/execution_agent.py` | `random.random()` not seeded -- non-deterministic execution simulation |
| W3 | `agents/execution_agent.py` | No defensive `.get()` fallback for config key access |
| W5 | `agents/market_analyst.py` | Uses raw `config.get()` instead of `self.config.get()` |

**From Sprint 2 Batch 2 review:**

| ID | File | Finding |
|----|------|---------|
| S7 | `communication/encoder.py:80-81` | Hidden dims `[512, 256]` creates redundant 512->512 first layer |
| S8 | `communication/encoder.py:115` | Per-layer `.to(device)` inconsistent with decoder's `self.to(device)` |
| S9 | `communication/decoder.py:260-263` | `decode_messages` processes sequentially -- could batch |
| S10 | `evolution/workflow_discovery.py:102` | `execution_history` is unbounded list |
| S11 | `evolution/workflow_discovery.py:366-367` | `save_library`/`load_library` no error handling on file I/O |

**From Sprint 2 Batch 3 review:**

| ID | File | Finding |
|----|------|---------|
| S3 | `communication/router.py:232-249` | Non-atomic receive across priority queues and LatentSpace |
| S4 | `communication/router.py:362-366` | N+1 encoding in `broadcast_with_attention` |
| S5 | Evolution modules | f-string logger calls instead of `%s`-style |
| S6 | `evolution/cooperative_evolution.py:389` | `cross_pollinate` cutoff inverts for negative rewards |
| S7 | Evolution modules | Sync file I/O inside async methods blocks event loop |
| S8 | `communication/router.py:18-20` | Absolute imports; rest of package uses relative imports |

### 10.3 Architectural Gaps (Sprint 3+)

| Gap | What is Missing | When |
|-----|----------------|------|
| Agent-Memory wiring | Agents are instantiated with `memory=None`; no runtime connection to AgeMem | Sprint 3 (TASK-015) |
| Agent-Communication wiring | Agents are instantiated with `communication=None`; no runtime LatentSpace connection | Sprint 3 (TASK-016) |
| End-to-end pipeline | The coordination cycle (Section 8.1) does not execute as a connected pipeline | Sprint 3 (TASK-017) |
| Nested Learning | Inner/outer loop meta-learning framework | Sprint 4 (TASK-013) |
| RepExp | Representation-based exploration for diversity | Sprint 4 (TASK-014) |
| Market data module | Real/mock data feeds | Sprint 5 (TASK-018) |
| Order management | Position tracking, P&L | Sprint 5 (TASK-019, TASK-020) |
| Test suite | Unit and integration tests | Sprint 5 (TASK-023) |
| Training infrastructure | OLMoE fine-tuning, AgeMem GRPO, LatentMAS encoder/decoder training | Not yet scheduled |

---

## 11. Sprint Roadmap Summary

### Sprint 1: Foundation and Core Abstractions -- COMPLETE

Delivered `BaseAgent` ABC, `AthenaConfig` hierarchy, `AgeMem` interface, project structure.

### Sprint 2: Parallel Layer Implementation -- COMPLETE

12 tasks, all accepted after 3 review batches (18 must-fix issues found and resolved).

| Task | Component | Status |
|------|-----------|--------|
| TASK-001 | MarketAnalystAgent | Accepted |
| TASK-002 | RiskManagerAgent | Accepted |
| TASK-003 | StrategyAgent | Accepted |
| TASK-004 | ExecutionAgent | Accepted |
| TASK-005 | CoordinatorAgent | Accepted |
| TASK-006 | LatentSpace | Accepted |
| TASK-007 | AgentStateEncoder | Accepted |
| TASK-008 | AgentStateDecoder | Accepted |
| TASK-009 | MessageRouter | Accepted |
| TASK-010 | WorkflowDiscovery | Accepted |
| TASK-011 | AgentGenerator | Accepted |
| TASK-012 | CooperativeEvolution | Accepted |

### Sprint 3: Layer Integration -- NEXT

| Task | Description | Dependencies | What Changes |
|------|-------------|-------------|--------------|
| TASK-015 | Integrate Agents with AgeMem | TASK-005 | Agents instantiated with live `AgeMem` instance. `_build_context()` populates `memory_context`. Agents can `remember()` and `recall()`. |
| TASK-016 | Integrate Agents with LatentMAS | TASK-005, TASK-009 | Agents instantiated with live `LatentSpace`. Messages flow through `send_message()` and are received in `_build_context()`. |
| TASK-017 | End-to-End Pipeline Integration Test | TASK-015, TASK-016 | Full coordination cycle runs: data in -> MarketAnalyst -> RiskManager + Strategy -> Coordinator -> ExecutionAgent. Validates message flow and decision pipeline. |

**What Sprint 3 enables:** The first runnable multi-agent pipeline. Agents will communicate via LatentMAS and persist knowledge via AgeMem. The coordination cycle (Section 8.1) will execute end-to-end.

### Sprint 4: Advanced Features and Learning Layer

| Task | Description | What it Delivers |
|------|-------------|-----------------|
| TASK-013 | Nested Learning Framework | Inner loop (task-specific adaptation) and outer loop (meta-learning). Knowledge consolidation. |
| TASK-014 | RepExp Exploration Module | Representation-space diversity measurement, exploration bonuses, test-time exploration. |

**What Sprint 4 enables:** Agents can learn and adapt through meta-learning. RepExp provides exploration diversity to prevent mode collapse in agent behavior.

### Sprint 5: Trading Domain and Testing

| Task | Description | What it Delivers |
|------|-------------|-----------------|
| TASK-018 | Trading Market Data Module | Mock and real data feeds (yahoo, alpaca). Feature engineering pipeline. |
| TASK-019 | Trading Order Management Module | Order execution interface, position tracking, P&L. |
| TASK-020 | Trading Portfolio Module | Portfolio construction, rebalancing, performance attribution. |
| TASK-021 | Data Scrapers | News/SEC filings, market data, social/sentiment scraping. |
| TASK-022 | Data Processors and Dataset Classes | Text cleaning, formatting, PyTorch datasets for training. |
| TASK-023 | Comprehensive Test Suite | Unit tests per component, integration tests, backtesting validation. |

**What Sprint 5 enables:** Connection to real financial data, a testable trading pipeline, and automated quality gates via comprehensive tests.

### Beyond Sprint 5 (Not Yet Scheduled)

- **Training pipeline**: OLMoE fine-tuning on financial data (Stage 1), AgeMem GRPO training (Stage 2)
- **LatentMAS encoder/decoder training**: Currently random weights; needs supervised or self-supervised training
- **Paper trading**: Real-time validation without capital risk
- **Production hardening**: Error recovery, monitoring, circuit breakers

---

*End of design document.*
