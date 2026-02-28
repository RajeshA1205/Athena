"""
ATHENA Interactive CLI
======================
Ask questions and get investment recommendations from the multi-agent system.

Usage:
    python cli.py
    python cli.py --verbose
    python cli.py --symbol AAPL
    python cli.py --log-level DEBUG
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional


def _load_dotenv():
    """Load .env file from project root if it exists."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()

from core.utils import setup_logging
from core.base_agent import AgentContext, AgentMessage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="athena-cli",
        description="ATHENA Interactive Financial Advisory CLI",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Default symbol to analyze (e.g. AAPL, TSLA)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Show full agent thought process, internal state, and timing",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Logging level (default: WARNING to keep output clean)",
    )
    return parser.parse_args()


DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"
MAGENTA = "\033[35m"


def _json_compact(obj, indent=4, max_depth=3, _depth=0) -> str:
    """Pretty-print a dict/list with controlled depth."""
    if _depth >= max_depth:
        return json.dumps(obj, default=str)
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        parts = []
        prefix = " " * indent * (_depth + 1)
        for k, v in obj.items():
            val = _json_compact(v, indent, max_depth, _depth + 1)
            parts.append(f"{prefix}{k}: {val}")
        return "{\n" + "\n".join(parts) + "\n" + " " * indent * _depth + "}"
    if isinstance(obj, list):
        if not obj:
            return "[]"
        if len(obj) <= 5 and all(not isinstance(x, (dict, list)) for x in obj):
            return "[" + ", ".join(str(x) for x in obj) + "]"
        parts = []
        prefix = " " * indent * (_depth + 1)
        for x in obj[:10]:
            parts.append(f"{prefix}{_json_compact(x, indent, max_depth, _depth + 1)}")
        if len(obj) > 10:
            parts.append(f"{prefix}... ({len(obj) - 10} more)")
        return "[\n" + "\n".join(parts) + "\n" + " " * indent * _depth + "]"
    if isinstance(obj, float):
        return f"{obj:.6f}" if abs(obj) < 1 else f"{obj:.4f}"
    return str(obj)


class AthenaShell:
    """Interactive shell for the Athena multi-agent system."""

    def __init__(self, default_symbol: Optional[str] = None, verbose: bool = False):
        from agents.coordinator import CoordinatorAgent
        from agents.market_analyst import MarketAnalystAgent
        from agents.risk_manager import RiskManagerAgent
        from agents.strategy_agent import StrategyAgent
        from agents.execution_agent import ExecutionAgent
        from trading.market_data import MarketDataFeed, MarketDataMode
        from memory.agemem import AgeMem
        from memory.graphiti_backend import GraphitiBackend
        from learning.nested_learning import NestedLearning
        from core.config import LearningConfig

        self.feed = MarketDataFeed(mode=MarketDataMode.MOCK)
        self.default_symbol = default_symbol or "AAPL"
        self.symbols = MarketDataFeed.MOCK_SYMBOLS
        self.verbose = verbose

        # Memory layer — AgeMem backed by Graphiti/Neo4j
        self.graphiti = GraphitiBackend(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
        )
        self.memory = AgeMem(backend=self.graphiti)
        self._memory_initialized = False

        # Nested learning — per-agent bilevel meta-learning
        learning_config = LearningConfig()
        self.learners: Dict[str, NestedLearning] = {
            "market_analyst": NestedLearning(learning_config, "market_analyst"),
            "risk_manager": NestedLearning(learning_config, "risk_manager"),
            "strategy_agent": NestedLearning(learning_config, "strategy_agent"),
            "execution_agent": NestedLearning(learning_config, "execution_agent"),
            "coordinator": NestedLearning(learning_config, "coordinator"),
        }
        self.query_count = 0

        # Agents — all wired to shared memory
        self.market_analyst = MarketAnalystAgent(memory=self.memory)
        self.risk_manager = RiskManagerAgent(memory=self.memory)
        self.strategy_agent = StrategyAgent(memory=self.memory)
        self.execution_agent = ExecutionAgent(memory=self.memory)

        self.coordinator = CoordinatorAgent(memory=self.memory)
        self.coordinator.register_agent("market_analyst", self.market_analyst)
        self.coordinator.register_agent("risk_manager", self.risk_manager)
        self.coordinator.register_agent("strategy_agent", self.strategy_agent)
        self.coordinator.register_agent("execution_agent", self.execution_agent)

    def _log(self, agent_name: str, phase: str, data: dict, elapsed_ms: float):
        """Print verbose agent trace."""
        if not self.verbose:
            return
        color = {
            "market_analyst": CYAN,
            "risk_manager": YELLOW,
            "strategy_agent": GREEN,
            "execution_agent": MAGENTA,
            "coordinator": BOLD,
        }.get(agent_name, "")

        print(f"\n{color}{'─' * 60}")
        print(f"  [{agent_name}] {phase.upper()}  ({elapsed_ms:.1f}ms)")
        print(f"{'─' * 60}{RESET}")
        print(f"{DIM}{_json_compact(data)}{RESET}")

    async def _ensure_memory(self):
        """Initialize memory on first use."""
        if not self._memory_initialized:
            ok = await self.memory.initialize()
            self._memory_initialized = True
            if self.verbose:
                stats = await self.memory.get_stats()
                using = "Graphiti/Neo4j" if stats["backend"].get("using_graphiti") else "in-memory fallback"
                episodes = stats["backend"].get("episode_count", 0)
                print(f"  {DIM}Memory: {using} ({episodes} episodes stored){RESET}")

    async def analyze_symbol(self, symbol: str) -> Dict:
        """Run the full agent pipeline for a symbol."""
        from learning.nested_learning import TaskTrajectory

        await self._ensure_memory()
        self.query_count += 1

        bar = await self.feed.get_realtime_data(symbol)
        if bar is None:
            return {"error": f"No data available for {symbol}"}

        bar_dict = asdict(bar)
        hist_bars = await self.feed.get_historical_data(symbol, days=60)
        prices = [asdict(b)["close"] for b in hist_bars]
        prices.append(bar_dict["close"])

        market_data = {
            "symbol": symbol,
            "prices": prices,
            "bar": bar_dict,
        }

        if self.verbose:
            print(f"\n{BOLD}{'=' * 60}")
            print(f"  Agent Pipeline: {symbol}  (query #{self.query_count})")
            print(f"  Price points: {len(prices)} (60d history + realtime)")
            print(f"{'=' * 60}{RESET}")

        # 1. Market Analyst
        t0 = time.perf_counter()
        analyst_ctx = AgentContext(
            task=f"Analyze {symbol} market conditions",
            metadata={"market_data": market_data, "symbol": symbol},
        )
        analyst_thought = await self.market_analyst.think(analyst_ctx)
        t_think = (time.perf_counter() - t0) * 1000
        self._log("market_analyst", "think", analyst_thought, t_think)

        t0 = time.perf_counter()
        analyst_action = await self.market_analyst.act(analyst_thought)
        t_act = (time.perf_counter() - t0) * 1000
        self._log("market_analyst", "act -> result", analyst_action.result or {}, t_act)

        # 2. Risk Manager
        t0 = time.perf_counter()
        risk_ctx = AgentContext(
            task=f"Assess risk for {symbol}",
            metadata={
                "market_data": market_data,
                "symbol": symbol,
                "portfolio": {"positions": {symbol: {"weight": 0.1}}},
            },
        )
        risk_thought = await self.risk_manager.think(risk_ctx)
        t_think = (time.perf_counter() - t0) * 1000
        self._log("risk_manager", "think", risk_thought, t_think)

        t0 = time.perf_counter()
        risk_action = await self.risk_manager.act(risk_thought)
        t_act = (time.perf_counter() - t0) * 1000
        self._log("risk_manager", "act -> result", risk_action.result or {}, t_act)

        # 3. Strategy Agent
        t0 = time.perf_counter()
        strategy_ctx = AgentContext(
            task=f"Generate trading strategy for {symbol}",
            metadata={
                "market_data": market_data,
                "symbol": symbol,
                "analysis": analyst_action.result if analyst_action.result else {},
                "risk": risk_action.result if risk_action.result else {},
            },
        )
        strategy_thought = await self.strategy_agent.think(strategy_ctx)
        t_think = (time.perf_counter() - t0) * 1000
        self._log("strategy_agent", "think", strategy_thought, t_think)

        t0 = time.perf_counter()
        strategy_action = await self.strategy_agent.act(strategy_thought)
        t_act = (time.perf_counter() - t0) * 1000
        self._log("strategy_agent", "act -> result", strategy_action.result or {}, t_act)

        # 4. Execution Agent (recommendation only)
        signal = {}
        if strategy_action.result and isinstance(strategy_action.result, dict):
            sigs = strategy_action.result.get("signals", [])
            if sigs and isinstance(sigs, list) and isinstance(sigs[0], dict):
                signal = sigs[0]

        from trading.enums import OrderSide
        side = OrderSide.BUY if signal.get("type") == "buy" else OrderSide.SELL

        t0 = time.perf_counter()
        exec_ctx = AgentContext(
            task=f"Generate execution recommendation for {symbol}",
            metadata={
                "market_data": market_data,
                "symbol": symbol,
                "trade_request": {
                    "symbol": symbol,
                    "side": side,
                    "quantity": signal.get("position_size", 100),
                    "urgency": "normal",
                },
                "market_conditions": {
                    "current_price": bar_dict["close"],
                    "avg_volume": bar_dict["volume"],
                    "volatility": analyst_thought.get("volatility", 0.02)
                        if isinstance(analyst_thought, dict) else 0.02,
                },
            },
        )
        exec_thought = await self.execution_agent.think(exec_ctx)
        t_think = (time.perf_counter() - t0) * 1000
        self._log("execution_agent", "think (execution plan)", exec_thought, t_think)

        t0 = time.perf_counter()
        exec_action = await self.execution_agent.act(exec_thought)
        t_act = (time.perf_counter() - t0) * 1000
        self._log("execution_agent", "act -> result", exec_action.result or {}, t_act)

        # 5. Coordinator aggregation
        recommendations = {}
        if analyst_action.result and isinstance(analyst_action.result, dict):
            recommendations["market_analyst"] = {
                "action": analyst_action.result.get("action", "hold"),
                "confidence": analyst_action.result.get("confidence", 0.5),
            }

        if risk_action.result and isinstance(risk_action.result, dict):
            recommendations["risk_manager"] = {
                "action": "hold",
                "risk_level": risk_action.result.get("risk_level", "unknown"),
                "confidence": 0.7,
            }

        if strategy_action.result and isinstance(strategy_action.result, dict):
            sigs = strategy_action.result.get("signals", [])
            if sigs and isinstance(sigs, list) and isinstance(sigs[0], dict):
                sig = sigs[0]
                recommendations["strategy_agent"] = {
                    "action": sig.get("type", "hold"),
                    "confidence": sig.get("strength", 0.5),
                }

        if self.verbose:
            print(f"\n{BOLD}{'─' * 60}")
            print(f"  [coordinator] AGGREGATING RECOMMENDATIONS")
            print(f"{'─' * 60}{RESET}")
            for name, rec in recommendations.items():
                action = rec.get("action", "?").upper()
                conf = rec.get("confidence", 0)
                print(f"  {DIM}{name:20s} -> {action:6s} (confidence: {conf:.0%}){RESET}")

        messages = [
            AgentMessage(
                sender=name,
                recipient="coordinator",
                content=rec,
                message_type="recommendation",
            )
            for name, rec in recommendations.items()
        ]

        t0 = time.perf_counter()
        coord_ctx = AgentContext(
            task=f"Make final recommendation for {symbol}",
            messages=messages,
            metadata={"market_data": market_data, "symbol": symbol},
        )
        coord_thought = await self.coordinator.think(coord_ctx)
        t_think = (time.perf_counter() - t0) * 1000
        self._log("coordinator", "think (orchestration plan)", coord_thought, t_think)

        t0 = time.perf_counter()
        coord_action = await self.coordinator.act(coord_thought)
        t_act = (time.perf_counter() - t0) * 1000
        self._log("coordinator", "act -> final decision", coord_action.result or {}, t_act)

        # ── 6. Nested Learning: build trajectories and adapt ──
        agent_actions = {
            "market_analyst": analyst_action,
            "risk_manager": risk_action,
            "strategy_agent": strategy_action,
            "execution_agent": exec_action,
            "coordinator": coord_action,
        }

        if self.verbose:
            print(f"\n{BOLD}{'─' * 60}")
            print(f"  NESTED LEARNING: Inner-loop adaptation")
            print(f"{'─' * 60}{RESET}")

        for agent_name, action in agent_actions.items():
            reward = 1.0 if action.success else -0.5
            # Confidence-weighted reward for agents that emit confidence
            if action.result and isinstance(action.result, dict):
                conf = action.result.get("confidence", None)
                if conf is not None:
                    reward *= conf

            trajectory = TaskTrajectory(
                task_id=f"{symbol}_analysis",
                agent_id=agent_name,
                states=[{"symbol": symbol, "query_num": self.query_count}],
                actions=[{"type": action.action_type, "success": action.success}],
                rewards=[reward],
                metadata={"symbol": symbol, "duration": action.duration},
            )

            learner = self.learners[agent_name]
            adapt_result = await learner.adapt_to_task(f"{symbol}_analysis", trajectory)

            if self.verbose:
                lr = adapt_result["meta_params_snapshot"]["lr_scale"]
                gain = adapt_result["adaptation_gain"]
                perf = adapt_result["task_performance"]
                baseline = adapt_result["meta_params_snapshot"]["baseline_performance"]
                print(f"  {DIM}{agent_name:20s}  reward={reward:+.3f}  "
                      f"perf={perf:.3f}  gain={gain:+.3f}  "
                      f"lr_scale={lr:.4f}  baseline={baseline:.4f}{RESET}")

        # Outer-loop update every 5 queries
        if self.query_count % 5 == 0:
            if self.verbose:
                print(f"\n{BOLD}  NESTED LEARNING: Outer-loop meta-update (every 5 queries){RESET}")

            for agent_name, learner in self.learners.items():
                all_trajs = []
                for trajs in learner.task_trajectories.values():
                    all_trajs.extend(trajs[-10:])  # recent 10 per task
                if all_trajs:
                    meta_result = await learner.update_meta_parameters(all_trajs)
                    if self.verbose:
                        mp = meta_result["updated_params"]
                        print(f"  {DIM}{agent_name:20s}  "
                              f"mean_perf={meta_result['mean_performance']:.4f}  "
                              f"lr_scale={mp['lr_scale']:.4f}  "
                              f"baseline={mp['baseline_performance']:.4f}  "
                              f"updates=#{meta_result['update_count']}{RESET}")

            # Knowledge consolidation every 20 queries
            if self.query_count % 20 == 0:
                if self.verbose:
                    print(f"\n{BOLD}  NESTED LEARNING: Knowledge consolidation{RESET}")
                for agent_name, learner in self.learners.items():
                    cons = await learner.consolidate_knowledge()
                    if self.verbose:
                        print(f"  {DIM}{agent_name:20s}  "
                              f"tasks={cons['consolidated_tasks']}  "
                              f"mean_reward={cons['mean_reward']:.4f}  "
                              f"pruned={cons['pruned_entries']}{RESET}")

        # ── 7. Get memory stats for output ──
        mem_stats = await self.memory.get_stats()

        return {
            "symbol": symbol,
            "price": bar_dict,
            "analyst": analyst_action.result,
            "risk": risk_action.result,
            "strategy": strategy_action.result,
            "execution": exec_action.result,
            "final_decision": coord_action.result.get("final_decision")
                if coord_action.result else None,
            "memory_stats": mem_stats,
        }

    async def handle_query(self, query: str) -> str:
        """Route a user query to the appropriate analysis."""
        query_lower = query.lower().strip()

        # Extract symbol from query if present
        symbol = self.default_symbol
        for s in self.symbols:
            if s.lower() in query_lower:
                symbol = s
                break

        # Command routing
        if query_lower in ("help", "?"):
            return self._help_text()

        if query_lower == "symbols":
            return f"Available symbols: {', '.join(self.symbols)}"

        if query_lower.startswith("set "):
            parts = query_lower.split()
            if len(parts) >= 2:
                new_sym = parts[1].upper()
                if new_sym in self.symbols:
                    self.default_symbol = new_sym
                    return f"Default symbol set to {new_sym}"
                return f"Unknown symbol: {new_sym}. Available: {', '.join(self.symbols)}"

        if query_lower == "verbose on":
            self.verbose = True
            return "Verbose mode ON"
        if query_lower == "verbose off":
            self.verbose = False
            return "Verbose mode OFF"

        if query_lower == "stats":
            return await self._show_stats()

        if query_lower == "memory":
            return await self._show_memory()

        # Run full analysis
        start = time.perf_counter()
        result = await self.analyze_symbol(symbol)
        elapsed = time.perf_counter() - start

        return self._format_result(result, query, elapsed)

    def _format_result(self, result: Dict, query: str, elapsed: float) -> str:
        """Format analysis result for display."""
        if "error" in result:
            return f"Error: {result['error']}"

        lines = []
        symbol = result["symbol"]
        price = result.get("price", {})

        lines.append(f"\n{'=' * 60}")
        lines.append(f"  ATHENA Analysis: {symbol}")
        lines.append(f"{'=' * 60}")

        # Price info
        if price:
            lines.append(f"\n  Price Data:")
            lines.append(f"    Open:   ${price.get('open', 0):.2f}")
            lines.append(f"    High:   ${price.get('high', 0):.2f}")
            lines.append(f"    Low:    ${price.get('low', 0):.2f}")
            lines.append(f"    Close:  ${price.get('close', 0):.2f}")
            lines.append(f"    Volume: {price.get('volume', 0):,.0f}")

        # Market Analysis
        analyst = result.get("analyst")
        if analyst and isinstance(analyst, dict):
            lines.append(f"\n  Market Analysis:")
            evidence = analyst.get("supporting_evidence", {})
            regime = evidence.get("regime", {}) if isinstance(evidence, dict) else {}
            if isinstance(regime, dict) and regime:
                lines.append(f"    Regime:     {regime.get('type', 'N/A')}")
                lines.append(f"    Trend:      {regime.get('trend', 'N/A')}")
                lines.append(f"    Volatility: {regime.get('volatility', 'N/A')}")
            indicators = evidence.get("indicators", {}) if isinstance(evidence, dict) else {}
            if isinstance(indicators, dict):
                rsi = indicators.get("rsi_14")
                if rsi is not None:
                    lines.append(f"    RSI(14):    {rsi:.1f}")
                macd = indicators.get("macd")
                if macd is not None:
                    lines.append(f"    MACD:       {macd:.4f}")
            reasons = analyst.get("reasons", [])
            if reasons:
                lines.append(f"    Signals:")
                for r in reasons[:4]:
                    lines.append(f"      - {r}")

        # Risk Assessment
        risk = result.get("risk")
        if risk and isinstance(risk, dict):
            lines.append(f"\n  Risk Assessment:")
            lines.append(f"    Risk Level: {risk.get('risk_level', 'N/A')}")
            var95 = risk.get("var_95")
            if var95 is not None:
                lines.append(f"    VaR (95%):  {var95:.4f}")
            es = risk.get("expected_shortfall")
            if es is not None:
                lines.append(f"    Exp. Short: {es:.4f}")
            violations = risk.get("compliance_violations", [])
            if violations:
                lines.append(f"    Violations: {', '.join(str(v) for v in violations)}")
            else:
                lines.append(f"    Compliance: OK")

        # Strategy
        strategy = result.get("strategy")
        if strategy and isinstance(strategy, dict):
            strat_info = strategy.get("strategy", {})
            if isinstance(strat_info, dict) and strat_info:
                lines.append(f"\n  Strategy:")
                lines.append(f"    Type:      {strat_info.get('type', 'N/A')}")
                lines.append(f"    Regime:    {strat_info.get('market_regime', 'N/A')}")
            sigs = strategy.get("signals", [])
            if sigs and isinstance(sigs, list):
                for i, sig in enumerate(sigs[:3]):
                    if not isinstance(sig, dict):
                        continue
                    label = f"Signal {i+1}" if len(sigs) > 1 else "Signal"
                    lines.append(f"\n  {label}:")
                    lines.append(f"    Action:      {sig.get('type', 'N/A')}")
                    lines.append(f"    Strength:    {sig.get('strength', 0):.2f}")
                    lines.append(f"    Reasoning:   {sig.get('reasoning', 'N/A')}")
                    if sig.get("entry_price"):
                        lines.append(f"    Entry:       ${sig['entry_price']:.2f}")
                    if sig.get("stop_loss"):
                        lines.append(f"    Stop Loss:   ${sig['stop_loss']:.2f}")
                    if sig.get("take_profit"):
                        lines.append(f"    Take Profit: ${sig['take_profit']:.2f}")
                    if sig.get("position_size"):
                        lines.append(f"    Position:    {sig['position_size']:.0%}")

        # Execution recommendation
        execution = result.get("execution")
        if execution and isinstance(execution, dict):
            order = execution.get("order", {})
            metrics = execution.get("metrics", {})
            if order:
                lines.append(f"\n  Execution Recommendation:")
                lines.append(f"    Order Type:  {order.get('order_type', 'N/A')}")
                lines.append(f"    Side:        {order.get('side', 'N/A')}")
                lines.append(f"    Quantity:    {order.get('quantity', 0)}")
                lines.append(f"    Avg Price:   ${order.get('avg_fill_price', 0):.2f}")
            if metrics:
                lines.append(f"    Slippage:    {metrics.get('slippage_bps', 0):.2f} bps")
                lines.append(f"    Fees:        ${metrics.get('total_fees', 0):.2f}")

        # Final Decision
        decision = result.get("final_decision")
        if decision and isinstance(decision, dict):
            lines.append(f"\n  {'─' * 56}")
            action = decision.get("action", "N/A").upper()
            confidence = decision.get("confidence", 0)
            risk_level = decision.get("risk_level", "N/A")
            lines.append(f"  RECOMMENDATION:  {action}")
            lines.append(f"  Confidence:      {confidence:.0%}")
            lines.append(f"  Risk Level:      {risk_level}")
            reasoning = decision.get("reasoning", "")
            if reasoning:
                lines.append(f"  Reasoning:       {reasoning}")

        # Memory + learning footer
        mem_stats = result.get("memory_stats", {})
        backend = mem_stats.get("backend", {})
        ops = mem_stats.get("operations", {})
        episodes = backend.get("episode_count", 0)
        adds = ops.get("add", {}).get("count", 0)
        retrieves = ops.get("retrieve", {}).get("count", 0)

        lines.append(f"\n{'=' * 60}")
        lines.append(f"  Analysis completed in {elapsed:.2f}s  |  Query #{self.query_count}")
        lines.append(f"  Memory: {episodes} episodes stored  |  {adds} adds, {retrieves} retrieves")
        lines.append(f"  NOTE: This is a simulated recommendation, not financial advice.")
        lines.append(f"{'=' * 60}\n")

        return "\n".join(lines)

    async def _show_stats(self) -> str:
        """Show learning and memory stats."""
        await self._ensure_memory()
        lines = [f"\n{'=' * 60}", "  ATHENA System Stats", f"{'=' * 60}"]

        lines.append(f"\n  Queries: {self.query_count}")

        # Memory stats
        mem = await self.memory.get_stats()
        backend = mem.get("backend", {})
        lines.append(f"\n  Memory Backend:")
        lines.append(f"    Using Graphiti: {backend.get('using_graphiti', False)}")
        lines.append(f"    Episodes:      {backend.get('episode_count', 0)}")
        lines.append(f"    Neo4j URI:     {backend.get('neo4j_uri', 'N/A')}")

        ops = mem.get("operations", {})
        lines.append(f"\n  Memory Operations:")
        for op_name, stats in ops.items():
            count = stats.get("count", 0)
            success = stats.get("success", 0)
            avg_ms = (stats.get("total_time", 0) / count * 1000) if count > 0 else 0
            lines.append(f"    {op_name:10s}  count={count:4d}  success={success:4d}  avg={avg_ms:.1f}ms")

        # Learning stats
        lines.append(f"\n  Nested Learning:")
        for agent_name, learner in self.learners.items():
            s = learner.get_stats()
            lines.append(f"    {agent_name}:")
            lines.append(f"      Trajectories: {s['total_trajectories']}  "
                         f"Tasks: {s['total_tasks']}  "
                         f"Meta updates: {s['meta_update_count']}")
            mp = s["meta_params"]
            lines.append(f"      lr_scale={mp['lr_scale']:.4f}  "
                         f"baseline={mp['baseline_performance']:.4f}  "
                         f"exploration={mp['exploration_weight']:.4f}")

        lines.append(f"\n{'=' * 60}\n")
        return "\n".join(lines)

    async def _show_memory(self) -> str:
        """Show recent memory episodes."""
        await self._ensure_memory()
        episodes = await self.memory.backend.get_all_episodes(limit=10)
        if not episodes:
            return "  No episodes stored yet. Run some analyses first."

        lines = [f"\n{'=' * 60}", f"  Recent Memory Episodes ({len(episodes)})", f"{'=' * 60}"]
        for ep in episodes:
            content = str(ep.content)[:80]
            lines.append(f"\n  [{ep.id[:8]}] {ep.source} @ {ep.timestamp[:19]}")
            lines.append(f"    {content}...")
        lines.append(f"\n{'=' * 60}\n")
        return "\n".join(lines)

    def _help_text(self) -> str:
        return f"""
ATHENA Interactive CLI
──────────────────────
Commands:
  <query>          Ask about any stock (e.g. "analyze AAPL", "should I buy TSLA?")
  symbols          List available symbols
  set <symbol>     Set default symbol (e.g. "set GOOGL")
  verbose on/off   Toggle agent thought process display
  stats            Show memory + learning statistics
  memory           Show recent memory episodes
  help / ?         Show this help
  quit / exit      Exit the CLI

Flags:
  -v / --verbose   Start with verbose mode on (shows all agent internals)

Each query:
  1. Runs 5 agents (analyst, risk, strategy, execution, coordinator)
  2. Stores decisions in AgeMem (Graphiti/Neo4j temporal knowledge graph)
  3. Retrieves past context from memory to inform decisions
  4. Runs nested learning inner-loop (per-agent adaptation)
  5. Runs outer-loop meta-update every 5 queries
  6. Consolidates knowledge every 20 queries

Examples:
  > analyze AAPL
  > what's the risk on TSLA?
  > should I buy MSFT?
  > stats
  > memory
"""


async def main_loop(shell: AthenaShell) -> None:
    """Main interactive loop."""
    print(f"\n{'=' * 60}")
    print(f"  ATHENA Financial Advisory System")
    print(f"  Multi-Agent Investment Recommendations")
    print(f"{'=' * 60}")
    print(f"  Default symbol: {shell.default_symbol}")
    print(f"  Available: {', '.join(shell.symbols)}")
    print(f"  Verbose: {'ON' if shell.verbose else 'OFF'} (toggle with 'verbose on/off' or -v flag)")
    print(f"  Type 'help' for commands, 'quit' to exit.")
    print(f"{'=' * 60}\n")

    while True:
        try:
            query = input("athena> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        response = await shell.handle_query(query)
        print(response)


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)

    shell = AthenaShell(default_symbol=args.symbol, verbose=args.verbose)
    asyncio.run(main_loop(shell))


if __name__ == "__main__":
    main()
