"""
ATHENA Main Entry Point
=======================
Production script to launch the ATHENA multi-agent financial advisory system.

Usage:
    python main.py --mode dry-run
    python main.py --mode paper-trade --config config.yaml
    python main.py --mode backtest --config config.yaml
    python main.py --help
"""

import argparse
import asyncio
import signal
import sys
from typing import Dict, Optional

from core.utils import setup_logging
from core.config import AthenaConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="athena",
        description="ATHENA Multi-Agent Financial Advisory System",
    )
    parser.add_argument(
        "--mode",
        choices=["dry-run", "paper-trade", "backtest"],
        default="dry-run",
        help="Operating mode (default: dry-run)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML or JSON config file",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path",
    )
    return parser.parse_args()


def load_config(config_path: Optional[str]) -> AthenaConfig:
    """Load configuration from file or use defaults."""
    if config_path is None:
        return AthenaConfig()
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        return AthenaConfig.from_yaml(config_path)
    if config_path.endswith(".json"):
        return AthenaConfig.from_json(config_path)
    raise ValueError(f"Unsupported config file format: {config_path}")


async def run(mode: str, config: AthenaConfig, logger) -> None:
    """
    Instantiate agents, memory, and communication layers, then run the main loop.

    The coordinator orchestrates the specialist agents (MarketAnalyst, RiskManager,
    StrategyAgent, ExecutionAgent) each round. ExecutionAgent output is treated as
    investment recommendations — no live order submission.
    """
    from agents.coordinator import CoordinatorAgent
    from agents.market_analyst import MarketAnalystAgent
    from agents.risk_manager import RiskManagerAgent
    from agents.strategy_agent import StrategyAgent
    from agents.execution_agent import ExecutionAgent
    from core.base_agent import AgentContext
    from trading.market_data import MarketDataFeed, MarketDataMode
    from dataclasses import asdict

    logger.info("Initialising ATHENA in %s mode", mode)

    # Instantiate agents (memory/router are optional; omitted in dry-run)
    market_analyst = MarketAnalystAgent()
    risk_manager = RiskManagerAgent()
    strategy_agent = StrategyAgent()
    execution_agent = ExecutionAgent()

    coordinator = CoordinatorAgent()
    coordinator.register_agent("market_analyst", market_analyst)
    coordinator.register_agent("risk_manager", risk_manager)
    coordinator.register_agent("strategy_agent", strategy_agent)
    coordinator.register_agent("execution_agent", execution_agent)

    if mode == "dry-run":
        # Single think-act cycle with empty context to confirm wiring
        context = AgentContext(task="dry-run health check")
        thought = await coordinator.think(context)
        action = await coordinator.act(thought)
        logger.info("Dry-run complete. Action type: %s", action.action_type)
        return

    if mode == "paper-trade":
        feed = MarketDataFeed(mode=MarketDataMode.MOCK)
        symbols = config.trading.markets if config.trading.markets != ["stocks"] else MarketDataFeed.MOCK_SYMBOLS
        logger.info("Paper-trade loop -- press Ctrl+C to stop")
        iteration = 0
        try:
            while True:
                iteration += 1
                for symbol in symbols:
                    bar = await feed.get_realtime_data(symbol)
                    if bar is None:
                        continue
                    bar_dict = asdict(bar)
                    context = AgentContext(
                        task=f"market cycle {iteration}: {symbol}",
                        metadata={
                            "market_data": {
                                "symbol": symbol,
                                "prices": [bar_dict["close"]],
                                "bar": bar_dict,
                            },
                            "symbol": symbol,
                        },
                    )
                    thought = await coordinator.think(context)
                    await coordinator.act(thought)
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            logger.info("Paper-trade loop cancelled after %d iterations", iteration)

    elif mode == "backtest":
        feed = MarketDataFeed(mode=MarketDataMode.MOCK)
        symbols = config.trading.markets if config.trading.markets != ["stocks"] else MarketDataFeed.MOCK_SYMBOLS
        backtest_days = 30  # Default backtest window
        logger.info("Backtest mode: replaying %d days of historical data", backtest_days)

        # Pre-fetch historical data for all symbols
        historical: Dict[str, list] = {}
        for symbol in symbols:
            bars = await feed.get_historical_data(symbol, days=backtest_days)
            historical[symbol] = [asdict(bar) for bar in bars]

        # Replay bar-by-bar
        num_bars = max((len(bars) for bars in historical.values()), default=0)
        for i in range(num_bars):
            for symbol in symbols:
                bars = historical[symbol]
                if i >= len(bars):
                    continue
                bar_dict = bars[i]
                window = [b["close"] for b in bars[: i + 1]]
                context = AgentContext(
                    task=f"backtest bar {i + 1}/{num_bars}: {symbol}",
                    metadata={
                        "market_data": {
                            "symbol": symbol,
                            "prices": window,
                            "bar": bar_dict,
                        },
                        "symbol": symbol,
                        "bar_index": i,
                    },
                )
                thought = await coordinator.think(context)
                await coordinator.act(thought)

        logger.info("Backtest complete: replayed %d bars", num_bars)


def main() -> None:
    args = parse_args()
    logger = setup_logging(level=args.log_level, log_file=args.log_file)
    config = load_config(args.config)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Graceful shutdown on SIGINT / SIGTERM
    task: Optional[asyncio.Task] = None

    def _shutdown(sig, frame):
        logger.info("Shutdown signal received (%s)", sig)
        if task and not task.done():
            task.cancel()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        task = loop.create_task(run(args.mode, config, logger))
        loop.run_until_complete(task)
    except Exception as exc:
        logger.error("ATHENA exited with error: %s", exc)
        sys.exit(1)
    finally:
        loop.close()
        logger.info("ATHENA shutdown complete")


if __name__ == "__main__":
    main()
