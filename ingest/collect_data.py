"""Data collection script - fetch and save stock data."""
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from src.data.collectors.yfinance_collector import YFinanceCollector
from src.data.collectors.alpha_vantage_collector import AlphaVantageCollector
from src.data.collectors.finnhub_collector import FinnhubCollector
from src.data.collectors.fred_collector import FREDCollector
import json
from pathlib import Path
import time
import yaml

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def collect_and_save_data(
    symbols: list[str],
    start_date: str,
    end_date: str = None,
    use_yfinance: bool = True,
    use_alpha_vantage: bool = False,
    use_finnhub: bool = False,
    use_fred: bool = False
):
    """
    Collect and save stock data from various sources.

    Args:
        symbols: List of stock symbols to collect
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (default: today)
        use_yfinance: Whether to use YFinance collector
        use_alpha_vantage: Whether to use Alpha Vantage collector
        use_finnhub: Whether to use Finnhub collector (news + fundamentals)
        use_fred: Whether to use FRED collector (economic indicators)
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    # Canonical OHLCV output directory shared with trading/market_data.py FILE mode
    canonical_dir = Path(__file__).resolve().parent.parent / "data" / "market"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "symbols": symbols,
        "start_date": start_date,
        "end_date": end_date,
        "results": []
    }

    # YFinance Collection
    if use_yfinance:
        logger.info("=" * 60)
        logger.info("Collecting data from YFinance")
        logger.info("=" * 60)

        yf_collector = YFinanceCollector(data_dir="data/raw/yfinance")

        for symbol in symbols:
            try:
                logger.info(f"\nProcessing {symbol}...")

                # Fetch historical data
                df = yf_collector.fetch_historical_data(symbol, start_date, end_date)

                if not df.is_empty():
                    # Validate data
                    is_valid = yf_collector.validate_data(df)

                    if is_valid:
                        # Save to parquet (also writes canonical copy)
                        filepath = yf_collector.save_to_parquet(df, symbol, "ohlcv", canonical_dir=canonical_dir)
                        logger.info(f"✓ Saved OHLCV data: {filepath} ({len(df)} rows)")

                        # Fetch and save fundamentals
                        fundamentals = yf_collector.fetch_fundamentals(symbol)

                        if fundamentals and "error" not in fundamentals:
                            # Save fundamentals as JSON
                            fund_path = Path("data/raw/yfinance") / f"{symbol}_fundamentals_{datetime.now().strftime('%Y%m%d')}.json"
                            fund_path.parent.mkdir(parents=True, exist_ok=True)

                            with open(fund_path, 'w') as f:
                                json.dump(fundamentals, f, indent=2)

                            logger.info(f"✓ Saved fundamentals: {fund_path}")

                        results["results"].append({
                            "symbol": symbol,
                            "source": "yfinance",
                            "status": "success",
                            "rows": len(df),
                            "ohlcv_file": str(filepath),
                            "fundamentals_file": str(fund_path) if fundamentals else None
                        })
                    else:
                        logger.warning(f"✗ Data validation failed for {symbol}")
                        results["results"].append({
                            "symbol": symbol,
                            "source": "yfinance",
                            "status": "validation_failed"
                        })
                else:
                    logger.warning(f"✗ No data returned for {symbol}")
                    results["results"].append({
                        "symbol": symbol,
                        "source": "yfinance",
                        "status": "no_data"
                    })

            except Exception as e:
                logger.error(f"✗ Error processing {symbol}: {e}")
                results["results"].append({
                    "symbol": symbol,
                    "source": "yfinance",
                    "status": "error",
                    "error": str(e)
                })

    # Alpha Vantage Collection
    if use_alpha_vantage:
        logger.info("\n" + "=" * 60)
        logger.info("Collecting data from Alpha Vantage")
        logger.info("=" * 60)

        av_collector = AlphaVantageCollector(data_dir="data/raw/alpha_vantage")

        if not av_collector.api_key:
            logger.warning("⚠ Alpha Vantage API key not configured. Skipping.")
            return results

        for symbol in symbols:
            try:
                logger.info(f"\nProcessing {symbol}...")

                # Fetch historical data
                df = av_collector.fetch_historical_data(symbol, start_date, end_date)

                if not df.is_empty():
                    # Validate data
                    is_valid = av_collector.validate_data(df)

                    if is_valid:
                        # Save to parquet (also writes canonical copy)
                        filepath = av_collector.save_to_parquet(df, symbol, "ohlcv", canonical_dir=canonical_dir)
                        logger.info(f"✓ Saved OHLCV data: {filepath} ({len(df)} rows)")

                        # Fetch and save fundamentals
                        fundamentals = av_collector.fetch_fundamentals(symbol)

                        if fundamentals and "error" not in fundamentals:
                            # Save fundamentals as JSON
                            fund_path = Path("data/raw/alpha_vantage") / f"{symbol}_fundamentals_{datetime.now().strftime('%Y%m%d')}.json"
                            fund_path.parent.mkdir(parents=True, exist_ok=True)

                            with open(fund_path, 'w') as f:
                                json.dump(fundamentals, f, indent=2)

                            logger.info(f"✓ Saved fundamentals: {fund_path}")

                        results["results"].append({
                            "symbol": symbol,
                            "source": "alpha_vantage",
                            "status": "success",
                            "rows": len(df),
                            "ohlcv_file": str(filepath),
                            "fundamentals_file": str(fund_path) if fundamentals else None
                        })
                    else:
                        logger.warning(f"✗ Data validation failed for {symbol}")
                        results["results"].append({
                            "symbol": symbol,
                            "source": "alpha_vantage",
                            "status": "validation_failed"
                        })
                else:
                    logger.warning(f"✗ No data returned for {symbol}")
                    results["results"].append({
                        "symbol": symbol,
                        "source": "alpha_vantage",
                        "status": "no_data"
                    })

            except Exception as e:
                logger.error(f"✗ Error processing {symbol}: {e}")
                results["results"].append({
                    "symbol": symbol,
                    "source": "alpha_vantage",
                    "status": "error",
                    "error": str(e)
                })

            # Rate limiting: wait between API calls
            time.sleep(15)  # Alpha Vantage free tier: 5 calls/minute

    # Finnhub Collection (News + Fundamentals only)
    if use_finnhub:
        logger.info("\n" + "=" * 60)
        logger.info("Collecting data from Finnhub")
        logger.info("=" * 60)

        fh_collector = FinnhubCollector(data_dir="data/raw/finnhub")

        if not fh_collector.client:
            logger.warning("⚠ Finnhub API key not configured. Skipping.")
        else:
            for symbol in symbols:
                try:
                    logger.info(f"\nProcessing {symbol}...")

                    # Fetch fundamentals
                    fundamentals = fh_collector.fetch_fundamentals(symbol)

                    fund_path = None
                    if fundamentals and "error" not in fundamentals:
                        # Save fundamentals as JSON
                        fund_path = Path("data/raw/finnhub") / f"{symbol}_fundamentals_{datetime.now().strftime('%Y%m%d')}.json"
                        fund_path.parent.mkdir(parents=True, exist_ok=True)

                        with open(fund_path, 'w') as f:
                            json.dump(fundamentals, f, indent=2)

                        logger.info(f"✓ Saved fundamentals: {fund_path}")

                    # Fetch company news (last 7 days)
                    news_from = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                    news_to = datetime.now().strftime("%Y-%m-%d")
                    news = fh_collector.fetch_company_news(symbol, news_from, news_to)

                    news_path = None
                    if news:
                        # Save news as JSON
                        news_path = Path("data/raw/finnhub") / f"{symbol}_news_{datetime.now().strftime('%Y%m%d')}.json"
                        news_path.parent.mkdir(parents=True, exist_ok=True)

                        # Calculate sentiment
                        sentiment = fh_collector.get_sentiment_score(news)

                        news_data = {
                            "symbol": symbol,
                            "timestamp": datetime.now().isoformat(),
                            "date_range": {"from": news_from, "to": news_to},
                            "article_count": len(news),
                            "avg_sentiment": sentiment,
                            "articles": news
                        }

                        with open(news_path, 'w') as f:
                            json.dump(news_data, f, indent=2)

                        logger.info(f"✓ Saved news: {news_path} ({len(news)} articles, sentiment: {sentiment:.2f})")

                    results["results"].append({
                        "symbol": symbol,
                        "source": "finnhub",
                        "status": "success",
                        "fundamentals_file": str(fund_path) if fund_path else None,
                        "news_file": str(news_path) if news_path else None,
                        "news_count": len(news) if news else 0,
                        "sentiment": sentiment if news else None
                    })

                except Exception as e:
                    logger.error(f"✗ Error processing {symbol}: {e}")
                    results["results"].append({
                        "symbol": symbol,
                        "source": "finnhub",
                        "status": "error",
                        "error": str(e)
                    })

                # Rate limiting: Finnhub free tier allows 60 calls/minute
                time.sleep(1)

    # FRED Collection (Economic Indicators)
    if use_fred:
        logger.info("\n" + "=" * 60)
        logger.info("Collecting economic data from FRED")
        logger.info("=" * 60)

        fred_collector = FREDCollector(data_dir="data/raw/fred")

        if not fred_collector.client:
            logger.warning("⚠ FRED API key not configured. Skipping.")
        else:
            try:
                logger.info(f"\nFetching economic indicators...")

                # Fetch all default economic indicators
                df = fred_collector.fetch_historical_data(start_date=start_date, end_date=end_date)

                if not df.is_empty():
                    # Validate data
                    is_valid = fred_collector.validate_data(df)

                    if is_valid:
                        # Save to parquet
                        filepath = fred_collector.save_to_parquet(df, "economic_indicators", "fred")
                        logger.info(f"✓ Saved economic data: {filepath} ({len(df)} rows)")

                        # Fetch and save metadata
                        metadata = fred_collector.fetch_fundamentals()

                        if metadata and "error" not in metadata:
                            # Save metadata as JSON
                            meta_path = Path("data/raw/fred") / f"metadata_{datetime.now().strftime('%Y%m%d')}.json"
                            meta_path.parent.mkdir(parents=True, exist_ok=True)

                            with open(meta_path, 'w') as f:
                                json.dump(metadata, f, indent=2)

                            logger.info(f"✓ Saved metadata: {meta_path}")

                        # Get list of indicators for results
                        indicators = [col for col in df.columns if col != 'date']

                        results["results"].append({
                            "source": "fred",
                            "status": "success",
                            "rows": len(df),
                            "indicators": indicators,
                            "data_file": str(filepath),
                            "metadata_file": str(meta_path) if metadata else None
                        })
                    else:
                        logger.warning(f"✗ Data validation failed for FRED")
                        results["results"].append({
                            "source": "fred",
                            "status": "validation_failed"
                        })
                else:
                    logger.warning(f"✗ No data returned from FRED")
                    results["results"].append({
                        "source": "fred",
                        "status": "no_data"
                    })

            except Exception as e:
                logger.error(f"✗ Error processing FRED data: {e}")
                results["results"].append({
                    "source": "fred",
                    "status": "error",
                    "error": str(e)
                })

    # Save collection summary
    summary_path = Path("data/raw") / f"collection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info(f"Collection complete! Summary saved to: {summary_path}")
    logger.info("=" * 60)

    # Print summary
    success_count = sum(1 for r in results["results"] if r.get("status") == "success")
    total_count = len(results["results"])

    logger.info(f"\nSummary: {success_count}/{total_count} successful")

    return results


if __name__ == "__main__":
    # Load configuration from YAML file (relative to this script's directory)
    _script_dir = Path(__file__).resolve().parent
    config_path = _script_dir / "symbols.yaml"

    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    symbols = config['symbols']
    settings = config['data_collection']

    logger.info("Starting comprehensive data collection...")
    logger.info(f"Symbols: {len(symbols)} stocks")
    logger.info(f"Symbols list: {', '.join(symbols)}")
    logger.info(f"Date range: {settings['start_date']} to today")

    results = collect_and_save_data(
        symbols=symbols,
        start_date=settings['start_date'],
        use_yfinance=settings['use_yfinance'],
        use_alpha_vantage=settings['use_alpha_vantage'],
        use_finnhub=settings['use_finnhub'],
        use_fred=settings['use_fred']
    )

    logger.info("\nData collection complete!")
