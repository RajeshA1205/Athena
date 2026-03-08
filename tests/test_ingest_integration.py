"""
Integration tests: Ingest pipeline -> MarketDataFeed FILE mode.

Verifies that canonical parquet files written by the ingest pipeline
can be read by MarketDataFeed in FILE mode, producing valid OHLCV bars.
All tests are offline (no network calls).
"""
import pytest
from pathlib import Path

pl = pytest.importorskip("polars")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_parquet(tmp_path: Path) -> Path:
    """Create a 30-row synthetic AAPL parquet in canonical schema.

    Returns the fake project root (tmp_path) so callers can monkeypatch
    _PROJECT_ROOT to redirect MarketDataFeed file reads.
    """
    market_dir = tmp_path / "data" / "market"
    market_dir.mkdir(parents=True)

    rows = 30
    data = {
        "symbol":    ["AAPL"] * rows,
        "timestamp": [f"2025-01-{i + 1:02d}T21:00:00+00:00" for i in range(rows)],
        "open":      [float(150 + i) for i in range(rows)],
        "high":      [float(152 + i) for i in range(rows)],
        "low":       [float(149 + i) for i in range(rows)],
        "close":     [float(151 + i) for i in range(rows)],
        "volume":    [1_000_000.0] * rows,
        "interval":  ["1d"] * rows,
    }
    schema = {
        "symbol": pl.Utf8, "timestamp": pl.Utf8,
        "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
        "close": pl.Float64, "volume": pl.Float64, "interval": pl.Utf8,
    }
    df = pl.DataFrame(data, schema=schema)
    df.write_parquet(market_dir / "AAPL_ohlcv.parquet")
    return tmp_path  # fake project root


@pytest.fixture
def file_mode_feed(synthetic_parquet: Path, monkeypatch):
    """MarketDataFeed in FILE mode with _PROJECT_ROOT redirected to tmp_path."""
    import trading.market_data as mdm
    monkeypatch.setattr(mdm, "_PROJECT_ROOT", synthetic_parquet)
    return mdm.MarketDataFeed(mode=mdm.MarketDataMode.FILE)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIngestIntegration:

    def test_parquet_schema(self, synthetic_parquet: Path):
        """Parquet file has all 8 canonical columns with correct dtypes."""
        path = synthetic_parquet / "data" / "market" / "AAPL_ohlcv.parquet"
        assert path.exists()

        df = pl.read_parquet(path)

        expected_columns = ["symbol", "timestamp", "open", "high", "low", "close", "volume", "interval"]
        assert list(df.columns) == expected_columns

        assert df["symbol"].dtype == pl.Utf8
        assert df["timestamp"].dtype == pl.Utf8
        assert df["open"].dtype == pl.Float64
        assert df["high"].dtype == pl.Float64
        assert df["low"].dtype == pl.Float64
        assert df["close"].dtype == pl.Float64
        assert df["volume"].dtype == pl.Float64
        assert df["interval"].dtype == pl.Utf8
        assert len(df) == 30

    @pytest.mark.asyncio
    async def test_market_data_feed_file_mode(self, file_mode_feed):
        """FILE mode returns the last row of the parquet as an OHLCV bar."""
        bar = await file_mode_feed.get_realtime_data("AAPL")

        assert bar is not None
        assert bar.symbol == "AAPL"
        assert isinstance(bar.open, float) and bar.open > 0
        assert isinstance(bar.high, float) and bar.high > 0
        assert isinstance(bar.low, float) and bar.low > 0
        assert isinstance(bar.close, float) and bar.close > 0
        assert isinstance(bar.volume, float) and bar.volume > 0
        # Last row: close = 151 + 29 = 180.0
        assert bar.close == pytest.approx(180.0)

    @pytest.mark.asyncio
    async def test_market_data_feed_file_mode_fallback(self, file_mode_feed):
        """Missing parquet file falls back to MOCK — no exception raised."""
        bar = await file_mode_feed.get_realtime_data("MISSING")

        assert bar is not None  # MOCK fallback
        assert isinstance(bar.close, float) and bar.close > 0

    @pytest.mark.asyncio
    async def test_get_historical_data_days_limit(self, file_mode_feed):
        """get_historical_data respects the days limit from a 30-row file."""
        bars = await file_mode_feed.get_historical_data("AAPL", days=7)

        assert isinstance(bars, list)
        assert len(bars) == 7

        from trading.market_data import OHLCV
        for bar in bars:
            assert isinstance(bar, OHLCV)
            assert bar.symbol == "AAPL"

        # Rows are in ascending timestamp order
        timestamps = [b.timestamp for b in bars]
        assert timestamps == sorted(timestamps)

        # Last bar is the last row of the file
        assert bars[-1].close == pytest.approx(180.0)


class TestCastToCanonical:
    """Direct tests for ingest/src/data/schema.py cast_to_canonical()."""

    @pytest.fixture(autouse=True)
    def _add_ingest_to_path(self):
        """Ensure ingest/ is on sys.path so src.data.schema is importable."""
        import sys, os
        ingest_dir = os.path.join(os.path.dirname(__file__), "..", "ingest")
        if ingest_dir not in sys.path:
            sys.path.insert(0, ingest_dir)

    def _raw_df(self, **overrides):
        """Return a minimal valid raw collector DataFrame (with 'date' column)."""
        base = {
            "symbol":  ["AAPL"],
            "date":    ["2025-01-01"],
            "open":    [150.0],
            "high":    [152.0],
            "low":     [149.0],
            "close":   [151.0],
            "volume":  [1_000_000.0],
        }
        base.update(overrides)
        return pl.DataFrame(base)

    def test_date_column_renamed_to_timestamp(self):
        """cast_to_canonical renames 'date' -> 'timestamp'."""
        from src.data.schema import cast_to_canonical
        result = cast_to_canonical(self._raw_df())
        assert "timestamp" in result.columns
        assert "date" not in result.columns

    def test_interval_added_when_absent(self):
        """cast_to_canonical adds the interval literal column."""
        from src.data.schema import cast_to_canonical
        result = cast_to_canonical(self._raw_df(), interval="1h")
        assert result["interval"][0] == "1h"

    def test_missing_required_column_raises(self):
        """cast_to_canonical raises ValueError when required columns are absent."""
        from src.data.schema import cast_to_canonical
        df = pl.DataFrame({
            "symbol":    ["AAPL"],
            "timestamp": ["2025-01-01T21:00:00+00:00"],
            # 'close' intentionally omitted
            "open": [150.0], "high": [152.0], "low": [149.0], "volume": [1e6],
        })
        with pytest.raises(ValueError, match="missing required"):
            cast_to_canonical(df)

    def test_validate_canonical_passes_on_valid_df(self):
        """validate_canonical returns True for a well-formed DataFrame."""
        from src.data.schema import cast_to_canonical, validate_canonical
        canonical = cast_to_canonical(self._raw_df())
        assert validate_canonical(canonical) is True
