"""Script to run the data processing pipeline with configuration."""

import yaml
from pathlib import Path
from src.data.processors.data_processor import DataProcessor


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary with configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_symbols(symbols_path: str = "symbols.yaml") -> list:
    """Load symbols from symbols config file.

    Args:
        symbols_path: Path to symbols configuration file

    Returns:
        List of stock symbols
    """
    with open(symbols_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['symbols']


def main():
    """Run the data processing pipeline."""
    # Load configuration
    config = load_config()
    symbols = load_symbols()  # Load from symbols.yaml

    # Extract configuration sections
    data_config = config['data_processing']

    print("=" * 80)
    print("Finance Assistant - Data Processing Pipeline")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Symbols: {len(symbols)} stocks")
    print(f"  Sequence Length: {data_config['sequence_length']} days")
    print(f"  Prediction Horizon: {data_config['prediction_horizon']} day(s)")
    print(f"  Train/Val/Test Split: {data_config['train_ratio']:.0%}/{data_config['val_ratio']:.0%}/{data_config['test_ratio']:.0%}")
    print(f"  Scaler Selection: {'Enabled' if config['scaler_selection']['enabled'] else 'Disabled'}")
    print(f"  Stationarity Tests: {'Enabled' if config['scaler_selection']['test_stationarity'] else 'Disabled'}")
    print()

    # Initialize processor
    processor = DataProcessor(
        raw_data_dir=data_config['raw_data_dir'],
        processed_data_dir=data_config['processed_data_dir'],
        sequence_length=data_config['sequence_length'],
        prediction_horizon=data_config['prediction_horizon'],
    )

    # Run the pipeline
    file_paths = processor.process_all(
        symbols=symbols,
        train_ratio=data_config['train_ratio'],
        val_ratio=data_config['val_ratio'],
        test_ratio=data_config['test_ratio'],
    )

    print("\n" + "=" * 80)
    print("Processing Complete!")
    print("=" * 80)
    print("\nOutput Files:")
    for key, path in file_paths.items():
        print(f"  {key:20s}: {path}")

    print("\nNext Steps:")
    print("  1. Review the scaler_analysis_report_*.json for feature analysis")
    print("  2. Check for stationarity warnings")
    print("  3. Consider adding differenced features if needed")
    print("  4. Proceed to model training")
    print()


if __name__ == "__main__":
    main()
