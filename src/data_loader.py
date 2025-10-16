import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data.scripts.get_data import (
    fetch_all_historical_data,
    get_available_symbols,
    load_local_data,
)
from src.config import config


class DataLoader:
    """
    Unified data loader that handles both local and live data

    Responsibilities:
    1. Abstract away data source (local vs live)
    2. Automatically fetch if local data does not exist
    3. Provide consistent interface across all services
    4. Handle multiple symbols efficiently
    """

    def __init__(self, use_local=None):
        """
        Initialize the DataLoader.

        Args:
            use_local (bool, optional): Override config.USE_LOCAL_DATA if set.
        """
        self.use_local = use_local if use_local is not None else config.USE_LOCAL_DATA
        print(f"DataLoader initialized in {config.data_mode} mode.")

    def load_all_data(self, force_refresh=False):
        """
        Load data for ALL available stocks from consolidated file

        Args:
            force_refresh (bool, optional): If True, fetch data from database even if local data exists.

        Returns:
            DataFrame with all stock data (consolidated)
        """
        print(f"\n{'='*80}")
        print("Loading all available stock data...")
        print(f"{'='*80}\n")

        if self.use_local and not force_refresh:
            # Try to load from consolidated local file
            df = load_local_data(format="parquet")

            if df is None:
                print("Local data not found. Fetching from database...")
                fetch_all_historical_data(save_local=True)
                df = load_local_data(format="parquet")
            else:
                print(
                    f"✅ Loaded {len(df):,} records for {df['symbol'].nunique()} stocks"
                )

            return df

        else:
            # Fetch from database
            print("Fetching data from database...")
            fetch_all_historical_data(save_local=self.use_local)
            return load_local_data(format="parquet")

    def split_train_test(self, df, test_size=0.2, method="time_based"):
        """
        Split data into train and test sets for time series.
        Works with consolidated DataFrame containing multiple symbols.

        Args:
            df: DataFrame with stock data (can contain multiple symbols)
            test_size: Proportion of data to use for testing. Defaults to 0.2.
            method: Splitting method. 'time_based' (Recommended) or 'last_n_days'. Defaults to 'time_based'.

        Returns:
            Tuple of (train_df, test_df)
        """
        if df is None or df.empty:
            return None, None

        # Split per symbol to maintain temporal integrity
        train_dfs = []
        test_dfs = []

        symbols = df["symbol"].unique()

        for symbol in symbols:
            symbol_df = df[df["symbol"] == symbol].copy()
            symbol_df = symbol_df.sort_values("timestamp").reset_index(drop=True)

            if method == "time_based":
                split_idx = int(len(symbol_df) * (1 - test_size))
                train_part = symbol_df.iloc[:split_idx]
                test_part = symbol_df.iloc[split_idx:]

            elif method == "last_n_days":
                # Use last N days for testing
                days = int(test_size * 100)  # Converts 0.2 to 20 days
                cutoff_date = symbol_df["timestamp"].max() - pd.Timedelta(days=days)

                train_part = symbol_df[symbol_df["timestamp"] < cutoff_date]
                test_part = symbol_df[symbol_df["timestamp"] >= cutoff_date]
            else:
                train_part = symbol_df
                test_part = pd.DataFrame()

            train_dfs.append(train_part)
            test_dfs.append(test_part)

        # Combine all symbols
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)

        print(f"📊 Split completed:")
        print(
            f"   Train: {len(train_df):,} records ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})"
        )
        print(
            f"   Test:  {len(test_df):,} records ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})"
        )

        return train_df, test_df

    def prepare_train_test_data(
        self, test_size=0.2, method="time_based", force_refresh=False
    ):
        """
        Load all data and split into train/test sets.

        Args:
            test_size (float, optional): Proportion of data to use for testing. Defaults to 0.2.
            method (str, optional): Splitting method. 'time_based' (Recommended) or 'last_n_days'. Defaults to 'time_based'.
            force_refresh (bool, optional): If True, fetch data from database even if local data exists. Defaults to False.

        Returns:
            Tuple of (train_df, test_df) - DataFrames with all stocks
        """
        all_data = self.load_all_data(force_refresh=force_refresh)

        if all_data is None or all_data.empty:
            print("❌ No data loaded")
            return None, None

        print(f"\n{'='*80}")
        print("✂️  Splitting data into train and test sets...")
        print(f"{'='*80}\n")

        train_df, test_df = self.split_train_test(
            all_data, test_size=test_size, method=method
        )

        print(f"\n{'='*80}")
        print(f"✅ Data preparation complete!")
        print(f"   Unique symbols: {all_data['symbol'].nunique()}")
        if train_df is not None and test_df is not None:
            print(f"   Train samples: {len(train_df):,}")
            print(f"   Test samples: {len(test_df):,}")
        print(f"{'='*80}\n")

        return train_df, test_df

    def refresh_data(self):
        """
        Force refresh data from database
        Useful for updating local cache

        Returns:
            Dictionary of DataFrames
        """
        print("Force refreshing data from database...")
        return fetch_all_historical_data(
            save_local=True
        )  # Fixed: call imported function

    def get_data_summary(self):
        """
        Get summary of available data

        Returns:
            Dictionary with data statistics
        """
        all_data = self.load_all_data()

        if all_data is None or all_data.empty:
            print("❌ No data loaded")
            return {
                "total_stocks": 0,
                "total_records": 0,
                "data_source": config.data_mode,
            }

        symbols = all_data["symbol"].unique()
        min_date = all_data["timestamp"].min()
        max_date = all_data["timestamp"].max()
        days_span = (max_date - min_date).days

        return {
            "total_stocks": len(symbols),
            "stock_symbols": list(symbols),
            "total_records": len(all_data),
            "data_source": config.data_mode,
            "date_range": {"start": min_date, "end": max_date, "days_span": days_span},
        }


if __name__ == "__main__":

    # Initialize DataLoader
    print("Testing DataLoader...")
    print("=" * 80)

    loader = DataLoader()

    # Data summary
    print("\nData Summary:")
    summary = loader.get_data_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Prepare train/test data
    print("\nPreparing Train/Test Data...")
    train_data, test_data = loader.prepare_train_test_data(
        test_size=0.2, method="time_based"
    )
