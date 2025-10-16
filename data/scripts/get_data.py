import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2
import pytz

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import config


def get_available_symbols():
    """
    Get all unique stock symbols available in the database

    Returns
        List of stock symbols
    """

    print("Fetching available stock symbols from database...")

    try:
        conn = psycopg2.connect(config.database_url)

        query = """
        SELECT DISTINCT symbol 
        FROM stock_prices
        ORDER BY symbol;
        """

        cursor = conn.cursor()
        cursor.execute(query)
        symbols = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()

        print(f"Found {len(symbols)} unique stock symbols: {', '.join(symbols)}")
        return symbols

    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return []


def get_date_range():
    """
    Get the min and max dates available in the stock_prices table

    Returns:
        Tuple of (min_date, max_date, total_records)
    """
    print("Fetching date range from database...")

    try:
        conn = psycopg2.connect(config.database_url)

        query = """
        SELECT 
            MIN(timestamp) AS min_date,
            MAX(timestamp) AS max_date,
            COUNT(*) AS total_records
        FROM stock_prices;
        """

        cursor = conn.cursor()
        cursor.execute(query)
        result = [row for row in cursor.fetchall()][0]
        cursor.close()
        conn.close()

        min_date = pd.to_datetime(result[0]).tz_localize(config.DB_TIMEZONE)
        max_date = pd.to_datetime(result[1]).tz_localize(config.DB_TIMEZONE)
        total_records = result[2]

        print(f"Date range: {min_date} to {max_date}, Total records: {total_records}")
        return min_date, max_date, total_records

    except Exception as e:
        print(f"Error fetching date range: {e}")
        return None, None, 0


def fetch_all_historical_data(save_local=False, fetch_from_min_date=True):
    """
    Fetch all historical stock data from the database.

    Args:
        save_local (bool, optional): Whether to save the data locally as a CSV file. Defaults to False.
        fetch_from_min_date (bool, optional): Whether to fetch data from the minimum date available. Defaults to True.

    Returns:
        Dictionary of DataFrames with stock data for each symbol
    """

    print(f"\n{'='*80}")
    print("Connecting to database: {config.database_url}")
    print(f"{'='*80}\n")

    try:
        conn = psycopg2.connect(config.database_url)

        print("Database connection established.")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

    # Get available symbols
    symbols = get_available_symbols()
    if not symbols:
        print("No symbols found. Exiting.")
        conn.close()
        return None

    # Get date range
    min_date, max_date, total_records = get_date_range()
    if min_date is None:
        print("Could not fetch date range. Exiting.")
        conn.close()
        return None

    # Use min_date from database
    start_time = min_date
    end_time = max_date
    days_span = (end_time - start_time).days

    print(f"Fetching data from {start_time} to {end_time} ({days_span} days)...\n")
    all_data = {}
    all_records = []
    total_records_fetched = 0

    for idx, symbol in enumerate(symbols, 1):
        print(f"[{idx}/{len(symbols)}] Fetching data for {symbol}...")

        query = """
            SELECT 
                symbol,
                open_price,
                high,
                low,
                close_price,
                volume,
                market_cap,
                timestamp,
                date_only
            FROM stock_prices
            WHERE symbol = %s
            AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp ASC;
            """

        try:
            df = pd.read_sql_query(query, conn, params=(symbol, start_time, end_time))

            if df.empty:
                print(f"  No data found for {symbol}. Skipping.")
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(
                config.DB_TIMEZONE
            )

            # Adding helpful columns
            df["timestamp_et"] = df["timestamp"].dt.tz_convert("America/New_York")
            df["hour"] = df["timestamp_et"].dt.hour
            df["day_of_week"] = df["timestamp_et"].dt.dayofweek
            df["is_trading_hours"] = (
                (df["hour"] >= 9) & (df["hour"] < 16) & (df["day_of_week"] < 5)
            )

            # Summary
            record_count = len(df)
            date_range = (df["timestamp"].min(), df["timestamp"].max())
            print(
                f"  Fetched {record_count} records from {date_range[0]} to {date_range[1]}"
            )
            all_data[symbol] = df
            total_records_fetched += record_count
            all_records.append(df)

        except Exception as e:
            print(f"  Error fetching data for {symbol}: {e}")
            continue

    conn.close()

    # Combine all records into a single DataFrame
    if all_records:
        print(f"\n{'='*80}")
        combined_df = pd.concat(all_records, ignore_index=True)
        print(f"Total records fetched: {total_records_fetched:,}")

        combined_df = combined_df.sort_values(["symbol", "timestamp"]).reset_index(
            drop=True
        )
        print(f"Combined DataFrame shape: {combined_df.shape}")
        print(f"Unique symbols in combined data: {combined_df['symbol'].nunique()}")
        print(
            f"Date range in combined data: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}"
        )

        if save_local:
            raw_dir = Path("data/raw")
            raw_dir.mkdir(parents=True, exist_ok=True)

            parquet_path = raw_dir / "all_stocks_historical.parquet"
            combined_df.to_parquet(parquet_path, index=False)
            print(f"Saved combined data to {parquet_path}")

            csv_path = raw_dir / "all_stocks_historical.csv"
            combined_df.to_csv(csv_path, index=False)
            print(f"Saved combined data to {csv_path}")

        print(f"{'='*80}\n")
        print("Data fetching complete.")
        return all_data
    else:
        print("No data fetched.")
        return None


def load_local_data(format="parquet"):
    """
    Load the consolidated local data file.

    Args:
        format (str, optional): File format to load ('parquet' or 'csv'). Defaults to 'parquet'.

    Returns:
        DataFrame with all stock data or None if file not found/error.
    """
    # Get project root directory (2 levels up from this script)
    project_root = Path(__file__).parent.parent.parent
    raw_dir = project_root / "data" / "raw"
    if format == "parquet":
        file_path = raw_dir / "all_stocks_historical.parquet"
        if file_path.exists():
            print(f"Loading data from {file_path}...")
            df = pd.read_parquet(file_path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            if "timestamp_et" in df.columns:
                df["timestamp_et"] = pd.to_datetime(df["timestamp_et"])
            print(f"Loaded {len(df):,} records from local parquet file.")
            return df
    else:
        file_path = raw_dir / "all_stocks_historical.csv"
        if file_path.exists():
            print(f"Loading data from {file_path}...")
            df = pd.read_csv(file_path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            if "timestamp_et" in df.columns:
                df["timestamp_et"] = pd.to_datetime(df["timestamp_et"])
            print(f"Loaded {len(df):,} records from local CSV file.")
            return df

    print(f"No local data file found at {file_path}.")
    return None


def get_symbol_data(symbol, df=None):
    """
    Extract data for a specific symbol from consolidated dataframe.

    Args:
        symbol: The stock symbol to filter data for.
        df (DataFrame, optional): Consolidated DataFrame (if None, will attempt to load local data).

    Returns:
        DataFrame with data for the specified symbol or None if not found.
    """

    if df is None:
        df = load_local_data()

    if df is None:
        return None

    symbol_df = df[df["symbol"] == symbol].copy()
    return symbol_df.reset_index(drop=True)


def get_data_summary():
    """
    Print a summary of the available data.
    """
    df = load_local_data()

    if df is None:
        print("No local data available for summary.")
        return

    symbols = df["symbol"].unique()

    print(f"Total records: {len(df):,}")
    print(f"Total unique symbols: {len(symbols)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Missing values:\n{df.isnull().sum()}")

    print("\nRecords per symbol:")
    for symbol in symbols:
        symbol_df = df[df["symbol"] == symbol]
        print(
            f"  {symbol}: {len(symbol_df):,} records from {symbol_df['timestamp'].min()} to {symbol_df['timestamp'].max()}"
        )


if __name__ == "__main__":

    print("\nFetching the data...\n")

    data = fetch_all_historical_data(save_local=True, fetch_from_min_date=True)

    if data:
        print("\nData fetched successfully.")
        get_data_summary()
    else:
        print("Failed to fetch data.")
