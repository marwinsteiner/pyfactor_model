import pandas as pd
from src.data.data_fetcher import DataFetcher
from typing import Dict


def prepare_returns_data(historical_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Convert historical price data to returns data."""
    returns_data = pd.DataFrame()
    for ticker, data in historical_data.items():
        returns_data[ticker] = data['close'].pct_change()
    return returns_data.dropna()


def fetch_benchmark_returns(start_date: str, end_date: str, benchmark: str) -> pd.Series:
    """Fetch benchmark returns for the given period."""
    # Fetch historical data using DataFetcher
    data_fetcher = DataFetcher(mode='persistent')
    benchmark_data = data_fetcher.fetch_historical_data(
        tickers=[benchmark],
        start_date=start_date,
        end_date=end_date
    )

    if not benchmark_data or benchmark not in benchmark_data:
        print(f'No data available for benchmark {benchmark}. Exiting...')
        exit(1)

    # Extract the 'close' prices for the benchmark and calculate returns
    benchmark_prices = benchmark_data[benchmark]['close']
    return benchmark_prices.pct_change().dropna()
