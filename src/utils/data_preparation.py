import pandas as pd
from src.data.data_fetcher import DataFetcher
from typing import Dict, Tuple, List
from pathlib import Path


def prepare_returns_data(historical_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    close_prices = {ticker: data['close'] for ticker, data in historical_data.items()}
    price_df = pd.DataFrame(close_prices)
    return price_df.pct_change(fill_method=None)


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


def load_snp_constituents() -> Tuple[pd.Series, List[str]]:
    """
    Load S&P 500 constituents and their weights from a CSV file.

    Returns:
        Tuple[pd.Series, List[str]]: A series with tickers as index and weights as values,
                                     and a list of all tickers.
    """
    file_path = Path(__file__).parent.parent / "data" / "constituents" / "snp_constituents.csv"
    constituents_df = pd.read_csv(file_path)

    # Remove percentage symbol and convert to float
    constituents_df['weight'] = constituents_df['weight'].str.rstrip('%').astype('float') / 100.0

    weights = constituents_df.set_index('ticker')['weight']
    tickers = constituents_df['ticker'].tolist()

    return weights, tickers
