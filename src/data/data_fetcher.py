from polygon import RESTClient
from dynaconf import Dynaconf
from typing import List, Dict
from datetime import datetime, timedelta
import time
import pandas as pd
from src.utils.path_utils import get_project_root

settings = Dynaconf(settings_files=['settings.json', '.secrets.json'])


class DataFetcher:
    def __init__(self, mode: str = 'on_demand'):
        self.client = RESTClient(api_key=settings.POLYGON_API_KEY)
        self.mode = mode
        self.data_dir = get_project_root() / 'src' / 'data' / 'data_download'
        if self.mode == 'persistent':
            self.data_dir.mkdir(exist_ok=True)
        self.rate_limit = 5 / 60
        self.last_request_time = 0
        self.request_count = 0
        self.last_reset_time = time.time()

    def fetch_historical_data(
            self,
            tickers: List[str],
            start_date: str,
            end_date: str,
            timespan: str = 'day',
            limit: int = 50000,
            adjusted: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple tickers from Polygon API.

        Args:
            tickers (List[str]): List of ticker symbols.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            timespan (str): The timespan to use for the data (e.g., 'day', 'hour', 'minute').
            limit (int): The maximum number of base aggregates to return.
            adjusted (bool): Whether to use adjusted data.

        Returns:
            Dict[str, Any]: A dictionary containing historical data for each ticker.
        """
        historical_data = {}

        if self.mode == 'persistent':
            # Read all CSV files in the data directory
            for csv_file in self.data_dir.glob('*.csv'):
                ticker = csv_file.stem  # Get filename without extension
                df = pd.read_csv(csv_file, parse_dates=['timestamp'])
                df.set_index('timestamp', inplace=True)
                historical_data[ticker] = df
                print(f'Loaded data for {ticker} from file')

            # Fetch data for tickers not present in the directory
            missing_tickers = set(tickers) - set(historical_data.keys())
            self._fetch_and_save_tickers(missing_tickers, start_date, end_date, timespan, limit, adjusted,
                                         historical_data)
        else:
            # On-demand mode: fetch data for all requested tickers
            self._fetch_and_save_tickers(tickers, start_date, end_date, timespan, limit, adjusted, historical_data)
        return historical_data

    def _fetch_and_save_tickers(self, tickers, start_date, end_date, timespan, limit, adjusted, historical_data):
        for ticker in tickers:
            try:
                df = self._fetch_ticker_data(ticker, start_date, end_date, timespan, limit, adjusted)

                if df.empty:
                    print(f'No data fetched for {ticker}')
                    continue

                if self.mode == 'persistent':
                    df.to_csv(self.data_dir / f'{ticker}.csv')

                historical_data[ticker] = df
                print(f'Successfully fetched data for {ticker}')
            except Exception as e:
                print(f'Error fetching data for {ticker}: {str(e)}')

    def _fetch_ticker_data(self, ticker, start_date, end_date, timespan, limit, adjusted):
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        chunk_size = timedelta(days=365)  # Fetch data in 1-year chunks

        all_data = []
        current_start = start

        while current_start < end:
            current_end = min(current_start + chunk_size, end)

            self._rate_limit()  # Apply rate limiting b/f each request
            aggs = self.client.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan=timespan,
                from_=current_start.strftime('%Y-%m-%d'),
                to=current_end.strftime('%Y-%m-%d'),
                limit=limit,
                adjusted=adjusted
            )

            chunk_data = pd.DataFrame([
                {
                    'timestamp': self._convert_timestamp_to_date(bar.timestamp),
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                    'vwap': bar.vwap,
                    'transactions': bar.transactions
                }
                for bar in aggs
            ])

            all_data.append(chunk_data)
            current_start = current_end + timedelta(days=1)

        if all_data:
            df = pd.concat(all_data)
            df.set_index('timestamp', inplace=True)
            return df
        else:
            return pd.DataFrame()

    def _rate_limit(self):
        current_time = time.time()

        # Reset counter if a minute has passed
        if current_time - self.last_reset_time >= 60:
            self.request_count = 0
            self.last_reset_time = current_time

        # If we've made 5 requests in the last minute, wait
        if self.request_count >= 5:
            sleep_time = 60 - (current_time - self.last_reset_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.request_count = 0
            self.last_reset_time = time.time()

        # Increment request count and update last request time
        self.request_count += 1
        self.last_request_time = time.time()

    def _convert_timestamp_to_date(self, timestamp: int) -> str:
        """
        Convert Unix timestamp to date string.

        Args:
            timestamp (int): Unix timestamp in milliseconds.

        Returns:
            str: Date string in 'YYYY-MM-DD' format.
        """
        return datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d')
