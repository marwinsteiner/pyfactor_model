import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from src.data.data_fetcher import DataFetcher


class TestDataFetcher(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test data
        self.test_data_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.test_data_dir,
                                              ignore_errors=True))  # lambda function to explicitly specify all
        # necessary arguments, `ignore_errors=True` added to ensure the cleanup process contineus even if it
        # encounters any issues (e.g. permissions)

    @patch('src.data.data_fetcher.RESTClient')
    def test_fetch_historical_data_on_demand(self, mock_rest_client):
        # Mock the Polygon API response
        mock_aggs = [
            MagicMock(timestamp=1609459200000, open=100, high=101, low=99, close=100.5, volume=1000000, vwap=100.2,
                      transactions=5000),
            MagicMock(timestamp=1609545600000, open=100.5, high=102, low=100, close=101.5, volume=1100000, vwap=101.0,
                      transactions=5500)
        ]
        mock_rest_client.return_value.get_aggs.return_value = mock_aggs

        fetcher = DataFetcher(mode="on_demand")
        fetcher.data_dir = Path(self.test_data_dir)

        data = fetcher.fetch_historical_data(
            tickers=["AAPL", "GOOGL"],
            start_date="2021-01-01",
            end_date="2021-01-02"
        )

        self.assertIn("AAPL", data)
        self.assertIn("GOOGL", data)
        self.assertEqual(len(data["AAPL"]), 2)
        self.assertEqual(len(data["GOOGL"]), 2)
        self.assertListEqual(list(data["AAPL"].columns),
                             ['open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions'])

    @patch('src.data.data_fetcher.RESTClient')
    def test_fetch_historical_data_persistent(self, mock_rest_client):
        # Create some test CSV files
        aapl_df = pd.DataFrame({
            'timestamp': ['2021-01-01', '2021-01-02'],
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102],
            'volume': [1000000, 1100000],
            'vwap': [100.5, 101.5],
            'transactions': [5000, 5500]
        })
        aapl_df.to_csv(Path(self.test_data_dir) / "AAPL.csv", index=False)

        fetcher = DataFetcher(mode="persistent")
        fetcher.data_dir = Path(self.test_data_dir)

        data = fetcher.fetch_historical_data(
            tickers=["AAPL", "GOOGL"],
            start_date="2021-01-01",
            end_date="2021-01-02"
        )

        self.assertIn("AAPL", data)
        self.assertEqual(len(data["AAPL"]), 2)
        self.assertListEqual(list(data["AAPL"].columns),
                             ['open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions'])

        # GOOGL should not be in the data as we didn't create a CSV for it
        self.assertNotIn("GOOGL", data)

    def test_convert_timestamp_to_date(self):
        fetcher = DataFetcher()
        date_str = fetcher._convert_timestamp_to_date(1609459200000)  # 2021-01-01 00:00:00 UTC
        self.assertEqual(date_str, "2021-01-01")

    @patch('src.data.data_fetcher.RESTClient')
    def test_fetch_and_save_tickers(self, mock_rest_client):
        # Mock the Polygon API response
        mock_aggs = [
            MagicMock(timestamp=1609459200000, open=100, high=101, low=99, close=100.5, volume=1000000, vwap=100.2,
                      transactions=5000),
            MagicMock(timestamp=1609545600000, open=100.5, high=102, low=100, close=101.5, volume=1100000, vwap=101.0,
                      transactions=5500)
        ]
        mock_rest_client.return_value.get_aggs.return_value = mock_aggs

        fetcher = DataFetcher(mode="persistent")
        fetcher.data_dir = Path(self.test_data_dir)

        historical_data = {}
        fetcher._fetch_and_save_tickers(
            tickers=["AAPL"],
            start_date="2021-01-01",
            end_date="2021-01-02",
            timespan="day",
            limit=50000,
            adjusted=True,
            historical_data=historical_data
        )

        self.assertIn("AAPL", historical_data)
        self.assertEqual(len(historical_data["AAPL"]), 2)
        self.assertTrue(Path(self.test_data_dir, "AAPL.csv").exists())


if __name__ == '__main__':
    unittest.main()
