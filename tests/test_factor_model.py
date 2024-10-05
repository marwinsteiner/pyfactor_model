import unittest
import pandas as pd
from src.models.factor_model import FactorModel
from src.utils.data_preparation import prepare_returns_data
from src.utils.portfolio_construction import construct_equal_weight_portfolio
from src.utils.performance_evaluation import PortfolioPerformance


class TestFactorModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Sample data for testing
        cls.sample_data = {
            'AAPL': pd.DataFrame({
                'close': [100, 101, 102, 103, 104],
                'volume': [1000000] * 5
            }, index=pd.date_range('2023-01-01', periods=5)),
            'GOOGL': pd.DataFrame({
                'close': [1500, 1510, 1520, 1530, 1540],
                'volume': [500000] * 5
            }, index=pd.date_range('2023-01-01', periods=5)),
            'SPY': pd.DataFrame({
                'close': [400, 402, 404, 406, 408],
                'volume': [5000000] * 5
            }, index=pd.date_range('2023-01-01', periods=5))
        }
        cls.benchmark_ticker = 'SPY'

    def test_factor_model_initialization(self):
        model = FactorModel(['Market', 'Size', 'Value', 'Momentum'], benchmark_ticker=self.benchmark_ticker)
        self.assertEqual(model.factors, ['Market', 'Size', 'Value', 'Momentum'])
        self.assertEqual(model.benchmark_ticker, 'SPY')

    def test_calculate_factor_exposures(self):
        model = FactorModel(['Market', 'Size', 'Value', 'Momentum'], benchmark_ticker=self.benchmark_ticker)
        exposures = model.calculate_factor_exposures(self.sample_data)
        self.assertIsInstance(exposures, pd.DataFrame)
        self.assertEqual(exposures.shape, (2, 4))  # 2 stocks, 4 factors
        self.assertNotIn(self.benchmark_ticker, exposures.index)

    def test_estimate_factor_returns(self):
        model = FactorModel(['Market', 'Size', 'Value', 'Momentum'], benchmark_ticker=self.benchmark_ticker)
        returns = model.estimate_factor_returns(self.sample_data)
        self.assertIsInstance(returns, pd.DataFrame)
        self.assertEqual(returns.shape[1], 4)  # 4 factors

    def test_construct_portfolio(self):
        model = FactorModel(['Market', 'Size', 'Value', 'Momentum'], benchmark_ticker=self.benchmark_ticker)
        target_exposures = {'Market': 1.0, 'Size': -0.2, 'Value': 0.5, 'Momentum': 0.3}
        portfolio = model.construct_portfolio(self.sample_data, target_exposures)
        self.assertIsInstance(portfolio, pd.Series)
        self.assertEqual(len(portfolio), 2)  # 2 stocks
        self.assertAlmostEqual(portfolio.sum(), 1.0, places=6)  # Weights should sum to 1

    def test_prepare_returns_data(self):
        returns = prepare_returns_data(self.sample_data)
        self.assertIsInstance(returns, pd.DataFrame)
        self.assertEqual(returns.shape, (4, 3))  # 4 periods, 3 stocks (including benchmark)

    def test_construct_equal_weight_portfolio(self):
        returns = prepare_returns_data(self.sample_data)
        equal_weight = construct_equal_weight_portfolio(returns)
        self.assertIsInstance(equal_weight, pd.Series)
        self.assertEqual(len(equal_weight), 4)  # 4 periods

    def test_portfolio_performance(self):
        returns = prepare_returns_data(self.sample_data)
        portfolio_returns = construct_equal_weight_portfolio(returns)
        benchmark_returns = returns[self.benchmark_ticker]
        performance = PortfolioPerformance(portfolio_returns, benchmark_returns)
        summary = performance.summary()
        self.assertIsInstance(summary, dict)
        self.assertIn('Sharpe Ratio', summary)
        self.assertIn('Max Drawdown', summary)


if __name__ == '__main__':
    unittest.main()
