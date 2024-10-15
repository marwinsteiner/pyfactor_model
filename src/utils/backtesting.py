from typing import Tuple, Dict
import pandas as pd
from src.models.factor_model import FactorModel
from src.utils import data_preparation
from loguru import logger


def backtest_strategy(historical_data: Dict[str, pd.DataFrame],
                      model: FactorModel,
                      rebalance_frequency: str,
                      window_size: int) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Implement rolling window backtesting approach.

    Args:
        historical_data (Dict[str, pd.DataFrame]): Historical price data
        model (FactorModel): The factor model to use
        rebalance_frequency (str): Frequency of rebalancing (e.g., 'M' for monthly)
        window_size (int): Size of the rolling window in days

    Returns:
        Tuple[pd.Series, pd.DataFrame]: Portfolio returns and weights over time
    """
    logger.info("Starting backtesting strategy...")
    returns_data = data_preparation.prepare_returns_data(historical_data)
    logger.info(f"Returns data shape: {returns_data.shape}")

    portfolio_returns = pd.Series(index=returns_data.index)
    portfolio_weights = pd.DataFrame(index=returns_data.index, columns=returns_data.columns)

    benchmark_returns = returns_data[model.benchmark_ticker]
    stock_returns = returns_data.drop(columns=[model.benchmark_ticker])

    for end_date in stock_returns.resample(rebalance_frequency).last().index:
        logger.info(f"Processing end date: {end_date}")
        start_date = end_date - pd.Timedelta(days=window_size)
        window_data = {ticker: historical_data[ticker].loc[start_date:end_date]
                       for ticker in historical_data if ticker != model.benchmark_ticker}
        logger.info(f"Window data size: {len(window_data)}")

        model.calculate_factor_exposures(window_data)
        model.estimate_factor_returns(window_data)

        target_exposures = {'Market': 1.0, 'Size': -0.2, 'Value': 0.5, 'Momentum': 0.3}
        weights = model.construct_portfolio(window_data, target_exposures)

        logger.info(f"Calculated weights shape: {weights.shape}")

        portfolio_weights.loc[end_date, weights.index] = weights
        future_returns = stock_returns.loc[end_date:].iloc[:20]  # Assume 20 trading days per month
        portfolio_returns.loc[end_date:] = (future_returns * weights).sum(axis=1)

    logger.info("Backtesting strategy completed.")
    return portfolio_returns.dropna(), portfolio_weights.dropna()
