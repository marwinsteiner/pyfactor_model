from typing import Tuple, Dict
import pandas as pd
from src.models.factor_model import FactorModel
from src.utils import data_preparation


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
    returns_data = data_preparation.prepare_returns_data(historical_data)
    portfolio_returns = pd.Series(index=returns_data.index)
    portfolio_weights = pd.DataFrame(index=returns_data.index, columns=returns_data.columns)

    for end_date in returns_data.resample(rebalance_frequency).last().index:
        start_date = end_date - pd.Timedelta(days=window_size)
        window_data = {ticker: data.loc[start_date:end_date] for ticker, data in historical_data.items()}

        model.calculate_factor_exposures(window_data)
        model.estimate_factor_returns(window_data)

        target_exposures = {'Market': 1.0, 'Size': -0.2, 'Value': 0.5, 'Momentum': 0.3}
        weights = model.construct_portfolio(window_data, target_exposures)

        portfolio_weights.loc[end_date] = weights
        future_returns = returns_data.loc[end_date:].iloc[:20]  # Assume 20 trading days per month
        portfolio_returns.loc[end_date:] = (future_returns * weights).sum(axis=1)

    return portfolio_returns.dropna(), portfolio_weights.dropna()
