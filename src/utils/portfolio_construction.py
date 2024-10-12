import pandas as pd
import numpy as np
from typing import Dict
from src.utils.data_preparation import load_snp_constituents, prepare_returns_data


def construct_equal_weight_portfolio(returns_data: pd.DataFrame) -> pd.Series:
    """Construct an equal-weighted portfolio."""
    weights = np.ones(len(returns_data.columns)) / len(returns_data.columns)
    return (returns_data * weights).sum(axis=1)


def construct_market_cap_weight_portfolio(historical_data: Dict[str, pd.DataFrame]) -> pd.Series:
    """Construct a market-cap weighted portfolio using S&P 500 weights."""
    returns_data = prepare_returns_data(historical_data)
    snp_weights = load_snp_constituents()

    # Filter weights for stocks present in our historical data
    portfolio_weights = snp_weights[snp_weights.index.isin(returns_data.columns)]

    # Normalize weights to sum to 1
    portfolio_weights = portfolio_weights / portfolio_weights.sum()

    return (returns_data * portfolio_weights).sum(axis=1)
