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
    snp_weights, _ = load_snp_constituents()

    # Ensure snp_weights is a Series
    if not isinstance(snp_weights, pd.Series):
        raise TypeError("Expected snp_weights to be a pandas Series")

    # Filter weights for stocks present in our historical data
    portfolio_weights = snp_weights[snp_weights.index.isin(returns_data.columns)]

    # Check if we have any valid weights
    if portfolio_weights.empty:
        raise ValueError("No matching stocks found between S&P 500 constituents and historical data")

    # Normalize weights to sum to 1
    portfolio_weights = portfolio_weights / portfolio_weights.sum()

    # Align the weights with the returns data
    aligned_weights = portfolio_weights.reindex(returns_data.columns, fill_value=0)

    return (returns_data * aligned_weights).sum(axis=1)
