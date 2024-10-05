import pandas as pd
import numpy as np
from typing import Dict
from src.utils import data_preparation


def construct_equal_weight_portfolio(returns_data: pd.DataFrame) -> pd.Series:
    """Construct an equal-weighted portfolio."""
    weights = np.ones(len(returns_data.columns)) / len(returns_data.columns)
    return (returns_data * weights).sum(axis=1)


def construct_market_cap_weight_portfolio(historical_data: Dict[str, pd.DataFrame]) -> pd.Series:
    """Construct a market-cap weighted portfolio."""
    market_caps = pd.Series({ticker: data['close'].iloc[-1] * data['volume'].iloc[-1]
                             for ticker, data in historical_data.items()})
    weights = market_caps / market_caps.sum()
    returns_data = data_preparation.prepare_returns_data(historical_data)
    return (returns_data * weights).sum(axis=1)
