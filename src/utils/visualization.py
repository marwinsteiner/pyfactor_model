import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import pandas as pd


def plot_cumulative_returns(returns_dict: Dict[str, pd.Series]) -> None:
    """Plot cumulative returns of multiple portfolios/benchmarks."""
    cumulative_returns = {name: (1 + returns).cumprod() for name, returns in returns_dict.items()}
    plt.figure(figsize=(12, 6))
    for name, returns in cumulative_returns.items():
        plt.plot(returns.index, returns.values, label=name)
    plt.title('Cumulative Returns Comparison')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()


def plot_drawdown(returns: pd.Series) -> None:
    """Plot drawdown of a given portfolio over time."""
    returns = returns.astype(float)  # make them all floats
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    plt.figure(figsize=(12, 6))
    plt.plot(drawdown.index, drawdown.values)
    plt.title('Portfolio Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.fill_between(drawdown.index, drawdown.values, 0, alpha=0.1)
    plt.show()


def plot_factor_attribution(attribution: pd.Series) -> None:
    """Create a bar plot of factor attribution results."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x=attribution.index, y=attribution.values)
    plt.title('Factor Attribution')
    plt.xlabel('Factors')
    plt.ylabel('Attribution')
    plt.xticks(rotation=45)
    plt.show()
