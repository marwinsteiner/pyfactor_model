import numpy as np
import pandas as pd
from typing import Tuple, Dict


class PortfolioPerformance:
    def __init__(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.02):
        """
        Initialize the PortfolioPerformance class.

        Args:
            portfolio_returns (pd.Series): Daily returns of the portfolio
            benchmark_returns (pd.Series): Daily returns of the benchmark
            risk_free_rate (float): Annualized risk-free rate (default: 2%)
        """
        self.portfolio_returns = portfolio_returns
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        self.daily_risk_free_rate = (1 + risk_free_rate) ** (1 / 252) - 1

    def total_return(self) -> float:
        """Calculate the total return of the portfolio."""
        return (self.portfolio_returns + 1).prod() - 1

    def annualized_return(self) -> float:
        """Calculate the annualized return of the portfolio."""
        total_return = self.total_return()
        num_years = len(self.portfolio_returns) / 252
        return (1 + total_return) ** (1 / num_years) - 1

    def sharpe_ratio(self) -> float:
        """Calculate the Sharpe ratio of the portfolio."""
        excess_returns = self.portfolio_returns - self.daily_risk_free_rate
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def max_drawdown(self) -> float:
        """Calculate the maximum drawdown of the portfolio."""
        cumulative_returns = (1 + self.portfolio_returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        return drawdown.min()

    def alpha_beta(self) -> Tuple[float, float]:
        """Calculate the alpha and beta of the portfolio relative to the benchmark."""
        excess_portfolio_returns = self.portfolio_returns - self.daily_risk_free_rate
        excess_benchmark_returns = self.benchmark_returns - self.daily_risk_free_rate

        beta = np.cov(excess_portfolio_returns, excess_benchmark_returns)[0, 1] / np.var(excess_benchmark_returns)
        alpha = excess_portfolio_returns.mean() - (beta * excess_benchmark_returns.mean())

        return alpha * 252, beta  # Annualize alpha

    def information_ratio(self) -> float:
        """Calculate the information ratio of the portfolio."""
        active_returns = self.portfolio_returns - self.benchmark_returns
        return np.sqrt(252) * active_returns.mean() / active_returns.std()

    def tracking_error(self) -> float:
        """Calculate the tracking error of the portfolio relative to the benchmark."""
        return np.sqrt(252) * (self.portfolio_returns - self.benchmark_returns).std()

    def sortino_ratio(self) -> float:
        """Calculate the Sortino ratio of the portfolio."""
        excess_returns = self.portfolio_returns - self.daily_risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        return np.sqrt(252) * excess_returns.mean() / downside_deviation

    def calmar_ratio(self) -> float:
        """Calculate the Calmar ratio of the portfolio."""
        return self.annualized_return() / abs(self.max_drawdown())

    def summary(self) -> Dict[str, float]:
        """Generate a summary of all performance metrics."""
        alpha, beta = self.alpha_beta()
        return {
            'Total Return': self.total_return(),
            'Annualized Return': self.annualized_return(),
            'Sharpe Ratio': self.sharpe_ratio(),
            'Max Drawdown': self.max_drawdown(),
            'Alpha': alpha,
            'Beta': beta,
            'Information Ratio': self.information_ratio(),
            'Tracking Error': self.tracking_error(),
            'Sortino Ratio': self.sortino_ratio(),
            'Calmar Ratio': self.calmar_ratio()
        }


def calculate_turnover(portfolio_weights: pd.DataFrame) -> float:
    """
    Calculate the average turnover of the portfolio.

    Args:
        portfolio_weights (pd.DataFrame): DataFrame of portfolio weights over time

    Returns:
        float: Average turnover
    """
    weight_changes = portfolio_weights.diff().abs().sum(axis=1)
    return weight_changes.mean() / 2  # Divide by 2 to avoid double counting


def perform_factor_attribution(factor_returns: pd.DataFrame, factor_exposures: pd.DataFrame,
                               portfolio_returns: pd.Series) -> pd.Series:
    """
    Perform factor attribution analysis.

    Args:
        factor_returns (pd.DataFrame): Factor returns over time
        factor_exposures (pd.DataFrame): Factor exposures of the portfolio
        portfolio_returns (pd.Series): Portfolio returns

    Returns:
        pd.Series: Attribution of returns to each factor
    """
    factor_contribution = factor_returns.multiply(factor_exposures.mean(), axis=1)
    total_factor_contribution = factor_contribution.sum()

    # Calculate residual return
    total_portfolio_return = portfolio_returns.sum()
    residual = total_portfolio_return - total_factor_contribution.sum()

    attribution = total_factor_contribution._append(pd.Series({'Residual': residual}))
    return attribution / total_portfolio_return  # Return as a percentage of total return
