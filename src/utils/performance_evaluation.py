import numpy as np
import pandas as pd
from typing import Tuple, Dict
from loguru import logger


class PortfolioPerformance:
    def __init__(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.02):
        self.portfolio_returns = portfolio_returns
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        self.daily_risk_free_rate = (1 + risk_free_rate) ** (1 / 252) - 1

        # Align the returns
        self.portfolio_returns, self.benchmark_returns = self.portfolio_returns.align(self.benchmark_returns,
                                                                                      join='inner')

        logger.info(f"Portfolio returns shape: {self.portfolio_returns.shape}")
        logger.info(f"Benchmark returns shape: {self.benchmark_returns.shape}")

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
        excess_portfolio_returns = self.portfolio_returns - self.daily_risk_free_rate
        excess_benchmark_returns = self.benchmark_returns - self.daily_risk_free_rate

        logger.info(f"Excess portfolio returns shape: {excess_portfolio_returns.shape}")
        logger.info(f"Excess benchmark returns shape: {excess_benchmark_returns.shape}")

        if len(excess_portfolio_returns) != len(excess_benchmark_returns):
            raise ValueError("Portfolio and benchmark returns have different lengths")

        # Calculate covariance manually to avoid issues with np.cov() edge case
        portfolio_var = np.var(excess_portfolio_returns)
        benchmark_var = np.var(excess_benchmark_returns)
        covariance = np.mean(excess_portfolio_returns * excess_benchmark_returns) - (
                    np.mean(excess_portfolio_returns) * np.mean(excess_benchmark_returns))

        beta = covariance / benchmark_var
        alpha = np.mean(excess_portfolio_returns) - (beta * np.mean(excess_benchmark_returns))

        return float(alpha * 252), float(beta)  # Annualize alpha and convert to float

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
