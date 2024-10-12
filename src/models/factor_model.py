import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.linear_model import LinearRegression
from pathlib import Path
import sys
from src.utils.data_preparation import load_snp_constituents

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


class FactorModel:
    def __init__(self, factors: List[str], benchmark_ticker: str = 'SPY'):
        self.factors = factors
        self.factor_exposures = None
        self.factor_returns = None
        self.benchmark_ticker = benchmark_ticker
        self.snp_weights, _ = load_snp_constituents()

    def calculate_factor_exposures(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = pd.DataFrame({ticker: data[ticker]['close'] for ticker in data})
        returns = df.pct_change(fill_method=None).dropna()

        market_returns = returns.mean(axis=1)

        exposures = pd.DataFrame(index=df.columns, columns=self.factors)

        for ticker in df.columns:
            if returns[ticker].notna().sum() > 1 and market_returns.notna().sum() > 1:
                cov_matrix = np.cov(returns[ticker].dropna(), market_returns.dropna())
                if cov_matrix.shape == (2, 2):
                    market_var = np.var(market_returns.dropna())
                    if market_var != 0:
                        exposures.loc[ticker, 'Market'] = cov_matrix[0, 1] / market_var
                    else:
                        exposures.loc[ticker, 'Market'] = np.nan
                else:
                    exposures.loc[ticker, 'Market'] = np.nan
            else:
                exposures.loc[ticker, 'Market'] = np.nan

            if not df[ticker].empty:
                exposures.loc[ticker, 'Size'] = np.log(df[ticker].iloc[-1]) if df[ticker].iloc[-1] > 0 else np.nan
                exposures.loc[ticker, 'Value'] = 1 / df[ticker].iloc[-1] if df[ticker].iloc[-1] != 0 else np.nan

                if len(df) >= 252:
                    exposures.loc[ticker, 'Momentum'] = (df[ticker].iloc[-1] / df[ticker].iloc[-252]) - 1 if \
                        df[ticker].iloc[-252] != 0 else np.nan
                else:
                    exposures.loc[ticker, 'Momentum'] = np.nan
            else:
                exposures.loc[ticker, ['Size', 'Value', 'Momentum']] = np.nan

        return exposures

    def estimate_factor_returns(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        # Exclude the benchmark from factor returns estimation
        stock_data = {ticker: df for ticker, df in data.items() if ticker != self.benchmark_ticker}

        # Convert data to DataFrame of closing prices
        df = pd.DataFrame({ticker: stock_data[ticker]['close'] for ticker in stock_data})

        # Calculate returns
        returns = df.pct_change(fill_method=None).dropna()

        # Ensure factor exposures are calculated
        if self.factor_exposures is None:
            self.calculate_factor_exposures(data)

        # Prepare for cross-sectional regression
        factor_returns = pd.DataFrame(index=returns.index, columns=self.factors)

        # Perform cross-sectional regression for each time period
        for date in returns.index:
            y = returns.loc[date]
            X = self.factor_exposures.dropna()  # Remove any stocks with NaN exposures

            if len(X) > 0 and not y.isnull().all():  # Only perform regression if we have valid data
                model = LinearRegression().fit(X, y[X.index])
                factor_returns.loc[date] = model.coef_  # type: ignore
            else:
                factor_returns.loc[date] = np.nan

        # After the loop, remove any dates with all NaN values
        factor_returns = factor_returns.dropna(how='all')
        self.factor_returns = factor_returns
        return factor_returns

    def construct_portfolio(self, data: Dict[str, pd.DataFrame],
                            target_exposures: Dict[str, float]) -> pd.Series:
        # Ensure factor exposures and returns are calculated
        if self.factor_exposures is None:
            self.calculate_factor_exposures(data)
        if self.factor_returns is None:
            self.estimate_factor_returns(data)

        # Exclude the benchmark from portfolio construction
        stock_tickers = [ticker for ticker in self.factor_exposures.index if ticker != self.benchmark_ticker]

        # Convert target exposures to Series
        target = pd.Series(target_exposures)

        # Simple portfolio construction: maximize expected return subject to target factor exposures
        # Note: This is a simplified approach. In practice, one might use optimization techniques.
        portfolio = pd.Series(0, index=stock_tickers)
        remaining_budget = 1.0

        for factor in self.factors:
            factor_portfolio = self.factor_exposures.loc[stock_tickers, factor] * target[factor]
            factor_sum = factor_portfolio.abs().sum()

            if factor_sum > 0:
                factor_portfolio = factor_portfolio / factor_sum
                allocation = min(remaining_budget, abs(target[factor]))
                portfolio += factor_portfolio * allocation
                remaining_budget -= allocation
            else:
                print(f'Warning: Unable to allocate to {factor} factor due to zero sum of exposures.')

        # Normalize weights to sum to 1, ignoring NaN values
        pd.set_option('future.no_silent_downcasting', True)
        """At the time of writing, downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will 
        change in a future version. Hence, we opt into the new behavior to deal with the FutureWarning."""
        portfolio = portfolio.fillna(0).infer_objects(copy=False)
        portfolio_sum = portfolio.abs().sum()
        if portfolio_sum > 0:
            portfolio = portfolio / portfolio_sum
        else:
            print('Warning: Unable to construct portfolio due to zero sum of weights.')
            return pd.Series(0, index=stock_tickers)

        return portfolio
