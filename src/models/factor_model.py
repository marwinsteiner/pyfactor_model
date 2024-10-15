import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.linear_model import LinearRegression
from pathlib import Path
import sys
from src.utils.data_preparation import load_snp_constituents
from loguru import logger
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


class FactorModel:
    def __init__(self, factors: List[str], benchmark_ticker: str = "SPY"):
        self.factors = factors
        self.factor_exposures = None
        self.factor_returns = None
        self.benchmark_ticker = benchmark_ticker
        self.snp_weights, _ = load_snp_constituents()
        logger.info(f"Loaded {len(self.factors)} factors")

    def construct_portfolio(self, data: Dict[str, pd.DataFrame], target_exposures: Dict[str, float]) -> pd.Series:
        logger.info("Constructing portfolio...")
        if self.factor_exposures is None:
            logger.warning("Factor exposures not calculated. Calculating now...")
            self.calculate_factor_exposures(data)

        if self.factor_exposures is None or self.factor_exposures.empty:
            logger.error("Failed to calculate factor exposures or exposures are empty")
            return pd.Series()

        logger.info(f"Factor exposures shape: {self.factor_exposures.shape}")

        stock_tickers = [ticker for ticker in self.factor_exposures.index if ticker != self.benchmark_ticker]

        # Convert target exposures to Series
        target = pd.Series(target_exposures)

        # Simple portfolio construction: maximize expected return subject to target factor exposures
        portfolio = pd.Series(0, index=stock_tickers)
        remaining_budget = 1.0

        for factor in self.factors:
            if factor in self.factor_exposures.columns:
                factor_portfolio = self.factor_exposures.loc[stock_tickers, factor] * target[factor]
                factor_sum = factor_portfolio.abs().sum()

                if factor_sum > 0:
                    factor_portfolio = factor_portfolio / factor_sum
                    allocation = min(remaining_budget, abs(target[factor]))
                    portfolio += factor_portfolio * allocation
                    remaining_budget -= allocation
                else:
                    logger.warning(f"Unable to allocate to {factor} factor due to zero sum of exposures.")
            else:
                logger.warning(f"Factor {factor} not found in factor exposures.")

        # Normalize weights to sum to 1, ignoring NaN values
        portfolio = portfolio.fillna(0)
        portfolio_sum = portfolio.abs().sum()
        if portfolio_sum > 0:
            portfolio = portfolio / portfolio_sum
        else:
            logger.warning("Unable to construct portfolio due to zero sum of weights.")
            return pd.Series(0, index=stock_tickers)

        return portfolio

    def calculate_factor_exposures(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        logger.info("Calculating factor exposures...")
        # Exclude the benchmark from calculations
        df = pd.DataFrame({ticker: data[ticker]['close'] for ticker in data if ticker != self.benchmark_ticker})
        logger.info(f"Data shape: {df.shape}")

        returns = df.pct_change(fill_method=None)
        logger.info(f"Returns shape before dropna: {returns.shape}")

        # Instead of filling NaN with 0, we'll drop rows with any NaN values
        returns = returns.dropna(how='any')
        logger.info(f"Returns shape after dropna: {returns.shape}")

        market_returns = returns.mean(axis=1)
        logger.info(f"Market returns shape: {market_returns.shape}")

        exposures = pd.DataFrame(index=df.columns, columns=self.factors)

        for ticker in df.columns:
            ticker_returns = returns[ticker]
            if ticker_returns.notna().sum() > 1 and market_returns.notna().sum() > 1:
                cov_matrix = np.cov(ticker_returns, market_returns)
                if cov_matrix.shape == (2, 2):
                    market_var = np.var(market_returns)
                    if market_var != 0:
                        exposures.loc[ticker, 'Market'] = cov_matrix[0, 1] / market_var
                    else:
                        exposures.loc[ticker, 'Market'] = np.nan
                else:
                    exposures.loc[ticker, 'Market'] = np.nan
            else:
                exposures.loc[ticker, 'Market'] = np.nan

            if not df[ticker].empty:
                last_price = df[ticker].iloc[-1]
                exposures.loc[ticker, 'Size'] = np.log(last_price) if last_price > 0 else np.nan
                exposures.loc[ticker, 'Value'] = 1 / last_price if last_price != 0 else np.nan

                if len(df) >= 2:
                    start_price = df[ticker].iloc[0]
                    exposures.loc[ticker, 'Momentum'] = (last_price / start_price) - 1 if start_price != 0 else np.nan
                else:
                    exposures.loc[ticker, 'Momentum'] = np.nan
            else:
                exposures.loc[ticker, ['Size', 'Value', 'Momentum']] = np.nan

        # Replace infinity values with NaN
        exposures = exposures.replace([np.inf, -np.inf], np.nan)

        # Drop any rows with NaN values
        exposures = exposures.dropna(how='any')

        logger.info(f"Calculated exposures shape: {exposures.shape}")
        self.factor_exposures = exposures
        return exposures

    def estimate_factor_returns(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        logger.info("Estimating factor returns...")
        # Exclude the benchmark from calculations
        df = pd.DataFrame({ticker: data[ticker]['close'] for ticker in data if ticker != self.benchmark_ticker})
        returns = df.pct_change(fill_method=None).fillna(0)

        if self.factor_exposures is None:
            self.calculate_factor_exposures(data)

        factor_returns = pd.DataFrame(index=returns.index, columns=self.factors)

        # Ensure factor_exposures and returns have the same stocks
        common_stocks = self.factor_exposures.index.intersection(returns.columns)
        X = self.factor_exposures.loc[common_stocks]

        # Create an imputer to handle NaN values
        imputer = SimpleImputer(strategy='mean')

        for date in returns.index:
            y = returns.loc[date, common_stocks]

            # Remove any remaining NaN values in y
            valid_indices = ~np.isnan(y)
            X_valid = X.loc[valid_indices]
            y_valid = y[valid_indices]

            if len(X_valid) > 0 and len(y_valid) > 0:
                # Impute NaN values in X_valid
                X_imputed = imputer.fit_transform(X_valid)

                model = LinearRegression().fit(X_imputed, y_valid)
                factor_returns.loc[date] = model.coef_
            else:
                factor_returns.loc[date] = np.nan

        self.factor_returns = factor_returns
        logger.info(f"Estimated factor returns shape: {factor_returns.shape}")
        return factor_returns
