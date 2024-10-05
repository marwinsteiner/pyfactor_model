import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.linear_model import LinearRegression
from src.data.data_fetcher import DataFetcher
from pathlib import Path
import sys

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


class FactorModel:
    def __init__(self, factors: List[str]):
        self.factors = factors
        self.factor_exposures = None
        self.factor_returns = None

    def calculate_factor_exposures(self, input_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        # Convert data to DataFrame of closing prices
        df_close = pd.DataFrame({ticker_data: input_data[ticker_data]['close'] for ticker_data in input_data})

        # Calculate returns
        returns_data = df_close.pct_change(fill_method=None).dropna()

        # Calculate market returns (assuming equal-weighted portfolio)
        market_returns = returns_data.mean(axis=1)

        # Calculate factor exposures
        portfolio_exposures = pd.DataFrame(index=df_close.columns, columns=self.factors)

        for ticker_data in df_close.columns:
            # Market factor (Beta)
            portfolio_exposures.loc[ticker_data, 'Market'] = np.cov(returns_data[ticker_data], market_returns)[
                                                                 0, 1] / np.var(
                market_returns)

            # Size factor (market cap as proxy, using last close price as simple proxy)
            portfolio_exposures.loc[ticker_data, 'Size'] = np.log(df_close[ticker_data].iloc[-1])

            # Value factor (assuming book value is available, using price-to-book ratio)
            # Note: In a real scenario, you would need to incorporate book value data
            portfolio_exposures.loc[ticker_data, 'Value'] = 1 / df_close[ticker_data].iloc[
                -1]  # Placeholder, inverse of price as proxy

            # Momentum factor (use available data if less than 12 months)
            if len(df_close) > 1:
                portfolio_exposures.loc[ticker_data, 'Momentum'] = (df_close[ticker_data].iloc[-1] /
                                                                    df_close[ticker_data].iloc[0]) - 1
            else:
                portfolio_exposures.loc[ticker_data, 'Momentum'] = np.nan

        self.factor_exposures = portfolio_exposures
        return portfolio_exposures

    def estimate_factor_returns(self, input_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        # Convert data to DataFrame of closing prices
        df_close = pd.DataFrame({ticker_data: input_data[ticker_data]['close'] for ticker_data in input_data})

        # Calculate returns
        returns_data = df_close.pct_change(fill_method=None).dropna()

        # Ensure factor exposures are calculated
        if self.factor_exposures is None:
            self.calculate_factor_exposures(input_data)

        # Prepare for cross-sectional regression
        factor_returns = pd.DataFrame(index=returns_data.index, columns=self.factors)

        # Perform cross-sectional regression for each time period
        for date in returns_data.index:
            y_data = returns_data.loc[date]
            x_data = self.factor_exposures.dropna()  # Remove any stocks with NaN exposures

            if len(x_data) > 0 and not y_data.isnull().all():  # Only perform regression if we have valid data
                model = LinearRegression().fit(x_data, y_data[x_data.index])
                factor_returns.loc[date] = model.coef_  # type: ignore
            else:
                factor_returns.loc[date] = np.nan

        # After the loop, remove any dates with all NaN values
        factor_returns = factor_returns.dropna(how='all')
        self.factor_returns = factor_returns
        return factor_returns

    def construct_portfolio(self, input_data: Dict[str, pd.DataFrame],
                            target_exposures: Dict[str, float]) -> pd.Series:
        # Ensure factor exposures and returns are calculated
        if self.factor_exposures is None:
            self.calculate_factor_exposures(input_data)
        if self.factor_returns is None:
            self.estimate_factor_returns(input_data)

        # Convert target exposures to Series
        target = pd.Series(target_exposures)

        # Simple portfolio construction: maximize expected return subject to target factor exposures
        # Note: This is a simplified approach. In practice, you might use optimization techniques.
        portfolio = pd.Series(0, index=self.factor_exposures.index)
        remaining_budget = 1.0

        for factor in self.factors:
            factor_portfolio = self.factor_exposures[factor] * target[factor]
            factor_sum = factor_portfolio.abs().sum()

            if factor_sum > 0:
                factor_portfolio = factor_portfolio / factor_sum
                allocation = min(remaining_budget, abs(target[factor]))
                portfolio += factor_portfolio * allocation
                remaining_budget -= allocation
            else:
                print(f"Warning: Unable to allocate to {factor} factor due to zero sum of exposures.")

        # Normalize weights to sum to 1, ignoring NaN values
        portfolio = portfolio.fillna(0)
        portfolio_sum = portfolio.abs().sum()
        if portfolio_sum > 0:
            portfolio = portfolio / portfolio_sum
        else:
            print("Warning: Unable to construct portfolio due to zero sum of weights.")
            return pd.Series(0, index=self.factor_exposures.index)
        return portfolio


if __name__ == "__main__":
    # Fetch historical data using DataFetcher
    data_fetcher = DataFetcher(mode="persistent")
    historical_data = data_fetcher.fetch_historical_data(
        tickers=["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"],
        start_date="2022-01-01",
        end_date="2023-01-01"
    )

    if not historical_data:
        print("No data available. Exiting.")
        exit(1)

    print(f"\nProcessing data for {len(historical_data)} tickers.")

    model = FactorModel(['Market', 'Size', 'Value', 'Momentum'])

    try:
        exposures = model.calculate_factor_exposures(historical_data)
        print("Factor Exposures:\n", exposures)

        returns = model.estimate_factor_returns(historical_data)
        print("\nFactor Returns (last 5 rows):\n", returns.tail())

        target_exposures = {'Market': 1.0, 'Size': -0.2, 'Value': 0.5, 'Momentum': 0.3}
        portfolio = model.construct_portfolio(historical_data, target_exposures)
        print("\nConstructed Portfolio:\n", portfolio)

        # Print some diagnostics
        print("\nData shapes:")
        for ticker, df in historical_data.items():
            print(f"{ticker}: {df.shape}")

        print("\nMissing values:")
        for ticker, df in historical_data.items():
            missing = df.isnull().sum()
            if missing.sum() > 0:
                print(f"{ticker}:\n{missing}")
            else:
                print(f"{ticker}: No missing values")

        print("\nDate ranges:")
        for ticker, df in historical_data.items():
            print(f"{ticker}: {df.index.min()} to {df.index.max()}")

    except Exception as e:
        print(f"An error occurred during model execution: {str(e)}")
        import traceback

        traceback.print_exc()
