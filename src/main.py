from src.data.data_fetcher import DataFetcher
from src.models.factor_model import FactorModel
from src.utils.data_preparation import prepare_returns_data, fetch_benchmark_returns, load_snp_constituents
from src.utils.portfolio_construction import construct_equal_weight_portfolio, construct_market_cap_weight_portfolio
from src.utils.backtesting import backtest_strategy
from src.utils.performance_evaluation import PortfolioPerformance, calculate_turnover, perform_factor_attribution
from src.utils.visualization import plot_cumulative_returns, plot_drawdown, plot_factor_attribution

from pathlib import Path
import sys

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)


def main():
    # 1. Data Preparation
    data_fetcher = DataFetcher(mode='persistent')

    # Define date range
    start_date = '2022-01-01'
    end_date = '2024-10-11'

    # Define stock tickers and benchmark ticker
    snp_weights, stock_tickers = load_snp_constituents()
    benchmark_ticker = 'SPY'

    # Fetch data for stocks
    historical_data = data_fetcher.fetch_historical_data(
        tickers=stock_tickers,
        start_date=start_date,
        end_date=end_date
    )

    if not historical_data:
        print('No data available for stocks. Exiting...')
        exit(1)

    print(f'Processing data for {len(historical_data)} stocks.')

    # Prepare returns data for stocks
    returns_data = prepare_returns_data(historical_data)

    # Fetch benchmark returns
    benchmark_returns = fetch_benchmark_returns(start_date, end_date, benchmark_ticker)

    # 2. Model Initialization
    model = FactorModel(['Market', 'Size', 'Value', 'Momentum'], benchmark_ticker=benchmark_ticker)

    # 3. Backtesting
    portfolio_returns, portfolio_weights = backtest_strategy(
        historical_data,
        model,
        rebalance_frequency='ME',  # Monthly rebalancing
        window_size=252  # One year of trading days
    )

    # Align benchmark returns with portfolio returns
    aligned_benchmark_returns = benchmark_returns.reindex(portfolio_returns.index)

    # 4. Benchmark Portfolios
    equal_weight_returns = construct_equal_weight_portfolio(returns_data).reindex(portfolio_returns.index)
    market_cap_weight_returns = construct_market_cap_weight_portfolio(historical_data).reindex(portfolio_returns.index)

    # 5. Performance Evaluation
    factor_portfolio_performance = PortfolioPerformance(portfolio_returns, aligned_benchmark_returns)
    equal_weight_performance = PortfolioPerformance(equal_weight_returns, aligned_benchmark_returns)
    market_cap_performance = PortfolioPerformance(market_cap_weight_returns, aligned_benchmark_returns)

    # Calculate turnover
    turnover = calculate_turnover(portfolio_weights)

    # Perform factor attribution
    factor_returns = model.factor_returns
    factor_exposures = model.factor_exposures
    attribution = perform_factor_attribution(factor_returns, factor_exposures, portfolio_returns)

    # 6. Results Presentation
    print('\nFactor Model Portfolio Performance:')
    print(factor_portfolio_performance.summary())
    print(f'\nPortfolio Turnover: {turnover:.2%}')

    print('\nEqual-Weight Portfolio Performance:')
    print(equal_weight_performance.summary())

    print('\nMarket-Cap-Weight Portfolio Performance:')
    print(market_cap_performance.summary())

    print('\nFactor Attribution:')
    print(attribution)

    print('\nFactor Portfolio Performance:')
    print(factor_portfolio_performance.summary())

    print('\nEqual Weight Portfolio Performance:')
    print(equal_weight_performance.summary())

    print('\nMarket-Cap Weight Portfolio Performance:')
    print(market_cap_performance.summary())

    # 7. Visualization
    plot_cumulative_returns({
        'Factor Model': portfolio_returns,
        'Equal-Weight': equal_weight_returns,
        'Market-Cap-Weight': market_cap_weight_returns,
        'Benchmark': aligned_benchmark_returns
    })

    if not portfolio_returns.empty and portfolio_returns.dtype == float:
        plot_drawdown(portfolio_returns)
    else:
        print("Unable to plot drawdown: portfolio returns are empty or contain non-numeric values")

    plot_factor_attribution(attribution)


if __name__ == "__main__":
    main()


# TODO: identify why cumulative returns are 1.0 until 2024 and only then change for the factor model. Suspicion is
#  that it has to do with not being able to calculate the factor exposures prior to this date.

# Good resource to find updated list of components: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies

# TODO:
#  """
#  Factor Exposure Calculation:
#   The log shows that the calculated exposures shape is (0, 4), which means no factor exposures are being calculated. This is likely the root cause of the flat performance.
#   Data Handling:
#   The log shows that after dropping NaN values, the returns shape becomes (0, 503), meaning all data is being removed. This suggests there might be an issue with how missing or invalid data is being handled.
#   Market Returns:
#   The market returns shape is (0,), which is consistent with the previous point and indicates no valid market returns are being calculated.
#   Potential causes and solutions:
#   Data Quality:
#   Check if the input data contains a large number of NaN or invalid values.
#   Review the data preprocessing steps to ensure data is being cleaned appropriately without removing all values.
#   Date Alignment:
#   Ensure that the dates for all stocks are properly aligned. Misaligned dates could cause the dropna() function to remove all data.
#   Factor Exposure Calculation Logic:
#   Review the factor exposure calculation method to ensure it's not too restrictive in filtering out data.
#   Consider using a rolling window approach for calculating exposures instead of requiring all data to be available from the start.
#   Error Handling:
#   Implement more robust error handling in the factor exposure calculation to prevent empty results from propagating through the system.
#   Lookback Period:
#   Check if the lookback period for calculating exposures is appropriate. If it's too long, it might not have enough data at the beginning of the backtest.
#   Data Frequency:
#   Verify that the data frequency matches what the model expects. For example, if the model expects daily data but receives monthly data, it might cause issues.
#   Factor Definition:
#   Review how each factor (Market, Size, Value, Momentum) is defined and calculated. Ensure they are appropriate for the data you have.
#   Next steps:
#   Add more detailed logging in the factor exposure calculation method to understand why all data is being dropped.
#   Implement a check to print out a sample of the input data before and after the dropna() operation to see what's being removed.
#   Consider implementing a more gradual approach to factor exposure calculation, allowing for partial data availability, especially at the beginning of the backtest period.
#   Review the backtesting strategy to ensure it's handling the initial period correctly when full factor exposures might not be available.
#  """