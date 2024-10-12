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
        rebalance_frequency='M',  # Monthly rebalancing
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

    plot_drawdown(portfolio_returns)
    plot_factor_attribution(attribution)


if __name__ == "__main__":
    main()
