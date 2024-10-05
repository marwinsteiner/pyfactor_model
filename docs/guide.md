# Project Guide and Instructions

## **1. Project Setup **

First we will need to create a base virtual environment, which is easy to do with
conda: `conda create -n pyfactor_model python=3.12`. Follow the instructions and agree to all suggested package
installs. Next, we'll want to use `git` for version control, so you can head to GitHub and create a new repository
there. I suggest you leave the project public as this project will be an opportunity to learn and show off your ability
to write clean, maintainable, and well-documented code. I use Pycharm, so `cd` into the appropriate directory (
called `*\PycharmProjects`)
and `git clone <link to your repo>`. Note that if you've opted to make your project private, you may need to generate an
access token. More information can be found
here: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens

To recap: we've created a virtual environment, made a new git repository and cloned this git repo to our Pycharm
Projects folder. Activate your conda environment by `conda activate pyfactor_model`. Now, `cd` into the project root
by `cd *\PycharmProjects\pyfactor_model>`. Now, we'll want to initialize poetry by `poetry init`. Click through the
options and specify anything you need.

Next, we'll set up our project with `dynaconf`. First, you'll want to add it as a dependency `poetry add dynaconf`, then
initialize dynaconf with `dynaconf init`. This creates a handful of config files, which we will use for storing
important variables which we may need to protect, such as API keys.

If you haven't already, you can open the project in PyCharm or your IDE of choice. We'll continue assuming you're using
PyCharm.

## **2. Project Structure Setup**

Now that you've set up the project, we have to set up our folder structure, which should be as follows:
```
C:.
│ .gitignore
│ .secrets.json
│ config.py
│ poetry.lock
│ pyproject.toml
│ README.md
│ settings.json
│
├───.idea
│ │ .gitignore
│ │ misc.xml
│ │ modules.xml
│ │ pyfactor_model.iml
│ │ vcs.xml
│ │ workspace.xml
│ │
│ └───inspectionProfiles
│ profiles_settings.xml
│
├───docs
│ guide.md
│ setup.md
│
├───src
│ │ main.py
│ │   __init__.py
│ │
│ ├───data
│ │ data_fetcher.py
│ │       __init__.py
│ │
│ ├───models
│ │ factor_model.py
│ │       __init__.py
│ │
│ └───utils
│ helpers.py
│           __init__.py
│
├───tests
│ test_data_fetcher.py
│ test_factor_model.py
│       __init__.py
│
└───__pycache__
config.cpython-312.pyc
```
Don't worry about the `.idea` or `__pycache__` folders, these are automatically generated. You must create the source
and tests folders and the files contained within.

## **3. Basic File Setup**

We're now ready to start coding! We'll set up some boilerplate code for several of our files to get going.

1. `src/data/data_fetcher.py`

```python
import os
from polygon import RESTClient
from dynaconf import Dynaconf

settings = Dynaconf(settings_files=["settings.toml", ".secrets.toml"])


class DataFetcher:
    def __init__(self):
        self.client = RESTClient(api_key=settings.POLYGON_API_KEY)

    def fetch_historical_data(self, ticker, start_date, end_date):
        # Implement data fetching logic here
        pass
```

2. `src/models/factor_model.py`

```python
import numpy as np
import pandas as pd


class FactorModel:
    def __init__(self, factors):
        self.factors = factors

    def calculate_factor_exposures(self, data):
        # Implement factor exposure calculation
        pass

    def estimate_factor_returns(self, data):
        # Implement factor returns estimation
        pass

    def construct_portfolio(self, data):
        # Implement portfolio construction logic
        pass
```

3. `src/main.py`

```python
import pandas as pd
import numpy as np
from src.data.data_fetcher import DataFetcher
from src.models.factor_model import FactorModel
from src.utils.performance_evaluation import (
    calculate_total_return,
    calculate_annualized_return,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_alpha_beta,
    calculate_information_ratio,
    perform_factor_attribution,
    calculate_turnover
)
from src.utils.visualization import plot_performance_comparison


def main():
    # Fetch historical data
    data_fetcher = DataFetcher(mode="persistent")
    historical_data = data_fetcher.fetch_historical_data(
        tickers=["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"],
        start_date="2020-01-01",
        end_date="2023-01-01"
    )

    # Initialize factor model
    model = FactorModel(['Market', 'Size', 'Value', 'Momentum'])

    # Backtesting parameters
    rebalance_frequency = 'M'  # Monthly rebalancing
    window_size = 252  # One year of trading days

    # Perform backtesting
    portfolio_returns = backtest_strategy(historical_data, model, rebalance_frequency, window_size)

    # Calculate performance metrics
    total_return = calculate_total_return(portfolio_returns)
    annualized_return = calculate_annualized_return(portfolio_returns)
    sharpe_ratio = calculate_sharpe_ratio(portfolio_returns)
    max_drawdown = calculate_max_drawdown(portfolio_returns)
    alpha, beta = calculate_alpha_beta(portfolio_returns, benchmark_returns)
    information_ratio = calculate_information_ratio(portfolio_returns, benchmark_returns)

    # Perform factor attribution
    factor_attribution = perform_factor_attribution(model, portfolio_returns)

    # Calculate turnover
    turnover = calculate_turnover(portfolio_weights_over_time)

    # Compare with benchmarks
    equal_weight_returns = calculate_equal_weight_returns(historical_data)
    market_cap_weight_returns = calculate_market_cap_weight_returns(historical_data)
    # Fetch benchmark (e.g., S&P 500) returns
    benchmark_returns = fetch_benchmark_returns(start_date, end_date)

    # Visualize results
    plot_performance_comparison(
        portfolio_returns,
        equal_weight_returns,
        market_cap_weight_returns,
        benchmark_returns
    )

    # Print performance summary
    print_performance_summary(
        total_return,
        annualized_return,
        sharpe_ratio,
        max_drawdown,
        alpha,
        beta,
        information_ratio,
        factor_attribution,
        turnover
    )


if __name__ == "__main__":
    main()
```

## **4. API Setup**

You'll want to have an existing or create a new account with Polygon, which will serve as our market data provider.
Store your API key, as we'll need this for our project. You'll want to modify your `.secrets.json` file at the project
root to something like this:

```json
{
  "POLYGON_API_KEY": "your_api_key"
}
```

## **5. Let's get Coding**

### `DataFetcher`

First, we'll need some data to work with our model, so let's start working
on `*\pyfactor_model\src\data\data_fetcher.py`.

The `data_fetcher.py` module is responsible for downloading data from Polygon. We make a few imports, `os`for potential
file path operations, though it's not yet used. We also import `typing` fo type hinting, improving
readability, `datetime` for timestamp conversions, and `polygon.RESTClient` is the main class we'll use to interact with
the Polygon API. Finally, `Dynaconf` is used for managing our configuration, including API keys.

We then set up our Dynaconf config, loading our settings from two files, `settings.json` and `.secrets.json`. This
practice is good for security and config management.

Then we initialize the `DataFetcher` class with a Polygon `RESTClient`. We use the API key from `.secrets.json`, which
keeps the key secure and separate from our code. The main method of our `DataFetcher` class is `fetch_historical_data`.
A key consideration when writing this method, is that we need it to be flexible. It accepts multiple tickers, allowing
us to fetch data for several stocks at once. We can specify the date range, timespan (daily, hourly, minute-date), and
whether we want adjusted data or not. The `limit` parameter lets us specify the maximum number of data points returned.

Inside the method, we iterate over each ticker we're retrieving data for. While slightly slow, this allows us more
flexibiltiy, because we can fetch data for each ticker independently, handle errors for individual tickers without
stopping the entire process, and customize the data retrieval parameters for each call. We then process the data into a
more usable format:

```python
historical_data[ticker] = [
    {
        "timestamp": self._convert_timestamp_to_date(bar.timestamp),
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume,
        "vwap": bar.vwap,
        "transactions": bar.transactions
    }
    for bar in aggs
]
```

This list comprehension converts the Polygon data into a list of dictionaries, making it easier to work with in Python.
It also simplifies later conversion to a `pandas` DataFrame.

We also write a helper method `_convert_timestamp_to_date`, which converts the Unix timestamp (in milliseconds) to a
readable date string. It's a private method as it's an internal utility.

To recap: we wrote out the `DataFetcher` class, with which we can now retrieve market data for any valid ticker,
timeframe, and interval from the Polygon API.

We've kept a few key principles in mind:

1. Flexibility: we can fetch data for multiple tickers with various parameters.
2. Error handling: errors can and do happen, but we avoid interrupting the entire data download as a result of this.
3. Data formatting: we format the data as a list of dictionaries, designed to ease later use with `pandas`.
4. Separation of concerns: API keys and other settings are managed separately from the code.
5. Ease of use: the class can be easily imported and used in other parts of the project.

### `*\pyfactor_model\models\factor_model.py`

We'll use the Fama-French Three-Factor Model as a basis and add a momentum factor, resulting in a four-factor model.
This approach is typical of what you'd find across academia and industry. We'll start by outlining our approach for
implementing the factor model:

1. Define the factors: Market, Size (SMB), Value (HML), and Momentum.
2. Calculate factor exposures.
3. Estimate factor returns.
4. Construct a portfolio based on the factor model.

Here's a wireframe of `*\pyfactor_model\models\factor_model.py`:

```python
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.linear_model import LinearRegression
from src.data.data_fetcher import DataFetcher


class FactorModel:
    def __init__(self):
        """
      Initialize the FactorModel with a list of factors.

      Args:
          factors (List[str]): List of factor names.
      """
        pass

    def calculate_factor_exposure(self):
        """
      Calculate factor exposures for each stock.

      Args:
          data (Dict[str, List[Dict[str, Any]]]): Historical data for multiple stocks.

      Returns:
          pd.DataFrame: Factor exposures for each stock.
      """
        pass

    def estimate_factor_returns(self):
        """
      Estimate factor returns using cross-sectional regression.

      Args:
          data (Dict[str, List[Dict[str, Any]]]): Historical data for multiple stocks.

      Returns:
          pd.DataFrame: Estimated factor returns over time.
      """
        pass

    def construct_portfolio(self):
        """
      Construct a portfolio based on target factor exposures.

      Args:
          data (Dict[str, List[Dict[str, Any]]]): Historical data for multiple stocks.
          target_exposures (Dict[str, float]): Target exposures for each factor.

      Returns:
          pd.Series: Portfolio weights for each stock.
      """
        pass

```

