# Project Guide and Instructions

The aim of this guide and the instructions herein are for you to replicate this project in an environment of your own.
If you just want to see it working and you've cloned this repo, simply follow the "Quick Start" instructions contained
in the Readme. More generally, the Readme also contains some information on what multifactor models are and why we might
care. So let's get cracking!

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
│   .gitignore
│   .secrets.json
│   config.py
│   poetry.lock
│   pyproject.toml
│   README.md
│   settings.json
│
├───.idea
│   │   .gitignore
│   │   misc.xml
│   │   modules.xml
│   │   other.xml
│   │   pyfactor_model.iml
│   │   vcs.xml
│   │   workspace.xml
│   │
│   └───inspectionProfiles
│           profiles_settings.xml
│
├───docs
│   │   guide.md
│   │   setup.md
│   │
│   └───img
│           cumulative_returns.png
│           drawdown.png
│           factor_attributes.png
│
├───src
│   │   main.py
│   │   __init__.py
│   │
│   ├───data
│   │   │   data_fetcher.py
│   │   │   __init__.py
│   │   │
│   │   ├───data_download
│   │   │       AAPL.csv
│   │   │       AMZN.csv
│   │   │       GOOGL.csv
│   │   │       MSFT.csv
│   │   │       NVDA.csv
│   │   │       SPY.csv
│   │   │
│   │   └───__pycache__
│   │           data_fetcher.cpython-312.pyc
│   │           __init__.cpython-312.pyc
│   │
│   ├───models
│   │   │   factor_model.py
│   │   │   __init__.py
│   │   │
│   │   └───__pycache__
│   │           factor_model.cpython-312.pyc
│   │           __init__.cpython-312.pyc
│   │
│   ├───utils
│   │   │   backtesting.py
│   │   │   data_preparation.py
│   │   │   path_utils.py
│   │   │   performance_evaluation.py
│   │   │   portfolio_construction.py
│   │   │   visualization.py
│   │   │   __init__.py
│   │   │
│   │   └───__pycache__
│   │           backtesting.cpython-312.pyc
│   │           data_preparation.cpython-312.pyc
│   │           path_utils.cpython-312.pyc
│   │           performance_evaluation.cpython-312.pyc
│   │           portfolio_construction.cpython-312.pyc
│   │           visualization.cpython-312.pyc
│   │           __init__.cpython-312.pyc
│   │
│   └───__pycache__
│           __init__.cpython-312.pyc
│
├───tests
│   │   test_data_fetcher.py
│   │   test_factor_model.py
│   │   __init__.py
│   │
│   └───__pycache__
│           test_data_fetcher.cpython-312.pyc
│           test_factor_model.cpython-312.pyc
│           __init__.cpython-312.pyc
│
└───__pycache__
        config.cpython-312.pyc
```

Don't worry about the `.idea` or `__pycache__` folders, these are automatically generated. You must create the source
and tests folders and the files contained within.

## **3. API Setup**

You'll want to have an existing or create a new account with Polygon, which will serve as our market data provider.
Store your API key, as we'll need this for our project. You'll want to modify your `.secrets.json` file at the project
root to something like this:

```json
{
  "POLYGON_API_KEY": "your_api_key"
}
```

## **4. Let's get Coding**

### `*\pyfactor_model\src\data\data_fetcher.py`

The overarching goal of the project is to let us implement a multifactor model, which requires access to values for
variables such as returns, which we can only calculate if we have a time series, in other words, market data. So, it's
integral to our project, that we design a script which is responsible strictly for downloading/fetching market data.
That's where `*\pyfactor_model\src\data\data_fetcher.py` comes in.

The `DataFetcher` class is integral to the functioning of our factor model -- if we don't have time series to use, we
can not test our model. The `DataFetcher` should let us fetch historical stock time series from our data provider,
Polygon. We design our implementation such that we have two usage modes: a 'persistent' and an 'on-demand' mode,
meaning, that wherever the class is instantiated, the use-case can dictate whether it makes more sense to download the
time series once, store them locally and access them locally to fit our model, or there might be a use-case where you
need this project to be extremely portable and you don't care how many API credits it will cost you, meaning you can
fetch data from the API on-demand each time you run it. Further, the 'persistent' variant is designed so as to fetch
data from the API when it can't find it locally. So, if you've tested the script with tickers A, B, and C, but you want
to know how it would fare with A, B, C, and D, then the persistent mode will notice A, B, and C exist and only fetch D.
The 'on-demand' mode would fetch all tickers on each new execution.

A final design considerationis that we need to provide a consistent interface for data retrieval. This means it needs to
be extremely easy to retrieve data from another script in our project. What good is a script if it isn't usable?

To reiterate, this class serves as the data foundation for our factor model, ensuring that we have reliable and
efficient access to the historical stock data needed for our analysis.

As mentioned before, it supports dual-mode operation, giving use-case flexibility. We use the Polygon RESTClient, which
negates any requirement to use requests, essentially simplifying our interactions with their API. In 'persistent' mode,
we save the time series as CSV files for later use, reducing API calls and speeding up any subsequent analysis. We also
use Dynaconf for managing API keys, which is good practice.

Let's get started!

```python
from polygon import RESTClient
from dynaconf import Dynaconf
from typing import List, Dict
from datetime import datetime
import pandas as pd
from src.utils.path_utils import get_project_root

settings = Dynaconf(settings_files=['settings.json', '.secrets.json'])
```

We import the polgygon `RESTClient` for interactions with the Polygon API, import Dynaconf for handing our API key (
which you should now have set up), we grab `List and Dict` from `typing`, which are for type hinting. Again, this is
good practice and is meant to make the code more maintainable and easier to understand. We also import `datetime` to
deal with times, `pandas` for working with dataframes and a helper module to get the project root. We then let Dynaconf
know which files contain our config.

```python
class DataFetcher:
    def __init__(self, mode: str = "on_demand"):
        self.client = RESTClient(api_key=settings.POLYGON_API_KEY)
        self.mode = mode
        self.data_dir = get_project_root() / "src" / "data" / "data_download"
        if self.mode == "persistent":
            self.data_dir.mkdir(exist_ok=True)
```

Next, we define the DataFetcher class along with its class constructor, which initializes the Polygon API client, sets
the operation mode, and prepares the data directory for persistent storage if required.

```python
def fetch_historical_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        timespan: str = "day",
        limit: int = 50000,
        adjusted: bool = True
) -> Dict[str, pd.DataFrame]:
    historical_data = {}

    if self.mode == "persistent":
        # Read all CSV files in the data directory
        for csv_file in self.data_dir.glob("*.csv"):
            ticker = csv_file.stem  # Get filename without extension
            df = pd.read_csv(csv_file, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            historical_data[ticker] = df
            print(f"Loaded data for {ticker} from file")

        # Fetch data for tickers not present in the directory
        missing_tickers = set(tickers) - set(historical_data.keys())
        self._fetch_and_save_tickers(missing_tickers, start_date, end_date, timespan, limit, adjusted,
                                     historical_data)
    else:
        # On-demand mode: fetch data for all requested tickers
        self._fetch_and_save_tickers(tickers, start_date, end_date, timespan, limit, adjusted, historical_data)
    return historical_data
```

Next, we define our main method, `fetch_historical_data`, which serves as our primary interface for retrieving
historical
data. It handles both persistent and on-demand modes. In persistent mode, it first checks for existing CSV files and
loads data for them. For missing data (in persistent mode) or all data (in on-demand mode), it fetches from the Polygon
API. The method returns a dictionary where keys are strings of the ticker symbols and values are pandas DataFrames
containing the historical data.

```python
    def _fetch_and_save_tickers(self, tickers, start_date, end_date, timespan, limit, adjusted, historical_data):


for ticker in tickers:
    try:
        aggs = self.client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan=timespan,
            from_=start_date,
            to=end_date,
            limit=limit,
            adjusted=adjusted
        )

        if not aggs:
            print(f"No data fetched for {ticker}")
            continue

        df = pd.DataFrame([
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
        ])
        df.set_index('timestamp', inplace=True)

        if self.mode == "persistent":
            df.to_csv(self.data_dir / f"{ticker}.csv")

        historical_data[ticker] = df
        print(f"Successfully fetched data for {ticker}")
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
```

Here we handle the actual API call through the rest client and get aggs. What we get back from the API we turn into a
dataframe with the following columns: `timestamp, open, high, low, close, volume, vwap, and transactions` for every bar
in the aggregates in the period. If in persistent mode, for every dictionary in our list of dictionaries, we use the key
as the filename and the value (dataframe) as the content of the csv file.

We also need to convert the timestamps we get from Polygon from a Unix timestamp to a more useful format like "
YYYY-MM-DD". We do this in the `_convert_timestamp_to_date` (private) helper method:

```python
def _convert_timestamp_to_date(self, timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d')
```

The `DataFetcher` class is designed to be the single point of interaction for data retrieval in the project. We'll use
it in the factor model and the backtesting module, to obtain the necessary historical data. Whenever you can opt to
simplify data management, you generally should.

The rationale behind the dual-mode operation is I want to leave it open for the use-case to determine whether on-demand
data or persistently stored data is required. This is not for me to decide. I chose the pandas DataFrame data structure
because its powerful and fast enough for this use-case. It's also a flexible data structure for time series analysis,
which is crucial for financial modeling. I chose to use Polygon as the data provider because there isn't really anywhere
else to get so much daily closing data for free. They have paid API plans and the way our project is set up means its
trivial to switch out the API key stored in the `.secrets.json` file and make use of the premium features. 

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

