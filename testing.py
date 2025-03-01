import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class PortfolioStrategy:
    def __init__(self, risk_free_rate=0.02, rebalance_threshold=0.05, lookback_window=30, volatility_window=20):
        """
        Initialize strategy parameters.
        """
        self.risk_free_rate = risk_free_rate  # Risk-free rate for Sharpe Ratio
        self.rebalance_threshold = rebalance_threshold  # Threshold for rebalancing
        self.lookback_window = lookback_window  # Number of days to look back for optimization
        self.volatility_window = volatility_window  # Window for calculating rolling volatility
        self.historical_data = []  # Stores historical market data for lookback
        self.optimal_weights = None  # Stores the optimal weights
        self.previous_weights = None  # Stores the previous weights for rebalancing

    def sharpe_ratio(self, weights, returns):
        """
        Calculate the Sharpe Ratio for a given set of weights and returns.
        """
        portfolio_return = np.dot(weights, np.mean(returns, axis=0))  # Mean returns across assets
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(np.cov(returns, rowvar=False), weights)))
        return -(portfolio_return - self.risk_free_rate) / portfolio_volatility  # Negative for minimization

    def diversification_score(self, weights):
        """
        Calculate the diversification score using the Herfindahl-Hirschman Index (HHI).
        Lower HHI means better diversification.
        """
        return np.sum(weights**2)

    def optimize_portfolio(self, returns, volatilities):
        """
        Optimize portfolio weights to maximize the Sharpe Ratio while ensuring diversification.
        """
        n_assets = returns.shape[1]
        initial_weights = np.ones(n_assets) / n_assets  # Equal weights as initial guess

        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

        # Bounds: no short selling (0 <= weight <= 1)
        bounds = tuple((0, 1) for _ in range(n_assets))

        # Objective function: Sharpe Ratio + Diversification Score
        def objective(weights):
            sharpe = self.sharpe_ratio(weights, returns)
            diversification = self.diversification_score(weights)
            return sharpe + 0.1 * diversification  # Adjust the diversification penalty as needed

        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x

    def rebalance_portfolio(self, current_weights, optimal_weights, volatilities):
        """
        Rebalance portfolio if weights deviate beyond the threshold or if volatility changes significantly.
        """
        deviation = np.abs(current_weights - optimal_weights)
        if np.any(deviation > self.rebalance_threshold):
            return optimal_weights
        return current_weights

    def allocate(self, market_data: dict) -> np.ndarray:
        """
        Allocate portfolio weights based on market data for the current time step.
        """
        # Append current market data to historical data
        self.historical_data.append(market_data)

        # Keep only the most recent data based on the lookback window
        if len(self.historical_data) > self.lookback_window:
            self.historical_data.pop(0)

        # If we don't have enough data, return equal weights
        if len(self.historical_data) < self.lookback_window:
            n_assets = len(market_data['close'])
            return np.ones(n_assets) / n_assets

        # Extract closing prices and volumes from historical data
        closes = np.array([day['close'] for day in self.historical_data])
        volumes = np.array([day['volume'] for day in self.historical_data])

        # Calculate daily returns
        returns = np.diff(closes, axis=0) / closes[:-1]

        # Calculate rolling volatility
        volatilities = np.std(returns[-self.volatility_window:], axis=0)

        # Optimize portfolio weights
        self.optimal_weights = self.optimize_portfolio(returns, volatilities)

        # Rebalance portfolio
        if self.previous_weights is None:
            self.previous_weights = self.optimal_weights
        else:
            self.previous_weights = self.rebalance_portfolio(self.previous_weights, self.optimal_weights, volatilities)

        return self.previous_weights


def backtest(strategy, historical_data):
    """
    Backtest the strategy on historical data.
    :param strategy: An instance of PortfolioStrategy.
    :param historical_data: List of dictionaries containing market data.
    :return: Cumulative returns and Sharpe Ratio.
    """
    portfolio_values = []
    for market_data in historical_data:
        weights = strategy.allocate(market_data)
        closes = market_data['close']
        daily_returns = np.diff(closes) / closes[:-1]
        portfolio_return = np.dot(weights, daily_returns)
        portfolio_values.append(portfolio_return)

    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + np.array(portfolio_values)) - 1

    # Calculate Sharpe Ratio
    sharpe_ratio = np.mean(portfolio_values) / np.std(portfolio_values)

    return cumulative_returns, sharpe_ratio


def parameter_tuning(historical_data):
    """
    Perform parameter tuning for the PortfolioStrategy.
    :param historical_data: List of dictionaries containing market data.
    :return: DataFrame with results.
    """
    # Define parameter grid
    parameter_grid = {
        'rebalance_threshold': [0.01, 0.05, 0.1],
        'lookback_window': [20, 30, 50],
        'volatility_window': [10, 20, 30]
    }

    results = []

    # Iterate through parameter combinations
    for rebalance_threshold in parameter_grid['rebalance_threshold']:
        for lookback_window in parameter_grid['lookback_window']:
            for volatility_window in parameter_grid['volatility_window']:
                # Initialize strategy with current parameters
                strategy = PortfolioStrategy(
                    rebalance_threshold=rebalance_threshold,
                    lookback_window=lookback_window,
                    volatility_window=volatility_window
                )

                # Backtest the strategy
                cumulative_returns, sharpe_ratio = backtest(strategy, historical_data)

                # Store results
                results.append({
                    'rebalance_threshold': rebalance_threshold,
                    'lookback_window': lookback_window,
                    'volatility_window': volatility_window,
                    'sharpe_ratio': sharpe_ratio,
                    'cumulative_returns': cumulative_returns[-1]  # Final cumulative return
                })

    # Convert results to a DataFrame for analysis
    results_df = pd.DataFrame(results)
    return results_df
    
    
# Function to load and transform JSON data into market_data format
def load_market_data(json_file):
    """
    Load JSON data and convert it into the market_data format.
    :param json_file: Path to the JSON file.
    :return: List of dictionaries, where each dictionary contains market data for a single day.
    """
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Extract dates and stock symbols
    dates = sorted(data['open'].keys())  # Sort dates in ascending order
    symbols = list(data['open'][dates[0]].keys())  # Get stock symbols

    # Initialize market_data as a list of dictionaries
    market_data = []

    # Populate market_data with values from the JSON file
    for date in dates:
        daily_data = {
            'open': np.array([data['open'][date][symbol] for symbol in symbols]),
            'close': np.array([data['close'][date][symbol] for symbol in symbols]),
            'high': np.array([data['high'][date][symbol] for symbol in symbols]),
            'low': np.array([data['low'][date][symbol] for symbol in symbols]),
            'volume': np.array([data['volume'][date][symbol] for symbol in symbols])
        }
        market_data.append(daily_data)

    return market_data

# Main script
if __name__ == "__main__":
    # Load market data from training.json
    historical_data = load_market_data('training.json')

    # Perform parameter tuning
    results_df = parameter_tuning(historical_data)

    # Sort by Sharpe Ratio
    results_df = results_df.sort_values(by='sharpe_ratio', ascending=False)

    # Print top 5 combinations
    print("Top 5 Parameter Combinations:")
    print(results_df.head())

    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['lookback_window'], results_df['sharpe_ratio'], c=results_df['rebalance_threshold'], cmap='viridis')
    plt.colorbar(label='Rebalance Threshold')
    plt.xlabel('Lookback Window')
    plt.ylabel('Sharpe Ratio')
    plt.title('Parameter Tuning Results')
    plt.show()