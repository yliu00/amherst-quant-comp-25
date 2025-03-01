import numpy as np
from scipy.optimize import minimize

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