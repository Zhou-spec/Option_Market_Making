import numpy as np
import torch
import torch.nn as nn
import gym
from gym import spaces
from scipy.stats import norm
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Define the EuropeanCallOption class
class EuropeanCallOption:
    def __init__(self, S, K, T_days, r, sigma_daily):
        """
        Initialize the European Call Option class with time and volatility in days.
        :param S: Current stock price
        :param K: Strike price
        :param T_days: Time to maturity (in days)
        :param r: Risk-free interest rate (annual)
        :param sigma_daily: Daily volatility of the underlying asset
        """
        self.S = S
        self.K = K
        self.T_days = T_days
        self.T = T_days / 252.0  # Convert days to years assuming 252 trading days per year
        self.r = r
        self.sigma_daily = sigma_daily
        self.sigma = sigma_daily * np.sqrt(252)  # Convert daily volatility to annual volatility

    def d1(self):
        if self.T <= 0 or self.S <= 0:
            return 0
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )

    def d2(self):
        if self.T <= 0 or self.S <= 0:
            return 0
        return self.d1() - self.sigma * np.sqrt(self.T)

    def price(self):
        if self.T <= 0 or self.S <= 0:
            return max(self.S - self.K, 0)
        d1 = self.d1()
        d2 = self.d2()
        return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)

    def delta(self):
        if self.T <= 0 or self.S <= 0:
            return 1.0 if self.S > self.K else 0.0
        d1 = self.d1()
        return norm.cdf(d1)

    def gamma(self):
        if self.T <= 0 or self.S <= 0:
            return 0.0
        d1 = self.d1()
        return norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))

    def theta(self):
        if self.T <= 0 or self.S <= 0:
            return 0.0
        d1 = self.d1()
        d2 = self.d2()
        theta = (
            - (self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
            - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        )
        return theta / 252  # Convert to per-day value

# Function to simulate stock price path
def stock_price_path(S0, sigma, T, dt):
    """
    Generate a stock price path using Geometric Brownian Motion.
    :param S0: Initial stock price.
    :param sigma: Daily volatility.
    :param T: Total time in trading days.
    :param dt: Time step in days.
    :return: A NumPy array representing the stock price path.
    """
    N = int(T / dt)
    rand = np.random.normal(0, 1, N)
    stock_path = np.zeros(N)
    stock_path[0] = S0
    for i in range(1, N):
        dW = rand[i - 1] * np.sqrt(dt)
        stock_path[i] = stock_path[i - 1] * np.exp(sigma * dW)
    return stock_path

# Function to simulate shocks
def shock_path(V, T, dt):
    """
    Simulate the shock path.
    :param V: Cholesky factor of the covariance matrix.
    :param T: Total time in trading days.
    :param dt: Time step in days.
    :return: A NumPy array representing the shock path.
    """
    N = int(T / dt)
    n = V.shape[0]
    shock = np.zeros((N, n))
    rand = np.random.normal(0, 1, size=(N, V.shape[1]))
    for i in range(1, N):
        dW = rand[i] * np.sqrt(dt)
        shock[i] = shock[i - 1] + V @ dW
    return shock

# Function to simulate option prices and Greeks
def option_simulation(V, S, T, dt, K, time, r, sigma):
    """
    Simulate option prices and Greeks.
    :param V: Cholesky factor of the covariance matrix.
    :param S: Stock price path.
    :param T: Total time in trading days.
    :param dt: Time step in days.
    :param K: Strike prices array.
    :param time: Time to maturity array.
    :param r: Risk-free rate.
    :param sigma: Daily volatility of the underlying asset.
    :return: option_prices, delta, gamma, theta arrays.
    """
    N = int(T / dt)
    n_options = len(K)
    option_prices = np.zeros((N, n_options))
    delta = np.zeros((N, n_options))
    gamma = np.zeros((N, n_options))
    theta = np.zeros((N, n_options))

    shocks = shock_path(V, T, dt)

    for i in range(N):
        for j in range(n_options):
            tau = time[j] - i * dt
            if tau <= 0:
                tau = 1e-6  # Avoid zero time to maturity
            option = EuropeanCallOption(S[i], K[j], tau, r, sigma)
            option_price = option.price() + shocks[i, j]
            option_prices[i, j] = option_price
            delta[i, j] = option.delta()
            gamma[i, j] = option.gamma()
            theta[i, j] = option.theta()

    return option_prices, delta, gamma, theta

# Function for market orders
def mkt_order(epsilon, dt, A, kappa):
    """
    Simulate market orders based on spread.
    :param epsilon: Spread array for each option.
    :param dt: Time step.
    :param A: Arrival rate parameter array.
    :param kappa: Sensitivity parameter array.
    :return: Number of market orders for each option.
    """
    arrival_rates = A * np.exp(-kappa * epsilon)
    expected_orders = arrival_rates * dt
    orders = np.random.poisson(expected_orders)
    return orders


# Define the custom Gym environment
class OptionMarketMakingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, paras):
        super(OptionMarketMakingEnv, self).__init__()
        # Define action and observation space
        self.paras = paras
        n_options = len(self.paras.K)
        self.n_options = n_options

        # Action space: bid and ask spreads for each option
        self.action_space = spaces.Box(
            low=np.array([self.paras.bid_ranges[i][0] for i in range(n_options)] +
                         [self.paras.ask_ranges[i][0] for i in range(n_options)]),
            high=np.array([self.paras.bid_ranges[i][1] for i in range(n_options)] +
                          [self.paras.ask_ranges[i][1] for i in range(n_options)]),
            dtype=np.float32
        )

        # Observation space: time, inventory, option prices, Greeks, stock price
        obs_size = 1 + 5 * n_options + 1  # Observation space: time, inventory, option prices, dp_dt, dp_dS, d2p_dS2, stock price
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
        self.reset()

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.done = False
        self.market_making_profits = 0.0
        n_options = self.n_options
        self.Q = np.zeros(n_options)
        self.S = stock_price_path(self.paras.S0, self.paras.sigma, self.paras.T, self.paras.dt)
        self.P, self.all_delta, self.all_gamma, self.all_theta = option_simulation(
            self.paras.V, self.S, self.paras.T, self.paras.dt,
            self.paras.K, self.paras.time, self.paras.r, self.paras.sigma
        )
        self.dp_dt = self.all_theta[self.current_step]
        self.dp_dS = self.all_delta[self.current_step]
        self.d2p_dS2 = self.all_gamma[self.current_step]
        self.dP_t = np.zeros(n_options)
        self.state = self.get_state()
        return self.state

    def step(self, action):
        # Ensure action is within bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)

        n_options = self.n_options
        t = self.current_step * self.paras.dt
        curr_S = self.S[self.current_step]

        # Split action into bid and ask spreads
        bid_spread = action[:n_options]
        ask_spread = action[n_options:]

        # Get A_i and kappa_i for buy and sell orders
        A_plus = self.paras.A_plus  # Array of A_i^+ for each option
        kappa_plus = self.paras.kappa_plus
        A_minus = self.paras.A_minus
        kappa_minus = self.paras.kappa_minus

        # Calculate buy and sell orders
        buy_orders = mkt_order(bid_spread, self.paras.dt, A_plus, kappa_plus)
        sell_orders = mkt_order(ask_spread, self.paras.dt, A_minus, kappa_minus)

        # Update inventory
        self.Q += buy_orders - sell_orders

        # Update option prices
        self.update_option_prices()

        # Compute immediate reward
        immediate_reward = self.compute_reward(
            buy_orders, sell_orders, bid_spread, ask_spread
        )

        # Check if the episode is done
        self.current_step += 1
        if self.current_step >= len(self.S) - 1:
            self.done = True

        # Terminal reward
        if self.done:
            final_reward = -self.paras.psi0 * np.sum((self.P[self.current_step - 1] * self.Q) ** 2)
            total_reward = immediate_reward + final_reward
            next_state = self.state  # No next state when done
        else:
            total_reward = immediate_reward
            next_state = self.get_state()

        info = {}

        self.state = next_state

        return next_state, total_reward, self.done, info

    def update_option_prices(self):
        self.dp_dt = self.all_theta[self.current_step]
        self.dp_dS = self.all_delta[self.current_step]
        self.d2p_dS2 = self.all_gamma[self.current_step]

    def get_state(self):
        t = np.array([self.current_step * self.paras.dt])
        curr_P = self.P[self.current_step]
        state = np.concatenate((
            t,
            self.Q,
            curr_P,
            self.dp_dt,
            self.dp_dS,
            self.d2p_dS2,
            np.array([self.S[self.current_step]])
        ))
        return state

    def compute_reward(self, buy_orders, sell_orders, bid_spread, ask_spread):
        # Immediate profit from spreads
        spread_profit = np.dot(bid_spread, buy_orders) + np.dot(ask_spread, sell_orders)

        # Profit from holding inventory
        if self.current_step >= 1:
            S_prev = self.S[self.current_step - 1]
        else:
            S_prev = self.S[self.current_step]
        inventory_profit = np.dot(
            self.Q,
            self.dp_dt * self.paras.dt +
            0.5 * self.paras.sigma ** 2 * S_prev ** 2 * self.d2p_dS2 * self.paras.dt
        )

        # Risk term (noise)
        dW = np.random.normal(0, np.sqrt(self.paras.dt), size=self.paras.V.shape[1])
        risk_term = np.dot(self.Q, self.paras.V @ dW)

        # Total immediate reward
        dX_t = spread_profit + inventory_profit + risk_term

        return dX_t

    def render(self, mode='human'):
        pass

# Custom feature extractor (if needed)
class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(n_input, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.net(observations)
