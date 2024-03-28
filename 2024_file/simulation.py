import numpy as np
import torch
from functions import *


# the following function is to generate the stock price path
def stock_price_path(S, sigma, T, dt, mu=0):
    """
    Generate a stock price path using Geometric Brownian Motion.
    
    :param S: Initial stock price.
    :param sigma: Daily volatility.
    :param T: Time of total trading days
    :param dt: Time step in days.
    :param mu: Drift (average return), default is 0.
    :return: A NumPy array representing the stock price path.
    """
    # Number of steps
    N = int(T / dt)
    # Time scale
    t = np.linspace(0, T, N)
    # Generate random normal values
    rand = np.random.normal(0, 1, N)
    # Calculate stock price path
    stock_path = np.zeros(N)
    stock_path[0] = S
    
    for i in range(1, N):
        # GBM formula applied in discrete time steps
        stock_path[i] = stock_path[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rand[i-1])
    
    return stock_path

############################################################################################################

# simulate the shock path
def shock_path(V, T, dt):
    # V: Cholesky factor of the covariance matrix
    # T: Time of total trading days
    # dt: Time step in days.

    # initialize the original shock

    n = V.shape[0]
    init_shock = np.random.multivariate_normal(np.zeros(n), np.eye(n))
    # d(shock) = V * d(W)
    shock = np.zeros((int(T/dt), n))
    
    brownian = np.random.multivariate_normal(np.zeros(n), np.eye(n), int(T/dt))
    shock[0] = np.matmul(V, init_shock)
    for i in range(1, int(T/dt)):
        shock[i] = shock[i-1] + np.matmul(V, brownian[i])

    return shock

############################################################################################################

# the following function is to compute the simulated option prices, delta, and gamma
def option_simulation(V, S, T, dt, K, r, sigma_daily):
    # options: a list of option objects
    # V: Cholesky factor of the covariance matrix
    # S: stock price path
    # T: Time of total trading days
    # dt: Time step in days.

    # K is np array of strike prices
    # r is the risk-free rate
    # sigma_daily is the daily volatility of the underlying asset

    # initialize the option prices, delta, and gamma
    N = int(T / dt)
    n = len(K)

    option_prices = np.zeros((N, n))
    delta = np.zeros((N, n))
    gamma = np.zeros((N, n))

    for i in range(N):
        for j in range(n):
            option = EuropeanCallOption(S[i], K[j], T - i * dt, r, sigma_daily)
            option_prices[i, j] = option.price()
            delta[i, j] = option.delta()
            gamma[i, j] = option.gamma()

    # add shock 
    shock = shock_path(V, T, dt)
    option_prices += shock

    return option_prices, delta, gamma
    


############################################################################################################

# the following function is to compute the number of market orders under policy 
def one_period_trading(policy, t, Q, P, S, dt, A, kappa):
    # policy: a function that takes in t, Q, P, S and returns the action
    # t: current time
    # Q: current inventory (dim: n)
    # P: current price (dim: n)
    # S: current shock (dim: 1)

    # get the action
    action = policy(t, Q, P, S) # dim: 2n

    # the action is a vector of bid and ask for each options 
    # for each option, we need to calculate the number of market orders 
    # let the first n be the bid and the second n be the ask
    n = int(len(action) / 2)
    bid_spread = action[:n]
    ask_spread = action[n:]

    # calculate the number of market orders
    # call the mkt_order function for each option
    # the result is a vector of number of market orders for each option
    # dim: 2n
    buy_orders = np.array([mkt_order(epsilon, dt, A, kappa).detach().numpy() for epsilon in bid_spread])
    sell_orders = np.array([mkt_order(epsilon, dt, A, kappa).detach().numpy() for epsilon in ask_spread])

    return buy_orders, sell_orders

############################################################################################################

# the following function is to compute the entire trading path
def entire_trading(policy, P, S, dt, A, kappa):
    # P is sequence of option prices (dim: N x n)
    # S is the sequence of stock (dim: N)
    # n means the number of options
    # N means the total trading days
    # return the inventory path, buy order path, and sell order path

    # initialize the inventory
    Q = np.zeros(P.shape[1])

    inventory_path = np.zeros((P.shape[0], P.shape[1]))
    buy_order_path = np.zeros((P.shape[0], P.shape[1]))
    sell_order_path = np.zeros((P.shape[0], P.shape[1]))

    for i in range(P.shape[0]):
        curr_time = np.array([i * dt])  
        curr_P = np.array(P[i])
        curr_S = np.array([S[i]])

        buy_orders, sell_orders = one_period_trading(policy, curr_time, Q, curr_P, curr_S, dt, A, kappa)
        Q += buy_orders - sell_orders
        inventory_path[i] = Q
        buy_order_path[i] = buy_orders
        sell_order_path[i] = sell_orders

    return inventory_path, buy_order_path, sell_order_path





    
    


    