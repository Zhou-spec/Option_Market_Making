import numpy as np
from functions import *
from policy_simulation import * 
import torch
import torch.optim as optim 
import random


############################################################################################################
'''
This file contains 5 functions:
1. stock_price_path(S, sigma, T, dt, mu=0) this is to generate the stock price path
2. shock_path(V, T, dt) this is to simulate the shock path (not directly used in the main file)
3. option_simulation(V, S, T, dt, K, r, sigma_daily) this is to compute the simulated option prices, delta, and gamma
4. one_period_trading(policy, t, Q, P, S, dt, A, kappa) this is to compute the number of market orders under policy (not directly used in the main file)
5. entire_trading(policy, P, S, dt, A, kappa) this is to compute the entire trading path
'''

############################################################################################################


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
    
    stock_path = np.array(stock_path)
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
def option_simulation(V, S, T, dt, K, time, r, sigma_daily):
    # options: a list of option objects
    # V: Cholesky factor of the covariance matrix
    # S: stock price path
    # T: Time of total trading days
    # dt: Time step in days.

    # K is np array of strike prices
    # time is np array of time to maturity
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
            option = EuropeanCallOption(S[i], K[j], time[j] - i * dt, r, sigma_daily)
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
    # policy: a trading policy object 
    # t: current time
    # Q: current inventory (dim: n)
    # P: current price (dim: n)
    # S: current shock (dim: 1)

    # get the action
    bid_spread, ask_spread = policy.policy_act(t, Q, P, S)

    # calculate the number of market orders
    # call the mkt_order function for each option
    # the result is a vector of number of market orders for each option
    # dim: 2n
    buy_orders = np.array([mkt_order(epsilon, dt, A, kappa) for epsilon in bid_spread])
    sell_orders = np.array([mkt_order(epsilon, dt, A, kappa) for epsilon in ask_spread])

    return buy_orders, sell_orders, bid_spread, ask_spread

############################################################################################################

# the following function is to compute the entire trading path
def entire_trading(policy, P, S, dt, A, kappa):
    # policy: a trading policy object
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

        buy_orders, sell_orders, bid_spread, ask_spread = one_period_trading(policy, curr_time, Q, curr_P, curr_S, dt, A, kappa)
        Q += buy_orders - sell_orders
        inventory_path[i] = Q
        buy_order_path[i] = buy_orders
        sell_order_path[i] = sell_orders



    return inventory_path, buy_order_path, sell_order_path


# final_version of entire_trading to calculate the profits
def entire_trading_final(policy, P, S, dt, A, kappa, phi):
    # policy: a trading policy object
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

    # market making profits
    market_making_profits = 0

    for i in range(P.shape[0]):
        curr_time = np.array([i * dt])  
        curr_P = np.array(P[i])
        curr_S = np.array([S[i]])

        buy_orders, sell_orders, bid_spread, ask_spread = one_period_trading(policy, curr_time, Q, curr_P, curr_S, dt, A, kappa)
        Q += buy_orders - sell_orders
        inventory_path[i] = Q
        buy_order_path[i] = buy_orders
        sell_order_path[i] = sell_orders

        # compute the culmulate profits
        market_making_profits += np.dot(bid_spread, buy_orders) + np.dot(ask_spread, sell_orders)
        
    # final inventory value 
    final_inventory_value = np.sum((inventory_path[-1] * P[-1])**2)
    reward = market_making_profits - phi * final_inventory_value
        

    return inventory_path, buy_order_path, sell_order_path, reward



############################################################################################################

# the following implement the loss function calculation for one path 
# once we have the a path of trading, we can compute the loss function 

def temporal_diff_error(new_value_net, policy, t, Q, P, S, t_prime, Q_prime, P_prime, S_prime, dt, delta, gamma):
    # new_value_net is the value network to estimate the value function of the current policy
    # policy is the trading policy object
    # (t, Q, P, S) is the current state
    # (t', Q', P', S') is the next state
    # dt is the time step
    # delta is the delta of the option
    # gamma is the gamma of the option

    # the difference between the value function
    # concatenate (t, Q, P, S) and (t', Q', P', S')
    state = np.concatenate([t, Q, P, S])
    state_prime = np.concatenate([t_prime, Q_prime, P_prime, S_prime])

    # transfer the state to tensor
    state = torch.tensor(state, dtype=torch.float32)
    state_prime = torch.tensor(state_prime, dtype=torch.float32)

    # check device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = state.to(device)
    state_prime = state_prime.to(device)

    value_diff = new_value_net.forward(state_prime) - new_value_net.forward(state)
    value_diff /= dt

    # compute the expected reward 
    profits = policy.expected_profits(t, Q, P, S)

    # compute the option related penalty
    option_penalty = np.dot(Q, (delta + gamma))

    # compute the entropy 
    entropy = policy.policy_entropy(t, Q, P, S)


    error = profits + value_diff + option_penalty + entropy

    return (error**2) * dt


# this is to compute the loss function of one simulated trajectories
# new_value_net is to estimate the value function of the current policy
# new_value_net is the only network to be trained    
def martingale_loss(new_value_net, policy, stock_price_path, options_price_path, options_delta_path, options_gamma_path, inv_path, dt, T, phi):
    # new_value_net is the value network to estimate the value function of the current policy
    # policy is the trading policy object
    # stock_price_path is the path of stock price
    # options_price_path is the path of option price
    # options_delta_path is the path of option delta
    # options_gamma_path is the path of option gamma
    # inv_path is the path of inventory
    # dt is the time step
    # T is the maturity of the option

    N = int(T / dt)
    loss = 0

    for i in range(N - 2):
        t = np.array([i * dt])
        t_prime = np.array([(i + 1) * dt])

        Q = inv_path[i]
        Q_prime = inv_path[i + 1]

        P = options_price_path[i]
        P_prime = options_price_path[i + 1]

        S = np.array([stock_price_path[i]])
        S_prime = np.array([stock_price_path[i + 1]])

        delta = options_delta_path[i]
        gamma = options_gamma_path[i]

        loss += temporal_diff_error(new_value_net, policy, t, Q, P, S, t_prime, Q_prime, P_prime, S_prime, dt, delta, gamma)

    # for the final stage loss, we have to use the terminal value
    final_inv = inv_path[N - 1]
    final_option_price = options_price_path[N - 1]  
    # compute the final value
    final_value = phi * np.sum((final_inv * final_option_price)**2) 

    # previous state 
    time = np.array([T - dt])
    S_previous = np.array([stock_price_path[N - 2]])
    previous_state = np.concatenate([time, inv_path[N - 2], options_price_path[N - 2], S_previous])

    previous_state = torch.tensor(previous_state, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    previous_state = previous_state.to(device)

    previous_value = new_value_net.forward(previous_state)
    
    final_profits = policy.expected_profits(time, inv_path[N - 2], options_price_path[N - 2], S_previous)
    final_option_penalty = np.dot(inv_path[N - 2], (options_delta_path[N - 2] + options_gamma_path[N - 2]))
    final_entropy = policy.policy_entropy(time, inv_path[N - 2], options_price_path[N - 2], S_previous)
    final_value_diff = (final_value - previous_value) / dt  

    loss += (final_profits + final_value_diff + final_option_penalty + final_entropy + final_value)**2 * dt

    return loss


############################################################################################################
# the following is to compute the loss function of one simulated trajectories
def value_iteration_one_epoch(new_value_net, policy, paras):
    # policy: TradingPolicy object
    # paras: TradingParameters object (there are too many parameters to pass
    
    ################################################################################################
    # this function is to simulate trading for one path (stock price, option price path)

    loss = 0

    stock_path = stock_price_path(paras.S0, paras.sigma, paras.T, paras.dt)
    option_price_path, delta_path, gamma_path = option_simulation(paras.V, stock_path, paras.T, paras.dt, paras.K, paras.time, paras.r, paras.sigma)
    for i in range(paras.epoch):
        inv, buy, sell = entire_trading(policy, option_price_path, stock_path, paras.dt, paras.A, paras.kappa)
        one_path_loss = martingale_loss(new_value_net, policy, stock_path, option_price_path, delta_path, gamma_path, inv, paras.dt, paras.T, paras.phi)
        loss += one_path_loss

    return loss / paras.epoch


############################################################################################################

# the following is the final function to get the policy_iteration, and value_estimation 
def value_estimation(value_net_to_train, policy, paras, optimizer, train_data, num_epoch = 10, lr = 0.01):
    loss_path = []
    for i in range(num_epoch):
        loss = 0
        # randomly sample the training data
        stock_path = random.choice(train_data.stock_path_repo)
        option_price_path = random.choice(train_data.option_price_path_repo)
        delta_path = random.choice(train_data.delta_path_repo)
        gamma_path = random.choice(train_data.gamma_path_repo)

        
        optimizer.zero_grad()

        for i in range(paras.epoch):
            inv, buy, sell = entire_trading(policy, option_price_path, stock_path, paras.dt, paras.A, paras.kappa)
            one_path_loss = martingale_loss(value_net_to_train, policy, stock_path, option_price_path, delta_path, gamma_path, inv, paras.dt, paras.T, paras.phi)
            loss += one_path_loss

        loss /= paras.epoch
        loss.backward()
        optimizer.step()
        print("loss: ", loss.item())
        loss_path.append(loss.item())

    return loss_path


############################################################################################################
# we are not going to use it
def policy_iteration(initial_policy, paras, device, num_iter = 5, num_epoch = 5, lr = 0.01):
    
    # this is the current policy (TradingPolicy object)
    curr_policy = initial_policy   
    # define a value_network to be trained
    n = len(paras.K)
    value_network = Net(n) 

    for i in range(num_iter):
        print("iteration: ", i)
        value_network = Net(n)
        value_network.to(device)
        if i != 0:
            value_network.load_state_dict(value_network.state_dict())
    
        # estimate the value network
        value_estimation(value_network, curr_policy, paras, device, num_epoch, lr)
        # update the policy
        curr_policy = TradingPolicy(value_network, paras.gamma, paras.A, paras.kappa, paras.bid_ranges, paras.ask_ranges)

    return curr_policy


    

    ####################################################################################################

# the following function is to compute the reward of the policy

def entire_trading_final(policy, P, S, dt, A, kappa, phi, option_gamma, option_theta, sigma, gamma):
    # policy: a trading policy object
    # P is sequence of option prices (dim: N x n)
    # S is the sequence of stock (dim: N)
    # n means the number of options
    # N means the total trading days
    # return the inventory path, buy order path, and sell order path
    # option_gamma is the gamma of the option
    # option_theta is the theta of the option
    # sigma is stock volatility

    # initialize the inventory
    Q = np.zeros(P.shape[1])

    inventory_path = np.zeros((P.shape[0], P.shape[1]))
    buy_order_path = np.zeros((P.shape[0], P.shape[1]))
    sell_order_path = np.zeros((P.shape[0], P.shape[1]))

    # market making profits
    market_making_profits = 0
    option_value = 0
    policy_entropy_value = 0

    for i in range(P.shape[0]):
        curr_time = np.array([i * dt])  
        curr_P = np.array(P[i])
        curr_S = np.array([S[i]])

        buy_orders, sell_orders, bid_spread, ask_spread = one_period_trading(policy, curr_time, Q, curr_P, curr_S, dt, A, kappa)
        Q += buy_orders - sell_orders
        inventory_path[i] = Q
        buy_order_path[i] = buy_orders
        sell_order_path[i] = sell_orders

        # compute the culmulate profits
        market_making_profits += np.dot(bid_spread, buy_orders) + np.dot(ask_spread, sell_orders)
        
        # option_theta, and option_gamma is 
        option_value += np.sum(np.dot(Q, option_gamma[i])) + 0.5 * sigma * np.sum(np.dot(Q, option_theta[i]))
        
        # policy entropy
        policy_entropy_value += policy.policy_entropy(curr_time, Q, curr_P, curr_S)
        
    
    final_inventory_value = -phi * np.sum((inventory_path[-1] * P[-1])**2)

    reward = market_making_profits + option_value * dt + gamma * policy_entropy_value * dt
        

    return inventory_path, buy_order_path, sell_order_path, reward


def option_simulation_final(V, S, T, dt, K, time, r, sigma_daily):
    # options: a list of option objects
    # V: Cholesky factor of the covariance matrix
    # S: stock price path
    # T: Time of total trading days
    # dt: Time step in days.

    # K is np array of strike prices
    # time is np array of time to maturity
    # r is the risk-free rate
    # sigma_daily is the daily volatility of the underlying asset

    # initialize the option prices, delta, and gamma
    N = int(T / dt)
    n = len(K)

    option_prices = np.zeros((N, n))
    delta = np.zeros((N, n))
    gamma = np.zeros((N, n))
    theta = np.zeros((N, n))
    for i in range(N):
        for j in range(n):
            option = EuropeanCallOption(S[i], K[j], time[j] - i * dt, r, sigma_daily)
            option_prices[i, j] = option.price()
            delta[i, j] = option.delta()
            gamma[i, j] = option.gamma()
            theta[i, j] = option.theta()

    # add shock 
    shock = shock_path(V, T, dt)
    option_prices += shock

    return option_prices, delta, gamma, theta


def reward_distribution(policy, paras, num_path):
    # num_path is the number of simulated trading of the current policy we are conducting

    terminal_wealths = [] 
    inventory_paths = []
    for i in range(num_path):
        stock_path = stock_path(paras.S0, paras.sigma, paras.T, paras.dt)
        option_price, option_delta, option_gamma, option_theta = option_simulation_final(paras.V, stock_path, paras.T, paras.dt, paras.K, paras.time, paras.r, paras.sigma)
        inv, buy, sell, reward = entire_trading_final(policy, option_price, stock_path, paras.dt, paras.A, paras.kappa, paras.phi, option_gamma, option_theta, paras.sigma, paras.gamma)
        # after getting the inventory, we can calculate the reward
        inventory_paths.append(inv)
        terminal_wealths.append(reward)

    return inventory_paths, terminal_wealths