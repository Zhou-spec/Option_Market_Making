# the policy is quite complicated, we need to first compute the numerator of the pdf given (bid, ask)
import numpy as np
import torch


def policy_numerator(ValueNet, bid, ask, t, Q, P, S, penalty, A, kappa):
    # ValueNet is the neural network for the value function
    # bid and ask are the bid and ask spreads
    # t is the current time
    # Q is the inventory (dim n)
    # P is the option price (dim n)
    # S is the stock price (dim 1)
    # penalty is penalty coefficient in the paper (gamma)
    # A: the constant in the rate function
    # kappa: the constant in the rate function

    n = len(Q)
    # compute the rate given bid and ask
    bid_rate = A * np.exp(-kappa * bid)
    ask_rate = A * np.exp(-kappa * ask)
    
    # compute the differential of value function
    bid_value = np.zeros(n)
    ask_value = np.zeros(n)
    for i in range(n):
        # bid_value is V(t, Q + e_i, P, S) - V(t, Q, P, S)
        bid_value[i] = ValueNet.forward(torch.tensor(np.concatenate([t, Q + np.eye(n)[i], P, S])).float()).detach().numpy() - ValueNet.forward(torch.tensor(np.concatenate([t, Q, P, S])).float()).detach().numpy()
        # ask_value is V(t, Q - e_i, P, S) - V(t, Q, P, S)
        ask_value[i] = ValueNet.forward(torch.tensor(np.concatenate([t, Q - np.eye(n)[i], P, S])).float()).detach().numpy() - ValueNet.forward(torch.tensor(np.concatenate([t, Q, P, S])).float()).detach().numpy()

    # compute the numerator of the policy
    numerator = 0
    for i in range(n):
        numerator += (bid_rate[i] * (bid[i] + bid_value[i]))
        numerator += (ask_rate[i] * (ask[i] + ask_value[i])) 
        

    numerator = numerator / penalty
    # here, we don't get the exponential term in the numerator
    # this is to avoid overflow
    return numerator

# compute the distribution of the policy 
def policy_distribution(ValueNet, t, Q, P, S, penalty, A, kappa, bid_range, ask_range):
    # ValueNet: the neural network for the value function
    # t: the time step
    # Q: the current state
    # P: the current price
    # S: the current inventory

    # penalty: the penalty coefficient in the paper (gamma)
    # A: the constant in the rate function
    # kappa: the constant in the rate function

    # bid_range: the range of bid spread
    # ask_range: the range of ask spread

    distribution = np.zeros((len(bid_range), len(ask_range)))
    for i in range(len(bid_range)):
        for j in range(len(ask_range)):
            curr_bid = bid_range[i]
            curr_ask = ask_range[j]

            distribution[i, j] = policy_numerator(ValueNet, curr_bid, curr_ask, t, Q, P, S, penalty, A, kappa)

    # apply softmax to the distribution
    distribution = np.exp(distribution) / np.sum(np.exp(distribution))

    return distribution

# compute the action given the policy 
def policy_act(ValueNet, t, Q, P, S, penalty, A, kappa, bid_range, ask_range):
    # ValueNet: the neural network for the value function
    # t: the time step
    # Q: the current state
    # P: the current price
    # S: the current inventory

    # penalty: the penalty coefficient in the paper (gamma)
    # A: the constant in the rate function
    # kappa: the constant in the rate function

    # bid_range: the range of bid spread
    # ask_range: the range of ask spread

    distribution = policy_distribution(ValueNet, t, Q, P, S, penalty, A, kappa, bid_range, ask_range)
    
    # sample the action from the distribution
    flattened_distribution = distribution.flatten()
    # Sample a single index from the flattened distribution
    sampled_index = np.random.choice(a=flattened_distribution.size, p=flattened_distribution)
    # Convert the sampled index back to 2D index in the distribution array
    i, j = np.unravel_index(sampled_index, distribution.shape)
    # Return the corresponding bid and ask values
    return bid_range[i], ask_range[j]


###########################################################################################################

class TradingPolicy:
    def __init__(self, ValueNet, penalty, A, kappa, bid_range, ask_range):
        """
        Initialize the trading policy with the given parameters.

        :param ValueNet: The neural network model for the value function.
        :param penalty: The penalty coefficient (gamma).
        :param A: The constant in the rate function.
        :param kappa: The constant in the rate function.
        """
        self.ValueNet = ValueNet
        self.penalty = penalty
        self.A = A
        self.kappa = kappa
        self.bid_range = bid_range  
        self.ask_range = ask_range

    def policy_numerator(self, bid, ask, t, Q, P, S):
        # ValueNet is the neural network for the value function
        # bid and ask are the bid and ask spreads
        # t is the current time
        # Q is the inventory (dim n)
        # P is the option price (dim n)
        # S is the stock price (dim 1)
        # penalty is penalty coefficient in the paper (gamma)
        # A: the constant in the rate function
        # kappa: the constant in the rate function

        n = len(Q)
        # compute the rate given bid and ask
        bid_rate = self.A * np.exp(-self.kappa * bid)
        ask_rate = self.A * np.exp(-self.kappa * ask)
        
        # compute the differential of value function
        bid_value = np.zeros(n)
        ask_value = np.zeros(n)
        for i in range(n):
            # bid_value is V(t, Q + e_i, P, S) - V(t, Q, P, S)
            bid_value[i] = self.ValueNet.forward(torch.tensor(np.concatenate([t, Q + np.eye(n)[i], P, S])).float()).detach().numpy() - self.ValueNet.forward(torch.tensor(np.concatenate([t, Q, P, S])).float()).detach().numpy()
            # ask_value is V(t, Q - e_i, P, S) - V(t, Q, P, S)
            ask_value[i] = self.ValueNet.forward(torch.tensor(np.concatenate([t, Q - np.eye(n)[i], P, S])).float()).detach().numpy() - self.ValueNet.forward(torch.tensor(np.concatenate([t, Q, P, S])).float()).detach().numpy()

        # compute the numerator of the policy
        numerator = 0
        for i in range(n):
            numerator += (bid_rate[i] * (bid[i] + bid_value[i]))
            numerator += (ask_rate[i] * (ask[i] + ask_value[i])) 
            
        numerator = numerator / self.penalty
        # here, we don't get the exponential term in the numerator
        # this is to avoid overflow
        return numerator
        

    def policy_distribution(self, t, Q, P, S):
        # ValueNet: the neural network for the value function
        # t: the time step
        # Q: the current state
        # P: the current price
        # S: the current inventory

        # penalty: the penalty coefficient in the paper (gamma)
        # A: the constant in the rate function
        # kappa: the constant in the rate function

        # bid_range: the range of bid spread
        # ask_range: the range of ask spread

        distribution = np.zeros((len(self.bid_range), len(self.ask_range)))
        for i in range(len(self.bid_range)):
            for j in range(len(self.ask_range)):
                curr_bid = self.bid_range[i]
                curr_ask = self.ask_range[j]

                distribution[i, j] = self.policy_numerator(curr_bid, curr_ask, t, Q, P, S)

        # apply softmax to the distribution
        distribution = np.exp(distribution) / np.sum(np.exp(distribution))

        return distribution
    
    def policy_act(self, t, Q, P, S):
        # ValueNet: the neural network for the value function
        # t: the time step
        # Q: the current state
        # P: the current price
        # S: the current inventory

        # penalty: the penalty coefficient in the paper (gamma)
        # A: the constant in the rate function
        # kappa: the constant in the rate function

        # bid_range: the range of bid spread
        # ask_range: the range of ask spread

        distribution = self.policy_distribution(t, Q, P, S)
        # sample the action from the distribution
        flattened_distribution = distribution.flatten()
        # Sample a single index from the flattened distribution
        sampled_index = np.random.choice(a=flattened_distribution.size, p=flattened_distribution)
        # Convert the sampled index back to 2D index in the distribution array
        i, j = np.unravel_index(sampled_index, distribution.shape)
        # Return the corresponding bid and ask values
        return self.bid_range[i], self.ask_range[j]
