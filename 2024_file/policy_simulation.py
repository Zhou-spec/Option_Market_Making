# the policy is quite complicated, we need to first compute the numerator of the pdf given (bid, ask)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 

""" 
This file define a trading policy object for the trading simulation.
"""

class TradingPolicy:
    def __init__(self, ValueNet, penalty, A, kappa, bid_range, ask_range):
        """
        Initialize the trading policy with the given parameters.

        :param ValueNet: The neural network model for the value function.
        :param penalty: The penalty coefficient (gamma).
        :param A: The constant in the rate function.
        :param kappa: The constant in the rate function.
        """
        # if ValueNet is on the GPU, move it to CPU
        
        self.ValueNet = ValueNet.to('cpu')  # Move the neural network to the CPU
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
        # overflow maybe a problem here
        # when encounter overflow, just use the maximum value to subtract
        shift_distribution = distribution - np.max(distribution)
        try:
            distribution = np.exp(shift_distribution) / np.sum(np.exp(shift_distribution))
        except OverflowError:
            # Directly catch the OverflowError to be more explicit
            max_index = np.argmax(distribution)
            distribution = np.zeros_like(distribution)
            distribution[max_index] = 1

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


    # This is for the computation of the loss function
    def policy_entropy(self, t, Q, P, S):
        distribution = self.policy_distribution(t, Q, P, S)
        # Compute entropy
        entropy = -np.sum(distribution * np.log(distribution), where=(distribution!=0))
        entropy = entropy * self.penalty

        return entropy
    

    # special function to compute expected value of \lambda \epsilon
    def expected_profits(self, t, Q, P, S):
        # this function is to compute the expected profits of the policy
        
        distribution = self.policy_distribution(t, Q, P, S)
        n = len(Q)
        profits = 0
        for i in range(n):
            for j in range(n):
                # notice that bid_range is m * n 
                # m is the number of different bid for n options

                bid_for_options = self.bid_range[i]
                ask_for_options = self.ask_range[j]

                # compute the rate given bid and ask
                bid_rate = self.A * np.exp(-self.kappa * bid_for_options)
                ask_rate = self.A * np.exp(-self.kappa * ask_for_options)

                profits += np.sum((bid_rate[i] * bid_for_options + ask_rate[j] * ask_for_options)) * distribution[i, j]
        
        return profits
    

    ############################################################################################################

    # the following is the network structure I am going to use for the value network
    # the value network is a simple feedforward neural network


class Net(torch.nn.Module):
    def __init__(self, n):
        super(Net, self).__init__()  # Call superclass constructor
        self.n = n
        self.fc1 = torch.nn.Linear(2 + 2 * n, 128)
        self.fc2 = torch.nn.Linear(128, 1024)
        self.fc3 = torch.nn.Linear(1024, 256)
        self.fc4 = torch.nn.Linear(256, 1)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
