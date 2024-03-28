import numpy as np
from scipy.stats import norm


# the following class define a European Call option
# the users can use it to compute greeks and price of the option
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
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def price(self):
        d1 = self.d1()
        d2 = self.d2()
        return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)

    def delta(self):
        d1 = self.d1()
        return norm.cdf(d1)

    def gamma(self):
        d1 = self.d1()
        return norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        d1 = self.d1()
        return self.S * norm.pdf(d1) * np.sqrt(self.T) / 100  # Vega is often represented per 1% change in volatility

    def theta(self):
        d1 = self.d1()
        d2 = self.d2()
        theta = - (self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        return theta / 365  # Convert to per-day value

    def rho(self):
        d2 = self.d2()
        return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) / 100  # Rho is often represented per 1% change in interest rate
    


############################################################################################################

def mkt_order(epsilon, dt, A, kappa):
    # epsilon: could be a pytorch tensor or numpy float number
    # return: the number 
    # A, kappa: hyperparameters
    # model the market order arrival as a poisson process
    # the arrival rate is A * exp(-kappa * epsilon)

    return np.random.poisson(A * np.exp(-kappa * epsilon) * dt)





    
    
    

