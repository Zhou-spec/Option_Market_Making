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

# this function is to generate the stock price path
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

def mkt_order(epsilon, dt, A, kappa):
    # epsilon: could be a pytorch tensor or numpy float number
    # return: the number 
    # A, kappa: hyperparameters
    # model the market order arrival as a poisson process
    # the arrival rate is A * exp(-kappa * epsilon)

    return np.random.poisson(A * np.exp(-kappa * epsilon) * dt)


    
    
    

