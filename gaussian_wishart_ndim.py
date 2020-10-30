import numpy as np
from scipy import stats


class Prior():
    """ N-dim Gaussian-Wishart distribution """
    def __init__(self, m=0.0, beta=1, nu=None, W=None):
        """
        (x, A) ~ NW(m, beta, W, nu) = N(x | m, (beta A)^-1 ) W(A | W, nu)
        """
        self.m = m
        self.beta = beta
        self.nu = nu # degree of freedom for Wishart dist.
        self.W = W
        
    def pdf(self, x, S):
        """ Probability Density Function """
        A = np.linalg.inv(S)
        pdf = stats.multivariate_normal.pdf(x, mean=self.m, cov=np.linalg.inv(self.beta * A))
        pdf = pdf * stats.wishart.pdf(A, df=self.nu, scale=self.W)
        return pdf
    
    def rvs(self, size=100):
        """ Random Variates """
        A = stats.wishart.rvs(df=self.nu, scale=self.W, size=size)
        x = stats.norm.rvs(mean=self.m, cov=np.linalg.inv(self.beta * A))
        return x, A
        

class PosteriorMean():
    """ N-dim Gaussian distribution """
    def __init__(self, data=None, m=None, beta=None, A=None):
        self.N = len(data)
        self.beta_hat = beta + self.N
        self.m_hat = (np.sum(data) + beta * m) / (self.beta_hat)
        self.A = A
        
        self.mean = self.m_hat
        self.cov = np.linalg.inv(self.beta_hat * self.A)
        
    def pdf(self, x):
        """ Probability Density Function """
        return stats.multivariate_norm.pdf(x, mean=self.mean, cov=self.cov)
    
    def rvs(self, size=100):
        """ Random Variates """
        return stats.multivariate_norm.rvs(mean=self.mean, cov=self.cov, size=size)


class PosteriorPrecision():
    """ N-dim Wishart distribution """
    def __init__(self, data=None, beta=None, beta_hat=None, nu=None, m=None, m_hat=None, W=None):
        self.N = len(data) # num
        self.dim = data.shape[1] # dim
        self.beta = beta
        self.beta_hat = beta_hat
        self.nu = nu
        self.m = m
        self.m_hat = m
        self.W = W

        # W_hat
        X = np.zeros((self.dim, self.dim))
        for x in data:
            X = X + np.outer(x, x)
        W_hat_inv = X + beta * np.outer(m, m) - beta_hat * np.outer(m_hat, m_hat) + np.linalg.inv(W)
        self.W_hat = np.linalg.inv(W_hat_inv)
        # nu_hat
        self.nu_hat = self.nu + self.N
        
    def pdf(self, x):
        """ Probability Density Function """
        return stats.wishart.pdf(x, df=self.nu_hat, scale=self.W_hat)
    
    def mean(self):
        return self.nu_hat * self.W_hat
    
    def rvs(self, size=100):
        """ Random Variates """
        return stats.wishart.rvs(df=self.nu_hat, scale=self.W_hat, size=size)

    
class Likelihood():
    """ N-dim Gaussian distribution """
    def __init__(self, mu=0.0, A=None):
        self.mu = mu
        self.A = A

        self.mean = mu
        self.cov = np.linalg.inv(A)
        
    def pdf(self, x):
        """ Probability Density Function """
        return stats.multivariate_norm.pdf(x, mean=self.mean, cov=self.cov)
    
    def logpdf(self, x):
        """ Log Probability Density Function """
        return stats.multivariate_norm.logpdf(x, mean=self.mean, cov=self.cov)
    
    def rvs(self, size=100):
        """ Random Variates """
        return stats.multivariate_norm.rvs(x, mean=self.mean, cov=self.cov, size=size)


class PredDist():
    """ N-dim Student's t-distribution """
    def __init__(self, dim=None, m=None, beta=None, W=None, nu=None):
        self.dim = dim
        self.m = m
        self.beta = beta
        self.W = W
        self.nu = nu

        self.df = 1 - dim + nu # degree of freedom
        self.loc = m
        self.scale = W * (self.df * beta) / (1 + beta)
        self.mean = m
        
    def pdf(self, x):
        """ Probability Density Function """
        return stats.t.pdf(x, self.df, self.loc, self.scale)
    
    def cdf(self, x):
        """ survival function """
        return stats.t.cdf(x, self.df, self.loc, self.scale)
    
    def logpdf(self, x):
        """ Log Probability Density Function """
        return stats.t.logpdf(x, self.df, self.loc, self.scale)
    
    def sf(self, x):
        return stats.t.sf(x, self.df, self.loc, self.scale)
    
    def error(self, x):
        return abs(x - self.mean)
    
    def rvs(self, size=100):
        """ Random Variates """
        return stats.t.rvs(self.df, self.loc, self.scale, size=size)

