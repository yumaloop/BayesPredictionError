import numpy as np
from scipy import stats

class PriorMean():
    def __init__(self, m=0.0, lam=0.5, beta=1):
        self.m = m
        self.lam = lam
        self.beta = beta
        
    def pdf(self, x):
        return stats.norm.pdf(x, loc=self.m, scale=1/(self.beta*self.lam))
    
    def rvs(self, size=100):
        return stats.norm.rvs(loc=self.m, scale=1/(self.beta*self.lam), size=size)
        

class PriorPrecision():
    def __init__(self, a=2, b=2):
        self.a = a
        self.b = b
        
    def pdf(self, x):
        return stats.gamma.pdf(x, self.a, loc=0, scale=self.b)
    
    def rvs(self, size=100):
        return stats.gamma.rvs(self.a, loc=0, scale=self.b, size=size)


class PosteriorMean():
    def __init__(self, data=None, m=None, lam=None, beta=None):
        self.data = data
        self.m = m
        self.lam = lam
        self.beta = beta
        
        self.N = len(data)
        self.m_hat = (np.sum(data) + beta * m) / (beta + N)
        self.beta_hat = beta + N
        
    def pdf(self, x):
        return stats.norm.pdf(x, loc=self.m_hat, scale=1/(self.beta_hat*self.lam))
    
    def rvs(self, size=100):
        return stats.norm.rvs(loc=self.m_hat, scale=1/(self.beta_hat*self.lam), size=size)


class PosteriorPrecision():
    def __init__(self, data=None, a=None, b=None):
        self.data = data
        self.a = a
        self.b = b
        
        self.N = len(data)
        self.a_hat = a + N / 2
        self.b_hat = b + 0.5 * (np.sum(data ** 2) + beta * (m ** 2) + (1/(beta + N)) * (np.sum(data ** 2) + beta * m) ** 2)
        
    def pdf(self, x):
        return stats.gamma.pdf(x, self.a_hat, loc=0, scale=self.b_hat)
    
    def rvs(self, size=100):
        return stats.gamma.rvs(self.a_hat, loc=0, scale=self.b_hat, size=size)

    
class Likelihood():
    def __init__(self, mu=0.0, lam=0.5):
        self.mu = mu
        self.lam = lam
        
    def pdf(self, x):
        return stats.norm.pdf(x, loc=self.mu, scale=1/self.lam)
    
    def logpdf(self, x):
        return stats.norm.logpdf(x, loc=self.mu, scale=1/self.lam)
    
    def rvs(self, size=100):
        return stats.norm.rvs(loc=self.mu, scale=1/self.lam, size=size)


class PredDist():
    def __init__(self, m=None, beta=None, a=None, b=None):
        self.df = 2 * a # degree of freedom
        self.loc = m
        # self.scale = beta / (beta+1) * (a/b)
        self.scale = np.sqrt(beta / (beta+1) * (a/b))
        
    def pdf(self, x):
        return stats.t.pdf(x, self.df, self.loc, self.scale)
    
    def rvs(self, size=100):
        return stats.t.rvs(self.df, self.loc, self.scale, size=size)


class TrueDist():
    def __init__(self, mean=1.2, precision=1.0):
        self.mean = mean
        self.precision = precision
        
    def pdf(self, x):
        return stats.norm.pdf(x, loc=self.mean, scale=1/self.precision)
    
    def rvs(self, size=100):
        return stats.norm.rvs(loc=self.mean, scale=1/self.precision, size=size)
    