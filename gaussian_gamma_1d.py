import numpy as np
from scipy import stats


class PriorMean():
    def __init__(self, m=0.0, lam=1, beta=1):
        self.m = m
        self.lam = lam
        self.beta = beta
        
        self.mean = stats.norm.mean(loc=self.m, scale=1/(self.beta*self.lam))
        self.var = stats.norm.var(loc=self.m, scale=1/(self.beta*self.lam))
        
    def pdf(self, x):
        return stats.norm.pdf(x, loc=self.m, scale=1/(self.beta*self.lam))
    
    def rvs(self, size=100):
        return stats.norm.rvs(loc=self.m, scale=1/(self.beta*self.lam), size=size)
        

class PriorPrecision():
    def __init__(self, a=2, b=2):
        self.a = a
        self.b = b
        
        self.mean = stats.gamma.mean(self.a, loc=0, scale=1./self.b)
        self.var = stats.gamma.var(self.a, loc=0, scale=1./self.b)
        
    def pdf(self, x):
        return stats.gamma.pdf(x, self.a, loc=0, scale=1./self.b)
    
    def rvs(self, size=100):
        return stats.gamma.rvs(self.a, loc=0, scale=1./self.b, size=size)


class PosteriorMean():
    def __init__(self, data=None, m=None, lam=None, beta=None):
        self.N = len(data)
        self.lam = lam
        self.m_hat = (np.sum(data) + beta * m) / (beta + self.N)
        self.beta_hat = beta + self.N
        
        self.mean = stats.norm.mean(loc=self.m_hat, scale=1/(self.beta_hat*self.lam))
        self.var = stats.norm.var(loc=self.m_hat, scale=1/(self.beta_hat*self.lam))
        
    def pdf(self, x):
        return stats.norm.pdf(x, loc=self.m_hat, scale=1/(self.beta_hat*self.lam))
    
    def rvs(self, size=100):
        return stats.norm.rvs(loc=self.m_hat, scale=1/(self.beta_hat*self.lam), size=size)


class PosteriorPrecision():
    def __init__(self, data=None, a=None, b=None, beta=None, m=None):        
        self.N = len(data)
        self.a_hat = a + self.N / 2
        self.b_hat = b + 0.5 * (np.sum(data ** 2) + beta * (m ** 2) - (1/(beta + self.N)) * (np.sum(data) + beta * m) ** 2)
        
        self.mean = stats.gamma.mean(self.a_hat, loc=0, scale=1./self.b_hat)
        self.var = stats.gamma.var(self.a_hat, loc=0, scale=1./self.b_hat)
        
    def pdf(self, x):
        return stats.gamma.pdf(x, self.a_hat, loc=0, scale=1./self.b_hat)
    
    def mean(self):
        return stats.gamma.mean(self.a_hat, loc=0, scale=1./self.b_hat)
    
    def rvs(self, size=100):
        return stats.gamma.rvs(self.a_hat, loc=0, scale=1./self.b_hat, size=size)

    
class Likelihood():
    def __init__(self, mu=0.0, lam=1.0):
        self.mu = mu
        self.lam = lam
        
        self.mean = stats.norm.mean(loc=self.mu, scale=1/self.lam)
        self.var = stats.norm.var(loc=self.mu, scale=1/self.lam)
        
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
        self.scale = 1./np.sqrt(beta / (beta+1) * (a/b))
        
        self.mean = stats.t.mean(self.df, self.loc, self.scale)
        self.var = stats.t.var(self.df, self.loc, self.scale)
        
    def pdf(self, x):
        return stats.t.pdf(x, self.df, self.loc, self.scale)
    
    def cdf(self, x):
        """ survival function """
        return stats.t.cdf(x, self.df, self.loc, self.scale)
    
    def logpdf(self, x):
        return stats.t.logpdf(x, self.df, self.loc, self.scale)
    
    def sf(self, x):
        return stats.t.sf(x, self.df, self.loc, self.scale)
    
    def error(self, x):
        return abs(x - self.mean)
    
    def rvs(self, size=100):
        return stats.t.rvs(self.df, self.loc, self.scale, size=size)

