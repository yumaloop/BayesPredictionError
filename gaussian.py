import numpy as np
from scipy import stats

def prior_mean(x, m=0.0, lam=1, beta=1):
    return stats.norm(x, loc=m, scale=1/(beta*lam))


def prior_precision(x, a=2, b=2):
    return stats.gamma.pdf(x, a, loc=0, scale=b)


def posterior_mean(x, data=None, m=None, lam=None, beta=None):
    N = len(data)
    mu_hat = (np.sum(data ** 2) + beta * m) / (beta + N)
    beta_hat = beta + N
    return stats.norm(x, loc=mu_hat, scale=1/(beta_hat*lam))


def posterior_precision(x, data=None, a=None, b=None):
    N = len(data)
    a_hat = a + N / 2
    b_hat = b + 0.5 * (np.sum(data ** 2) + beta * (m ** 2) - (1/(beta + N)) * (np.sum(data ** 2) + beta * m) ** 2)
    return stats.gamma.pdf(x, a_hat, loc=0, scale=b_hat)


def likelihood(x, mu=0.0, lam=1.0):
    return stats.norm.pdf(x, loc=mu, scale=1/lam)


def pred_dist(x, m=None, beta=None, a=None, b=None):
    df = 2 * a # degree of freedom
    loc = m
    scale = (beta/(beta + 1)) * (a/b)
    return stats.t.pdf(x, df, loc, scale)
