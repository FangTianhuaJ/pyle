import numpy as np
from scipy.special import gammaln

def lnbinom(k, n):
    return gammaln(n+1) - gammaln(n-k+1) - gammaln(k+1)

def lnproba(p1, stats):
    k = np.arange(stats+1)
    return lnbinom(k, stats) + k * np.log(p1) + (stats-k) * np.log(1-p1)


def uncertainty(p, stats):
    """Uncertainty of our measurement, i.e. a binomial process with
    probability p and stats trials"""
    
    return np.sqrt(p*(1-p)/stats)
