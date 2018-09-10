from scipy.optimize import leastsq

from numpy.fft import rfft
from pylab import iterable
from numpy import ones, zeros, arange

def optimalwindow(n, badness):
    nfft = (len(badness)-1)*2
    def errfunc(params):
        x = zeros(n,dtype=float)
        x[0:-1]=params
        x[-1] = n-sum(params)
        return abs(rfft(x,n=nfft))*badness
    params = ones(n-1,dtype=float)
    result, success = leastsq(errfunc,params)
    x = zeros(n, dtype=float)
    x[0:-1] = result
    x[-1] = n-sum(result)
    return x


def badness(nfft, badspots, sigma, bgstart, bgh):
    i = arange(nfft/2+1)
    badness = 0.0

    if iterable(sigma):
        for j, b in enumerate(badspots):
            badness += 1.0/(1+((i-float(b))/sigma[j])**2)
    else:
        for b in badspots:
            badness += 1.0/(1.0+((i-float(b))/sigma)**2)
        
    return badness

    
    


