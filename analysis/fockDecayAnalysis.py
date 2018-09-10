from plye.plotting import dstools
from pyle.analysis.numberAnalysis import photonnumbers1 as photonnumbers
import numpy as np

def fockdecay(ds, p0, p1, freq, dataset=None, session=None, nmax=5):
    #read data
    data = dstools.getDataset(ds, dataset, session)

    borders = np.argwhere(np.diff(data[:,1]) > 0)[:,0]
    borders = np.hstack((0, 1+borders, np.shape(data)[0]))
    n = len(borders)-1
    result = np.zeros((1+nmax,n))
    result[0,:] = data[borders[:-1],1]
    for i in np.arange(n):
        p = data[borders[i]:borders[i+1],2]
        dt = data[borders[i]+1,0]-data[borders[i],0]
        result[1:,i] = photonnumbers(p, freq, dt, p0, p1, n=nmax)
    return result


        
        
