'''
Created back in the day ('09?)
author: Max Hofheinz and/or probably Matthew Neeley
Recovered by Erik Lucero May 2011

Needs some retouching to get aligned with current pyle release.
Looks useful for Resonator measurement analysis
'''


import numpy as np
import matplotlib.pyplot as plt

import dstools
#from numberanalysis import *
import datareader as ds

ds.cd(['','090202','r7c3','090302'])

def catanalysis(ds, vis, coherent, wigner, session=None, N=6,
                K=10, KCalc=None, qT1=650, qT2=300, rT1=3500,
                minContrast=2.0, scale=None, normalize=True,
                normalizeRho=False, doPlot=0):
    data = dstools.getDataset(ds, dataset=wigner, session=session)
    n = 1 + np.argwhere((np.diff(data[:,0])!=0) | (np.diff(data[:,1]) != 0))[:,0]
    n = np.append([0], n)
    if any(n != np.arange(0, len(n))*n[1]):
        raise Exception('Data file not understood')

    alpha = data[n,0] + 1.0j*data[n,1]
    n  = n[1]
    time = data[:n,2]
    print time
    data = np.reshape(data[:,3:], (np.size(alpha), n, 4))
    even = 100.0*data[:,:,3] / np.sum(data[:,:,2:], axis=2)
    odd = 100.0*data[:,:,1] / np.sum(data[:,:,0:2], axis=2)
    total = sum(data[:,:,1:4:2],axis=2)
    plt.figure(1)
    plt.clf()
    plt.plot(even[0,:])
    plt.plot(odd[0,:])
    plt.plot(total[0,:])
    even = np.transpose(even)
    odd = np.transpose(odd)
    total = np.transpose(total)
    
    
    p0, p1 = getVis(ds, vis, session, quiet=True)
    rabifreq, maxima, amplscale, visibilities = \
        coherentanalysis(ds,dataset=coherent, session=session,
                         minContrast=minContrast, chopForVis=time[-1]+0.1,
                         p0=p0, p1=p1, doPlot=doPlot & PLOT_COHERENT,
                         scale=scale)
    if scale is not None:
        amplscale = scale
    even = numberProb(even, rabifreq, time[1]-time[0], p0, p1, n=K, nCalc=KCalc,qT1=qT1, qT2=qT2, rT1=rT1,
                      normalize=normalize)
    odd = numberProb(odd, rabifreq, time[1]-time[0], p0, p1, n=K, nCalc=KCalc,qT1=qT1, qT2=qT2, rT1=rT1,
                      normalize=normalize)
    total = numberProb(total, rabifreq, time[1]-time[0], p0, p1, n=K, nCalc=KCalc,qT1=qT1, qT2=qT2, rT1=rT1,
                      normalize=normalize)
    even = np.transpose(even)
    odd = np.transpose(odd)
    total = np.transpose(total)
    rhoEven = QktoRho(alpha*amplscale, even, N=N)
    rhoOdd = QktoRho(alpha*amplscale, odd, N=N)
    rho = QktoRho(alpha*amplscale, total, N=N)
    if normalizeRho:
        rhoEven /= np.trace(rhoEven)
        rhoOdd /= np.trace(rhoOdd)
        rho /= np.trace(rho)
    return rho, rhoEven, rhoOdd

