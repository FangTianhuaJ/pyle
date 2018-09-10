import numpy as np
import scipy
from numpy import floor, log, exp, pi, cos
import matplotlib.pyplot as plt

from labrad.units import Unit
V, mV, us, ns, GHz, MHz = [Unit(s) for s in ('V', 'mV', 'us', 'ns', 'GHz', 'MHz')]

from labrad.types import Value

from scipy.optimize import leastsq
from pyle.util import structures


def ramseyTomo(dataset, T1=None, timeRange=None):
    """Fit routine for ramsey tomography. This is seperate from the
    fitting functions in FluxNoiseAnalysis and is to be used by
    general users"""
    time = dataset.data[:,0]
    probabilities = dataset.data[:,1:]
    freq = dataset.parameters['fringeFrequency']['GHz']
    
    if T1 is None:
        T1 = float(raw_input('T1 [ns]: '))
    T1=float(T1)
    colors = ['.','r.','g.','k.'] 
    xp = probabilities[:,0]
    yp = probabilities[:,1]
    xm = probabilities[:,2]
    ym = probabilities[:,3]
    env = np.sqrt( (xp-xm)**2 + (yp-ym)**2)
    plt.figure()
    for i in range(4):
        plt.plot(time,probabilities[:,i],colors[i])
    plt.figure()
    plt.plot(time,env)
    if timeRange is None:
        minTime = float(raw_input('Minimum time to keep [ns]: '))
        maxTime = float(raw_input('Maximum time to keep [ns]: '))
        timeRange = (minTime,maxTime)
    #Generate array mask for elements you want to keep
    indexList = (time>timeRange[0]) * (time<timeRange[1])
    t = time[indexList]
    e = env[indexList]

    fitFunc = lambda v,x: v[0]*exp(-((x/v[1])**2))*\
              exp(-x/(2*T1))*exp(-x/v[2])+v[3]
    errorFunc = lambda v,x,y: fitFunc(v,x)-y
    
    vGuess = [0.85, 200.0, 700.0, 0.02]
    result = leastsq(errorFunc, vGuess, args=(t,e), maxfev=10000, full_output=True)
    v = result[0]
    cov=result[1]
    success = result[4]
    #Jim's formula for error bars. Got this from the Resonator fit server.
    rsd = (result[2]['fvec']**2).sum()/(len(t)-len(v))
    bars = np.sqrt(rsd*np.diag(result[1]))
    #Plot the fitted curve
    plt.plot(time, fitFunc(v,time),'g')
    plt.text(120,0.6,r'$p(t)=\exp \left[ -\frac{t}{2T_1}-(\frac{t}{T_2})^2 \right]$')
    T2 = v[1]*ns
    dT2 = bars[1]*ns
    Tphi = v[2]*ns
    dTphi = bars[2]*ns
    plt.text(100,0.5,'T2 = %f' %T2['ns'])
    return structures.AttrDict({'T2':T2, 'dT2':dT2, 'Tphi':Tphi, 'dTphi':dTphi, 'success':success})
