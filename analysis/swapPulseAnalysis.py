import numpy as np
import matplotlib.pyplot as plt

from pyle.plotting.dstools import getParameters

def swappulses(ds, files, session):
    times = []
    ampl = []
    for f in files:
        p = getParameters(ds, f, session)
        plt.figure(1)
        t = p['swapTimes']
        a = p['swapAmplitudes']
        times += t
        ampl += a
        t = np.array(t)
        a = np.array(a)        
        plt.plot(t, a-p['swapAmplitude'])
        plt.figure(2)
        plt.plot(1+np.arange(len(t)), t, 'x')
    n = np.arange(0.1, 20, 0.1)
    times = np.array(times)
    ampl = np.array(ampl)
    
    plt.plot(n, 26/np.sqrt(n))
    plt.show()        
             
        
