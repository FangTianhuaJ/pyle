from pyle import gateCompiler as gc
from pyle.gateCompiler import PiPulse,Wait,Measure
import matplotlib.pyplot as plt
import numpy as np
from pyle import envelopes as env
from labrad.units import ns

#THIS IS A COMPLETE HACK MEANT ONLY AS A DEMO OF THE GATECOMPILER!!!!!

tBuffer = 200

def T1(qubit, time):
    alg = gc.Algorithm(agents=[qubit])
    alg[gc.PiPulse(qubit)][gc.Wait(qubit, time)][gc.Measure(qubit)]
    alg.compile()
    
def showT1Scan(qubit, timeScan):
    fig = plt.figure()
    n = len(timeScan)
    nPts = timeScan[-1]+2*tBuffer
    t = np.linspace(timeScan[0]['ns']-tBuffer,0,nPts)
    for i, time in enumerate(timeScan):
        alg = gc.Algorithm(agents=[qubit])
        #end hack
        T1(qubit, time)
        #Plot that shit        
        ax = fig.add_subplot(n,1,i+1)
        plt.grid()
        ax.plot(t, qubit['xy'](t))
        ax.plot(t, qubit['z'](t))
        ax.text(0,0,str(time))