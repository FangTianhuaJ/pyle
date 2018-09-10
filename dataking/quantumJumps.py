'''
Created August 20, 2011
author: Daniel Sank
'''

#CHANGELOG


#import math
#import time

#import numpy as np
#import matplotlib.pyplot as plt

from labrad.units import Unit
V, mV, us, ns, GHz, MHz, sec = [Unit(s) for s in ('V', 'mV', 'us', 'ns', 'GHz', 'MHz', 's')]

from pyle import envelopes as env
from pyle.dataking import envelopehelpers as eh
from pyle.dataking import util
#from pyle.pipeline import returnValue, FutureList
from pyle.util import sweeptools as st
from pyle.dataking import sweeps
from pyle.dataking.fpgaseq import runQubits #,runInterlacedSRAM
#from pyle.plotting import dstools
#from pyle.analysis import signalProcessing as sp
#from pyle.analysis import FluxNoiseAnalysis as fna
#from pyle.dataking import utilMultilevels as ml

#from pyle.fitting import fitting
#import labrad

def singleShotSort(sample, delay=st.r[-10:1000:2,ns], stats=600L, measure=0,
                   name='T1 Single shot sort', collect=True,
                   noisy=True,plot=True,raw=False):
    """A single pi-pulse on one qubit, with other qubits also operated."""
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    
    #axes = [(delay, 'Measure pulse delay')]
    #kw = {'stats': stats}
    #dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)
    
    def func(server, delay):
        q.xy = eh.boostState(q, 0, state=1)
        q.z = eh.measurePulse(q, delay, state=1)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1],raw=raw)
    data = sweeps.run(func, delay, save=False, dataset=None, abortable=True,
                      abortPrefix=[], collect=True, noisy=True, pipesize=10)
    return data

def swapWithCoherent(Sample, measureC, measureP, measureR, paramName, swapFraction,
                     pulseTime = None, pulseAmp = None,
                     name=None, collect=False, noisy=False, save=True):
    """Swap a part of a coherent state into Control qubit and measure the resonator with the Probe qubit"""
    sample, qubits = util.loadDeviceType(Sample, 'phaseQubit')
    sample, resonators = util.loadDeviceType(Sample, 'resonator')
    qC = qubits[measureC]
    qP = qubits[measureP]
    r = resonators[measureR]
    
    if pulseTime is None:
        pulseTime = 20*ns
    if pulseAmp is None:
        pulseAmp = 0.5
    
    swapTime = qC[paramName+'SwapTime']*swapFraction
    
    def func(server):
        t=0*ns
        qC.xy = qC.z = qP.xy = qP.z = env.NOTHING
        #Drive the resonator
        r.xy = env.flattop(t,pulseTime,r['piFWHM'],pulseAmp)
        t += (pulseTime + 2*r['piFWHM'])
        #Grab some state with the control qubit
        qC.z += env.rect(t,swapTime,qC[paramName+'SwapAmp'])
        t += swapTime
        #Now measure the resonator with the other qubit
        
