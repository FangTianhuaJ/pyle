import math
import time

import numpy as np
import matplotlib.pyplot as plt

from labrad.units import Unit
V, mV, us, ns, GHz, MHz, sec = [Unit(s) for s in ('V', 'mV', 'us', 'ns', 'GHz', 'MHz', 's')]

from pyle import envelopes as env
from pyle.dataking import envelopehelpers as eh
from pyle.dataking import util
from pyle.pipeline import returnValue, FutureList
from pyle.util import sweeptools as st
from pyle.dataking import sweeps
from pyle.dataking.fpgaseq import runQubits
from pyle.dataking import utilMultilevels as ml

import os
from msvcrt import getch, kbhit


DATA_FORMAT = '%.12G'

def getKBuffer():
    buf = []
    while kbhit():
        buf.append(getch())
    return buf

def clearKBuffer():
    while kbhit():
        getch()

def randomNums(s, measure, mpa, repsPerFile, delay=4*ns, stats=300,
               name='Random Numbers', save=True, collect=False, noisy=False):
    """Generate a string of random numbers by measuring a state on the equator"""
    sample, qubits, Qubits = util.loadQubits(s,write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]
    q['measureAmp'] = mpa
    q['readout'] = True
    q.z = eh.measurePulse(q, 0)
    
    iterations = st.r[0:repsPerFile:1]
            
    axes = [(iterations, 'iteration')]
    deps = [('Tunnelled','','')]
    kw = {}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)
    
    def func(server,arg):
        result = yield runQubits(server, qubits, stats, dataFormat = 'phases')
        returnValue(np.vstack(result).T)
    data = sweeps.grid(func, axes, save=save, dataset=save and dataset,
                       noisy=noisy, collect=collect)
    if collect:
        return data
    
    
def test(s, measure, mpa, repsPerFile, numFiles, stats):
    
    os.chdir('U:\\daniel\\projects\RandomBits')
    
    clearKBuffer()
    
    for numFile in range(numFiles):
        print numFile
        filename = str(numFile)+'.csv'
        #Take some data
        data = randomNums(s, measure, mpa, repsPerFile, collect=True, save=False, stats=stats)
        #Save that data
        np.savetxt(filename, data, fmt=DATA_FORMAT, delimiter=',')
        #Convert to packed binary file
        os.system('ra2bin %s' %filename)
        #Remove .csv file
        if not numFile%10==0:
            os.remove(filename)
        #Check for user quit command
        buf = getKBuffer()
        if 'Q' in buf:
            break