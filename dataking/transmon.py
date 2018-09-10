import numpy as np
from scipy.optimize import leastsq
from scipy.special import erf, erfc
import matplotlib.pyplot as plt

from labrad.units import Unit,Value
V, mV, us, ns, GHz, MHz, dBm, rad = [Unit(s) for s in ('V', 'mV', 'us', 'ns', 'GHz', 'MHz', 'dBm', 'rad')]

import pyle.envelopes as env
from pyle.pipeline import returnValue, FutureList
from pyle.dataking import envelopehelpers as eh
from pyle.dataking import measurement
from pyle.dataking import adjust
from pyle.dataking.fpgaseqTransmon import runQubits as runQubits
from pyle.util import sweeptools as st
from math import atan2
from pyle.dataking import utilMultilevels as ml
from pyle.plotting import dstools
from pyle.fitting import fitting
from pyle.util import structures

from pyle import gateCompiler as gc
from pyle.gateCompiler import PiPulse, Spectroscopy, Wait, Measure, Algorithm


import scipy.optimize as opt

import qubitpulsecal as qpc
import sweeps
import util
import labrad

def s_scanning(sample, freq=None, power=None, reset = None,  sb_freq = -50*MHz, measure=0, stats=150,
               save=True, name='S parameter scanning', collect=False, noisy=True, update=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    axes = [(freq, 'Frequency')]
    deps = [('Phase', 'S11 for %s'%q.__name__, rad), ('Amplitude','S11 for %s'%q.__name__,'')]
    kw = {'stats': stats, 'power':power}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    if power is not None:
        q['readout power'] = power

    q['readout'] = True

    q = gc.AttrDict2Qubit(q, gc.Transmon)

    def func(server, f):
        q['readout frequency']=f
        q['readout fc'] = q['readout frequency'] - sb_freq

        alg = Algorithm()
        alg[Measure([q])]
        alg.compile()

        if noisy: print f
        data = yield FutureList([runQubits(server, [q], stats, dataFormat='iq')])
        I = np.mean(data[0][0][0])
        Q = np.mean(data[0][0][1])
        amp = abs(I+1j*Q)
        phase = (atan2(Q,I))
        returnValue([phase, amp])

    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=False)
    if update:
        phase = np.asarray([[row[0], row[1]] for row in data])
        adjust.adjust_s_scanning(Q, phase)
    if collect:
        return data

def readoutIq(sample, freq=None, power=None, sb_freq=-50.0*MHz, measure=0, stats=150,
              save=True, name='Readout IQ', collect=False, noisy=True, plot=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    # this allows us to look at the resonance circle in the IQ plane
    # update readoutIqOffset with the additive translation to move circle to the origin

    # since s_scanning operates at -50*MHz sideband,
    # after this scan, set readout fc to be 50*MHz above readout frequency

    if freq is None:
        f = st.nearest(q['readout frequency'][GHz], 1e-5)
        freq = st.r[f-0.001:f+0.001:1e-5, GHz]
    if power is None:
        power = q['readout power']

    axes = [(freq, 'Frequency'),(power,'Power')]
    deps = [('Clicks', 'I for %s' %q.__name__, rad), ('Clicks', 'Q for %s'%q.__name__,'')]
    kw = {'stats': stats, 'power':power}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    q['readout'] = True

    q = gc.AttrDict2Qubit(q, gc.Transmon)

    def func(server, f, power):
        q['readout frequency']=f
        q['readout fc'] = q['readout frequency'] - sb_freq
        q['readout power'] = power

        alg = Algorithm()
        alg[Measure([q])]
        alg.compile()

        if noisy: print f
        data = yield FutureList([runQubits(server, [q], stats, dataFormat='iq', debug=True)])
        I = np.mean(data[0][0][0])
        Q = np.mean(data[0][0][1])
        returnValue([I,Q])

    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=False)
    if plot:
        plt.figure()
        plt.plot(data[:,1],data[:,2],'.')
        plt.grid()
        for i,freq in enumerate(data[:,0]):
            plt.text(data[i,1],data[i,2],str(data[i,0]))
        plt.xlabel('I')
        plt.ylabel('Q')
    if collect:
        return data

def phaseSpectroscopy(sample, freq=None,  sb_freq = -50*MHz, measure=0, stats=150,
               save=True, name='Phase Spectroscopy', collect=False, noisy=True, update=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    axes = [(freq, 'Frequency')]
    deps = [('Phase', 'S21 for %s'%q.__name__, rad), ('Amplitude','S21 for %s'%q.__name__,'')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    q['readout'] = True

    q = gc.AttrDict2Qubit(q, gc.Transmon)

    def func(server, f):
        q['fc']=f

        alg = Algorithm()
        alg[Spectroscopy([q])]
        alg[Measure([q])]
        alg.compile()

        if noisy: print f
        data = yield FutureList([runQubits(server, [q], stats, dataFormat='iq')])
        I = np.mean(data[0][0][0])
        Q = np.mean(data[0][0][1])
        amp = abs(I+1j*Q)
        phase = (atan2(Q,I))
        returnValue([phase, amp])

    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=False)

    if collect:
        return data

def t1(sample, delay=st.r[-10:1000:2,ns], stats=600L, measure=0,
       name='T1', save=True, collect=True, noisy=True, state=1,
       update=True, plot=True):
    """A single pi-pulse on one qubit, with other qubits also operated."""
    if update and (state>1):
        raise Exception('updating with states above |1> not yet implemented')
    sample, qubits, Qubits = util.loadQubits(sample,write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]
    q = gc.AttrDict2Qubit(q, gc.Transmon)
    ml.setMultiKeys(q,state)
    if state>1: name=name+' for |'+str(state)+'>'

    axes = [(delay, 'Readout delay')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)
    def func(server, delay):
        # gate compiler shit
        alg = Algorithm()
        alg[PiPulse([q])]
        alg[Wait([q], delay)]
        alg[Measure([q])]
        alg.compile()
        q['readout'] = True
        return runQubits(server, [q], stats, probs=[1], debug=True)
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return data

