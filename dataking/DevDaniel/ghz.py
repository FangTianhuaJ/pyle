# James Wenner

import numpy as np

from labrad.units import Unit,Value
V, mV, us, ns, GHz, MHz, dBm, rad = [Unit(s) for s in ('V', 'mV', 'us', 'ns', 'GHz', 'MHz', 'dBm', 'rad')]

import pyle.envelopes as env
from pyle.pipeline import returnValue, FutureList
from pyle.dataking import envelopehelpers as eh
from pyle.dataking import measurement
from pyle.dataking import adjust
from pyle.dataking.fpgaseq import runQubits as runQubits
from pyle.dataking import dephasingSweeps
from pyle.util import sweeptools as st
from math import atan2
from pyle.dataking import utilMultilevels as ml
from pyle.plotting import dstools
from pyle.fitting import fitting
from pyle.util import structures
from pyle.dataking.benchmarking import danBench as db

from pyle.dataking import sweeps
from pyle.dataking import util
import pyle
import labrad

def wSwap1(s, measure, paramName, swapTime = st.r[0:200:1,ns], piQubit=0,
                    name = 'W-Single Swap', stats=1200,
                    collect=False, save=True, noisy=True):

    sample, qubits = util.loadQubits(s)
    qP = qubits[piQubit]

    axes = [(swapTime,'Swap Time')]
    kw = {'stats': stats, 'piQubit':piQubit}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, delay):
        t = 0.0*ns
        for q in qubits:
            q['z'] = env.NOTHING
        #Excite piQubit
        qP['xy'] = eh.piPulse(qP, t, phase = 0)
        t += 0.5*qP['piLen']
        #Do iSwap and measure pulse for all qubits
        t += delay
        for q in [qubits[qubit] for qubit in measure]:
            q['z'] += env.rect(t-delay, delay, q['swapAmp'+paramName])
            q['readout'] = True
            q['z'] += eh.measurePulse(q, t)
        #Apply sideband mixing
        qP['xy'] = eh.mix(qP, qP['xy'])
        eh.correctCrosstalkZ(qubits)
        return runQubits(server, qubits, stats)

    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if collect:
        return data

def wSwap2(s, measure, paramName, swapTime = st.r[0:200:1,ns], piQubit=0,
                    name = 'W-Two Swaps', stats=1200,
                    collect=False, save=True, noisy=True,
                    tomo=False, tBuf = 0.0*ns):

    sample, qubits = util.loadQubits(s)
    qP = qubits[piQubit]
    N = len(qubits)

    if tomo:
        measureFunc = measurement.Tomo(N, measure, tBuf = tBuf)
    else:
        measureFunc = measurement.Simult(N, measure, tBuf = tBuf)

    axes = [(swapTime,'Swap Time')]
    kw = {'stats': stats, 'piQubit':piQubit}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measureFunc, kw=kw)

    def func(server, delay):
        t = 0.0*ns
        for q in qubits:
            q['z'] = env.NOTHING
        #Excite piQubit
        qP['xy'] = eh.piPulse(qP, t, phase = 0)
        t += 0.5*qP['piLen']
        #iSwap excitation into resonator
        qP['z'] += env.rect(t, qP['swapTime'+paramName], qP['swapAmp'+paramName])
        t += qP['swapTime'+paramName]+tBuf
        t += delay
        #Do iSwap and measure pulse for all qubits
        for q in [qubits[m] for m in measure]:
            q['z'] += env.rect(t-delay, delay, q['swapAmp'+paramName])
            q['readout'] = True

        #Apply sideband mixing
        qP['xy'] = eh.mix(qP, qP['xy'])
        eh.correctCrosstalkZ(qubits)
        return measureFunc(server, qubits, t, stats=stats)

    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if collect:
        return data

def getRho(path, dataset,N=1,tomo='Tomo'):
    with labrad.connect() as cxn:
        dv = cxn.data_vault
        dataset = dstools.getDeviceDataset(dv, datasetId=dataset, session=path)
        probs = dataset.data[:,1:]
        F = db.fMatrix(dataset)
        if tomo == 'Tomo':
            tomoNum = 3**N
        elif tomo == 'Octomo':
            tomoNum = 6**N
        pxmsArray = probs.reshape((-1,tomoNum,2**N))
        Us, U = pyle.tomo._qst_transforms[tomo.lower()+str(N)]
        rhos = np.array([pyle.tomo.qst_mle(pxms, Us, F) for pxms in pxmsArray])
        return rhos

def getWPhases(rho):
    phase12 = np.angle(rho[2][1])
    phase02 = np.angle(rho[4][1])
    phase01 = np.angle(rho[4][2])


