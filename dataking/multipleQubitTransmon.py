import numpy as np
import scipy.optimize as opt
from scipy.optimize import leastsq, curve_fit
from scipy.special import erf, erfc
import matplotlib.pyplot as plt
from scipy.integrate import quad

import time
import itertools

import labrad
import labrad.units as U

import pyle
from pyle import tomo, fidelity
import pyle.envelopes as env
from pyle.pipeline import returnValue, FutureList
from pyle.dataking import envelopehelpers as eh
from pyle.dataking import adjust
from pyle.util import sweeptools as st
from pyle.dataking import utilMultilevels as ml
from pyle.plotting import dstools
from pyle.fitting import fitting
from pyle.dataking import qubitpulsecal as qpc
from pyle.dataking import sweeps
from pyle.dataking import util
from pyle.dataking.fpgaseqTransmonV7 import runQubits
from pyle import gateCompiler as gc
from pyle import gates
from pyle.plotting import tomography
from pyle.interpol import interp1d_cubic
from pyle.analysis import readout

# COLORS
BLUE   = "#348ABD"
RED    = "#E24A33"
PURPLE = "#988ED5"
YELLOW = "#FBC15E"
GREEN  = "#8EBA42"
PINK   = "#FFB5B8"
GRAY   = "#777777"


V, mV, us, ns, GHz, MHz, dBm, rad = [U.Unit(s) for s in ('V', 'mV', 'us', 'ns', 'GHz', 'MHz', 'dBm', 'rad')]

def testSpectroscopy2DZxtalk(Sample, measure=(0,1), freqScan=st.r[4.5:6:0.01, GHz], zbias=st.r[-2:2:0.1],
                             readoutPower=None, readoutFrequency=None, demodFreq=10*MHz, sb_freq=0*GHz,
                             stats=300L, name='test 2D Z-pulse xtalk spectroscopy', save=True, noisy=True):
    """
    measure = [ measure qubit, z pulse qubit ]
    z_pulse on qubits[z_pulse], measure spectroscopy on qubits[measure]
    """
    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, True)

    axes = [(zbias, 'Z pulse Amplitude'), (freqScan, 'Frequency')]
    deps = [("Mag", "", ""), ("Phase", "", "rad")]

    if readoutPower is None:
        readoutPower = devs[measure[0]]['readoutPower']
    if readoutFrequency is None:
        readoutFrequency = devs[measure[0]]['readoutFrequency']

    kw = {'stats': stats, 'readoutPower': readoutPower, "readoutFrequency": readoutFrequency,
          "demodFreq": demodFreq, "sb_freq": sb_freq, "qubit measure": measure[0], "qubit z pulse": measure[1]}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure[0], kw=kw)
    idx = sorted(measure).index(measure[0])
    qubitNameCarrier = util.otherQubitNamesOnCarrier(devs[measure[0]], devs)

    def func(server, z, freq):
        alg = gc.Algorithm(devs)
        qm = alg.q0
        qz = alg.q1
        qm['readoutPower'] = readoutPower
        qm['readoutFrequency'] = readoutFrequency
        qm['readoutDevice']['carrierFrequency'] = readoutFrequency - demodFreq
        qz['readoutPower'] = -100*dBm # do not readout qz
        qm['fc'] = freq - sb_freq
        for name in qubitNameCarrier:
            alg.agents_dict[name]['fc'] = freq - sb_freq
        alg[gates.Spectroscopy([qm], df=sb_freq)]
        alg[gates.Detune([qz], tlen=(qm['spectroscopyLen']), amp=z)]
        alg[gates.Measure([qm, qz], align='start')]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        # only record data for qm
        mag, ang = readout.iqToPolar(readout.parseDataFormat(data[idx][np.newaxis, ], 'iq'))
        returnValue([mag, ang])

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data

def testSwapSpectroscopy(Sample, measure=(0,1), delay=st.r[0:200:2, ns], swapAmp=st.r[-1:1:0.1], stats=300L,
                         tBuf=5*ns, name='test swap spectroscopy 2 qubits', save=True, noisy=True):
    """
    swap between measure[0] and measure[1],  piPulse on the qubit(measure[0]) and swap to qubit(measure[1])
    measure both qubits
    """
    sample, devs, qubits = gc.loadQubits(Sample, measure=measure)

    axes = [(swapAmp, 'swap amplitude'), (delay, 'swap time')]
    qNames = [dev.__name__ for dev in devs if dev.get("readout", False)]

    deps = [("Mag", "%s" %(qNames[0]), ""), ("Phase", "%s" %(qNames[0]), "rad"),
            ("Mag", "%s" %(qNames[1]), ""), ("Phase", "%s" %(qNames[1]), "rad")]
    kw = {"stats": stats, 'tBuf': tBuf}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, currAmp, currDelay):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        q1 = alg.q1
        alg[gates.PiPulse([q0])]
        alg[gates.Detune([q0], currDelay, currAmp)]
        alg[gates.Wait([q0], tBuf)]
        alg[gates.Measure([q0, q1])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        mags, phases = readout.iqToPolar(readout.parseDataFormat(data, 'iq'))
        returnValue([mags[0], phases[0], mags[1], phases[1]])

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data

def testPiPulseXtalk(Sample, measure=(0,1), piPulse=(False, False),delay=st.r[0:200:2, ns],
                 stats=300L, tBuf=10*ns, name='test PiPulse xtalk', save=True, noisy=True):
    """
    piPulse = (False, True), represents no piPulse on measure[0] and piPulse on measure[1]
    """
    sample, devs, qubits = gc.loadQubits(Sample, measure)

    qNames = [dev.__name__ for dev in devs if dev.get("readout", False)]
    axes = [(delay, 'delay')]
    deps = [[("Mag", "%s" %(q), ""), ("Phase", "%s" %(q), "rad")] for q in qNames]
    deps = sum(deps, [])

    kw = {'stats': stats, 'tBuf': tBuf, "piPulse": piPulse}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, currDelay):
        alg = gc.Algorithm(devs)
        for q, pump in zip(alg.qubits, piPulse):
            if pump:
                alg[gates.PiPulse([q])]
                alg[gates.Wait([q], currDelay)]
        if not any(piPulse):
            alg[gates.Wait([alg.q0], currDelay)]
        alg[gates.Sync(alg.qubits)]
        for q in alg.qubits:
            alg[gates.Wait([q], tBuf)]
        alg[gates.Measure(alg.qubits)]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        mags, phases = readout.iqToPolar(readout.parseDataFormat(data, 'iq'))
        returnValue(np.array(zip(mags, phases)).flatten())

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data

def testRabiXtalk(Sample, measure=(0,1), rabiAmp=(0.1, 0.1), rabiLen=st.r[0:200:2, ns],
                  stats=300L, name='test Rabi xtalk', save=True, noisy=True):
    """
    rabi drive on qubits
    """
    sample, devs, qubits = gc.loadQubits(Sample, measure)

    qNames = [dev.__name__ for dev in devs if dev.get("readout", False)]
    axes = [(rabiLen, 'rabi drive time')]
    deps = [[("Mag", "%s" %(q), ""), ("Phase", "%s" %(q), "rad")] for q in qNames]
    deps = sum(deps, [])

    kw = {'stats': stats, "rabi amplitude": rabiAmp}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, currTime):
        alg = gc.Algorithm(devs)
        for q, amp in zip(alg.qubits, rabiAmp):
            alg[gates.RabiDrive([q], amp, currTime)]
        alg[gates.Measure(alg.qubits)]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        mags, phases = readout.iqToPolar(readout.parseDataFormat(data, 'iq'))
        returnValue(np.array(zip(mags, phases)).flatten())

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data

def testScurve(Sample, measure=(0, 1), readoutPower=st.r[-30:0:0.2, dBm], name='test Scurve Xtalk',
               stats=600L, save=True, noisy=True):
    """
    piPulse on other qubits, measure the scurve of the measure[0]
    """
    sample, devs, qubits = gc.loadQubits(Sample, measure)
    qname = devs[measure[0]].__name__
    idx = devs[measure[0]]['readoutOrder']

    axes = [(readoutPower, 'readoutPower')]
    deps = [("Mag", "%s-|0>" %qname, ""), ("Phase", "%s-|0>" %qname, "rad"),
            ("Mag", "%s-|1>" %qname, ""), ("Phase", "%s-|1>" %qname, "rad")]
    kw = {"stats": stats, }

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, currPower):
        alg = gc.Algorithm(devs)
        qm = alg.q0
        qm['readoutPower'] = currPower
        for q in alg.qubits[1:]:
            alg[gates.PiPulse([q])]
        alg[gates.Measure(alg.qubits)]
        alg.compile()
        data0 = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')

        alg = gc.Algorithm(devs)
        alg.q0['readoutPower'] = currPower
        for q in alg.qubits:
            alg[gates.PiPulse([q])]
        alg[gates.Measure(alg.qubits)]
        alg.compile()
        data1 = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')

        mag0, ang0 = readout.iqToPolar(readout.parseDataFormat(data0[idx][np.newaxis, ], 'iq'))
        mag1, ang1 = readout.iqToPolar(readout.parseDataFormat(data1[idx][np.newaxis, ], 'iq'))
        returnValue([mag0, ang0, mag1, ang1])

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)
    return data

def SparameterState(Sample, measure=(0,1), states=(0,0), freqScan=st.r[6:7:0.01, GHz], readoutPower=-30*dBm,
                    readoutLen=15*us, demodFreq=10*MHz, name='S parameter State', stats=600L, save=True, noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure)

    axes = [(freqScan, 'freq')]
    state = "".join(str(s) for s in states)
    state = " |" + state + ">"
    name += state
    deps = [("Mag", state, ""), ("Phase", state, "rad")]
    kw = {"stats": stats, "state": state, 'readoutPower': readoutPower}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, f):
        alg = gc.Algorithm(devs)
        for state, q in zip(states, alg.qubits):
            if state:
                alg[gates.MoveToState([q], 0, state)]
            else:
                alg[gates.Wait([q], q['piLen'])]
        for q in alg.qubits[1:]:
            q['readoutPower'] = -100*dBm
        q0 = alg.q0
        q0['readoutDevice']['carrierFrequency'] = f-demodFreq
        q0['readoutFrequency'] = f
        q0['readoutPower'] = readoutPower
        q0['readoutLen'] = readoutLen
        alg[gates.Measure(alg.qubits)]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        dat = readout.parseDataFormat(data[q0['readoutOrder']][np.newaxis, ], dataFormat='iq')
        mag, ang = readout.iqToPolar(dat)
        returnValue([mag, ang])

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data

def spectroscopy2DZxtalk(Sample, measure=(0,1), freqScan=st.r[4.5:6:0.01, GHz], zbias=st.r[-2:2:0.1],
                         sb_freq=0*GHz, stats=300L, name='Z-pulse xtalk spectroscopy 2D', save=True, noisy=True):
    """
    measure = [ measure qubit, z pulse qubit ]
    z_pulse on qubits[z_pulse], measure spectroscopy on qubits[measure]
    """
    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, True)

    axes = [(zbias, 'Z pulse Amplitude'), (freqScan, 'Frequency')]
    deps = [("Probability |1>", "%s" %(qubits[measure[0]].__name__), "")]

    kw = {'stats': stats, "sb_freq": sb_freq, "qubit measure": measure[0], "qubit z pulse": measure[1]}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure[0], kw=kw)
    # idx = sorted(measure).index(measure[0])
    idx = qubits[measure[0]]['readoutOrder']
    qubitNameCarrier = util.otherQubitNamesOnCarrier(devs[measure[0]], devs)

    def func(server, z, freq):
        alg = gc.Algorithm(devs)
        qm = alg.q0
        qz = alg.q1
        qz['readoutPower'] = -100*dBm # do not readout qz
        qm['fc'] = freq - sb_freq
        for name in qubitNameCarrier:
            alg.agents_dict[name]['fc'] = freq - sb_freq
        alg[gates.Spectroscopy([qm], df=sb_freq)]
        alg[gates.Detune([qz], tlen=(qm['spectroscopyLen']), amp=z)]
        alg[gates.Measure([qm, qz], align='start')]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        probs = readout.iqToProbs(data, alg.qubits, states=[0,1], correlated=False)
        returnValue([np.squeeze(probs[idx])[1]]) # only record data for qm

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data

def swapSpectroscopy(Sample, measure=(0,1), delay=st.r[0:200:2, ns], swapAmp=st.r[-1:1:0.1], stats=300L, herald=False,
                     herald_delay=500*ns, prob_correlated=False, tBuf=5*ns, name='swap spectroscopy 2 qubits', save=True, noisy=True):
    """
    swap between measure[0] and measure[1],  piPulse on the qubit(measure[0]) and swap to qubit(measure[1])
    measure both qubits
    """
    sample, devs, qubits = gc.loadQubits(Sample, measure=measure)

    axes = [(swapAmp, 'swap amplitude'), (delay, 'swap time')]
    qNames = [dev.__name__ for dev in devs if dev.get("readout", False)]

    deps = readout.genProbDeps(qubits, measure, correlated=prob_correlated)
    kw = {"stats": stats, 'tBuf': tBuf, 'prob_correlated': prob_correlated, 'herald': herald,
          'herald_delay': herald_delay}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, currAmp, currDelay):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        q1 = alg.q1
        if herald:
            alg[gates.Measure([q0, q1], name='herald')]
            alg[gates.Wait([q0], herald_delay)]
            alg[gates.Wait([q1], herald_delay)]
            alg[gates.Sync([q0, q1])]
        alg[gates.PiPulse([q0])]
        alg[gates.Detune([q0], currDelay, currAmp)]
        alg[gates.Wait([q0], tBuf)]
        alg[gates.Measure([q0, q1])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        probs = readout.iqToProbs(data, alg.qubits, correlated=prob_correlated, herald=herald)
        returnValue(np.squeeze(probs).flatten())

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data

def sqrtSwapSpectroscopy(Sample, measure=(0,1), swapAmp=st.r[-1:1:0.05], swapTime=st.r[0:100:1, ns], state=1,
                         tBuf=5*ns, stats=600, name='sqrt swap spectroscopy', prob_correlated=False,
                         save=True, noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure=measure)

    axes = [(swapAmp, 'swap amplitude'), (swapTime, 'swap length')]
    deps = readout.genProbDeps(qubits, measure, correlated=prob_correlated)
    kw = {'prob_correlated': prob_correlated, "stats": stats, "tBuf": tBuf, 'state': state}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, currAmp, currTime):
        alg = gc.Algorithm(devs)
        q0, q1 = alg.qubits
        alg[gates.MoveToState([q0], initState=0, endState=state)]
        alg[gates.QubitSqrtSwap([q0, q1], tlen=currTime, amp=currAmp)]
        alg[gates.Wait([q0], tBuf)]
        alg[gates.QubitSqrtSwap([q0, q1], tlen=currTime, amp=currAmp)]
        alg[gates.Wait([q0], tBuf)]
        alg[gates.Sync([q0, q1])]
        alg[gates.Measure([q0, q1])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        probs = readout.iqToProbs(data, alg.qubits, correlated=prob_correlated)
        returnValue(probs.flatten())

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data