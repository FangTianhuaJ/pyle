import numpy as np
import pyle

from labrad.units import Unit,Value
V, mV, us, ns, GHz, MHz, dBm, rad = [Unit(s) for s in ('V', 'mV', 'us', 'ns', 'GHz', 'MHz', 'dBm', 'rad')]

from pyle.dataking import multiqubitPQ as mq
import pyle.envelopes as env
from pyle.pipeline import returnValue, FutureList
from pyle.dataking import envelopehelpers as eh
from pyle.dataking import measurement
from pyle.dataking.fpgaseq import runQubits as runQubits
from pyle.util import sweeptools as st
from pyle.dataking import utilMultilevels as ml
from pyle.plotting import dstools
from pyle.fitting import fitting
from pyle.util import structures
from pyle.dataking.benchmarking import danBench as db
from pyle.dataking import sweeps
from pyle.dataking import util

import pyle.analysis as analysis


import labrad

# CHECKLIST FOR CZ TUNE UP
# . Room temperature z pulse calibrations
# . S-curve. Rough, just to get pulse calibration to work
# . Testdelay
# . Pulse shape calibration
# . Better scurve
# . Pi amp and frequency
# . calscurve
# . zpa func search
# . swapSpectroscopy to find bus both states
# . swapTuner
# . timing with respect to master
# . zpa crosstalk!
# . Measurement crosstalk matrix business
# . Ramseys

def overshootTuner(sample, paramName, measure=0, swapAmp=None, swapTime=None, overshoot=None,
                   iterations=3, overshootBound=0.025, state=1,
                   save=False,update=True,stats=1200,noisy=False):
    """Finds the best qubit z-pulse amplitude and pulse length to do a swap into a resonator"""
    name = 'Overshoot Tuner |%d>' %state
    sample, qubits, Qubits = util.loadDeviceType(sample, 'phaseQubit', write_access=True)
    qubit=qubits[measure]
    Qubit=Qubits[measure]
    paramName += str(state) if state>1 else ''
    if swapAmp is None:
        swapAmp = qubit['swapAmp'+paramName]
    if swapTime is None:
        swapTime= qubit['swapTime'+paramName]
    if overshoot is None:
        overshoot = qubit['swapOvershoot'+paramName]
    if noisy:
        print 'Original swap overshoot: %f' %qubit['swapOvershoot'+paramName]
    for iteration in range(iterations):
        dataStats=stats*(2**iteration)
        dSwapOvershoot = overshootBound/(2.0**iteration)
        overshootScan = np.linspace(overshoot-dSwapOvershoot,overshoot+dSwapOvershoot,21)
        overshootScan = overshootScan[overshootScan>0]
        data = mq.swapSpectroscopy(sample,
                                   swapLen=swapTime, swapAmp=swapAmp,
                                   overshoot = overshootScan,
                                   measure=measure, state=state, collect=True, save=save,
                                   stats=dataStats, noisy=noisy, name=name)
        overshoot,probability = fitting.findMinimum(np.array(data),fit=True)
    return overshoot

def czRamseyControl(s, control, target, paramNameC, paramNameT,
                    targetCompensationAngle=0.0, controlCompensationAngle=None,
                    delay=3.0*ns,
                    name = 'czRamseyControl', stats=600,
                    collect=False, save=True, noisy=True):

    sample, qubits = util.loadQubits(s)
    qC = qubits[control]
    qT = qubits[target]

    qC['readout'] = True

    if controlCompensationAngle is None:
        controlCompensationAngle = np.linspace(-2*np.pi*0.15,2*np.pi*1.75,100)

    axes = [(targetCompensationAngle,'targetCompensationAngle'),(controlCompensationAngle,'controlCompensationAngle')]
    kw = {'stats': stats, 'targetCompensationAngle': targetCompensationAngle}
    dataset = sweeps.prepDataset(sample, name, axes, measure=control, kw=kw)

    def func(server, targetCompensationAngle, controlCompensationAngle):
        t = 0.0*ns
        qC['xy'] = qT['xy'] = env.NOTHING
        for q in qubits:
            q['z'] = env.NOTHING
        #Excite control
        qC['xy'] += eh.piHalfPulse(qC, t, phase = 0)
        t += 0.5*max(qC['piLen'],qT['piLen'])
        #iSwap control into resonator
        qC['z'] += env.rect(t, qC['swapTime'+paramNameC], qC['swapAmp'+paramNameC])
        t += qC['swapTime'+paramNameC]
        #Do (iSwap)**2 on target
        qT['z'] += env.rect(t, qT['swapTime'+paramNameT+'2']*2, qT['swapAmp'+paramNameT+'2'])
        t += qT['swapTime'+paramNameT+'2']*2
        #Retrieve photon from resonator into control
        qC['z'] += env.rect(t, qC['swapTime'+paramNameC], qC['swapAmp'+paramNameC])
        t += qC['swapTime'+paramNameC]
        #Compensate control and target phases
        t += 0.5*max(qC['piLen'],qT['piLen'])
        t += delay
        qC['z'] += eh.rotPulseZ(qC, t, controlCompensationAngle)
        t += 0.5*max(qC['piLen'],qT['piLen'])
        #Final pi pulses for Ramsey
        t += 0.5*max(qC['piLen'],qT['piLen'])
        qC['xy'] += eh.piHalfPulse(qC, t, phase = 0)
        t += 0.5*qC['piLen']
        #Measure control
        qC['z'] += eh.measurePulse(qC, t)

        #Apply sideband mixing
        qC['xy'] = eh.mix(qC, qC['xy'])
        eh.correctCrosstalkZ(qubits)

        return runQubits(server, qubits, stats, probs=[1])

    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if collect:
        return data

def czRamseyTarget(s, control, target, paramNameC, paramNameT, controlCompensationAngle=None, targetCompensationAngle=None,
                   delay=3.0*ns,
                   name = 'czRamseyTarget', stats=600,
                   collect=False, save=True, noisy=True):

    sample, qubits = util.loadQubits(s)
    qC = qubits[control]
    qT = qubits[target]

    qT['readout'] = True

    if targetCompensationAngle is None:
        targetCompensationAngle = np.linspace(-2*np.pi*0.15,2*np.pi*1.75,100)

    if controlCompensationAngle is None:
        controlCompensationAngle = qC['czControlCompAngle']

    axes = [(targetCompensationAngle,'targetCompensationAngle'),(controlCompensationAngle,'controlCompensationAngle')]
    deps = [('Probability', 'Control = |%d>'%state, '') for state in [0,1]]
    kw = {'stats': stats, 'targetCompensationAngle': targetCompensationAngle}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=target, kw=kw)

    def func(server, targetCompensationAngle, controlCompensationAngle):
        reqs=[]
        for controlPi in [False, True]:
            t = 0.0*ns
            for q in qubits:
                q['z'] = env.NOTHING
                q['xy'] = env.NOTHING
            #Excite control
            if controlPi:
                qC['xy'] += eh.piPulse(qC, t, phase = 0)
            qT['xy'] += eh.piHalfPulse(qT, t, phase = 0)
            t += 0.5*max(qC['piLen'],qT['piLen'])
            #iSwap control into resonator
            qC['z'] += env.rect(t, qC['swapTime'+paramNameC], qC['swapAmp'+paramNameC])
            t += qC['swapTime'+paramNameC]
            #Do (iSwap)**2 on target
            qT['z'] += env.rect(t, qT['swapTime'+paramNameT+'2']*2, qT['swapAmp'+paramNameT+'2'])
            t += qT['swapTime'+paramNameT+'2']*2
            #Retrieve photon from resonator into control
            qC['z'] += env.rect(t, qC['swapTime'+paramNameC], qC['swapAmp'+paramNameC])
            t += qC['swapTime'+paramNameC]
            #Compensate control and target phases
            t += 0.5*max(qC['piLen'],qT['piLen'])
            t += delay
            qC['z'] += eh.rotPulseZ(qC, t, controlCompensationAngle)
            qT['z'] += eh.rotPulseZ(qT, t, targetCompensationAngle)
            t += 0.5*max(qC['piLen'],qT['piLen'])
            #Final pi pulses for Ramsey
            t += 0.5*max(qC['piLen'],qT['piLen'])
            qT['xy'] += eh.piHalfPulse(qT, t, phase = 0)
            t += 0.5*qT['piLen']
            #Measure target
            qT['z'] += eh.measurePulse(qT, t)

            #Apply sideband mixing
            qC['xy'] = eh.mix(qC, qC['xy'])
            qT['xy'] = eh.mix(qT, qT['xy'])
            eh.correctCrosstalkZ(qubits)

            reqs.append(runQubits(server, qubits, stats, probs=[1]))
        data = yield FutureList(reqs)
        probs = [p[0] for p in data]
        returnValue(probs)

    data = sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)
    if collect:
        return data

def czBellStateTomo(s, control, target, paramNameC, paramNameT, controlCompensationAngle=None, targetCompensationAngle=None,
                    tBuf=3.0*ns,
                    name = 'czBellTomo', stats=600,
                    collect=False, save=True, noisy=True):

    sample, qubits = util.loadQubits(s)
    qC = qubits[control]
    qT = qubits[target]
    measure = [control, target]
    qT['readout'] = True
    qC['readout'] = True
    N = len(s['config'])
    measureFunc = measurement.Tomo(N, measure, tBuf=tBuf,)

    if targetCompensationAngle is None:
        targetCompensationAngle = qT['czTargetCompAngle']

    if controlCompensationAngle is None:
        controlCompensationAngle = qC['czControlCompAngle']

    axes = [(targetCompensationAngle,'targetCompensationAngle'),(controlCompensationAngle,'controlCompensationAngle')]
    kw = {'stats': stats, 'targetCompensationAngle': targetCompensationAngle}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measureFunc, kw=kw)

    def func(server, targetCompensationAngle, controlCompensationAngle):
        reqs=[]
        #Initialize everything
        t = 0.0*ns
        for q in qubits:
            q['z'] = env.NOTHING
            q['xy'] = env.NOTHING
        #Single qubit preparations
        qC['xy'] += eh.piHalfPulse(qC, t, phase = np.pi/2)
        qT['xy'] += eh.piHalfPulse(qT, t, phase = np.pi/2)
        t += 0.5*max(qC['piLen'],qT['piLen'])
        ### BEGIN CONTROLLED Z GATE ###
        #iSwap control into resonator
        qC['z'] += env.rect(t, qC['swapTime'+paramNameC], qC['swapAmp'+paramNameC])
        t += qC['swapTime'+paramNameC]
        #Do (iSwap)**2 on target
        qT['z'] += env.rect(t, qT['swapTime'+paramNameT+'2']*2, qT['swapAmp'+paramNameT+'2'])
        t += qT['swapTime'+paramNameT+'2']*2
        #Retrieve photon from resonator into control
        qC['z'] += env.rect(t, qC['swapTime'+paramNameC], qC['swapAmp'+paramNameC])
        t += qC['swapTime'+paramNameC]
        #Compensate control and target phases
        t += 0.5*max(qC['piLen'],qT['piLen'])
        t += tBuf
        qC['z'] += eh.rotPulseZ(qC, t, controlCompensationAngle)
        qT['z'] += eh.rotPulseZ(qT, t, targetCompensationAngle)
        t += 0.5*max(qC['piLen'],qT['piLen'])
        ### END CONTROLLED Z GATE ###
        t += qT['piLen']/2
        qT['xy'] += eh.piHalfPulse(qT, t, phase=np.pi/2)
        t += qT['piLen']/2
        #Apply sideband mixing
        qC['xy'] = eh.mix(qC, qC['xy'])
        qT['xy'] = eh.mix(qT, qT['xy'])
        eh.correctCrosstalkZ(qubits)
        return measureFunc(server, qubits, t+tBuf, stats=stats)
    data = sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy, pipesize=1)
    if collect:
        return data

def czRamseyControlRect(s, control, target, paramNameC, paramNameT, doTargetComp = False,
                        controlCompensationAmp=None,
                        tBuf=5.0*ns,
                        name = 'czRamseyControl', stats=600,
                        collect=False, save=True, noisy=True):

    sample, qubits = util.loadQubits(s)
    qC = qubits[control]
    qT = qubits[target]

    qC['readout'] = True

    if controlCompensationAmp is None:
        controlCompensationAmp = np.linspace(-0.05, 0.05, 100)

    axes = [(controlCompensationAmp,'controlCompensationAmp')]
    kw = {'stats': stats,'control':control,'target':target}
    if doTargetComp:
        kw['targetCompensationAmp'] = qT['czTargetCompAmp']
    dataset = sweeps.prepDataset(sample, name, axes, measure=control, kw=kw)

    def func(server, controlCompensationAmp):
        t = 0.0*ns
        qC['xy'] = qT['xy'] = env.NOTHING
        for q in qubits:
            q['z'] = env.NOTHING
        #Excite control
        qC['xy'] += eh.piHalfPulse(qC, t, phase = 0)
        t += 0.5*max(qC['piLen'],qT['piLen']) + tBuf
        #iSwap control into resonator
        qC['z'] += env.rect(t, qC['swapTime'+paramNameC], qC['swapAmp'+paramNameC])
        t += qC['swapTime'+paramNameC] + tBuf
        #Do (iSwap)**2 on target
        qT['z'] += env.rect(t, qT['swapTime'+paramNameT+'2']*2, qT['swapAmp'+paramNameT+'2'])
        t += qT['swapTime'+paramNameT+'2']*2 + tBuf
        #Retrieve photon from resonator into control
        qC['z'] += env.rect(t, qC['swapTime'+paramNameC], qC['swapAmp'+paramNameC])
        t += qC['swapTime'+paramNameC] + tBuf
        #Compensate control and target phases
        qC['z'] += env.rect(t, qC['czControlCompTime'], controlCompensationAmp)
        if doTargetComp:
            qT['z'] += env.rect(t, qT['czTargetCompTime'], qT['czTargetCompAmp'])
        t += max(qC['czControlCompTime'], qT['czTargetCompTime']) + tBuf
        #Final pi pulses for Ramsey
        t += 0.5*max(qC['piLen'],qT['piLen'])
        qC['xy'] += eh.piHalfPulse(qC, t, phase = 0)
        t += 0.5*qC['piLen'] + tBuf
        #Measure control
        qC['z'] += eh.measurePulse(qC, t)

        #Apply sideband mixing
        qC['xy'] = eh.mix(qC, qC['xy'])
        eh.correctCrosstalkZ(qubits)

        return runQubits(server, qubits, stats, probs=[1])

    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if collect:
        return data

def czRamseyTargetRect(s, control, target, paramNameC, paramNameT, controlCompensationAmp=None, targetCompensationAmp=None,
                       tBuf=5.0*ns,
                       name = 'czRamseyTarget', stats=600,
                       collect=False, save=True, noisy=True):

    sample, qubits = util.loadQubits(s)
    qC = qubits[control]
    qT = qubits[target]

    qT['readout'] = True

    if targetCompensationAmp is None:
        targetCompensationAmp = np.linspace(-0.1,0.1,100)

    if controlCompensationAmp is None:
        controlCompensationAmp = qC['czControlCompAmp']

    axes = [(targetCompensationAmp,'targetCompensationAmp'),(controlCompensationAmp,'controlCompensationAmp')]
    deps = [('Probability', 'Control = |%d>'%state, '') for state in [0,1]]
    kw = {'stats': stats, 'controlCompensationAmp': controlCompensationAmp,'control':control,'target':target}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=target, kw=kw)

    def func(server, targetCompensationAmp, controlCompensationAmp):
        reqs=[]
        for controlPi in [False, True]:
            t = 0.0*ns
            for q in qubits:
                q['z'] = env.NOTHING
                q['xy'] = env.NOTHING
            #Excite control
            if controlPi:
                qC['xy'] += eh.piPulse(qC, t, phase = 0)
            qT['xy'] += eh.piHalfPulse(qT, t, phase = 0)
            t += 0.5*max(qC['piLen'],qT['piLen']) + tBuf
            #iSwap control into resonator
            qC['z'] += env.rect(t, qC['swapTime'+paramNameC], qC['swapAmp'+paramNameC])
            t += qC['swapTime'+paramNameC] + tBuf
            #Do (iSwap)**2 on target
            qT['z'] += env.rect(t, qT['swapTime'+paramNameT+'2']*2, qT['swapAmp'+paramNameT+'2'])
            t += qT['swapTime'+paramNameT+'2']*2 + tBuf
            #Retrieve photon from resonator into control
            qC['z'] += env.rect(t, qC['swapTime'+paramNameC], qC['swapAmp'+paramNameC])
            t += qC['swapTime'+paramNameC] + tBuf
            #Compensate control and target phases
            qC['z'] += env.rect(t, qC['czControlCompTime'], controlCompensationAmp)
            qT['z'] += env.rect(t, qT['czTargetCompTime'], targetCompensationAmp)
            t += max(qC['czControlCompTime'],qT['czTargetCompTime']) + tBuf
            #Final pi pulses for Ramsey
            t += 0.5*max(qC['piLen'],qT['piLen'])
            qT['xy'] += eh.piHalfPulse(qT, t, phase = 0)
            t += 0.5*qT['piLen'] + tBuf
            #Measure target
            qT['z'] += eh.measurePulse(qT, t)

            #Apply sideband mixing
            qC['xy'] = eh.mix(qC, qC['xy'])
            qT['xy'] = eh.mix(qT, qT['xy'])
            eh.correctCrosstalkZ(qubits)

            reqs.append(runQubits(server, qubits, stats, probs=[1]))
        data = yield FutureList(reqs)
        probs = [p[0] for p in data]
        returnValue(probs)

    data = sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)
    if collect:
        return data

def czBellStateTomoRect(s, control, target, paramNameC, paramNameT, controlCompensationAmp=None, targetCompensationAmp=None,
                        tBuf=5.0*ns, tBufMeasure = 5.0*ns, tomoType = 'tomo',
                        name = 'czBellTomo', stats=600,
                        collect=False, save=True, noisy=True):

    sample, qubits = util.loadQubits(s)
    qC = qubits[control]
    qT = qubits[target]
    measure = [control, target]
    qT['readout'] = True
    qC['readout'] = True
    N = len(s['config'])

    if tomoType == 'tomo':
        measureFunc = measurement.Tomo(N, measure, tBuf=tBufMeasure)
    elif tomoType == 'octomo':
        measureFunc = measurement.Octomo(N, measure, tBuf=tBufMeasure)

    if targetCompensationAmp is None:
        targetCompensationAmp = qT['czTargetCompAmp']

    if controlCompensationAmp is None:
        controlCompensationAmp = qC['czControlCompAmp']

    axes = [(targetCompensationAmp,'targetCompensationAmp'),(controlCompensationAmp,'controlCompensationAmp')]
    kw = {'stats': stats, 'targetCompensationAmp': targetCompensationAmp, 'controlCompensationAmp': controlCompensationAmp, 'measureType': tomoType,'control':control,'target':target}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measureFunc, kw=kw)

    def func(server, targetCompensationAmp, controlCompensationAmp):
        reqs=[]
        #Initialize everything
        t = 0.0*ns
        for q in qubits:
            q['z'] = env.NOTHING
            q['xy'] = env.NOTHING
        #Single qubit preparations
        qC['xy'] += eh.piHalfPulse(qC, t, phase = np.pi/2)
        qT['xy'] += eh.piHalfPulse(qT, t, phase = np.pi/2)
        t += 0.5*max(qC['piLen'],qT['piLen']) + tBuf
        ### BEGIN CONTROLLED Z GATE ###
        #iSwap control into resonator
        qC['z'] += env.rect(t, qC['swapTime'+paramNameC], qC['swapAmp'+paramNameC])
        t += qC['swapTime'+paramNameC] + tBuf
        #Do (iSwap)**2 on target
        qT['z'] += env.rect(t, qT['swapTime'+paramNameT+'2']*2, qT['swapAmp'+paramNameT+'2'])
        t += qT['swapTime'+paramNameT+'2']*2 + tBuf
        #Retrieve photon from resonator into control
        qC['z'] += env.rect(t, qC['swapTime'+paramNameC], qC['swapAmp'+paramNameC])
        t += qC['swapTime'+paramNameC] + tBuf
        #Compensate control and target phases
        qC['z'] += env.rect(t, qC['czControlCompTime'], controlCompensationAmp)
        qT['z'] += env.rect(t, qT['czTargetCompTime'], targetCompensationAmp)
        t += max(qC['czControlCompTime'],qT['czTargetCompTime']) + tBuf
        ### END CONTROLLED Z GATE ###
        t += qT['piLen']/2
        qT['xy'] += eh.piHalfPulse(qT, t, phase=np.pi/2)
        t += qT['piLen']/2 + tBuf
        #Apply sideband mixing
        qC['xy'] = eh.mix(qC, qC['xy'])
        qT['xy'] = eh.mix(qT, qT['xy'])
        eh.correctCrosstalkZ(qubits)
        return measureFunc(server, qubits, t, stats=stats)
    data = sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy, pipesize=1)
    if collect:
        return data

def czQPTRect(s, control, target, paramNameC, paramNameT,
                        tBuf=5.0*ns, tBufMeasure = 5.0*ns, tomoType = 'tomo',
                        name = 'czBellTomo', stats=600,
                        collect=False, save=True, noisy=True):

    sample, qubits = util.loadQubits(s)
    qC = qubits[control]
    qT = qubits[target]
    measure = [control, target]
    qT['readout'] = True
    qC['readout'] = True
    N = len(s['config'])

    if tomoType == 'tomo':
        measureFunc = measurement.Tomo(N, measure, tBuf=tBufMeasure)
    elif tomoType == 'octomo':
        measureFunc = measurement.Octomo(N, measure, tBuf=tBufMeasure)

    prepOps = [
        ('I', 0, 0),
        ('Xpi', 1, 0),
        ('Ypi/2', 0.5, 0.5),
        ('Xpi/2', 0.5, 1.0)]
    opNumbers = range(len(prepOps)**2)
    opNames = []
    opAmps = []
    opPhases = []
    for opC in prepOps:
        for opT in prepOps:
            opNames.append(opC[0]+opT[0])       #name
            opAmps.append([opC[1],opT[1]])      #amplitude
            opPhases.append([opC[2],opT[2]])    #phase

    axes = [(opNumbers,'Prep Op Number')]
    kw = {'stats': stats, 'target': target, 'control': control, 'prepOps': prepOps, 'prepNames': opNames}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measureFunc, kw=kw)

    def func(server, opNumber):
        reqs=[]
        #Initialize everything
        t = 0.0*ns
        for q in qubits:
            q['z'] = env.NOTHING
            q['xy'] = env.NOTHING
        #Single qubit preparations
        qC['xy'] += eh.rotPulse(qC, t, angle = opAmps[opNumber][0]*np.pi, phase = opPhases[opNumber][0]*np.pi)
        qT['xy'] += eh.rotPulse(qT, t, angle = opAmps[opNumber][1]*np.pi, phase = opPhases[opNumber][1]*np.pi)
        t += 0.5*max(qC['piLen'],qT['piLen']) + tBuf
        ### BEGIN CONTROLLED Z GATE ###
        #iSwap control into resonator
        qC['z'] += env.rect(t, qC['swapTime'+paramNameC], qC['swapAmp'+paramNameC])
        t += qC['swapTime'+paramNameC] + tBuf
        #Do (iSwap)**2 on target
        qT['z'] += env.rect(t, qT['swapTime'+paramNameT+'2']*2, qT['swapAmp'+paramNameT+'2'])
        t += qT['swapTime'+paramNameT+'2']*2 + tBuf
        #Retrieve photon from resonator into control
        qC['z'] += env.rect(t, qC['swapTime'+paramNameC], qC['swapAmp'+paramNameC])
        t += qC['swapTime'+paramNameC] + tBuf
        #Compensate control and target phases
        qC['z'] += env.rect(t, qC['czControlCompTime'], qC['czControlCompAmp'])
        qT['z'] += env.rect(t, qT['czTargetCompTime'], qT['czTargetCompAmp'])
        t += max(qC['czControlCompTime'],qT['czTargetCompTime']) + tBuf
        ### END CONTROLLED Z GATE ###
        #Apply sideband mixing
        qC['xy'] = eh.mix(qC, qC['xy'])
        qT['xy'] = eh.mix(qT, qT['xy'])
        eh.correctCrosstalkZ(qubits)
        return measureFunc(server, qubits, t, stats=stats)
    data = sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy, pipesize=1)
    if collect:
        return data

