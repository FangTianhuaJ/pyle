import numpy as np
from scipy.optimize import leastsq
from scipy.special import erf, erfc
import pylab as plt

import labrad
from labrad.units import Unit
V, mV, us, ns, GHz, MHz = [Unit(s) for s in ('V', 'mV', 'us', 'ns', 'GHz', 'MHz')]

import pyle
from pyle.pipeline import returnValue, FutureList
from pyle.dataking import util
from pyle import envelopes as env
from pyle.dataking import envelopehelpers as eh
from pyle.dataking import measurement as meas
from pyle.dataking import sweeps
from pyle.dataking import adjust
from pyle.dataking.fpgaseq import runQubits
from pyle.util import sweeptools as st
from pyle.analysis import entanglementAnalysis as et
from pyle.plotting import dstools
#import pdb

def iSwap(sample, swapLen=st.r[-20:1000:1,ns], measure=[0,1], stats=1500L,
         name='iSwap MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]
    #nameEx = [' q0->q1',' q1->q0']
    axes = [(swapLen, 'swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)#+nameEx[measure[0]]

    def func(server, curr):
        start = 0
        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2
        q0.z = env.rect(start, curr, q0.cZControlAmp)#amplitude for an iSWAP
        start += q0.cZControlLen+curr
        q1.z = env.rect(start, q1.cZControlLen, q1.cZControlAmp)#amplitude & length for an iSWAP
        start += q1.cZControlLen
        q0.z += eh.measurePulse(q0, start)
        q1.z += eh.measurePulse(q1, start)

        q0['readout'] = True
        q1['readout'] = True
        return runQubits(server, qubits, stats=stats)
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)



def bellStateiSwap(sample, reps=5, measure=[0,1], stats=1500L, corrAmp=None,
         name='Bell State with an iSwap MQ', save=True, collect=False, noisy=True):
    """Make any of the four Bell states by varying the corrAmplitude (empirically for now)
    TODO: make this a function of Zpi amplitude so there is some meaning to the amplitude and
    consequent phase shift. """
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    measurement = meas.Octomo(4, measure)

    if corrAmp is None:
        corrAmp = q0['piAmpZ']
    repetition = range(reps)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measurement, kw=kw)

    def func(server, curr):
        start = 0
        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start, phase=0.5*np.pi))
        start += q0['piLen']/2
        q0.z = env.rect(start, q0.cZControlLen/2.0, q0.cZControlAmp)#TIME for an SQRT(iSWAP)
        start += q0.cZControlLen/2.0
        q1.z = env.rect(start, q1.cZControlLen, q1.cZControlAmp)#amplitude & length for an iSWAP
        #add compensation pulse for different Bell-states
        q0.z += env.rect(start, q0['piFWHM'], corrAmp) #calibrate around Z-pi pulses
        #pad time by half the length of pi-pulse for no-overlap with tomo-pulses
        start += q1.cZControlLen + q0['piLen']/2
        return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    rho_ideal = np.array([[0.0j,0.0j,0.0j,0.0j],[0.0j,0.5,0.5j,0.0j],[0j,-0.5j,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)

    Qk = np.reshape(result[1:],(36,4))
    Qk = readoutFidCorr(Qk, [q0.measureF0,q0.measureF1,q1.measureF0,q1.measureF1])
#

    rho_cal = pyle.tomo.qst(Qk,'octomo2')
    plotRhoSingle(rho_cal)
    eFormation = et.eof(rho_cal)
    rho_caln = rho_cal.copy()
#    rho_caln[1,2] = abs(rho_caln[1,2])
#    rho_caln[2,1] = abs(rho_caln[2,1])
    print 'Fidelity is %g ' % np.trace(np.dot(rho_caln,rho_ideal))
    print 'Entangle of Formation is %g' % eFormation

    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal))


def twoiSwap(sample, swapLen=st.r[-20:1000:1,ns], measure=[0,1], stats=1500L,
         name='multiSwapper MQ', save=True, collect=False, noisy=True):
    """Fock state = 2, by pumping one photon from two qubits, measuring with second qubit """
    sample, qubits = util.loadQubits(sample)
    qi = qubits[measure[0]]
    qj = qubits[measure[1]]

    axes = [(swapLen, 'swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)#+nameEx[measure[0]]

    def func(server, curr):
        start = 0
        #pi, then iSWAP e->1
        qi.xy = eh.mix(qi, eh.piPulseHD(qi, start))
        start += qi['piLen']/2
        qi.z = env.rect(start, qi.cZControlLen, qi.cZControlAmp)
        start += qi.cZControlLen
        #pi, then iSWAP e->2
        qj.xy = eh.mix(qj, eh.piPulseHD(qj, start))
        start += qj['piLen']/2
        qj.z = env.rect(start, curr, qj.cZControlAmp)#same amp, but swap is faster controlLen is cal'd for N=1 photon in Resonator
        start += curr

        qi.z += eh.measurePulse(qi, start)
        qj.z += eh.measurePulse(qj, start)


        qi['readout'] = True
        qj['readout'] = True

        return runQubits(server, qubits, stats=stats)
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

def multiSwapper(sample, swapLen=st.r[-20:1000:1,ns], measure=[0,1,2], stats=1500L,
         name='multiSwapper MQ', save=True, collect=False, noisy=True):
    """Fock state = 2, by pumping one photon from two qubits [i,j], readout in third [k] """
    sample, qubits = util.loadQubits(sample)
    qi = qubits[measure[0]]
    qj = qubits[measure[1]]
    qk = qubits[measure[2]]

    #nameEx = [' q0->q1',' q1->q0']
    axes = [(swapLen, 'swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)#+nameEx[measure[0]]

    def func(server, curr):
        start = 0
        #pi, then iSWAP e->1
        qi.xy = eh.mix(qi, eh.piPulseHD(qi, start))
        start += qi['piLen']/2
        qi.z = env.rect(start, qi.cZControlLen, qi.cZControlAmp)
        start += qi.cZControlLen
        #pi, then iSWAP e->2
        qj.xy = eh.mix(qj, eh.piPulseHD(qj, start))
        start += qj['piLen']/2
        qj.z = env.rect(start, qj.cZTargetLen/2, qj.cZControlAmp)#same Control amp, but swap is faster TargetLen is cal'd for N=1 photon in Resonator
        start += qj.cZTargetLen/2
        #pull out photon N=2 into third ('k') qubit
        qk.z = env.rect(start, curr, qk.cZControlAmp)
        start += curr

        qi.z += eh.measurePulse(qi, start)
        qj.z += eh.measurePulse(qj, start)
        qk.z += eh.measurePulse(qk, start)

        qi['readout'] = True
        qj['readout'] = True
        qk['readout'] = True
        return runQubits(server, qubits, stats=stats)
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)


def cZCalP1(sample, targetAmp=st.r[-0.25:0:0.001], measureC=0, measureT=1, stats=1500L,
         name='Control-Z Step 1 TargetCal MQ', save=True, collect=False, noisy=True, update=False):
    """Generalized Ramsey. Performs the controlled-Z gate Z-pulse sequence
    with pi/2 pulse on target and no microwaves on control qubit [qt, qc]
    to calibrate the phase correction on the target qubit. Find any maximum of the Ramsey."""

    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    qc = qubits[measureC] #control qubit
    qt, Qt = qubits[measureT], Qubits[measureT] #target qubit

    # repetition = range(repetition)
    axes = [(targetAmp, 'target amp phase correction')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measureT, kw=kw)

    def func(server, targetAmp):
        start = 0
        #pad time for equal state prep time in Step 2 Cal
        start += qc['piLen']/2 + qc['cZControlLen']
        #state prep
        #Control qubit no microwaves, Target qubit pi/2
        qt.xy = eh.mix(qt, eh.piHalfPulseHD(qt, start))
        start += qt['piLen']/2
        #Control is IDLE

        #Target Phase swap Q21 with R21 for iswap^2 time
        qt.z = env.rect(start, qt.cZTargetLen, qt.cZTargetAmp)
        start += qt.cZTargetLen

        #Target phase correction, time is fixed sweeping amplitude
        qt.z += env.rect(start, qt.cZTargetPhaseCorrLen, targetAmp)
        start += qt.cZTargetPhaseCorrLen + qt['piLen']/2
        #Final pi/2 for Ramsey, rotate about X
        qt.xy += eh.mix(qt, eh.piHalfPulseHD(qt, start, phase=0.0*np.pi))
        start += qt['piLen']/2

        #Measure only the Control
        qt.z += eh.measurePulse(qt, start)

        qt['readout'] = True

        return runQubits(server, qubits, stats=stats, probs=[1])
#    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    if update:
        adjust.adjust_cZTargetPhaseCorrAmp(Qt, data)

def cZCalP2(sample, targetAmp=st.r[-0.25:0:0.001], measureC=0, measureT=1, stats=1500L,
         name='Control-Z Step 2 TargetCal MQ', save=True, collect=False, noisy=True):
    """Generalized Ramsey. Performs the controlled-Z gate Z-pulse sequence
    with pi/2 pulse on target ann a pi-pulse on the control qubit [qc, qt]
    to verify the "pi" phase shift from cZCalP1 on the target qubit.
    Look for a Min, should be really close to Max from Part 1"""

    sample, qubits = util.loadQubits(sample)
    qc = qubits[measureC] #control qubit
    qt = qubits[measureT] #target qubit

    axes = [(targetAmp, 'target amp phase correction')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measureT, kw=kw)

    def func(server, targetAmp):
        start = 0
        #state prep
        #Control g -> e
        qc.xy = eh.mix(qc, eh.piPulseHD(qc, start))
        start += qc['piLen']/2
        #Control iSWAP with Resonator
        qc.z = env.rect(start, qc.cZControlLen, qt.cZControlAmp)
        start += qt.cZControlLen
        #state prep Target
        qt.xy = eh.mix(qt, eh.piHalfPulseHD(qt, start))
        start += qt['piLen']/2
        #Target Phase swap Q21 with R21 for iswap^2 time
        qt.z = env.rect(start, qt.cZTargetLen, qc.cZTargetAmp)
        start += qt.cZTargetLen

        #Target phase correction, time is fixed sweeping amplitude
        qt.z += env.rect(start, qt.cZTargetPhaseCorrLen, targetAmp)
        start += qt.cZTargetPhaseCorrLen + qt['piLen']/2
        #Final pi/2 for Ramsey, rotate about X
        qt.xy += eh.mix(qt, eh.piHalfPulseHD(qt, start, phase=0.0*np.pi))
        start += qt['piLen']/2

        #Measure
        qt.z += eh.measurePulse(qt, start)

        qt['readout'] = True

        return runQubits(server, qubits, stats=stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

def cZCalP3(sample,controlAmp=st.r[-0.25:0:0.001], measureC=0, measureT=1, stats=1500L,
         name='Control-Z Step 3 ControlCal MQ', save=True, collect=False, noisy=True):
    """Generalized Ramsey. Performs the controlled-Z gate Z-pulse sequence
    with pi/2 pulse on control and no microwaves on target qubit [qc, qt].
    This is a check experiment to calibrate the phase correction on the control qubit.
    Look for Max (probably near 0.0).
    Note this experiment is orthogonal to Cal P1 and Cal P2, you do not need to iterate"""

    sample, qubits = util.loadQubits(sample)
    qc = qubits[measureC] #control qubit
    qt = qubits[measureT] #target qubit

    # repetition = range(repetition)
    axes = [(controlAmp, 'control amp phase correction')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measureC, kw=kw)

    def func(server, controlAmp):
        start = 0
        #state prep
        #Control g -> g + e
        qc.xy = eh.mix(qc, eh.piHalfPulseHD(qc, start))
        start += qc['piLen']/2
        #Control iSWAP with Resonator
        qc.z = env.rect(start, qc.cZControlLen, qc.cZControlAmp)
        start += qc.cZControlLen
        #Target NO microwaves, but need to pad time for consistent sequence with CZCal Part 1,2
        start += qt['piLen']/2
        #Target Phase swap Q21 with R21 for iswap^2 time
        qt.z = env.rect(start, qt.cZTargetLen, qt.cZTargetAmp)
        start += qt.cZTargetLen

        #Target phase correction, time is fixed amp calibrated in Step 1,2
        qt.z += env.rect(start, qt.cZTargetPhaseCorrLen, qt.cZTargetPhaseCorrAmp)
        #Control iSWAP with Resonator
        qc.z += env.rect(start, qc.cZControlLen, qc.cZControlAmp)
        start += qt.cZControlLen

        #Control phase correction, time is fixed, amp is swept:
        qc.z +=env.rect(start, qc.cZControlPhaseCorrLen, controlAmp)
        start += qc.cZControlPhaseCorrLen + qc['piLen']/2
        #Final pi/2 for Ramsey, rotate about X
        qc.xy += eh.mix(qc, eh.piHalfPulseHD(qc, start, phase=0.0*np.pi))
        start += qc['piLen']/2

        #Measure
        qc.z += eh.measurePulse(qc, start)

        qc['readout'] = True

        return runQubits(server, qubits, stats=stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)


def BellStateCPi_debug1(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=False,
                phase=0.5*np.pi, name='BellState QST with controlled-Pi',
                save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    qc = qubits[measure[0]]
    qt = qubits[measure[1]]

    measurement = meas.Octomo(4, measure)
    #measurement = meas.Tomo(2)

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measurement, kw=kw)

    def func(server, curr):
        start = 0
        ph = phase

        #state prep
        #Control g -> g + e
        qc.xy = eh.mix(qc, eh.piHalfPulseHD(qc, start, phase=ph))
        #start += qc['piLen']/2
        #qt.xy = eh.mix(qt, eh.piPulseHD(qt, start))
        #start += qt['piLen']/2
        #Control iSWAP with Resonator
#        qc.z = env.rect(start, qc.cZControlLen, qc.cZControlAmp)
#        start += qc.cZControlLen
#
#        #state prep Target
#        qt.xy = eh.mix(qt, eh.piHalfPulseHD(qt, start, phase=ph))
#        start += qt['piLen']/2
#        #Target Phase swap Q21 with R21 for iswap^2 time
#        qt.z = env.rect(start, qt.cZTargetLen, qt.cZTargetAmp)
#        start += qt.cZTargetLen
#
#        #Target phase correction, time is fixed amp calibrated in Step 1,2
#        qt.z += env.rect(start, qt.cZTargetPhaseCorrLen, qt.cZTargetPhaseCorrAmp)
#
#        #Control iSWAP with Resonator
#        qc.z += env.rect(start, qc.cZControlLen, qc.cZControlAmp)
#        start += qt.cZControlLen
#
#        #Control phase correction, time is fixed, amp calibrated in Step 3:
#        qc.z +=env.rect(start, qc.cZControlPhaseCorrLen, qc.cZControlPhaseCorrAmp)
#        start += qc.cZControlPhaseCorrLen + qc['piLen']/2
#
#        #Final pi/2 for Ramsey, rotate about X
#        qt.xy += eh.mix(qt, eh.piHalfPulseHD(qt, start, phase=ph))
        start += qt['piLen']/2
#
#        #Measure

        return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
#    rho_ideal = np.array([[0.5,0.0,0.0,0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.5,0.0,0.0,0.5]])
#    rho_ideal = np.array([[0.0,0.0,0.0,0.0],[0.0,0.5,-0.5,0.0],[0.0,-0.5,0.5,0.0],[0.0,0.0,0.0,0.0]])
#    rho_ideal = np.array([[0.5,0.0,0.0,-0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[-0.5,0.0,0.0,0.5]])
    result = np.sum(result,axis=0)/len(repetition)
    Qk = np.reshape(result[1:],(36,4))
    #Qk = np.reshape(result[1:],(9,4))
    Qk = readoutFidCorr(Qk, [qc.measureF0,qc.measureF1,qt.measureF0,qt.measureF1])
#
    rho_cal = pyle.tomo.qst(Qk,'octomo2')
    #rho_cal = pyle.tomo.qst(Qk,'tomo2')
    plotRhoSingle(rho_cal)
    rho_caln = rho_cal.copy()
#    rho_caln[1,2] = abs(rho_caln[1,2])
#    rho_caln[2,1] = abs(rho_caln[2,1])
    print 'Fidelity is %g ' % np.trace(np.dot(rho_caln,rho_ideal))

    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal))


def BellStateCPi(sample, reps=10, measure=[0,1], stats=1500L, overshoot=False,
                phase=0.5*np.pi, name='BellState QST with controlled-Pi',
                save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    qc = qubits[measure[0]]
    qt = qubits[measure[1]]

    measurement = meas.Octomo(4, measure)

    repetition = range(reps)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measurement, kw=kw)

    def func(server, curr):
        start = 0
        ph = phase

        #state prep
        #Control g -> g + e
        qc.xy = eh.mix(qc, eh.piHalfPulseHD(qc, start, phase=ph))
        start += qc['piLen']/2
        #Control iSWAP with Resonator
        qc.z = env.rect(start, qc.cZControlLen, qc.cZControlAmp)
        start += qc.cZControlLen

        #state prep Target
        qt.xy = eh.mix(qt, eh.piHalfPulseHD(qt, start, phase=ph))
        start += qt['piLen']/2
        #Target Phase swap Q21 with R21 for iswap^2 time
        qt.z = env.rect(start, qt.cZTargetLen, qt.cZTargetAmp)
        start += qt.cZTargetLen

        #Target phase correction, time is fixed amp calibrated in Step 1,2
        qt.z += env.rect(start, qt.cZTargetPhaseCorrLen, qt.cZTargetPhaseCorrAmp)

        #Final pi/2 for Target, rotate about X
        qt.xy += eh.mix(qt, eh.piHalfPulseHD(qt, start+qt.cZTargetPhaseCorrLen+qt['piLen']/2, phase=ph))

        #Control iSWAP with Resonator
        qc.z += env.rect(start, qc.cZControlLen, qc.cZControlAmp)
        start += qt.cZControlLen

        #Control phase correction, time is fixed, amp calibrated in Step 3:
        qc.z +=env.rect(start, qc.cZControlPhaseCorrLen, qc.cZControlPhaseCorrAmp)
        start += qc.cZControlPhaseCorrLen


        #Measure padding for tomopulses
        start += 1.0*ns

        return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

#    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]]) #max, max
#    rho_ideal = np.array([[0.5,0.0,0.0,0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.5,0.0,0.0,0.5]]) #max,min
#    rho_ideal = np.array([[0.0,0.0,0.0,0.0],[0.0,0.5,-0.5,0.0],[0.0,-0.5,0.5,0.0],[0.0,0.0,0.0,0.0]]) #min,max
#    rho_ideal = np.array([[0.5,0.0,0.0,-0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[-0.5,0.0,0.0,0.5]]) #min,min
    result = np.sum(result,axis=0)/len(repetition)

    Qk = np.reshape(result[1:],(36,4))
    Qk = readoutFidCorr(Qk, [qc.measureF0,qc.measureF1,qt.measureF0,qt.measureF1])
#

    rho_cal = pyle.tomo.qst(Qk,'octomo2')
    plotRhoSingle(rho_cal)
    eFormation = et.eof(rho_cal)
    rho_caln = rho_cal.copy()
#    rho_caln[1,2] = abs(rho_caln[1,2])
#    rho_caln[2,1] = abs(rho_caln[2,1])
    print 'Fidelity is %g ' % np.trace(np.dot(rho_caln,rho_ideal))
    print 'Entangle of Formation is %g' % eFormation

    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal))


def readoutFidCorr(data,measFidMat):
    data = data.copy()
    sd = np.shape(data)
    if sd[1]==5:
        x = data[:,0]
        data = data[:,1:]
    # f0q1 = probability (%) of correctly reading a |0> on qubit 0
    f0q1 = measFidMat[0] # .956;
    # f1q1 = probability (%) of correctly reading a |1> on qubit 0
    f1q1 = measFidMat[1] # .891;
    # f0q2 = probability (%) of correctly reading a |0> on qubit 1
    f0q2 = measFidMat[2] # .91;
    # f1q2 = probability (%) of correctly reading a |1> on qubit 1
    f1q2 = measFidMat[3] # .894;
    # matrix of fidelities
    fidC = np.matrix([[   f0q1*f0q2        , f0q1*(1-f1q2)    , (1-f1q1)*f0q2    , (1-f1q1)*(1-f1q2) ],
                            [   f0q1*(1-f0q2)    , f0q1*f1q2        , (1-f1q1)*(1-f0q2), (1-f1q1)*f1q2     ],
                            [   (1-f0q1)*f0q2    , (1-f0q1)*(1-f1q2), f1q1*f0q2        , f1q1*(1-f1q2)     ],
                            [   (1-f0q1)*(1-f0q2), (1-f0q1)*f1q2    , f1q1*(1-f0q2)    , f1q1*f1q2         ]])
    fidCinv = fidC.I
    dataC = data*0
    for i in range(len(data[:,0])):
        dataC[i,:] = np.dot(fidCinv,data[i,:])

    if sd[1]==5:
        dataC0 = np.zeros(sd)
        dataC0[:,0] = x
        dataC0[:,1:] = dataC
        dataC = dataC0
    return  dataC

def plotRhoSingle(rho, scale=1.0, color=None, width=0.05, headwidth=0.1, headlength=0.1, chopN=None, amp=1.0 ):#figNo=100
    plt.figure()
#    pylab.clf()
    rho=rho.copy()
    s=np.shape(rho)
    if chopN!=None:
        rho = rho[:chopN,:chopN]
    s=np.shape(rho)
    rho = rho*amp
    ax = plt.gca()
    ax.set_aspect(1.0)
    pos = ax.get_position()
    r = np.real(rho)
    i = np.imag(rho)
    x = np.arange(s[0])[None,:] + 0*r
    y = np.arange(s[1])[:,None] + 0*i
    plt.quiver(x,y,r,i,units='x',scale=1.0/scale, width=width, headwidth=headwidth, headlength=headlength, color=color)
    plt.xticks(np.arange(s[1]))
    plt.yticks(np.arange(s[0]))
    plt.xlim(-0.9999,s[1]-0.0001)
    plt.ylim(-0.9999,s[0]-0.0001)
    return rho


def cZ_old(sample, repetition=10, measure=[0,1], stats=1500L, control=True,
         name='Controlled Z MQ', save=True, collect=False, noisy=True):
    """Performs a controlled-Z gate. Including mw and z-pulses on control and target qubits [qc, qt]"""

    sample, qubits = util.loadQubits(sample)
    qc = qubits[measure[0]] #control qubit
    qt = qubits[measure[1]] #target qubit

    repetition = range(repetition)
    axes = [(repetition, 'number of repetitions')]
    kw = {'stats': stats}
    if control:
        name = name + 'Control=1'
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, curr):
        start = 0
        #gate prep
        #Target g -> e
        qt.xy = eh.mix(qt, eh.piPulseHD(qt, start))
        start += qt['piLen']/2
        #Target iSWAP with Resonator
        qt.z = env.rect(start, qt.cZTargetLen, qt.cZTargetAmp)
        start += qt.cZTargetLen
        #control
        if control:
            qc.xy = eh.mix(qc,eh.piPulseHD(qc,start))

        start += qc['piLen']/2
        #Control Phase swap Q21 with R21 for iswap^2 time
        qc.z = env.rect(start, qc.cZControlLen, qc.cZControlAmp)
        start += qc.cZControlLen

        #control phase correction, time is fixed amp calibrated in Step 1,2
        qc.z += env.rect(start, qc.cZControlPhaseCorrLen, qc.cZControlPhaseCorrAmp)
        #Target iSWAP with Resonator, phase adjusted state back in Target
        qt.z += env.rect(start, qt.cZTargetLen, qt.cZTargetAmp)
        start += qt.cZTargetLen

        #target phase correction, time is fixed, amp is calibrated in Step 3:
        qt.z +=env.rect(start, qt.cZTargetPhaseCorrLen, qt.cZTargetPhaseCorrAmp)
        start += qt.cZTargetPhaseCorrLen

        #Measure
        qc.z += eh.measurePulse(qc, start)
        qt.z += eh.measurePulse(qt, start)


        qc['readout'] = True
        qt['readout'] = True
        return runQubits(server, qubits, stats=stats)
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

def cNOT_old(sample, repetition=10, measure=[0,1], stats=1500L, control=True, target=True,
         name='Controlled NOT', save=True, collect=False, noisy=True):
    """Performs a controlled-NOT gate using control and target qubits enumerated as [qc, qt]"""

    sample, qubits = util.loadQubits(sample)
    qc = qubits[measure[0]] #control qubit
    qt = qubits[measure[1]] #target qubit

    repetition = range(repetition)
    axes = [(repetition, 'number of repetitions')]
    kw = {'stats': stats}
    if control:
        name = name + 'Control=1'
    else:
        name = name + 'Control=0'
    if target:
        name = name + 'Target=1'
    else:
        name = name + 'Target=0'
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, curr):
        start = 0
        #gate prep
        if target:#Target g -> e
            qt.xy = eh.mix(qt, eh.piPulseHD(qt,start))
            start = qt['piLen']/2
            #Hadamard or Xpi/2
            qt.xy += eh.mix(qt, eh.piHalfPulseHD(qt, start))
        else:
            start = qt['piLen']/2
            #Hadamard or Xpi/2
            qt.xy = eh.mix(qt, eh.piHalfPulseHD(qt, start))

        start += qt['piLen']/2
        #Target iSWAP with Resonator
        qt.z = env.rect(start, qt.cZTargetLen, qt.cZTargetAmp)
        start += qt.cZTargetLen
        #control
        if control:
            qc.xy = eh.mix(qc,eh.piPulseHD(qc,start))

        start += qc['piLen']/2
        #Control Phase swap Q21 with R21 for iswap^2 time
        qc.z = env.rect(start, qc.cZControlLen, qc.cZControlAmp)
        start += qc.cZControlLen

        #control phase correction, time is fixed amp calibrated in Step 1,2
        qc.z += env.rect(start, qc.cZControlPhaseCorrLen, qc.cZControlPhaseCorrAmp)
        #Target iSWAP with Resonator, phase adjusted state back in Target
        qt.z += env.rect(start, qt.cZTargetLen, qt.cZTargetAmp)
        start += qt.cZTargetLen

        #target phase correction, time is fixed, amp is calibrated in Step 3:
        qt.z +=env.rect(start, qt.cZTargetPhaseCorrLen, qt.cZTargetPhaseCorrAmp)
        start += qt.cZTargetPhaseCorrLen

        #Final Hadamard or -Xpi/2
        qt.xy += eh.mix(qt, eh.piHalfPulseHD(qt, start))
        start += qt['piLen']/2

        #Measure
        qc.z += eh.measurePulse(qc, start)
        qt.z += eh.measurePulse(qt, start)


        qc['readout'] = True
        qt['readout'] = True
        return runQubits(server, qubits, stats=stats)
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)


def cZCalPhaseCntrl_OLD(sample, controlAmp=st.r[-0.01:0.01:0.0001], measure=[0,1], stats=1500L,
         name='Controlled Z ControlCal MQ', save=True, collect=False, noisy=True):
    """Performs the controlled-Z gate Z-pulse sequence
    with pi/2 pulse on control and no microwaves on target qubit [qc, qt]
    to calibrate the phase correction on the control qubit"""

    sample, qubits = util.loadQubits(sample)
    qc = qubits[measure[0]] #control qubit
    qt = qubits[measure[1]] #target qubit

    # repetition = range(repetition)
    axes = [(controlAmp, 'control amp phase correction')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, controlAmp):
        start = 0
        #state prep
        #Target qubit no microwaves
        qc.xy = eh.mix(qc, eh.piHalfPulseHD(qc, start))
        if qc['piLen'] > qt['piLen']:
            start += qc['piLen']/2
        else:
            start += qt['piLen']/2
        #Target SWAP with Resonator
        qt.z = env.rect(start, qt.cZTargetLen, qt.cZTargetAmp)
        #compensating Z pulse for control qubit phase correction
        qc.z = env.rect(start, qt.cZTargetLen, controlAmp)
        start += qt.cZTargetLen

        #Control Phase swap Q21 with R21 for iswap^2 time
        qc.z += env.rect(start, qc.cZControlLen, qc.cZControlAmp)
        start += qc.cZControlLen

        #Target SWAP with Resonator
        qt.z += env.rect(start, qt.cZTargetLen, qt.cZTargetAmp)
        #compensating Z pulse for control qubit phase correction
        qc.z += env.rect(start, qt.cZTargetLen, controlAmp)
        start += qt.cZTargetLen
        #Final pi/2 for Ramsey, rotate about y
        qc.xy += eh.mix(qc, eh.piHalfPulseHD(qc, start, phase=0.0*np.pi))
        if qc['piLen'] > qt['piLen']:
            start += qc['piLen']/2
        else:
            start += qt['piLen']/2

        #Measure
        qc.z += eh.measurePulse(qc, start)
        qt.z += eh.measurePulse(qt, start)


        qc['readout'] = True
        qt['readout'] = True
        return runQubits(server, qubits, stats=stats)
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

def cZCalPhaseTrgt_OLD(sample, targetAmp=st.r[-0.01:0.01:0.0001], measure=[0,1], stats=1500L,
         name='Controlled Z TargetCal MQ', save=True, collect=False, noisy=True):
    """Performs the controlled-Z gate Z-pulse sequence
    with pi/2 pulse on target and no microwaves on control qubit [qc, qt]
    to calibrate the phase correction on the target due to the control qubit z-pulses"""

    sample, qubits = util.loadQubits(sample)
    qc = qubits[measure[0]] #control qubit
    qt = qubits[measure[1]] #target qubit

    # repetition = range(repetition)
    axes = [(targetAmp, 'target amp phase correction')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, targetAmp):
        start = 0
        #state prep
        #Control qubit no microwaves
        qt.xy = eh.mix(qt, eh.piHalfPulseHD(qt, start))
        if qc['piLen'] > qt['piLen']:
            start += qc['piLen']/2
        else:
            start += qt['piLen']/2
        #Target SWAP with Resonator
        qt.z = env.rect(start, qt.cZTargetLen, qt.cZTargetAmp)
        #compensating Z pulse for control qubit phase correction calibrated from cZCalPhaseCntrl
        qc.z = env.rect(start, qt.cZTargetLen, qc.cZControlPhaseCorrAmp)
        start += qt.cZTargetLen

        #Control Phase swap Q21 with R21 for iswap^2 time
        qc.z += env.rect(start, qc.cZControlLen, qc.cZControlAmp)
        #compensating Z pulse for target qubit phase
        qt.z += env.rect(start, qc.cZControlLen, targetAmp)
        start += qc.cZControlLen

        #Target SWAP with Resonator
        qt.z += env.rect(start, qt.cZTargetLen, qt.cZTargetAmp)
        #compensating Z pulse for control qubit phase correction
        qc.z += env.rect(start, qt.cZTargetLen, qc.cZControlPhaseCorrAmp)
        start += qt.cZTargetLen

        #Final pi/2 for Ramsey, rotate about y
        qt.xy += eh.mix(qt, eh.piHalfPulseHD(qt, start, phase=0.5*np.pi))
        if qc['piLen'] > qt['piLen']:
            start += qc['piLen']/2
        else:
            start += qt['piLen']/2

        #Measure
        qc.z += eh.measurePulse(qc, start)
        qt.z += eh.measurePulse(qt, start)


        qc['readout'] = True
        qt['readout'] = True
        return runQubits(server, qubits, stats=stats)
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)



def fourQShor_old(sample, repetition=10, measure=[0,1,2,3], stats=1500L,
         name='Four Qubit Shor MQ', save=True, collect=False, noisy=True):
    """Factors 15."""

    sample, qubits = util.loadQubits(sample)
    qa = qubits[measure[0]] #control qubit
    qb = qubits[measure[2]] #control / control / target qubit
    qc = qubits[measure[1]] #target qubit
    qd = qubits[measure[3]] #target qubit

    repetition = range(repetition)
    axes = [(repetition, 'number of repetitions')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, curr):
        start = 0
        #state prep: Hadamards on 3 of 4 qubits {qa,qb,qc}
        qa.xy = eh.mix(qa, eh.rotPulseHD(qa, start, angle=np.pi/np.sqrt(2)))
        qa.z  = eh.rotPulseZ(qa, start, angle=np.pi/np.sqrt(2))

        qb.xy = eh.mix(qb, eh.rotPulseHD(qb, start, angle=np.pi/np.sqrt(2)))
        qb.z  = eh.rotPulseZ(qb, start, angle=np.pi/np.sqrt(2))

        qc.xy = eh.mix(qc, eh.rotPulseHD(qc, start, angle=np.pi/np.sqrt(2)))
        qc.z  = eh.rotPulseZ(qc, start, angle=np.pi/np.sqrt(2))

        #First CNOT: Qubits B & C participating
        #qubit B is Control, qubit C is Target
        if qb['piLen'] > qc['piLen']:
            start += qb['piLen']/2
        else:
            start += qc['piLen']/2
        #Target SWAP with Resonator
        qc.z  += env.rect(start, qc.cZTargetLen, qc.cZTargetAmp)
        start += qc.cZTargetLen

        #Control Phase swap Q21 with R21
        qb.z  += env.rect(start, qb.cZControlLen, qb.cZControlAmp)
        start += qb.cZControlLen

        #Target SWAP with Resonator
        qc.z  += env.rect(start, qc.cZTargetLen, qc.cZTargetAmp)
        start += qc.cZTargetLen

        #Finish C-NOT: Hadamard on Target Qubit
        start += qc['cPadTime'] #pad the time after the swap
        qc.xy += eh.mix(qc, eh.rotPulseHD(qc, start, angle=np.pi/np.sqrt(2)))
        qc.z  += eh.rotPulseZ(qc, start, angle=np.pi/np.sqrt(2))

        #Second CNOT: Qubits B & D participating
        #qubit B is Control, qubit D is Target

        #State prep: Hadamard on qubit qd.
        #Timing is coincident with ending of First CNOT
        qd.xy = eh.mix(qd, eh.rotPulseHD(qd, start, angle=np.pi/np.sqrt(2)))
        qd.z  = eh.rotPulseZ(qd, start, angle=np.pi/np.sqrt(2))
        start += qd['piLen']

        #Measure qc:
        qc.z += eh.measurePulse(qc, start)

        #Target SWAP with Resonator
        qd.z  += env.rect(start, qd.cZTargetLen, qd.cZTargetAmp)
        start += qd.cZTargetLen

        #Control Phase swap Q21 with R21
        qb.z  += env.rect(start, qb.cZControlLen, qb.cZControlAmp)
        start += qb.cZControlLen

        #Target SWAP with Resonator
        qd.z  += env.rect(start, qd.cZTargetLen, qd.cZTargetAmp)
        start += qd.cZTargetLen

        #Finish C-NOT: Hadamard on Target Qubit
        start += qd['cPadTime'] #pad the time after the swap
        qd.xy += eh.mix(qd, eh.rotPulseHD(qd, start, angle=np.pi/np.sqrt(2)))
        qd.z  += eh.rotPulseZ(qd, start, angle=np.pi/np.sqrt(2))

        #Controlled-S (a.k.a Controlled Z-pi/2): Qubits A & B participating
        #qubit A is Control, qubit C is Target

        #State prep: Hadamard on qubit qa and qb.
        #Timing is coincident with ending of Second CNOT
        qa.xy += eh.mix(qa, eh.rotPulseHD(qa, start, angle=np.pi/np.sqrt(2)))
        qa.z  += eh.rotPulseZ(qa, start, angle=np.pi/np.sqrt(2))
        qb.xy += eh.mix(qb, eh.rotPulseHD(qb, start, angle=np.pi/np.sqrt(2)))
        qb.z  += eh.rotPulseZ(qb, start, angle=np.pi/np.sqrt(2))
        start += qb['piLen']

        #Measure qd:
        qd.z += eh.measurePulse(qc, start)

        #Target SWAP with Resonator
        qb.z  += env.rect(start, qb.cZTargetLen, qb.cZTargetAmp)
        start += qb.cZTargetLen

        #Control Phase SQRT(swap) Q21 with R21
        #Half time of full Controlled Z
        qa.z  += env.rect(start, (qa.cZControlLen)/2, qa.cZControlAmp)
        start += qa.cZControlLen/2
        #Measure qa:
        qa.z += eh.measurePulse(qa, (start+qa.cZControlLen/2))

        #Target SWAP with Resonator
        qb.z  += env.rect(start, qb.cZTargetLen, qb.cZTargetAmp)
        start += qb.cZTargetLen

        #Finish C-S: Hadamard on Target Qubit
        start += qb['cPadTime'] #pad the time after the swap
        qb.xy += eh.mix(qb, eh.rotPulseHD(qb, start, angle=np.pi/np.sqrt(2)))
        qb.z  += eh.rotPulseZ(qb, start, angle=np.pi/np.sqrt(2))
        start +=qb['piLen']
        #Measure qb:
        qb.z += eh.measurePulse(qb, start)

        qa['readout'] = True
        qb['readout'] = True
        qc['readout'] = True
        qd['readout'] = True
        return runQubits(server, qubits, stats=stats)
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
