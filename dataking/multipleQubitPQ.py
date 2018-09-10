import numpy as np
import scipy.optimize as opt
from scipy.optimize import leastsq, curve_fit
from scipy.special import erf, erfc
import matplotlib.pyplot as plt

import random
import itertools

import labrad
from labrad.units import Unit,Value

from pyle import gateCompiler as gc
from pyle import gates
from pyle import tomo, fidelity
import pyle.envelopes as env
from pyle.pipeline import returnValue, FutureList
from pyle.dataking import envelopehelpers as eh
from pyle.dataking import measurement
from pyle.dataking import adjust
from pyle.dataking.fpgaseq import runQubits as runQubits
from pyle.util import sweeptools as st
from pyle.dataking import utilMultilevels as ml
from pyle.plotting import dstools
from pyle.fitting import fitting
from pyle.util import structures
from pyle.dataking import sweeps
from pyle.dataking import util
from pyle.plotting import tomography

from pyle.dataking import singleQubitPQ as sq
from pyle.dataking.singleQubitPQ import get_mpa_func, find_mpa

# some function in singleQubit
ramseySpec = sq.ramseySpec
swapTuner = sq.swapTuner

V, mV, us, ns, GHz, MHz, dBm, rad = [Unit(s) for s in ('V', 'mV', 'us', 'ns', 'GHz', 'MHz', 'dBm', 'rad')]



def spectroscopy2Dxtalk(sample, freqScan=None, measure=0, bias=0, stats=300L,
                        fluxBelow=2*mV, fluxAbove=2*mV, fluxStep=0.1*mV, sb_freq=0*GHz,
                        save=True, name='2D Flux spectroscopy xtalk', collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    qubit = qubits[measure]
    qbias = qubits[bias]

    if measure == bias:
        raise Exception('must bias a different qubit from the one measured')

    if freqScan is None:
        freq = st.nearest(qubit['f10'][GHz], 0.001)
        freqScan = np.arange(freq-0.1, freq+1.0, 0.005)
    else:
        freqScan = np.array([f[GHz] for f in freqScan])
    freqScan = freqScan[np.argsort(abs(freqScan-qubit['f10'][GHz]))]

    fluxBelow = fluxBelow[V]
    fluxAbove = fluxAbove[V]
    fluxStep = fluxStep[V]
    fluxScan = np.arange(-fluxBelow, fluxAbove, fluxStep)
    fluxScan = fluxScan[np.argsort(abs(fluxScan))]
    fluxPoints = len(fluxScan)

    sweepData = {
        'fluxFunc': np.array([st.nearest(qbias['biasOperate'][V], fluxStep) - qbias['biasStepEdge']]),
        'fluxIndex': 0,
        'freqIndex': 0,
        'flux': 0*fluxScan,
        'prob': 0*fluxScan,
        'maxima': 0*freqScan,
    }

    axes = [('Flux Bias', 'V'), ('Frequency', 'GHz')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def sweep():
        for f in freqScan:
            center = np.polyval(sweepData['fluxFunc'], f**4) + qbias['biasStepEdge']
            center = st.nearest(center, fluxStep)
            for flx in center + fluxScan:
                yield flx*V, f*GHz

    def func(server, args):
        flux, freq = args
        for q in qubits:
            q['fc'] = freq - sb_freq # set all frequencies since they share a common microwave source
        qbias['biasOperate'] = flux
        qubit.xy = eh.spectroscopyPulse(qubit, 0, sb_freq)
        qubit.z = eh.measurePulse(qubit, qubit['spectroscopyLen'])
        qubit['readout'] = True
        prob = yield runQubits(server, qubits, stats, probs=[1])

        flux_idx = sweepData['fluxIndex']
        sweepData['flux'][flux_idx] = flux[V]
        sweepData['prob'][flux_idx] = prob[0]
        if flux_idx + 1 == fluxPoints:
            # one row is done.  find the maximum and update the spectroscopy fit
            freq_idx = sweepData['freqIndex']
            sweepData['maxima'][freq_idx] = sweepData['flux'][np.argmax(sweepData['prob'])]
            sweepData['fluxFunc'] = np.polyfit(freqScan[:freq_idx+1]**4,
                                               sweepData['maxima'][:freq_idx+1] - qubit['biasStepEdge'],
                                               freq_idx > 5)
            sweepData['fluxIndex'] = 0
            sweepData['freqIndex'] += 1
        else:
            # just go to the next point
            sweepData['fluxIndex'] = flux_idx + 1
        returnValue([flux, freq, prob])
    return sweeps.run(func, sweep(), dataset=dataset, save=save, collect=collect, noisy=noisy)


def spectroscopy2DZxtalk(sample, freqScan=None, measAmplFunc=None, measure=0, z_pulse=1,
                         fluxBelow=0.5, fluxAbove=0.5, fluxStep=0.025, sb_freq=0*GHz, stats=300L, plot=False,
                         name='2D Z-pulse xtalk spectroscopy', save=True, collect=False, update=True, noisy=True):
    sample, qubits = util.loadQubits(sample)
    qubit = qubits[measure]

    if measAmplFunc is None:
        measAmplFunc = get_mpa_func(qubit)

    if freqScan is None:
        freq = st.nearest(qubit['f10'][GHz], 0.001)
        freqScan = np.arange(freq-0.01, freq+0.01, 0.0002)
    elif isinstance(freqScan, tuple) and len(freqScan) == 2:
        freq = st.nearest(qubit['f10'][GHz], 0.001)
        rng = freqScan
        dfs = np.logspace(-3, -0.3, 25)
        freqScan = freq + np.hstack(([0], dfs, -dfs))
        freqScan = np.array([st.nearest(f, 0.001) for f in freqScan])
        freqScan = np.unique(freqScan)
        freqScan = np.compress((rng[0][GHz] < freqScan) * (freqScan < rng[1][GHz]), freqScan)
    else:
        freqScan = np.array([f[GHz] for f in freqScan])
    freqScan = freqScan[np.argsort(abs(freqScan-qubit['f10'][GHz]))]

    fluxScan = np.arange(-fluxBelow, fluxAbove, fluxStep)
    fluxScan = fluxScan[np.argsort(abs(fluxScan))]

    sweepData = {
        'fluxFunc': np.array([0]),
        'fluxIndex': 0,
        'freqIndex': 0,
        'flux': 0*fluxScan,
        'prob': 0*fluxScan,
        'maxima': 0*freqScan,
        'freqScan': freqScan,
        'fluxScan': fluxScan,
        'rowComplete': False
    }

    axes = [('Z-pulse amplitude', ''), ('Frequency', 'GHz')]
    kw = {'stats': stats, 'pulseQubit': z_pulse}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)



    def sweep():
        for indexF,f in enumerate(sweepData['freqScan']):
            center = np.polyval(sweepData['fluxFunc'], f**4)
            center = st.nearest(center, fluxStep)
            for indexZ,zpa in enumerate(center + sweepData['fluxScan']):
                if abs(zpa) < 2.0: # skip if zpa is too large
                    yield zpa, f*GHz
                if abs(zpa) >= 2.0:
                    zpaCut = 2.0*zpa/abs(zpa)
                    sweepData['fluxScan']=np.array([])
                    sweepData['freqScan']=freqScan[:(indexF+1)]
                    print 'Reaching maximum zpa at frequency %f GHz.' %f
                    yield zpaCut, f*GHz

    def func(server, args):
        zpa, freq = args
        for q in qubits:
            q['fc'] = freq - sb_freq # set all frequencies since they share a common microwave source
        dt = qubit['spectroscopyLen']
        qubits[z_pulse].z = env.rect(0, dt, zpa)
        qubits[measure].xy = eh.spectroscopyPulse(qubits[measure], 0, sb_freq)
        qubits[measure].z = eh.measurePulse(qubits[measure], dt)
        qubits[measure]['readout'] = True
        prob = yield runQubits(server, qubits, stats, probs=[1])

        flux_idx = sweepData['fluxIndex']
        sweepData['flux'][flux_idx] = zpa
        sweepData['prob'][flux_idx] = prob[0]
        if (flux_idx + 1 == len(sweepData['fluxScan'])) or (len(sweepData['fluxScan']) == 0 and not sweepData['rowComplete']):
            # one row is done.  find the maximum and update the spectroscopy fit
            freq_idx = sweepData['freqIndex']
            sweepData['maxima'][freq_idx] = sweepData['flux'][np.argmax(sweepData['prob'])]
            sweepData['fluxFunc'] = np.polyfit(sweepData['freqScan'][:freq_idx+1]**4,
                                               sweepData['maxima'][:freq_idx+1],
                                               freq_idx > 5)
            sweepData['fluxIndex'] = 0
            sweepData['freqIndex'] += 1
            sweepData['rowComplete'] = True
        else:
            # just go to the next point
            sweepData['fluxIndex'] = flux_idx + 1
            sweepData['rowComplete'] = not bool(len(sweepData['fluxScan']))
        returnValue([zpa, freq, prob[0]])
    sweeps.run(func, sweep(), dataset=dataset, save=save, collect=collect, noisy=noisy)

    # create a flux function and return it
    p = sweepData['fluxFunc']
    freq_idx = sweepData['freqIndex']
    #return (freqScan[:freq_idx],sweepData['maxima'][:freq_idx],p)
    if plot:
        indices = np.argsort(sweepData['maxima'][:freq_idx])
        data = np.vstack((sweepData['maxima'][:freq_idx][indices],freqScan[0:freq_idx][indices])).T
        fig=dstools.plotDataset1D(data,[('Z-Pulse Amplitude','')],[('Frequency','','GHz')],style='.',legendLocation=None,show=False,
                                  markersize=15,title='')
        fig.get_axes()[0].plot(np.polyval(p,data[:,1]**4),data[:,1],'r',linewidth=3)
        fig.show()
    return sweepData['fluxFunc']

def spectroscopyZ_xtalk_matrix(s):
    """Measure the z-pulse crosstalk between each pair of qubits.

    We then create the crosstalk matrix and store it in the registry.
    In addition, we invert the crosstalk matrix, since this is needed
    to correct the z-pulse signals if desired.

    Assumes you have already run find_zpa_func, so that the
    cal_zpa_func has already been set, as xtalk is relative to this.

    zpulseCrosstalk in crosstalk does the similar thing in a different way.
    """
    s, qubits, Qubits = util.loadQubits(s, write_access=True)
    A = np.eye(len(qubits))
    for i, qi in enumerate(qubits):
        for j, _qj in enumerate(qubits):
            if i == j:
                continue
            print 'measuring crosstalk on %s from z-pulse on %s' % (i, j)
            xtfunc = spectroscopy2DZxtalk(s, measure=i, z_pulse=j, noisy=False)
            aii = float(qi['calZpaFunc'][0])
            aij = float(xtfunc[0])
            print 'crosstalk =', aii/aij
            A[i,j] = aii/aij
    Ainv = np.linalg.inv(A)
    print
    print 'xtalk matrix:\n', A
    print
    print 'inverse xtalk matrix:\n', Ainv
    for i, Qi in enumerate(Qubits):
        Qi['calZpaXtalk'] = A[i]
        Qi['calZpaXtalkInv'] = Ainv[i]


def meas_xtalk(sample, trabi=st.r[0:150:1,ns], amp=None, df=None, stats=600L,
               drive=0, simult=True, name='meas_xtalk', save=True, collect=True, noisy=True):
    """Drive rabis on one qubit, then measure others to see measurement crosstalk.

    This works best when the qubits are off resonance from each other,
    so that direct microwave crosstalk is not an issue (although microwave
    crosstalk should appear at a different frequency, while measurement
    crosstalk will appear at the main Rabi frequency, so a Fourier transform
    should be able to separate these two effects, anyway.
    """
    sample, qubits = util.loadQubits(sample)
    driver = qubits[drive]
    N = len(qubits)

    if amp is not None:
        driver['piAmp'] = amp
    if df is not None:
        driver['piDf'] = 0

    axes = [(trabi, 'rabi pulse length')]
    deps = [('Probability', '|%s>' % ''.join('1' if i==n else 'x' for i in range(N)), '')
            for n in range(N)]
    kw = {
        'stats': stats,
        'simult': simult,
        'drive': drive,
    }
    dataset = sweeps.prepDataset(sample, name, axes, deps, kw=kw)

    if simult:
        def func(server, dt):
            driver.xy = eh.mix(driver, env.flattop(0, dt, driver['piFWHM'], amp=driver['piAmp']))
            for q in qubits:
                q.z = eh.measurePulse(q, dt + driver['piLen'])
                q['readout'] = True
            return runQubits(server, qubits, stats, dataFormat='probs_separate')
    else:
        def func(server, dt):
            driver.xy = eh.mix(driver, env.flattop(0, dt, driver['piFWHM'], amp=driver['piAmp']))
            reqs = []
            for i in range(N):
                for j, q in enumerate(qubits):
                    q.z = eh.measurePulse(q, dt + driver['piLen']) if i == j else env.NOTHING
                    q['readout'] = (i == j)
                reqs.append(runQubits(server, qubits, stats, probs=[1]))
            probs = yield FutureList(reqs)
            returnValue([p[0] for p in probs])
    return sweeps.grid(func, axes, dataset=dataset, save=save, collect=collect, noisy=noisy)


def meas_xtalk_timer(sample, adjust, ref, t=st.r[-40:40:0.5,ns], amp=None,
                     df=None, stats=600L, drive=0, simult=True, correctZ=False,
                     name='meas_xtalk_timer', save=True, collect=False, noisy=True, update=True):
    """Adjust timing between measure pulses and look at xtalk signal to adjust timing."""
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)

    qa, Qa = qubits[adjust], Qubits[adjust]
    qr = qubits[ref]

    print 'Adjusting measure-pulses for 50% tunneling...'
    find_mpa(sample, measure=adjust, target=0.5, noisy=False)
    find_mpa(sample, measure=ref, target=0.5, noisy=False)

    axes = [(t, 'adjusted measure pulse time')]
    measure = sorted([adjust, ref])
    kw = {
        'stats': stats,
        'adjust': adjust,
        'ref': ref,
    }
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, t):
        for q in qubits:
            q['z'] = env.NOTHING
        qa.z += eh.measurePulse(qa, t)
        qr.z += eh.measurePulse(qr, 0)
        qa['readout'] = True
        qr['readout'] = True
        eh.correctCrosstalkZ(qubits)
        return runQubits(server, qubits, stats)
    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    if update:
        t = adjust.adjust_time(data)
        if t is not None:
            print 'timing lag corrected by %g ns' % t
            Qa['timingLagWrtMaster'] += t*ns




def testdelayBusSwap(sample, measure, drive, paramName, startTime=st.r[-100:100:2,ns], pulseDelay=5*ns, stats=600L,
                     name='Qubit-Qubit delay using Resonator "test delay" ',
                     save=True, collect=True, noisy=True, plot=False, update=True):
    """
    A single pi-pulse on the drive qubit, iSWAP with R, delay, iSwap with measure qubit.

    Measure sweeps iSWAP through sequence.
    """
    sample, qubits = util.loadQubits(sample)
    qM = qubits[measure]
    qD = qubits[drive]

    axes = [(startTime, 'Control swap start time')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, curr):
        tControlPulse = qD['piLen']/2+qD['swapTime'+paramName]+pulseDelay+curr
        if tControlPulse < 0:
            t = -tControlPulse
        else:
            t = 0
        qD.xy = eh.mix(qD, eh.piPulseHD(qD, t))
        t += qD['piLen']/2
        qD.z = env.rect(t, qD['swapTime'+paramName], qD['swapAmp'+paramName])
        t += qD['swapTime'+paramName] + pulseDelay + curr
        qM.z = env.rect(t, qM['swapTime'+paramName], qM['swapAmp'+paramName])
        t += qM['swapTime'+paramName]
        qM.z += eh.measurePulse(qM, t)
        qM['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)
    if collect:
        return result

def testdelayBusSwapTuner(Sample, qubitUpdate, qubitMaster, paramName,
                          delay=st.r[-100:50:2,ns], stats=1500L, pulseDelay=5*ns,
                          name=None, save=False, collect=False, noisy=True, update=True, plot=True):
    sample, qubits, Qubits = util.loadQubits(Sample,write_access=True)
    qU = qubits[qubitUpdate]
    qM = qubits[qubitMaster]
    QU = Qubits[qubitUpdate]
    if name is None:
        name = 'Timing delay between %s and %s' %(Sample['config'][qubitUpdate],Sample['config'][qubitMaster])

    result0 = testdelayBusSwap(sample, qubitUpdate, qubitMaster, paramName, name=name, startTime=delay, save=save, collect=True, noisy=noisy)
    result1 = testdelayBusSwap(sample, qubitMaster, qubitUpdate, paramName, name=name, startTime=delay, save=save, collect=True, noisy=noisy)

    topLenM = qM['swapTime'+paramName]['ns']
    transLenM = pulseDelay
    def fitfunc0(x, p):
        return (p[1] +
                p[2] * 0.5*erf((x - (p[0] + topLenM/2.0)) / transLenM))
    x0, y0 = result0.T
    fit0, _ok0 = leastsq(lambda p: fitfunc0(x0, p) - y0, [0.0, 0.05, 0.9])

    topLenU = qU['swapTime'+paramName]['ns']
    transLenU = pulseDelay
    def fitfunc1(x, p):
        return (p[1] +
                p[2] * 0.5*erf((x - (p[0] + topLenU/2.0)) / transLenU))
    x1, y1 = result1.T
    fit1, _ok1 = leastsq(lambda p: fitfunc1(x1, p) - y1, [0.0, 0.05, 0.9])
    if plot:
        plt.figure()
        plt.plot(x0, y0, 'b.')
        plt.plot(x0, fitfunc0(x0, fit0), 'b-')
        plt.plot(x1, y1, 'r.')
        plt.plot(x1, fitfunc1(x1, fit1), 'r-')
    if update:
        print 'correct lag between two qubits by adding %g ns to %s' %(((fit0[0]-fit1[0])/2.0),Sample['config'][qubitUpdate])
        QU['timingLagWrtMaster']  += ((fit0[0]-fit1[0])/2.0)*ns
    if collect:
        return result0,result1

def testdelay_x(sample, t0=st.r[-40:40:0.5,ns], zpa=-1.5, zpl=20*ns, tm=65*ns,
                measure=0, z_pulse=1, stats=1200, update=True,
                save=True, name='Qubit-Qubit Test Delay ', plot=False, noisy=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    assert measure != z_pulse
    q = qubits[measure]
    other, Other = qubits[z_pulse], Qubits[z_pulse]

    axes = [(t0, 'Detuning pulse center')]
    kw = {
        'stats': stats,
        'z_pulse': z_pulse,
        'detuning pulse length': zpl,
        'detuning pulse height': zpa,
        'start of measurement pulse': tm
    }
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, t0):
        q.xy = eh.mix(q, eh.piPulse(q, 0))
        q.z = eh.measurePulse(q, tm)
        q['readout'] = True
        other.z = env.rect(t0-zpl/2.0, zpl, zpa)
        return runQubits(server, qubits, stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    zpl = zpl[ns]
    translength = 0.4*q['piFWHM'][ns]
    def fitfunc(x, p):
        return (p[1] +
                p[2] * 0.5*erfc((x - (p[0] - zpl/2.0)) / translength) +
                p[3] * 0.5*erf((x - (p[0] + zpl/2.0)) / translength))
    x, y = result.T
    fit, _ok = leastsq(lambda p: fitfunc(x, p) - y, [0.0, 0.05, 0.85, 0.85])

    if plot:
        plt.figure()
        plt.plot(x, y, '.')
        plt.plot(x, fitfunc(x, fit))
    print 'uwave lag:', fit[0]
    if update:
        print 'uwave lag corrected by %g ns' % fit[0]
        Other['timingLagWrtMaster'] += fit[0]*ns


def Swap(sample, paramName='', swapLen=st.r[0:400:4,ns], swapAmp=np.arange(-0.05,0.05,0.002),
         overshoot=0.0, measure=0, stats=600L, name='Swap MQ',
         save=True, collect=False, noisy=True, state=1):
    """
    measure swap spectroscopy, same with singleQubit.swapSpectroscopy
    swapLen, swapAmp can be iterable or single value.
    if state > 1, then the qubit will be moved to |state> then swaps.
    """
    return sq.swapSpectroscopy(sample, swapLen=swapLen, swapAmp=swapAmp, overshoot=overshoot,
                               measure=measure, stats=stats, name=name, save=save, collect=collect,
                               noisy=noisy, state=state, piPulse=True, paramName=paramName)


def FockScanReset(sample, n=1, scanLen=0.0*ns, scanOS=0.0, tuneOS=False, probeFlag=False,
                  paraName='0', stats=1500L, measure=0, delay=0*ns, resetPoint=0,
                  name='Fock state swap length scan MQ', save=False, collect=True, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    axes = [(scanLen, 'Swap length adjust'),(scanOS, 'Amplitude overshoot')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, 'res '+paraName+' '+name+' '+str(n)+' Fock state', axes, measure=measure, kw=kw)

    sl = sample['q'+str(measure)]['resetLens'+paraName][resetPoint]
    #sl = q['resetLens'+paraName][resetPoint]
    #print 'Optimizing n=%g for the swap length = %g ns...' %(n,sl[n-1])
    #sa = sample['q'+str(measure)]['resetAmps'+paraName][resetPoint]
    sa = q['resetAmps'+paraName][resetPoint]

    if not tuneOS:
        so = np.array([0.0]*n)
    else:
        so = q['noonSwapAmp'+paraName+'OSs']

    def func(server, currLen, currOS):
        q.xy = env.NOTHING
        q.z = env.NOTHING
        start = -q.piLen/2
        for i in range(n-1):
            q.xy += eh.mix(q, eh.piPulseHD(q, start+q.piLen/2))
            start += q.piLen+delay
            q.z += env.rect(start, sl, sa, overshoot=so[i])
            start += sl+delay
        q.xy += eh.mix(q, eh.piPulseHD(q, start+q.piLen/2))
        start += q.piLen+delay
        if not probeFlag:
            q.z += env.rect(start, sl+currLen, sa, overshoot=so[n-1]+currOS)
            start += sl+currLen+delay
            q.z += eh.measurePulse(q, start)
        else:
            q.z += env.rect(start, sl, sa, overshoot=so[n-1]+currOS)
            start += sl+delay
            q.z += env.rect(start, currLen, sa)
            start += currLen+delay
            q.z += eh.measurePulse(q, start)

        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy, collect=collect)

def FockScanResetCounter(sample, n=1, scanLen=0.0*ns, scanOS=0.0, tuneOS=False, probeFlag=False,
                         paraName='0', stats=1500L, measure=0, delay=0*ns, resetPoint=0,
                         name='Fock state counter-swap length scan MQ', save=False, collect=True, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    axes = [(scanLen, 'Counter-swap length adjust'),(scanOS, 'Amplitude overshoot')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, 'res '+paraName+' '+name+' '+str(n)+' Fock state', axes, measure=measure, kw=kw)

    sl = sample['q'+str(measure)]['resetLens'+paraName][resetPoint]
    #sl = q['resetLens'+paraName][resetPoint]
    #print 'Optimizing n=%g for the swap length = %g ns...' %(n,sl[n-1])
    #sa = sample['q'+str(measure)]['resetAmps'+paraName][resetPoint]
    sa = q['resetAmps'+paraName][resetPoint]


    if not tuneOS:
        so = np.array([0.0]*n)
    else:
        so = q['noonSwapAmp'+paraName+'OSs']

    def func(server, currLen, currOS):
        q.xy = env.NOTHING
        q.z = env.NOTHING
        start = -q.piLen/2
        for i in range(n-1):
            q.xy += eh.mix(q, eh.piPulseHD(q, start+q.piLen/2))
            start += q.piLen+delay
            q.z += env.rect(start, sl, sa, overshoot=so[i])
            start += sl+delay
        q.xy += eh.mix(q, eh.piPulseHD(q, start+q.piLen/2))
        start += q.piLen+delay
        if not probeFlag:
            q.z += env.rect(start, currLen, sa, overshoot=so[n-1]+currOS)
            start += currLen+delay
            q.z += eh.measurePulse(q, start)
        else:
            # q.z += env.rect(start, sl, sa, overshoot=so[n-1]+currOS)
            # start += sl+delay
            q.z += env.rect(start, currLen, sa)
            start += currLen+delay
            q.z += eh.measurePulse(q, start)

        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy, collect=collect)

def FockResetTunerEZ(sample, n=1, iteration=3, tuneOS=False, paraName='0', resetPoint=0, stats=1500L, measure=0, delay=0*ns,
                     save=False, collect=True, noisy=True, update=False):
    sample, qubits, Qubits = util.loadQubits(sample,write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]

    for iter in range(iteration):
        rf = 2**iter
        print 'iteration %g...' % iter
        sl = sample['q'+str(measure)]['resetLens'+paraName][resetPoint]
        results = FockScanReset(sample, n=n, scanLen=st.PQlinspace(-max([0.3*sl['ns']/rf,1]),max([0.3*sl['ns']/rf,1]),21,'ns'), resetPoint=resetPoint,
                                paraName=paraName, stats=stats, measure=measure, probeFlag=False, delay=delay,
                                save=False, collect=collect, noisy=noisy)
        new, percent = sq.datasetMinimum(results, 0, -max([0.3*sl['ns']/rf,1]), max([0.3*sl['ns']/rf,1]))
        sample['q'+str(measure)]['resetLens'+paraName][resetPoint] += new

        if save:
            FockScanReset(sample, n=n, scanLen=st.arangePQ(0,100,2,'ns'),
                          paraName=paraName, stats=stats, measure=measure, probeFlag=True, delay=delay, resetPoint=resetPoint,
                          save=save, collect=collect, noisy=noisy)

    # if update:
    #    Q['resetLens'+paraName][resetPoint] = sample['q'+str(measure)]['resetLens'+paraName][resetPoint]

    return sample['q'+str(measure)]['resetLens'+paraName][resetPoint]

def FockResetTunerCounterEZ(sample, n=1, iteration=3, tuneOS=False, paraName='0', resetPoint=0, stats=1500L, measure=0, delay=0*ns,
                            save=False, collect=True, noisy=True, update=False):
    sample, qubits, Qubits = util.loadQubits(sample,write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]

    for iter in range(iteration):
        rf = 2**iter
        print 'iteration %g...' % iter
        sl = sample['q'+str(measure)]['resetLens'+paraName][resetPoint]
        results = FockScanResetCounter(sample, n=n, scanLen=st.PQlinspace(-max([0.3*sl['ns']/rf,1]),max([0.3*sl['ns']/rf,1]),21,'ns'), resetPoint=resetPoint,
                                       paraName=paraName, stats=stats, measure=measure, probeFlag=False, delay=delay,
                                       save=False, collect=collect, noisy=noisy)
        new, percent = sq.datasetMinimum(results, 0, -max([0.3*sl['ns']/rf,1]), max([0.3*sl['ns']/rf,1]))
        sample['q'+str(measure)]['resetLens'+paraName][resetPoint] += new

        if save:
            FockScanResetCounter(sample, n=n, scanLen=st.arangePQ(0,100,2,'ns'),
                                 paraName=paraName, stats=stats, measure=measure, probeFlag=True, delay=delay, resetPoint=resetPoint,
                                 save=save, collect=collect, noisy=noisy)

    # if update:
    #    Q['resetLens'+paraName][resetPoint] = sample['q'+str(measure)]['resetLens'+paraName][resetPoint]

    return sample['q'+str(measure)]['resetLens'+paraName][resetPoint]

def iSwapBus(sample, swapLen=st.r[-20:1000:1,ns], measure=[0,1], stats=1500L,
             name='iSwap through Bus MQ', save=True, collect=True, noisy=True, delay=0*ns,
             paraName='C', plot=True):
    """
    iSwap two qubit through a bus resonator.
    pump the first qubit to |1>, swap into resonatorBus (with varies swap length),
    and then second qubit swap with the resonator bus
    """
    sample, qubits, _ = gc.loadQubits(sample, measure)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]
    nameEx = ' {0}->{1}'.format(str(q0), str(q1))

    axes = [(swapLen, 'swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx, axes, measure=measure, kw=kw)

    def func(server, curr):
        alg = gc.Algorithm(qubits)
        alg[gates.PiPulse([alg.q0])]
        ml.setMultiLevels(alg.q0, 'swapTime'+paraName, curr, 1) # change swapTime of q0
        alg[gates.iSwap([alg.q0, alg.q1] ,delay=delay, paramName=paraName)]
        alg[gates.Wait([alg.q0], delay)]
        alg[gates.Wait([alg.q1], delay)]
        alg[gates.MeasurePQ([alg.q0, alg.q1])]
        alg.q0['readout'] = True
        alg.q1['readout'] = True
        alg.compile(correctXtalkZ=False)
        return runQubits(server, alg.agents, stats=stats)
    result = sweeps.grid(func, axes, dataset=dataset, noisy=noisy, save=save)
    resultC = readoutFidCorr(result, [q0['measureF0'],q0['measureF1'],
                                      q1['measureF0'],q1['measureF1']])
    if plot:
        color = ('b','r','g','c')
        plt.figure()
        plt.clf()
        for i in np.arange(0,4,1):
            plt.plot(result[:,0],result[:,i+1],color[i]+'.')
            plt.plot(resultC[:,0],resultC[:,i+1],color[i]+'-')
            plt.hold('on')
    if collect:
       return result, resultC

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


def bellStateScan(sample, measure=[0,1], sweepQubit=1, stats=1500L, paraName='C',
                  swapLen=None, save=True, collect=True,
                  name='Bell State Scan through Bus MQ', noisy=True):
    """
    excite first qubit to |1>, sqrt(iSWAP) with Bus resonator,
    the second qubit swap with the Bus resonator
    """
    if sweepQubit not in measure:
        raise Exception("sweepQubit {} must in measure {}".format(sweepQubit, measure))

    sample, qubits, _ = gc.loadQubits(sample, measure)

    paraNameLen = 'swapTime' + paraName

    if swapLen is None:
        swapLen0 = (qubits[sweepQubit])[paraNameLen]
        swapLen = np.linspace(-10, 10, 41) + swapLen0
        swapLen = swapLen[swapLen>0]

    axes = [(swapLen, 'scan parameter')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, curr):
        alg = gc.Algorithm(qubits)
        q_scan = alg.agents[sweepQubit]
        q_scan[paraNameLen] = curr
        alg.q0['readout'] = True
        alg.q1['readout'] = True
        alg[gates.PiPulse([alg.q0])]
        alg[gates.SwapHalf([alg.q0], paramName=paraName)] # sqrtSwap with Bus
        alg[gates.Sync([alg.q0, alg.q1])]
        alg[gates.Swap([alg.q1], paramName=paraName)] # Bus swap to q1
        alg[gates.MeasurePQ([alg.q0, alg.q1], sync=True)]
        alg.compile(correctXtalkZ=False)
        return runQubits(server, alg.agents, stats=stats)
    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)
    if collect:
        return data


def bellStateScan_SQRT(sample, measure=[0,1], sweepQubit=1, stats=1500L,
                  paraName='C', swapLen=None, save=True, collect=True,
                  name='Bell State Scan (SQRT) through Bus MQ', noisy=True):
    """
    excite first qubit to |1>, sqrt(iSWAP) with Bus resonator,
    the second qubit swap with the Bus resonator
    use registry key sqrtSwapTime+paraName, sqrtSwapAmp+paraName
    """
    if sweepQubit not in measure:
        raise Exception("sweepQubit {} must in measure {}".format(sweepQubit, measure))

    sample, qubits, _ = gc.loadQubits(sample, measure)

    paraNameLen = 'sqrtSwapTime' + paraName

    if swapLen is None:
        swapLen0 = (qubits[sweepQubit])[paraNameLen]
        swapLen = np.linspace(-10, 10, 41) + swapLen0
        swapLen = swapLen[swapLen>0]

    axes = [(swapLen, 'scan parameter')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, curr):
        alg = gc.Algorithm(qubits)
        q_scan = alg.agents[sweepQubit]
        q_scan[paraNameLen] = curr
        alg.q0['readout'] = True
        alg.q1['readout'] = True
        alg[gates.PiPulse([alg.q0])]
        alg[gates.SqrtSwap([alg.q0], paramName=paraName)] # sqrtSwap with Bus
        alg[gates.Sync([alg.q0, alg.q1])]
        alg[gates.Swap([alg.q1], paramName=paraName)] # Bus swap to q1
        alg[gates.MeasurePQ([alg.q0, alg.q1], sync=True)]
        alg.compile(correctXtalkZ=False)
        return runQubits(server, alg.agents, stats=stats)
    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)
    if collect:
        return data


def bellState(sample, repetition=10, measure=[0,1], stats=1500L, paraName='C',
         name='Bell state through Bus MQ', save=True, collect=True, noisy=True):
    """
    bell state, -|10>+|01>
    excite the first qubit, SQRT(iSWAP) with Bus, and the second iSwap with Bus
    """
    sample, qubits, _ = gc.loadQubits(sample, measure)

    repetition = range(repetition)
    axes = [(repetition, 'repetitions')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, rep):
        alg = gc.Algorithm(qubits)
        alg[gates.PiPulse([alg.q0])]
        alg[gates.SwapHalf([alg.q0], paramName=paraName)]
        alg[gates.Sync([alg.q0, alg.q1])]
        alg[gates.Swap([alg.q1], paramName=paraName)]
        alg[gates.MeasurePQ([alg.q0, alg.q1], sync=True)]
        alg.q0['readout'] = True
        alg.q1['readout'] = True
        alg.compile(correctXtalkZ=False)
        return runQubits(server, alg.agents, stats=stats)
    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)
    if collect:
        return data


def bellState_SQRT(sample, repetition=10, measure=[0,1], stats=1500L,
                   paraName='C', save=True, noisy=True, collect=True,
                   name='Bell state (SQRT) through Bus MQ'):
    """
    bell state, -|10>+|01>
    excite the first qubit, SQRT(iSWAP) with Bus, and the second iSwap with Bus
    use registry key sqrtSwapTime+paraName, sqrtSwapAmp+paraName
    """
    sample, qubits, _ = gc.loadQubits(sample, measure)

    repetition = range(repetition)
    axes = [(repetition, 'repetitions')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, rep):
        alg = gc.Algorithm(qubits)
        alg[gates.PiPulse([alg.q0])]
        alg[gates.SqrtSwap([alg.q0], paramName=paraName)]
        alg[gates.Sync([alg.q0, alg.q1])]
        alg[gates.Swap([alg.q1], paramName=paraName)]
        alg[gates.MeasurePQ([alg.q0, alg.q1], sync=True)]
        alg.q0['readout'] = True
        alg.q1['readout'] = True
        alg.compile(correctXtalkZ=False)
        return runQubits(server, alg.agents, stats=stats)
    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)
    if collect:
        return data

# the tomo below, we do not use gateCompiler

def bellStateTomo(sample, repetition=10, measure=[0,1], stats=1500L, paraName='C',
         name='Bell state QST through Bus MQ', save=True, collect=True, noisy=True):
    """
    bell state, -|10>+|01>
    excite the first qubit, SQRT(iSWAP) with Bus, and the second iSwap with Bus
    """
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    repetition = range(repetition)
    axes = [(repetition, 'repetitions')]
    kw = {'stats': stats}
    measFunc = measurement.Octomo(len(qubits), measure=measure)
    dataset = sweeps.prepDataset(sample, name, axes, measure=measFunc, kw=kw)

    swapTimeName = 'swapTime'+paraName
    swapAmpName = 'swapAmp'+paraName
    def func(server, rep):
        start = 0
        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2
        q0.z = env.rect(start, q0[swapTimeName]/2, q0[swapAmpName])
        start += q0[swapTimeName]/2
        q1.z = env.rect(start, q1[swapTimeName], q1[swapAmpName])
        start += q1[swapTimeName]
        return measFunc(server, qubits, start, stats=stats)

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)
    if collect:
        return data

def bellStateTomo_SQRT(sample, repetition=10, measure=[0,1], stats=1500L,
                   paraName='C', save=True, noisy=True, collect=True,
                   name='Bell state QST (SQRT) through Bus MQ'):
    """
    bell state,
    the first qubit SQRT(iSWAP) with Bus, and the second iSwap with Bus
    use registry key sqrtSwapTime+paraName
    """
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    repetition = range(repetition)
    axes = [(repetition, 'repetitions')]
    kw = {'stats': stats}
    measFunc = measurement.Octomo(len(qubits), measure=measure)
    dataset = sweeps.prepDataset(sample, name, axes, measure=measFunc, kw=kw)

    swapTimeNameSQRT = 'sqrtSwapTime'+paraName
    swapAmpNameSQRT = 'sqrtSwapAmp'+paraName
    swapTimeName = 'swapTime'+paraName
    swapAmpName = 'swapAmp'+paraName
    def func(server, rep):
        start = 0
        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2
        q0.z = env.rect(start, q0[swapTimeNameSQRT], q0[swapAmpNameSQRT])
        start += q0[swapTimeNameSQRT]
        q1.z = env.rect(start, q1[swapTimeName], q1[swapAmpName])
        start += q1[swapTimeName]
        return measFunc(server, qubits, start, stats=stats)
    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)
    if collect:
        return data

def bellStateTomo2(sample, repetition=10, measure=[0,1], stats=1500L,
                      corrAmp=None, paraName='C', save=True, noisy=True,
                      name='Bell State QST through Bus with z compensation'):
    """
    make any of the bell state by varying the corrAmp, this is done by adding an
    additional z-pulse to the first qubit to adjust the phase
    """
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    repetition = range(repetition)
    axes = [(repetition, 'repetitions')]
    kw = {'stats': stats}
    measFunc = measurement.Octomo(len(qubits), measure=measure)
    dataset = sweeps.prepDataset(sample, name, axes, measure=measFunc, kw=kw)

    swapTimeName = 'swapTime'+paraName
    swapAmpName = 'swapAmp'+paraName

    if corrAmp is None:
        corrAmp = q0['piAmpZ']

    def func(server, rep):
        start = 0
        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2
        q0.z = env.rect(start, q0[swapTimeName]/2, q0[swapAmpName])
        start += q0[swapTimeName]/2
        q1.z = env.rect(start, q1[swapTimeName], q1[swapAmpName])
        q0.z += env.rect(start, q0['piFWHM'], corrAmp)
        # calibrate around Z-pi pulse
        start += q1[swapTimeName] + q0['piLen']/2
        return measFunc(server, qubits, start, stats=stats)

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)
    return data

def bellStateTomo2_SQRT(sample, repetition=10, measure=[0,1], stats=1500L,
                      corrAmp=None, paraName='C', save=True, noisy=True,
                      name='Bell State QST (SQRT) through Bus with z compensation'):
    """
    make any of the bell state by varying the corrAmp, this is done by adding an
    additional z-pulse to the first qubit to adjust the phase
    """
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    repetition = range(repetition)
    axes = [(repetition, 'repetitions')]
    kw = {'stats': stats}
    measFunc = measurement.Octomo(len(qubits), measure=measure)
    dataset = sweeps.prepDataset(sample, name, axes, measure=measFunc, kw=kw)

    swapTimeNameSQRT = 'sqrtSwapTime'+paraName
    swapAmpNameSQRT = 'sqrtSwapAmp'+paraName
    swapTimeName = 'swapTime'+paraName
    swapAmpName = 'swapAmp'+paraName

    if corrAmp is None:
        corrAmp = q0['piAmpZ']

    def func(server, rep):
        start = 0
        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2
        q0.z = env.rect(start, q0[swapTimeNameSQRT], q0[swapAmpNameSQRT])
        start += q0[swapTimeNameSQRT]
        q1.z = env.rect(start, q1[swapTimeName], q1[swapAmpName])
        q0.z += env.rect(start, q0['piFWHM'], corrAmp)
        # calibrate around Z-pi pulse
        start += q1[swapTimeName] + q0['piLen']/2
        return measFunc(server, qubits, start, stats=stats)

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)
    return data

### control-Z gate calibration from shor
def cZCal_Step1(sample, targetAmp=st.r[-0.25:0:0.001], measureC=0, measureT=1,
                stats=1500L,save=True, collect=False, noisy=True, update=True,
                name='Control-Z Step 1 TargetCal'):
    """
    Control-Z step 1, through Bus
    Generalized Ramsey.
    Performs the controlled-Z gate Z-pulse sequence
    with pi/2 pulse on target and no microwaves on control qubit
    [qt, qc]: [qubit_target, qubit_control]
    to calibrate the phase correction on the target qubit.
    use rect envelope to do the phase correction
    Find any maximum of the Ramsey.

    The Maximum value is the cZTargetPhaseCorrAmp

    Registry:
    control qubit:
    cZControlLen,
    target qubit:
    cZTargetLen, cZTargetAmp, cZTargetPhaseCorrLen
    cZTargetPhaseCorrAmpMax, cZTargetPhaseCorrAmpMin
    """

    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    qc = qubits[measureC] #control qubit
    qt, Qt = qubits[measureT], Qubits[measureT] #target qubit

    # repetition = range(repetition)
    axes = [(targetAmp, 'target amp phase correction')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measureT, kw=kw)

    def func(server, targetAmp):
        start = 0
        # pad time for equal state prep time in Step 2 Cal
        start += qc['piLen']/2 + qc['cZControlLen']
        # state prep
        # Control qubit no microwaves, Target qubit pi/2
        qt.xy = eh.mix(qt, eh.piHalfPulseHD(qt, start))
        start += qt['piLen']/2
        # Control is IDLE

        # Target Phase swap Q21 with R21 for iswap^2 time
        qt.z = env.rect(start, qt['cZTargetLen'], qt['cZTargetAmp'])
        start += qt['cZTargetLen']

        # Target phase correction, time is fixed sweeping amplitude
        qt.z += env.rect(start, qt['cZTargetPhaseCorrLen'], targetAmp)
        start += qt['cZTargetPhaseCorrLen'] + qt['piLen']/2
        # Final pi/2 for Ramsey, rotate about X
        qt.xy += eh.mix(qt, eh.piHalfPulseHD(qt, start, phase=0.0*np.pi))
        start += qt['piLen']/2

        # Measure only the Target
        qt.z += eh.measurePulse(qt, start)

        qt['readout'] = True

        eh.correctCrosstalkZ(qubits)
        return runQubits(server, qubits, stats=stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    if update:
        adjust.adjust_cZTargetPhaseCorrAmp(Qt, data)
    if collect:
        return data

def cZCal_Step2(sample, targetAmp=st.r[-0.25:0:0.001], measureC=0, measureT=1,
                stats=1500L, name='Control-Z Step 2 TargetCal',
                save=True, collect=True, noisy=True):
    """
    Generalized Ramsey.
    Control-Z step 2, through Bus
    Performs the controlled-Z gate Z-pulse sequence
    with pi/2 pulse on target ann a pi-pulse on the control qubit
    [qc, qt]: [qubit_control, qubit_target]
    to verify the "pi" phase shift from cZCal_Step1 on the target qubit.
    Look for a Min, should be really close to Max from Part 1

    Registry:
    control qubit:
    cZControlLen,
    target qubit:
    cZTargetLen, cZTargetAmp, cZTargetPhaseCorrLen
    cZTargetPhaseCorrAmpMax, cZTargetPhaseCorrAmpMin
    """

    sample, qubits = util.loadQubits(sample)
    qc = qubits[measureC] #control qubit
    qt = qubits[measureT] #target qubit

    axes = [(targetAmp, 'target amp phase correction')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measureT, kw=kw)

    def func(server, targetAmp):
        start = 0
        # state prep
        # Control g -> e
        qc.xy = eh.mix(qc, eh.piPulseHD(qc, start))
        start += qc['piLen']/2
        # Control iSWAP with Resonator
        qc.z = env.rect(start, qc['cZControlLen'], qc['cZControlAmp'])
        start += qc['cZControlLen']
        # state prep Target
        qt.xy = eh.mix(qt, eh.piHalfPulseHD(qt, start))
        start += qt['piLen']/2
        # Target Phase swap Q21 with R21 for iswap^2 time
        qt.z = env.rect(start, qt['cZTargetLen'], qt['cZTargetAmp'])
        start += qt['cZTargetLen']

        # Target phase correction, time is fixed sweeping amplitude
        qt.z += env.rect(start, qt['cZTargetPhaseCorrLen'], targetAmp)
        start += qt['cZTargetPhaseCorrLen'] + qt['piLen']/2
        # Final pi/2 for Ramsey, rotate about X
        qt.xy += eh.mix(qt, eh.piHalfPulseHD(qt, start, phase=0.0*np.pi))
        start += qt['piLen']/2

        # Measure
        qt.z += eh.measurePulse(qt, start)

        qt['readout'] = True
        eh.correctCrosstalkZ(qubits)
        return runQubits(server, qubits, stats=stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=dataset, noisy=noisy, save=save)
    if collect:
        return data


def cZCal_Step3(sample,controlAmp=st.r[-0.25:0:0.001], measureC=0, measureT=1,
                stats=1500L,save=True, collect=True, noisy=True, update=True,
                name='Control-Z Step 3 ControlCal'):
    """
    Generalized Ramsey.
    Control-Z step 3, through Bus
    Performs the controlled-Z gate Z-pulse sequence
    with pi/2 pulse on control and no microwaves on target qubit
    [qc, qt]: [qubit_control, qubit_target]
    This is a check experiment to calibrate the phase correction on the control qubit.
    use rect envelope to do the phase correction
    Look for Max (probably near 0.0).
    Note this experiment is orthogonal to Cal step 1 and Cal step 2,
    you do not need to iterate

    Registry:
    control qubit:
    cZControlLen, cZControlAmp, cZControlPhaseCorrLen
    cZControlPhaseCorrAmpMax, cZControlPhaseCorrAmpMin
    target qubit:
    cZTargetLen, cZTargetAmp, cZTargetPhaseCorrLen,
    cZTargetPhaseCorrAmp
    cZTargetPhaseCorrAmpMax, czTargetPhaseCorrAmpMin
    """

    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    qc = qubits[measureC] #control qubit
    qt = qubits[measureT] #target qubit
    Qc = Qubits[measureC]

    # repetition = range(repetition)
    axes = [(controlAmp, 'control amp phase correction')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measureC, kw=kw)

    def func(server, controlAmp):
        start = 0
        # state prep
        # Control g -> g + e
        qc.xy = eh.mix(qc, eh.piHalfPulseHD(qc, start))
        start += qc['piLen']/2
        # Control iSWAP with Resonator
        qc.z = env.rect(start, qc['cZControlLen'], qc['cZControlAmp'])
        start += qc['cZControlLen']
        # Target NO microwaves,
        # but need to pad time for consistent sequence with cZCal Step 1,2
        start += qt['piLen']/2
        # Target Phase swap Q21 with R21 for iswap^2 time
        qt.z = env.rect(start, qt['cZTargetLen'], qt['cZTargetAmp'])
        start += qt['cZTargetLen']

        # Target phase correction, time is fixed amp calibrated in Step 1,2
        qt.z += env.rect(start, qt['cZTargetPhaseCorrLen'], qt['cZTargetPhaseCorrAmp'])
        # Control iSWAP with Resonator
        qc.z += env.rect(start, qc['cZControlLen'], qc['cZControlAmp'])
        start += qt['cZControlLen']

        # Control phase correction, time is fixed, amp is swept:
        qc.z +=env.rect(start, qc['cZControlPhaseCorrLen'], controlAmp)
        start += qc['cZControlPhaseCorrLen'] + qc['piLen']/2
        # Final pi/2 for Ramsey, rotate about X
        qc.xy += eh.mix(qc, eh.piHalfPulseHD(qc, start, phase=0.0*np.pi))
        start += qc['piLen']/2

        # Measure
        qc.z += eh.measurePulse(qc, start)

        qc['readout'] = True
        eh.correctCrosstalkZ(qubits)
        return runQubits(server, qubits, stats=stats, probs=[1])

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    if update:
        adjust.adjust_cZControlPhaseCorrAmp(Qc, data)

    if collect:
        return data

### bellstate by cPi
def BellStatetTomo_cPi(sample, reps=10, measure=[0,1], stats=1200L, phase=np.pi/2.0,
                       save=True, collect=True, noisy=True,
                       name='BellState QST with controlled-Pi'):
    """
    generate bell state by controlled-Pi gate
    """
    sample, qubits = util.loadQubits(sample)
    qc = qubits[measure[0]]
    qt = qubits[measure[1]]

    measFunc = measurement.Octomo(len(qubits), measure)

    repetition = range(reps)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measFunc, kw=kw)

    def func(server, curr):
        start = 0
        ph = phase

        # state prep
        # Control g -> g + e
        qc.xy = eh.mix(qc, eh.piHalfPulseHD(qc, start, phase=ph))
        start += qc['piLen']/2
        # Control iSWAP with Resonator
        qc.z = env.rect(start, qc['cZControlLen'], qc['cZControlAmp'])
        start += qc['cZControlLen']

        # state prep Target
        qt.xy = eh.mix(qt, eh.piHalfPulseHD(qt, start, phase=ph))
        start += qt['piLen']/2
        # Target Phase swap Q21 with R21 for iswap^2 time
        qt.z = env.rect(start, qt['cZTargetLen'], qt['cZTargetAmp'])
        start += qt['cZTargetLen']

        # Target phase correction, time is fixed amp calibrated in Step 1,2
        qt.z += env.rect(start, qt['cZTargetPhaseCorrLen'], qt['cZTargetPhaseCorrAmp'])

        # Final pi/2 for Target, rotate about X
        qt.xy += eh.mix(qt, eh.piHalfPulseHD(qt, start+qt['cZTargetPhaseCorrLen']+qt['piLen']/2, phase=ph))

        # Control iSWAP with Resonator
        qc.z += env.rect(start, qc['cZControlLen'], qc['cZControlAmp'])
        start += qt['cZControlLen']

        # Control phase correction, time is fixed, amp calibrated in Step 3:
        qc.z +=env.rect(start, qc['cZControlPhaseCorrLen'], qc['cZControlPhaseCorrAmp'])
        start += qc['cZControlPhaseCorrLen']

        # Measure padding for tomopulses
        start += 1.0*ns
        eh.correctCrosstalkZ(qubits)
        return measFunc(server, qubits, start, **kw)

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    result = np.sum(data, axis=0)/len(repetition)
    Qk = np.reshape(result[1:], (36, 4))
    dataC = readoutFidCorr(Qk, [qc['measureF0'], qc['measureF1'],
                             qt['measureF0'], qt['measureF1']])
    if collect:
        return data, dataC

### control-Z gate calibration functions
# use gateCompiler style
def cZControlSwap(sample, measure, swapLen=st.r[0:100:1, ns],
                  swapAmp=np.arange(-0.05,0.05,0.002),
                  stats=600, name='cZ Control Swap', correctXtalkZ=True,
                  collect=True, save=True, noisy=True):
    """
    calibrate swap amp and swap len of control qubit for cZ gate
    excite the control qubit, swap into the bus resonator

    @param measure: control qubit
    """
    sample, qubits, _ = gc.loadQubits(sample, measure)

    axes = [(swapAmp, 'swap pulse amplitude'), (swapLen, 'swap pulse length')]
    deps = [('Probability', '|1>', '')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)
    def func(server, currAmp, currLen):
        alg = gc.Algorithm(qubits)
        q = alg.q0
        # excite the control qubit into |1>
        alg[gates.PiPulse([q])]
        # swap with the bus resonator
        alg[gates.cZControlSwap([q], tlen=currLen, amp=currAmp)]
        alg[gates.MeasurePQ([q])]
        alg.compile(correctXtalkZ=correctXtalkZ)
        return runQubits(server, alg.agents, stats, probs=[1])

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    if collect:
        return data

def cZTargetSwap20(sample, measure, swapLen=st.r[0:100:1, ns],
                 swapAmp=np.arange(-0.05, 0.05, 0.002),
                 stats=600, name='cZ Target Swap |20>', correctXtalkZ=True,
                 collect=True, save=True, noisy=True, state=2):
    """
    calibrate swap amp and swap length of target qubit
    excite the target qubit into |2>, swap into the bus resonator
    and swap back into the qubit, |20> <-> |11>

    @param measure: target qubit
    @param state: measure state, default is 2
    """
    sample, qubits, _ = gc.loadQubits(sample, measure)

    axes = [(swapAmp, 'swap pulse amplitude'), (swapLen, 'swap pulse length')]
    deps = [('Probability', '|{}>'.format(state), '')]
    kw = {'stats': stats, 'state': state}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)
    def func(server, currAmp, currLen):
        alg = gc.Algorithm(qubits)
        q = alg.q0
        # excite the qubit into |2>
        alg[gates.MoveToState([q], initState=0, endState=2)]
        # swap with the bus resonator
        alg[gates.cZTargetSwap([q], tlen=currLen, amp=currAmp)]
        alg[gates.MeasurePQ([q], state=state)]
        alg.compile(correctXtalkZ=correctXtalkZ)
        return runQubits(server, alg.agents, stats, probs=[1])

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    if collect:
        return data

def cZTargetSwap11(sample, measure, swapLen=st.r[0:100:1, ns],
                 swapAmp=np.arange(-0.05, 0.05, 0.002), delay=0*ns,
                 stats=600, name='cZ Target Swap |11>', correctXtalkZ=True,
                 collect=True, save=True, noisy=True, state=2):
    """
    calibrate swap amp and swap length of target qubit
    excite the control qubit into |1>, swap into the bus resonator
    excite the target qubit, swap with the resonator

    @param measure: [control, target]
    @param state: measure state, default is 2
    """
    sample, qubits, _ = gc.loadQubits(sample, measure)

    axes = [(swapAmp, 'swap pulse amplitude'), (swapLen, 'swap pulse length')]
    deps = [('Probability', '|{}>'.format(state), '')]
    kw = {'stats': stats, 'state': state}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure[1], kw=kw)

    def func(server, currAmp, currLen):
        alg = gc.Algorithm(qubits)
        qC, qT = alg.q0, alg.q1
        qC['readout'] = False
        # excite the control qubit
        alg[gates.PiPulse([qC])]
        alg[gates.PiPulse([qT])]
        alg[gates.cZControlSwap([qC])]
        alg[gates.Sync([qC, qT])]
        # swap with the bus resonator
        alg[gates.cZTargetSwap([qT], tlen=currLen, amp=currAmp)]
        alg[gates.Wait([qT], delay)]
        alg[gates.MeasurePQ([qT], state=state)]
        alg.compile(correctXtalkZ=correctXtalkZ)
        return runQubits(server, alg.agents, stats, probs=[1])

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    if collect:
        return data

def cZControlRamsey(sample, measure, controlAmp=st.r[-0.25:0:0.001], stats=600,
                    name='cZ Control Ramsey', collect=True, correctXtalkZ=True,
                    save=True, noisy=True, update=True):
    """
    Generalized Ramsey, Control-Z through Bus
    Performs the controlled-Z gate Z-pulse sequence
    with pi/2 pulse on control and no microwaves on target qubit
    This is a check experiment to calibrate the phase correction on the control qubit.
    use rect envelope to do the phase correction
    Look for Max (probably near 0.0).
    you do not need to iterate

    @param measure: [control, target]
    @param controlAmp: the amplitude of phase compensation pulse
    """
    sample, qubits, _, Qubits = gc.loadQubits(sample, measure, write_access=True)
    QC = Qubits[measure[0]]

    targetAmp = 0.0

    axes = [(controlAmp, 'control amp phase correction')]
    kw = {'stats': stats}
    deps = [('Probability', 'Control Qubit', '')]
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure[0], kw=kw)

    def func(server, amp_c):
        alg = gc.Algorithm(qubits)
        qC = alg.q0 # control qubit
        qT = alg.q1 # target qubit
        qT['readout'] = False
        qC['cZControlPhaseCorrAmp'] = amp_c
        qT['cZTargetPhaseCorrAmp'] = targetAmp
        # excite the control qubit g -> g+e
        alg[gates.PiHalfPulse([qC])]
        alg[gates.CZ([qC, qT])]
        alg[gates.Sync([qC, qT])]
        alg[gates.PiHalfPulse([qC])]
        alg[gates.MeasurePQ([qC])]
        alg.compile(correctXtalkZ=correctXtalkZ)
        return runQubits(server, alg.agents, stats=stats, probs=[1])

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    if update:
        adjust.adjust_cZControlPhaseCorrAmp(QC, data)

    if collect:
        return data

def cZTargetRamsey(sample, measure, targetAmp=st.r[-0.25:0:0.001], stats=600,
                   name='cZ Target Ramsey', collect=True, correctXtalkZ=True,
                   save=True, noisy=True, update=True):
    """
    Generalized Ramsey.
    Performs the controlled-Z gate Z-pulse sequence
    with pi/2 pulse on target and no microwaves (or Pi) on control qubit
    to calibrate the phase correction on the target qubit.
    use rect envelope to do the phase correction
    Find any maximum of the Ramsey (when no microwaves on control qubit)

    The Maximum value is the cZTargetPhaseCorrAmp

    @param measure: [control, target]
    @param targetAmp: the amp of phase compensation pulse
    """
    sample, qubits, _, Qubits = gc.loadQubits(sample, measure, write_access=True)
    QT = Qubits[measure[1]]

    axes = [(targetAmp, 'target amp phase correction')]
    kw = {'stats': stats}
    deps = [('Probability', 'Control = |%d>'%state, '') for state in [0,1]]
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure[1], kw=kw)

    def func(server, amp_t):
        reqs = []
        for cpi in [False, True]:
            alg = gc.Algorithm(qubits)
            qC = alg.q0
            qT = alg.q1
            qC['readout'] = False
            qT['cZTargetPhaseCorrAmp'] = amp_t
            if cpi:
                alg[gates.PiPulse([qC])]
            alg[gates.PiHalfPulse([qT])]
            alg[gates.Sync([qC, qT])]
            alg[gates.CZ([qC, qT])]
            alg[gates.Sync([qC, qT])]
            alg[gates.PiHalfPulse([qT])]
            alg[gates.MeasurePQ([qT])]
            alg.compile(correctXtalkZ=correctXtalkZ)
            reqs.append(
                runQubits(server, alg.agents, stats=stats, probs=[1])
            )
        data = yield FutureList(reqs)
        probs = [p[0] for p in data]
        returnValue(probs)

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)
    # data = np.array(data)

    if update:
        adjust.adjust_cZTargetPhaseCorrAmp(QT, data[:, (0, 1)])

    if collect:
        return data

def cZControlRamsey2(sample, measure, control_phase=np.linspace(-2*np.pi, 2*np.pi, 251), stats=600,
                     name='cZ Control Ramsey with Phase', collect=True, correctXtalkZ=True,
                     save=True, noisy=True, update=True):
    """
    Generalized Ramsey, Control-Z through Bus
    Performs the controlled-Z gate Z-pulse sequence
    with pi/2 pulse on control and no microwaves on target qubit
    This is a check experiment to calibrate the phase correction on the control qubit.
    use xy_phase to do the phase correction
    Look for Max (probably near 0.0).
    you do not need to iterate

    @param measure: [control, target]
    @param control_phase: the phase correction of control qubit
    """
    sample, qubits, _, Qubits = gc.loadQubits(sample, measure, write_access=True)
    qC = qubits[measure[0]]
    QC = Qubits[measure[0]]


    axes = [(control_phase, 'control phase correction')]
    kw = {'stats': stats}
    deps = [('Probability', 'Control Qubit', '')]
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure[0], kw=kw)

    def func(server, curr_phase):
        alg = gc.Algorithm(qubits)
        qC = alg.q0 # control qubit
        qT = alg.q1 # target qubit
        qT['readout'] = False
        qC['cZControlPhaseCorr'] = curr_phase
        # excite the control qubit g -> g+e
        alg[gates.PiHalfPulse([qC])]
        alg[gates.CZ2([qC, qT])]
        alg[gates.Sync([qC, qT])]
        alg[gates.PiHalfPulse([qC])]
        alg[gates.MeasurePQ([qC])]
        alg.compile(correctXtalkZ=correctXtalkZ)
        return runQubits(server, alg.agents, stats=stats, probs=[1])

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    if update:
        phase, prob = data.T
        val = qC.get('cZControlPhaseCorr', 0.0)
        traces = [{'x': phase, 'y': prob, 'args': ('b.-', )}]
        params = [{'name': 'control phase', 'val': float(val),
                   'range': (min(control_phase), max(control_phase)),
                   'axis': 'x', 'color': 'b'}]
        result = adjust.adjust(params, traces)
        if result is not None:
            sele_phase = result['control phase']
            key = 'cZControlPhaseCorr'
            print "Old {}: {}".format(key, qC.get(key, 0.0))
            index = (abs(phase - sele_phase) < 0.5)
            phase_keep = phase[index]
            prob_keep = prob[index]
            p = np.polyfit(phase_keep, prob_keep, 2)
            fit_val = -p[1]/p[0]/2.0
            print "Selected {}: {}".format(key, sele_phase)
            print "Fitting {}: {}".format(key, fit_val)
            QC[key] = fit_val

    if collect:
        return data


def cZTargetRamsey2(sample, measure, targetLen=st.r[0:100:1, ns],
                    targetAmp=np.arange(-0.05, 0.05, 0.002),
                    targetPhase=np.linspace(-np.pi, np.pi, 51),
                    stats=600, name='cZ Target Ramsey with Phase', update=True,
                    correctXtalkZ=True, collect=True, save=True, noisy=True, plot=True):
    """
    Generalized Ramsey.
    Performs the controlled-Z gate Z-pulse sequence
    with pi/2 pulse on target and no microwaves (or Pi) on control qubit
    to calibrate the phase correction on the target qubit.
    use xy_phase to do the phase correction
    Find any maximum of the Ramsey (when no microwaves on control qubit, or control=|0>)

    the maximum value of P1 when control=|0> should be the same x-value with
    the minimum value of P1 when control=|1>.

    @param measure: [control, target]
    @param targetAmp: the amp of target swap pulse, if None, use the registry key, 'cZTargetAmp'
    @param targetLen: the length of target swap pulse, if None, use the registry key, 'cZTargetLen'
    @param targetPhase: the phase correction, if None, use the registry key, 'cZTargetPhaseCorr'
    @param update: targetLen and targetAmp are scalars, and targetPhase is a list,
                   then adjust for phase correction
    """

    sample, qubits, _, Qubits = gc.loadQubits(sample, measure, True)
    target_qubit = qubits[measure[1]]
    QT = Qubits[measure[1]]

    if targetLen is None:
        targetLen = target_qubit['cZTargetLen']
    if targetAmp is None:
        targetAmp = target_qubit['cZTargetAmp']
    if targetPhase is None:
        targetPhase = target_qubit['cZTargetPhaseCorr']

    axes = [(targetAmp, 'cZ target swap pulse amplitude'),
            (targetLen, 'cZ target swap pulse length'),
            (targetPhase, 'cZ target phase correction')]
    deps = [('Probability', 'Control = |{}>'.format(s), '') for s in [0, 1]]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure[1], kw=kw)

    def func(server, currAmp, currLen, currPhase):
        reqs = []
        for cpi in [False, True]:
            alg = gc.Algorithm(qubits)
            qC, qT = alg.q0, alg.q1
            qC['readout'] = False
            qT['cZTargetPhaseCorr'] = currPhase
            qT['cZTargetLen'] = currLen
            qT['cZTargetAmp'] = currAmp
            if cpi:
                alg[gates.PiPulse([qC])]
            alg[gates.PiHalfPulse([qT])]
            alg[gates.Sync([qC, qT])]
            alg[gates.CZ2([qC, qT])]
            alg[gates.Sync([qC, qT])]
            alg[gates.PiHalfPulse([qT])]
            alg[gates.MeasurePQ([qT])]
            alg.compile(correctXtalkZ=correctXtalkZ)
            reqs.append(
                runQubits(server, alg.agents, stats=stats, probs=[1])
            )
        data = yield FutureList(reqs)
        probs = [p[0] for p in data]
        returnValue(probs)

    local_dataset = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    dim_check = (np.iterable(targetLen) or np.iterable(targetAmp))
    dim_check = (not dim_check) and np.iterable(targetPhase)

    if plot and dim_check:
        phase, prob0, prob1 = local_dataset.T
        plt.figure()
        plt.plot(phase, prob0, 'b.-', label='control=|0>')
        plt.plot(phase, prob1, 'r.-', label='control=|1>')
        plt.grid()
        plt.xlabel("Target Phase Correction")
        plt.ylabel("P1")
        plt.legend(loc=0)

    if update and dim_check:
        phase, prob0, prob1 = local_dataset.T
        val = target_qubit.get('cZTargetPhaseCorr', 0.0)
        traces = [{'x': phase, 'y': prob0, 'args': ('b.-', )}]
        params = [{'name': 'target phase', 'val': float(val),
                   'range': (min(targetPhase), max(targetPhase)),
                   'axis': 'x', 'color': 'b'}]
        result = adjust.adjust(params, traces)
        if result is not None:
            sele_phase = result['target phase']
            key = 'cZTargetPhaseCorr'
            print "Old {}: {}".format(key, target_qubit.get(key, 0.0))
            index = (abs(phase - sele_phase) < 0.5)
            phase_keep = phase[index]
            prob_keep = prob0[index]
            p = np.polyfit(phase_keep, prob_keep, 2)
            fit_val = -p[1]/p[0]/2.0
            print "Selected {}: {}".format(key, sele_phase)
            print "Fitting {}: {}".format(key, fit_val)
            QT[key] = fit_val

    if collect:
        return local_dataset


def qstBellStateCZ(sample, control, target, repetition=10,
                   bellstate='Psi+', correctXtalkZ=True, cz='CZ',
                   name='QST - Bell State', stats=1200, tBuf=0*ns,
                   collect=True, save=True, noisy=True):
    """
    generate bell state by cZ gate, and apply QST measurement,
    Four bell state:
    |Psi+> = |01> + |10>
    |Psi-> = |01> - |10>
    |Phi+> = |00> + |11>
    |Phi-> = |00> - |11>

    @param control: control qubit in cZ gate
    @param target:  target qubit in cZ gate
    @param repetition: number of repetition
    @param cz: should be in ["CZ", "CZ2"]
    @param bellstate: one of [ "Psi+", "Psi-", "Phi+", "Phi-" ]
    """
    sample, qubits, _ = gc.loadQubits(sample, measure=[control, target])
    cz = cz.upper()
    assert cz in ["CZ", "CZ2"]
    assert bellstate in ['Psi+', 'Psi-', 'Phi+', 'Phi-']
    name += ' {} - {}'.format(cz, bellstate)

    if cz == "CZ":
        CZ = gates.CZ
    elif cz == "CZ2":
        CZ = gates.CZ2

    axes = [(range(repetition), 'repetition')]
    kw = {'stats': stats, 'repetition': repetition, 'BellState': bellstate, 'tBuf': tBuf}
    measFunc = measurement.Octomo(len(qubits), measure=[control, target], tBuf=tBuf)
    dataset = sweeps.prepDataset(sample, name, axes, measure=measFunc, kw=kw)

    def func(server, i):
        alg = gc.Algorithm(qubits)
        qC = alg.q0
        qT = alg.q1
        if bellstate == 'Psi+':
            # 01 + 10
            alg[gates.PiHalfPulse([qC])] # Xpi/2
            alg[gates.PiHalfPulse([qT])] # Xpi/2
            alg[CZ([qC, qT])]
            alg[gates.PiHalfPulse([qC])] # Xpi/2
        elif bellstate == 'Psi-':
            # 01 - 10
            alg[gates.PiHalfPulse([qC])] # Xpi/2
            alg[gates.PiHalfPulse([qT], phase=np.pi)] # -Xpi/2
            alg[CZ([qC, qT])]
            alg[gates.PiHalfPulse([qC])] # Xpi/2
        elif bellstate == 'Phi+':
            # 00 + 11
            alg[gates.PiHalfPulse([qC])] # Xpi/2
            alg[gates.PiHalfPulse([qT])] # Xpi/2
            alg[CZ([qC, qT])]
            alg[gates.PiHalfPulse([qC], phase=np.pi)] # -Xpi/2
        elif bellstate == 'Phi-':
            # 00 - 11
            alg[gates.PiHalfPulse([qC])] # Xpi/2
            alg[gates.PiHalfPulse([qT], phase=np.pi)] # -Xpi/2
            alg[CZ([qC, qT])]
            alg[gates.PiHalfPulse([qC], phase=np.pi)] # -Xpi/2
        else:
            raise Exception("not implemented")
        alg[gates.Sync([qC, qT])]
        alg.compile(correctXtalkZ=False)
        t = max([q['_t'] for q in alg.qubits])
        return measFunc(server, alg.agents, t, correctCrosstalkZ=correctXtalkZ, stats=stats)

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    if collect:
        return data


def qstBellStateSqrtSwap(sample, measure, repetition=10, bellstate='Phi+', paramName='',
                     correctXtalkZ=True, name='QST - Bell State SqrtSwap', stats=1200,
                     tomoPhase=0.0, tBuf=0*ns, collect=True, save=True, noisy=True):
    """
    generate bell state swap, and apply QST measurement,
    Only two bell state can be generated by swap operation
    |Phi+> = |01> + |10>
    |Phi-> = |01> - |10>
    use sqrtSwap registry key
    @param measure: qubits, the first element sqrtSwap with the bus
    @param repetition: number of reptition
    @param bellstate: should be one of ["Phi+", "Phi-"]
    """
    sample, qubits, _ = gc.loadQubits(sample, measure=measure)
    name += ' - {}'.format(bellstate)

    assert bellstate in ["Phi+", "Phi-"]
    axes = [(range(repetition), 'repetition')]
    kw = {'stats': stats, 'repetition': repetition, 'BellState': bellstate, 'tBuf': tBuf,
          'tomoPhase': tomoPhase}
    measFunc = measurement.Octomo(len(qubits), measure=measure, tBuf=tBuf)
    dataset = sweeps.prepDataset(sample, name, axes, measure=measFunc, kw=kw)

    def func(server, i):
        alg = gc.Algorithm(qubits)
        # excite the first qubit
        alg[gates.PiPulse([alg.q0])]
        # sqrtSwap with Bus resonator
        alg[gates.SqrtSwap([alg.q0], paramName=paramName)]
        alg[gates.Sync([alg.q0, alg.q1])]
        # the second qubit iSwap with bus resonator
        alg[gates.Swap([alg.q1], paramName=paramName)]
        if bellstate == 'Phi+':
            alg[gates.PiPulseZ(alg.q0)]
        alg[gates.Sync([alg.q0, alg.q1])]
        alg.compile(correctXtalkZ=False)
        t = max([q['_t'] for q in alg.qubits])
        alg.q0['tomoPhase'] = tomoPhase
        return measFunc(server, alg.agents, t, correctCrosstalkZ=correctXtalkZ, stats=stats)

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    if collect:
        return data

def qstBellStateSwap(sample, measure, repetition=10, bellstate='Psi+', paramName='',
                     correctXtalkZ=True, name='QST - Bell State SwapHalf', stats=1200,
                     tBuf=0*ns, collect=True, save=True, noisy=True, tomoPhase=0.0):
    """
    generate bell state swap, and apply QST measurement,
    Only two bell state can be generated by swap operation
    |Psi+> = |01> + |10>
    |Psi-> = |01> - |10>
    use Swap registry key (SwapHalf Gate)
    @param measure: qubits, the first element sqrtSwap with the bus
    @param repetition: number of reptition
    @param bellstate: should be one of ["Phi+", "Phi-"]
    """
    sample, qubits, _ = gc.loadQubits(sample, measure=measure)
    name += ' - {}'.format(bellstate)

    assert bellstate in ["Psi+", "Psi-"]
    axes = [(range(repetition), 'repetition')]
    kw = {'stats': stats, 'repetition': repetition, 'BellState': bellstate, 'tBuf': tBuf,
          'tomoPhase': tomoPhase}
    measFunc = measurement.Octomo(len(qubits), measure=measure, tBuf=tBuf)
    dataset = sweeps.prepDataset(sample, name, axes, measure=measFunc, kw=kw)

    def func(server, i):
        alg = gc.Algorithm(qubits)
        # excite the first qubit
        alg[gates.PiPulse([alg.q0])]
        # sqrtSwap with Bus resonator
        alg[gates.SwapHalf([alg.q0], paramName=paramName)]
        alg[gates.Sync([alg.q0, alg.q1])]
        # the second qubit iSwap with bus resonator
        alg[gates.Swap([alg.q1], paramName=paramName)]
        if bellstate == 'Psi+':
            alg[gates.PiPulseZ([alg.q0])]
        alg[gates.Sync([alg.q0, alg.q1])]
        alg.compile(correctXtalkZ=False)
        t = max([q['_t'] for q in alg.qubits])
        alg.q0['tomoPhase'] = tomoPhase
        return measFunc(server, alg.agents, t, correctCrosstalkZ=correctXtalkZ, stats=stats)

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    if collect:
        return data

def qstQBQSwap(sample, paramName, measure=[0, 1], repetition=10, swapAmp=None, swapTime=None, axis=('X', 'X'),
                name='QST - Qubt Bus Qubit Swap', delay=0.0*ns, collect=True, save=True, stats=1200,
                noisy=True, correctXtalkZ=False, plot=True):
    """
    excite qubit with Pi (Pi/2) or None pulse, the axis is given by axis,
    then Full Swap, and finally QST
    @param axis, should be in tomo operators, axis = [op1, op2], op1, op2 should in
        ["I", "X", "Y", "X/2", "Y/2", "-X", "-Y", "-X/2", "-Y/2"]
    """
    sample, qubits, _ = gc.loadQubits(sample, measure)
    axes = [(range(repetition), 'repetition')]
    measFunc = measurement.Octomo(len(qubits), measure=measure)
    kw = {"stats": stats, "tomo ops": axis}
    dataset = sweeps.prepDataset(sample, name, axes=axes, measure=measFunc, kw=kw)


    def func(server, i):
        alg = gc.Algorithm(qubits)
        q0 = alg.q0
        q1 = alg.q1
        q0['readout'] = True
        alg[gates.Tomography([q0, q1], axis)]
        alg[gates.Swap([q0], paramName=paramName)]
        alg[gates.Wait([q0], waitTime=delay)]
        alg[gates.Sync([q0, q1])]
        alg[gates.Swap([q1], paramName=paramName)]
        alg[gates.Sync([q0, q1])]
        alg[gates.Swap([q0], paramName=paramName)]
        alg[gates.Sync([q0, q1])]
        alg.compile()
        t = q1['_t']
        return measFunc(server, alg.agents, t, stats=stats)

    local_dataset = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)
    if plot:
        probs = np.asarray(local_dataset[:, 1:]).reshape(-1, 36, 4)
        rhos = np.array([tomo.qst(diag, 'octomo2') for diag in probs])
        rho_mean = np.mean(rhos, axis=0)
        tomography.manhattan3d(rho_mean.real)
        tomography.manhattan3d(rho_mean.imag)
    if collect:
        return local_dataset

def qptiSwapBus(sample, measure, repetition=10, paramName='', delay=0*ns,
                correctXtalkZ=True, name='QPT - iSwap through Bus MQ',
                stats=1200, collect=True, save=True, noisy=True, plot=True):
    """
    qpt of iSwap, q0 -> Bus -> q1, and Bus -> q0
    @param measure: [q0, q1]
    """
    sample, qubits, _ = gc.loadQubits(sample, measure=measure)

    qptPrepOps = list(tomo.gen_qptPrep_tomo_ops(tomo.octomo_names, 2))
    qstOps = list(tomo.gen_qst_tomo_ops(tomo.octomo_names, 2))
    opList = list(itertools.product(qptPrepOps, qstOps))

    index = range(len(opList))
    index_gen = st.averagedScan(index, repetition, noisy=noisy)
    axes = [(index_gen(), "Tomo Operation")]
    deps = [('Probability', '|00>', ''),
            ('Probability', '|01>', ''),
            ('Probability', '|10>', ''),
            ('Probability', '|11>', '')]
    kw = {'stats': stats, 'repetition': repetition}
    dataset = sweeps.prepDataset(sample, name, axes=axes, dependents=deps, measure=measure, kw=kw)

    def func(server, i):
        qptOp, qstOp = opList[i]
        if noisy:
            print qptOp, qstOp
        alg = gc.Algorithm(qubits)
        q0, q1 = alg.q0, alg.q1
        # for qpt pre operation
        alg[gates.Tomography([q0, q1], qptOp)]
        alg[gates.iSwap([q0, q1], delay=delay, paramName=paramName)]
        alg[gates.Swap([q0], paramName=paramName)]
        alg[gates.Sync([q0, q1])]
        # for qst measurement
        alg[gates.Tomography([q0, q1], qstOp)]
        alg[gates.MeasurePQ([q0, q1], sync=True)]
        alg.compile(correctXtalkZ=correctXtalkZ)
        return runQubits(server, alg.agents, stats=stats)

    data = sweeps.grid(func, axes, collect=collect, noisy=noisy, save=save, dataset=dataset)

    probs = data[:, 1:].reshape((-1, 36, 4))
    rhos = np.array([tomo.qst(diag, 'octomo2') for diag in probs])
    rho_in = np.array([
                      [[ 1.0+0.j ,  0.0+0.j ], [ 0.0+0.j ,  0.0+0.j ]],    # I
                      [[0.5 + 0.j, 0.0 + 0.5j], [0.0 - 0.5j, 0.5 + 0.j]],  # Xpi/2
                      [[ 0.5+0.j ,  0.5+0.j ], [ 0.5+0.j ,  0.5+0.j ]],    # Ypi/2
                      [[0.5, -0.5j], [0.5j, 0.5]], # -Xpi/2
                      [[0.5, -0.5], [-0.5, 0.5]],  # -Ypi/2
                      [[0, 0], [0, 1]],            # Xpi
                      ])
    rhos_in = np.array([np.kron(i, j) for i, j in itertools.product(rho_in, rho_in)])
    rhos = np.split(rhos, repetition)
    chis = np.array([tomo.qpt(rhos_in, rs, 'sigma2') for rs in rhos])
    if plot:
        axesLabel = ['I', "X", "Y", "Z"]
        axesLabel = itertools.product(axesLabel, axesLabel)
        axesLabel = [v[0]+v[1] for v in axesLabel]
        tomography.manhattan3d(np.average(chis, axis=0), axesLabels=axesLabel)
    if collect:
        return data, chis

def qptCZGate(sample, measure, repetition=10, correctXtalkZ=True, cz='CZ',
              stats=1200, collect=True, save=True, noisy=True, plot=True):
    """
    qpt of cZ gate
    @param measure: [control, target]
    @param cz: should be in ["CZ", "CZ2"]
    """
    sample, qubits, _ = gc.loadQubits(sample, measure=measure)
    cz = cz.upper()
    assert cz in ["CZ", "CZ2"]
    if cz == "CZ":
        cZgate = gates.CZ
        name = "QPT - CZ Gate"
    elif cz == "CZ2":
        cZgate = gates.CZ2
        name =  "QPT - CZ2 Gate"
    qptPrepOps = list(tomo.gen_qptPrep_tomo_ops(tomo.octomo_names, 2))
    qstOps = list(tomo.gen_qst_tomo_ops(tomo.octomo_names, 2))
    opList = list(itertools.product(qptPrepOps, qstOps))

    index = range(len(opList))
    index_gen = st.averagedScan(index, repetition, noisy=noisy)
    axes = [(index_gen(), "Tomo Operation")]
    deps = [('Probability', '|00>', ''),
            ('Probability', '|01>', ''),
            ('Probability', '|10>', ''),
            ('Probability', '|11>', '')]
    kw = {'stats': stats, 'repetition': repetition}
    dataset = sweeps.prepDataset(sample, name, axes=axes, dependents=deps, measure=measure,
                                 kw=kw)

    def func(server, i):
        qptOp, qstOp = opList[i]
        if noisy:
            print qptOp, qstOp
        alg = gc.Algorithm(qubits)
        qC, qT = alg.q0, alg.q1
        # for qpt pre operation
        alg[gates.Tomography([qC, qT], qptOp)]
        alg[cZgate([qC, qT])]
        alg[gates.Sync([qC, qT])]
        # for qst measurement
        alg[gates.Tomography([qC, qT], qstOp)]
        alg[gates.MeasurePQ([qC, qT], sync=True)]
        alg.compile(correctXtalkZ=correctXtalkZ)
        return runQubits(server, alg.agents, stats=stats)

    data = sweeps.grid(func, axes, collect=collect, noisy=noisy, save=save, dataset=dataset)
    # print data.shape
    # return data
    probs = data[:, 1:].reshape((-1, 36, 4))
    rhos = np.array([tomo.qst(diag, 'octomo2') for diag in probs])
    rho_in = np.array([
                      [[ 1.0+0.j ,  0.0+0.j ], [ 0.0+0.j ,  0.0+0.j ]],    # I
                      [[0.5 + 0.j, 0.0 + 0.5j], [0.0 - 0.5j, 0.5 + 0.j]],  # Xpi/2
                      [[ 0.5+0.j ,  0.5+0.j ], [ 0.5+0.j ,  0.5+0.j ]],    # Ypi/2
                      [[0.5, -0.5j], [0.5j, 0.5]], # -Xpi/2
                      [[0.5, -0.5], [-0.5, 0.5]],  # -Ypi/2
                      [[0, 0], [0, 1]],            # Xpi
                      ])
    rhos_in = np.array([np.kron(i, j) for i, j in itertools.product(rho_in, rho_in)])
    rhos = np.split(rhos, repetition)
    chis = np.array([tomo.qpt(rhos_in, rs, 'sigma2') for rs in rhos])
    if plot:
        axesLabel = ['I', "X", "Y", "Z"]
        axesLabel = itertools.product(axesLabel, axesLabel)
        axesLabel = [v[0] + v[1] for v in axesLabel]
        tomography.manhattan3d(np.average(chis, axis=0), axesLabels=axesLabel)
    if collect:
        return data, chis

def qptCNOTGate(sample, measure, repetition=10, correctXtalkZ=True, cnot='CNOT',
              name='QPT - CNOT Gate', stats=1200, collect=True,
              save=True, noisy=True, plot=True):
    """
    qpt of cNOT gate
    @param measure: [control, target]
    @param cnot: should be in ["CNOT", "CNOT2"]
    """
    sample, qubits, _ = gc.loadQubits(sample, measure=measure)

    cnot = cnot.upper()
    assert cnot in ["CNOT", "CNOT2"]
    if cnot == "CNOT":
        cNOTgate = gates.CNOT
        name = "QPT - CNOT Gate"
    elif cnot == "CNOT2":
        cNOTgate = gates.CNOT2
        name = "QPT - CNOT2 Gate"

    qptPrepOps = list(tomo.gen_qptPrep_tomo_ops(tomo.octomo_names, 2))
    qstOps = list(tomo.gen_qst_tomo_ops(tomo.octomo_names, 2))
    opList = list(itertools.product(qptPrepOps, qstOps))

    # axes = [(range(repetition), 'repetition')]
    # measFunc = measurement.Octomo(len(qubits), measure=measure, tBuf=tBuf)
    # mdeps = measFunc.dependents()
    # deps = [("prepOps: {}, {}".format(ops, n), l, u) for ops in qptPrepOps for n,l,u in mdeps]

    index = range(len(opList))
    index_gen = st.averagedScan(index, repetition, noisy=noisy)
    axes = [(index_gen(), "Tomo Operation")]
    deps = [('Probability', '|00>', ''),
            ('Probability', '|01>', ''),
            ('Probability', '|10>', ''),
            ('Probability', '|11>', '')]
    kw = {'stats': stats, 'repetition': repetition}
    dataset = sweeps.prepDataset(sample, name, axes=axes, dependents=deps, measure=measure,
                                 kw=kw)

    def func(server, i):
        qptOp, qstOp = opList[i]
        if noisy:
            print qptOp, qstOp
        alg = gc.Algorithm(qubits)
        qC, qT = alg.q0, alg.q1
        # for qpt pre operation
        alg[gates.Tomography([qC, qT], qptOp)]
        alg[cNOTgate([qC, qT], sync=True)]
        alg[gates.Sync([qC, qT])]
        # for qst measurement
        alg[gates.Tomography([qC, qT], qstOp)]
        alg[gates.MeasurePQ([qC, qT], sync=True)]
        alg.compile(correctXtalkZ=correctXtalkZ)
        return runQubits(server, alg.agents, stats=stats)

    data = sweeps.grid(func, axes, collect=collect, noisy=noisy, save=save, dataset=dataset)
    # print data.shape
    # return data
    probs = data[:, 1:].reshape((-1, 36, 4))
    rhos = np.array([tomo.qst(diag, 'octomo2') for diag in probs])
    rho_in = np.array([
        [[1.0 + 0.j, 0.0 + 0.j], [0.0 + 0.j, 0.0 + 0.j]],  # I
        [[0.5 + 0.j, 0.0 + 0.5j], [0.0 - 0.5j, 0.5 + 0.j]],  # Xpi/2
        [[0.5 + 0.j, 0.5 + 0.j], [0.5 + 0.j, 0.5 + 0.j]],  # Ypi/2
        [[0.5, -0.5j], [0.5j, 0.5]],  # -Xpi/2
        [[0.5, -0.5], [-0.5, 0.5]],  # -Ypi/2
        [[0, 0], [0, 1]],  # Xpi
    ])
    rhos_in = np.array([np.kron(i, j) for i, j in itertools.product(rho_in, rho_in)])
    rhos = np.split(rhos, repetition)
    chis = np.array([tomo.qpt(rhos_in, rs, 'sigma2') for rs in rhos])
    if plot:
        axesLabel = ['I', "X", "Y", "Z"]
        axesLabel = itertools.product(axesLabel, axesLabel)
        axesLabel = [v[0] + v[1] for v in axesLabel]
        tomography.manhattan3d(np.average(chis, axis=0), axesLabels=axesLabel)
    if collect:
        return data, chis


### control-Z gate calibration from DevDaniel/cz.py
### similar with cZCal_StepX

def czRamseyControlGaussian(sample, control, target, cZControlPhaseCorr=None,
                    delay=3.0*ns, name='cZ Ramsey Control Gaussian', stats=600,
                    collect=False, save=True, noisy=True):
    """
    similar with cZCal Step 3,
    But here in czRamsey, it should be the first step in cZ-gate calibration.
    calibrate the phase compensation of the control qubit.
    Use gaussian pulse instead of rectangle pulse
    """
    sample, qubits = util.loadQubits(sample)
    qC = qubits[control]
    qT = qubits[target]

    qC['readout'] = True

    if cZControlPhaseCorr is None:
        cZControlPhaseCorr = np.linspace(-2*np.pi*0.25, 2*np.pi*1.75, 101)

    cZTargetPhaseCorr=0.0

    axes = [(cZTargetPhaseCorr, 'cZTargetPhaseCorr'), (cZControlPhaseCorr, 'cZControlPhaseCorr')]
    kw = {'stats': stats, 'cZTargetPhaseCorr': cZTargetPhaseCorr}
    dataset = sweeps.prepDataset(sample, name, axes, measure=control, kw=kw)

    def func(server, target_corr_phase, control_corr_phase):
        t = 0.0*ns
        qC['xy'] = qT['xy'] = env.NOTHING
        for q in qubits:
            q['z'] = env.NOTHING
        # Excite control
        qC['xy'] += eh.piHalfPulse(qC, t, phase = 0)
        t += 0.5*max(qC['piLen'],qT['piLen'])
        # iSwap control into resonator
        qC['z'] += env.rect(t, qC['cZControlLen'], qC['cZControlAmp'])
        t += qC['cZControlLen']
        # Do (iSwap)**2 on target
        qT['z'] += env.rect(t, qT['cZTargetLen'], qT['cZTargetAmp'])
        t += qT['cZTargetLen']
        # Retrieve photon from resonator into control
        qC['z'] += env.rect(t, qC['cZControlLen'], qC['cZControlAmp'])
        t += qC['cZControlLen']
        # Compensate control and target phases
        t += 0.5*max(qC['piLen'],qT['piLen'])
        t += delay
        qC['z'] += eh.rotPulseZ(qC, t, control_corr_phase)
        t += 0.5*max(qC['piLen'],qT['piLen'])
        # Final pi pulses for Ramsey
        t += 0.5*max(qC['piLen'],qT['piLen'])
        qC['xy'] += eh.piHalfPulse(qC, t, phase = 0)
        t += 0.5*qC['piLen']
        # Measure control
        qC['z'] += eh.measurePulse(qC, t)

        # Apply sideband mixing
        qC['xy'] = eh.mix(qC, qC['xy'])
        eh.correctCrosstalkZ(qubits)

        return runQubits(server, qubits, stats, probs=[1])

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    if collect:
        return data


def czRamseyTargetGaussian(sample, control, target, cZControlPhaseCorr=None, cZTargetPhaseCorr=None,
                   delay=3.0*ns, name='czRamseyTarget Gaussian', stats=600,
                   collect=True, save=True, noisy=True):
    """
    similar with cZCal Step 1 and 2,
    But here in czRamsey, it should be the second step in cZ-gate calibration.
    calibrate the phase compensation of the target qubit.
    It combines cZCal Step 1 and 2
    Use gaussian pulse instead of rectangle pulse
    """
    sample, qubits = util.loadQubits(sample)
    qC = qubits[control]
    qT = qubits[target]

    qT['readout'] = True

    if cZTargetPhaseCorr is None:
        cZTargetPhaseCorr = np.linspace(-2*np.pi*0.25, 2*np.pi*1.75, 101)

    if cZControlPhaseCorr is None:
        cZControlPhaseCorr = qC['czControlCompAngle']

    axes = [(cZTargetPhaseCorr, 'cZTargetPhaseCorr'), (cZControlPhaseCorr, 'cZControlPhaseCorr')]
    deps = [('Probability', 'Control = |%d>'%state, '') for state in [0,1]]
    kw = {'stats': stats, 'cZControlPhaseCorr': cZControlPhaseCorr}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=target, kw=kw)

    def func(server, target_phase_corr, control_phase_corr):
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
            qC['z'] += env.rect(t, qC['cZControlLen'], qC['cZControlAmp'])
            t += qC['cZControlAmp']
            #Do (iSwap)**2 on target
            qT['z'] += env.rect(t, qT['cZTargetLen'], qT['cZTargetAmp'])
            t += qT['cZTargetLen']
            #Retrieve photon from resonator into control
            qC['z'] += env.rect(t, qC['cZControlLen'], qC['cZControlAmp'])
            t += qC['cZControlLen']
            #Compensate control and target phases
            t += 0.5*max(qC['piLen'],qT['piLen'])
            t += delay
            qC['z'] += eh.rotPulseZ(qC, t, control_phase_corr)
            qT['z'] += eh.rotPulseZ(qT, t, target_phase_corr)
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

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)
    if collect:
        return data


def czBellStateTomoGaussian(sample, control, target, repetition=10, cZTargetPhaseCorr=None,
                    cZControlPhaseCorr=None, tBuf=3.0*ns, save=True, noisy=True,
                    name='cZ BellState Gaussian QST', stats=600, collect=True):
    """
    Bell State Using cZ Gate
    (using gaussian pulse to do the phase compensation)
    """
    sample, qubits = util.loadQubits(sample)
    qC = qubits[control]
    qT = qubits[target]
    measure = [control, target]
    qT['readout'] = True
    qC['readout'] = True
    N = len(sample['config'])
    measureFunc = measurement.Tomo(N, measure, tBuf=tBuf,)

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measureFunc, kw=kw)
    if cZControlPhaseCorr is not None:
        qC['cZControlPhaseCorr'] = cZControlPhaseCorr
    if cZTargetPhaseCorr is not None:
        qT['cZTargetPhaseCorr'] = cZTargetPhaseCorr

    def func(server, repi):
        # Initialize everything
        t = 0.0*ns
        for q in qubits:
            q['z'] = env.NOTHING
            q['xy'] = env.NOTHING
        # Single qubit preparations
        qC['xy'] += eh.piHalfPulse(qC, t, phase = np.pi/2)
        qT['xy'] += eh.piHalfPulse(qT, t, phase = np.pi/2)
        t += 0.5*max(qC['piLen'],qT['piLen'])
        ### BEGIN CONTROLLED Z GATE ###
        #iSwap control into resonator
        qC['z'] += env.rect(t, qC['cZControlLen'], qC['cZControlAmp'])
        t += qC['cZControlLen']
        # Do (iSwap)**2 on target
        qT['z'] += env.rect(t, qT['cZTargetLen'], qT['cZTargetAmp'])
        t += qT['cZTargetLen']
        # Retrieve photon from resonator into control
        qC['z'] += env.rect(t, qC['cZControlLen'], qC['cZControlLen'])
        t += qC['cZControlLen']
        # Compensate control and target phases
        t += 0.5*max(qC['piLen'],qT['piLen'])
        t += tBuf
        qC['z'] += eh.rotPulseZ(qC, t, qC['cZControlPhaseCorr'])
        qT['z'] += eh.rotPulseZ(qT, t, qT['cZTargetPhaseCorr'])
        t += 0.5*max(qC['piLen'],qT['piLen'])
        ### END CONTROLLED Z GATE ###
        t += qT['piLen']/2
        qT['xy'] += eh.piHalfPulse(qT, t, phase=np.pi/2)
        t += qT['piLen']/2
        # Apply sideband mixing
        qC['xy'] = eh.mix(qC, qC['xy'])
        qT['xy'] = eh.mix(qT, qT['xy'])
        eh.correctCrosstalkZ(qubits)
        return measureFunc(server, qubits, t+tBuf, stats=stats)
    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy, pipesize=1)

    result = np.sum(data, axis=0)/len(repetition)
    Qk = np.reshape(result[1:], (36, 4))
    dataC = readoutFidCorr(Qk, [qC['measureF0'], qC['measureF1'],
                                qT['measureF0'], qT['measureF1']])
    if collect:
        return data, dataC
