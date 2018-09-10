from numpy import argmax, polyfit, polyval, eye, kron, abs, ones, sqrt, exp, conjugate, arange, angle, shape, reshape, trace, dot, floor, asarray, size, newaxis, transpose, zeros, pi, real,imag, outer, inner, nonzero, identity, diag, min, max, linspace, polyval, resize, sign, array, arctan, sum
from pylab import bar, iterable, imshow, xlabel, ylabel, xticks, yticks, ion, ioff, show, draw, clf, colorbar, gca, axes, figure, close, plot, subplot, title, log10, hold, arrow, xlim, ylim, legend
import numpy as np
from scipy.optimize import leastsq
from scipy.special import erf, erfc
import pylab as plt
import pylab
import numpy
import labrad
from labrad.units import ns, GHz, MHz, mV, V
import random

import pyle
from pyle.pipeline import returnValue, FutureList
from pyle.dataking import util
from pyle import envelopes as env
from pyle.dataking import envelopehelpers as eh
from pyle.dataking import measurement
from pyle.dataking import multiqubitPQ
from pyle.dataking import sweeps
from pyle.dataking.fpgaseq import runQubits
from pyle.util import sweeptools as st
from pyle.plotting import dstools
from pyle.dataking.multiqubitPQ import freqtuner, find_mpa, rabihigh
import pdb
#from fourierplot import fitswap
import time
from pyle import tomo
from pyle.dataking.multiqubitPQ import testdelay, pulseshape

from pyle.dataking import fpgaseq
fpgaseq.PREPAD = 350
#fpgaseq.PREPAD = 1100

# added on 2010.06.06 Sun
# txt files saved in U:\Matteo\Eclipse\datataking
import pyle.plotting.dstools as ds
import numpy as np
#import fourierplot1 as fp1
import labrad
# end added on 2010.06.06 Sun

extraDelay = 8*ns

def test(sample):
    npoints = 16
    a = 0.6 * np.exp(1.0j*np.linspace(0,2*np.pi,npoints,endpoint=False))
    b = 0.6 * np.exp(1.0j*np.linspace(0,2*np.pi,npoints,endpoint=False))
    result=noonTomo(sample,probeLen=st.r[0:300:2,ns],n=1,disp0=a,disp1=b,measure=[1,0],
                stats=1200)

    a = 0.7 * np.exp(1.0j*np.linspace(0,2*np.pi,npoints,endpoint=False))
    b = 0.7 * np.exp(1.0j*np.linspace(0,2*np.pi,npoints,endpoint=False))
    result=noonTomoSchemeA(sample,probeLen=st.r[0:300:2,ns],n=2,disp0=a,disp1=b,measure=[1,0],
               stats=1200)

    a = 0.5 * np.exp(1.0j*np.linspace(0,2*np.pi,npoints,endpoint=False))
    b = 0.5 * np.exp(1.0j*np.linspace(0,2*np.pi,npoints,endpoint=False))
    result=noonTomoSchemeA(sample,probeLen=st.r[0:300:2,ns],n=2,disp0=a,disp1=b,measure=[1,0],
               stats=1200)

    a = 0.9 * np.exp(1.0j*np.linspace(0,2*np.pi,npoints,endpoint=False))
    b = 0.9 * np.exp(1.0j*np.linspace(0,2*np.pi,npoints,endpoint=False))
    result=noonTomoSchemeA(sample,probeLen=st.r[0:300:2,ns],n=2,disp0=a,disp1=b,measure=[1,0],
               stats=1200)
      # swap1 = swap21(sample, measure=0, swapLen=st.r[0:250:4,ns], swapAmp=st.r[-0.3:0.5:0.008], collect=True)
      # swap0 = swap(sample, measure=0, swapLen=st.r[0:500:5,ns], swapAmp=st.r[0.1:0.6:0.008], collect=True)
     #------------------------------------------------------ return swap0, swap1

def complexSweep(displacement0, displacement1, sweepTime):
    return [[0,0,sT] for sT in sweepTime] + [[d0,d1,sT] for d0 in displacement0 for d1 in displacement1 for sT in sweepTime]

def datasetMinimum(data, default, llim, rlim, dataset=None):
    coeffs = np.polyfit(data[:,0],data[:,1],2)
    if coeffs[0] <= 0:
        print 'No minimum found, keeping value'
        return default, np.polyval(coeffs, default)
    result = np.clip(-0.5*coeffs[1]/coeffs[0],llim,rlim)
    return result, np.polyval(coeffs, result)

def swap10tuner(sample, swapLen=None, swapLenBND=1*ns, swapAmp=None, swapAmpBND=0.005, paraName='C',
                iteration=3, measure=0, stats=1200L,
         name='Q10-resonator swap tuner MQ', save=False, noisy=True, update=True):
    sample, qubits, Qubits= util.loadQubits(sample,write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]

    if swapAmp is None:
        swapAmp = q['noonSwapAmp'+paraName]
    if swapLen is None:
        swapLen = q['noonSwapLen'+paraName]

    for i in range(iteration):
        rf = 2**i
        swapLenOld = swapLen
        swapAmpOld = swapAmp

        print 'Tuning the swap amplitude'
        results = swap10(sample, swapLen=swapLen,
                        swapAmp=np.linspace(swapAmp-swapAmpBND/rf,swapAmp+swapAmpBND/rf,21),
                        measure=measure, stats=600L,
                        name='Q10-resonator swap MQ', save=save, collect=True, noisy=noisy)

        new, percent = datasetMinimum(results, swapAmpOld, swapAmpOld-swapAmpBND/rf, swapAmpOld+swapAmpBND/rf)
        swapAmp = new
        print 'New swap amplitude is %g' % swapAmp

        print 'Tuning the swap length'
        results = swap10(sample, swapLen=st.PQlinspace(max([swapLen['ns']*(1-0.2/rf),swapLen['ns']-swapLenBND['ns']]),
                                                       min([swapLen['ns']*(1+0.2/rf),swapLen['ns']+swapLenBND['ns']]),21,ns),
                        swapAmp=swapAmp, measure=measure, stats=600L,
                        name='Q10-resonator swap MQ', save=save, collect=True, noisy=noisy)

        new, percent = datasetMinimum(results, swapLenOld['ns'], max([swapLen['ns']*(1-0.2/rf),swapLen['ns']-swapLenBND['ns']]),
                                      min([swapLen['ns']*(1+0.2/rf),swapLen['ns']+swapLenBND['ns']]))
        swapLen = new*ns
        print 'New swap length is %g ns' % swapLen['ns']

    if update:
        Q['noonSwapAmp'+paraName] = swapAmp
        Q['noonSwapLen'+paraName] = swapLen

    return swapLen, swapAmp

def swap10(sample, swapLen=st.arangePQ(0,200,4,ns), swapAmp=np.arange(-0.05,0.05,0.002), measure=0, stats=600L,
         name='Q10-resonator swap MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    if swapAmp is None:
        swapAmp = q.swapAmp

    axes = [(swapAmp, 'swap pulse amplitude'), (swapLen, 'swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, currAmp, currLen):
        q.xy = eh.mix(q, eh.piPulseHD(q, 0))
        q.z = env.rect(q['piLen']/2, currLen, currAmp) + eh.measurePulse(q, q['piLen']/2 + currLen)
        q['readout'] = True
        return runQubits(server, qubits, stats=stats, probs=[1])

    return sweeps.grid(func, axes, save=save, dataset=dataset, collect=collect, noisy=noisy)

def swap10night(sample):

    swap10(sample, swapLen=st.arangePQ(0,505,1,ns), swapAmp=np.arange(-0.30,0.35,0.001), measure=0, save=True)
    swap10(sample, swapLen=st.arangePQ(0,505,1,ns), swapAmp=np.arange(-0.40,0.25,0.001), measure=1, save=True)

def ramseySpec(sample, delay=st.r[0:500:1,ns], swapAmp=np.arange(-0.05,0.05,0.002), phase=0, df=50*MHz, measure=0, stats=600L,
               name='Ramsey spectroscopy 2D MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    if swapAmp is None:
        swapAmp = q.swapAmp

    axes = [(swapAmp, 'swap pulse amplitude'), (delay, 'Ramsey delay time')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, currAmp, currLen):
        dt = q['piLen']
        tp = dt/2.0 + currLen + dt/2.0
        tm = tp + dt/2.0

        ph = phase - 2*np.pi*df[GHz]*currLen[ns]

        q.z = env.rect(dt/2.0, tp-dt, currAmp)

        q.xy = eh.mix(q, eh.piHalfPulse(q, 0) + eh.piHalfPulse(q, tp, phase=ph))

        q.z += eh.measurePulse(q, tm)

        q['readout'] = True

        return runQubits(server, qubits, stats=stats, probs=[1])

    return sweeps.grid(func, axes, save=save, dataset=dataset, collect=collect, noisy=noisy)
    # return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

def ramseySpecNight(sample):

    ramseySpec(sample, delay=st.r[0:505:1,ns], swapAmp=np.arange(-0.10,0.57,0.001), measure=0, save=True)
    ramseySpec(sample, delay=st.r[0:505:1,ns], swapAmp=np.arange(-0.40,0.25,0.001), measure=1, save=True)

def swap10resetTunerEZ(sample, swapLen=10.0*ns, swapLenBND=1*ns, swapAmp=-0.036, swapAmpBND=0.005, paraName='0',
                       iteration=3, measure=0, stats=1200L,
                       name='Q10-reset env swap tuner MQ', save=False, noisy=True, update=False):
    sample, qubits, Qubits= util.loadQubits(sample,write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]

    for i in range(iteration):
        rf = 2**i
        swapLenOld = swapLen
        swapAmpOld = swapAmp

        print 'Tuning the swap amplitude'
        results = swap10(sample, swapLen=swapLen,
                         swapAmp=np.linspace(swapAmp-swapAmpBND/rf,swapAmp+swapAmpBND/rf,21),
                         measure=measure, stats=600L,
                         name='Q10-reset env swap tuner MQ', save=save, collect=True, noisy=noisy)

        new, percent = datasetMinimum(results, swapAmpOld, swapAmpOld-swapAmpBND/rf, swapAmpOld+swapAmpBND/rf)
        swapAmp = new
        print 'New swap amplitude is %g' % swapAmp

        print 'Tuning the swap length'
        results = swap10(sample, swapLen=st.PQlinspace(max([swapLen['ns']*(1-0.2/rf),swapLen['ns']-swapLenBND['ns']]),
                                                       min([swapLen['ns']*(1+0.2/rf),swapLen['ns']+swapLenBND['ns']]),21,ns),
                        swapAmp=swapAmp, measure=measure, stats=600L,
                        name='Q10-reset env swap tuner MQ', save=save, collect=True, noisy=noisy)

        new, percent = datasetMinimum(results, swapLenOld['ns'], max([swapLen['ns']*(1-0.2/rf),swapLen['ns']-swapLenBND['ns']]),
                                      min([swapLen['ns']*(1+0.2/rf),swapLen['ns']+swapLenBND['ns']]))
        swapLen = new*ns
        print 'New swap length is %g ns' % swapLen['ns']

    return swapLen, swapAmp

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

#    if tuneOS is False:
#        so = np.array([0.0]*n)

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
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)

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

#    if tuneOS is False:
#        so = np.array([0.0]*n)

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
#            q.z += env.rect(start, sl, sa, overshoot=so[n-1]+currOS)
#            start += sl+delay
            q.z += env.rect(start, currLen, sa)
            start += currLen+delay
            q.z += eh.measurePulse(q, start)

        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)

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
        new, percent = datasetMinimum(results, 0, -max([0.3*sl['ns']/rf,1]), max([0.3*sl['ns']/rf,1]))
        sample['q'+str(measure)]['resetLens'+paraName][resetPoint] += new

        if save:
            FockScanReset(sample, n=n, scanLen=st.arangePQ(0,100,2,'ns'),
                          paraName=paraName, stats=stats, measure=measure, probeFlag=True, delay=delay, resetPoint=resetPoint,
                          save=save, collect=collect, noisy=noisy)

#    if update:
#        Q['resetLens'+paraName][resetPoint] = sample['q'+str(measure)]['resetLens'+paraName][resetPoint]

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
        new, percent = datasetMinimum(results, 0, -max([0.3*sl['ns']/rf,1]), max([0.3*sl['ns']/rf,1]))
        sample['q'+str(measure)]['resetLens'+paraName][resetPoint] += new

        if save:
            FockScanResetCounter(sample, n=n, scanLen=st.arangePQ(0,100,2,'ns'),
                                 paraName=paraName, stats=stats, measure=measure, probeFlag=True, delay=delay, resetPoint=resetPoint,
                                 save=save, collect=collect, noisy=noisy)

#    if update:
#        Q['resetLens'+paraName][resetPoint] = sample['q'+str(measure)]['resetLens'+paraName][resetPoint]

    return sample['q'+str(measure)]['resetLens'+paraName][resetPoint]

def swap21tuner(sample, swapLen=None, swapLenBND=1*ns, swapAmp=None, swapAmpBND=0.005,
                paraName='1', iteration=3, measure=0, stats=1200L, update=True,
         name='Q21-resonator swap tuner MQ', save=False, noisy=True):
    sample, qubits, Qubits= util.loadQubits(sample,write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]

    if swapAmp is None:
        swapAmp = q['noonSwapAmp'+paraName]
    if swapLen is None:
        swapLen = q['noonSwapLen'+paraName]

    for i in range(iteration):
        rf = 2**i
        swapLenOld = swapLen
        swapAmpOld = swapAmp

        print 'Tuning the swap amplitude'
        results = swap21(sample, swapLen=swapLen,
                        swapAmp=np.linspace(swapAmp-swapAmpBND,swapAmp+swapAmpBND,21),
                        measure=measure, stats=600L,
                        name='Q21-resonator swap MQ', save=save, collect=True, noisy=noisy)

        new, percent = datasetMinimum(results, swapAmpOld, swapAmpOld-swapAmpBND/rf, swapAmpOld+swapAmpBND/rf)
        swapAmp = new
        print 'New swap amplitude is %g' % swapAmp

        print 'Tuning the swap length'
        results = swap21(sample, swapLen=st.PQlinspace(max([swapLen['ns']*(1-0.2/rf),swapLen['ns']-swapLenBND['ns']]),
                                                       min([swapLen['ns']*(1+0.2/rf),swapLen['ns']+swapLenBND['ns']]),21,ns),
                        swapAmp=swapAmp, measure=measure, stats=600L,
                        name='Q21-resonator swap MQ', save=save, collect=True, noisy=noisy)

        new, percent = datasetMinimum(results, swapLenOld['ns'], max([swapLen['ns']*(1-0.2/rf),swapLen['ns']-swapLenBND['ns']]),
                                      min([swapLen['ns']*(1+0.2/rf),swapLen['ns']+swapLenBND['ns']]))

        swapLen = new*ns
        print 'New swap length is %g ns' % swapLen['ns']

    if update:
        Q['noonSwapAmp'+paraName] = swapAmp
        Q['noonSwapLen'+paraName] = swapLen

    return swapLen, swapAmp

def swap21(sample, swapLen=st.arangePQ(0,200,4,ns), swapAmp=np.arange(-0.05,0.05,0.002), measure=0, stats=600L,
         name='Q21-resonator swap MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    axes = [(swapAmp, 'swap pulse amplitude'), (swapLen, 'swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, currAmp, currLen):
        q.xy = eh.mix(q, eh.piPulseHD(q, 0))+eh.mix(q, env.gaussian(q.piLen, q.piFWHM, q.piAmp21, df=q.piDf21), freq = 'f21')
        q.z = env.rect(q['piLen']*1.5, currLen, currAmp) + eh.measurePulse2(q, q['piLen']*1.5 + currLen)
        q['readout'] = True
        return runQubits(server, qubits, stats=stats, probs=[1])

    return sweeps.grid(func, axes, save=save, dataset=dataset, collect=collect, noisy=noisy)

def pituner10(sample, measure=0, iterations=2, npoints=21, stats=1200, save=False, update=True, noisy=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    amp = q['piAmp']
    for _ in xrange(iterations):
        # optimize amplitude

        data = rabihigh10(sample, amplitude=np.linspace(0.6*amp, 1.4*amp, npoints),
                           measure=measure, stats=stats, collect=True, noisy=noisy)
        amp_fit = np.polyfit(data[:,0], data[:,1], 2)
        amp = -0.5 * amp_fit[1] / amp_fit[0]
        print 'Amplitude: %g' % amp

        freq = freqtuner(sample, iterations=1, tEnd=100*ns, timeRes=1*ns, nfftpoints=4000, stats=1200, df=50*MHz,
              measure=measure, save=False, plot=False, noisy=noisy)
        sample['q'+str(measure)].f10 = freq
    # save updated values
    if update:
        Q.piAmp = amp
        Q.f10 = freq
    return amp

def rabihigh10(sample, amplitude=st.r[0.0:1.5:0.05], measureDelay=None, measure=0, stats=1500L,
                name='Rabi-pulse height MQ', save=False, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    if measureDelay is None: measureDelay = q['piLen'] /2.0

    axes = [(amplitude, 'pulse height'),
            (measureDelay, 'measure delay')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, amp, delay):
        q['piAmp'] = amp
        q.xy = eh.mix(q, eh.piPulseHD(q, 0))
        q.z = eh.measurePulse(q, delay)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)

def pituner21(sample, measure=0, iterations=2, npoints=21, stats=1500L, save=False, update=True, noisy=True, findMPA=False):

    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    amp = q.piAmp21
    df = q.piDf21['MHz']
    if findMPA:
        Q.measureAmp2 = find_mpa(sample, stats=60, target=0.05, mpa_range=(-2.0, 2.0), pi_pulse=True,
                 measure=measure, pulseFunc=None, resolution=0.005, blowup=0.05,
                 falling=None, statsinc=1.25,
                 save=False, name='SCurve Search MQ', collect=True, update=True, noisy=True)
    for _ in xrange(iterations):
        # optimize amplitude
        data = rabihigh21(sample, amplitude=np.linspace(0.75*amp, 1.25*amp, npoints), detuning=df*MHz,
                        measure=measure, stats=stats, collect=True, noisy=noisy, save=save)
        amp_fit = np.polyfit(data[:,0], data[:,1], 2)
        amp = -0.5 * amp_fit[1] / amp_fit[0]
        print 'Amplitude for 1->2 transition: %g' % amp
        # optimize detuning
        data = rabihigh21(sample, amplitude=amp, detuning=st.PQlinspace(df-20, df+20, npoints, MHz),
                        measure=measure, stats=stats, collect=True, noisy=noisy, save=save)
        df_fit = np.polyfit(data[:,0], data[:,1], 2)
        Delta_df = -0.5 * df_fit[1] / df_fit[0]-df
        if np.abs(Delta_df)>20:
            df += np.sign(Delta_df)*20
        else:
            df += Delta_df
        print 'Detuning frequency for 1->2 transition: %g MHz' % df
    # save updated values
    if update:
        Q['piAmp21'] = amp
        Q['piDf21'] = df*MHz
    return amp, df*MHz

def rabihigh21(sample, amplitude=st.r[0.0:1.5:0.05], detuning=0*MHz, measure=0, stats=1500L,
                name='Rabi-pulse 12 MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    axes = [(amplitude, 'pulse height'),
            (detuning, 'frequency detuning')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, amp, df):
        q.xy = eh.mix(q, eh.piPulseHD(q, 0)) + eh.mix(q, env.gaussian(q.piLen, q.piFWHM, amp, df=df), freq = 'f21')
        q.z = eh.measurePulse2(q, q.piLen*1.5)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)

def T1(sample, delay=st.arangePQ(-10,20,1,ns)+st.arangePQ(20,100,2,ns)+st.arangePQ(100,500,4,ns)+st.arangePQ(500,1500,10,ns), stats=600L, measure=0,
       name='T1 level1 MQ', save=True, collect=True, noisy=True):
    """A single pi-pulse on one qubit, with other qubits also operated."""
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    axes = [(delay, 'Measure pulse delay')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, delay):
        q.xy = eh.mix(q, eh.piPulseHD(q, 0))
        q.z = eh.measurePulse(q, delay)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

def T1zPulse(sample, delay=st.arangePQ(-10,20,1,ns)+st.arangePQ(20,100,2,ns)+st.arangePQ(100,500,4,ns)+st.arangePQ(500,1500,10,ns),
             zpa=0, stats=600L, measure=0,
       name='T1 level1 zPulse MQ', save=True, collect=True, noisy=True):
    """A single pi-pulse on one qubit, with other qubits also operated."""
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    axes = [(delay, 'Measure pulse delay')]
    kw = {'stats': stats,
          'zpa': zpa}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, delay):
        q.xy = eh.mix(q, eh.piPulseHD(q, -q.piLen/2))
        q.z = env.rect(0, delay, zpa)
        q.z += eh.measurePulse(q, delay)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

def T1L2(sample, delay=st.arangePQ(-10,100,2,ns)+st.arangePQ(100,500,4,ns)+st.arangePQ(500,750,10,ns), stats=600L, measure=0,
       name='T1 level2 MQ', save=True, collect=True, noisy=True):
    """A single pi-pulse on one qubit, with other qubits also operated."""
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    axes = [(delay, 'Measure pulse delay')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, delay):
        q.xy = eh.mix(q, eh.piPulseHD(q, -q.piLen))+eh.mix(q, env.gaussian(0, q.piFWHM, q.piAmp21, df = q.piDf21), freq = 'f21')
        q.z = eh.measurePulse2(q, delay)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)


def resonatorT1(sample, delay=st.arangePQ(0,1,0.01,'us')+st.arangePQ(1,8,0.1,'us'),paraName='C',stats=1200L, measure=0,
       name='resonator T1 MQ', save=True, collect=True, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    axes = [(delay, 'Measure pulse delay')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, paraName+' '+name, axes, measure=measure, kw=kw)

    sl = q['noonSwapLen'+paraName]
    sa = q['noonSwapAmp'+paraName]

    def func(server, delay):
        q.xy = eh.mix(q, eh.piPulseHD(q, 0))
        q.z = env.rect(q.piLen/2, sl, sa)
        q.z += env.rect(q.piLen/2+sl+delay, sl, sa)
        q.z += eh.measurePulse(q, q.piLen/2+sl+delay+sl)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

def iSwap(sample, swapLen=st.r[-20:1000:1,ns], measure=[0,1], stats=1500L,
         name='iSwap MQ', save=True, collect=False, noisy=True, delay=0*ns):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]
    nameEx = [' q0->q1',' q1->q0']
    axes = [(swapLen, 'swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx[measure[0]], axes, measure=measure, kw=kw)

    def func(server, curr):
        start = 0
        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2+delay
        q0.z = env.rect(start, curr, q0.noonSwapAmpC)
        start += curr+delay
        q1.z = env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)
        start += q1.noonSwapLenC+delay
        q0.z += eh.measurePulse(q0, start)
        q1.z += eh.measurePulse(q1, start)

        q0['readout'] = True
        q1['readout'] = True
        return runQubits(server, qubits, stats=stats)
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    resultC = readoutFidCorr(result, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])
    color = ('b','r','g','c')
    pylab.figure(98)
    pylab.clf()
    for i in np.arange(0,4,1):
        pylab.plot(result[:,0],result[:,i+1],color[i]+'.')
        pylab.plot(resultC[:,0],resultC[:,i+1],color[i]+'-')
        pylab.hold('on')


def bellStateTomo(sample, repetition=10, measure=[0,1], stats=1500L, delay=0*ns,
         name='Bell state MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    measurement = pyle.dataking.measurement.Tomo(2)

    nameEx = [' q0->q1',' q1->q0']
    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx[measure[0]], axes, measure=measurement, kw=kw)

    def func(server, curr):

        start = 0
        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2+delay
        q0.z = env.rect(start, q0.noonSwapLenCSQRT, q0.noonSwapAmpC)
        start += q0.noonSwapLenCSQRT+delay
        q1.z = env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)
        start += q1.noonSwapLenC+delay
        return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)
    Qk = np.reshape(result[1:],(9,4))
    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    rho_cal = tomo.qst(Qk,'tomo2')
    plotRhoSingle(rho_cal,figNo=100)
    pylab.title('Exp.')
    rho_caln = rho_cal.copy()
    rho_caln[1,2] = abs(rho_caln[1,2])
    rho_caln[2,1] = abs(rho_caln[2,1])
    print 'Fidelity is %g ' % np.trace(np.dot(rho_caln,rho_ideal))

    Us =tomo._qst_transforms['tomo2'][0]
    rho_calLike = tomo.qst_mle(Qk,Us,rho0 = rho_ideal)
    rho_calLiken = rho_calLike.copy()
    rho_calLiken[1,2] = abs(rho_calLiken[1,2])
    rho_calLiken[2,1] = abs(rho_calLiken[2,1])
    plotRhoSingle(rho_calLike,figNo=101)
    pylab.title('Exp. likely')
    print 'Fidelity for likelyhood is %g ' % np.trace(np.dot(rho_calLiken,rho_ideal))
    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal)), rho_calLike, np.trace(np.dot(rho_calLiken,rho_ideal))

def bellStateTomoQStorage(sample, repetition=10, measure=[0,1], stats=1500L, delay=0*ns, storage=250.0*ns,
                          name='Bell state qubit storage MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    measurement = pyle.dataking.measurement.Tomo(2)

    nameEx = [' q0->q1',' q1->q0']
    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx[measure[0]], axes, measure=measurement, kw=kw)

    def func(server, curr):
        start = 0

        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2+delay

        q0.z = env.rect(start, q0.noonSwapLenCSQRT, q0.noonSwapAmpC)
        start += q0.noonSwapLenCSQRT+delay

        q1.z = env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)
        start += q1.noonSwapLenC+delay+storage

        return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)
    Qk = np.reshape(result[1:],(9,4))
    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    rho_cal = tomo.qst(Qk,'tomo2')
    plotRhoSingle(rho_cal,figNo=100)
    pylab.title('Exp.')
    rho_caln = rho_cal.copy()
    rho_caln[1,2] = abs(rho_caln[1,2])
    rho_caln[2,1] = abs(rho_caln[2,1])
    print 'Fidelity is %g ' % np.trace(np.dot(rho_caln,rho_ideal))

    Us =tomo._qst_transforms['tomo2'][0]
    rho_calLike = tomo.qst_mle(Qk,Us,rho0 = rho_ideal)
    rho_calLiken = rho_calLike.copy()
    rho_calLiken[1,2] = abs(rho_calLiken[1,2])
    rho_calLiken[2,1] = abs(rho_calLiken[2,1])
    plotRhoSingle(rho_calLike,figNo=101)
    pylab.title('Exp. likely')
    print 'Fidelity for likelyhood is %g ' % np.trace(np.dot(rho_calLiken,rho_ideal))
    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal)), rho_calLike, np.trace(np.dot(rho_calLiken,rho_ideal))

def bellStateMidTransTomo(sample, repetition=10, measure=[0,1], stats=1500L, delay=0*ns,
         name='Bell state mid transfer MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    measurement = pyle.dataking.measurement.Tomo(2)

    nameEx = [' q0->q1',' q1->q0']
    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx[measure[0]], axes, measure=measurement, kw=kw)

    def func(server, curr):

        start = 0
        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2+delay
        q0.z = env.rect(start, q0.noonSwapLenCSQRT, q0.noonSwapAmpC)
        start += q0.noonSwapLenCSQRT+delay
        q1.z = env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)
        start += q1.noonSwapLenC+delay
        q0.z += env.rect(start, q0.noonSwapLen0, q0.noonSwapAmp0)
        q1.z += env.rect(start, q1.noonSwapLen0, q1.noonSwapAmp0)
        start += max([q0.noonSwapLen0,q1.noonSwapLen0])
        return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)
    Qk = np.reshape(result[1:],(9,4))
    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    rho_cal = tomo.qst(Qk,'tomo2')
    plotRhoSingle(rho_cal,figNo=100)
    pylab.title('Exp.')
    rho_caln = rho_cal.copy()
    rho_caln[1,2] = abs(rho_caln[1,2])
    rho_caln[2,1] = abs(rho_caln[2,1])
    print 'Fidelity is %g ' % np.trace(np.dot(rho_caln,rho_ideal))

    Us =tomo._qst_transforms['tomo2'][0]
    rho_calLike = tomo.qst_mle(Qk,Us,rho0 = rho_ideal)
    rho_calLiken = rho_calLike.copy()
    rho_calLiken[1,2] = abs(rho_calLiken[1,2])
    rho_calLiken[2,1] = abs(rho_calLiken[2,1])
    plotRhoSingle(rho_calLike,figNo=101)
    pylab.title('Exp. likely')
    print 'Fidelity for likelyhood is %g ' % np.trace(np.dot(rho_calLiken,rho_ideal))
    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal)), rho_calLike, np.trace(np.dot(rho_calLiken,rho_ideal))

def bellStatePostTransTomo(sample, repetition=10, measure=[0,1], stats=1500L, delay=0*ns,
         name='Bell state post transfer MQ', save=True, collect=False, noisy=True, extraDelay=extraDelay):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    measurement = pyle.dataking.measurement.Tomo(2)

    nameEx = [' q0->q1',' q1->q0']
    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx[measure[0]], axes, measure=measurement, kw=kw)

    def func(server, curr):

        start = 0
        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2+delay
        q0.z = env.rect(start, q0.noonSwapLenCSQRT, q0.noonSwapAmpC)
        start += q0.noonSwapLenCSQRT+delay
        q1.z = env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)
        start += q1.noonSwapLenC+delay
        q0.z += env.rect(start, q0.noonSwapLen0, q0.noonSwapAmp0)
        q1.z += env.rect(start, q1.noonSwapLen0, q1.noonSwapAmp0)
        start += max([q0.noonSwapLen0,q1.noonSwapLen0])+extraDelay
        q0.z += env.rect(start, q0.noonSwapLen0, q0.noonSwapAmp0)
        q1.z += env.rect(start, q1.noonSwapLen0, q1.noonSwapAmp0)
        start += max([q0.noonSwapLen0,q1.noonSwapLen0])
        return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)
    Qk = np.reshape(result[1:],(9,4))
    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    rho_cal = tomo.qst(Qk,'tomo2')
    plotRhoSingle(rho_cal,figNo=100)
    pylab.title('Exp.')
    rho_caln = rho_cal.copy()
    rho_caln[1,2] = abs(rho_caln[1,2])
    rho_caln[2,1] = abs(rho_caln[2,1])
    print 'Fidelity is %g ' % np.trace(np.dot(rho_caln,rho_ideal))

    Us =tomo._qst_transforms['tomo2'][0]
    rho_calLike = tomo.qst_mle(Qk,Us,rho0 = rho_ideal)
    rho_calLiken = rho_calLike.copy()
    rho_calLiken[1,2] = abs(rho_calLiken[1,2])
    rho_calLiken[2,1] = abs(rho_calLiken[2,1])
    plotRhoSingle(rho_calLike,figNo=101)
    pylab.title('Exp. likely')
    print 'Fidelity for likelyhood is %g ' % np.trace(np.dot(rho_calLiken,rho_ideal))
    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal)), rho_calLike, np.trace(np.dot(rho_calLiken,rho_ideal))

def iSwap2(sample, swapLen=st.r[-20:1000:1,ns], measure=[0,1], stats=1500L,
           name='iSwap 2 MQ', save=True, collect=False, noisy=True, delay=0*ns):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]
    nameEx = [' q0->q1',' q1->q0']
    axes = [(swapLen, 'swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx[measure[0]], axes, measure=measure, kw=kw)

    def func(server, curr):

        start = 0

        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2+delay

        q0.z = env.rect(start, q0.noonSwapLenCSQRT, q0.noonSwapAmpC)
        start += q0.noonSwapLenCSQRT+delay
        q1.z = env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)
        start += q1.noonSwapLenC+delay
        # Bell state 1

        q0.z += env.rect(start, q0.noonSwapLen0, q0.noonSwapAmp0)
        q1.z += env.rect(start, q1.noonSwapLen0, q1.noonSwapAmp0)
        start += max([q0.noonSwapLen0,q1.noonSwapLen0])+17.0*ns
        # write Bell state 1 into NOON resonators

        q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2+delay

        q0.z += env.rect(start, curr, q0.noonSwapAmpC)
        start += curr+delay
        q1.z += env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)
        start += q1.noonSwapLenC+delay

        q0.z += eh.measurePulse(q0, start)
        q1.z += eh.measurePulse(q1, start)

        q0['readout'] = True
        q1['readout'] = True
        return runQubits(server, qubits, stats=stats)
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    resultC = readoutFidCorr(result, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])
    color = ('b','r','g','c')
    pylab.figure(98)
    pylab.clf()
    for i in np.arange(0,4,1):
        pylab.plot(result[:,0],result[:,i+1],color[i]+'.')
        pylab.plot(resultC[:,0],resultC[:,i+1],color[i]+'-')
        pylab.hold('on')

def bellState1TransQubit(sample, swapLen=st.r[0:500:1,ns], measure=[0,1], stats=1500L, delay=0*ns,
                         name='Bell state 1 transfer qubit MQ', save=True, collect=False, noisy=True, extraDelay=extraDelay):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    nameEx = [' q0->q1',' q1->q0']
    axes = [(swapLen, 'swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx[measure[0]], axes, measure=measure, kw=kw)

    def func(server, curr):

        start = 0

        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2+delay

        q0.z = env.rect(start, q0.noonSwapLenCSQRT, q0.noonSwapAmpC)
        start += q0.noonSwapLenCSQRT+delay
        q1.z = env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)
        start += q1.noonSwapLenC+delay
        # Bell state 1

        q0.z += env.rect(start, q0.noonSwapLen0, q0.noonSwapAmp0)
        q1.z += env.rect(start, q1.noonSwapLen0, q1.noonSwapAmp0)
        start += max([q0.noonSwapLen0,q1.noonSwapLen0])+extraDelay
        # write Bell state 1 into NOON resonators

        start += curr
        q0.z += eh.measurePulse(q0, start)
        q1.z += eh.measurePulse(q1, start)

        q0['readout'] = True
        q1['readout'] = True
        return runQubits(server, qubits, stats=stats)
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

def bellState1TransQubitTomo(sample, repetition=10, measure=[0,1], stats=1500L, delay=0*ns,
                             name='Bell state 1 transfer qubit TOMO MQ', save=True, collect=False, noisy=True, extraDelay=extraDelay):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    measurement = pyle.dataking.measurement.Tomo(2)

    nameEx = [' q0->q1',' q1->q0']
    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx[measure[0]], axes, measure=measurement, kw=kw)

    def func(server, curr):

        start = 0

        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2+delay

        q0.z = env.rect(start, q0.noonSwapLenCSQRT, q0.noonSwapAmpC)
        start += q0.noonSwapLenCSQRT+delay
        q1.z = env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)
        start += q1.noonSwapLenC+delay
        # +4.0*ns
        # Bell state 1

        q0.z += env.rect(start, q0.noonSwapLen0, q0.noonSwapAmp0)
        q1.z += env.rect(start, q1.noonSwapLen0, q1.noonSwapAmp0)
        start += max([q0.noonSwapLen0,q1.noonSwapLen0])+extraDelay
        # write Bell state 1 into NOON resonators

        return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)
    Qk = np.reshape(result[1:],(9,4))
    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    rho_cal = tomo.qst(Qk,'tomo2')
    plotRhoSingle(rho_cal,figNo=100)
    pylab.title('Exp.')
    rho_caln = rho_cal.copy()
    rho_caln[1,2] = abs(rho_caln[1,2])
    rho_caln[2,1] = abs(rho_caln[2,1])
    print 'Fidelity is %g ' % np.trace(np.dot(rho_caln,rho_ideal))

    Us =tomo._qst_transforms['tomo2'][0]
    rho_calLike = tomo.qst_mle(Qk,Us,rho0 = rho_ideal)
    rho_calLiken = rho_calLike.copy()
    rho_calLiken[1,2] = abs(rho_calLiken[1,2])
    rho_calLiken[2,1] = abs(rho_calLiken[2,1])
    plotRhoSingle(rho_calLike,figNo=101)
    pylab.title('Exp. likely')
    print 'Fidelity for likelyhood is %g ' % np.trace(np.dot(rho_calLiken,rho_ideal))
    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal)), rho_calLike, np.trace(np.dot(rho_calLiken,rho_ideal))

def bellState2Tomo(sample, repetition=10, measure=[0,1], stats=1500L, delay=0*ns,
                   name='Bell state 2 MQ', save=True, collect=False, noisy=True, extraDelay=extraDelay):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    measurement = pyle.dataking.measurement.Tomo(2)

    nameEx = [' q0->q1',' q1->q0']
    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx[measure[0]], axes, measure=measurement, kw=kw)

    def func(server, curr):

        start = 0

        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2+delay

        q0.z = env.rect(start, q0.noonSwapLenCSQRT, q0.noonSwapAmpC)
        start += q0.noonSwapLenCSQRT+delay
        q1.z = env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)
        start += q1.noonSwapLenC+delay
        # Bell state 1

        q0.z += env.rect(start, q0.noonSwapLen0, q0.noonSwapAmp0)
        q1.z += env.rect(start, q1.noonSwapLen0, q1.noonSwapAmp0)
        start += max([q0.noonSwapLen0,q1.noonSwapLen0])+extraDelay
        # write Bells state 1 into NOON resonators

        q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2+delay

        q0.z += env.rect(start, q0.noonSwapLenCSQRT, q0.noonSwapAmpC)
        start += q0.noonSwapLenCSQRT+delay
        q1.z += env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)
        start += q1.noonSwapLenC+delay+extraDelay
        # Bell state 2

        return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)
    Qk = np.reshape(result[1:],(9,4))
    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    rho_cal = tomo.qst(Qk,'tomo2')
    plotRhoSingle(rho_cal,figNo=100)
    pylab.title('Exp.')
    rho_caln = rho_cal.copy()
    rho_caln[1,2] = abs(rho_caln[1,2])
    rho_caln[2,1] = abs(rho_caln[2,1])
    print 'Fidelity is %g ' % np.trace(np.dot(rho_caln,rho_ideal))

    Us =tomo._qst_transforms['tomo2'][0]
    rho_calLike = tomo.qst_mle(Qk,Us,rho0 = rho_ideal)
    rho_calLiken = rho_calLike.copy()
    rho_calLiken[1,2] = abs(rho_calLiken[1,2])
    rho_calLiken[2,1] = abs(rho_calLiken[2,1])
    plotRhoSingle(rho_calLike,figNo=101)
    pylab.title('Exp. likely')
    print 'Fidelity for likelyhood is %g ' % np.trace(np.dot(rho_calLiken,rho_ideal))
    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal)), rho_calLike, np.trace(np.dot(rho_calLiken,rho_ideal))

def bellState2Reset(sample, swapLen=st.r[0:500:1,ns], measure=[1,0], stats=1500L, delay=0*ns,
                    name='Bell state 2 reset MQ', save=True, collect=False, noisy=True, extraDelay=extraDelay):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    nameEx = [' q0->q1',' q1->q0']
    axes = [(swapLen, 'swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx[measure[0]], axes, measure=measure, kw=kw)

    def func(server, curr):

        start = 0

        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2+delay

        q0.z = env.rect(start, q0.noonSwapLenCSQRT, q0.noonSwapAmpC)
        start += q0.noonSwapLenCSQRT+delay
        q1.z = env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)
        start += q1.noonSwapLenC+delay
        # Bell state 1

        q0.z += env.rect(start, q0.noonSwapLen0, q0.noonSwapAmp0)
        q1.z += env.rect(start, q1.noonSwapLen0, q1.noonSwapAmp0)
        start += max([q0.noonSwapLen0,q1.noonSwapLen0])+extraDelay
        # write Bells state 1 into NOON resonators

        q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2+delay

        q0.z += env.rect(start, q0.noonSwapLenCSQRT, q0.noonSwapAmpC)
        start += q0.noonSwapLenCSQRT+delay
        q1.z += env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)
        start += q1.noonSwapLenC+delay+extraDelay
        # Bell state 2

        q0.z += env.rect(start, q0.resetLens[0], q0.resetAmps[0])
        q1.z += env.rect(start, q1.resetLens[2], q1.resetAmps[2])
        start += max([q0.resetLens[0],q1.resetLens[2]])+extraDelay

        start += curr
        q0.z += eh.measurePulse(q0, start)
        q1.z += eh.measurePulse(q1, start)

        q0['readout'] = True
        q1['readout'] = True
        return runQubits(server, qubits, stats=stats)
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

def bellState2ResetTomo(sample, repetition=10, measure=[1,0], stats=1500L, delay=0*ns,
                        name='Bell state 2 MQ', save=True, collect=False, noisy=True, extraDelay=extraDelay):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    measurement = pyle.dataking.measurement.Tomo(2)

    nameEx = [' q0->q1',' q1->q0']
    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx[measure[0]], axes, measure=measurement, kw=kw)

    def func(server, curr):

        start = 0

        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2+delay

        q0.z = env.rect(start, q0.noonSwapLenCSQRT, q0.noonSwapAmpC)
        start += q0.noonSwapLenCSQRT+delay
        q1.z = env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)
        start += q1.noonSwapLenC+delay
        # Bell state 1

        q0.z += env.rect(start, q0.noonSwapLen0, q0.noonSwapAmp0)
        q1.z += env.rect(start, q1.noonSwapLen0, q1.noonSwapAmp0)
        start += max([q0.noonSwapLen0,q1.noonSwapLen0])+extraDelay
        # write Bells state 1 into NOON resonators

        q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2+delay

        q0.z += env.rect(start, q0.noonSwapLenCSQRT, q0.noonSwapAmpC)
        start += q0.noonSwapLenCSQRT+delay
        q1.z += env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)
        start += q1.noonSwapLenC+delay+extraDelay
        # Bell state 2

        q0.z += env.rect(start, q0.resetLens[0], q0.resetAmps[0])
        q1.z += env.rect(start, q1.resetLens[2], q1.resetAmps[2])
        start += max([q0.resetLens[0],q1.resetLens[2]])+extraDelay

        return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)
    Qk = np.reshape(result[1:],(9,4))
    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    rho_cal = tomo.qst(Qk,'tomo2')
    plotRhoSingle(rho_cal,figNo=100)
    pylab.title('Exp.')
    rho_caln = rho_cal.copy()
    rho_caln[1,2] = abs(rho_caln[1,2])
    rho_caln[2,1] = abs(rho_caln[2,1])
    print 'Fidelity is %g ' % np.trace(np.dot(rho_caln,rho_ideal))

    Us =tomo._qst_transforms['tomo2'][0]
    rho_calLike = tomo.qst_mle(Qk,Us,rho0 = rho_ideal)
    rho_calLiken = rho_calLike.copy()
    rho_calLiken[1,2] = abs(rho_calLiken[1,2])
    rho_calLiken[2,1] = abs(rho_calLiken[2,1])
    plotRhoSingle(rho_calLike,figNo=101)
    pylab.title('Exp. likely')
    print 'Fidelity for likelyhood is %g ' % np.trace(np.dot(rho_calLiken,rho_ideal))
    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal)), rho_calLike, np.trace(np.dot(rho_calLiken,rho_ideal))

def Bell1PostResetBell2(sample, swapLen=st.r[0:500:1,ns], measure=[1,0], stats=1500L, delay=0*ns,
                        name='Bell 1 post-reset Bell 2 MQ', save=True, collect=False, noisy=True, extraDelay=extraDelay):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    nameEx = [' q0->q1',' q1->q0']
    axes = [(swapLen, 'swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx[measure[0]], axes, measure=measure, kw=kw)

    def func(server, curr):

        start = 0

        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2+delay

        q0.z = env.rect(start, q0.noonSwapLenCSQRT, q0.noonSwapAmpC)
        start += q0.noonSwapLenCSQRT+delay
        q1.z = env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)
        start += q1.noonSwapLenC+delay
        # Bell state 1

        q0.z += env.rect(start, q0.noonSwapLen0, q0.noonSwapAmp0)
        q1.z += env.rect(start, q1.noonSwapLen0, q1.noonSwapAmp0)
        start += max([q0.noonSwapLen0,q1.noonSwapLen0])+extraDelay
        # write Bell state 1 into NOON resonators

        q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2+delay

        q0.z += env.rect(start, q0.noonSwapLenCSQRT, q0.noonSwapAmpC)
        start += q0.noonSwapLenCSQRT+delay
        q1.z += env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)
        start += q1.noonSwapLenC+delay+extraDelay
        # Bell state 2

        q0.z += env.rect(start, q0.resetLens[0], q0.resetAmps[0])
        q1.z += env.rect(start, q1.resetLens[2], q1.resetAmps[2])
        start += max([q0.resetLens[0],q1.resetLens[2]])+extraDelay
        # reset Bell state 2

        q0.z += env.rect(start, q0.noonSwapLen0, q0.noonSwapAmp0)
        q1.z += env.rect(start, q1.noonSwapLen0, q1.noonSwapAmp0)
        start += max([q0.noonSwapLen0,q1.noonSwapLen0])

        start += curr
        q0.z += eh.measurePulse(q0, start)
        q1.z += eh.measurePulse(q1, start)

        q0['readout'] = True
        q1['readout'] = True
        return runQubits(server, qubits, stats=stats)
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

def Bell1PostResetBell2Tomo(sample, repetition=10, measure=[1,0], stats=1500L, delay=0*ns, extraDelayStore=100.0*ns,
                            name='Bell 1 post-reset Bell 2 TOMO MQ', save=True, collect=False, noisy=True, extraDelay=extraDelay):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    measurement = pyle.dataking.measurement.Tomo(2)

    nameEx = [' q0->q1',' q1->q0']
    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx[measure[0]], axes, measure=measurement, kw=kw)

    def func(server, curr):

        start = 0

        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2+delay

        q0.z = env.rect(start, q0.noonSwapLenCSQRT, q0.noonSwapAmpC)
        start += q0.noonSwapLenCSQRT+delay
        q1.z = env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)
        start += q1.noonSwapLenC+delay
        # Bell state 1

        q0.z += env.rect(start, q0.noonSwapLen0, q0.noonSwapAmp0)
        q1.z += env.rect(start, q1.noonSwapLen0, q1.noonSwapAmp0)
        start += max([q0.noonSwapLen0,q1.noonSwapLen0])+extraDelay
        # write Bell state 1 into NOON resonators

        q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2+delay

        q0.z += env.rect(start, q0.noonSwapLenCSQRT, q0.noonSwapAmpC)
        start += q0.noonSwapLenCSQRT+delay
        q1.z += env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)
        start += q1.noonSwapLenC+delay+extraDelay
        # Bell state 2

        q0.z += env.rect(start, q0.resetLens[0], q0.resetAmps[0])
        q1.z += env.rect(start, q1.resetLens[2], q1.resetAmps[2])
        start += max([q0.resetLens[0],q1.resetLens[2]])+extraDelayStore
        # reset Bell state 2

        q0.z += env.rect(start, q0.noonSwapLen0, q0.noonSwapAmp0)
        q1.z += env.rect(start, q1.noonSwapLen0, q1.noonSwapAmp0)
        start += max([q0.noonSwapLen0,q1.noonSwapLen0])

        return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)
    Qk = np.reshape(result[1:],(9,4))
    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    rho_cal = tomo.qst(Qk,'tomo2')
    plotRhoSingle(rho_cal,figNo=100)
    pylab.title('Exp.')
    rho_caln = rho_cal.copy()
    rho_caln[1,2] = abs(rho_caln[1,2])
    rho_caln[2,1] = abs(rho_caln[2,1])
    print 'Fidelity is %g ' % np.trace(np.dot(rho_caln,rho_ideal))

    Us =tomo._qst_transforms['tomo2'][0]
    rho_calLike = tomo.qst_mle(Qk,Us,rho0 = rho_ideal)
    rho_calLiken = rho_calLike.copy()
    rho_calLiken[1,2] = abs(rho_calLiken[1,2])
    rho_calLiken[2,1] = abs(rho_calLiken[2,1])
    plotRhoSingle(rho_calLike,figNo=101)
    pylab.title('Exp. likely')
    print 'Fidelity for likelyhood is %g ' % np.trace(np.dot(rho_calLiken,rho_ideal))
    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal)), rho_calLike, np.trace(np.dot(rho_calLiken,rho_ideal))

def CPHASECompPulseQ1Fock1(sample, compAmp=np.arange(-0.05,0.05,0.001), measure=0, stats=1200L,
                           CPHASETime=None*ns, compPulseLen=10.0*ns, delay=0.0*ns, phase=0,
                           name='CPHASE compensation pulse qubit 1 Fock state 1', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(compAmp, 'compensation pulse amplitude')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

    def func(server, curr):

         start = 0

         ph = phase

#         # excite q0 from g to e
#         q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))

         # first Ramsey pulse on q1
         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2+2.0*ns

         # START CZ GATE

         # generate Fock state 1 in rc
         q0.z = env.rect(start, sl[0], q0.noonSwapAmpC)
         start += sl[0]
#         +4.0*ns

         # CZ gate
         q1.z = env.rect(start, CPHASETime, q1.noonSwapAmpC21)
         start += CPHASETime+2.0*ns

         # map back rc into q0
         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC)
         start += sl[0]
#         start += max(compPulseLen, sl[0])+compPulseLen+2.0*ns+q0['piLen']/2

         # compensation pulse for q1
         q1.z += env.rect(start, compPulseLen, curr)
         start += compPulseLen+2.0*ns+q0['piLen']/2

         # END CZ GATE

         # second Ramsey pulse on q1
         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2

         # measure pulse and readout q1
         q1.z += eh.measurePulse(q1, start)
         q1['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def CPHASECompPulseQ0Fock1(sample, compAmp=np.arange(-0.05,0.05,0.001), measure=0, stats=1200L,
                           CPHASETime=None*ns, compPulseLen=10.0*ns, compPulseAmpQ1=0.051, delay=0.0*ns, phase=0,
                           name='CPHASE Compensation Pulse qubit 0 Fock state 1', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(compAmp, 'compensation pulse amplitude')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

    def func(server, curr):

         start = 0

         ph = phase

         # first Ramsey pulse on q0
         q0.xy = eh.mix(q0, eh.piHalfPulse(q0, start, phase=ph))
         start += q0['piLen']/2+2.0*ns

         # START CZ GATE

         # generate Fock state 1 in rc
         q0.z = env.rect(start, sl[0], q0.noonSwapAmpC)
         start += sl[0]

         # CZ gate
         q1.z = env.rect(start, CPHASETime, q1.noonSwapAmpC21)
         start += CPHASETime+2.0*ns

         # map back rc into q0
         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC)
         start += sl[0]

         # compensation pulse for q1
         q1.z += env.rect(start, compPulseLen, compPulseAmpQ1)

         # compensation pulse for q0
         q0.z += env.rect(start, compPulseLen, curr)
         start += compPulseLen+2.0*ns+q0['piLen']/2

         # END CZ GATE

         # second Ramsey pulse on q0
         q0.xy += eh.mix(q0, eh.piHalfPulse(q0, start, phase=ph))
         start += q0['piLen']/2

         # measure pulse and readout q0
         q0.z += eh.measurePulse(q0, start)
         q0['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def CPHASETomoTest(sample, repetition=10, measure=[0,1], stats=1500L, phase=0.5*np.pi,
                   name='CPHASE Tomo Test MQ', extraName='0p1 q0', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    measurement = pyle.dataking.measurement.Tomo(2)

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+' '+extraName, axes, measure=measurement, kw=kw)

    def func(server, curr):

         start = 0

         ph = phase

         # Ramsey pulse on q0
         q0.xy = eh.mix(q0, eh.piHalfPulse(q0, start, phase=ph))
         start += q0['piLen']/2

         return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)
    Qk = np.reshape(result[1:],(9,4))
    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    rho_cal = tomo.qst(Qk,'tomo2')
    plotRhoSingle(rho_cal,figNo=100)
    pylab.title('Exp.')
    rho_caln = rho_cal.copy()
    rho_caln[1,2] = abs(rho_caln[1,2])
    rho_caln[2,1] = abs(rho_caln[2,1])
    print 'Fidelity is %g ' % np.trace(np.dot(rho_caln,rho_ideal))

    Us =tomo._qst_transforms['tomo2'][0]
    rho_calLike = tomo.qst_mle(Qk,Us,rho0 = rho_ideal)
    rho_calLiken = rho_calLike.copy()
    rho_calLiken[1,2] = abs(rho_calLiken[1,2])
    rho_calLiken[2,1] = abs(rho_calLiken[2,1])
    plotRhoSingle(rho_calLike,figNo=101)
    pylab.title('Exp. likely')
    print 'Fidelity for likelyhood is %g ' % np.trace(np.dot(rho_calLiken,rho_ideal))
    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal)), rho_calLike, np.trace(np.dot(rho_calLiken,rho_ideal))

def CPHASEGateCompPulseBellTomo(sample, repetition=10, measure=[0,1], stats=1500L,
                                CPHASETime=None*ns, compPulseLen=10.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                delay=2.0*ns, phase=0.5*np.pi,
                                name='CPHASE Gate Compensation Pulse Bell TOMO MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

#    measurement = pyle.dataking.measurement.Tomo(2)
    measurement = pyle.dataking.measurement.Octomo(2)

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measurement, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

    def func(server, curr):

         start = 0

         ph = phase

         # Ramsey pulse on q0
         q0.xy = eh.mix(q0, eh.piHalfPulse(q0, start, phase=ph))
         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q0['piLen']/2+delay

         # START CZ GATE

         # generate Fock state 1 in rc
         q0.z = env.rect(start, sl[0], q0.noonSwapAmpC)
         start += sl[0]

         # CZ gate
         q1.z = env.rect(start, CPHASETime, q1.noonSwapAmpC21)
         start += CPHASETime+delay

         # map back rc into q0
         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC)
         start += sl[0]

         # compensation pulse for q1
         q1.z += env.rect(start, compPulseLen, compPulseAmpQ1)

         # compensation pulse for q0
         q0.z += env.rect(start, compPulseLen, compPulseAmpQ0)
         start += compPulseLen+delay+q0['piLen']/2

         # END CZ GATE

         # Ramsey pulse on q0
         q0.xy += eh.mix(q0, eh.piHalfPulse(q0, start, phase=ph))
         start += q0['piLen']/2

         return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)
#    Qk = np.reshape(result[1:],(9,4))
    Qk = np.reshape(result[1:],(36,4))
    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

#    rho_cal = tomo.qst(Qk,'tomo2')
    rho_cal = tomo.qst(Qk,'octomo2')
    plotRhoSingle(rho_cal,figNo=100)
    pylab.title('Exp.')
    rho_caln = rho_cal.copy()
    rho_caln[1,2] = abs(rho_caln[1,2])
    rho_caln[2,1] = abs(rho_caln[2,1])
    print 'Fidelity is %g ' % np.trace(np.dot(rho_caln,rho_ideal))

#    Us =tomo._qst_transforms['tomo2'][0]
#    rho_calLike = tomo.qst_mle(Qk,Us,rho0 = rho_ideal)
#    rho_calLiken = rho_calLike.copy()
#    rho_calLiken[1,2] = abs(rho_calLiken[1,2])
#    rho_calLiken[2,1] = abs(rho_calLiken[2,1])
#    plotRhoSingle(rho_calLike,figNo=101)
#    pylab.title('Exp. likely')
#    print 'Fidelity for likelyhood is %g ' % np.trace(np.dot(rho_calLiken,rho_ideal))
#    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal)), rho_calLike, np.trace(np.dot(rho_calLiken,rho_ideal))

def CPHASECompPulseQ1Fock1OS(sample, compAmp=np.arange(-0.1,0.1,0.001), measure=0, stats=1500L,
                            CPHASETime=None*ns, compPulseLen=7.0*ns, delay=0.0*ns, phase=0, overshoot=False,
                            name='CPHASE compensation pulse qubit 1 Fock state 1 OS', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(compAmp, 'compensation pulse amplitude')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

    def func(server, curr):

         start = 0

         ph = phase

         # excite q0 from g to e
         q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
         start += q0['piLen']/2

         # START CZ GATE

         # generate Fock state 1 in rc
         q0.z = env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)
         start += sl[0]+delay-q1['piLen']/2

         # first Ramsey pulse on q1
         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2

         # CZ gate
         q1.z = env.rect(start, CPHASETime, q1.noonSwapAmpC21)
         start += CPHASETime

         # map back rc into q0
         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)

         # compensation pulse for q1
         q1.z += env.rect(start, compPulseLen, curr)
         start += compPulseLen+q1['piLen']/2

         # second Ramsey pulse on q1
         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2

         # END CZ GATE

         # measure pulse and readout q1
         q1.z += eh.measurePulse(q1, start)
         q1['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def genCPHASECompPulseQ1Fock1OS(sample, compAmp=np.arange(-0.15,0.1,0.001), measure=0, stats=1500L,
                                CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                delay=0.0*ns, phase=0, overshoot=False,
                                name='Generalized CPHASE compensation pulse qubit 1 Fock state 1 OS',
                                nameEx='',
                                save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(compAmp, 'compensation pulse amplitude')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx, axes, measure=measure, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASEAmp == None:

         CPHASEAmp = q1.noonSwapAmpC21

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

    def func(server, curr):

         start = 0

         ph = phase

         # excite q0 from g to e
         q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
         start += q0['piLen']/2

         # START Cphi GATE

         # generate Fock state 1 in rc
         q0.z = env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)
         start += sl[0]+delay-q1['piLen']/2

         # first Ramsey pulse on q1
         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2

         # Cphi gate
         q1.z = env.rect(start, CPHASETime, CPHASEAmp)
         start += CPHASETime

         # map back rc into q0
         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)

         # compensation pulse for q1
         q1.z += env.rect(start, compPulseLen, curr)
         start += compPulseLen+q1['piLen']/2

         # second Ramsey pulse on q1
         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2

         # END Cphi GATE

         # measure pulse and readout q1
         q1.z += eh.measurePulse(q1, start)
         q1['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])

    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)

def findCpaQ1F1Fourier(sample, compAmp=np.arange(-0.15,0.1,0.001), measure=0, stats=1500L,
                       compPulseLen=7.0*ns, delay=0.0*ns, phase=0, overshoot=False,
                       name='Find compensation pulse amplitude qubit 1 Fock state 1 Fourier',
                       nameEx='',
                       save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(compAmp, 'compensation pulse amplitude')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx, axes, measure=measure, kw=kw)

    sl = q0['noonSwapLenCs']

    def func(server, curr):

         start = 0

         ph = phase

         # excite q0 from g to e
         q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
         start += q0['piLen']/2

         # START Fourier transform

         # map q0 onto rc
         q0.z = env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)
         start += sl[0]+delay-q1['piLen']/2

         # first Ramsey pulse on q1
         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2

         # half CZ gate
         q1.z = env.rect(start, q1.noonSwapLenC21s[0], q1.noonSwapAmpC21)
         start += q1.noonSwapLenC21s[0]+1.0*ns

         # entanglement
         q1.z += env.rect(start, 0.5*q1.noonSwapLenCs[0], q1.noonSwapAmpC)
         start += 0.5*q1.noonSwapLenCs[0]+1.0*ns

         # half CZ gate
         q1.z += env.rect(start, q1.noonSwapLenC21s[0], q1.noonSwapAmpC21)
         start += q1.noonSwapLenC21s[0]

         # map back rc onto q0
         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)

         # compensation pulse for q1
         q1.z += env.rect(start, compPulseLen, curr)
         start += compPulseLen+q1['piLen']/2

         # second Ramsey pulse on q1
         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2

         # END Fourier transform

         # measure pulse and readout q1
         q1.z += eh.measurePulse(q1, start)
         q1['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])

    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)

def genCPHASECompPulseQ1Fock1OSLong(sample, compAmp=np.arange(-0.15,0.1,0.001), measure=0, stats=1500L,
                                    CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                    delay=0.0*ns, phase=0, overshoot=False,
                                    name='Generalized CPHASE compensation pulse qubit 1 Fock state 1 OS long',
                                    nameEx='',
                                    save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(compAmp, 'compensation pulse amplitude')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx, axes, measure=measure, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASEAmp == None:

         CPHASEAmp = q1.noonSwapAmpC21

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

    def func(server, curr):

         start = 0

         ph = phase

         # excite q0 from g to e
         q0.xy = eh.mix(q0, eh.piPulseHD(q0, start-q0['piLen']/2))

         # first Ramsey pulse on q1
         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start-q1['piLen']/2, phase=ph))

         # START Cphi GATE

         # generate Fock state 1 in rc
         q0.z = env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)
         start += sl[0]+delay

         # Cphi gate
         q1.z = env.rect(start, CPHASETime, CPHASEAmp)
         start += CPHASETime

         # map back rc into q0
         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)

         # compensation pulse for q1
         q1.z += env.rect(start, compPulseLen, curr)
         start += max([(sl[0]['ns']+delay['ns']+compPulseLen['ns']), (compPulseLen['ns']+0.5*q1['piLen']['ns'])])

         # second Ramsey pulse on q1
         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2

         # END Cphi GATE

         # measure pulse and readout q1
         q1.z += eh.measurePulse(q1, start)
         q1['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])

    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)

def repeatGenCPHASECompPulseQ1Fock1OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=1500L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, plotFlag=False,
                                      name='Repeat generalized CPHASE compensation pulse qubit 1 Fock state 1 OS',
                                      save=True, collect=True, noisy=True):

#    CPHASEAmp = readArray('swapAmp20100606.txt')[0][13:45]
# [0:]
    CPHASEAmp = readArray('swapAmp20100618.txt')[0][0:220]
    #CPHASEAmp = readArray('swapAmp20100612.txt')[0][2:4]


    # CPHASEAmp = readArray('swapAmp20100606.txt')[0][0:2]
#    CPHASETime = readArray('swapTime20100606.txt')[0][13:45]
    CPHASETime = readArray('swapTime20100618.txt')[0][0:220]
    #CPHASETime = readArray('swapTime20100612.txt')[0][2:4]


    # CPHASETime = readArray('swapTime20100606.txt')[0][0:2]
#    [29:31]
#    CPHASEAmp = readArray('swapAmp.txt')[0][30:-1]
#    CPHASETime = readArray('swapTime.txt')[0][30:-1]

    zpaMin = np.zeros((len(CPHASEAmp),1))
    zpaMax = np.zeros((len(CPHASEAmp),1))
    parray = np.zeros((len(CPHASEAmp),4))

    def fitfunc(cpa, p):

        return p[0]+p[1]*np.cos(2*np.pi*(p[2]*cpa-p[3]))

    for i in range(len(CPHASETime)):

        print 'CPHASE time is %g' % CPHASETime[i]

        # pdb.set_trace()

        data = genCPHASECompPulseQ1Fock1OS(sample, compAmp=compAmp, measure=measure, stats=stats,
                                           CPHASEAmp=CPHASEAmp[i], CPHASETime=CPHASETime[i]*ns, compPulseLen=compPulseLen,
                                           delay=delay, phase=phase, overshoot=overshoot,
                                           nameEx=str(i), save=save, collect=True, noisy=noisy)

        def errfunc(p):

            return data[:,1]-fitfunc(data[:,0], p)

        p,ok = leastsq(errfunc, [0.5,0.8,10.0,0.0])

        parray[i,:] = p

        zpafit = np.linspace(data[0,0],data[-1,0],1000)
        prob = fitfunc(zpafit,p)

        plt.figure(10)
        plt.clf()
        plt.plot(zpafit,prob,'r-')
        plt.plot(data[:,0],data[:,1],'b.')

        # to check plots
#        pdb.set_trace()

        minProb = np.min(prob)
        zpaMin[i] = zpafit[np.argmin(prob)]
        maxProb = np.max(prob)
        zpaMax[i] = zpafit[np.argmax(prob)]

    saveArray('cpaQubit1Fock1min20100618.txt',zpaMin)
    saveArray('cpaQubit1Fock1max20100618.txt',zpaMax)
    saveArray('cpaFittingParsQubit1Fock1_20100618.txt',parray)

def genCPHASECompPulseQ1Fock0OS(sample, compAmp=np.arange(-0.15,0.1,0.001), measure=0, stats=1500L,
                                CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                delay=0.0*ns, phase=0, overshoot=False,
                                name='Generalized CPHASE compensation pulse qubit 1 Fock state 0 OS',
                                nameEx='',
                                save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(compAmp, 'compensation pulse amplitude')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx, axes, measure=measure, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASEAmp == None:

         CPHASEAmp = q1.noonSwapAmpC21

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

    def func(server, curr):

         start = 0

         ph = phase

         # START Cphi GATE

         # fake generating Fock state 1 in rc
         q0.z = env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)
         start += sl[0]+delay-q1['piLen']/2

         # first Ramsey pulse on q1
         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2

         # Cphi gate
         q1.z = env.rect(start, CPHASETime, CPHASEAmp)
         start += CPHASETime

         # fake mapping back rc into q0
         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)

         # compensation pulse for q1
         q1.z += env.rect(start, compPulseLen, curr)
         start += compPulseLen+q1['piLen']/2

         # second Ramsey pulse on q1
         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2

         # END Cphi GATE

         # measure pulse and readout q1
         q1.z += eh.measurePulse(q1, start)
         q1['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])

    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)

def findCpaQ1F0Fourier(sample, compAmp=np.arange(-0.15,0.1,0.001), measure=0, stats=1500L,
                       compPulseLen=7.0*ns, delay=0.0*ns, phase=0, overshoot=False,
                       name='Find compensation pulse amplitude qubit 1 Fock state 0 Fourier',
                       nameEx='',
                       save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(compAmp, 'compensation pulse amplitude')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx, axes, measure=measure, kw=kw)

    sl = q0['noonSwapLenCs']

    def func(server, curr):

         start = 0

         ph = phase

#         # excite q0 from g to e
#         q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
#         start += q0['piLen']/2

         # START Fourier transform

         # map q0 onto rc
         q0.z = env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)
         start += sl[0]+delay-q1['piLen']/2

         # first Ramsey pulse on q1
         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2

         # half CZ gate
         q1.z = env.rect(start, q1.noonSwapLenC21s[0], q1.noonSwapAmpC21)
         start += q1.noonSwapLenC21s[0]+1.0*ns

         # entanglement
         q1.z += env.rect(start, 0.5*q1.noonSwapLenCs[0], q1.noonSwapAmpC)
         start += 0.5*q1.noonSwapLenCs[0]+1.0*ns

         # half CZ gate
         q1.z += env.rect(start, q1.noonSwapLenC21s[0], q1.noonSwapAmpC21)
         start += q1.noonSwapLenC21s[0]

         # map back rc onto q0
         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)

         # compensation pulse for q1
         q1.z += env.rect(start, compPulseLen, curr)
         start += compPulseLen+q1['piLen']/2

         # second Ramsey pulse on q1
         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2

         # END Fourier transform

         # measure pulse and readout q1
         q1.z += eh.measurePulse(q1, start)
         q1['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])

    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)

def genCPHASECompPulseQ1Fock0OSLong(sample, compAmp=np.arange(-0.15,0.1,0.001), measure=0, stats=1500L,
                                    CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                    delay=0.0*ns, phase=0, overshoot=False,
                                    name='Generalized CPHASE compensation pulse qubit 1 Fock state 0 OS long',
                                    nameEx='',
                                    save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(compAmp, 'compensation pulse amplitude')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx, axes, measure=measure, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASEAmp == None:

         CPHASEAmp = q1.noonSwapAmpC21

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

    def func(server, curr):

         start = 0

         ph = phase

         # first Ramsey pulse on q1
         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start-q1['piLen']/2, phase=ph))

         # START Cphi GATE

         # generate Fock state 1 in rc
         q0.z = env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)
         start += sl[0]+delay

         # Cphi gate
         q1.z = env.rect(start, CPHASETime, CPHASEAmp)
         start += CPHASETime

         # map back rc into q0
         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)

         # compensation pulse for q1
         q1.z += env.rect(start, compPulseLen, curr)
         start += max([(sl[0]['ns']+delay['ns']+compPulseLen['ns']), (compPulseLen['ns']+0.5*q1['piLen']['ns'])])

         # second Ramsey pulse on q1
         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2

         # END Cphi GATE

         # measure pulse and readout q1
         q1.z += eh.measurePulse(q1, start)
         q1['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])

    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)

def repeatGenCPHASECompPulseQ1Fock0OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=1500L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, plotFlag=False,
                                      name='Repeat generalized CPHASE compensation pulse qubit 1 Fock state 0 OS',
                                      nameEx=None,
                                      save=True, collect=True, noisy=True):

#    CPHASEAmp = readArray('swapAmp20100606.txt')[0][13:45]
    CPHASEAmp = readArray('swapAmp20100618.txt')[0][0:220]
    #CPHASEAmp = readArray('swapAmp20100612.txt')[0][2:4]


    # CPHASEAmp = readArray('swapAmp20100606.txt')[0][0:2]
#    CPHASETime = readArray('swapTime20100606.txt')[0][13:45]
    CPHASETime = readArray('swapTime20100618.txt')[0][0:220]
    #CPHASETime = readArray('swapTime20100612.txt')[0][2:4]


    # CPHASETime = readArray('swapTime20100606.txt')[0][0:2]
#    [29:31]
#    CPHASEAmp = readArray('swapAmp.txt')[0][30:-1]
#    CPHASETime = readArray('swapTime.txt')[0][30:-1]

    zpaMin = np.zeros((len(CPHASEAmp),1))
    zpaMax = np.zeros((len(CPHASEAmp),1))
    parray = np.zeros((len(CPHASEAmp),4))

    def fitfunc(cpa, p):

        return p[0]+p[1]*np.cos(2*np.pi*(p[2]*cpa-p[3]))

    for i in range(len(CPHASETime)):

        print 'CPHASE time is %g' % CPHASETime[i]

        # pdb.set_trace()

        data = genCPHASECompPulseQ1Fock0OS(sample, compAmp=compAmp, measure=measure, stats=stats,
                                           CPHASEAmp=CPHASEAmp[i], CPHASETime=CPHASETime[i]*ns, compPulseLen=compPulseLen,
                                           delay=delay, phase=phase, overshoot=overshoot,
                                           nameEx=str(i), save=save, collect=True, noisy=noisy)

        def errfunc(p):

            return data[:,1]-fitfunc(data[:,0], p)

        p,ok = leastsq(errfunc, [0.5,0.8,10.0,0.0])

        parray[i,:] = p

        zpafit = np.linspace(data[0,0],data[-1,0],1000)
        prob = fitfunc(zpafit,p)

        plt.figure(11)
        plt.clf()
        plt.plot(zpafit,prob,'r-')
        plt.plot(data[:,0],data[:,1],'b.')

        # to check plots
#        pdb.set_trace()

        minProb = np.min(prob)
        zpaMin[i] = zpafit[np.argmin(prob)]
        maxProb = np.max(prob)
        zpaMax[i] = zpafit[np.argmax(prob)]

    saveArray('cpaQubit1Fock0min20100618.txt',zpaMin)
    saveArray('cpaQubit1Fock0max20100618.txt',zpaMax)
    saveArray('cpaFittingParsQubit1Fock0_20100618.txt',parray)

def CPHASECompPulseQ0Fock1OS(sample, compAmp=np.arange(-0.1,0.1,0.001), measure=0, stats=1500L,
                             CPHASETime=None*ns, compPulseLen=7.0*ns, compPulseAmpQ1=0.051, delay=0.0*ns, phase=0, overshoot=False,
                             name='CPHASE compensation pulse qubit 0 Fock state 1 OS', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(compAmp, 'compensation pulse amplitude')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

    def func(server, curr):

         start = 0

         ph = phase

         # first Ramsey pulse on q0
         q0.xy = eh.mix(q0, eh.piHalfPulse(q0, start, phase=ph))
         start += q0['piLen']/2

         # START CZ GATE

         # generate Fock state 1 in rc
         q0.z = env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)
         start += sl[0]+delay

         # CZ gate
         q1.z = env.rect(start, CPHASETime, q1.noonSwapAmpC21)
         start += CPHASETime

         # map back rc into q0
         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)

         # compensation pulse for q1
         q1.z += env.rect(start, compPulseLen, compPulseAmpQ1)
         start += sl[0]+delay
#         start += max((sl[0]+delay), compPulseLen)

         # compensation pulse for q0
         q0.z += env.rect(start, compPulseLen, curr)
         start += compPulseLen+q0['piLen']/2

         # END CZ GATE

         # second Ramsey pulse on q0
         q0.xy += eh.mix(q0, eh.piHalfPulse(q0, start, phase=ph))
         start += q0['piLen']/2

         # measure pulse and readout q0
         q0.z += eh.measurePulse(q0, start)
         q0['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def genCPHASECompPulseQ0Fock1OS(sample, compAmp=np.arange(-0.1,0.1,0.001), measure=0, stats=1500L,
                                CPHASEAmp=None, CPHASETime=None*ns, compPulseLen=7.0*ns, compPulseAmpQ1=0.051,
                                delay=0.0*ns, phase=0, overshoot=False,
                                name='Generalized CPHASE compensation pulse qubit 0 Fock state 1 OS',
                                nameEx='',
                                save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(compAmp, 'compensation pulse amplitude')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx, axes, measure=measure, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASEAmp == None:

         CPHASEAmp = q1.noonSwapAmpC21

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

    def func(server, curr):

         start = 0

         ph = phase

         # first Ramsey pulse on q0
         q0.xy = eh.mix(q0, eh.piHalfPulse(q0, start, phase=ph))
         start += q0['piLen']/2

         # START Cphi GATE

         # generate Fock state 1 in rc
         q0.z = env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)
         start += sl[0]+delay

         # CZ gate
         q1.z = env.rect(start, CPHASETime, CPHASEAmp)
         start += CPHASETime

         # map back rc into q0
         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)

         # compensation pulse for q1
         q1.z += env.rect(start, compPulseLen, compPulseAmpQ1)
         start += sl[0]+delay
#         start += max((sl[0]+delay), compPulseLen)

         # compensation pulse for q0
         q0.z += env.rect(start, compPulseLen, curr)
         start += compPulseLen+q0['piLen']/2

         # END Cphi GATE

         # second Ramsey pulse on q0
         q0.xy += eh.mix(q0, eh.piHalfPulse(q0, start, phase=ph))
         start += q0['piLen']/2

         # measure pulse and readout q0
         q0.z += eh.measurePulse(q0, start)
         q0['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])

    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)

def findCpaQ0Fourier(sample, compAmp=np.arange(-0.15,0.1,0.001), measure=0, stats=1500L,
                     compPulseLen=7.0*ns, compPulseAmpQ1=0.0, delay=0.0*ns, phase=0, overshoot=False,
                     name='Find compensation pulse amplitude qubit 0 Fourier',
                     nameEx='',
                     save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(compAmp, 'compensation pulse amplitude')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx, axes, measure=measure, kw=kw)

    sl = q0['noonSwapLenCs']

    def func(server, curr):

         start = 0

         ph = phase

         # first Ramsey pulse on q0
         q0.xy = eh.mix(q0, eh.piHalfPulse(q0, start, phase=ph))
         start += q0['piLen']/2

         # START Fourier transform

         # map q0 onto rc
         q0.z = env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)
         start += sl[0]+delay

         # half CZ gate
         q1.z = env.rect(start, q1.noonSwapLenC21s[0], q1.noonSwapAmpC21)
         start += q1.noonSwapLenC21s[0]+1.0*ns

         # entanglement
         q1.z += env.rect(start, 0.5*q1.noonSwapLenCs[0], q1.noonSwapAmpC)
         start += 0.5*q1.noonSwapLenCs[0]+1.0*ns

         # half CZ gate
         q1.z += env.rect(start, q1.noonSwapLenC21s[0], q1.noonSwapAmpC21)
         start += q1.noonSwapLenC21s[0]

         # map back rc onto q0
         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)

         # compensation pulse for q1
         q1.z += env.rect(start, compPulseLen, compPulseAmpQ1)
         start += sl[0]+delay

         # compensation pulse for q0
         q0.z += env.rect(start, compPulseLen, curr)
         start += compPulseLen+q0['piLen']/2

         # END Fourier transform

         # second Ramsey pulse on q0
         q0.xy += eh.mix(q0, eh.piHalfPulse(q0, start, phase=ph))
         start += q0['piLen']/2

         # measure pulse and readout q0
         q0.z += eh.measurePulse(q0, start)
         q0['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])

    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)

def genCPHASECompPulseQ0Fock1OSLong(sample, compAmp=np.arange(-0.1,0.1,0.001), measure=0, stats=1500L,
                                    CPHASEAmp=None, CPHASETime=None*ns, compPulseLen=7.0*ns, compPulseAmpQ1=0.051,
                                    delay=0.0*ns, phase=0, overshoot=False,
                                    name='Generalized CPHASE compensation pulse qubit 0 Fock state 1 OS long',
                                    nameEx='',
                                    save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(compAmp, 'compensation pulse amplitude')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx, axes, measure=measure, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASEAmp == None:

         CPHASEAmp = q1.noonSwapAmpC21

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

    def func(server, curr):

         start = 0

         ph = phase

         # first Ramsey pulse on q0
         q0.xy = eh.mix(q0, eh.piHalfPulse(q0, start-q0['piLen']/2, phase=ph))

         # START Cphi GATE

         # generate Fock state 1 in rc
         q0.z = env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)
         start += sl[0]+delay

         # CPHASE gate
         q1.z = env.rect(start, CPHASETime, CPHASEAmp)
         start += CPHASETime

         # map back rc into q0
         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)

         # compensation pulse for q1
         q1.z += env.rect(start, compPulseLen, compPulseAmpQ1)
         start += sl[0]+delay
#         start += max((sl[0]+delay), compPulseLen)

         # compensation pulse for q0
         q0.z += env.rect(start, compPulseLen, curr)
         start += compPulseLen+q0['piLen']/2

         # END CPHASE GATE

         # second Ramsey pulse on q0
         q0.xy += eh.mix(q0, eh.piHalfPulse(q0, start, phase=ph))
         start += q0['piLen']/2

         # measure pulse and readout q0
         q0.z += eh.measurePulse(q0, start)
         q0['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])

    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)

def repeatGenCPHASECompPulseQ0Fock1OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=1500L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, plotFlag=False,
                                      name='Repeat generalized CPHASE compensation pulse qubit 0 Fock state 1 OS',
                                      nameEx=None,
                                      save=True, collect=True, noisy=True):

#    CPHASEAmp = readArray('swapAmp20100606.txt')[0][13:45]
    CPHASEAmp = readArray('swapAmp20100618.txt')[0][0:220]
    #CPHASEAmp = readArray('swapAmp20100612.txt')[0][2:4]


    # CPHASEAmp = readArray('swapAmp20100606.txt')[0][0:2]
#    CPHASETime = readArray('swapTime20100606.txt')[0][13:45]
    CPHASETime = readArray('swapTime20100618.txt')[0][0:220]
    #CPHASETime = readArray('swapTime20100612.txt')[0][2:4]


    # CPHASETime = readArray('swapTime20100606.txt')[0][0:2]
#    [29:31]
#    CPHASEAmp = readArray('swapAmp.txt')[0][30:-1]
#    CPHASETime = readArray('swapTime.txt')[0][30:-1]
#    compPulseAmpQ1 = readArray('cpaQubit1Fock1min20100529.txt')[0]
    compPulseAmpQ1 = readArray('cpaQubit1Fock1min20100618.txt')[:,]

#    pdb.set_trace()

    zpaMin = np.zeros((len(CPHASEAmp),1))
    zpaMax = np.zeros((len(CPHASEAmp),1))
    parray = np.zeros((len(CPHASEAmp),4))

    def fitfunc(cpa, p):

        return p[0]+p[1]*np.cos(2*np.pi*(p[2]*cpa-p[3]))

    for i in range(len(CPHASETime)):

        print 'CPHASE time is %g' % CPHASETime[i]

#        pdb.set_trace()

        data = genCPHASECompPulseQ0Fock1OS(sample, compAmp=compAmp, measure=measure, stats=stats,
                                           CPHASEAmp=CPHASEAmp[i], CPHASETime=CPHASETime[i]*ns, compPulseLen=7.0*ns,
                                           compPulseAmpQ1=compPulseAmpQ1[i],
                                           delay=delay, phase=phase, overshoot=overshoot,
                                           nameEx=str(i), save=save, collect=True, noisy=noisy)

        def errfunc(p):

            return data[:,1]-fitfunc(data[:,0], p)

        p,ok = leastsq(errfunc, [0.5,0.8,10.0,0.0])

        parray[i,:] = p

        zpafit = np.linspace(data[0,0],data[-1,0],1000)
        prob = fitfunc(zpafit,p)

        plt.figure(12)
        plt.clf()
        plt.plot(zpafit,prob,'r-')
        plt.plot(data[:,0],data[:,1],'b.')

        # to check plots
#        pdb.set_trace()

        minProb = np.min(prob)
        zpaMin[i] = zpafit[np.argmin(prob)]
        maxProb = np.max(prob)
        zpaMax[i] = zpafit[np.argmax(prob)]

    saveArray('cpaQubit0Fock1min20100618.txt',zpaMin)
    saveArray('cpaQubit0Fock1max20100618.txt',zpaMax)
    saveArray('cpaFittingParsQubit0Fock1_20100618.txt',parray)

def genCPHASECompPulses(sample):

    repeatGenCPHASECompPulseQ1Fock1OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=1500L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatGenCPHASECompPulseQ1Fock0OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=1500L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatGenCPHASECompPulseQ0Fock1OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=1500L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

def CPHASEGateCompPulseBellTomoOS(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=False,
                                 CPHASETime=None*ns, compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                 delay=0.0*ns, phase=0.5*np.pi,
                                 name='CPHASE gate compensation pulse Bell TOMO OS MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    measurement = pyle.dataking.measurement.Tomo(2)
    # measurement = pyle.dataking.measurement.Octomo(2)

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measurement, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

    def func(server, curr):

         start = 0

         ph = phase

         # Ramsey pulse on q0
         q0.xy = eh.mix(q0, eh.piHalfPulse(q0, start, phase=ph))
         start += q0['piLen']/2

         # START CZ GATE

         # generate Fock state 1 in rc
         q0.z = env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)
         start += sl[0]+delay-q1['piLen']/2

         # Ramsey pulse on q1
         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2

         # CZ gate
         q1.z = env.rect(start, CPHASETime, q1.noonSwapAmpC21)
         start += CPHASETime

         # map back rc into q0
         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)

         # compensation pulse for q1
         q1.z += env.rect(start, compPulseLen, compPulseAmpQ1)

         # compensation pulse for q0
         q0.z += env.rect(start+sl[0]+delay, compPulseLen, compPulseAmpQ0)

         # END CZ GATE

         # Ramsey pulse on q1
         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start+compPulseLen+q1['piLen']/2, phase=ph))
         start += max([(sl[0]['ns']+delay['ns']+compPulseLen['ns']),(compPulseLen['ns']+q1['piLen']['ns'])])
#         start += max((sl[0]+delay+compPulseLen),(compPulseLen+q1['piLen']))

         return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    # rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    # rho_ideal = np.array([[0.5,0.0,0.0,0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.5,0.0,0.0,0.5]])
    # rho_ideal = np.array([[0.0,0.0,0.0,0.0],[0.0,-0.5,0.5,0.0],[0.0,0.5,-0.5,0.0],[0.0,0.0,0.0,0.0]])
    rho_ideal = np.array([[0.5,0.0,0.0,-0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[-0.5,0.0,0.0,0.5]])
    result = np.sum(result,axis=0)/len(repetition)
    Qk = np.reshape(result[1:],(-1,4))
    # Qk = np.reshape(result[1:],(9,4))
    # Qk = np.reshape(result[1:],(36,4))
    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    rho_cal = tomo.qst(Qk,'tomo2')
    # rho_cal = tomo.qst(Qk,'octomo2')
    plotRhoSingle(rho_cal,figNo=100)
    pylab.title('Exp.')
    rho_caln = rho_cal.copy()
#    rho_caln[1,2] = abs(rho_caln[1,2])
#    rho_caln[2,1] = abs(rho_caln[2,1])
    print 'Fidelity is %g ' % np.trace(np.dot(rho_caln,rho_ideal))

#    Us =tomo._qst_transforms['tomo2'][0]
#    rho_calLike = tomo.qst_mle(Qk,Us,rho0 = rho_ideal)
#    rho_calLiken = rho_calLike.copy()
#    rho_calLiken[1,2] = abs(rho_calLiken[1,2])
#    rho_calLiken[2,1] = abs(rho_calLiken[2,1])
#    plotRhoSingle(rho_calLike,figNo=101)
#    pylab.title('Exp. likely')
#    print 'Fidelity for likelyhood is %g ' % np.trace(np.dot(rho_calLiken,rho_ideal))
#    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal)), rho_calLike, np.trace(np.dot(rho_calLiken,rho_ideal))
    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal))

def genCPHASEGateCompPulseBellTomoOS(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=False,
                                     CPHASEAmp=None, CPHASETime=None*ns,
                                     compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                     delay=0.0*ns, delayCPHASE=0.0*ns, phase=0.5*np.pi,
                                     name='Generalized CPHASE gate compensation pulse Bell TOMO OS MQ',
                                     nameEx='',
                                     save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

#    measurement = pyle.dataking.measurement.Tomo(2)
    measurement = pyle.dataking.measurement.Octomo(2)

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measurement, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASEAmp == None:

         CPHASEAmp = q1.noonSwapAmpC21

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

    def func(server, curr):

         start = 0

         ph = phase

         # Ramsey pulse on q0
         q0.xy = eh.mix(q0, eh.piHalfPulse(q0, start, phase=ph))
         start += q0['piLen']/2

         # START Cphi GATE

         # generate Fock state 1 in rc
         q0.z = env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)
         start += sl[0]+delay-q1['piLen']/2

         # Ramsey pulse on q1
         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2

         # Cphi gate
         q1.z = env.rect(start, CPHASETime+delayCPHASE, CPHASEAmp)
         start += CPHASETime+delayCPHASE

         # map back rc into q0
         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)

         # compensation pulse for q1
         q1.z += env.rect(start, compPulseLen, compPulseAmpQ1)

         # compensation pulse for q0
         q0.z += env.rect(start+sl[0]+delay, compPulseLen, compPulseAmpQ0)

         # END Cphi GATE

         # Ramsey pulse on q1
         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start+compPulseLen+q1['piLen']/2, phase=ph))
         start += max([(sl[0]['ns']+delay['ns']+compPulseLen['ns']),(compPulseLen['ns']+q1['piLen']['ns'])])

         return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

#    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)
#    return result

#    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
#    rho_ideal = np.array([[0.5,0.0,0.0,0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.5,0.0,0.0,0.5]])
    rho_ideal = np.array([[0.0,0.0,0.0,0.0],[0.0,0.5,-0.5,0.0],[0.0,-0.5,0.5,0.0],[0.0,0.0,0.0,0.0]])
#    rho_ideal = np.array([[0.5,0.0,0.0,-0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[-0.5,0.0,0.0,0.5]])
    result = np.sum(result,axis=0)/len(repetition)
#    Qk = np.reshape(result[1:],(9,4))
    Qk = np.reshape(result[1:],(36,4))
    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])
#
#    rho_cal = tomo.qst(Qk,'tomo2')
    rho_cal = tomo.qst(Qk,'octomo2')
    plotRhoSingle(rho_cal,figNo=100)
    pylab.title('Exp.')
    rho_caln = rho_cal.copy()
#    rho_caln[1,2] = abs(rho_caln[1,2])
#    rho_caln[2,1] = abs(rho_caln[2,1])
    print 'Fidelity is %g ' % np.trace(np.dot(rho_caln,rho_ideal))

#    Us =tomo._qst_transforms['tomo2'][0]
#    rho_calLike = tomo.qst_mle(Qk,Us,rho0 = rho_ideal)
#    rho_calLiken = rho_calLike.copy()
#    rho_calLiken[1,2] = abs(rho_calLiken[1,2])
#    rho_calLiken[2,1] = abs(rho_calLiken[2,1])
#    plotRhoSingle(rho_calLike,figNo=101)
#    pylab.title('Exp. likely')
#    print 'Fidelity for likelyhood is %g ' % np.trace(np.dot(rho_calLiken,rho_ideal))
#    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal)), rho_calLike, np.trace(np.dot(rho_calLiken,rho_ideal))
    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal))

def genCPHASEGateCompPulseBellTomoOSforRep(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=False,
                                           CPHASEAmp=None, CPHASETime=None*ns,
                                           compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                           delay=0.0*ns, phase=0.5*np.pi,
                                           name='Generalized CPHASE gate compensation pulse Bell TOMO OS MQ for repeat',
                                           nameEx='',
                                           save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

#    measurement = pyle.dataking.measurement.Tomo(2)
    measurement = pyle.dataking.measurement.Octomo(2)

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx, axes, measure=measurement, kw=kw)
#    dataset = sweeps.prepDataset(sample, name, axes, measure=measurement, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASEAmp == None:

         CPHASEAmp = q1.noonSwapAmpC21

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

    def func(server, curr):

         start = 0

         ph = phase

         # Ramsey pulse on q0
         q0.xy = eh.mix(q0, eh.piHalfPulse(q0, start, phase=ph))
         start += q0['piLen']/2

         # START Cphi GATE

         # generate Fock state 1 in rc
         q0.z = env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)
         start += sl[0]+delay-q1['piLen']/2

         # Ramsey pulse on q1
         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2

         # Cphi gate
         q1.z = env.rect(start, CPHASETime, CPHASEAmp)
         start += CPHASETime

         # map back rc into q0
         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)

         # compensation pulse for q1
         q1.z += env.rect(start, compPulseLen, compPulseAmpQ1)

         # compensation pulse for q0
         q0.z += env.rect(start+sl[0]+delay, compPulseLen, compPulseAmpQ0)

         # END Cphi GATE

         # Ramsey pulse on q1
         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start+compPulseLen+q1['piLen']/2, phase=ph))
         start += max([(sl[0]['ns']+delay['ns']+compPulseLen['ns']),(compPulseLen['ns']+q1['piLen']['ns'])])
#         start += max((sl[0]+delay+compPulseLen),(compPulseLen+q1['piLen']))

         return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

#    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)
#    return result

#    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
#    rho_ideal = np.array([[0.5,0.0,0.0,0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.5,0.0,0.0,0.5]])
#    rho_ideal = np.array([[0.0,0.0,0.0,0.0],[0.0,0.5,-0.5,0.0],[0.0,-0.5,0.5,0.0],[0.0,0.0,0.0,0.0]])
#    rho_ideal = np.array([[0.5,0.0,0.0,-0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[-0.5,0.0,0.0,0.5]])
    result = np.sum(result,axis=0)/len(repetition)
#    Qk = np.reshape(result[1:],(9,4))
    Qk = np.reshape(result[1:],(36,4))
    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

#    rho_cal = tomo.qst(Qk,'tomo2')
    rho_cal = tomo.qst(Qk,'octomo2')
    plotRhoSingle(rho_cal,figNo=100)
    pylab.title('Exp.')

    rhoCoherence = rho_cal[1,2]

    rhoPhi = np.arctan2(np.imag(rhoCoherence), np.real(rhoCoherence))

    rhoAmp = np.sqrt(np.real(rhoCoherence)**2+np.imag(rhoCoherence)**2)

    return rhoPhi, rhoAmp

#    rho_caln = rho_cal.copy()
#    rho_caln[1,2] = abs(rho_caln[1,2])
#    rho_caln[2,1] = abs(rho_caln[2,1])
#    print 'Fidelity is %g ' % np.trace(np.dot(rho_caln,rho_ideal))

#    Us =tomo._qst_transforms['tomo2'][0]
#    rho_calLike = tomo.qst_mle(Qk,Us,rho0 = rho_ideal)
#    rho_calLiken = rho_calLike.copy()
#    rho_calLiken[1,2] = abs(rho_calLiken[1,2])
#    rho_calLiken[2,1] = abs(rho_calLiken[2,1])
#    plotRhoSingle(rho_calLike,figNo=101)
#    pylab.title('Exp. likely')
#    print 'Fidelity for likelyhood is %g ' % np.trace(np.dot(rho_calLiken,rho_ideal))
#    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal)), rho_calLike, np.trace(np.dot(rho_calLiken,rho_ideal))
#    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal))

# ************************
# initial states generator
# ************************

_prepOps = [
    ('I', 0, 0),
    ('Xpi', 1, 0),
    ('Ypi/2', 0.5, 0.5),
    ('Xpi/2', 0.5, 1.0)
]

def build_prepOps(n):
    def prepOps(n):
        if n == 0:
            yield ()
        else:
            for op in _prepOps:
                for rest in prepOps(n-1):
                    yield (op,) + rest
    res=[]
    for x in prepOps(n):
        res.append(x)
    return res

# ********
# QPT slow
# ********

def genCPHASEGateCompPulseQuanProcTomoOS(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                         CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                         amp0=1.0, phi0=0.0,
                                         amp1=1.0, phi1=0.0,
                                         compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167, delay=0.0*ns,
                                         name='Generalized CPHASE gate with cp QPT OS MQ mQ1 MQ0',
                                         nameEx='',
                                         save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

#    measurement = pyle.dataking.measurement.Tomo(2)
    measurement = pyle.dataking.measurement.Octomo(2)

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx, axes, measure=measurement, kw=kw)
#    dataset = sweeps.prepDataset(sample, name, axes, measure=measurement, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASEAmp == None:

         CPHASEAmp = q1.noonSwapAmpC21

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

#    ampTest0 = 0.5
#    ampTest1 = 0.5

    def func(server, curr):

         start = 0

         # initial pulse on q0
         q0.xy = eh.mix(q0, amp0*eh.piPulse(q0, start-q0['piLen']/2, phase=phi0*np.pi))

         # initial pulse on q1
         q1.xy = eh.mix(q1, amp1*eh.piPulse(q1, start-q1['piLen']/2, phase=phi1*np.pi))
         start += 2.0*ns
#         start += max([q0['piLen'],q1['piLen']])+2.0*ns
#
#         # test pulse on q0
#         q0.xy += eh.mix(q0, ampTest0*eh.piPulse(q0, start-q0['piLen']/2))
#
#         # test pulse on q1
#         q1.xy += eh.mix(q1, ampTest1*eh.piPulse(q1, start-q1['piLen']/2))
#         start += 2.0*ns

         # START CPHASE GATE

         # generate Fock state 1 in rc
         q0.z = env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)
         start += sl[0]+delay

         # CPHASE gate
         q1.z = env.rect(start, CPHASETime+delayCPHASE, CPHASEAmp)
         start += CPHASETime+delayCPHASE

         # map back rc into q0
         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)

         # compensation pulse for q1
         q1.z += env.rect(start, compPulseLen, compPulseAmpQ1)

         # compensation pulse for q0
         q0.z += env.rect(start+sl[0]+delay, compPulseLen, compPulseAmpQ0)

         # END CPHASE GATE

         start += sl[0]+delay+compPulseLen

         return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

#    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)
#    return result

#    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
#    rho_ideal = np.array([[0.5,0.0,0.0,0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.5,0.0,0.0,0.5]])
#    rho_ideal = np.array([[0.0,0.0,0.0,0.0],[0.0,0.5,-0.5,0.0],[0.0,-0.5,0.5,0.0],[0.0,0.0,0.0,0.0]])
#    rho_ideal = np.array([[0.5,0.0,0.0,-0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[-0.5,0.0,0.0,0.5]])
#    result = np.sum(result,axis=0)/len(repetition)
#    Qk = np.reshape(result[1:],(9,4))
#    Qk = np.reshape(result[1:],(36,4))
#    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

#    rho_cal = tomo.qst(Qk,'tomo2')
#    rho_cal = tomo.qst(Qk,'octomo2')
#    plotRhoSingle(rho_cal,figNo=100)
#    pylab.title('Exp.')

#    rhoCoherence = rho_cal[1,2]

#    rhoPhi = np.arctan2(np.imag(rhoCoherence), np.real(rhoCoherence))

#    rhoAmp = np.sqrt(np.real(rhoCoherence)**2+np.imag(rhoCoherence)**2)

#    return rhoPhi, rhoAmp

def repeatGenCPHASEGateCompPulseQuanProcTomoOS(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                               CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                               amp0=1.0, phi0=0,
                                               amp1=1.0, phi1=0,
                                               compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167, delay=0.0,
                                               name='Repeat generalized CPHASE gate with cp QPT OS MQ mQ1 MQ0',
                                               nameEx='',
                                               save=True, collect=True, noisy=True):

    preOps = build_prepOps(2)
    dimXi = len(preOps)

    for i in range(dimXi):

        print '*****************************'
        print 'density matrix number %g' % i
        print '*****************************'

#        pdb.set_trace()

        genCPHASEGateCompPulseQuanProcTomoOS(sample, repetition=repetition, measure=measure, stats=stats, overshoot=overshoot,
                                             CPHASEAmp=CPHASEAmp, CPHASETime=CPHASETime, delayCPHASE=delayCPHASE,
                                             amp0=preOps[i][0][1], phi0=preOps[i][0][2],
                                             amp1=preOps[i][1][1], phi1=preOps[i][1][2],
                                             compPulseLen=compPulseLen, compPulseAmpQ0=compPulseAmpQ0, compPulseAmpQ1=compPulseAmpQ1, delay=delay,
                                             name=name, nameEx=str(i),
                                             save=save, collect=False, noisy=noisy)

# ************
# pre QPT slow
# ************

def genCPHASEGateCompPulseQuanProcTomoOSPre(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                            amp0=1.0, phi0=0,
                                            amp1=1.0, phi1=0.0,
                                            delay=0.0*ns,
                                            name='Generalized CPHASE gate with cp QPT OS MQ pre',
                                            nameEx='',
                                            save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

#    measurement = pyle.dataking.measurement.Tomo(2)
    measurement = pyle.dataking.measurement.Octomo(2)

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx, axes, measure=measurement, kw=kw)
#    dataset = sweeps.prepDataset(sample, name, axes, measure=measurement, kw=kw)

    sl = q0['noonSwapLenCs']

    def func(server, curr):

         start = 0

         # initial pulse on q0
         q0.xy = eh.mix(q0, amp0*eh.piPulse(q0, start-q0['piLen']/2, phase=phi0*np.pi))

         # initial pulse on q1
         q1.xy = eh.mix(q1, amp1*eh.piPulse(q1, start-q1['piLen']/2, phase=phi1*np.pi))
         start += 2.0*ns

         return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

#    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)
#    return result

#    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
#    rho_ideal = np.array([[0.5,0.0,0.0,0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.5,0.0,0.0,0.5]])
#    rho_ideal = np.array([[0.0,0.0,0.0,0.0],[0.0,0.5,-0.5,0.0],[0.0,-0.5,0.5,0.0],[0.0,0.0,0.0,0.0]])
#    rho_ideal = np.array([[0.5,0.0,0.0,-0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[-0.5,0.0,0.0,0.5]])
#    result = np.sum(result,axis=0)/len(repetition)
#    Qk = np.reshape(result[1:],(9,4))
#    Qk = np.reshape(result[1:],(36,4))
#    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

#    rho_cal = tomo.qst(Qk,'tomo2')
#    rho_cal = tomo.qst(Qk,'octomo2')
#    plotRhoSingle(rho_cal,figNo=100)
#    pylab.title('Exp.')

#    rhoCoherence = rho_cal[1,2]

#    rhoPhi = np.arctan2(np.imag(rhoCoherence), np.real(rhoCoherence))

#    rhoAmp = np.sqrt(np.real(rhoCoherence)**2+np.imag(rhoCoherence)**2)

#    return rhoPhi, rhoAmp

def repeatGenCPHASEGateCompPulseQuanProcTomoOSPre(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                  amp0=1.0, phi0=0,
                                                  amp1=1.0, phi1=0,
                                                  delay=0.0,
                                                  name='Repeat generalized CPHASE gate with cp QPT OS MQ pre',
                                                  nameEx='',
                                                  save=True, collect=True, noisy=True):

    preOps = build_prepOps(2)
    dimXi = len(preOps)

    for i in range(dimXi):

        print '*****************************'
        print 'density matrix number %g' % i
        print '*****************************'

#        pdb.set_trace()

        genCPHASEGateCompPulseQuanProcTomoOSPre(sample, repetition=repetition, measure=measure, stats=stats, overshoot=overshoot,
                                                amp0=preOps[i][0][1], phi0=preOps[i][0][2],
                                                amp1=preOps[i][1][1], phi1=preOps[i][1][2],
                                                delay=delay,
                                                name=name,
                                                nameEx=str(i),
                                                save=save, collect=False, noisy=noisy)

# ********
# QPT fast
# ********

def genCPHASEGateCompPulseQuanProcTomoOSInt(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                            CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                            amp0=1.0, phi0=0,
                                            amp1=1.0, phi1=0.0,
                                            compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167, delay=0.0*ns,
                                            name='Generalized CPHASE gate compensation pulse quantum process TOMO OS MQ mQ1 MQ0 int',
                                            nameEx='',
                                            save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

#    measurement = pyle.dataking.measurement.Tomo(2)
    measurement = pyle.dataking.measurement.Octomo(2)

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx, axes, measure=measurement, kw=kw)
#    dataset = sweeps.prepDataset(sample, name, axes, measure=measurement, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASEAmp == None:

         CPHASEAmp = q1.noonSwapAmpC21

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

#    ampTest0 = 0.0
#    ampTest1 = 1.0

    def func(server, curr):

         start = 0

         # initial pulse on q0
         q0.xy = eh.mix(q0, amp0*eh.piPulse(q0, start-q0['piLen']/2, phase=phi0*np.pi))

         # generate Fock state 1 in rc
         q0.z = env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)
         start += sl[0]+delay-q1['piLen']/2

         # initial pulse on q1
         q1.xy = eh.mix(q1, amp1*eh.piPulse(q1, start, phase=phi1*np.pi))
         start += q1['piLen']/2
#         start += max([q0['piLen'],q1['piLen']])+2.0*ns

#         # test pulse on q0
#         q0.xy += eh.mix(q0, ampTest0*eh.piPulse(q0, start-q0['piLen']/2))
#
#         # test pulse on q1
#         q1.xy += eh.mix(q1, ampTest1*eh.piPulse(q1, start-q1['piLen']/2))
#         start += 2.0*ns

         # START CPHASE GATE

         # CPHASE gate
         q1.z = env.rect(start, CPHASETime+delayCPHASE, CPHASEAmp)
         start += CPHASETime+delayCPHASE

         # map back rc into q0
         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)

         # compensation pulse for q1
         q1.z += env.rect(start, compPulseLen, compPulseAmpQ1)

         # compensation pulse for q0
         q0.z += env.rect(start+sl[0]+delay, compPulseLen, compPulseAmpQ0)

         # END CPHASE GATE

         start += sl[0]+delay+compPulseLen

         return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

#    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)
#    return result

#    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
#    rho_ideal = np.array([[0.5,0.0,0.0,0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.5,0.0,0.0,0.5]])
#    rho_ideal = np.array([[0.0,0.0,0.0,0.0],[0.0,0.5,-0.5,0.0],[0.0,-0.5,0.5,0.0],[0.0,0.0,0.0,0.0]])
#    rho_ideal = np.array([[0.5,0.0,0.0,-0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[-0.5,0.0,0.0,0.5]])
#    result = np.sum(result,axis=0)/len(repetition)
#    Qk = np.reshape(result[1:],(9,4))
#    Qk = np.reshape(result[1:],(36,4))
#    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

#    rho_cal = tomo.qst(Qk,'tomo2')
#    rho_cal = tomo.qst(Qk,'octomo2')
#    plotRhoSingle(rho_cal,figNo=100)
#    pylab.title('Exp.')

#    rhoCoherence = rho_cal[1,2]

#    rhoPhi = np.arctan2(np.imag(rhoCoherence), np.real(rhoCoherence))

#    rhoAmp = np.sqrt(np.real(rhoCoherence)**2+np.imag(rhoCoherence)**2)

#    return rhoPhi, rhoAmp

def repeatGenCPHASEGateCompPulseQuanProcTomoOSInt(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                  CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                                  amp0=1.0, phi0=0,
                                                  amp1=1.0, phi1=0,
                                                  compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167, delay=0.0,
                                                  name='Repeat generalized CPHASE gate compensation pulse quantum process TOMO OS MQ mQ1 MQ0 int',
                                                  nameEx='',
                                                  save=True, collect=True, noisy=True):

    preOps = build_prepOps(2)
    dimXi = len(preOps)

    for i in range(dimXi):

        print '*****************************'
        print 'density matrix number %g' % i
        print '*****************************'

#        pdb.set_trace()

        genCPHASEGateCompPulseQuanProcTomoOSInt(sample, repetition=repetition, measure=measure, stats=stats, overshoot=overshoot,
                                                CPHASEAmp=CPHASEAmp, CPHASETime=CPHASETime, delayCPHASE=delayCPHASE,
                                                amp0=preOps[i][0][1], phi0=preOps[i][0][2],
                                                amp1=preOps[i][1][1], phi1=preOps[i][1][2],
                                                compPulseLen=compPulseLen, compPulseAmpQ0=compPulseAmpQ0, compPulseAmpQ1=compPulseAmpQ1, delay=delay,
                                                name=name, nameEx=str(i),
                                                save=save, collect=False, noisy=noisy)

# **********
# QPT on QFT
# **********

def quanFourierTransformQPT(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                            delayGates=0.0*ns,
                            amp0=1.0, phi0=0,
                            amp1=1.0, phi1=0.0,
                            compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167, delay=0.0*ns,
                            name='QPT on quantum Fourier transform',
                            nameEx='',
                            save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

#    measurement = pyle.dataking.measurement.Tomo(2)
    measurement = pyle.dataking.measurement.Octomo(2)

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx, axes, measure=measurement, kw=kw)
#    dataset = sweeps.prepDataset(sample, name, axes, measure=measurement, kw=kw)

    sl = q0['noonSwapLenCs']

    def func(server, curr):

         start = 0

         # initial pulse on q0
         q0.xy = eh.mix(q0, amp0*eh.piPulse(q0, start-q0['piLen']/2, phase=phi0*np.pi))

         # map q0 onto rc
         q0.z = env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)
         start += sl[0]+delay-q1['piLen']/2

         # initial pulse on q1
         q1.xy = eh.mix(q1, amp1*eh.piPulse(q1, start, phase=phi1*np.pi))
         start += q1['piLen']/2

         # START Fourier transform

         # half CZ gate
         q1.z = env.rect(start, q1.noonSwapLenC21s[0], q1.noonSwapAmpC21)
         start += q1.noonSwapLenC21s[0]+1.0*ns

         # entanglement
         q1.z += env.rect(start, 0.5*q1.noonSwapLenCs[0], q1.noonSwapAmpC)
         start += 0.5*q1.noonSwapLenCs[0]+1.0*ns

         # half CZ gate
         q1.z += env.rect(start, q1.noonSwapLenC21s[0], q1.noonSwapAmpC21)
         start += q1.noonSwapLenC21s[0]

         # map back rc onto q0
         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)

         # compensation pulse for q1
         q1.z += env.rect(start, compPulseLen, compPulseAmpQ1)
         start += sl[0]+delay

         # compensation pulse for q0
         q0.z += env.rect(start, compPulseLen, compPulseAmpQ0)
         start += compPulseLen

         return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

#    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)
#    return result

#    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
#    rho_ideal = np.array([[0.5,0.0,0.0,0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.5,0.0,0.0,0.5]])
#    rho_ideal = np.array([[0.0,0.0,0.0,0.0],[0.0,0.5,-0.5,0.0],[0.0,-0.5,0.5,0.0],[0.0,0.0,0.0,0.0]])
#    rho_ideal = np.array([[0.5,0.0,0.0,-0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[-0.5,0.0,0.0,0.5]])
#    result = np.sum(result,axis=0)/len(repetition)
#    Qk = np.reshape(result[1:],(9,4))
#    Qk = np.reshape(result[1:],(36,4))
#    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

#    rho_cal = tomo.qst(Qk,'tomo2')
#    rho_cal = tomo.qst(Qk,'octomo2')
#    plotRhoSingle(rho_cal,figNo=100)
#    pylab.title('Exp.')

#    rhoCoherence = rho_cal[1,2]

#    rhoPhi = np.arctan2(np.imag(rhoCoherence), np.real(rhoCoherence))

#    rhoAmp = np.sqrt(np.real(rhoCoherence)**2+np.imag(rhoCoherence)**2)

#    return rhoPhi, rhoAmp

def repeatQuanFourierTransformQPT(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                  delayGates=0.0*ns,
                                  amp0=1.0, phi0=0,
                                  amp1=1.0, phi1=0,
                                  compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167, delay=0.0,
                                  name='Repeat QPT on quantum Fourier transform',
                                  nameEx='',
                                  save=True, collect=True, noisy=True):

    preOps = build_prepOps(2)
    dimXi = len(preOps)

    for i in range(dimXi):

        print '*****************************'
        print 'density matrix number %g' % i
        print '*****************************'

#        pdb.set_trace()

        quanFourierTransformQPT(sample, repetition=repetition, measure=measure, stats=stats, overshoot=overshoot,
                                delayGates=delayGates,
                                amp0=preOps[i][0][1], phi0=preOps[i][0][2],
                                amp1=preOps[i][1][1], phi1=preOps[i][1][2],
                                compPulseLen=compPulseLen, compPulseAmpQ0=compPulseAmpQ0, compPulseAmpQ1=compPulseAmpQ1, delay=delay,
                                name=name, nameEx=str(i),
                                save=save, collect=False, noisy=noisy)

# ************
# pre QPT fast
# ************

def genCPHASEGateCompPulseQuanProcTomoOSIntPre(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                               amp0=1.0, phi0=0,
                                               amp1=1.0, phi1=0.0,
                                               delay=0.0*ns,
                                               name='Generalized CPHASE gate with cp QPT OS MQ int pre',
                                               nameEx='',
                                               save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

#    measurement = pyle.dataking.measurement.Tomo(2)
    measurement = pyle.dataking.measurement.Octomo(2)

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx, axes, measure=measurement, kw=kw)
#    dataset = sweeps.prepDataset(sample, name, axes, measure=measurement, kw=kw)

    sl = q0['noonSwapLenCs']

    def func(server, curr):

         start = 0

         # initial pulse on q0
         q0.xy = eh.mix(q0, amp0*eh.piPulse(q0, start-q0['piLen']/2, phase=phi0*np.pi))

#         # generate Fock state 1 in rc
#         q0.z = env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)
         start += sl[0]+delay-q1['piLen']/2

         # initial pulse on q1
         q1.xy = eh.mix(q1, amp1*eh.piPulse(q1, start, phase=phi1*np.pi))
         start += q1['piLen']/2

         return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

#    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)
#    return result

#    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
#    rho_ideal = np.array([[0.5,0.0,0.0,0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.5,0.0,0.0,0.5]])
#    rho_ideal = np.array([[0.0,0.0,0.0,0.0],[0.0,0.5,-0.5,0.0],[0.0,-0.5,0.5,0.0],[0.0,0.0,0.0,0.0]])
#    rho_ideal = np.array([[0.5,0.0,0.0,-0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[-0.5,0.0,0.0,0.5]])
#    result = np.sum(result,axis=0)/len(repetition)
#    Qk = np.reshape(result[1:],(9,4))
#    Qk = np.reshape(result[1:],(36,4))
#    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

#    rho_cal = tomo.qst(Qk,'tomo2')
#    rho_cal = tomo.qst(Qk,'octomo2')
#    plotRhoSingle(rho_cal,figNo=100)
#    pylab.title('Exp.')

#    rhoCoherence = rho_cal[1,2]

#    rhoPhi = np.arctan2(np.imag(rhoCoherence), np.real(rhoCoherence))

#    rhoAmp = np.sqrt(np.real(rhoCoherence)**2+np.imag(rhoCoherence)**2)

#    return rhoPhi, rhoAmp

def repeatGenCPHASEGateCompPulseQuanProcTomoOSIntPre(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                     amp0=1.0, phi0=0,
                                                     amp1=1.0, phi1=0,
                                                     delay=0.0,
                                                     name='Repeat generalized CPHASE gate with cp QPT OS MQ int pre',
                                                     nameEx='',
                                                     save=True, collect=True, noisy=True):

    preOps = build_prepOps(2)
    dimXi = len(preOps)

    for i in range(dimXi):

        print '*****************************'
        print 'density matrix number %g' % i
        print '*****************************'

#        pdb.set_trace()

        genCPHASEGateCompPulseQuanProcTomoOSIntPre(sample, repetition=repetition, measure=measure, stats=stats, overshoot=overshoot,
                                                   amp0=preOps[i][0][1], phi0=preOps[i][0][2],
                                                   amp1=preOps[i][1][1], phi1=preOps[i][1][2],
                                                   delay=delay,
                                                   name=name,
                                                   nameEx=str(i),
                                                   save=save, collect=False, noisy=noisy)

def CZGatesQPT20100604Fri(sample):

    repeatGenCPHASEGateCompPulseQuanProcTomoOSIntPre(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                     amp0=1.0, phi0=0,
                                                     amp1=1.0, phi1=0,
                                                     delay=0.0,
                                                     name='CZ_pre_QPT_int',
                                                     nameEx='',
                                                     save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseQuanProcTomoOSInt(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                  CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                                  amp0=1.0, phi0=0,
                                                  amp1=1.0, phi1=0,
                                                  compPulseLen=7.0*ns, compPulseAmpQ0=-0.0850, compPulseAmpQ1=-0.0558, delay=0.0,
                                                  name='CZ_mQ1_MQ0_U11_int',
                                                  nameEx='',
                                                  save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseQuanProcTomoOSInt(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                  CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                                  amp0=1.0, phi0=0,
                                                  amp1=1.0, phi1=0,
                                                  compPulseLen=7.0*ns, compPulseAmpQ0=-0.0850, compPulseAmpQ1=-0.0982, delay=0.0,
                                                  name='CZ_MQ1_MQ0_U01_int',
                                                  nameEx='',
                                                  save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseQuanProcTomoOSInt(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                  CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.1*ns,
                                                  amp0=1.0, phi0=0,
                                                  amp1=1.0, phi1=0,
                                                  compPulseLen=7.0*ns, compPulseAmpQ0=-0.0402, compPulseAmpQ1=-0.0558, delay=0.0,
                                                  name='CZ_mQ1_mQ0_U10_int',
                                                  nameEx='',
                                                  save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseQuanProcTomoOSInt(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                  CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                                  amp0=1.0, phi0=0,
                                                  amp1=1.0, phi1=0,
                                                  compPulseLen=7.0*ns, compPulseAmpQ0=-0.0402, compPulseAmpQ1=-0.0982, delay=0.0,
                                                  name='CZ_MQ1_mQ0_U00_int',
                                                  nameEx='',
                                                  save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseQuanProcTomoOSPre(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                  amp0=1.0, phi0=0,
                                                  amp1=1.0, phi1=0,
                                                  delay=0.0,
                                                  name='CZ_pre_QPT',
                                                  nameEx='',
                                                  save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseQuanProcTomoOS(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                               CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                               amp0=1.0, phi0=0,
                                               amp1=1.0, phi1=0,
                                               compPulseLen=7.0*ns, compPulseAmpQ0=-0.0850, compPulseAmpQ1=-0.0558, delay=0.0,
                                               name='CZ_mQ1_MQ0_U11',
                                               nameEx='',
                                               save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseQuanProcTomoOS(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                               CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                               amp0=1.0, phi0=0,
                                               amp1=1.0, phi1=0,
                                               compPulseLen=7.0*ns, compPulseAmpQ0=-0.0850, compPulseAmpQ1=-0.0982, delay=0.0,
                                               name='CZ_MQ1_MQ0_U01',
                                               nameEx='',
                                               save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseQuanProcTomoOS(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                               CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.1*ns,
                                               amp0=1.0, phi0=0,
                                               amp1=1.0, phi1=0,
                                               compPulseLen=7.0*ns, compPulseAmpQ0=-0.0402, compPulseAmpQ1=-0.0558, delay=0.0,
                                               name='CZ_mQ1_mQ0_U10',
                                               nameEx='',
                                               save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseQuanProcTomoOS(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                               CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                               amp0=1.0, phi0=0,
                                               amp1=1.0, phi1=0,
                                               compPulseLen=7.0*ns, compPulseAmpQ0=-0.0402, compPulseAmpQ1=-0.0982, delay=0.0,
                                               name='CZ_MQ1_mQ0_U00_int',
                                               nameEx='',
                                               save=True, collect=True, noisy=True)

    swap21(sample, swapLen=st.arangePQ(0,600,2,ns), swapAmp=np.arange(0.02,0.8,0.001), measure=1)

def genCPHASEGateCompPulseQuanProcTomoOSIntBlank1(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                  CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                                  amp0=1.0, phi0=0,
                                                  amp1=1.0, phi1=0.0,
                                                  compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167, delay=0.0*ns,
                                                  name='Generalized CPHASE gate compensation pulse quantum process TOMO OS MQ mQ1 MQ0 int blank 1',
                                                  nameEx='',
                                                  save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

#    measurement = pyle.dataking.measurement.Tomo(2)
    measurement = pyle.dataking.measurement.Octomo(2)

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx, axes, measure=measurement, kw=kw)
#    dataset = sweeps.prepDataset(sample, name, axes, measure=measurement, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASEAmp == None:

         CPHASEAmp = q1.noonSwapAmpC21

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

#    ampTest0 = 0.0
#    ampTest1 = 1.0

    def func(server, curr):

         start = 0

         # initial pulse on q0
         q0.xy = eh.mix(q0, amp0*eh.piPulse(q0, start-q0['piLen']/2, phi0))

#         # generate Fock state 1 in rc
#         q0.z = env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)
         start += sl[0]+delay-q1['piLen']/2

         # initial pulse on q1
         q1.xy = eh.mix(q1, amp1*eh.piPulse(q1, start, phi1))
         start += q1['piLen']/2
#         start += max([q0['piLen'],q1['piLen']])+2.0*ns

#         # test pulse on q0
#         q0.xy += eh.mix(q0, ampTest0*eh.piPulse(q0, start-q0['piLen']/2))
#
#         # test pulse on q1
#         q1.xy += eh.mix(q1, ampTest1*eh.piPulse(q1, start-q1['piLen']/2))
#         start += 2.0*ns

#         # START CPHASE GATE
#
#         # CPHASE gate
#         q1.z = env.rect(start, CPHASETime+delayCPHASE, CPHASEAmp)
         start += CPHASETime+delayCPHASE
#
#         # map back rc into q0
#         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)
#
#         # compensation pulse for q1
#         q1.z += env.rect(start, compPulseLen, compPulseAmpQ1)
#
#         # compensation pulse for q0
#         q0.z += env.rect(start+sl[0]+delay, compPulseLen, compPulseAmpQ0)
#
#         # END CPHASE GATE
#
         start += sl[0]+delay+compPulseLen

         return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

#    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)
#    return result

#    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
#    rho_ideal = np.array([[0.5,0.0,0.0,0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.5,0.0,0.0,0.5]])
#    rho_ideal = np.array([[0.0,0.0,0.0,0.0],[0.0,0.5,-0.5,0.0],[0.0,-0.5,0.5,0.0],[0.0,0.0,0.0,0.0]])
#    rho_ideal = np.array([[0.5,0.0,0.0,-0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[-0.5,0.0,0.0,0.5]])
#    result = np.sum(result,axis=0)/len(repetition)
#    Qk = np.reshape(result[1:],(9,4))
#    Qk = np.reshape(result[1:],(36,4))
#    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

#    rho_cal = tomo.qst(Qk,'tomo2')
#    rho_cal = tomo.qst(Qk,'octomo2')
#    plotRhoSingle(rho_cal,figNo=100)
#    pylab.title('Exp.')

#    rhoCoherence = rho_cal[1,2]

#    rhoPhi = np.arctan2(np.imag(rhoCoherence), np.real(rhoCoherence))

#    rhoAmp = np.sqrt(np.real(rhoCoherence)**2+np.imag(rhoCoherence)**2)

#    return rhoPhi, rhoAmp

def repeatGenCPHASEGateCompPulseQuanProcTomoOSIntBlank1(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                        CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                                        amp0=1.0, phi0=0,
                                                        amp1=1.0, phi1=0,
                                                        compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167, delay=0.0,
                                                        name='Repeat generalized CPHASE gate compensation pulse quantum process TOMO OS MQ mQ1 MQ0 int blank 1',
                                                        nameEx='',
                                                        save=True, collect=True, noisy=True):

    preOps = build_prepOps(2)
    dimXi = len(preOps)

    for i in range(dimXi):

        print '*****************************'
        print 'density matrix number %g' % i
        print '*****************************'

#        pdb.set_trace()

        genCPHASEGateCompPulseQuanProcTomoOSIntBlank1(sample, repetition=repetition, measure=measure, stats=stats, overshoot=overshoot,
                                                      CPHASEAmp=CPHASEAmp, CPHASETime=CPHASETime, delayCPHASE=delayCPHASE,
                                                      amp0=preOps[i][0][1], phi0=preOps[i][0][2],
                                                      amp1=preOps[i][1][1], phi1=preOps[i][1][2],
                                                      compPulseLen=compPulseLen, compPulseAmpQ0=compPulseAmpQ0, compPulseAmpQ1=compPulseAmpQ1, delay=delay,
                                                      name=name, nameEx=str(i),
                                                      save=save, collect=False, noisy=noisy)

def genCPHASEGateCompPulseQuanProcTomoOSIntBlank2(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                  CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                                  amp0=1.0, phi0=0,
                                                  amp1=1.0, phi1=0.0,
                                                  compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167, delay=0.0*ns,
                                                  name='Generalized CPHASE gate compensation pulse quantum process TOMO OS MQ MQ1 MQ0 int blank 2',
                                                  nameEx='',
                                                  save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

#    measurement = pyle.dataking.measurement.Tomo(2)
    measurement = pyle.dataking.measurement.Octomo(2)

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx, axes, measure=measurement, kw=kw)
#    dataset = sweeps.prepDataset(sample, name, axes, measure=measurement, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASEAmp == None:

         CPHASEAmp = q1.noonSwapAmpC21

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

#    ampTest0 = 0.0
#    ampTest1 = 1.0

    def func(server, curr):

         start = 0

         # initial pulse on q0
         q0.xy = eh.mix(q0, amp0*eh.piPulse(q0, start-q0['piLen']/2, phase=phi0*np.pi))

#         # generate Fock state 1 in rc
#         q0.z = env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)
         start += q0['piLen']+sl[0]+delay-q1['piLen']/2

         # initial pulse on q1
         q1.xy = eh.mix(q1, amp1*eh.piPulse(q1, start, phase=phi1*np.pi))
         start += q1['piLen']/2
#         start += max([q0['piLen'],q1['piLen']])+2.0*ns

#         # test pulse on q0
#         q0.xy += eh.mix(q0, ampTest0*eh.piPulse(q0, start-q0['piLen']/2))
#
#         # test pulse on q1
#         q1.xy += eh.mix(q1, ampTest1*eh.piPulse(q1, start-q1['piLen']/2))
#         start += 2.0*ns

#         # START CPHASE GATE
#
#         # CPHASE gate
#         q1.z = env.rect(start, CPHASETime+delayCPHASE, CPHASEAmp)
#         start += CPHASETime+delayCPHASE
#
#         # map back rc into q0
#         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)
#
#         # compensation pulse for q1
#         q1.z += env.rect(start, compPulseLen, compPulseAmpQ1)
#
#         # compensation pulse for q0
#         q0.z += env.rect(start+sl[0]+delay, compPulseLen, compPulseAmpQ0)
#
#         # END CPHASE GATE
#
#         start += sl[0]+delay+compPulseLen

         return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

#    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)
#    return result

#    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
#    rho_ideal = np.array([[0.5,0.0,0.0,0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.5,0.0,0.0,0.5]])
#    rho_ideal = np.array([[0.0,0.0,0.0,0.0],[0.0,0.5,-0.5,0.0],[0.0,-0.5,0.5,0.0],[0.0,0.0,0.0,0.0]])
#    rho_ideal = np.array([[0.5,0.0,0.0,-0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[-0.5,0.0,0.0,0.5]])
#    result = np.sum(result,axis=0)/len(repetition)
#    Qk = np.reshape(result[1:],(9,4))
#    Qk = np.reshape(result[1:],(36,4))
#    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

#    rho_cal = tomo.qst(Qk,'tomo2')
#    rho_cal = tomo.qst(Qk,'octomo2')
#    plotRhoSingle(rho_cal,figNo=100)
#    pylab.title('Exp.')

#    rhoCoherence = rho_cal[1,2]

#    rhoPhi = np.arctan2(np.imag(rhoCoherence), np.real(rhoCoherence))

#    rhoAmp = np.sqrt(np.real(rhoCoherence)**2+np.imag(rhoCoherence)**2)

#    return rhoPhi, rhoAmp

def repeatGenCPHASEGateCompPulseQuanProcTomoOSIntBlank2(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                        CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                                        amp0=1.0, phi0=0,
                                                        amp1=1.0, phi1=0,
                                                        compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167, delay=0.0,
                                                        name='Repeat generalized CPHASE gate compensation pulse quantum process TOMO OS MQ MQ1 MQ0 int blank 2',
                                                        nameEx='',
                                                        save=True, collect=True, noisy=True):

    preOps = build_prepOps(2)
    dimXi = len(preOps)

    for i in range(dimXi):

        print '*****************************'
        print 'density matrix number %g' % i
        print '*****************************'

#        pdb.set_trace()

        genCPHASEGateCompPulseQuanProcTomoOSIntBlank2(sample, repetition=repetition, measure=measure, stats=stats, overshoot=overshoot,
                                                      CPHASEAmp=CPHASEAmp, CPHASETime=CPHASETime, delayCPHASE=delayCPHASE,
                                                      amp0=preOps[i][0][1], phi0=preOps[i][0][2],
                                                      amp1=preOps[i][1][1], phi1=preOps[i][1][2],
                                                      compPulseLen=compPulseLen, compPulseAmpQ0=compPulseAmpQ0, compPulseAmpQ1=compPulseAmpQ1, delay=delay,
                                                      name=name, nameEx=str(i),
                                                      save=save, collect=False, noisy=noisy)

def CZGateCompPulsesQPT20100603ThuInt(sample):

    repeatGenCPHASEGateCompPulseQuanProcTomoOSIntBlank1(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                        CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=-0.1*ns,
                                                        amp0=1.0, phi0=0,
                                                        amp1=1.0, phi1=0,
                                                        compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167, delay=0.0,
                                                        name='CZ_pre_QPT_U11_1_m0p1',
                                                        nameEx='',
                                                        save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseQuanProcTomoOSIntBlank2(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                        CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                                        amp0=1.0, phi0=0,
                                                        amp1=1.0, phi1=0,
                                                        compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167, delay=0.0,
                                                        name='CZ_pre_QPT_U11_2',
                                                        nameEx='',
                                                        save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseQuanProcTomoOSInt(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                  CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=-0.1*ns,
                                                  amp0=1.0, phi0=0,
                                                  amp1=1.0, phi1=0,
                                                  compPulseLen=7.0*ns, compPulseAmpQ0=-0.0835, compPulseAmpQ1=-0.0434, delay=0.0,
                                                  name='CZ_mQ1_MQ0_U11',
                                                  nameEx='',
                                                  save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseQuanProcTomoOSInt(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                  CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.1*ns,
                                                  amp0=1.0, phi0=0,
                                                  amp1=1.0, phi1=0,
                                                  compPulseLen=7.0*ns, compPulseAmpQ0=-0.0835, compPulseAmpQ1=-0.0873, delay=0.0,
                                                  name='CZ_MQ1_MQ0_U01',
                                                  nameEx='',
                                                  save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseQuanProcTomoOSInt(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                  CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.1*ns,
                                                  amp0=1.0, phi0=0,
                                                  amp1=1.0, phi1=0,
                                                  compPulseLen=7.0*ns, compPulseAmpQ0=-0.0406, compPulseAmpQ1=-0.0434, delay=0.0,
                                                  name='CZ_mQ1_mQ0_U10',
                                                  nameEx='',
                                                  save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseQuanProcTomoOSInt(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                  CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                                  amp0=1.0, phi0=0,
                                                  amp1=1.0, phi1=0,
                                                  compPulseLen=7.0*ns, compPulseAmpQ0=-0.0406, compPulseAmpQ1=-0.0873, delay=0.0,
                                                  name='CZ_MQ1_mQ0_U00',
                                                  nameEx='',
                                                  save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseQuanProcTomoOSIntBlank1(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                        CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.1*ns,
                                                        amp0=1.0, phi0=0,
                                                        amp1=1.0, phi1=0,
                                                        compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167, delay=0.0,
                                                        name='CZ_pre_QPT_U11_1_0p1',
                                                        nameEx='',
                                                        save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseQuanProcTomoOSIntBlank1(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                        CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                                        amp0=1.0, phi0=0,
                                                        amp1=1.0, phi1=0,
                                                        compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167, delay=0.0,
                                                        name='CZ_pre_QPT_U11_1_0p0',
                                                        nameEx='',
                                                        save=True, collect=True, noisy=True)

    swap21(sample, swapLen=st.arangePQ(0,600,2,ns), swapAmp=np.arange(0.02,0.8,0.001), measure=1)

def repeatGenCPHASEGateCompPulseBellTomoOSmQ1F1MQ0(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                                   delay=0.0*ns, phase=0.5*np.pi,
                                                   name='Repeat generalized CPHASE gate compensation pulse Bell TOMO OS MQ mQ1F1 MQ0',
                                                   nameEx=None, nameExFile='run00',
                                                   save=True, collect=True, noisy=True):

    CPHASEAmp = readArray('swapAmp20100530.txt')[0][13:45]
    CPHASETime = readArray('swapTime20100530.txt')[0][13:45]

#    cpaMinQ11 = readArray('cpaQubit1Fock1min20100529.txt')[0]
#    cpaMaxQ11 = readArray('cpaQubit1Fock1max20100529.txt')[0]
#
#    cpaMinQ10 = readArray('cpaQubit1Fock0min20100529.txt')[0]
#    cpaMaxQ10 = readArray('cpaQubit1Fock0max20100529.txt')[0]
#
#    cpaMinQ01 = readArray('cpaQubit0Fock1min20100529.txt')[0]
#    cpaMaxQ01 = readArray('cpaQubit0Fock1max20100529.txt')[0]

    cpaMinQ11 = readArray('cpaQubit1Fock1min20100530.txt')[:,]
    cpaMaxQ11 = readArray('cpaQubit1Fock1max20100530.txt')[:,]

    cpaMinQ10 = readArray('cpaQubit1Fock0min20100530.txt')[:,]
    cpaMaxQ10 = readArray('cpaQubit1Fock0max20100530.txt')[:,]

    cpaMinQ01 = readArray('cpaQubit0Fock1min20100530.txt')[:,]
    cpaMaxQ01 = readArray('cpaQubit0Fock1max20100530.txt')[:,]

    rhoPhi = np.zeros(len(CPHASETime))
    rhoAmp = np.zeros(len(CPHASETime))
    outFile = open('U:\\Matteo\\CPHASEPars20100530Sun\\coherencePhiAmp\\%s%s.dat' % (name, nameExFile), 'w')

    for i in range(len(CPHASETime)):

        print 'CPHASE time is %g' % CPHASETime[i]

#        pdb.set_trace()

        rhoPhi[i], rhoAmp[i] = genCPHASEGateCompPulseBellTomoOSforRep(sample, repetition=repetition, measure=measure,
                                                                      stats=stats, overshoot=overshoot,
                                                                      CPHASEAmp=CPHASEAmp[i], CPHASETime=CPHASETime[i]*ns,
                                                                      compPulseLen=compPulseLen,
                                                                      compPulseAmpQ1=cpaMinQ11[i], compPulseAmpQ0=cpaMaxQ01[i],
                                                                      delay=delay, phase=phase,
                                                                      nameEx=str(i),
                                                                      save=True, collect=True, noisy=True)

        timeNew = CPHASETime[i]

        outString = str(timeNew)+'    '+ str(rhoPhi[i])+'    '+ str(rhoAmp[i])
        print >> outFile, outString
    outFile.close()

    plt.figure(200)
    plt.plot(CPHASETime, rhoPhi, 'bs-')

    plt.figure(201)
    plt.plot(CPHASETime, rhoAmp, 'rs-')

#    return CPHASETime, rhoPhi, rhoAmp

def repeatGenCPHASEGateCompPulseBellTomoOSmQ1F1mQ0(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                                   delay=0.0*ns, phase=0.5*np.pi,
                                                   name='Repeat generalized CPHASE gate compensation pulse Bell TOMO OS MQ mQ1F1 mQ0',
                                                   nameEx=None, nameExFile='run01',
                                                   save=True, collect=True, noisy=True):

    CPHASEAmp = readArray('swapAmp20100530.txt')[0][13:45]
    CPHASETime = readArray('swapTime20100530.txt')[0][13:45]

#    cpaMinQ11 = readArray('cpaQubit1Fock1min20100529.txt')[0]
#    cpaMaxQ11 = readArray('cpaQubit1Fock1max20100529.txt')[0]
#
#    cpaMinQ10 = readArray('cpaQubit1Fock0min20100529.txt')[0]
#    cpaMaxQ10 = readArray('cpaQubit1Fock0max20100529.txt')[0]
#
#    cpaMinQ01 = readArray('cpaQubit0Fock1min20100529.txt')[0]
#    cpaMaxQ01 = readArray('cpaQubit0Fock1max20100529.txt')[0]

    cpaMinQ11 = readArray('cpaQubit1Fock1min20100530.txt')[:,]
    cpaMaxQ11 = readArray('cpaQubit1Fock1max20100530.txt')[:,]

    cpaMinQ10 = readArray('cpaQubit1Fock0min20100530.txt')[:,]
    cpaMaxQ10 = readArray('cpaQubit1Fock0max20100530.txt')[:,]

    cpaMinQ01 = readArray('cpaQubit0Fock1min20100530.txt')[:,]
    cpaMaxQ01 = readArray('cpaQubit0Fock1max20100530.txt')[:,]

    rhoPhi = np.zeros(len(CPHASETime))
    rhoAmp = np.zeros(len(CPHASETime))
    outFile = open('U:\\Matteo\\CPHASEPars20100530Sun\\coherencePhiAmp\\%s%s.dat' % (name, nameExFile), 'w')

    for i in range(len(CPHASETime)):

        print 'CPHASE time is %g' % CPHASETime[i]

#        pdb.set_trace()

        rhoPhi[i], rhoAmp[i] = genCPHASEGateCompPulseBellTomoOSforRep(sample, repetition=repetition, measure=measure,
                                                                      stats=stats, overshoot=overshoot,
                                                                      CPHASEAmp=CPHASEAmp[i], CPHASETime=CPHASETime[i]*ns,
                                                                      compPulseLen=compPulseLen,
                                                                      compPulseAmpQ1=cpaMinQ11[i], compPulseAmpQ0=cpaMinQ01[i],
                                                                      delay=delay, phase=phase,
                                                                      nameEx=str(i),
                                                                      save=True, collect=True, noisy=True)

        timeNew = CPHASETime[i]

        outString = str(timeNew)+'    '+ str(rhoPhi[i])+'    '+ str(rhoAmp[i])
        print >> outFile, outString
    outFile.close()

    plt.figure(202)
    plt.plot(CPHASETime, rhoPhi, 'bs-')

    plt.figure(203)
    plt.plot(CPHASETime, rhoAmp, 'rs-')

#    return CPHASETime, rhoPhi, rhoAmp

def repeatGenCPHASEGateCompPulseBellTomoOSMQ1F1mQ0(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                                   delay=0.0*ns, phase=0.5*np.pi,
                                                   name='Repeat generalized CPHASE gate compensation pulse Bell TOMO OS MQ MQ1F1 mQ0',
                                                   nameEx=None, nameExFile='run02',
                                                   save=True, collect=True, noisy=True):

    CPHASEAmp = readArray('swapAmp20100530.txt')[0][13:45]
    CPHASETime = readArray('swapTime20100530.txt')[0][13:45]

#    cpaMinQ11 = readArray('cpaQubit1Fock1min20100529.txt')[0]
#    cpaMaxQ11 = readArray('cpaQubit1Fock1max20100529.txt')[0]
#
#    cpaMinQ10 = readArray('cpaQubit1Fock0min20100529.txt')[0]
#    cpaMaxQ10 = readArray('cpaQubit1Fock0max20100529.txt')[0]
#
#    cpaMinQ01 = readArray('cpaQubit0Fock1min20100529.txt')[0]
#    cpaMaxQ01 = readArray('cpaQubit0Fock1max20100529.txt')[0]

    cpaMinQ11 = readArray('cpaQubit1Fock1min20100530.txt')[:,]
    cpaMaxQ11 = readArray('cpaQubit1Fock1max20100530.txt')[:,]

    cpaMinQ10 = readArray('cpaQubit1Fock0min20100530.txt')[:,]
    cpaMaxQ10 = readArray('cpaQubit1Fock0max20100530.txt')[:,]

    cpaMinQ01 = readArray('cpaQubit0Fock1min20100530.txt')[:,]
    cpaMaxQ01 = readArray('cpaQubit0Fock1max20100530.txt')[:,]

    rhoPhi = np.zeros(len(CPHASETime))
    rhoAmp = np.zeros(len(CPHASETime))
    outFile = open('U:\\Matteo\\CPHASEPars20100530Sun\\coherencePhiAmp\\%s%s.dat' % (name, nameExFile), 'w')

    for i in range(len(CPHASETime)):

        print 'CPHASE time is %g' % CPHASETime[i]

#        pdb.set_trace()

        rhoPhi[i], rhoAmp[i] = genCPHASEGateCompPulseBellTomoOSforRep(sample, repetition=repetition, measure=measure,
                                                                      stats=stats, overshoot=overshoot,
                                                                      CPHASEAmp=CPHASEAmp[i], CPHASETime=CPHASETime[i]*ns,
                                                                      compPulseLen=compPulseLen,
                                                                      compPulseAmpQ1=cpaMaxQ11[i], compPulseAmpQ0=cpaMinQ01[i],
                                                                      delay=delay, phase=phase,
                                                                      nameEx=str(i),
                                                                      save=True, collect=True, noisy=True)

        timeNew = CPHASETime[i]

        outString = str(timeNew)+'    '+ str(rhoPhi[i])+'    '+ str(rhoAmp[i])
        print >> outFile, outString
    outFile.close()

    plt.figure(204)
    plt.plot(CPHASETime, rhoPhi, 'bs-')

    plt.figure(205)
    plt.plot(CPHASETime, rhoAmp, 'rs-')

#    return CPHASETime, rhoPhi, rhoAmp

def repeatGenCPHASEGateCompPulseBellTomoOSMQ1F1MQ0(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                                   delay=0.0*ns, phase=0.5*np.pi,
                                                   name='Repeat generalized CPHASE gate compensation pulse Bell TOMO OS MQ MQ1F1 MQ0',
                                                   nameEx=None, nameExFile='run03',
                                                   save=True, collect=True, noisy=True):

    CPHASEAmp = readArray('swapAmp20100530.txt')[0][13:45]
    CPHASETime = readArray('swapTime20100530.txt')[0][13:45]

#    cpaMinQ11 = readArray('cpaQubit1Fock1min20100529.txt')[0]
#    cpaMaxQ11 = readArray('cpaQubit1Fock1max20100529.txt')[0]
#
#    cpaMinQ10 = readArray('cpaQubit1Fock0min20100529.txt')[0]
#    cpaMaxQ10 = readArray('cpaQubit1Fock0max20100529.txt')[0]
#
#    cpaMinQ01 = readArray('cpaQubit0Fock1min20100529.txt')[0]
#    cpaMaxQ01 = readArray('cpaQubit0Fock1max20100529.txt')[0]

    cpaMinQ11 = readArray('cpaQubit1Fock1min20100530.txt')[:,]
    cpaMaxQ11 = readArray('cpaQubit1Fock1max20100530.txt')[:,]

    cpaMinQ10 = readArray('cpaQubit1Fock0min20100530.txt')[:,]
    cpaMaxQ10 = readArray('cpaQubit1Fock0max20100530.txt')[:,]

    cpaMinQ01 = readArray('cpaQubit0Fock1min20100530.txt')[:,]
    cpaMaxQ01 = readArray('cpaQubit0Fock1max20100530.txt')[:,]

    rhoPhi = np.zeros(len(CPHASETime))
    rhoAmp = np.zeros(len(CPHASETime))
    outFile = open('U:\\Matteo\\CPHASEPars20100530Sun\\coherencePhiAmp\\%s%s.dat' % (name, nameExFile), 'w')

    for i in range(len(CPHASETime)):

        print 'CPHASE time is %g' % CPHASETime[i]

#        pdb.set_trace()

        rhoPhi[i], rhoAmp[i] = genCPHASEGateCompPulseBellTomoOSforRep(sample, repetition=repetition, measure=measure,
                                                                      stats=stats, overshoot=overshoot,
                                                                      CPHASEAmp=CPHASEAmp[i], CPHASETime=CPHASETime[i]*ns,
                                                                      compPulseLen=compPulseLen,
                                                                      compPulseAmpQ1=cpaMaxQ11[i], compPulseAmpQ0=cpaMaxQ01[i],
                                                                      delay=delay, phase=phase,
                                                                      nameEx=str(i),
                                                                      save=True, collect=True, noisy=True)

        timeNew = CPHASETime[i]

        outString = str(timeNew)+'    '+ str(rhoPhi[i])+'    '+ str(rhoAmp[i])
        print >> outFile, outString
    outFile.close()

    plt.figure(206)
    plt.plot(CPHASETime, rhoPhi, 'bs-')

    plt.figure(207)
    plt.plot(CPHASETime, rhoAmp, 'rs-')

#    return CPHASETime, rhoPhi, rhoAmp

def repeatGenCPHASEGateCompPulseBellTomoOSmQ1F0mQ0(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                                   delay=0.0*ns, phase=0.5*np.pi,
                                                   name='Repeat generalized CPHASE gate compensation pulse Bell TOMO OS MQ mQ1F0 mQ0',
                                                   nameEx=None, nameExFile='run04',
                                                   save=True, collect=True, noisy=True):

    CPHASEAmp = readArray('swapAmp20100530.txt')[0][13:45]
    CPHASETime = readArray('swapTime20100530.txt')[0][13:45]

#    cpaMinQ11 = readArray('cpaQubit1Fock1min20100529.txt')[0]
#    cpaMaxQ11 = readArray('cpaQubit1Fock1max20100529.txt')[0]
#
#    cpaMinQ10 = readArray('cpaQubit1Fock0min20100529.txt')[0]
#    cpaMaxQ10 = readArray('cpaQubit1Fock0max20100529.txt')[0]
#
#    cpaMinQ01 = readArray('cpaQubit0Fock1min20100529.txt')[0]
#    cpaMaxQ01 = readArray('cpaQubit0Fock1max20100529.txt')[0]

    cpaMinQ11 = readArray('cpaQubit1Fock1min20100530.txt')[:,]
    cpaMaxQ11 = readArray('cpaQubit1Fock1max20100530.txt')[:,]

    cpaMinQ10 = readArray('cpaQubit1Fock0min20100530.txt')[:,]
    cpaMaxQ10 = readArray('cpaQubit1Fock0max20100530.txt')[:,]

    cpaMinQ01 = readArray('cpaQubit0Fock1min20100530.txt')[:,]
    cpaMaxQ01 = readArray('cpaQubit0Fock1max20100530.txt')[:,]

    rhoPhi = np.zeros(len(CPHASETime))
    rhoAmp = np.zeros(len(CPHASETime))
    outFile = open('U:\\Matteo\\CPHASEPars20100530Sun\\coherencePhiAmp\\%s%s.dat' % (name, nameExFile), 'w')

    for i in range(len(CPHASETime)):

        print 'CPHASE time is %g' % CPHASETime[i]

#        pdb.set_trace()

        rhoPhi[i], rhoAmp[i] = genCPHASEGateCompPulseBellTomoOSforRep(sample, repetition=repetition, measure=measure,
                                                                      stats=stats, overshoot=overshoot,
                                                                      CPHASEAmp=CPHASEAmp[i], CPHASETime=CPHASETime[i]*ns,
                                                                      compPulseLen=compPulseLen,
                                                                      compPulseAmpQ1=cpaMinQ10[i], compPulseAmpQ0=cpaMinQ01[i],
                                                                      delay=delay, phase=phase,
                                                                      nameEx=str(i),
                                                                      save=True, collect=True, noisy=True)

        timeNew = CPHASETime[i]

        outString = str(timeNew)+'    '+ str(rhoPhi[i])+'    '+ str(rhoAmp[i])
        print >> outFile, outString
    outFile.close()

    plt.figure(208)
    plt.plot(CPHASETime, rhoPhi, 'bs-')

    plt.figure(209)
    plt.plot(CPHASETime, rhoAmp, 'rs-')

#    return CPHASETime, rhoPhi, rhoAmp

def repeatGenCPHASEGateCompPulseBellTomoOSmQ1F0MQ0(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                                   delay=0.0*ns, phase=0.5*np.pi,
                                                   name='Repeat generalized CPHASE gate compensation pulse Bell TOMO OS MQ mQ1F0 MQ0',
                                                   nameEx=None, nameExFile='run05',
                                                   save=True, collect=True, noisy=True):

#    CPHASEAmp = readArray('swapAmp20100530.txt')[0][13:45]
    CPHASEAmp = readArray('swapAmp20100612.txt')[0][2:77]
    #CPHASEAmp = readArray('swapAmp20100612.txt')[0][2:4]


#    CPHASETime = readArray('swapTime20100530.txt')[0][13:45]
    CPHASETime = readArray('swapTime20100612.txt')[0][2:77]
    #CPHASETime = readArray('swapTime20100612.txt')[0][2:4]



#    cpaMinQ11 = readArray('cpaQubit1Fock1min20100529.txt')[0]
#    cpaMaxQ11 = readArray('cpaQubit1Fock1max20100529.txt')[0]
#
#    cpaMinQ10 = readArray('cpaQubit1Fock0min20100529.txt')[0]
#    cpaMaxQ10 = readArray('cpaQubit1Fock0max20100529.txt')[0]
#
#    cpaMinQ01 = readArray('cpaQubit0Fock1min20100529.txt')[0]
#    cpaMaxQ01 = readArray('cpaQubit0Fock1max20100529.txt')[0]

    cpaMinQ11 = readArray('cpaQubit1Fock1min20100612.txt')[:,]
    cpaMaxQ11 = readArray('cpaQubit1Fock1max20100612.txt')[:,]

    cpaMinQ10 = readArray('cpaQubit1Fock0min20100612.txt')[:,]
    cpaMaxQ10 = readArray('cpaQubit1Fock0max20100612.txt')[:,]

    cpaMinQ01 = readArray('cpaQubit0Fock1min20100612.txt')[:,]
    cpaMaxQ01 = readArray('cpaQubit0Fock1max20100612.txt')[:,]

    rhoPhi = np.zeros(len(CPHASETime))
    rhoAmp = np.zeros(len(CPHASETime))
    outFile = open('U:\\Matteo\\CPHASEPars20100612SatTest\\coherencePhiAmp\\%s%s.dat' % (name, nameExFile), 'w')

    for i in range(len(CPHASETime)):

        print 'CPHASE time is %g' % CPHASETime[i]

#        pdb.set_trace()

        rhoPhi[i], rhoAmp[i] = genCPHASEGateCompPulseBellTomoOSforRep(sample, repetition=repetition, measure=measure,
                                                                      stats=stats, overshoot=overshoot,
                                                                      CPHASEAmp=CPHASEAmp[i], CPHASETime=CPHASETime[i]*ns,
                                                                      compPulseLen=compPulseLen,
                                                                      compPulseAmpQ1=cpaMinQ10[i], compPulseAmpQ0=cpaMaxQ01[i],
                                                                      delay=delay, phase=phase,
                                                                      nameEx=str(i),
                                                                      save=True, collect=True, noisy=True)

        timeNew = CPHASETime[i]

        outString = str(timeNew)+'    '+ str(rhoPhi[i])+'    '+ str(rhoAmp[i])
        print >> outFile, outString
    outFile.close()

    plt.figure(210)
    plt.plot(CPHASETime, rhoPhi, 'bs-')

    plt.figure(211)
    plt.plot(CPHASETime, rhoAmp, 'rs-')

#    return CPHASETime, rhoPhi, rhoAmp

def repeatGenCPHASEGateCompPulseBellTomoOSMQ1F0mQ0(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                                   delay=0.0*ns, phase=0.5*np.pi,
                                                   name='Repeat generalized CPHASE gate compensation pulse Bell TOMO OS MQ MQ1F0 mQ0',
                                                   nameEx=None, nameExFile='run06',
                                                   save=True, collect=True, noisy=True):

#    CPHASEAmp = readArray('swapAmp20100530.txt')[0][13:45]
    CPHASEAmp = readArray('swapAmp20100606.txt')[0][0:]
#    CPHASETime = readArray('swapTime20100530.txt')[0][13:45]
    CPHASETime = readArray('swapTime20100606.txt')[0][0:]

#    cpaMinQ11 = readArray('cpaQubit1Fock1min20100529.txt')[0]
#    cpaMaxQ11 = readArray('cpaQubit1Fock1max20100529.txt')[0]
#
#    cpaMinQ10 = readArray('cpaQubit1Fock0min20100529.txt')[0]
#    cpaMaxQ10 = readArray('cpaQubit1Fock0max20100529.txt')[0]
#
#    cpaMinQ01 = readArray('cpaQubit0Fock1min20100529.txt')[0]
#    cpaMaxQ01 = readArray('cpaQubit0Fock1max20100529.txt')[0]

    cpaMinQ11 = readArray('cpaQubit1Fock1min20100606.txt')[:,]
    cpaMaxQ11 = readArray('cpaQubit1Fock1max20100606.txt')[:,]

    cpaMinQ10 = readArray('cpaQubit1Fock0min20100606.txt')[:,]
    cpaMaxQ10 = readArray('cpaQubit1Fock0max20100606.txt')[:,]

    cpaMinQ01 = readArray('cpaQubit0Fock1min20100606.txt')[:,]
    cpaMaxQ01 = readArray('cpaQubit0Fock1max20100606.txt')[:,]

    rhoPhi = np.zeros(len(CPHASETime))
    rhoAmp = np.zeros(len(CPHASETime))
    outFile = open('U:\\Matteo\\CPHASEPars20100606Sun\\coherencePhiAmp\\%s%s.dat' % (name, nameExFile), 'w')

    for i in range(len(CPHASETime)):

        print 'CPHASE time is %g' % CPHASETime[i]

#        pdb.set_trace()

        rhoPhi[i], rhoAmp[i] = genCPHASEGateCompPulseBellTomoOSforRep(sample, repetition=repetition, measure=measure,
                                                                      stats=stats, overshoot=overshoot,
                                                                      CPHASEAmp=CPHASEAmp[i], CPHASETime=CPHASETime[i]*ns,
                                                                      compPulseLen=compPulseLen,
                                                                      compPulseAmpQ1=cpaMaxQ10[i], compPulseAmpQ0=cpaMinQ01[i],
                                                                      delay=delay, phase=phase,
                                                                      nameEx=str(i),
                                                                      save=True, collect=True, noisy=True)

        timeNew = CPHASETime[i]

        outString = str(timeNew)+'    '+ str(rhoPhi[i])+'    '+ str(rhoAmp[i])
        print >> outFile, outString
    outFile.close()

    plt.figure(212)
    plt.plot(CPHASETime, rhoPhi, 'bs-')

    plt.figure(213)
    plt.plot(CPHASETime, rhoAmp, 'rs-')

#    return CPHASETime, rhoPhi, rhoAmp

def repeatGenCPHASEGateCompPulseBellTomoOSMQ1F0MQ0(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                                   delay=0.0*ns, phase=0.5*np.pi,
                                                   name='Repeat generalized CPHASE gate compensation pulse Bell TOMO OS MQ MQ1F0 MQ0',
                                                   nameEx=None, nameExFile='run07',
                                                   save=True, collect=True, noisy=True):

#    CPHASEAmp = readArray('swapAmp20100530.txt')[0][13:45]
    # CPHASEAmp = readArray('swapAmp20100606.txt')[0][2:77]
    # CPHASEAmp = readArray('swapAmp20100606.txt')[0][0:2]
    CPHASEAmp = readArray('swapAmp20100618.txt')[0][133:155]
#    CPHASETime = readArray('swapTime20100530.txt')[0][13:45]
    # CPHASETime = readArray('swapTime20100606.txt')[0][2:77]
    # CPHASETime = readArray('swapTime20100606.txt')[0][0:2]
    CPHASETime = readArray('swapTime20100618.txt')[0][133:155]

#    cpaMinQ11 = readArray('cpaQubit1Fock1min20100529.txt')[0]
#    cpaMaxQ11 = readArray('cpaQubit1Fock1max20100529.txt')[0]
#
#    cpaMinQ10 = readArray('cpaQubit1Fock0min20100529.txt')[0]
#    cpaMaxQ10 = readArray('cpaQubit1Fock0max20100529.txt')[0]
#
#    cpaMinQ01 = readArray('cpaQubit0Fock1min20100529.txt')[0]
#    cpaMaxQ01 = readArray('cpaQubit0Fock1max20100529.txt')[0]

    cpaMinQ11 = readArray('cpaQubit1Fock1min20100618.txt')[:,]
    cpaMaxQ11 = readArray('cpaQubit1Fock1max20100618.txt')[:,]

    cpaMinQ10 = readArray('cpaQubit1Fock0min20100618.txt')[:,]
    cpaMaxQ10 = readArray('cpaQubit1Fock0max20100618.txt')[:,]

    cpaMinQ01 = readArray('cpaQubit0Fock1min20100618.txt')[:,]
    cpaMaxQ01 = readArray('cpaQubit0Fock1max20100618.txt')[:,]

    rhoPhi = np.zeros(len(CPHASETime))
    rhoAmp = np.zeros(len(CPHASETime))
    outFile = open('U:\\Matteo\\CPHASEPars20100618Fri\\coherencePhiAmp\\%s%s.dat' % (name, nameExFile), 'w')

    for i in range(len(CPHASETime)):

        print 'CPHASE time is %g' % CPHASETime[i]

#        pdb.set_trace()

        rhoPhi[i], rhoAmp[i] = genCPHASEGateCompPulseBellTomoOSforRep(sample, repetition=repetition, measure=measure,
                                                                      stats=stats, overshoot=overshoot,
                                                                      CPHASEAmp=CPHASEAmp[i], CPHASETime=CPHASETime[i]*ns,
                                                                      compPulseLen=compPulseLen,
                                                                      compPulseAmpQ1=cpaMaxQ10[i], compPulseAmpQ0=cpaMaxQ01[i],
                                                                      delay=delay, phase=phase,
                                                                      nameEx=str(i),
                                                                      save=True, collect=True, noisy=True)

        timeNew = CPHASETime[i]

        outString = str(timeNew)+'    '+ str(rhoPhi[i])+'    '+ str(rhoAmp[i])
        print >> outFile, outString
    outFile.close()

    plt.figure(214)
    plt.plot(CPHASETime, rhoPhi, 'bs-')

    plt.figure(215)
    plt.plot(CPHASETime, rhoAmp, 'rs-')

#    return CPHASETime, rhoPhi, rhoAmp

def genCPHASECompPulsesRepeatTomo20100530Sun(sample):

    repeatGenCPHASEGateCompPulseBellTomoOSmQ1F1MQ0(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                                   delay=0.0*ns, phase=0.5*np.pi,
                                                   name='Repeat generalized CPHASE gate compensation pulse Bell TOMO OS MQ mQ1F1 MQ0',
                                                   nameEx=None, nameExFile='run00',
                                                   save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseBellTomoOSmQ1F1mQ0(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                                   delay=0.0*ns, phase=0.5*np.pi,
                                                   name='Repeat generalized CPHASE gate compensation pulse Bell TOMO OS MQ mQ1F1 mQ0',
                                                   nameEx=None, nameExFile='run01',
                                                   save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseBellTomoOSMQ1F1mQ0(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                                   delay=0.0*ns, phase=0.5*np.pi,
                                                   name='Repeat generalized CPHASE gate compensation pulse Bell TOMO OS MQ MQ1F1 mQ0',
                                                   nameEx=None, nameExFile='run02',
                                                   save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseBellTomoOSMQ1F1MQ0(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                                   delay=0.0*ns, phase=0.5*np.pi,
                                                   name='Repeat generalized CPHASE gate compensation pulse Bell TOMO OS MQ MQ1F1 MQ0',
                                                   nameEx=None, nameExFile='run03',
                                                   save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseBellTomoOSmQ1F0mQ0(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                                   delay=0.0*ns, phase=0.5*np.pi,
                                                   name='Repeat generalized CPHASE gate compensation pulse Bell TOMO OS MQ mQ1F0 mQ0',
                                                   nameEx=None, nameExFile='run04',
                                                   save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseBellTomoOSmQ1F0MQ0(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                                   delay=0.0*ns, phase=0.5*np.pi,
                                                   name='Repeat generalized CPHASE gate compensation pulse Bell TOMO OS MQ mQ1F0 MQ0',
                                                   nameEx=None, nameExFile='run05',
                                                   save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseBellTomoOSMQ1F0mQ0(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                                   delay=0.0*ns, phase=0.5*np.pi,
                                                   name='Repeat generalized CPHASE gate compensation pulse Bell TOMO OS MQ MQ1F0 mQ0',
                                                   nameEx=None, nameExFile='run06',
                                                   save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseBellTomoOSMQ1F0MQ0(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                                   delay=0.0*ns, phase=0.5*np.pi,
                                                   name='Repeat generalized CPHASE gate compensation pulse Bell TOMO OS MQ MQ1F0 MQ0',
                                                   nameEx=None, nameExFile='run07',
                                                   save=True, collect=True, noisy=True)

def swapReader(dataFrame):

    # dataFrame = 139
#    session = ['','Matteo','N00N','wfr20100219','ADR01','100606n']
    session = ['','Matteo','N00N','wfr20100219','ADR01','100618']
    with labrad.connec() as cxn:
        data0 = ds.getDataset(cxn.data_vault, dataFrame, session=session)

    # update chopping bounds
    # data0 = data0[1425:18525]

    # data0 = data0[1710:18810]
    # data0 = data0[5985:14535]

    p=fp1.fitswap(data0, axis=1, return_fit=True)

    data, info = ds.columns2Matrix(data0)

    swapAmp = np.arange(info['Xmin'],info['Xmax']+info['Xstep'],info['Xstep'])

    swapFreq = np.sqrt(p[0]**2 + (np.polyval(p[1:], swapAmp))**2)

    # update date for amp and freq
    saveArray('swapAmp20100618.txt',swapAmp)
    saveArray('swapFreq20100618.txt',swapFreq)

    intTime = np.arange(info['Ymin'],info['Ymax']+info['Ystep'],info['Ystep'])
    stepTime = info['Ystep']
    swapTime = swapAmp.copy()*0.0

    # update date for freq and amp
    swapFreq = readArray('swapFreq20100618.txt')[0]
    swapAmp = readArray('swapAmp20100618.txt')[0]

    for i in range(len(swapAmp)):
        timeLoc = 1000.0/swapFreq[i]
        timeLocInd = int(timeLoc/stepTime)
        probP = data[i,timeLocInd+1]
        prob = data[i,timeLocInd]
        probM = data[i,timeLocInd-1]
        if probP>prob:
            for j in range(len(swapTime)):
                if data[i,timeLocInd+j+1]<data[i,timeLocInd+j]:
                    break
            swapTime[i] = intTime[timeLocInd+j]
        elif probM>prob:
            for j in range(len(swapTime)):
                if data[i,timeLocInd-j-1]<data[i,timeLocInd-j]:
                    break
            swapTime[i] = intTime[timeLocInd-j]
        else:
            swapTime[i] = intTime[timeLocInd]

    #figure(1000)
    #clf()
    #plot(intTime,data[i,:],'b.-')
    #plot([swapTime[i],swapTime[i]],[0,1],'k:')
    #plot([timeLoc,timeLoc],[0,1],'r:')
    #print swapAmp[i], swapFreq[i]
    #pdb.set_trace()

    # update date for time
    saveArray('swapTime20100618.txt',swapTime,format='%0.2f,')

def test20100606(sample):

    swap21(sample, swapLen=st.arangePQ(0,200,4,ns), swapAmp=np.arange(0.075,0.095,0.002), measure=1, stats=150L)

    swapReader(141)

    repeatGenCPHASECompPulseQ1Fock1OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=150L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatGenCPHASECompPulseQ1Fock0OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=150L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatGenCPHASECompPulseQ0Fock1OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=150L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatGenCPHASEGateCompPulseBellTomoOSMQ1F0MQ0(sample, repetition=10, measure=[0,1], stats=90L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                                   delay=0.0*ns, phase=0.5*np.pi,
                                                   name='Repeat generalized CPHASE gate compensation pulse Bell TOMO OS MQ MQ1F0 MQ0',
                                                   nameEx=None, nameExFile='run07',
                                                   save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseQuanProcTomoOSIntX(sample, repetition=10, measure=[0,1], stats=60L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                                   amp0=1.0, phi0=0,
                                                   amp1=1.0, phi1=0,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0850, compPulseAmpQ1=-0.0558, delay=0.0,
                                                   name='QPT_CPHASE_MQ1F0_MQ0',
                                                   nameEx='',
                                                   save=True, collect=True, noisy=True)

def ramseyScopeMod(sample, zAmp=st.r[0:0.1:0.005], measure=0, stats=900L,
                   name='Z-Pulse Ramsey Fringe Modified',
                   save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    axes = [(zAmp, 'Z-Pulse Amplitude')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, zAmp):
        dt = q['piLen']
        q.xy = eh.mix(q, eh.piHalfPulse(q, 0) + eh.piHalfPulse(q, 2*dt))
        q.z = env.rect(dt/2, dt, zAmp) + eh.measurePulse(q, 2*dt + dt/2)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

# *************
# * QFT - V00 *
# *************

def quanFourierTransQPTVer00(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                             CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                             amp0=1.0, phi0=0,
                             amp1=1.0, phi1=0.0,
                             compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167, delay=0.0*ns,
                             name='Quantum Fourier Transform QPT MQ MQ10 MQ0',
                             nameEx='',
                             save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

#    measurement = pyle.dataking.measurement.Tomo(2)
    measurement = pyle.dataking.measurement.Octomo(2)

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx, axes, measure=measurement, kw=kw)
#    dataset = sweeps.prepDataset(sample, name, axes, measure=measurement, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASEAmp == None:

         CPHASEAmp = q1.noonSwapAmpC21

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

#    ampTest0 = 0.0
#    ampTest1 = 1.0

    def func(server, curr):

         start = 0

         # initial pulse on q0
         q0.xy = eh.mix(q0, amp0*eh.piPulse(q0, start-q0['piLen']/2, phase=phi0*np.pi))
         start += q0['piLen']/2

         # Ry on q0
         q0.xy += eh.mix(q0, eh.piHalfPulse(q0, start, phase=0.5*np.pi))
         start += q0['piLen']/2

         # map q0 into rc
         q0.z = env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)
         start += sl[0]+delay-q1['piLen']/2

         # initial pulse on q1
         q1.xy = eh.mix(q1, amp1*eh.piPulse(q1, start, phase=phi1*np.pi))
         start += q1['piLen']/2
#         start += max([q0['piLen'],q1['piLen']])+2.0*ns

#         # test pulse on q0
#         q0.xy += eh.mix(q0, ampTest0*eh.piPulse(q0, start-q0['piLen']/2))
#
#         # test pulse on q1
#         q1.xy += eh.mix(q1, ampTest1*eh.piPulse(q1, start-q1['piLen']/2))
#         start += 2.0*ns

         # START QFT

         # CPHASE gate
         q1.z = env.rect(start, CPHASETime+delayCPHASE, CPHASEAmp)
         start += CPHASETime+delayCPHASE

         # map back rc into q0
         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)

         # compensation pulse for q1
         q1.z += env.rect(start, compPulseLen, compPulseAmpQ1)

         # Ry gate on q1
         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start+compPulseLen+q1['piLen']/2, phase=0.5*np.pi))

         # compensation pulse for q0
         q0.z += env.rect(start+sl[0]+delay, compPulseLen, compPulseAmpQ0)

         # END QFT

         start += sl[0]+delay+compPulseLen

         return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

#    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)
#    return result

#    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
#    rho_ideal = np.array([[0.5,0.0,0.0,0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.5,0.0,0.0,0.5]])
#    rho_ideal = np.array([[0.0,0.0,0.0,0.0],[0.0,0.5,-0.5,0.0],[0.0,-0.5,0.5,0.0],[0.0,0.0,0.0,0.0]])
#    rho_ideal = np.array([[0.5,0.0,0.0,-0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[-0.5,0.0,0.0,0.5]])
#    result = np.sum(result,axis=0)/len(repetition)
#    Qk = np.reshape(result[1:],(9,4))
#    Qk = np.reshape(result[1:],(36,4))
#    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

#    rho_cal = tomo.qst(Qk,'tomo2')
#    rho_cal = tomo.qst(Qk,'octomo2')
#    plotRhoSingle(rho_cal,figNo=100)
#    pylab.title('Exp.')

#    rhoCoherence = rho_cal[1,2]

#    rhoPhi = np.arctan2(np.imag(rhoCoherence), np.real(rhoCoherence))

#    rhoAmp = np.sqrt(np.real(rhoCoherence)**2+np.imag(rhoCoherence)**2)

#    return rhoPhi, rhoAmp

def repeatQuanFourierTransQPTVer00(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                   CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                   amp0=1.0, phi0=0,
                                   amp1=1.0, phi1=0,
                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167, delay=0.0,
                                   name='Repeat quantum Fourier Transform QPT MQ MQ10 MQ0',
                                   nameEx='',
                                   save=True, collect=True, noisy=True):

    preOps = build_prepOps(2)
    dimXi = len(preOps)

    for i in range(dimXi):

        print '*****************************'
        print 'density matrix number %g' % i
        print '*****************************'

#        pdb.set_trace()

        quanFourierTransQPTVer00(sample, repetition=repetition, measure=measure, stats=stats, overshoot=overshoot,
                                 CPHASEAmp=CPHASEAmp, CPHASETime=CPHASETime, delayCPHASE=delayCPHASE,
                                 amp0=preOps[i][0][1], phi0=preOps[i][0][2],
                                 amp1=preOps[i][1][1], phi1=preOps[i][1][2],
                                 compPulseLen=compPulseLen, compPulseAmpQ0=compPulseAmpQ0, compPulseAmpQ1=compPulseAmpQ1, delay=delay,
                                 name=name, nameEx=str(i),
                                 save=save, collect=False, noisy=noisy)

def repeatQuanFourierTransQPTVer00X(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                    CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                    amp0=1.0, phi0=0,
                                    amp1=1.0, phi1=0,
                                    compPulseLen=7.0*ns, compPulseAmpQ0=-0.0850, compPulseAmpQ1=-0.0558, delay=0.0,
                                    name='QPT_QFT_MQ1F0_MQ0',
                                    nameEx='',
                                    save=True, collect=True, noisy=True):

    #    CPHASEAmp = readArray('swapAmp20100530.txt')[0][13:45]
    CPHASEAmp = readArray('swapAmp20100618.txt')[0][133:155]
    #CPHASEAmp = readArray('swapAmp20100612.txt')[0][2:4]


    # [2:77]
    # CPHASEAmp = readArray('swapAmp20100606.txt')[0][0:2]
#    CPHASETime = readArray('swapTime20100530.txt')[0][13:45]
    CPHASETime = readArray('swapTime20100618.txt')[0][133:155]
    #CPHASETime = readArray('swapTime20100612.txt')[0][2:4]


    # CPHASETime = readArray('swapTime20100606.txt')[0][0:2]

#    cpaMinQ11 = readArray('cpaQubit1Fock1min20100529.txt')[0]
#    cpaMaxQ11 = readArray('cpaQubit1Fock1max20100529.txt')[0]
#
#    cpaMinQ10 = readArray('cpaQubit1Fock0min20100529.txt')[0]
#    cpaMaxQ10 = readArray('cpaQubit1Fock0max20100529.txt')[0]
#
#    cpaMinQ01 = readArray('cpaQubit0Fock1min20100529.txt')[0]
#    cpaMaxQ01 = readArray('cpaQubit0Fock1max20100529.txt')[0]

    cpaMinQ11 = readArray('cpaQubit1Fock1min20100618.txt')[:,]
    cpaMaxQ11 = readArray('cpaQubit1Fock1max20100618.txt')[:,]

    cpaMinQ10 = readArray('cpaQubit1Fock0min20100618.txt')[:,]
    cpaMaxQ10 = readArray('cpaQubit1Fock0max20100618.txt')[:,]

    cpaMinQ01 = readArray('cpaQubit0Fock1min20100618.txt')[:,]
    cpaMaxQ01 = readArray('cpaQubit0Fock1max20100618.txt')[:,]

    for j in range(len(CPHASETime)):

        print 'CPHASE time is %g' % CPHASETime[j]

#        pdb.set_trace()

        repeatQuanFourierTransQPTVer00(sample, repetition=repetition, measure=measure,
                                       stats=stats, overshoot=overshoot,
                                       CPHASEAmp=CPHASEAmp[j], CPHASETime=CPHASETime[j]*ns, delayCPHASE=delayCPHASE,
                                       amp0=1.0, phi0=0,
                                       amp1=1.0, phi1=0,
                                       compPulseLen=compPulseLen, compPulseAmpQ0=cpaMaxQ01[j], compPulseAmpQ1=cpaMaxQ10[j], delay=delay,
                                       name=name,
                                       nameEx=str(j),
                                       save=True, collect=True, noisy=True)

# ****************
# * QPT Hadamard *
# ****************

def piPulseMod(q, t0, phase=0, alpha=0.5, df=0.0*GHz):
    """Pi pulse using a gaussian envelope with half-derivative Y quadrature with constant detuning."""
    return rotPulseMod(q, t0, angle=np.pi, phase=phase, df=df)

def piHalfPulseMod(q, t0, phase=0, alpha=0.5):
    """Pi/2 pulse using a gaussian envelope with half-derivative Y quadrature."""
    return rotPulseMod(q, t0, angle=np.pi/2, phase=phase)

def rotPulseMod(q, t0, angle=np.pi, phase=0, alpha=0.5, df=0.0*GHz):
    """Rotation pulse using a gaussian envelope with half-derivative Y quadrature."""
    r = angle / np.pi
    delta = 2*np.pi * (q['f21'] - q['f10'])['GHz']
    x = env.gaussian(t0, w=q['piFWHM'], amp=q['piAmp']*r, phase=phase, df=df)
    y = -alpha * env.deriv(x) / delta
    return x + 1j*y

def mixMod(q, seq, freq='f10', df=0.0*GHz):
    """Apply microwave mixing to a sequence.

    This mixes to a particular frequency from the carrier frequency.
    Also, adjusts the microwave phase according to the phase calibration.
    """
    if isinstance(freq, str):
        freq = q[freq]+df
    return env.mix(seq, freq - q['fc']) * np.exp(1j*q['uwavePhase'])

def rabihighZ(sample, amplitude=st.r[0.0:1.5:0.05], df=0.0*GHz, measureDelay=None, measure=0, stats=1500L,
              name='Rabi-pulse height HD with z MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    if amplitude is None: amplitude = q['piAmp']
    if measureDelay is None: measureDelay = q['piLen'] # /2.0

    axes = [(amplitude, 'pulse height'),
            (measureDelay, 'measure delay')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, amp, delay):
        q['piAmp'] = amp
        q.xy = eh.mix(q, eh.piPulseHD(q, 0), freq=q['f10']+df)
        q.z = env.rect(-q['piLen']/2, q['piLen'], q['piAmpZ']/np.sqrt(2))
        q.z += eh.measurePulse(q, delay)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)

def HadamardQPT(sample, repetition=10, measure=[0,1], stats=1500L, df=0.0*GHz, alpha=0.5,
                amp0=1.0, phi0=0,
                amp1=1.0, phi1=0.0,
                name='Hadamard QPT MQ',
                nameEx='',
                save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

#    measurement = pyle.dataking.measurement.Tomo(2)
    measurement = pyle.dataking.measurement.Octomo(2)

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx, axes, measure=measurement, kw=kw)
#    dataset = sweeps.prepDataset(sample, name, axes, measure=measurement, kw=kw)

#    ampTest0 = 0.0
#    ampTest1 = 1.0

    def func(server, curr):

         start = 0

         # initial pulse on q0 and q1
         q0.xy = eh.mix(q0, amp0*eh.piPulse(q0, start-q0['piLen']/2, phase=phi0*np.pi))
         q1.xy = eh.mix(q1, amp1*eh.piPulse(q1, start-q1['piLen']/2, phase=phi1*np.pi))
         start += q1['piLen']/2

         # Hadamard gate on q1
         q1.xy += eh.mix(q1, (1/np.sqrt(2))*eh.piPulse(q1, start, phase=0.0*np.pi, alpha=alpha), freq=q1['f10']+df)
         # -1.0*0.031215645944890014*GHz
         q1.z = env.rect(start-q1['piLen']/2, q1['piLen'], q1['piAmpZ']/np.sqrt(2))
         start += q1['piLen']/2

         return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

#    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)
#    return result

#    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
#    rho_ideal = np.array([[0.5,0.0,0.0,0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.5,0.0,0.0,0.5]])
#    rho_ideal = np.array([[0.0,0.0,0.0,0.0],[0.0,0.5,-0.5,0.0],[0.0,-0.5,0.5,0.0],[0.0,0.0,0.0,0.0]])
#    rho_ideal = np.array([[0.5,0.0,0.0,-0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[-0.5,0.0,0.0,0.5]])
#    result = np.sum(result,axis=0)/len(repetition)
#    Qk = np.reshape(result[1:],(9,4))
#    Qk = np.reshape(result[1:],(36,4))
#    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

#    rho_cal = tomo.qst(Qk,'tomo2')
#    rho_cal = tomo.qst(Qk,'octomo2')
#    plotRhoSingle(rho_cal,figNo=100)
#    pylab.title('Exp.')

#    rhoCoherence = rho_cal[1,2]

#    rhoPhi = np.arctan2(np.imag(rhoCoherence), np.real(rhoCoherence))

#    rhoAmp = np.sqrt(np.real(rhoCoherence)**2+np.imag(rhoCoherence)**2)

#    return rhoPhi, rhoAmp

def repeatHadamardQPT(sample, repetition=10, measure=[0,1], stats=1500L, df=0.0*GHz, alpha=0.5,
                      amp0=1.0, phi0=0,
                      amp1=1.0, phi1=0,
                      name='Repeat Hadamard QPT MQ',
                      nameEx='',
                      save=True, collect=True, noisy=True):

    preOps = build_prepOps(2)
    dimXi = len(preOps)

    for i in range(dimXi):

        print '*****************************'
        print 'density matrix number %g' % i
        print '*****************************'

#        pdb.set_trace()

        HadamardQPT(sample, repetition=repetition, measure=measure, stats=stats, df=df, alpha=alpha,
                    amp0=preOps[i][0][1], phi0=preOps[i][0][2],
                    amp1=preOps[i][1][1], phi1=preOps[i][1][2],
                    name=name, nameEx=str(i),
                    save=save, collect=False, noisy=noisy)

# *************
# * QFT - V01 *
# *************

def quanFourierTransQPTVer01(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                             CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                             amp0=1.0, phi0=0,
                             amp1=1.0, phi1=0.0,
                             compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167, delay=0.0*ns,
                             name='Quantum Fourier Transform H QPT MQ MQ10 MQ0',
                             nameEx='',
                             save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

#    measurement = pyle.dataking.measurement.Tomo(2)
    measurement = pyle.dataking.measurement.Octomo(2)

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx, axes, measure=measurement, kw=kw)
#    dataset = sweeps.prepDataset(sample, name, axes, measure=measurement, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASEAmp == None:

         CPHASEAmp = q1.noonSwapAmpC21

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

#    ampTest0 = 0.0
#    ampTest1 = 1.0

    def func(server, curr):

         start = 0

         # initial pulse on q0
         q0.xy = eh.mix(q0, amp0*eh.piPulse(q0, start-q0['piLen']/2, phase=phi0*np.pi))
         start += q0['piLen']/2

         # Hadamard gate on q0
         q0.xy += eh.mix(q0, (1.0/np.sqrt(2))*eh.piPulse(q0, start, phase=0.0*np.pi))
         q0.z = env.rect(start-q0['piLen']/2, q0['piLen'], q0['piAmpZ']/np.sqrt(2))
         start += q0['piLen']/2

         # map q0 into rc
         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)
         start += sl[0]+delay-q1['piLen']/2

         # initial pulse on q1
         q1.xy = eh.mix(q1, amp1*eh.piPulse(q1, start, phase=phi1*np.pi))
         start += q1['piLen']/2
#         start += max([q0['piLen'],q1['piLen']])+2.0*ns

#         # test pulse on q0
#         q0.xy += eh.mix(q0, ampTest0*eh.piPulse(q0, start-q0['piLen']/2))
#
#         # test pulse on q1
#         q1.xy += eh.mix(q1, ampTest1*eh.piPulse(q1, start-q1['piLen']/2))
#         start += 2.0*ns

         # START QFT

         # CPHASE gate
         q1.z = env.rect(start, CPHASETime+delayCPHASE, CPHASEAmp)
         start += CPHASETime+delayCPHASE

         # map back rc into q0
         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[0]*overshoot)

         # compensation pulse for q1
         q1.z += env.rect(start, compPulseLen, compPulseAmpQ1)

         # Hadamard gate gate on q1
         q1.xy += eh.mix(q1, (1.0/np.sqrt(2))*eh.piPulse(q1, start+compPulseLen+q1['piLen']/2, phase=0.0*np.pi))
         q1.z += env.rect(start+compPulseLen, q1['piLen'], q1['piAmpZ']/np.sqrt(2))

         # compensation pulse for q0
         q0.z += env.rect(start+sl[0]+delay, compPulseLen, compPulseAmpQ0)

         # END QFT

         start += sl[0]+delay+compPulseLen

         return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

#    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)
#    return result

#    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
#    rho_ideal = np.array([[0.5,0.0,0.0,0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.5,0.0,0.0,0.5]])
#    rho_ideal = np.array([[0.0,0.0,0.0,0.0],[0.0,0.5,-0.5,0.0],[0.0,-0.5,0.5,0.0],[0.0,0.0,0.0,0.0]])
#    rho_ideal = np.array([[0.5,0.0,0.0,-0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[-0.5,0.0,0.0,0.5]])
#    result = np.sum(result,axis=0)/len(repetition)
#    Qk = np.reshape(result[1:],(9,4))
#    Qk = np.reshape(result[1:],(36,4))
#    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

#    rho_cal = tomo.qst(Qk,'tomo2')
#    rho_cal = tomo.qst(Qk,'octomo2')
#    plotRhoSingle(rho_cal,figNo=100)
#    pylab.title('Exp.')

#    rhoCoherence = rho_cal[1,2]

#    rhoPhi = np.arctan2(np.imag(rhoCoherence), np.real(rhoCoherence))

#    rhoAmp = np.sqrt(np.real(rhoCoherence)**2+np.imag(rhoCoherence)**2)

#    return rhoPhi, rhoAmp

def repeatQuanFourierTransQPTVer01(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                   CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                   amp0=1.0, phi0=0,
                                   amp1=1.0, phi1=0,
                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167, delay=0.0,
                                   name='Repeat quantum Fourier H Transform QPT MQ MQ10 MQ0',
                                   nameEx='',
                                   save=True, collect=True, noisy=True):

    preOps = build_prepOps(2)
    dimXi = len(preOps)

    for i in range(dimXi):

        print '*****************************'
        print 'density matrix number %g' % i
        print '*****************************'

#        pdb.set_trace()

        quanFourierTransQPTVer01(sample, repetition=repetition, measure=measure, stats=stats, overshoot=overshoot,
                                 CPHASEAmp=CPHASEAmp, CPHASETime=CPHASETime, delayCPHASE=delayCPHASE,
                                 amp0=preOps[i][0][1], phi0=preOps[i][0][2],
                                 amp1=preOps[i][1][1], phi1=preOps[i][1][2],
                                 compPulseLen=compPulseLen, compPulseAmpQ0=compPulseAmpQ0, compPulseAmpQ1=compPulseAmpQ1, delay=delay,
                                 name=name, nameEx=str(i),
                                 save=save, collect=False, noisy=noisy)

def repeatQuanFourierTransQPTVer01X(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                    CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                    amp0=1.0, phi0=0,
                                    amp1=1.0, phi1=0,
                                    compPulseLen=7.0*ns, compPulseAmpQ0=-0.0850, compPulseAmpQ1=-0.0558, delay=0.0,
                                    name='QPT_QFTH_MQ1F0_MQ0',
                                    nameEx='',
                                    save=True, collect=True, noisy=True):

    #    CPHASEAmp = readArray('swapAmp20100530.txt')[0][13:45]
    CPHASEAmp = readArray('swapAmp20100618.txt')[0][133:155]
    #CPHASEAmp = readArray('swapAmp20100612.txt')[0][2:4]


    # [2:77]
    # CPHASEAmp = readArray('swapAmp20100606.txt')[0][0:2]
#    CPHASETime = readArray('swapTime20100530.txt')[0][13:45]
    CPHASETime = readArray('swapTime20100618.txt')[0][133:155]
    #CPHASETime = readArray('swapTime20100612.txt')[0][2:4]


    # CPHASETime = readArray('swapTime20100606.txt')[0][0:2]

#    cpaMinQ11 = readArray('cpaQubit1Fock1min20100529.txt')[0]
#    cpaMaxQ11 = readArray('cpaQubit1Fock1max20100529.txt')[0]
#
#    cpaMinQ10 = readArray('cpaQubit1Fock0min20100529.txt')[0]
#    cpaMaxQ10 = readArray('cpaQubit1Fock0max20100529.txt')[0]
#
#    cpaMinQ01 = readArray('cpaQubit0Fock1min20100529.txt')[0]
#    cpaMaxQ01 = readArray('cpaQubit0Fock1max20100529.txt')[0]

    cpaMinQ11 = readArray('cpaQubit1Fock1min20100618.txt')[:,]
    cpaMaxQ11 = readArray('cpaQubit1Fock1max20100618.txt')[:,]

    cpaMinQ10 = readArray('cpaQubit1Fock0min20100618.txt')[:,]
    cpaMaxQ10 = readArray('cpaQubit1Fock0max20100618.txt')[:,]

    cpaMinQ01 = readArray('cpaQubit0Fock1min20100618.txt')[:,]
    cpaMaxQ01 = readArray('cpaQubit0Fock1max20100618.txt')[:,]

    for j in range(len(CPHASETime)):

        print 'CPHASE time is %g' % CPHASETime[j]

#        pdb.set_trace()

        repeatQuanFourierTransQPTVer01(sample, repetition=repetition, measure=measure,
                                       stats=stats, overshoot=overshoot,
                                       CPHASEAmp=CPHASEAmp[j], CPHASETime=CPHASETime[j]*ns, delayCPHASE=delayCPHASE,
                                       amp0=1.0, phi0=0,
                                       amp1=1.0, phi1=0,
                                       compPulseLen=compPulseLen, compPulseAmpQ0=cpaMaxQ01[j], compPulseAmpQ1=cpaMaxQ10[j], delay=delay,
                                       name=name,
                                       nameEx=str(j),
                                       save=True, collect=True, noisy=True)

def repeatGenCPHASEGateCompPulseQuanProcTomoOSIntX(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                                   amp0=1.0, phi0=0,
                                                   amp1=1.0, phi1=0,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0850, compPulseAmpQ1=-0.0558, delay=0.0,
                                                   name='QPT_CPHASE_mQ1F0_MQ0',
                                                   nameEx='',
                                                   save=True, collect=True, noisy=True):

    #    CPHASEAmp = readArray('swapAmp20100530.txt')[0][13:45]
    CPHASEAmp = readArray('swapAmp20100612.txt')[0][2:77]
    #CPHASEAmp = readArray('swapAmp20100612.txt')[0][2:4]


    # [2:77]
    # CPHASEAmp = readArray('swapAmp20100606.txt')[0][0:2]
#    CPHASETime = readArray('swapTime20100530.txt')[0][13:45]
    CPHASETime = readArray('swapTime20100612.txt')[0][2:77]
    #CPHASETime = readArray('swapTime20100612.txt')[0][2:4]


    # CPHASETime = readArray('swapTime20100606.txt')[0][0:2]

#    cpaMinQ11 = readArray('cpaQubit1Fock1min20100529.txt')[0]
#    cpaMaxQ11 = readArray('cpaQubit1Fock1max20100529.txt')[0]
#
#    cpaMinQ10 = readArray('cpaQubit1Fock0min20100529.txt')[0]
#    cpaMaxQ10 = readArray('cpaQubit1Fock0max20100529.txt')[0]
#
#    cpaMinQ01 = readArray('cpaQubit0Fock1min20100529.txt')[0]
#    cpaMaxQ01 = readArray('cpaQubit0Fock1max20100529.txt')[0]

    cpaMinQ11 = readArray('cpaQubit1Fock1min20100612.txt')[:,]
    cpaMaxQ11 = readArray('cpaQubit1Fock1max20100612.txt')[:,]

    cpaMinQ10 = readArray('cpaQubit1Fock0min20100612.txt')[:,]
    cpaMaxQ10 = readArray('cpaQubit1Fock0max20100612.txt')[:,]

    cpaMinQ01 = readArray('cpaQubit0Fock1min20100612.txt')[:,]
    cpaMaxQ01 = readArray('cpaQubit0Fock1max20100612.txt')[:,]

    for j in range(len(CPHASETime)):

        print 'CPHASE time is %g' % CPHASETime[j]

#        pdb.set_trace()

        repeatGenCPHASEGateCompPulseQuanProcTomoOSInt(sample, repetition=repetition, measure=measure,
                                                      stats=stats, overshoot=overshoot,
                                                      CPHASEAmp=CPHASEAmp[j], CPHASETime=CPHASETime[j]*ns, delayCPHASE=delayCPHASE,
                                                      amp0=1.0, phi0=0,
                                                      amp1=1.0, phi1=0,
                                                      compPulseLen=compPulseLen, compPulseAmpQ0=cpaMaxQ01[j], compPulseAmpQ1=cpaMinQ10[j], delay=delay,
                                                      name=name,
                                                      nameEx=str(j),
                                                      save=True, collect=True, noisy=True)

def CPHASEGateQSTandQPT20100606(sample):

    swap21(sample, swapLen=st.arangePQ(0,235,1,ns), swapAmp=np.arange(0.070,0.100,0.0002), measure=1, stats=1200L)

    # update dataFrame
    swapReader(20)

    repeatGenCPHASECompPulseQ1Fock1OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=900L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatGenCPHASECompPulseQ1Fock0OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=900L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatGenCPHASECompPulseQ0Fock1OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=900L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatGenCPHASEGateCompPulseBellTomoOSMQ1F0MQ0(sample, repetition=10, measure=[0,1], stats=600L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                                   delay=0.0*ns, phase=0.5*np.pi,
                                                   name='Repeat generalized CPHASE gate compensation pulse Bell TOMO OS MQ MQ1F0 MQ0',
                                                   nameEx=None, nameExFile='run07',
                                                   save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseQuanProcTomoOSIntX(sample, repetition=10, measure=[0,1], stats=150L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                                   amp0=1.0, phi0=0,
                                                   amp1=1.0, phi1=0,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0850, compPulseAmpQ1=-0.0558, delay=0.0,
                                                   name='QPT_CPHASE_MQ1F0_MQ0',
                                                   nameEx='',
                                                   save=True, collect=True, noisy=True)

#    repeatGenCPHASEGateCompPulseBellTomoOSMQ1F0mQ0(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
#                                                   CPHASEAmp=None, CPHASETime=None*ns,
#                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
#                                                   delay=0.0*ns, phase=0.5*np.pi,
#                                                   name='Repeat generalized CPHASE gate compensation pulse Bell TOMO OS MQ MQ1F0 mQ0',
#                                                   nameEx=None, nameExFile='run06',
#                                                   save=True, collect=True, noisy=True)
#
#    www
#
#    repeatGenCPHASEGateCompPulseBellTomoOSmQ1F0MQ0(sample, repetition=10, measure=[0,1], stats=1500L, overshoot=True,
#                                                   CPHASEAmp=None, CPHASETime=None*ns,
#                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
#                                                   delay=0.0*ns, phase=0.5*np.pi,
#                                                   name='Repeat generalized CPHASE gate compensation pulse Bell TOMO OS MQ mQ1F0 MQ0',
#                                                   nameEx=None, nameExFile='run05',
#                                                   save=True, collect=True, noisy=True)
#
#    www

def CPHASEGateQSTandQPT20100611(sample):

    swap21(sample, swapLen=st.arangePQ(0,235,1,ns), swapAmp=np.arange(0.070,0.100,0.0002), measure=1, stats=1200L)

    # update dataFrame
    swapReader(29)

    repeatGenCPHASECompPulseQ1Fock1OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=900L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatGenCPHASECompPulseQ1Fock0OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=900L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatGenCPHASECompPulseQ0Fock1OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=900L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatGenCPHASEGateCompPulseBellTomoOSmQ1F0MQ0(sample, repetition=10, measure=[0,1], stats=600L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                                   delay=0.0*ns, phase=0.5*np.pi,
                                                   name='Repeat generalized CPHASE gate compensation pulse Bell TOMO OS MQ mQ1F0 MQ0',
                                                   nameEx=None, nameExFile='run05',
                                                   save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseQuanProcTomoOSIntX(sample, repetition=10, measure=[0,1], stats=150L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                                   amp0=1.0, phi0=0,
                                                   amp1=1.0, phi1=0,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0850, compPulseAmpQ1=-0.0558, delay=0.0,
                                                   name='QPT_CPHASE_mQ1F0_MQ0',
                                                   nameEx='',
                                                   save=True, collect=True, noisy=True)

def test20100612(sample):

    swap21(sample, swapLen=st.arangePQ(0,200,4,ns), swapAmp=np.arange(0.075,0.095,0.002), measure=1, stats=90L)

    swapReader(2)

    repeatGenCPHASECompPulseQ1Fock1OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=60L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatGenCPHASECompPulseQ1Fock0OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=60L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatGenCPHASECompPulseQ0Fock1OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=60L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatGenCPHASEGateCompPulseBellTomoOSmQ1F0MQ0(sample, repetition=10, measure=[0,1], stats=30L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                                   delay=0.0*ns, phase=0.5*np.pi,
                                                   name='Repeat generalized CPHASE gate compensation pulse Bell TOMO OS MQ mQ1F0 MQ0',
                                                   nameEx=None, nameExFile='run05',
                                                   save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseQuanProcTomoOSIntX(sample, repetition=10, measure=[0,1], stats=30L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                                   amp0=1.0, phi0=0,
                                                   amp1=1.0, phi1=0,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0850, compPulseAmpQ1=-0.0558, delay=0.0,
                                                   name='QPT_CPHASE_mQ1F0_MQ0',
                                                   nameEx='',
                                                   save=True, collect=True, noisy=True)

def CPHASEGateQSTandQPT20100612(sample):

    swap21(sample, swapLen=st.arangePQ(0,235,1,ns), swapAmp=np.arange(0.070,0.100,0.0002), measure=1, stats=1200L)

    # update dataFrame
    swapReader(60)

    repeatGenCPHASECompPulseQ1Fock1OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=900L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatGenCPHASECompPulseQ1Fock0OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=900L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatGenCPHASECompPulseQ0Fock1OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=900L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatGenCPHASEGateCompPulseBellTomoOSmQ1F0MQ0(sample, repetition=10, measure=[0,1], stats=600L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                                   delay=0.0*ns, phase=0.5*np.pi,
                                                   name='Repeat generalized CPHASE gate compensation pulse Bell TOMO OS MQ mQ1F0 MQ0',
                                                   nameEx=None, nameExFile='run05',
                                                   save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseQuanProcTomoOSIntX(sample, repetition=10, measure=[0,1], stats=150L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                                   amp0=1.0, phi0=0,
                                                   amp1=1.0, phi1=0,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0850, compPulseAmpQ1=-0.0558, delay=0.0,
                                                   name='QPT_CPHASE_mQ1F0_MQ0',
                                                   nameEx='',
                                                   save=True, collect=True, noisy=True)

def quanFourierTransQPT20100615(sample):

    swap21(sample, swapLen=st.arangePQ(0,235,1,ns), swapAmp=np.arange(0.070,0.100,0.0002), measure=1, stats=1200L)

    # update dataFrame
    swapReader(20)

    repeatGenCPHASECompPulseQ1Fock1OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=900L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatGenCPHASECompPulseQ1Fock0OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=900L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatGenCPHASECompPulseQ0Fock1OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=900L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatQuanFourierTransQPTVer01X(sample, repetition=10, measure=[0,1], stats=150L, overshoot=True,
                                    CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                    amp0=1.0, phi0=0,
                                    amp1=1.0, phi1=0,
                                    compPulseLen=7.0*ns, compPulseAmpQ0=-0.0850, compPulseAmpQ1=-0.0558, delay=0.0,
                                    name='QPT_QFTH_MQ1F0_MQ0',
                                    nameEx='',
                                    save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseBellTomoOSMQ1F0MQ0(sample, repetition=10, measure=[0,1], stats=450L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                                   delay=0.0*ns, phase=0.5*np.pi,
                                                   name='Repeat generalized CPHASE gate compensation pulse Bell TOMO OS MQ MQ1F0 MQ0',
                                                   nameEx=None, nameExFile='run07',
                                                   save=True, collect=True, noisy=True)

def quanFourierTransQPT20100615Part2(sample):

    repeatQuanFourierTransQPTVer01X(sample, repetition=10, measure=[0,1], stats=150L, overshoot=True,
                                    CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                    amp0=1.0, phi0=0,
                                    amp1=1.0, phi1=0,
                                    compPulseLen=7.0*ns, compPulseAmpQ0=-0.0850, compPulseAmpQ1=-0.0558, delay=0.0,
                                    name='QPT_QFTH_MQ1F0_MQ0',
                                    nameEx='',
                                    save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseBellTomoOSMQ1F0MQ0(sample, repetition=10, measure=[0,1], stats=450L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                                   delay=0.0*ns, phase=0.5*np.pi,
                                                   name='Repeat generalized CPHASE gate compensation pulse Bell TOMO OS MQ MQ1F0 MQ0',
                                                   nameEx=None, nameExFile='run07',
                                                   save=True, collect=True, noisy=True)

def quanFourierTransQPT20100616(sample):

    swap21(sample, swapLen=st.arangePQ(0,235,1,ns), swapAmp=np.arange(0.070,0.100,0.0002), measure=1, stats=1200L)

    # update dataFrame
    swapReader(20)

    repeatGenCPHASECompPulseQ1Fock1OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=900L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatGenCPHASECompPulseQ1Fock0OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=900L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatGenCPHASECompPulseQ0Fock1OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=900L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatQuanFourierTransQPTVer00X(sample, repetition=10, measure=[0,1], stats=150L, overshoot=True,
                                    CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                    amp0=1.0, phi0=0,
                                    amp1=1.0, phi1=0,
                                    compPulseLen=7.0*ns, compPulseAmpQ0=-0.0850, compPulseAmpQ1=-0.0558, delay=0.0,
                                    name='QPT_QFT_MQ1F0_MQ0',
                                    nameEx='',
                                    save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseBellTomoOSMQ1F0MQ0(sample, repetition=10, measure=[0,1], stats=450L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                                   delay=0.0*ns, phase=0.5*np.pi,
                                                   name='Repeat generalized CPHASE gate compensation pulse Bell TOMO OS MQ MQ1F0 MQ0',
                                                   nameEx=None, nameExFile='run07',
                                                   save=True, collect=True, noisy=True)

def quanFourierTransQPT20100617(sample):

    swap21(sample, swapLen=st.arangePQ(0,180,1,ns), swapAmp=np.arange(0.060,0.130,0.0002), measure=1, stats=900L)

    # update dataFrame
    swapReader(31)

    repeatGenCPHASECompPulseQ1Fock1OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=450L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatGenCPHASECompPulseQ1Fock0OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=450L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatGenCPHASECompPulseQ0Fock1OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=450L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatQuanFourierTransQPTVer01X(sample, repetition=10, measure=[0,1], stats=150L, overshoot=True,
                                    CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                    amp0=1.0, phi0=0,
                                    amp1=1.0, phi1=0,
                                    compPulseLen=7.0*ns, compPulseAmpQ0=-0.0850, compPulseAmpQ1=-0.0558, delay=0.0,
                                    name='QPT_QFTH_MQ1F0_MQ0',
                                    nameEx='',
                                    save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseQuanProcTomoOSIntBlank2(sample, repetition=10, measure=[0,1], stats=150L, overshoot=True,
                                                        CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                                        amp0=1.0, phi0=0,
                                                        amp1=1.0, phi1=0,
                                                        compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167, delay=0.0,
                                                        name='CPHASE_pre_QPT',
                                                        nameEx='',
                                                        save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseBellTomoOSMQ1F0MQ0(sample, repetition=10, measure=[0,1], stats=300L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                                   delay=0.0*ns, phase=0.5*np.pi,
                                                   name='Repeat generalized CPHASE gate compensation pulse Bell TOMO OS MQ MQ1F0 MQ0',
                                                   nameEx=None, nameExFile='run07',
                                                   save=True, collect=True, noisy=True)

    repeatQuanFourierTransQPTVer00X(sample, repetition=10, measure=[0,1], stats=150L, overshoot=True,
                                    CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                    amp0=1.0, phi0=0,
                                    amp1=1.0, phi1=0,
                                    compPulseLen=7.0*ns, compPulseAmpQ0=-0.0850, compPulseAmpQ1=-0.0558, delay=0.0,
                                    name='QPT_QFT_MQ1F0_MQ0',
                                    nameEx='',
                                    save=True, collect=True, noisy=True)

def quanFourierTransQPT20100618(sample):

    swap21(sample, swapLen=st.arangePQ(0,180,1,ns), swapAmp=np.arange(0.060,0.130,0.0002), measure=1, stats=900L)

    # update dataFrame
    swapReader(25)

    repeatGenCPHASECompPulseQ1Fock1OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=450L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatGenCPHASECompPulseQ1Fock0OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=450L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatGenCPHASECompPulseQ0Fock1OS(sample, compAmp=np.arange(-0.15,0.0,0.001), measure=0, stats=450L,
                                      CPHASETime=None*ns, CPHASEAmp=None, compPulseLen=7.0*ns,
                                      delay=0.0*ns, phase=0, overshoot=True, save=True)

    repeatQuanFourierTransQPTVer01X(sample, repetition=10, measure=[0,1], stats=150L, overshoot=True,
                                    CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                    amp0=1.0, phi0=0,
                                    amp1=1.0, phi1=0,
                                    compPulseLen=7.0*ns, compPulseAmpQ0=-0.0850, compPulseAmpQ1=-0.0558, delay=0.0,
                                    name='QPT_QFTH_MQ1F0_MQ0',
                                    nameEx='',
                                    save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseQuanProcTomoOSIntBlank2(sample, repetition=10, measure=[0,1], stats=150L, overshoot=True,
                                                        CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                                        amp0=1.0, phi0=0,
                                                        amp1=1.0, phi1=0,
                                                        compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167, delay=0.0,
                                                        name='CPHASE_pre_QPT',
                                                        nameEx='',
                                                        save=True, collect=True, noisy=True)

    repeatGenCPHASEGateCompPulseBellTomoOSMQ1F0MQ0(sample, repetition=10, measure=[0,1], stats=300L, overshoot=True,
                                                   CPHASEAmp=None, CPHASETime=None*ns,
                                                   compPulseLen=7.0*ns, compPulseAmpQ0=-0.0351, compPulseAmpQ1=-0.0167,
                                                   delay=0.0*ns, phase=0.5*np.pi,
                                                   name='Repeat generalized CPHASE gate compensation pulse Bell TOMO OS MQ MQ1F0 MQ0',
                                                   nameEx=None, nameExFile='run07',
                                                   save=True, collect=True, noisy=True)

    repeatQuanFourierTransQPTVer00X(sample, repetition=10, measure=[0,1], stats=150L, overshoot=True,
                                    CPHASEAmp=None, CPHASETime=None*ns, delayCPHASE=0.0*ns,
                                    amp0=1.0, phi0=0,
                                    amp1=1.0, phi1=0,
                                    compPulseLen=7.0*ns, compPulseAmpQ0=-0.0850, compPulseAmpQ1=-0.0558, delay=0.0,
                                    name='QPT_QFT_MQ1F0_MQ0',
                                    nameEx='',
                                    save=True, collect=True, noisy=True)

def multiPhotonCZCompPulseQ11(sample, compAmp11=np.arange(-0.15,0.1,0.001), measure=0, stats=1500L,
                              n=1, compPulseLen=7.0*ns, phase=0, loadReset=False, overshoot=False,
                              name='Multi photon CZ gate - compAmp11', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(compAmp11, 'compensation pulse amplitude 11')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+'_n'+str(n), axes, measure=measure, kw=kw)

    sLen = q0['noonSwapLenCs']

    CZAmp = q1.noonSwapAmpC21

    multiPhotonCZTime = 2.0*q1.noonSwapLenC21s[n-1]

    def func(server, curr):

        ph = phase
        start = 0

        q0.xy = env.NOTHING
        q0.z = env.NOTHING

        for i in range(n-1):

            q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
            start += q0['piLen']/2
            q0.z += env.rect(start, sLen[i], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[i]*overshoot)
            start += sLen[i]+q0['piLen']/2

        q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2
        q0.z += env.rect(start, sLen[n-1], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[n-1]*overshoot)
        start += sLen[n-1]+2.0*ns
        # prepare rc in Fock state n

        if loadReset:

            q0.z += env.rect(start, q0.noonSwapLenR1s[0], q0.noonSwapAmpR1[0])
            start += q0.noonSwapLenR1s[0]
            # reset q0

        start += q1['piLen']/2

        # first Ramsey pulse on q1
        q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
        start += q1['piLen']/2

        # START MULTI-PHOTON CZ GATE

        # CZ gate
        q1.z = env.rect(start, multiPhotonCZTime, CZAmp)
        start += multiPhotonCZTime

        # map one excitation of coupling resonator into q0
        q0.z += env.rect(start, sLen[n-1], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[n-1]*overshoot)

        # compensation pulse for q1
        q1.z += env.rect(start, compPulseLen, curr)
        start += compPulseLen+q1['piLen']/2

        # END MULTI-PHOTON CZ GATE

        # second Ramsey pulse on q1
        q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
        start += q1['piLen']/2

        # measure pulse and readout q1
        q1.z += eh.measurePulse(q1, start)
        q1['readout'] = True

        return runQubits(server, qubits, stats=stats, probs=[1])

    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)

def multiPhotonCZCompPulseQ10(sample, compAmp10=np.arange(-0.15,0.1,0.001), measure=0, stats=1500L,
                              n=1, compPulseLen=7.0*ns, phase=0, overshoot=False,
                              name='Multi photon CZ gate - compAmp10', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(compAmp10, 'compensation pulse amplitude 10')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+'_n'+str(n), axes, measure=measure, kw=kw)

    sLen = q0['noonSwapLenCs']

    CZAmp = q1.noonSwapAmpC21

    multiPhotonCZTime = 2.0*q1.noonSwapLenC21s[n-1]

    def func(server, curr):

        ph = phase
        start = 0

        # first Ramsey pulse on q1
        q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
        start += q1['piLen']/2

        # START MULTI-PHOTON CZ GATE

        # CZ gate
        q1.z = env.rect(start, multiPhotonCZTime, CZAmp)
        start += multiPhotonCZTime

        # map one excitation of rc into q0
        q0.z = env.rect(start, sLen[n-1], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[n-1]*overshoot)

        # compensation pulse for q1
        q1.z += env.rect(start, compPulseLen, curr)
        start += compPulseLen+q1['piLen']/2

        # END MULTI-PHOTON CZ GATE

        # second Ramsey pulse on q1
        q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
        start += q1['piLen']/2

        # measure pulse and readout q1
        q1.z += eh.measurePulse(q1, start)
        q1['readout'] = True

        return runQubits(server, qubits, stats=stats, probs=[1])

    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)

def multiPhotonCZCompPulseQ0(sample, compAmp0=np.arange(-0.15,0.1,0.001), measure=0, stats=1500L,
                             n=1, compPulseLen=7.0*ns, compPulseAmpQ1=0.0, phase=0, loadReset=False, overshoot=False,
                             name='Multi photon CZ gate - compAmp0',
                             save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(compAmp0, 'compensation pulse amplitude 0')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+'_n'+str(n), axes, measure=measure, kw=kw)

    sLen = q0['noonSwapLenCs']

    CZAmp = q1.noonSwapAmpC21

    multiPhotonCZTime = 2.0*q1.noonSwapLenC21s[n-1]

    def func(server, curr):

        ph = phase
        start = 0

        q0.xy = env.NOTHING
        q0.z = env.NOTHING

        for i in range(n-1):

            q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
            start += q0['piLen']/2
            q0.z += env.rect(start, sLen[i], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[i]*overshoot)
            start += sLen[i]+q0['piLen']/2 # heck: it can be optimized

        if loadReset:

            q0.z += env.rect(start, q0.noonSwapLenR1s[0], q0.noonSwapAmpR1[0])
            start += q0.noonSwapLenR1s[0]+q0['piLen']/2
            # reset q0

        # first Ramsey pulse on q0
        q0.xy += eh.mix(q0, eh.piHalfPulse(q0, start, phase=ph))
        start += q0['piLen']/2

        # START MULTI-PHOTON CZ GATE

        # map q0 into rc
        q0.z += env.rect(start, sLen[n-1], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[n-1]*overshoot)
        start += sLen[n-1]+2.0*ns

        # CZ gate
        q1.z = env.rect(start, multiPhotonCZTime, CZAmp)
        start += multiPhotonCZTime

        # map back one excitation of rc into q0
        q0.z += env.rect(start, sLen[n-1], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[n-1]*overshoot)

        # compensation pulse for q1
        q1.z += env.rect(start, compPulseLen, compPulseAmpQ1)
        start += sLen[n-1]
#        start += max(sl[n-1], compPulseLen)

        # compensation pulse for q0
        q0.z += env.rect(start, compPulseLen, curr)
        start += compPulseLen+q0['piLen']/2

        # END MULTI-PHOTON CZ GATE

        # second Ramsey pulse on q0
        q0.xy += eh.mix(q0, eh.piHalfPulse(q0, start, phase=ph))
        start += q0['piLen']/2

        # measure pulse and readout q0
        q0.z += eh.measurePulse(q0, start)
        q0['readout'] = True

        return runQubits(server, qubits, stats=stats, probs=[1])

    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)

def multiPhotonCZBellsQST(sample, repetition=10, measure=[0,1], stats=1500L,
                          n=1, compPulseAmpQ0=0.0, compPulseAmpQ1=0.0, compPulseLen=7.0*ns,
                          phase=0.5*np.pi, loadReset=False, overshoot=False,
                          name='Multi photon CZ Bell states QST', exName='U00', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    # measurement = pyle.dataking.measurement.Tomo(2)
    measurement = pyle.dataking.measurement.Octomo(2)

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+'_'+exName+'_n'+str(n), axes, measure=measurement, kw=kw)

    sLen = q0['noonSwapLenCs']

    CZAmp = q1.noonSwapAmpC21

    multiPhotonCZTime = 2.0*q1.noonSwapLenC21s[n-1]

    def func(server, curr):

        ph = phase
        start = 0

        q0.xy = env.NOTHING
        q0.z = env.NOTHING

        for i in range(n-1):

            q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
            start += q0['piLen']/2
            q0.z += env.rect(start, sLen[i], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[i]*overshoot)
            start += sLen[i]+q0['piLen']/2 # heck: it can be optimized

        if loadReset:

            q0.z += env.rect(start, q0.noonSwapLenR1s[0], q0.noonSwapAmpR1[0])
            start += q0.noonSwapLenR1s[0]+q0['piLen']/2
            # reset q0

        # Ramsey pulse on q0
        q0.xy += eh.mix(q0, eh.piHalfPulse(q0, start, phase=ph))
        start += q0['piLen']/2

        # START MULTI-PHOTON CZ GATE

        # map q0 into rc
        q0.z += env.rect(start, sLen[n-1], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[n-1]*overshoot)
        start += sLen[n-1]+2.0*ns-q1['piLen']/2

        # Ramsey pulse on q1
        q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
        start += q1['piLen']/2

        # CZ gate
        q1.z = env.rect(start, multiPhotonCZTime, CZAmp)
        start += multiPhotonCZTime

        # map back one excitation of rc into q0
        q0.z += env.rect(start, sLen[n-1], q0.noonSwapAmpC, q0.noonSwapAmpCOSs[n-1]*overshoot)

        # compensation pulse for q1
        q1.z += env.rect(start, compPulseLen, compPulseAmpQ1)
#        start += sLen[n-1]
#        start += max(sl[n-1], compPulseLen)

        # compensation pulse for q0
        q0.z += env.rect(start+sLen[n-1]+5.0*ns, compPulseLen, compPulseAmpQ0)

        # END MULTI-PHOTON CZ GATE

        # Ramsey pulse on q1
        q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start+compPulseLen+q1['piLen']/2, phase=ph))
        start += max([(sLen[n-1]['ns']+compPulseLen['ns']),(compPulseLen['ns']+q1['piLen']['ns'])])+6.0*ns
#        start += max((sl[n-1]+delay+compPulseLen),(compPulseLen+q1['piLen']))

        return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    # rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    rho_ideal = np.array([[0.5,0.0,0.0,0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.5,0.0,0.0,0.5]])
    # rho_ideal = np.array([[0.0,0.0,0.0,0.0],[0.0,-0.5,0.5,0.0],[0.0,0.5,-0.5,0.0],[0.0,0.0,0.0,0.0]])
    # rho_ideal = np.array([[0.5,0.0,0.0,-0.5],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[-0.5,0.0,0.0,0.5]])
    result = np.sum(result,axis=0)/len(repetition)
    # Qk = np.reshape(result[1:],(-1,4))
    # Qk = np.reshape(result[1:],(9,4))
    Qk = np.reshape(result[1:],(36,4))
    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    # rho_cal = tomo.qst(Qk,'tomo2')
    rho_cal = tomo.qst(Qk,'octomo2')
    plotRhoSingle(rho_cal,figNo=100)
    pylab.title('Exp.')
    rho_caln = rho_cal.copy()
#    rho_caln[1,2] = abs(rho_caln[1,2])
#    rho_caln[2,1] = abs(rho_caln[2,1])
    print 'Fidelity is %g ' % np.trace(np.dot(rho_caln,rho_ideal))
    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal))

def ToffoliOR(sample,
              load=1, read=1,
              loadReset=False, readReset=True,
              gateControl=(False, False),
              stats=900L,
              delay=0.0*ns, CZPad=0.0*ns, phase=st.r[0:2*np.pi:np.pi/30.0], overshoot=True,
              name='Toffoli Ramsey OS',
              save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0, q1 = qubits[:2]
    qLoad = qubits[load]
    qRead = qubits[read]

    axes = [(phase, 'Ramsey phase'),
            (CZPad, 'CZPad')]
    kw = {'stats': stats, 'load': load, 'read': read, 'gateControl': gateControl,
          'loadReset': loadReset, 'readReset': readReset, 'delay': delay, 'overshoot': overshoot}
    #args = kw.copy()
    #del args['overshoot']
    #del args['delay']
    #del args['loadReset']
    #del args['readReset']
    #name += ': ' + ', '.join('%s=%s' % item for item in sorted(kw.items()))
    # hack: this expects 4 probs for the qubit states.  We return 4 probs for the controls
    dataset = sweeps.prepDataset(sample, name, axes, measure=[0,1], kw=kw)

    def swap(t0, q, par, mult=1, overshoot=False):
        lenPar = 'noonSwapLen%ss' % par
        ampPar = 'noonSwapAmp%s' % par
        if overshoot:
            osPar = 'noonSwapAmp%sOSs' % par
            osAmp = q[osPar][0]*overshoot
        else:
            osAmp = 0
        return env.rect(t0, q[lenPar][0]*mult, q[ampPar], osAmp), q[lenPar][0]*mult

    def func(server, ph, CZPad):
        seqs = []
        for control in [(0,0), (0,1), (1,0), (1,1)]:
            start = 0*ns

            for q in qubits[:2]:
                q.xy = env.NOTHING
                q.z = env.NOTHING

            # load resonator
            qLoad.xy += eh.piHalfPulseHD(qLoad, start)
            start += qLoad['piLen']/2

            swapPulse, dt = swap(start, qLoad, 'C', overshoot=overshoot)
            qLoad.z += swapPulse
            start += dt

            if loadReset:
                resetPulse, dt = swap(start, qLoad, 'R0', overshoot=False)
                qLoad.z += resetPulse
                start += dt

            # set up control qubits
            piLen = max([q['piLen'] for q in qubits[:2]])
            for q, on in zip(qubits, control):
                if on:
                    q.xy += eh.piPulseHD(q, start + piLen - q['piLen']/2.0)
            start += piLen

            # Toffoli gate
            halfCZ1, dt = swap(start, q1, 'C21', overshoot=False)
            if gateControl[1]:
                q1.z += halfCZ1
            start += dt

            CZ, dt = swap(start, q0, 'C21', 2, overshoot=False)
            if gateControl[0]:
                q0.z += CZ
            start += dt+CZPad

            halfCZ2, dt = swap(start, q1, 'C21', overshoot=False)
            if gateControl[1]:
                q1.z += halfCZ2
            start += dt

            # readout of resonator
            if readReset:
                resetPulse, dt = swap(start, qRead, 'R0', overshoot=False)
                qRead.z += resetPulse
                start += dt + 3*ns # extra delay after reset

            swapPulse, dt = swap(start, qRead, 'C', overshoot=overshoot)
            qRead.z += swapPulse
            start += dt

            qRead.xy += eh.piHalfPulseHD(qLoad, start+qLoad['piLen']/2, phase=ph)
            start += qLoad['piLen']

            # measure readout qubit
            qRead.z += eh.measurePulse(qRead, start)
            qRead['readout'] = True

            for q in qubits[:2]:
                q.xy = eh.mix(q, q.xy)

            seqs += [runQubits(server, qubits, stats=stats, probs=[1])]

        ans = yield FutureList(seqs)
        probs = [p[0] for p in ans]
        returnValue(probs)

    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)

def ToffoliORfull(sample,
                  load=0, read=0,
                  loadReset=False, readReset=True,
                  gateControl=(False, False),
                  measure=[0, 1, 2], controlBits=[(0,0), (0,1), (1,0), (1,1)],
                  stats=900L,
                  CZPad=0.0*ns, CZDet=0.0, compAmp0=0.0, compAmp1=0.0, compLen=5.0*ns, resDelay=0.0*ns,
                  phase=st.r[0:2*np.pi:np.pi/30.0], overshoot=True,
                  name='Toffoli Ramsey OS',
                  save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0, q1 = qubits[:2]
    qLoad = qubits[load]
    qRead = qubits[read]

    axes = [(phase, 'Ramsey phase'),
            (CZPad, 'CZPad'),
            (CZDet, 'CZDet'),
            (compAmp0, 'compAmp0'),
            (compAmp1, 'compAmp1'),
            (resDelay, 'resDelay')]
    kw = {'stats': stats, 'load': load, 'read': read, 'gateControl': gateControl,
          'loadReset': loadReset, 'readReset': readReset, 'overshoot': overshoot}

    # assemble the list of initial control states
    controls = []
    for i in measure:
        for bits in controlBits:
            bits = list(bits)
            bits.insert(i, 0.5)
            controls.append(bits)

    # build list of dependent variables
    deps = [('Probability', ''.join('r' if a == 0.5 else str(a) for a in control), '') for control in controls]

    # prepare the dataset
    dataset = sweeps.prepDataset(sample, name, axes, dependents=deps, measure=[0], kw=kw)

    def setupPulse(state, q, time):
        """Create a pulse to prepare an initial state, either nothing, pi/2 or pi"""
        if state == 0: return env.NOTHING
        if state == 0.5: return eh.piHalfPulseHD(q, time)
        if state == 1: return eh.piPulseHD(q, time)
        raise Exception('invalid state')

    def swap(t0, q, par, mult=1, overshoot=False):
        """Create a swap pulse using some set of parameters and potentially scaling it in time"""
        lenPar = 'noonSwapLen%ss' % par
        ampPar = 'noonSwapAmp%s' % par
        if overshoot:
            osPar = 'noonSwapAmp%sOSs' % par
            osAmp = q[osPar][0]*overshoot
        else:
            osAmp = 0
        return env.rect(t0, q[lenPar][0]*mult, q[ampPar], osAmp), q[lenPar][0]*mult

    def func(server, ph, CZPad, CZDet, compAmp0, compAmp1, resDelay):
        seqs = []
        for control in controls:
            start = 0*ns

            for q in qubits[:2]:
                q.xy = env.NOTHING
                q.z = env.NOTHING
                q['readout'] = False

            # load resonator
            qLoad.xy += setupPulse(control[2], qLoad, start)
            start += qLoad['piLen']/2

            swapPulse, dt = swap(start, qLoad, 'C', overshoot=overshoot)
            qLoad.z += swapPulse
            start += dt

            if loadReset:
                resetPulse, dt = swap(start, qLoad, 'R0', overshoot=False)
                qLoad.z += resetPulse
                start += dt

            # set up control qubits
            piLen = max([q['piLen'] for q in qubits[:2]])
            for q, state in zip(qubits[:2], control[:2]):
                q.xy += setupPulse(state, q, start + piLen - q['piLen']/2.0)
            start += piLen

            # Toffoli OR-gate
            halfCZ1, dt = swap(start, q1, 'C21', overshoot=False)
            if gateControl[1]:
                q1.z += halfCZ1
            start += dt

            CZ, dt = swap(start, q0, 'C21', 2, overshoot=False)
            if gateControl[0]:
                q0.z += CZ
            q1.z += env.rect(start, dt+CZPad, CZDet) # add detuning to shelved qubit
            start += dt+CZPad


            halfCZ2, dt = swap(start, q1, 'C21', overshoot=False)
            if gateControl[1]:
                q1.z += halfCZ2
            start += dt

            # qubit compensation pulses
            q0.z += env.rect(start, compLen, compAmp0)
            q1.z += env.rect(start, compLen, compAmp1)
            start += compLen

            if 0.5 in control[:2]:
                # readout a qubit
                i = control[:2].index(0.5)
                qi = qubits[i]
                qi.xy += eh.piHalfPulseHD(qi, start+qi['piLen']/2, phase=ph)
                start += qi['piLen']

                # measure readout qubit
                qi.z += eh.measurePulse(qi, start)
                qi['readout'] = True

            if control[2] == 0.5:
                # readout the resonator
                if readReset:
                    resetPulse, dt = swap(start, qRead, 'R1', overshoot=False)
                    qRead.z += resetPulse
                    start += dt

                # extra delay for resonator
                start += resDelay

                swapPulse, dt = swap(start, qRead, 'C', overshoot=overshoot)
                qRead.z += swapPulse
                start += dt

                qRead.xy += eh.piHalfPulseHD(qLoad, start+qLoad['piLen']/2, phase=ph)
                start += qLoad['piLen']

                # measure readout qubit
                qRead.z += eh.measurePulse(qRead, start)
                qRead['readout'] = True

            for q in qubits[:2]:
                q.xy = eh.mix(q, q.xy)

            seqs += [runQubits(server, qubits, stats=stats, probs=[1])]

        ans = yield FutureList(seqs)
        probs = [p[0] for p in ans]
        returnValue(probs)

    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)

def ToffoliXORfull(sample,
                  load=0, read=0,
                  loadReset=False, readReset=True,
                  gateControl=(False, False),
                  measure=[0, 1, 2], controlBits=[(0,0), (0,1), (1,0), (1,1)],
                  stats=900L,
                  compAmp0=0.0, compAmp1=0.0, compLen=5.0*ns, resDelay=0.0*ns,
                  phase=st.r[0:2*np.pi:np.pi/30.0], overshoot=True,
                  name='Toffoli Ramsey OS',
                  save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0, q1 = qubits[:2]
    qLoad = qubits[load]
    qRead = qubits[read]

    axes = [(phase, 'Ramsey phase'),
            #(CZPad, 'CZPad'),
            (compAmp0, 'compAmp0'),
            (compAmp1, 'compAmp1'),
            (resDelay, 'resDelay')]
    kw = {'stats': stats, 'load': load, 'read': read, 'gateControl': gateControl,
          'loadReset': loadReset, 'readReset': readReset, 'overshoot': overshoot}

    # assemble the list of initial control states
    controls = []
    for i in measure:
        for bits in controlBits:
            bits = list(bits)
            bits.insert(i, 0.5)
            controls.append(bits)

    # build list of dependent variables
    deps = [('Probability', ''.join('r' if a == 0.5 else str(a) for a in control), '') for control in controls]

    # prepare the dataset
    dataset = sweeps.prepDataset(sample, name, axes, dependents=deps, measure=[0], kw=kw)

    def setupPulse(state, q, time):
        """Create a pulse to prepare an initial state, either nothing, pi/2 or pi"""
        if state == 0: return env.NOTHING
        if state == 0.5: return eh.piHalfPulseHD(q, time)
        if state == 1: return eh.piPulseHD(q, time)
        raise Exception('invalid state')

    def swap(t0, q, par, mult=1, overshoot=False):
        """Create a swap pulse using some set of parameters and potentially scaling it in time"""
        lenPar = 'noonSwapLen%ss' % par
        ampPar = 'noonSwapAmp%s' % par
        if overshoot:
            osPar = 'noonSwapAmp%sOSs' % par
            osAmp = q[osPar][0]*overshoot
        else:
            osAmp = 0
        return env.rect(t0, q[lenPar][0]*mult, q[ampPar], osAmp), q[lenPar][0]*mult

    def func(server, ph, compAmp0, compAmp1, resDelay):#CZPad
        seqs = []
        for control in controls:
            start = 0*ns

            for q in qubits[:2]:
                q.xy = env.NOTHING
                q.z = env.NOTHING
                q['readout'] = False

            # load resonator
            qLoad.xy += setupPulse(control[2], qLoad, start)
            start += qLoad['piLen']/2

            swapPulse, dt = swap(start, qLoad, 'C', overshoot=overshoot)
            qLoad.z += swapPulse
            start += dt

            if loadReset:
                resetPulse, dt = swap(start, qLoad, 'R0', overshoot=False)
                qLoad.z += resetPulse
                start += dt

            # set up control qubits
            piLen = max([q['piLen'] for q in qubits[:2]])
            for q, state in zip(qubits[:2], control[:2]):
                q.xy += setupPulse(state, q, start + piLen - q['piLen']/2.0)
            start += piLen

            # Toffoli XOR gate
            CZ1, dt = swap(start, q1, 'C21', 2, overshoot=False)
            if gateControl[1]:
                q1.z += CZ1
            start += dt

            CZ2, dt = swap(start, q0, 'C21', 2, overshoot=False)
            if gateControl[0]:
                q0.z += CZ2
            start += dt#+CZPad

            # qubit compensation pulses
            q0.z += env.rect(start, compLen, compAmp0)
            q1.z += env.rect(start, compLen, compAmp1)
            start += compLen

            if 0.5 in control[:2]:
                # readout a qubit
                i = control[:2].index(0.5)
                qi = qubits[i]
                qi.xy += eh.piHalfPulseHD(qi, start+qi['piLen']/2, phase=ph)
                start += qi['piLen']

                # measure readout qubit
                qi.z += eh.measurePulse(qi, start)
                qi['readout'] = True

            if control[2] == 0.5:
                # readout the resonator
                if readReset:
                    resetPulse, dt = swap(start, qRead, 'R1', overshoot=False)
                    qRead.z += resetPulse
                    start += dt

                # extra delay for resonator
                start += resDelay

                swapPulse, dt = swap(start, qRead, 'C', overshoot=overshoot)
                qRead.z += swapPulse
                start += dt

                qRead.xy += eh.piHalfPulseHD(qLoad, start+qLoad['piLen']/2, phase=ph)
                start += qLoad['piLen']

                # measure readout qubit
                qRead.z += eh.measurePulse(qRead, start)
                qRead['readout'] = True

            for q in qubits[:2]:
                q.xy = eh.mix(q, q.xy)

            seqs += [runQubits(server, qubits, stats=stats, probs=[1])]

        ans = yield FutureList(seqs)
        probs = [p[0] for p in ans]
        returnValue(probs)

    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)

def ToffoliXOR(sample,
               load=1, read=1,
               loadReset=False, readReset=True,
               gateControl=(False, False),
               stats=900L,
               delay=0.0*ns, CZPad=0.0*ns, phase=st.r[0:2*np.pi:np.pi/30.0], overshoot=True,
               name='Toffoli XOR Ramsey OS',
               save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0, q1 = qubits[:2]
    qLoad = qubits[load]
    qRead = qubits[read]

    axes = [(phase, 'Ramsey phase'),
            (CZPad, 'CZPad')]
    kw = {'stats': stats, 'load': load, 'read': read, 'gateControl': gateControl,
          'loadReset': loadReset, 'readReset': readReset, 'delay': delay, 'overshoot': overshoot}
    #args = kw.copy()
    #del args['overshoot']
    #del args['delay']
    #del args['loadReset']
    #del args['readReset']
    #name += ': ' + ', '.join('%s=%s' % item for item in sorted(kw.items()))
    # hack: this expects 4 probs for the qubit states.  We return 4 probs for the controls
    dataset = sweeps.prepDataset(sample, name, axes, measure=[0,1], kw=kw)

    def swap(t0, q, par, mult=1, overshoot=False):
        lenPar = 'noonSwapLen%ss' % par
        ampPar = 'noonSwapAmp%s' % par
        if overshoot:
            osPar = 'noonSwapAmp%sOSs' % par
            osAmp = q[osPar][0]*overshoot
        else:
            osAmp = 0
        return env.rect(t0, q[lenPar][0]*mult, q[ampPar], osAmp), q[lenPar][0]*mult

    def func(server, ph, CZPad):
        seqs = []
        for control in [(0,0), (0,1), (1,0), (1,1)]:
            start = 0*ns

            for q in qubits[:2]:
                q.xy = env.NOTHING
                q.z = env.NOTHING

            # load resonator
            qLoad.xy += eh.piHalfPulseHD(qLoad, start)
            start += qLoad['piLen']/2

            swapPulse, dt = swap(start, qLoad, 'C', overshoot=overshoot)
            qLoad.z += swapPulse
            start += dt

            if loadReset:
                resetPulse, dt = swap(start, qLoad, 'R0', overshoot=False)
                qLoad.z += resetPulse
                start += dt

            # set up control qubits
            piLen = max([q['piLen'] for q in qubits[:2]])
            for q, on in zip(qubits, control):
                if on:
                    q.xy += eh.piPulseHD(q, start + piLen - q['piLen']/2.0)
            start += piLen

            # Toffoli XOR gate
            CZ1, dt = swap(start, q1, 'C21', 2, overshoot=False)
            if gateControl[1]:
                q1.z += CZ1
            start += dt

            CZ2, dt = swap(start, q0, 'C21', 2, overshoot=False)
            if gateControl[0]:
                q0.z += CZ2
            start += dt+CZPad

            # readout of resonator
            if readReset:
                resetPulse, dt = swap(start, qRead, 'R0', overshoot=False)
                qRead.z += resetPulse
                start += dt + 3*ns # extra delay after reset

            swapPulse, dt = swap(start, qRead, 'C', overshoot=overshoot)
            qRead.z += swapPulse
            start += dt

            qRead.xy += eh.piHalfPulseHD(qLoad, start+qLoad['piLen']/2, phase=ph)
            start += qLoad['piLen']

            # measure readout qubit
            qRead.z += eh.measurePulse(qRead, start)
            qRead['readout'] = True

            for q in qubits[:2]:
                q.xy = eh.mix(q, q.xy)

            seqs += [runQubits(server, qubits, stats=stats, probs=[1])]

        ans = yield FutureList(seqs)
        probs = [p[0] for p in ans]
        returnValue(probs)

    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)

def CPHASECompPulseQ0Fock1Test(sample, compAmp=np.arange(-0.05,0.05,0.001), measure=0, stats=1200L,
                               CPHASETime=None*ns, compPulseLen=10.0*ns, compPulseAmpQ1=0.051, delay=0.0*ns, phase=0,
                               name='CPHASE Compensation Pulse qubit 0 Fock state 1', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(compAmp, 'compensation pulse amplitude')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

    def func(server, curr):

         start = 0

         ph = phase

         # first Ramsey pulse on q0
         q0.xy = eh.mix(q0, eh.piHalfPulse(q0, start, phase=ph))
         start += q0['piLen']/2+2.0*ns

         # START CZ GATE

         # generate Fock state 1 in rc
         q0.z = env.rect(start, sl[0], q0.noonSwapAmpC)
         start += sl[0]

         # CZ gate
         q1.z = env.rect(start, CPHASETime, q1.noonSwapAmpC21)
         start += CPHASETime+2.0*ns

         # map back rc into q0
         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC)

         # compensation pulse for q1
         q1.z += env.rect(start, compPulseLen, compPulseAmpQ1)
         start += max(q0.noonSwapAmpC, compPulseLen)

         # compensation pulse for q0
         q0.z += env.rect(start, compPulseLen, curr)
         start += compPulseLen+2.0*ns+q0['piLen']/2

         # END CZ GATE

         # second Ramsey pulse on q0
         q0.xy += eh.mix(q0, eh.piHalfPulse(q0, start, phase=ph))
         start += q0['piLen']/2

         # measure pulse and readout q0
         q0.z += eh.measurePulse(q0, start)
         q0['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def singlePhotonQND1(sample, repetition=10, measure=[0,1], stats=1500L, delay=0*ns, delayA=0.0*ns, extraAmp=0.0,
                     name='Single photon QND 1 TOMO MQ', save=True, collect=False, noisy=True, extraDelay=extraDelay):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    measurement = pyle.dataking.measurement.Tomo(2, [1])

    nameEx = [' q0->q1',' q1->q0']
    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx[measure[0]], axes, measure=measurement, kw=kw)

    def func(server, curr):

        start = 0

        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2
        q0.z = env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
        start += q0.noonSwapLenCs[0]-q0['piLen']/2+delay
        # 1 photon in resC

        q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
        start += q1['piLen']/2+delay

        q1.z = env.rect(start, 2.0*q1.noonSwapLenC21s[0]+delayA, q1.noonSwapAmpC21+extraAmp)
        start += 2.0*q1.noonSwapLenC21s[0]+delayA+delay

#        q1.z = env.rect(start, 0.0*q1.noonSwapLenC21s[0]+delayA, q1.noonSwapAmpC21+extraAmp)
#        start += 0.0*q1.noonSwapLenC21s[0]+delayA+delay

        return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    # rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)
    # pdb.set_trace()
    Qk = np.reshape(result[1:],(3,2))
    # Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    rho_cal = tomo.qst(Qk,'tomo')
    plotRhoSingle(rho_cal,figNo=100)
    pylab.title('Exp.')

    rhoCoherence = rho_cal[0,1]

    return np.arctan2(np.imag(rhoCoherence), np.real(rhoCoherence))
    # /np.pi/2

#    rho_caln = rho_cal.copy()
#    rho_caln[1,2] = abs(rho_caln[1,2])
#    rho_caln[2,1] = abs(rho_caln[2,1])
#    print 'Fidelity is %g ' % np.trace(np.dot(rho_caln,rho_ideal))
#
#    Us =tomo._qst_transforms['tomo2'][0]
#    rho_calLike = tomo.qst_mle(Qk,Us,rho0 = rho_ideal)
#    rho_calLiken = rho_calLike.copy()
#    rho_calLiken[1,2] = abs(rho_calLiken[1,2])
#    rho_calLiken[2,1] = abs(rho_calLiken[2,1])
#    plotRhoSingle(rho_calLike,figNo=101)
#    pylab.title('Exp. likely')
#    print 'Fidelity for likelyhood is %g ' % np.trace(np.dot(rho_calLiken,rho_ideal))
#    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal)), rho_calLike, np.trace(np.dot(rho_calLiken,rho_ideal))

def repeatSinglePhotonQND1(sample, save=False, delayA = np.arange(-2.0,2.0,0.5), stats=1500L, measure=[0,1]):

    deltaPhi = np.zeros(len(delayA))
    for i in range(len(delayA)):
        print 'Delay is %g' % delayA[i]
        deltaPhi[i] = singlePhotonQND1(sample, measure = measure, save = save, delayA = delayA[i]*ns,
                                       noisy = False, stats = stats)
    plt.figure(102)
    plt.plot(delayA, deltaPhi, 'bs-')

    pDeltaPhi = np.polyfit(delayA, deltaPhi, 1)

    return delayA, deltaPhi, pDeltaPhi

def singlePhotonQND2(sample, probeLen=st.arangePQ(0,500,1,ns), measure=0, stats=1200L, delay=0.0*ns, delayA=0.0*ns,
                     name='Single photon QND 2 MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    # nameEx = [' q1->q0', ' q0->q1']

    axes = [(probeLen, 'swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw) # +nameEx[measure]

    def func(server, curr):
         start = 0

         q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
         start += q0['piLen']/2
         q0.z = env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
         start += q0.noonSwapLen0s[0]-q0['piLen']/2+delay
         # 1 photon in resC

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay

         q1.z = env.rect(start, 2.0*q1.noonSwapLenC21s[0]+delayA, q1.noonSwapAmpC21)
         start += 2.0*q1.noonSwapLenC21s[0]+delayA+delay

         q0.z += env.rect(start, curr, q0.noonSwapAmpCRead)

         start += curr
         q0.z += eh.measurePulse(q0, start)

         q0['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def singlePhotonQNDPre(sample, swap21Length=st.arangePQ(0,250,1,ns), measure=0, stats=1200L, delay=0.0*ns, delayA=0.0*ns, phase=0,
                       name='Single photon QND variable 21 swap length MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    # nameEx = [' q1->q0', ' q0->q1']

    axes = [(swap21Length, '21 swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw) # +nameEx[measure]

    def func(server, curr):
         start = 0

         q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
         start += q0['piLen']/2
         q0.z = env.rect(start, q0.noonSwapLenCs[0]/2.0, q0.noonSwapAmpC)
         start += q0.noonSwapLen0s[0]/2.0-q0['piLen']/2+delay
         # 1 photon in resC

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay

         q1.z = env.rect(start, curr, q1.noonSwapAmpC21)
         start += curr+q1['piLen']/2

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase))
         start += q1['piLen']/2

         q1.z += eh.measurePulse(q1, start)

         q1['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def singlePhotonQNDPre0p1(sample, swap21Length=st.arangePQ(0,250,1,ns), measure=0, stats=1200L, delay=0.0*ns, delayA=0.0*ns, phase=0, dfRot=0.0*MHz,
                          name='Fock 0+1 variable 21 swap length MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    # nameEx = [' q1->q0', ' q0->q1']

    axes = [(swap21Length, '21 swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw) # +nameEx[measure]

    def func(server, curr):
         start = 0
         ph = phase + 2*np.pi*dfRot[GHz]*curr[ns]

         q0.xy = eh.mix(q0, eh.piHalfPulse(q0, start))
         start += q0['piLen']/2+delay

         q0.z = env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
         start += q0.noonSwapLen0s[0]-q0['piLen']/2+delay
         # 0+1 photons in resC

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay

         q1.z = env.rect(start, curr, q1.noonSwapAmpC21)
         start += curr+q1['piLen']/2

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2

         q1.z += eh.measurePulse(q1, start)

         q1['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def singlePhotonQNDPre0p1SQ(sample, swap21Length=st.arangePQ(0,250,1,ns), measure=0, stats=1200L, delay=0.0*ns, delayA=0.0*ns, phase=0, dfRot=0.0*MHz,
                            name='Fock 0+1 variable 21 swap length MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[1-measure]
    q1 = qubits[measure]

    # nameEx = [' q1->q0', ' q0->q1']

    axes = [(swap21Length, '21 swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw) # +nameEx[measure]

    def func(server, curr):
         start = 0
         ph = phase + 2*np.pi*dfRot[GHz]*curr[ns]

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay

         q1.z = env.rect(start, q1.noonSwapLen0s[0], q1.noonSwapAmp0)
         start += q1.noonSwapLen0s[0]+30.0*ns+delay
         # 0+1 photons in resC

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay

         q1.z += env.rect(start, curr, q1.noonSwapAmp021)
         start += curr+q1['piLen']/2

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2

         q1.z += eh.measurePulse(q1, start)

         q1['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def singlePhotonQNDPreRotScan(sample, swap21Length=st.arangePQ(0,250,1,ns), measure=0, stats=1800L, delay=0.0*ns, delayA=0.0*ns, phase=0, dfRot=0.0*GHz,
                              name='Single photon QND variable 21 swap length - rotating frame scan MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(swap21Length, '21 swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, curr):
         start = 0
         # dfRot=q1['f21']-q1['fResC']
         ph = phase + 2*np.pi*dfRot[GHz]*curr[ns]
         # do FFT and define dfRot
         # ph = phase

         q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
         start += q0['piLen']/2
         q0.z = env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
         start += q0.noonSwapLen0s[0]-q0['piLen']/2+delay
         # 1 photon in resC

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay

         q1.z = env.rect(start, curr, q1.noonSwapAmpC21)
         start += curr+q1['piLen']/2

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2

         q1.z += eh.measurePulse(q1, start)

         q1['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def singlePhotonQNDPreRotScanArb(sample, n=1, swap21Length=st.arangePQ(0,250,1,ns), measure=0, stats=1800L,
                                 delay=0.0*ns, delayA=0.0*ns, phase=0, dfRot=0.0*GHz,
                                 name='Single photon QND variable 21 swap length - rotating frame scan arbitray Fock MQ',
                                 save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(swap21Length, '21 swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state', axes, measure=measure, kw=kw)

    sl = q0['noonSwapLenCs']

    def func(server, curr):
         start = 0
         # dfRot=q1['f21']-q1['fResC']
         ph = phase + 2*np.pi*dfRot[GHz]*curr[ns]
         # do FFT and define dfRot
         # ph = phase

         q0.xy = env.NOTHING
         q0.z = env.NOTHING

         for i in range(n-1):

             q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
             start += q0['piLen']/2
             q0.z += env.rect(start, sl[i], q0.noonSwapAmpC)
             start += sl[i]+q0['piLen']/2+delay

         q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
         start += q0['piLen']/2
         q0.z += env.rect(start, sl[n-1], q0.noonSwapAmpC)
         start += sl[n-1]-q0['piLen']/2+delay
         # generate Fock states in res C

#         q0.z += env.rect(start+q0['piLen']/2, curr, q0.noonSwapAmpCRead)
#
#         start += q0['piLen']/2+curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z = env.rect(start, curr, q1.noonSwapAmpC21)
         start += curr+q1['piLen']/2
         # CPHASE gate

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2
         # pre-measurement pulse

         q1.z += eh.measurePulse(q1, start)

         q1['readout'] = True

#         q0.z += env.rect(start, curr, q0.noonSwapAmpCRead)
#
#         start += curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def SingleQuHadTOMO(sample, repetition=10, stats=1500L, measure=[0,1], delay=0.0*ns, delayA=0.0*ns, phase=0.0,
                    name='Single qubit Hadamard TOMO MQ',
                    save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    measurement = pyle.dataking.measurement.Tomo(2, [1])

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measurement, kw=kw)

    def func(server, curr):
         start = 0

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start, phase=phase))
         start += q1['piLen']/2+delay+2.0*ns

         return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    # rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)
    # pdb.set_trace()
    Qk = np.reshape(result[1:],(3,2))
    # Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    rho_cal = tomo.qst(Qk,'tomo')
    plotRhoSingle(rho_cal,figNo=100)
    pylab.title('Exp.')

    rhoCoherenceSingle = rho_cal[0,1]
    rhoPhiSingle = np.arctan2(np.imag(rhoCoherenceSingle), np.real(rhoCoherenceSingle))

    return rhoPhiSingle
    # /np.pi/2

def QuResCPHASEPreRotArbTOMO(sample, n=1, repetition=10, stats=1500L, CPHASETime=None*ns, phase=0, dfRot=0.0*GHz, measure=[0,1],
                             delay=0.0*ns, delayA=0.0*ns, plotFlag=True,
                             name='Single photon QND variable 21 swap length - rotating frame arbitray Fock TOMO MQ',
                             save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    measurement = pyle.dataking.measurement.Tomo(2, [1])

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state', axes, measure=measurement, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

    def func(server, curr):
         start = 0
         # dfRot=q1['f21']-q1['fResC']
         # ph = phase + 2*np.pi*dfRot[GHz]*curr[ns]
         # do FFT and define dfRot
         # ph = phase

         q0.xy = env.NOTHING
         q0.z = env.NOTHING

         for i in range(n-1):

             q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
             start += q0['piLen']/2
             q0.z += env.rect(start, sl[i], q0.noonSwapAmpC)
             start += sl[i]+q0['piLen']/2+delay

         q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
         start += q0['piLen']/2
         q0.z += env.rect(start, sl[n-1], q0.noonSwapAmpC)
         start += sl[n-1]-q0['piLen']/2+delay
         # generate Fock states in res C

#         q0.z += env.rect(start+q0['piLen']/2, curr, q0.noonSwapAmpCRead)
#
#         start += q0['piLen']/2+curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay+2.0*ns

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z = env.rect(start, CPHASETime, q1.noonSwapAmpC21)
         start += CPHASETime+2.0*ns
         # CPHASE gate

#         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
#         start += q1['piLen']/2
#         # pre-measurement pulse

         return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    # rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)
    Qk = np.reshape(result[1:],(3,2))
    # Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    rho_cal = tomo.qst(Qk,'tomo')

    if plotFlag:
         plotRhoSingle(rho_cal,figNo=101)
         pylab.title('Exp.')

    rhoCoherence = rho_cal[0,1]
    rhoPhi = np.arctan2(np.imag(rhoCoherence), np.real(rhoCoherence))

    # pdb.set_trace()
    rhoPhiDynamic = 2.0*np.pi*dfRot*CPHASETime

    rhoPhiCorrect = rhoPhi-rhoPhiDynamic

    rhoAmp = np.sqrt(np.real(rhoCoherence)**2+np.imag(rhoCoherence)**2)
    aPrime = rhoAmp*np.cos(rhoPhiCorrect)
    bPrime = rhoAmp*np.sin(rhoPhiCorrect)
    rhoCoherenceCorrect = aPrime+1.0j*bPrime

#    rho_cal[0,1] = 1.0*rhoCoherenceCorrect
#    rho_cal[1,0] = -1.0*rhoCoherenceCorrect
    rho_cal[0,1] = aPrime+1.0j*bPrime
    rho_cal[1,0] = aPrime-1.0j*bPrime

    if plotFlag:
         plotRhoSingle(rho_cal, figNo=102)
         pylab.title('Phase correct')

    # return rhoPhi, rhoPhiDynamic, rhoPhiCorrect
    return rhoPhi, rhoAmp
    # /np.pi/2

def repeatQuResCPHASEPreRotArbTOMO(sample, n=1, save=True, CPHASETime=st.arangePQ(0,120,1,ns), dfRot=0.0*GHz, stats=1500L, measure=[0,1], plotFlag=False,
                                   name='Repeat qubit-resonator CPHASE TOMO', nameEx='run00'):

    rhoPhi = np.zeros(len(CPHASETime))
    rhoAmp = np.zeros(len(CPHASETime))
    outFile = open('U:\\Matteo\\20100514Fri\\%s%s.dat' % (name, nameEx), 'w')
    for i in range(len(CPHASETime)):
        print 'CPHASE time is %g' % CPHASETime[i]
        rhoPhi[i], rhoAmp[i] = QuResCPHASEPreRotArbTOMO(sample, n = n, measure = measure, save = save, plotFlag = plotFlag, CPHASETime = CPHASETime[i],
                                                        phase = 0, dfRot = dfRot, noisy = False, stats = stats)
        time = CPHASETime[i]
        # print time
        outString = str(time.value)+'    '+ str(rhoPhi[i])+'    '+ str(rhoAmp[i])
        print >> outFile, outString
    outFile.close()

    plt.figure(103)
    plt.plot(CPHASETime, rhoPhi, 'bs-')

    plt.figure(104)
    plt.plot(CPHASETime, rhoAmp, 'rs-')

    # pDeltaPhi = np.polyfit(delayA, deltaPhi, 1)

    return CPHASETime, rhoPhi, rhoAmp

def QuResCPHASEPreRotArbTOMO2(sample, n=1, repetition=10, stats=1500L, CPHASETime=None*ns, phase=0, dfRot=0.0*GHz, measure=[0,1],
                              delay=0.0*ns, delayA=0.0*ns, plotFlag=True,
                              name='Single photon QND variable 21 swap length - rotating frame arbitray Fock TOMO2 MQ',
                              save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    measurement = pyle.dataking.measurement.Tomo2(2, [1])

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state', axes, measure=measurement, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

    def func(server, curr):
         start = 0
         # dfRot=q1['f21']-q1['fResC']
         # ph = phase + 2*np.pi*dfRot[GHz]*curr[ns]
         # do FFT and define dfRot
         # ph = phase

         q0.xy = env.NOTHING
         q0.z = env.NOTHING

         for i in range(n-1):

             q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
             start += q0['piLen']/2
             q0.z += env.rect(start, sl[i], q0.noonSwapAmpC)
             start += sl[i]+q0['piLen']/2+delay

         q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
         start += q0['piLen']/2
         q0.z += env.rect(start, sl[n-1], q0.noonSwapAmpC)
         start += sl[n-1]-q0['piLen']/2+delay
         # generate Fock states in res C

#         q0.z += env.rect(start+q0['piLen']/2, curr, q0.noonSwapAmpCRead)
#
#         start += q0['piLen']/2+curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay+2.0*ns

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z = env.rect(start, CPHASETime, q1.noonSwapAmpC21)
         start += CPHASETime+2.0*ns
         # CPHASE gate

#         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
#         start += q1['piLen']/2
#         # pre-measurement pulse

         return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    # rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)
    Qk = np.reshape(result[1:],(3,2))
    # Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    rho_cal = tomo.qst(Qk,'tomo')

    if plotFlag:
         plotRhoSingle(rho_cal,figNo=101)
         pylab.title('Exp.')

    rhoCoherence = rho_cal[0,1]
    rhoPhi = np.arctan2(np.imag(rhoCoherence), np.real(rhoCoherence))

    # pdb.set_trace()
    rhoPhiDynamic = 2.0*np.pi*dfRot*CPHASETime

    rhoPhiCorrect = rhoPhi-rhoPhiDynamic

    rhoAmp = np.sqrt(np.real(rhoCoherence)**2+np.imag(rhoCoherence)**2)
    aPrime = rhoAmp*np.cos(rhoPhiCorrect)
    bPrime = rhoAmp*np.sin(rhoPhiCorrect)
    rhoCoherenceCorrect = aPrime+1.0j*bPrime

#    rho_cal[0,1] = 1.0*rhoCoherenceCorrect
#    rho_cal[1,0] = -1.0*rhoCoherenceCorrect
    rho_cal[0,1] = aPrime+1.0j*bPrime
    rho_cal[1,0] = aPrime-1.0j*bPrime

    if plotFlag:
         plotRhoSingle(rho_cal, figNo=102)
         pylab.title('Phase correct')

    # return rhoPhi, rhoPhiDynamic, rhoPhiCorrect
    return rhoPhi, rhoAmp
    # /np.pi/2

def repeatQuResCPHASEPreRotArbTOMO2(sample, n=1, save=True, CPHASETime=st.arangePQ(0,120,1,ns), dfRot=0.0*GHz, stats=1500L, measure=[0,1], plotFlag=False,
                                    name='Repeat qubit-resonator CPHASE TOMO2', nameEx='run00'):

    rhoPhi = np.zeros(len(CPHASETime))
    rhoAmp = np.zeros(len(CPHASETime))
    outFile = open('U:\\Matteo\\20100514Fri\\%s%s.dat' % (name, nameEx), 'w')
    for i in range(len(CPHASETime)):
        print 'CPHASE time is %g' % CPHASETime[i]
        rhoPhi[i], rhoAmp[i] = QuResCPHASEPreRotArbTOMO2(sample, n = n, measure = measure, save = save, plotFlag = plotFlag, CPHASETime = CPHASETime[i],
                                                         phase = 0, dfRot = dfRot, noisy = False, stats = stats)
        time = CPHASETime[i]
        # print time
        outString = str(time.value)+'    '+ str(rhoPhi[i])+'    '+ str(rhoAmp[i])
        print >> outFile, outString
    outFile.close()

    plt.figure(103)
    plt.plot(CPHASETime, rhoPhi, 'bs-')

    plt.figure(104)
    plt.plot(CPHASETime, rhoAmp, 'rs-')

    # pDeltaPhi = np.polyfit(delayA, deltaPhi, 1)

    return CPHASETime, rhoPhi, rhoAmp

def QuResCPHASEPreRotArbZeroTOMO(sample, n=0, repetition=10, stats=1500L, CPHASETime=None*ns, phase=0, dfRot=0.0*GHz, measure=[0,1],
                                 delay=0.0*ns, delayA=0.0*ns, plotFlag=True,
                                 name='Single photon QND variable 21 swap length - rotating frame arbitrary Fock TOMO MQ',
                                 save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    measurement = pyle.dataking.measurement.Tomo(2, [1])

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state', axes, measure=measurement, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

    def func(server, curr):
         start = 0
         # dfRot=q1['f21']-q1['fResC']
         # ph = phase + 2*np.pi*dfRot[GHz]*curr[ns]
         # do FFT and define dfRot
         # ph = phase

#         q0.xy = env.NOTHING
#         q0.z = env.NOTHING
#
#         for i in range(n-1):
#
#             q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
#             start += q0['piLen']/2
#             q0.z += env.rect(start, sl[i], q0.noonSwapAmpC)
#             start += sl[i]+q0['piLen']/2+delay
#
#         q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
#         start += q0['piLen']/2
#         q0.z += env.rect(start, sl[n-1], q0.noonSwapAmpC)
#         start += sl[n-1]-q0['piLen']/2+delay
#         # generate Fock states in res C

#         q0.z += env.rect(start+q0['piLen']/2, curr, q0.noonSwapAmpCRead)
#
#         start += q0['piLen']/2+curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay+2.0*ns

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z = env.rect(start, CPHASETime, q1.noonSwapAmpC21)
         start += CPHASETime+2.0*ns
         # CPHASE gate

#         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
#         start += q1['piLen']/2
#         # pre-measurement pulse

         return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    # rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)
    Qk = np.reshape(result[1:],(3,2))
    # Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    rho_cal = tomo.qst(Qk,'tomo')

    if plotFlag:
         plotRhoSingle(rho_cal,figNo=101)
         pylab.title('Exp.')

    rhoCoherence = rho_cal[0,1]
    rhoPhi = np.arctan2(np.imag(rhoCoherence), np.real(rhoCoherence))

    # pdb.set_trace()
    rhoPhiDynamic = 2.0*np.pi*dfRot*CPHASETime

    rhoPhiCorrect = rhoPhi-rhoPhiDynamic

    rhoAmp = np.sqrt(np.real(rhoCoherence)**2+np.imag(rhoCoherence)**2)
    aPrime = rhoAmp*np.cos(rhoPhiCorrect)
    bPrime = rhoAmp*np.sin(rhoPhiCorrect)
    rhoCoherenceCorrect = aPrime+1.0j*bPrime

#    rho_cal[0,1] = 1.0*rhoCoherenceCorrect
#    rho_cal[1,0] = -1.0*rhoCoherenceCorrect
    rho_cal[0,1] = aPrime+1.0j*bPrime
    rho_cal[1,0] = aPrime-1.0j*bPrime

    if plotFlag:
         plotRhoSingle(rho_cal, figNo=102)
         pylab.title('Phase correct')

    # return rhoPhi, rhoPhiDynamic, rhoPhiCorrect
    return rhoPhi, rhoAmp
    # /np.pi/2

def repeatQuResCPHASEPreRotArbZeroTOMO(sample, n=0, save=True, CPHASETime=st.arangePQ(0,120,1,ns), dfRot=0.0*GHz, stats=1500L, measure=[0,1], plotFlag=False,
                                       name='Repeat qubit-resonator CPHASE TOMO', nameEx='run00'):

    rhoPhi = np.zeros(len(CPHASETime))
    rhoAmp = np.zeros(len(CPHASETime))
    outFile = open('U:\\Matteo\\20100516Sun\\%s%s.dat' % (name, nameEx), 'w')
    for i in range(len(CPHASETime)):
        print 'CPHASE time is %g' % CPHASETime[i]
        rhoPhi[i], rhoAmp[i] = QuResCPHASEPreRotArbZeroTOMO(sample, n = n, measure = measure, save = save, plotFlag=plotFlag, CPHASETime = CPHASETime[i],
                                                            phase=0, dfRot=dfRot, noisy = False, stats = stats)
        time = CPHASETime[i]
        # print time
        outString = str(time.value)+'    '+ str(rhoPhi[i])+'    '+ str(rhoAmp[i])
        print >> outFile, outString
    outFile.close()

    plt.figure(103)
    plt.plot(CPHASETime, rhoPhi, 'bs-')

    plt.figure(104)
    plt.plot(CPHASETime, rhoAmp, 'rs-')

    # pDeltaPhi = np.polyfit(delayA, deltaPhi, 1)

    return CPHASETime, rhoPhi, rhoAmp

def QuResCPHASEPreRotArbZeroTOMO2(sample, n=0, repetition=10, stats=1500L, CPHASETime=None*ns, phase=0, dfRot=0.0*GHz, measure=[0,1],
                                  delay=0.0*ns, delayA=0.0*ns, plotFlag=True,
                                  name='Single photon QND variable 21 swap length - rotating frame arbitrary Fock TOMO2 MQ',
                                  save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    measurement = pyle.dataking.measurement.Tomo2(2, [1])

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state', axes, measure=measurement, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

    def func(server, curr):
         start = 0
         # dfRot=q1['f21']-q1['fResC']
         # ph = phase + 2*np.pi*dfRot[GHz]*curr[ns]
         # do FFT and define dfRot
         # ph = phase

#         q0.xy = env.NOTHING
#         q0.z = env.NOTHING
#
#         for i in range(n-1):
#
#             q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
#             start += q0['piLen']/2
#             q0.z += env.rect(start, sl[i], q0.noonSwapAmpC)
#             start += sl[i]+q0['piLen']/2+delay
#
#         q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
#         start += q0['piLen']/2
#         q0.z += env.rect(start, sl[n-1], q0.noonSwapAmpC)
#         start += sl[n-1]-q0['piLen']/2+delay
#         # generate Fock states in res C

#         q0.z += env.rect(start+q0['piLen']/2, curr, q0.noonSwapAmpCRead)
#
#         start += q0['piLen']/2+curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay+2.0*ns

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z = env.rect(start, CPHASETime, q1.noonSwapAmpC21)
         start += CPHASETime+2.0*ns
         # CPHASE gate

#         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
#         start += q1['piLen']/2
#         # pre-measurement pulse

         return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    # rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)
    Qk = np.reshape(result[1:],(3,2))
    # Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    rho_cal = tomo.qst(Qk,'tomo')

    if plotFlag:
         plotRhoSingle(rho_cal,figNo=101)
         pylab.title('Exp.')

    rhoCoherence = rho_cal[0,1]
    rhoPhi = np.arctan2(np.imag(rhoCoherence), np.real(rhoCoherence))

    # pdb.set_trace()
    rhoPhiDynamic = 2.0*np.pi*dfRot*CPHASETime

    rhoPhiCorrect = rhoPhi-rhoPhiDynamic

    rhoAmp = np.sqrt(np.real(rhoCoherence)**2+np.imag(rhoCoherence)**2)
    aPrime = rhoAmp*np.cos(rhoPhiCorrect)
    bPrime = rhoAmp*np.sin(rhoPhiCorrect)
    rhoCoherenceCorrect = aPrime+1.0j*bPrime

#    rho_cal[0,1] = 1.0*rhoCoherenceCorrect
#    rho_cal[1,0] = -1.0*rhoCoherenceCorrect
    rho_cal[0,1] = aPrime+1.0j*bPrime
    rho_cal[1,0] = aPrime-1.0j*bPrime

    if plotFlag:
         plotRhoSingle(rho_cal, figNo=102)
         pylab.title('Phase correct')

    # return rhoPhi, rhoPhiDynamic, rhoPhiCorrect
    return rhoPhi, rhoAmp
    # /np.pi/2

def repeatQuResCPHASEPreRotArbZeroTOMO2(sample, n=0, save=True, CPHASETime=st.arangePQ(0,120,1,ns), dfRot=0.0*GHz, stats=1500L, measure=[0,1], plotFlag=False,
                                        name='Repeat qubit-resonator CPHASE TOMO2', nameEx='run00'):

    rhoPhi = np.zeros(len(CPHASETime))
    rhoAmp = np.zeros(len(CPHASETime))
    outFile = open('U:\\Matteo\\20100516Sun\\%s%s.dat' % (name, nameEx), 'w')
    for i in range(len(CPHASETime)):
        print 'CPHASE time is %g' % CPHASETime[i]
        rhoPhi[i], rhoAmp[i] = QuResCPHASEPreRotArbZeroTOMO2(sample, n = n, measure = measure, save = save, plotFlag=plotFlag, CPHASETime = CPHASETime[i],
                                                             phase=0, dfRot=dfRot, noisy = False, stats = stats)
        time = CPHASETime[i]
        # print time
        outString = str(time.value)+'    '+ str(rhoPhi[i])+'    '+ str(rhoAmp[i])
        print >> outFile, outString
    outFile.close()

    plt.figure(103)
    plt.plot(CPHASETime, rhoPhi, 'bs-')

    plt.figure(104)
    plt.plot(CPHASETime, rhoAmp, 'rs-')

    # pDeltaPhi = np.polyfit(delayA, deltaPhi, 1)

    return CPHASETime, rhoPhi, rhoAmp

def QuResCPHASEPreRotArb0p1TOMO(sample, repetition=10, stats=1500L, CPHASETime=None*ns, phase=0, dfRot=0.0*GHz, measure=[0,1],
                                delay=0.0*ns, delayA=0.0*ns, plotFlag=True,
                                name='Single photon QND variable 21 swap length - rotating frame Fock 0+1 TOMO MQ',
                                save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    measurement = pyle.dataking.measurement.Tomo(2, [1])

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measurement, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

    def func(server, curr):
         start = 0
         # dfRot=q1['f21']-q1['fResC']
         # ph = phase + 2*np.pi*dfRot[GHz]*curr[ns]
         # do FFT and define dfRot
         # ph = phase

         q0.xy = eh.mix(q0, eh.piHalfPulse(q0, start))
         start += q0['piLen']/2+delay

         q0.z = env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
         start += q0.noonSwapLenCs[0]-q0['piLen']/2+delay
         # 0+1 photons in resC

#         q0.z += env.rect(start+q0['piLen']/2, curr, q0.noonSwapAmpCRead)
#
#         start += q0['piLen']/2+curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay+2.0*ns

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z = env.rect(start, CPHASETime, q1.noonSwapAmpC21)
         start += CPHASETime+2.0*ns
         # CPHASE gate

#         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
#         start += q1['piLen']/2
#         # pre-measurement pulse

         return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    # rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)
    Qk = np.reshape(result[1:],(3,2))
    # Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    rho_cal = tomo.qst(Qk,'tomo')

    if plotFlag:
         plotRhoSingle(rho_cal,figNo=101)
         pylab.title('Exp.')

    rhoCoherence = rho_cal[0,1]
    rhoPhi = np.arctan2(np.imag(rhoCoherence), np.real(rhoCoherence))

    # pdb.set_trace()
    rhoPhiDynamic = 2.0*np.pi*dfRot*CPHASETime

    rhoPhiCorrect = rhoPhi-rhoPhiDynamic

    rhoAmp = np.sqrt(np.real(rhoCoherence)**2+np.imag(rhoCoherence)**2)
    aPrime = rhoAmp*np.cos(rhoPhiCorrect)
    bPrime = rhoAmp*np.sin(rhoPhiCorrect)
    rhoCoherenceCorrect = aPrime+1.0j*bPrime

#    rho_cal[0,1] = 1.0*rhoCoherenceCorrect
#    rho_cal[1,0] = -1.0*rhoCoherenceCorrect
    rho_cal[0,1] = aPrime+1.0j*bPrime
    rho_cal[1,0] = aPrime-1.0j*bPrime

    if plotFlag:
         plotRhoSingle(rho_cal, figNo=102)
         pylab.title('Phase correct')

    # return rhoPhi, rhoPhiDynamic, rhoPhiCorrect
    return rhoPhi, rhoAmp
    # /np.pi/2

def repeatQuResCPHASEPreRotArb0p1TOMO(sample, save=True, CPHASETime=st.arangePQ(0,120,1,ns), dfRot=0.0*GHz, stats=1500L, measure=[0,1], plotFlag=False,
                                      name='Repeat qubit-resonator CPHASE TOMO', nameEx='run00'):

    rhoPhi = np.zeros(len(CPHASETime))
    rhoAmp = np.zeros(len(CPHASETime))
    outFile = open('U:\\Matteo\\20100516Sun\\%s%s.dat' % (name, nameEx), 'w')
    for i in range(len(CPHASETime)):
        print 'CPHASE time is %g' % CPHASETime[i]
        rhoPhi[i], rhoAmp[i] = QuResCPHASEPreRotArb0p1TOMO(sample, measure = measure, save = save, plotFlag=plotFlag, CPHASETime = CPHASETime[i],
                                                           phase=0, dfRot=dfRot, noisy = False, stats = stats)
        time = CPHASETime[i]
        # print time
        outString = str(time.value)+'    '+ str(rhoPhi[i])+'    '+ str(rhoAmp[i])
        print >> outFile, outString
    outFile.close()

    plt.figure(103)
    plt.plot(CPHASETime, rhoPhi, 'bs-')

    plt.figure(104)
    plt.plot(CPHASETime, rhoAmp, 'rs-')

    # pDeltaPhi = np.polyfit(delayA, deltaPhi, 1)

    return CPHASETime, rhoPhi, rhoAmp

def QuResCPHASEPreRotArb0p1TOMO2(sample, repetition=10, stats=1500L, CPHASETime=None*ns, phase=0, dfRot=0.0*GHz, measure=[0,1],
                                 delay=0.0*ns, delayA=0.0*ns, plotFlag=True,
                                 name='Single photon QND variable 21 swap length - rotating frame Fock 0+1 TOMO MQ',
                                 save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    measurement = pyle.dataking.measurement.Tomo2(2, [1])

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measurement, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

    def func(server, curr):
         start = 0
         # dfRot=q1['f21']-q1['fResC']
         # ph = phase + 2*np.pi*dfRot[GHz]*curr[ns]
         # do FFT and define dfRot
         # ph = phase

         q0.xy = eh.mix(q0, eh.piHalfPulse(q0, start))
         start += q0['piLen']/2+delay

         q0.z = env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
         start += q0.noonSwapLenCs[0]-q0['piLen']/2+delay
         # 0+1 photons in resC

#         q0.z += env.rect(start+q0['piLen']/2, curr, q0.noonSwapAmpCRead)
#
#         start += q0['piLen']/2+curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay+2.0*ns

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z = env.rect(start, CPHASETime, q1.noonSwapAmpC21)
         start += CPHASETime+2.0*ns
         # CPHASE gate

#         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
#         start += q1['piLen']/2
#         # pre-measurement pulse

         return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    # rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)
    Qk = np.reshape(result[1:],(3,2))
    # Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    rho_cal = tomo.qst(Qk,'tomo')

    if plotFlag:
         plotRhoSingle(rho_cal,figNo=101)
         pylab.title('Exp.')

    rhoCoherence = rho_cal[0,1]
    rhoPhi = np.arctan2(np.imag(rhoCoherence), np.real(rhoCoherence))

    # pdb.set_trace()
    rhoPhiDynamic = 2.0*np.pi*dfRot*CPHASETime

    rhoPhiCorrect = rhoPhi-rhoPhiDynamic

    rhoAmp = np.sqrt(np.real(rhoCoherence)**2+np.imag(rhoCoherence)**2)
    aPrime = rhoAmp*np.cos(rhoPhiCorrect)
    bPrime = rhoAmp*np.sin(rhoPhiCorrect)
    rhoCoherenceCorrect = aPrime+1.0j*bPrime

#    rho_cal[0,1] = 1.0*rhoCoherenceCorrect
#    rho_cal[1,0] = -1.0*rhoCoherenceCorrect
    rho_cal[0,1] = aPrime+1.0j*bPrime
    rho_cal[1,0] = aPrime-1.0j*bPrime

    if plotFlag:
         plotRhoSingle(rho_cal, figNo=102)
         pylab.title('Phase correct')

    # return rhoPhi, rhoPhiDynamic, rhoPhiCorrect
    return rhoPhi, rhoAmp
    # /np.pi/2

def repeatQuResCPHASEPreRotArb0p1TOMO2(sample, save=True, CPHASETime=st.arangePQ(0,120,1,ns), dfRot=0.0*GHz, stats=1500L, measure=[0,1], plotFlag=False,
                                       name='Repeat qubit-resonator CPHASE TOMO', nameEx='run00'):

    rhoPhi = np.zeros(len(CPHASETime))
    rhoAmp = np.zeros(len(CPHASETime))
    outFile = open('U:\\Matteo\\20100516Sun\\%s%s.dat' % (name, nameEx), 'w')
    for i in range(len(CPHASETime)):
        print 'CPHASE time is %g' % CPHASETime[i]
        rhoPhi[i], rhoAmp[i] = QuResCPHASEPreRotArb0p1TOMO2(sample, measure = measure, save = save, plotFlag=plotFlag, CPHASETime = CPHASETime[i],
                                                            phase=0, dfRot=dfRot, noisy = False, stats = stats)
        time = CPHASETime[i]
        # print time
        outString = str(time.value)+'    '+ str(rhoPhi[i])+'    '+ str(rhoAmp[i])
        print >> outFile, outString
    outFile.close()

    plt.figure(103)
    plt.plot(CPHASETime, rhoPhi, 'bs-')

    plt.figure(104)
    plt.plot(CPHASETime, rhoAmp, 'rs-')

    # pDeltaPhi = np.polyfit(delayA, deltaPhi, 1)

    return CPHASETime, rhoPhi, rhoAmp

#def visibility(sample, mpa=st.r[0:2:0.05], stats=300, measure=0,
#               save=True, name='Visibility MQ', collect=True, update=False, noisy=True):
#    sample, qubits = util.loadQubits(sample)
#    q = qubits[measure]
#
#    axes = [(mpa, 'Measure pulse amplitude')]
#    deps = [('Probability', '|0>', ''),
#            ('Probability', '|1>', ''),
#            ('Visibility', '|1> - |0>', ''),
#            ('Probability', '|2>', ''),
#            ('Visibility', '|2> - |1>', '')
#            ]
#    kw = {'stats': stats}
#    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)
#
#    def func(server, mpa):
#        t_pi = 0
#        t_meas = q['piLen']/2.0
#
#        # without pi-pulse
#        q['readout'] = True
#        q['measureAmp'] = mpa
#        q.xy = env.NOTHING
#        q.z = eh.measurePulse(q, t_meas)
#        req0 = runQubits(server, qubits, stats, probs=[1])
#
#        # with pi-pulse
#        q['readout'] = True
#        q['measureAmp'] = mpa
#        q.xy = eh.mix(q, eh.piPulseHD(q, t_pi))
#        q.z = eh.measurePulse(q, t_meas)
#        req1 = runQubits(server, qubits, stats, probs=[1])
#
#        # |2> with pi-pulse
#        q['readout'] = True
#        q['measureAmp'] = mpa
#        q.xy = eh.mix(q, eh.piPulseHD(q, t_pi-q.piLen))+eh.mix(q, env.gaussian(t_pi, q.piFWHM, q.piAmp21, df=q.piDf21), freq = 'f21')
#        q.z = eh.measurePulse(q, t_meas)
#        req2 = runQubits(server, qubits, stats, probs=[1])
#
#        probs = yield FutureList([req0, req1, req2])
#        p0, p1, p2 = [p[0] for p in probs]
#
#        returnValue([p0, p1, p1-p0, p2, p2-p1])
#    return sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)

def QuResCPHASEPreRotArbP01P12(sample, n=1, swap21Length=st.arangePQ(0,120,0.1,ns), measure=0, stats=1200L,
                               delay=0.0*ns, delayA=0.0*ns, phase=0, dfRot=0.0*MHz,
                               name='Single photon QND variable 21 swap length - rotating frame scan arbitray Fock - P01P12 MQ',
                               save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(swap21Length, '21 swap pulse length')]
    deps = [('Probability', '|1>', ''),
            ('Probability', '|2>', ''),
            ('Visibility', '|2> - |1>', '')
            ]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state', axes, deps, measure=measure, kw=kw)

    sl = q0['noonSwapLenCs']

    def func(server, curr):

         # measure Pe

         start = 0
         # dfRot=q1['f21']-q1['fResC']
         ph = phase + 2*np.pi*dfRot[GHz]*curr[ns]
         # do FFT and define dfRot
         # ph = phase

         q0.xy = env.NOTHING
         q0.z = env.NOTHING

         for i in range(n-1):

             q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
             start += q0['piLen']/2
             q0.z += env.rect(start, sl[i], q0.noonSwapAmpC)
             start += sl[i]+q0['piLen']/2+delay

         q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
         start += q0['piLen']/2
         q0.z += env.rect(start, sl[n-1], q0.noonSwapAmpC)
         start += sl[n-1]-q0['piLen']/2+delay
         # generate Fock states in res C

#         q0.z += env.rect(start+q0['piLen']/2, curr, q0.noonSwapAmpCRead)
#
#         start += q0['piLen']/2+curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay+2.0*ns

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z = env.rect(start, curr, q1.noonSwapAmpC21)
         start += curr+q1['piLen']/2
         # CPHASE gate

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2
         # pre-measurement pulse

         q1.z += eh.measurePulse(q1, start)
         q1['readout'] = True
         req0 = runQubits(server, qubits, stats, probs=[1])

#         q0.z += env.rect(start, curr, q0.noonSwapAmpCRead)
#
#         start += curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         # measure P2

         start = 0
         # dfRot=q1['f21']-q1['fResC']
         ph = phase + 2*np.pi*dfRot[GHz]*curr[ns]
         # do FFT and define dfRot
         # ph = phase

         q0.xy = env.NOTHING
         q0.z = env.NOTHING

         for i in range(n-1):

             q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
             start += q0['piLen']/2
             q0.z += env.rect(start, sl[i], q0.noonSwapAmpC)
             start += sl[i]+q0['piLen']/2+delay

         q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
         start += q0['piLen']/2
         q0.z += env.rect(start, sl[n-1], q0.noonSwapAmpC)
         start += sl[n-1]-q0['piLen']/2+delay
         # generate Fock states in res C

#         q0.z += env.rect(start+q0['piLen']/2, curr, q0.noonSwapAmpCRead)
#
#         start += q0['piLen']/2+curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay+2.0*ns

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z = env.rect(start, curr, q1.noonSwapAmpC21)
         start += curr+q1['piLen']/2
         # CPHASE gate

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2
         # pre-measurement pulse

         q1.z += eh.measurePulse2(q1, start)
         q1['readout'] = True
         req1 = runQubits(server, qubits, stats, probs=[1])

#         q0.z += env.rect(start, curr, q0.noonSwapAmpCRead)
#
#         start += curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         # return runQubits(server, qubits, stats=stats, probs=[1])
         probs = yield FutureList([req0, req1])
         p0, p1 = [p[0] for p in probs]

         returnValue([p0, p1, p0-p1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def QuResCPHASEPreRotArb0p1P01P12(sample, swap21Length=st.arangePQ(0,120,0.1,ns), measure=0, stats=1200L,
                                  delay=0.0*ns, delayA=0.0*ns, phase=0, dfRot=0.0*MHz,
                                  name='Single photon QND variable 21 swap length - rotating frame Fock 0+1 - P01P12 MQ',
                                  save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(swap21Length, '21 swap pulse length')]
    deps = [('Probability', '|1>', ''),
            ('Probability', '|2>', ''),
            ('Visibility', '|2> - |1>', '')
            ]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+' Fock state 0+1', axes, deps, measure=measure, kw=kw)

    def func(server, curr):

         # measure Pe

         start = 0
         # dfRot=q1['f21']-q1['fResC']
         ph = phase + 2*np.pi*dfRot[GHz]*curr[ns]
         # do FFT and define dfRot
         # ph = phase

         q0.xy = eh.mix(q0, eh.piHalfPulse(q0, start))
         start += q0['piLen']/2+delay

         q0.z = env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
         start += q0.noonSwapLen0s[0]-q0['piLen']/2+delay
         # 0+1 photons in resC

#         q0.z += env.rect(start+q0['piLen']/2, curr, q0.noonSwapAmpCRead)
#
#         start += q0['piLen']/2+curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay+2.0*ns

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z = env.rect(start, curr, q1.noonSwapAmpC21)
         start += curr+q1['piLen']/2
         # CPHASE gate

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2
         # pre-measurement pulse

         q1.z += eh.measurePulse(q1, start)
         q1['readout'] = True
         req0 = runQubits(server, qubits, stats, probs=[1])

#         q0.z += env.rect(start, curr, q0.noonSwapAmpCRead)
#
#         start += curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         # measure P2

         start = 0
         # dfRot=q1['f21']-q1['fResC']
         ph = phase + 2*np.pi*dfRot[GHz]*curr[ns]
         # do FFT and define dfRot
         # ph = phase

         q0.xy = eh.mix(q0, eh.piHalfPulse(q0, start))
         start += q0['piLen']/2+delay

         q0.z = env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
         start += q0.noonSwapLen0s[0]-q0['piLen']/2+delay
         # 0+1 photons in resC

#         q0.z += env.rect(start+q0['piLen']/2, curr, q0.noonSwapAmpCRead)
#
#         start += q0['piLen']/2+curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay+2.0*ns

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z = env.rect(start, curr, q1.noonSwapAmpC21)
         start += curr+q1['piLen']/2
         # CPHASE gate

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2
         # pre-measurement pulse

         q1.z += eh.measurePulse2(q1, start)
         q1['readout'] = True
         req1 = runQubits(server, qubits, stats, probs=[1])

#         q0.z += env.rect(start, curr, q0.noonSwapAmpCRead)
#
#         start += curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         # return runQubits(server, qubits, stats=stats, probs=[1])
         probs = yield FutureList([req0, req1])
         p0, p1 = [p[0] for p in probs]

         returnValue([p0, p1, p0-p1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

#def twoQubitCPHASEPreRotArbP01P12(sample, n=1, swap21Length=st.arangePQ(0,120,0.1,ns), measure=[0,1], stats=1200L,
#                                  delay=0.0*ns, delayA=0.0*ns, phase=0, dfRot0=0.0*MHz, dfRot1=0.0*MHz,
#                                  name='Two-qubit CPHASE variable 21 swap length - rotating frame - arbitray Fock MQ',
#                                  save=True, collect=False, noisy=True):
#    sample, qubits = util.loadQubits(sample)
#    q0 = qubits[measure[0]]
#    q1 = qubits[measure[1]]
#
#    axes = [(swap21Length, '21 swap pulse length')]
##    deps = [('Probability', '|1>', ''),
##            ('Probability', '|2>', ''),
##            ('Probability', '|2> - |1>', '')
##            ]
#    kw = {'stats': stats}
##    dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state', axes, deps, measure=measure, kw=kw)
#    dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state', axes, measure=measure, kw=kw)
#
#    sl = q0['noonSwapLenCs']
#
#    def func(server, curr):
#
#         # measure Pe
#
#         start = 0
#         # dfRot=q1['f21']-q1['fResC']
#         ph0 = phase + 2*np.pi*dfRot0[GHz]*curr[ns]
#         ph1 = phase + 2*np.pi*dfRot1[GHz]*curr[ns]
#         # do FFT and define dfRot
#         # ph = phase
#
#         q0.xy = env.NOTHING
#         q0.z = env.NOTHING
#
#         for i in range(n-1):
#
#             q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
#             start += q0['piLen']/2
#             q0.z += env.rect(start, sl[i], q0.noonSwapAmpC)
#             start += sl[i]+q0['piLen']/2+delay
#
#         q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
#         start += q0['piLen']/2
#         q0.z += env.rect(start, sl[n-1], q0.noonSwapAmpC)
#         start += sl[n-1]-q0['piLen']/2+delay
#         # generate Fock states in res C
#
##         q0.xy = eh.mix(q0, eh.piHalfPulse(q0, start))
##         start += q0['piLen']/2+delay
##         # Hadamard on q0
##
##         q0.z = env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
##         start += q0.noonSwapLenCs[0]-q0['piLen']/2+delay
##         # 0+1 photons in resC
#
##         q0.z += env.rect(start+q0['piLen']/2, curr, q0.noonSwapAmpCRead)
##
##         start += q0['piLen']/2+curr
##         q0.z += eh.measurePulse(q0, start)
##
##         q0['readout'] = True
#
#         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
#         start += q1['piLen']/2+delay+2.0*ns
#         # Hadamard on q1
#
#         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
#         q1.z = env.rect(start, curr, q1.noonSwapAmpC21)
#         start += curr+q1['piLen']/2
#         # CPHASE gate
#
#         q0.z += env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
#         start += q0.noonSwapLenCs[0]+delay+2.0*ns
#         # back mapping q0
#
#         # q0.xy += eh.mix(q0, eh.piHalfPulse(q0, start, phase=ph0))
#         # pre-measurement pulse on q0
#         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph1))
#         # pre-measurement pulse on q1
#         start += q1['piLen']/2
#
#         q0.z += eh.measurePulse(q0, start)
#         q1.z += eh.measurePulse(q1, start)
#         q0['readout'] = True
#         q1['readout'] = True
#
#         return runQubits(server, qubits, stats=stats)
#    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
##
##         req0 = runQubits(server, qubits, stats)
##
###         q0.z += env.rect(start, curr, q0.noonSwapAmpCRead)
###
###         start += curr
###         q0.z += eh.measurePulse(q0, start)
###
###         q0['readout'] = True
##
##         # measure P2
##
##         start = 0
##         # dfRot=q1['f21']-q1['fResC']
##         ph0 = phase + 2*np.pi*dfRot0[GHz]*curr[ns]
##         ph1 = phase + 2*np.pi*dfRot1[GHz]*curr[ns]
##         # do FFT and define dfRot
##         # ph = phase
##
##         q0.xy = eh.mix(q0, eh.piHalfPulse(q0, start))
##         start += q0['piLen']/2+delay
##         # Hadamard on q0
##
##         q0.z = env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
##         start += q0.noonSwapLenCs[0]-q0['piLen']/2+delay
##         # 0+1 photons in resC
##
###         q0.z += env.rect(start+q0['piLen']/2, curr, q0.noonSwapAmpCRead)
###
###         start += q0['piLen']/2+curr
###         q0.z += eh.measurePulse(q0, start)
###
###         q0['readout'] = True
##
##         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
##         start += q1['piLen']/2+delay+2.0*ns
##         # Hadamard on q1
##
##         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
##         q1.z = env.rect(start, curr, q1.noonSwapAmpC21)
##         start += curr+q1['piLen']/2
##         # CPHASE gate
##
##         q0.z += env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
##         start += q0.noonSwapLenCs[0]+delay+2.0*ns
##         # back mapping q0
##
##         q0.xy += eh.mix(q0, eh.piHalfPulse(q0, start, phase=ph0))
##         # pre-measurement pulse on q0
##         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph1))
##         # pre-measurement pulse on q1
##         start += q1['piLen']/2
##
##         q0.z += eh.measurePulse2(q0, start)
##         q1.z += eh.measurePulse2(q1, start)
##         q0['readout'] = True
##         q1['readout'] = True
##         req1 = runQubits(server, qubits, stats)
##
###         q0.z += env.rect(start, curr, q0.noonSwapAmpCRead)
###
###         start += curr
###         q0.z += eh.measurePulse(q0, start)
###
###         q0['readout'] = True
##
##         # return runQubits(server, qubits, stats=stats, probs=[1])
##         probs = yield FutureList([req0, req1])
##         # pdb.set_trace()
##         p0, p1 = [p[0] for p in probs]
##
##         returnValue([p0, p1, p0-p1])
##    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
##    return

#def twoQubitCPHASEPreRotArbTOMO(sample, repetition=10, measure=[1,0], stats=1500L, delay=0.0*ns, delayA=0.0*ns,
#                                CPHASETime=None*ns, phase=0, dfRot0=0.0*GHz, dfRot1=0.0*GHz,
#                                name='Two-Qubit CPHASE Pre Rot 0+1 TOMO MQ', save=True, collect=False, noisy=True):
#    sample, qubits = util.loadQubits(sample)
#    q0 = qubits[measure[0]]
#    q1 = qubits[measure[1]]
#
#    measurement = pyle.dataking.measurement.Tomo(2)
#
#    repetition = range(repetition)
#    axes = [(repetition, 'repetition')]
#    kw = {'stats': stats}
#    dataset = sweeps.prepDataset(sample, name+' Fock state 0+1', axes, measure=measurement, kw=kw)
#
#    sl = q0['noonSwapLenCs']
#
#    if CPHASETime == None*ns:
#
#         CPHASETime = 2.0*q1.noonSwapLenC21s[0]
#
#    def func(server, curr):
#
#         start = 0
#         # dfRot=q1['f21']-q1['fResC']
#         ph0 = phase + 2*np.pi*dfRot0[GHz]*CPHASETime[ns]
#         ph1 = phase + 2*np.pi*dfRot1[GHz]*CPHASETime[ns]
#         # do FFT and define dfRot
#         # ph = phase
#
##         q0.xy = eh.mix(q0, eh.piHalfPulse(q0, start))
##         start += q0['piLen']/2+delay
##         # Hadamard on q0
##
##         q0.z = env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
##         start += q0.noonSwapLenCs[0]-q0['piLen']/2+delay
##         # 0+1 photons in resC
#
##         q0.z += env.rect(start+q0['piLen']/2, curr, q0.noonSwapAmpCRead)
##
##         start += q0['piLen']/2+curr
##         q0.z += eh.measurePulse(q0, start)
##
##         q0['readout'] = True
#
#         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
#         start += q1['piLen']/2+delay+2.0*ns
#         # Hadamard on q1
#
#         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
#         q1.z = env.rect(start, CPHASETime, q1.noonSwapAmpC21)
#         start += CPHASETime+q1['piLen']/2
#         # CPHASE gate
#
#         q0.z = env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
#         start += q0.noonSwapLenCs[0]+delay+2.0*ns
#         # back mapping q0
#
#         # q0.xy += eh.mix(q0, eh.piHalfPulse(q0, start, phase=ph0))
#         # pre-measurement pulse on q0
#         # q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph1))
#         # pre-measurement pulse on q1
#         # start += q1['piLen']/2
#
#         return measurement(server, qubits, start, **kw)
#
#    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
#
#    # pdb.set_trace()
#
#    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
#    result = np.sum(result,axis=0)/len(repetition)
#    Qk = np.reshape(result[1:],(9,4))
#    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])
#
#    rho_cal = tomo.qst(Qk,'tomo2')
#    plotRhoSingle(rho_cal,figNo=100)
#    pylab.title('Exp.')
#    rho_caln = rho_cal.copy()
#    rho_caln[1,2] = abs(rho_caln[1,2])
#    rho_caln[2,1] = abs(rho_caln[2,1])
#    print 'Fidelity is %g ' % np.trace(np.dot(rho_caln,rho_ideal))
#
#    Us =tomo._qst_transforms['tomo2'][0]
#    rho_calLike = tomo.qst_mle(Qk,Us,rho0 = rho_ideal)
#    rho_calLiken = rho_calLike.copy()
#    rho_calLiken[1,2] = abs(rho_calLiken[1,2])
#    rho_calLiken[2,1] = abs(rho_calLiken[2,1])
#    plotRhoSingle(rho_calLike,figNo=101)
#    pylab.title('Exp. likely')
#    print 'Fidelity for likelyhood is %g ' % np.trace(np.dot(rho_calLiken,rho_ideal))
#    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal)), rho_calLike, np.trace(np.dot(rho_calLiken,rho_ideal))

def QuResCPHASEPreRotArbP01P12TwoD(sample, n=1, swap21Length=st.arangePQ(0,120,0.1,ns), phase=np.arange(0,2*np.pi,np.pi/50),
                                   measure=0, stats=1200L, delay=0.0*ns, delayA=0.0*ns, dfRot=0.0*MHz,
                                   name='Single photon QND variable 21 swap length - rotating frame arbitray Fock - P01P12 2D MQ',
                                   save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(swap21Length, '21 swap pulse length'), (phase, 'Extra pi-half pulse phase')]
    deps = [('Probability', '|1>', ''),
            ('Probability', '|2>', ''),
            ('Visibility', '|2> - |1>', '')
            ]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state', axes, deps, measure=measure, kw=kw)

    sl = q0['noonSwapLenCs']

    def func(server, currLen, currPh):

         # measure Pe

         start = 0
         # dfRot=q1['f21']-q1['fResC']
         ph = currPh + 2*np.pi*dfRot[GHz]*currLen[ns]
         # do FFT and define dfRot
         # ph = phase

         q0.xy = env.NOTHING
         q0.z = env.NOTHING

         for i in range(n-1):

             q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
             start += q0['piLen']/2
             q0.z += env.rect(start, sl[i], q0.noonSwapAmpC)
             start += sl[i]+q0['piLen']/2+delay

         q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
         start += q0['piLen']/2
         q0.z += env.rect(start, sl[n-1], q0.noonSwapAmpC)
         start += sl[n-1]-q0['piLen']/2+delay
         # generate Fock states in res C

#         q0.z += env.rect(start+q0['piLen']/2, curr, q0.noonSwapAmpCRead)
#
#         start += q0['piLen']/2+curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay+2.0*ns

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z = env.rect(start, currLen, q1.noonSwapAmpC21)
         start += currLen+q1['piLen']/2
         # CPHASE gate

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2
         # pre-measurement pulse

         q1.z += eh.measurePulse(q1, start)
         q1['readout'] = True
         req0 = runQubits(server, qubits, stats, probs=[1])

#         q0.z += env.rect(start, curr, q0.noonSwapAmpCRead)
#
#         start += curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         # measure P2

         start = 0
         # dfRot=q1['f21']-q1['fResC']
         ph = currPh + 2*np.pi*dfRot[GHz]*currLen[ns]
         # do FFT and define dfRot
         # ph = phase

         q0.xy = env.NOTHING
         q0.z = env.NOTHING

         for i in range(n-1):

             q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
             start += q0['piLen']/2
             q0.z += env.rect(start, sl[i], q0.noonSwapAmpC)
             start += sl[i]+q0['piLen']/2+delay

         q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
         start += q0['piLen']/2
         q0.z += env.rect(start, sl[n-1], q0.noonSwapAmpC)
         start += sl[n-1]-q0['piLen']/2+delay
         # generate Fock states in res C

#         q0.z += env.rect(start+q0['piLen']/2, curr, q0.noonSwapAmpCRead)
#
#         start += q0['piLen']/2+curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay+2.0*ns

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z = env.rect(start, currLen, q1.noonSwapAmpC21)
         start += currLen+q1['piLen']/2
         # CPHASE gate

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2
         # pre-measurement pulse

         q1.z += eh.measurePulse2(q1, start)
         q1['readout'] = True
         req1 = runQubits(server, qubits, stats, probs=[1])

#         q0.z += env.rect(start, curr, q0.noonSwapAmpCRead)
#
#         start += curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         # return runQubits(server, qubits, stats=stats, probs=[1])
         probs = yield FutureList([req0, req1])
         p0, p1 = [p[0] for p in probs]

         returnValue([p0, p1, p0-p1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def QuResCPHASEPreRotArb0p1P01P12TwoD(sample, swap21Length=st.arangePQ(0,171,0.2,ns), phase=np.arange(0,2*np.pi,np.pi/50),
                                      measure=0, stats=900L, delay=0.0*ns, delayA=0.0*ns, dfRot=0.0*MHz,
                                      name='Single photon QND variable 21 swap length - rotating frame Fock 0+1 - P01P12 2D MQ',
                                      save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(swap21Length, '21 swap pulse length'), (phase, 'Extra pi-half pulse phase')]
    deps = [('Probability', '|1>', ''),
            ('Probability', '|2>', ''),
            ('Visibility', '|2> - |1>', '')
            ]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+' '+' Fock state 0+1', axes, deps, measure=measure, kw=kw)

    sl = q0['noonSwapLenCs']

    def func(server, currLen, currPh):

         # measure Pe

         start = 0
         # dfRot=q1['f21']-q1['fResC']
         ph = currPh + 2*np.pi*dfRot[GHz]*currLen[ns]
         # do FFT and define dfRot
         # ph = phase

         q0.xy = eh.mix(q0, eh.piHalfPulse(q0, start))
         start += q0['piLen']/2+delay

         q0.z = env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
         start += q0.noonSwapLen0s[0]-q0['piLen']/2+delay
         # 0+1 photons in resC

#         q0.z += env.rect(start+q0['piLen']/2, curr, q0.noonSwapAmpCRead)
#
#         start += q0['piLen']/2+curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay+2.0*ns

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z = env.rect(start, currLen, q1.noonSwapAmpC21)
         start += currLen+q1['piLen']/2
         # CPHASE gate

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2
         # pre-measurement pulse

         q1.z += eh.measurePulse(q1, start)
         q1['readout'] = True
         req0 = runQubits(server, qubits, stats, probs=[1])

#         q0.z += env.rect(start, curr, q0.noonSwapAmpCRead)
#
#         start += curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         # measure P2

         start = 0
         # dfRot=q1['f21']-q1['fResC']
         ph = currPh + 2*np.pi*dfRot[GHz]*currLen[ns]
         # do FFT and define dfRot
         # ph = phase

         q0.xy = eh.mix(q0, eh.piHalfPulse(q0, start))
         start += q0['piLen']/2+delay

         q0.z = env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
         start += q0.noonSwapLen0s[0]-q0['piLen']/2+delay
         # 0+1 photons in resC

#         q0.z += env.rect(start+q0['piLen']/2, curr, q0.noonSwapAmpCRead)
#
#         start += q0['piLen']/2+curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay+2.0*ns

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z = env.rect(start, currLen, q1.noonSwapAmpC21)
         start += currLen+q1['piLen']/2
         # CPHASE gate

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2
         # pre-measurement pulse

         q1.z += eh.measurePulse2(q1, start)
         q1['readout'] = True
         req1 = runQubits(server, qubits, stats, probs=[1])

#         q0.z += env.rect(start, curr, q0.noonSwapAmpCRead)
#
#         start += curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         # return runQubits(server, qubits, stats=stats, probs=[1])
         probs = yield FutureList([req0, req1])
         p0, p1 = [p[0] for p in probs]

         returnValue([p0, p1, p0-p1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def QuResCPHASEPreRotArbP01P12ZeroTwoD(sample, n=0, swap21Length=st.arangePQ(0,121,0.2,ns), phase=np.arange(0,2*np.pi,np.pi/50),
                                       measure=0, stats=1200L, delay=0.0*ns, delayA=0.0*ns, dfRot=0.0*MHz,
                                       name='Single photon QND variable 21 swap length - rotating frame Fock 0 - P01P12 2D MQ',
                                       save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(swap21Length, '21 swap pulse length'), (phase, 'Extra pi-half pulse phase')]
    deps = [('Probability', '|1>', ''),
            ('Probability', '|2>', ''),
            ('Visibility', '|2> - |1>', '')
            ]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state', axes, deps, measure=measure, kw=kw)

    sl = q0['noonSwapLenCs']

    def func(server, currLen, currPh):

         # measure Pe

         start = 0
         # dfRot=q1['f21']-q1['fResC']
         ph = currPh + 2*np.pi*dfRot[GHz]*currLen[ns]
         # do FFT and define dfRot
         # ph = phase

#         q0.xy = env.NOTHING
#         q0.z = env.NOTHING
#
#         for i in range(n-1):
#
#             q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
#             start += q0['piLen']/2
#             q0.z += env.rect(start, sl[i], q0.noonSwapAmpC)
#             start += sl[i]+q0['piLen']/2+delay
#
#         q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
#         start += q0['piLen']/2
#         q0.z += env.rect(start, sl[n-1], q0.noonSwapAmpC)
#         start += sl[n-1]-q0['piLen']/2+delay
#         # generate Fock states in res C

#         q0.z += env.rect(start+q0['piLen']/2, curr, q0.noonSwapAmpCRead)
#
#         start += q0['piLen']/2+curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay+2.0*ns

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z = env.rect(start, currLen, q1.noonSwapAmpC21)
         start += currLen+q1['piLen']/2
         # CPHASE gate

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2
         # pre-measurement pulse

         q1.z += eh.measurePulse(q1, start)
         q1['readout'] = True
         req0 = runQubits(server, qubits, stats, probs=[1])

#         q0.z += env.rect(start, curr, q0.noonSwapAmpCRead)
#
#         start += curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         # measure P2

         start = 0
         # dfRot=q1['f21']-q1['fResC']
         ph = currPh + 2*np.pi*dfRot[GHz]*currLen[ns]
         # do FFT and define dfRot
         # ph = phase

#         q0.xy = env.NOTHING
#         q0.z = env.NOTHING
#
#         for i in range(n-1):
#
#             q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
#             start += q0['piLen']/2
#             q0.z += env.rect(start, sl[i], q0.noonSwapAmpC)
#             start += sl[i]+q0['piLen']/2+delay
#
#         q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
#         start += q0['piLen']/2
#         q0.z += env.rect(start, sl[n-1], q0.noonSwapAmpC)
#         start += sl[n-1]-q0['piLen']/2+delay
#         # generate Fock states in res C

#         q0.z += env.rect(start+q0['piLen']/2, curr, q0.noonSwapAmpCRead)
#
#         start += q0['piLen']/2+curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay+2.0*ns

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z = env.rect(start, currLen, q1.noonSwapAmpC21)
         start += currLen+q1['piLen']/2
         # CPHASE gate

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2
         # pre-measurement pulse

         q1.z += eh.measurePulse2(q1, start)
         q1['readout'] = True
         req1 = runQubits(server, qubits, stats, probs=[1])

#         q0.z += env.rect(start, curr, q0.noonSwapAmpCRead)
#
#         start += curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         # return runQubits(server, qubits, stats=stats, probs=[1])
         probs = yield FutureList([req0, req1])
         p0, p1 = [p[0] for p in probs]

         returnValue([p0, p1, p0-p1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def CPHASECompPulse0Arb(sample, n=1, compAmp=np.arange(-0.06,0.06,0.001), measure=0, stats=1200L,
                        CPHASETime=None*ns, compPulseLen=15.0*ns, delay=0.0*ns, delayA=0.0*ns, phase=0,
                        name='CPHASE Compensation Pulse 0 arbitrary Fock states', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(compAmp, 'compensation pulse amplitude')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state', axes, measure=measure, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

    def func(server, curr):

         # measure Pe

         start = 0

         ph = phase

         q0.xy = env.NOTHING
         q0.z = env.NOTHING

         for i in range(n-1):

             q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
             start += q0['piLen']/2
             q0.z += env.rect(start, sl[i], q0.noonSwapAmpC)
             start += sl[i]+q0['piLen']/2+delay

         q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
         start += q0['piLen']/2
         q0.z += env.rect(start, sl[n-1], q0.noonSwapAmpC)
         start += sl[n-1]-q0['piLen']/2+delay
         # generate Fock states in res C

#         q0.z += env.rect(start+q0['piLen']/2, curr, q0.noonSwapAmpCRead)
#
#         start += q0['piLen']/2+curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay+2.0*ns

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z = env.rect(start, CPHASETime, q1.noonSwapAmpC21)
         start += CPHASETime
         # CPHASE gate

         q1.z += env.rect(start, compPulseLen, curr)
         start += compPulseLen+q1['piLen']/2
         # compensation pulse 0

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2
         # pre-measurement pulse

         q1.z += eh.measurePulse(q1, start)
         q1['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def CPHASECompPulse1Arb(sample, n=1, compAmp=np.arange(-0.06,0.06,0.001), measure=0, stats=1200L,
                        CPHASETime=None*ns, compPulseLen=15.0*ns, compPulseAmp0=-0.0351,
                        delay=0.0*ns, delayA=0.0*ns, phase=0,
                        name='CPHASE Compensation Pulse 1 arbitrary Fock states', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(compAmp, 'compensation pulse amplitude')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state', axes, measure=measure, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

    def func(server, curr):

         # measure Pe

         start = 0

         ph = phase

         q0.xy = eh.mix(q0, eh.piHalfPulse(q0, start))
         start += q0['piLen']/2+delay+2.0*ns

#         q0.z += env.rect(start+q0['piLen']/2, curr, q0.noonSwapAmpCRead)
#
#         start += q0['piLen']/2+curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z = env.rect(start, CPHASETime, q1.noonSwapAmpC21)
         start += CPHASETime
         # CPHASE gate

         q1.z += env.rect(start, compPulseLen, compPulseAmp0)
         q0.z = env.rect(start, compPulseLen, curr)
         start += compPulseLen+q0['piLen']/2
         # compensation pulse 0+1

         q0.xy += eh.mix(q0, eh.piHalfPulse(q0, start, phase=ph))
         start += q0['piLen']/2
         # pre-measurement pulse

         q0.z += eh.measurePulse(q0, start)
         q0['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def CPHASEGateCompPulseTomo(sample, repetition=10, measure=[0,1], n=1, stats=1500L,
                            CPHASETime=None*ns, compPulseLen=15.0*ns, compPulseAmp0=-0.0351, compPulseAmp1=-0.0167,
                            delay=0.0*ns, delayA=0.0*ns, phase=0,
                            name='CPHASE-gate compensation pulse TOMO MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    measurement = pyle.dataking.measurement.Tomo(2)

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state', axes, measure=measurement, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

    def func(server, curr):

         start = 0

         ph = phase

         q0.xy = env.NOTHING
         q0.z = env.NOTHING

         for i in range(n-1):

             q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
             start += q0['piLen']/2
             q0.z += env.rect(start, sl[i], q0.noonSwapAmpC)
             start += sl[i]+q0['piLen']/2+delay

         q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
         start += q0['piLen']/2
         q0.z += env.rect(start, sl[n-1], q0.noonSwapAmpC)
         start += sl[n-1]-q0['piLen']/2+delay
         # generate Fock states in res C

#         q0.z += env.rect(start+q0['piLen']/2, curr, q0.noonSwapAmpCRead)
#
#         start += q0['piLen']/2+curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay+2.0*ns

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z = env.rect(start, CPHASETime, q1.noonSwapAmpC21)
         start += CPHASETime
         # CPHASE gate

         q0.z += env.rect(start, compPulseLen, compPulseAmp1)
         q1.z += env.rect(start, compPulseLen, compPulseAmp0)
         start += compPulseLen+q0['piLen']/2
         # compensation pulse 0

         q0.xy += eh.mix(q0, eh.piHalfPulse(q0, start, phase=ph))
         start += q0['piLen']/2
         # pre-measurement pulse

         return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)
    Qk = np.reshape(result[1:],(9,4))
    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    rho_cal = tomo.qst(Qk,'tomo2')
    plotRhoSingle(rho_cal,figNo=100)
    pylab.title('Exp.')
    rho_caln = rho_cal.copy()
    rho_caln[1,2] = abs(rho_caln[1,2])
    rho_caln[2,1] = abs(rho_caln[2,1])
    print 'Fidelity is %g ' % np.trace(np.dot(rho_caln,rho_ideal))

    Us =tomo._qst_transforms['tomo2'][0]
    rho_calLike = tomo.qst_mle(Qk,Us,rho0 = rho_ideal)
    rho_calLiken = rho_calLike.copy()
    rho_calLiken[1,2] = abs(rho_calLiken[1,2])
    rho_calLiken[2,1] = abs(rho_calLiken[2,1])
    plotRhoSingle(rho_calLike,figNo=101)
    pylab.title('Exp. likely')
    print 'Fidelity for likelyhood is %g ' % np.trace(np.dot(rho_calLiken,rho_ideal))
    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal)), rho_calLike, np.trace(np.dot(rho_calLiken,rho_ideal))

def CPHASEGateCompPulseTomoTest(sample, repetition=10, measure=[0,1], n=1, stats=1500L,
                                CPHASETime=None*ns, compPulseLen=15.0*ns, compPulseAmp0=-0.0351, compPulseAmp1=-0.0167,
                                delay=0.0*ns, delayA=0.0*ns, phase=0,
                                name='CPHASE-gate compensation pulse TOMO MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    measurement = pyle.dataking.measurement.Tomo(2)

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state', axes, measure=measurement, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

    def func(server, curr):

         start = 0

         ph = phase

         q0.xy = env.NOTHING
         q0.z = env.NOTHING

         for i in range(n-1):

             q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
             start += q0['piLen']/2
             q0.z += env.rect(start, sl[i], q0.noonSwapAmpC)
             start += sl[i]+q0['piLen']/2+delay

         q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
         start += q0['piLen']/2
         q0.z += env.rect(start, sl[n-1], q0.noonSwapAmpC)
         start += sl[n-1]-q0['piLen']/2+delay
         # generate Fock states in res C

#         q0.z += env.rect(start+q0['piLen']/2, curr, q0.noonSwapAmpCRead)
#
#         start += q0['piLen']/2+curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay+2.0*ns

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z = env.rect(start, CPHASETime, q1.noonSwapAmpC21)
         start += CPHASETime
         # CPHASE gate

         q0.z += env.rect(start, compPulseLen, compPulseAmp1)
         q1.z += env.rect(start, compPulseLen, compPulseAmp0)
         start += compPulseLen+q0['piLen']/2
         # compensation pulse 0

         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC)
         start += sl[0]

         q0.xy += eh.mix(q0, eh.piHalfPulse(q0, start, phase=ph))
         start += q0['piLen']/2
         # pre-measurement pulse

         return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)
    Qk = np.reshape(result[1:],(9,4))
    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    rho_cal = tomo.qst(Qk,'tomo2')
    plotRhoSingle(rho_cal,figNo=100)
    pylab.title('Exp.')
    rho_caln = rho_cal.copy()
    rho_caln[1,2] = abs(rho_caln[1,2])
    rho_caln[2,1] = abs(rho_caln[2,1])
    print 'Fidelity is %g ' % np.trace(np.dot(rho_caln,rho_ideal))

    Us =tomo._qst_transforms['tomo2'][0]
    rho_calLike = tomo.qst_mle(Qk,Us,rho0 = rho_ideal)
    rho_calLiken = rho_calLike.copy()
    rho_calLiken[1,2] = abs(rho_calLiken[1,2])
    rho_calLiken[2,1] = abs(rho_calLiken[2,1])
    plotRhoSingle(rho_calLike,figNo=101)
    pylab.title('Exp. likely')
    print 'Fidelity for likelyhood is %g ' % np.trace(np.dot(rho_calLiken,rho_ideal))
    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal)), rho_calLike, np.trace(np.dot(rho_calLiken,rho_ideal))

def CPHASECompPulseQ1Arb(sample, compAmp=np.arange(-0.06,0.06,0.001), measure=0, stats=1200L,
                         CPHASETime=None*ns, compPulseLen=10.0*ns, delay=0.0*ns, delayA=0.0*ns, phase=0,
                         name='CPHASE Compensation Pulse qubit 1 Fock state 1', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(compAmp, 'compensation pulse amplitude')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

    def func(server, curr):

         # measure Pe

         start = 0

         ph = phase

         q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
         # excite q0 from g to e

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+2.0*ns

         # START CZ GATE
         q0.z = env.rect(start, sl[0], q0.noonSwapAmpC)
         start += sl[0]
         # generate Fock state 1 in rc

         q1.z = env.rect(start, CPHASETime, q1.noonSwapAmpC21)
         start += CPHASETime
         # CZ gate

         q1.z += env.rect(start, compPulseLen, curr)
         start += compPulseLen
         # compensation pulse q1

         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC)
         start += sl[0]+q1['piLen']/2
         # END CZ GATE

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2
         # pre-measurement pulse

         q1.z += eh.measurePulse(q1, start)
         q1['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def CPHASECompPulseQ0Arb(sample, compAmp=np.arange(-0.06,0.06,0.001), measure=0, stats=1200L,
                         CPHASETime=None*ns, compPulseLen=10.0*ns, compPulseAmpQ1=0.051, delay=0.0*ns, delayA=0.0*ns, phase=0,
                         name='CPHASE Compensation Pulse qubit 0 Fock state 1', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(compAmp, 'compensation pulse amplitude')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

    def func(server, curr):

         # measure Pe

         start = 0

         ph = phase

         q0.xy = eh.mix(q0, eh.piHalfPulse(q0, start))
         start += q0['piLen']/2+2.0*ns
         # q0 Ramsey pulse

         # START CZ GATE
         q0.z = env.rect(start, 2*sl[0], q0.noonSwapAmpC)
         start += 2*sl[0]
         # generate Fock state 1 in rc

         q1.z = env.rect(start, CPHASETime, q1.noonSwapAmpC21)
         start += CPHASETime
         # CZ gate

         q1.z += env.rect(start, compPulseLen, compPulseAmpQ1)
         # compensation pulse q1

         q0.z += env.rect(start, compPulseLen, curr)
         start += compPulseLen+q0['piLen']/2
         # compensation pulse q0

#         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC)
#         start += sl[0]+q0['piLen']/2
         # END CZ GATE

         q0.xy += eh.mix(q0, eh.piHalfPulse(q0, start, phase=ph))
         start += q0['piLen']/2
         # pre-measurement pulse

         q0.z += eh.measurePulse(q0, start)
         q0['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def CPHASEGateCompPulseTomoFast(sample, repetition=10, measure=[0,1], n=1, stats=1500L,
                                CPHASETime=None*ns, compPulseLen=15.0*ns, compPulseAmp0=-0.0351, compPulseAmp1=-0.0167,
                                delay=0.0*ns, delayA=0.0*ns, phase=0,
                                name='CPHASE-gate compensation pulse TOMO MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    measurement = pyle.dataking.measurement.Tomo(2)

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state', axes, measure=measurement, kw=kw)

    sl = q0['noonSwapLenCs']

    if CPHASETime == None*ns:

         CPHASETime = 2.0*q1.noonSwapLenC21s[0]

    def func(server, curr):

         start = 0

         ph = phase

         q0.xy = env.NOTHING
         q0.z = env.NOTHING

         for i in range(n-1):

             q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
             start += q0['piLen']/2
             q0.z += env.rect(start, sl[i], q0.noonSwapAmpC)
             start += sl[i]+q0['piLen']/2+delay

         q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
         start += q0['piLen']/2
         q0.z += env.rect(start, sl[n-1], q0.noonSwapAmpC)
         start += sl[n-1]-q0['piLen']/2+delay
         # generate Fock states in res C

#         q0.z += env.rect(start+q0['piLen']/2, curr, q0.noonSwapAmpCRead)
#
#         start += q0['piLen']/2+curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z = env.rect(start, CPHASETime, q1.noonSwapAmpC21)
         start += CPHASETime
         # CPHASE gate

         q0.z += env.rect(start, compPulseLen, compPulseAmp1)
         q1.z += env.rect(start, compPulseLen, compPulseAmp0)
         start += compPulseLen+q0['piLen']/2
         # compensation pulse 0

         q0.z += env.rect(start, sl[0], q0.noonSwapAmpC)
         start += sl[0]

         q0.xy += eh.mix(q0, eh.piHalfPulse(q0, start, phase=ph))
         start += q0['piLen']/2
         # pre-measurement pulse

         return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)
    Qk = np.reshape(result[1:],(9,4))
    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    rho_cal = tomo.qst(Qk,'tomo2')
    plotRhoSingle(rho_cal,figNo=100)
    pylab.title('Exp.')
    rho_caln = rho_cal.copy()
    rho_caln[1,2] = abs(rho_caln[1,2])
    rho_caln[2,1] = abs(rho_caln[2,1])
    print 'Fidelity is %g ' % np.trace(np.dot(rho_caln,rho_ideal))

    Us =tomo._qst_transforms['tomo2'][0]
    rho_calLike = tomo.qst_mle(Qk,Us,rho0 = rho_ideal)
    rho_calLiken = rho_calLike.copy()
    rho_calLiken[1,2] = abs(rho_calLiken[1,2])
    rho_calLiken[2,1] = abs(rho_calLiken[2,1])
    plotRhoSingle(rho_calLike,figNo=101)
    pylab.title('Exp. likely')
    print 'Fidelity for likelyhood is %g ' % np.trace(np.dot(rho_calLiken,rho_ideal))
    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal)), rho_calLike, np.trace(np.dot(rho_calLiken,rho_ideal))

def singlePhotonQNDPreRotScanArbSQ(sample, n=1, swap21Length=st.arangePQ(0,250,1,ns), measure=1, stats=1800L,
                                   delay=0.0*ns, delayA=0.0*ns, phase=0, dfRot=0.0*GHz,
                                   name='Single photon QND variable 21 swap length - rotating frame scan arbitray Fock q1 MQ',
                                   save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[1-measure]
    q1 = qubits[measure]

    axes = [(swap21Length, '21 swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state', axes, measure=measure, kw=kw)

    sl = q1['noonSwapLen0s']

    def func(server, curr):
         start = 0
         # dfRot=q1['f21']-q1['fRes0']
         ph = phase + 2*np.pi*dfRot[GHz]*curr[ns]
         # do FFT and define dfRot
         # ph = phase

         q1.xy = env.NOTHING
         q1.z = env.NOTHING

         for i in range(n-1):

             q1.xy += eh.mix(q1, eh.piPulseHD(q1, start))
             start += q1['piLen']/2
             q1.z += env.rect(start, sl[i], q1.noonSwapAmp0)
             start += sl[i]+q1['piLen']/2+delay

         q1.xy += eh.mix(q1, eh.piPulseHD(q1, start))
         start += q1['piLen']/2
         q1.z += env.rect(start, sl[n-1], q1.noonSwapAmp0)
         start += sl[n-1]+delay+30.0*ns
         # generate Fock states in res 0

#         q1.z += env.rect(start+q1['piLen']/2, curr, q1.noonSwapAmp0Read)
#
#         start += q1['piLen']/2+curr
#         q1.z += eh.measurePulse(q1, start)
#
#         q1['readout'] = True

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z += env.rect(start, curr, q1.noonSwapAmp021)
         start += curr+q1['piLen']/2
         # CPHASE gate

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2
         # pre-measurement pulse

         q1.z += eh.measurePulse(q1, start)

         q1['readout'] = True

#         q0.z += env.rect(start, curr, q0.noonSwapAmpCRead)
#
#         start += curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def singlePhotonQNDPreRotScanArbSQReset(sample, n=1, swap21Length=st.arangePQ(0,250,1,ns), measure=1, stats=1800L,
                                        delay=0.0*ns, delayA=0.0*ns, phase=0, dfRot=0.0*GHz,
                                        name='Single photon QND variable 21 swap length - rotating frame scan arbitray Fock q1 reset MQ',
                                        save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[1-measure]
    q1 = qubits[measure]

    axes = [(swap21Length, '21 swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state', axes, measure=measure, kw=kw)

    sl = q1['noonSwapLen0s']

    def func(server, curr):
         start = 0
         # dfRot=q1['f21']-q1['fRes0']
         ph = phase + 2*np.pi*dfRot[GHz]*curr[ns]
         # do FFT and define dfRot
         # ph = phase

         q1.xy = env.NOTHING
         q1.z = env.NOTHING

         for i in range(n-1):

             q1.xy += eh.mix(q1, eh.piPulseHD(q1, start))
             start += q1['piLen']/2
             q1.z += env.rect(start, sl[i], q1.noonSwapAmp0)
             start += sl[i]+q1['piLen']/2+delay

         q1.xy += eh.mix(q1, eh.piPulseHD(q1, start))
         start += q1['piLen']/2
         q1.z += env.rect(start, sl[n-1], q1.noonSwapAmp0)
         start += sl[n-1]+delay+4.0*ns
         # generate Fock states in res 0

#         q1.z += env.rect(start+q1['piLen']/2, curr, q1.noonSwapAmp0Read)
#
#         start += q1['piLen']/2+curr
#         q1.z += eh.measurePulse(q1, start)
#
#         q1['readout'] = True

         q1.z += env.rect(start, q1.resetLens1[2], q1.resetAmps1[2])
         start += q1.resetLens1[2]+q1['piLen']/2+15.0*ns
         # reset q1

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z += env.rect(start, curr, q1.noonSwapAmp021)
         start += curr+q1['piLen']/2
         # CPHASE gate

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2
         # pre-measurement pulse

         q1.z += eh.measurePulse(q1, start)

         q1['readout'] = True

#         q0.z += env.rect(start, curr, q0.noonSwapAmpCRead)
#
#         start += curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

    alpha = np.array([1.1,1.45])[:,None] * np.exp(1.0j*np.linspace(0,2*np.pi,30,endpoint=False))[None,:]
    alpha = np.reshape(alpha, size(alpha))
#    np.plot(np.real(alpha), np.imag(alpha))
#    alpha = np.linspace(-2.0,2.0,25)
#    alpha = alpha[:,None]+1.0j*alpha[None,:]
#    alpha = np.reshape(alpha,np.size(alpha))

def complexSweepSQ(displacement, sweepTime):
    return [[d,sT] for d in displacement for sT in sweepTime]

def CPHASEPreRotArbSQ_TOMO(sample, n=1, probeLen=st.arangePQ(0,200,2,ns), CPHASETime = 42.203*ns, disp = [0.5], measure=1, stats=1800L,
                           delay=0.0*ns, delayA=0.0*ns, phase=np.pi, dfRot=99.1*MHz,
                           name='Single photon QND variable 21 swap length - rotating frame scan arbitray Fock q1 MQ TOMO',
                           save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[1-measure]
    q1 = qubits[measure]
    r1 = qubits[measure+2]

    sweepPara = complexSweepSQ(np.array(disp)/r1.noonAmpScale.value,probeLen)

    kw = {'stats': stats,
    'measure': measure}
    dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state',
       axes = [('r0 displacement', 're'),('r0 displacement', 'im'),
               ('swap pulse length', 'ns')], measure=measure, kw=kw)

    # axes = [(swap21Length, '21 swap pulse length')]
    # kw = {'stats': stats}
    # dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state', axes, measure=measure, kw=kw)

    sl = q1['noonSwapLen0s']

    def func(server, curr):
         a1 = curr[0]
         currLen = curr[1]

         start = 0
         # dfRot=q1['f21']-q1['fRes0']
         ph = phase + 2*np.pi*dfRot[GHz]*currLen[ns]
         # do FFT and define dfRot
         # ph = phase

         q1.xy = env.NOTHING
         q1.z = env.NOTHING

         for i in range(n-1):

             q1.xy += eh.mix(q1, eh.piPulseHD(q1, start))
             start += q1['piLen']/2
             q1.z += env.rect(start, sl[i], q1.noonSwapAmp0)
             start += sl[i]+q1['piLen']/2+delay

         q1.xy += eh.mix(q1, eh.piPulseHD(q1, start))
         start += q1['piLen']/2
         q1.z += env.rect(start, sl[n-1], q1.noonSwapAmp0)
         start += sl[n-1]+delay+30.0*ns
         # generate Fock states in res 0

#         q1.z += env.rect(start+q1['piLen']/2, curr, q1.noonSwapAmp0Read)
#
#         start += q1['piLen']/2+curr
#         q1.z += eh.measurePulse(q1, start)
#
#         q1['readout'] = True

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z += env.rect(start, CPHASETime, q1.noonSwapAmp021)
         start += CPHASETime+q1['piLen']/2
         # CPHASE gate

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2+2.0*ns
         # pre-measurement pulse

         q1.z += env.rect(start, q1.noonSwapLenR21s[0]+q1.noonSwapLenCs[0], q1.noonSwapAmpR21)-env.rect(start+q1.noonSwapLenR21s[0], q1.noonSwapLenCs[0], q1.noonSwapAmpR21-q1.noonSwapAmpC)
         start += q1.noonSwapLenR21s[0]+q1.noonSwapLenCs[0]+4.0*ns

#         q1.z += eh.measurePulse(q1, start)
#
#         q1['readout'] = True

#         q0.z += env.rect(start, curr, q0.noonSwapAmpCRead)
#
#         start += curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True


         r1.xy = eh.mix(r1, env.gaussian(start+r1.piLen/2, r1.piFWHM,
                                         np.conjugate(a1*r1.noonDrivePhase)), freq = 'fRes0')
         start += r1.piLen+8.0*ns

         q1.z += env.rect(start, currLen, q1.noonSwapAmp0Read)

         start += currLen
         q1.z += eh.measurePulse(q1, start)

         q1['readout'] = True

         data = yield runQubits(server, qubits, stats=stats, probs=[1])

         data = np.hstack(([a1.real, a1.imag, currLen], data))
         returnValue(data)
    result = sweeps.run(func, sweepPara, dataset=save and dataset, noisy=noisy)
    return

def CPHASEPreRotArbSQ_TOMOZero(sample, n=0, probeLen=st.arangePQ(0,200,2,ns), CPHASETime = 42.082*ns, disp = [0.5], measure=1, stats=1800L,
                               delay=0.0*ns, delayA=0.0*ns, phase=np.pi, dfRot=99.1*MHz,
                               name='Single photon QND variable 21 swap length - rotating frame scan Fock 0 q1 MQ TOMO',
                               save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[1-measure]
    q1 = qubits[measure]
    r1 = qubits[measure+2]

    sweepPara = complexSweepSQ(np.array(disp)/r1.noonAmpScale.value,probeLen)

    kw = {'stats': stats,
    'measure': measure}
    dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state',
       axes = [('r0 displacement', 're'),('r0 displacement', 'im'),
               ('swap pulse length', 'ns')], measure=measure, kw=kw)

    # axes = [(swap21Length, '21 swap pulse length')]
    # kw = {'stats': stats}
    # dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state', axes, measure=measure, kw=kw)

    sl = q1['noonSwapLen0s']

    def func(server, curr):
         a1 = curr[0]
         currLen = curr[1]

         start = 0
         # dfRot=q1['f21']-q1['fRes0']
         ph = phase + 2*np.pi*dfRot[GHz]*currLen[ns]
         # do FFT and define dfRot
         # ph = phase

         q1.xy = env.NOTHING
         q1.z = env.NOTHING

#         q1.z += env.rect(start+q1['piLen']/2, curr, q1.noonSwapAmp0Read)
#
#         start += q1['piLen']/2+curr
#         q1.z += eh.measurePulse(q1, start)
#
#         q1['readout'] = True

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z += env.rect(start, CPHASETime, q1.noonSwapAmp021)
         start += CPHASETime+q1['piLen']/2
         # CPHASE gate

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2+2.0*ns
         # pre-measurement pulse

         q1.z += env.rect(start, q1.noonSwapLenR21s[0]+q1.noonSwapLenCs[0], q1.noonSwapAmpR21)-env.rect(start+q1.noonSwapLenR21s[0], q1.noonSwapLenCs[0], q1.noonSwapAmpR21-q1.noonSwapAmpC)
         start += q1.noonSwapLenR21s[0]+q1.noonSwapLenCs[0]+4.0*ns

#         q1.z += eh.measurePulse(q1, start)
#
#         q1['readout'] = True

#         q0.z += env.rect(start, curr, q0.noonSwapAmpCRead)
#
#         start += curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True


         r1.xy = eh.mix(r1, env.gaussian(start+r1.piLen/2, r1.piFWHM,
                                         np.conjugate(a1*r1.noonDrivePhase)), freq = 'fRes0')
         start += r1.piLen+8.0*ns

         q1.z += env.rect(start, currLen, q1.noonSwapAmp0Read)

         start += currLen
         q1.z += eh.measurePulse(q1, start)

         q1['readout'] = True

         data = yield runQubits(server, qubits, stats=stats, probs=[1])

         data = np.hstack(([a1.real, a1.imag, currLen], data))
         returnValue(data)
    result = sweeps.run(func, sweepPara, dataset=save and dataset, noisy=noisy)
    return

def CPHASEPreRotArbSQ_TOMOReset(sample, n=1, probeLen=st.arangePQ(0,200,2,ns), CPHASETime = 42.203*ns, disp = [0.5], measure=1, stats=1800L,
                                delay=0.0*ns, delayA=0.0*ns, phase=np.pi, dfRot=99.1*MHz,
                                name='Single photon QND variable 21 swap length - rotating frame scan arbitray Fock q1 MQ TOMO',
                                save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[1-measure]
    q1 = qubits[measure]
    r1 = qubits[measure+2]

    sweepPara = complexSweepSQ(np.array(disp)/r1.noonAmpScale.value,probeLen)

    kw = {'stats': stats,
    'measure': measure}
    dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state',
       axes = [('r0 displacement', 're'),('r0 displacement', 'im'),
               ('swap pulse length', 'ns')], measure=measure, kw=kw)

    # axes = [(swap21Length, '21 swap pulse length')]
    # kw = {'stats': stats}
    # dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state', axes, measure=measure, kw=kw)

    sl = q1['noonSwapLen0s']

    def func(server, curr):
         a1 = curr[0]
         currLen = curr[1]

         start = 0
         # dfRot=q1['f21']-q1['fRes0']
         ph = phase + 2*np.pi*dfRot[GHz]*currLen[ns]
         # do FFT and define dfRot
         # ph = phase

         q1.xy = env.NOTHING
         q1.z = env.NOTHING

         for i in range(n-1):

             q1.xy += eh.mix(q1, eh.piPulseHD(q1, start))
             start += q1['piLen']/2
             q1.z += env.rect(start, sl[i], q1.noonSwapAmp0)
             start += sl[i]+q1['piLen']/2+delay

         q1.xy += eh.mix(q1, eh.piPulseHD(q1, start))
         start += q1['piLen']/2
         q1.z += env.rect(start, sl[n-1], q1.noonSwapAmp0)
         start += sl[n-1]+delay+4.0*ns
         # generate Fock states in res 0

#         q1.z += env.rect(start+q1['piLen']/2, curr, q1.noonSwapAmp0Read)
#
#         start += q1['piLen']/2+curr
#         q1.z += eh.measurePulse(q1, start)
#
#         q1['readout'] = True

         q1.z += env.rect(start, q1.resetLens1[2], q1.resetAmps1[2])
         start += q1.resetLens1[2]+q1['piLen']/2+15.0*ns
         # reset q1

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z += env.rect(start, CPHASETime, q1.noonSwapAmp021)
         start += CPHASETime+q1['piLen']/2
         # CPHASE gate

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2+2.0*ns
         # pre-measurement pulse

         q1.z += env.rect(start, q1.noonSwapLenR21s[0]+q1.noonSwapLenCs[0], q1.noonSwapAmpR21)-env.rect(start+q1.noonSwapLenR21s[0], q1.noonSwapLenCs[0], q1.noonSwapAmpR21-q1.noonSwapAmpC)
         start += q1.noonSwapLenR21s[0]+q1.noonSwapLenCs[0]+4.0*ns

#         q1.z += eh.measurePulse(q1, start)
#
#         q1['readout'] = True

#         q0.z += env.rect(start, curr, q0.noonSwapAmpCRead)
#
#         start += curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True


         r1.xy = eh.mix(r1, env.gaussian(start+r1.piLen/2, r1.piFWHM,
                                         np.conjugate(a1*r1.noonDrivePhase)), freq = 'fRes0')
         start += r1.piLen+8.0*ns

         q1.z += env.rect(start, currLen, q1.noonSwapAmp0Read)

         start += currLen
         q1.z += eh.measurePulse(q1, start)

         q1['readout'] = True

         data = yield runQubits(server, qubits, stats=stats, probs=[1])

         data = np.hstack(([a1.real, a1.imag, currLen], data))
         returnValue(data)
    result = sweeps.run(func, sweepPara, dataset=save and dataset, noisy=noisy)
    return

def CPHASEPreRotArbSQ0p1_TOMO(sample, n=1, probeLen=st.arangePQ(0,200,2,ns), CPHASETime = 42.203*ns, disp = [0.5], measure=1, stats=1800L,
                              delay=0.0*ns, delayA=0.0*ns, phase=np.pi, dfRot=88.2*MHz,
                              name='Single photon QND variable 21 swap length - rotating frame scan arbitray Fock q1 0p1 MQ TOMO',
                              save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[1-measure]
    q1 = qubits[measure]
    r1 = qubits[measure+2]

    sweepPara = complexSweepSQ(np.array(disp)/r1.noonAmpScale.value,probeLen)

    kw = {'stats': stats,
    'measure': measure}
    dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state',
       axes = [('r0 displacement', 're'),('r0 displacement', 'im'),
               ('swap pulse length', 'ns')], measure=measure, kw=kw)

    # axes = [(swap21Length, '21 swap pulse length')]
    # kw = {'stats': stats}
    # dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state', axes, measure=measure, kw=kw)

    sl = q1['noonSwapLen0s']

    def func(server, curr):
         a1 = curr[0]
         currLen = curr[1]

         start = 0
         # dfRot=q1['f21']-q1['fRes0']
         ph = phase + 2*np.pi*dfRot[GHz]*currLen[ns]
         # do FFT and define dfRot
         # ph = phase

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay

         q1.z = env.rect(start, q1.noonSwapLen0s[0], q1.noonSwapAmp0)
         start += q1.noonSwapLen0s[0]+30.0*ns+delay
         # 0+1 photons in res0

#         q1.z += env.rect(start+q1['piLen']/2, curr, q1.noonSwapAmp0Read)
#
#         start += q1['piLen']/2+curr
#         q1.z += eh.measurePulse(q1, start)
#
#         q1['readout'] = True

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z += env.rect(start, CPHASETime, q1.noonSwapAmp021)
         start += CPHASETime+q1['piLen']/2
         # CPHASE gate

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2+2.0*ns
         # pre-measurement pulse

         q1.z += env.rect(start, q1.noonSwapLenR21s[0]+q1.noonSwapLenCs[0], q1.noonSwapAmpR21)-env.rect(start+q1.noonSwapLenR21s[0], q1.noonSwapLenCs[0], q1.noonSwapAmpR21-q1.noonSwapAmpC)
         start += q1.noonSwapLenR21s[0]+q1.noonSwapLenCs[0]+4.0*ns

#         q1.z += eh.measurePulse(q1, start)
#
#         q1['readout'] = True

#         q0.z += env.rect(start, curr, q0.noonSwapAmpCRead)
#
#         start += curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True


         r1.xy = eh.mix(r1, env.gaussian(start+r1.piLen/2, r1.piFWHM,
                                         np.conjugate(a1*r1.noonDrivePhase)), freq = 'fRes0')
         start += r1.piLen+8.0*ns

         q1.z += env.rect(start, currLen, q1.noonSwapAmp0Read)

         start += currLen
         q1.z += eh.measurePulse(q1, start)

         q1['readout'] = True

         data = yield runQubits(server, qubits, stats=stats, probs=[1])

         data = np.hstack(([a1.real, a1.imag, currLen], data))
         returnValue(data)
    result = sweeps.run(func, sweepPara, dataset=save and dataset, noisy=noisy)
    return

def CPHASEPreRotArbSQSwap(sample, n=1, probeLen=st.arangePQ(0,500,1,ns), swap21Length=st.arangePQ(0,120,0.1,ns), CPHASETime = 42.203*ns, measure=1, stats=1800L,
                          delay=0.0*ns, delayA=0.0*ns, phase=np.pi, dfRot=99.1*MHz,
                          name='Single photon QND variable 21 swap length - rotating frame scan arbitray Fock q1 MQ swaps',
                          save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[1-measure]
    q1 = qubits[measure]

    axes = [(probeLen, 'swap pulse length'), (swap21Length, '21 swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state', axes, measure=measure, kw=kw)

    sl = q1['noonSwapLen0s']

    def func(server, currLen, currSwap):

         start = 0
         # dfRot=q1['f21']-q1['fRes0']
         ph = phase + 2*np.pi*dfRot[GHz]*currLen[ns]
         # do FFT and define dfRot
         # ph = phase

         q1.xy = env.NOTHING
         q1.z = env.NOTHING

         for i in range(n-1):

             q1.xy += eh.mix(q1, eh.piPulseHD(q1, start))
             start += q1['piLen']/2
             q1.z += env.rect(start, sl[i], q1.noonSwapAmp0)
             start += sl[i]+q1['piLen']/2+delay

         q1.xy += eh.mix(q1, eh.piPulseHD(q1, start))
         start += q1['piLen']/2
         q1.z += env.rect(start, sl[n-1], q1.noonSwapAmp0)
         start += sl[n-1]+delay+30.0*ns
         # generate Fock states in res 0

#         q1.z += env.rect(start+q1['piLen']/2, curr, q1.noonSwapAmp0Read)
#
#         start += q1['piLen']/2+curr
#         q1.z += eh.measurePulse(q1, start)
#
#         q1['readout'] = True

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z += env.rect(start, currSwap, q1.noonSwapAmp021)
         start += currSwap+q1['piLen']/2
         # CPHASE gate

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2+2.0*ns
         # pre-measurement pulse

         q1.z += env.rect(start, q1.noonSwapLenR21s[0]+q1.noonSwapLenCs[0], q1.noonSwapAmpR21)-env.rect(start+q1.noonSwapLenR21s[0], q1.noonSwapLenCs[0], q1.noonSwapAmpR21-q1.noonSwapAmpC)
         start += q1.noonSwapLenR21s[0]+q1.noonSwapLenCs[0]+4.0*ns

#         q1.z += eh.measurePulse(q1, start)
#
#         q1['readout'] = True

#         q0.z += env.rect(start, curr, q0.noonSwapAmpCRead)
#
#         start += curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         q1.z += env.rect(start, currLen, q1.noonSwapAmp0Read)

         start += currLen
         q1.z += eh.measurePulse(q1, start)

         q1['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def QuResCPHASEPreRotArbSwap(sample, n=1, probeLen=st.arangePQ(0,500,1,ns), swap21Length=st.arangePQ(0,121,1,ns), CPHASETime = 37.06*ns,
                             measure=1, stats=900L, delay=0.0*ns, delayA=0.0*ns, phase=np.pi, dfRot=287.45*MHz,
                             name='Qubit-Resonator CPHASE Pre Rot Arb Fock SWAP q0-rC Variable 21 Length',
                             save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[1-measure]
    q1 = qubits[measure]

    axes = [(probeLen, 'swap pulse length'), (swap21Length, '21 swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state', axes, measure=measure, kw=kw)

    sl = q0['noonSwapLenCs']
    auxAmp = 0.044

    def func(server, currLen, currSwap):

         start = 0
         # dfRot=q1['f21']-q1['fRes0']
         ph = phase + 2*np.pi*dfRot[GHz]*currLen[ns]
         # do FFT and define dfRot
         # ph = phase

         q0.xy = env.NOTHING
         q0.z = env.NOTHING

         for i in range(n-1):

             q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
             start += q0['piLen']/2
             q0.z += env.rect(start, sl[i], q0.noonSwapAmpC)
             start += sl[i]+q0['piLen']/2+delay

         q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
         start += q0['piLen']/2
#         q0.z += env.rect(start, sl[n-1], q0.noonSwapAmpC)
#         start += sl[n-1]+delay+2.0*ns
#         # generate Fock states in res C
#
#         q0.z += env.rect(start, q0.noonSwapLen0s[0], q0.noonSwapAmp0)
#         start += q0.noonSwapLen0s[0]-q0['piLen']/2
#         # reset q0

         q0.z += env.rect(start, sl[n-1]+q0.noonSwapLen0s[0]+delay+currSwap+q1['piLen']/2+q1['piLen']/2+currLen, q0.noonSwapAmp0)-env.rect(start, sl[n-1], q0.noonSwapAmp0-q0.noonSwapAmpC)-env.rect(start+sl[n-1]+q0.noonSwapLen0s[0], delay+currSwap+q1['piLen']/2+q1['piLen']/2+currLen, auxAmp)-env.rect(start+sl[n-1]+q0.noonSwapLen0s[0]+delay+currSwap+q1['piLen']/2+q1['piLen']/2, currLen, q0.noonSwapAmp0-auxAmp-q0.noonSwapAmpCRead)
         start += sl[n-1]+q0.noonSwapLen0s[0]+delay-q0['piLen']/2
         # generate Fock states in res C and reset q0

#         q1.z += env.rect(start+q1['piLen']/2, curr, q1.noonSwapAmp0Read)
#
#         start += q1['piLen']/2+curr
#         q1.z += eh.measurePulse(q1, start)
#
#         q1['readout'] = True

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z = env.rect(start, currSwap, q1.noonSwapAmpC21)
         start += currSwap+q1['piLen']/2
         # CPHASE gate

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2
         # pre-measurement pulse

#         q1.z += eh.measurePulse(q1, start)
#
#         q1['readout'] = True

#         q0.z += env.rect(start, curr, q0.noonSwapAmpCRead)
#
#         start += curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

#         q0.z += env.rect(start, currLen, q0.noonSwapAmpCRead)

         start += currLen
         q0.z += eh.measurePulse(q0, start)

         q0['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def QuResCPHASEPreRot0p1Swap(sample, probeLen=st.arangePQ(0,500,1,ns), swap21Length=st.arangePQ(0,121,1,ns), CPHASETime = 37.06*ns,
                             measure=1, stats=900L, delay=0.0*ns, delayA=0.0*ns, phase=np.pi, dfRot=277.57*MHz,
                             name='Qubit-Resonator CPHASE Pre Rot Fock 0+1 SWAP q0-rC Variable 21 Length',
                             save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[1-measure]
    q1 = qubits[measure]

    axes = [(probeLen, 'swap pulse length'), (swap21Length, '21 swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    auxAmp = 0.044

    def func(server, currLen, currSwap):

         start = 0
         # dfRot=q1['f21']-q1['fRes0']
         ph = phase + 2*np.pi*dfRot[GHz]*currLen[ns]
         # do FFT and define dfRot
         # ph = phase

         q0.xy = eh.mix(q0, eh.piHalfPulse(q0, start))
         start += q0['piLen']/2+delay

#         q0.z = env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
#         start += q0.noonSwapLenCs[0]+delay+2.0*ns
#         # 0+1 photons in resC

#         q0.z += env.rect(start, q0.noonSwapLen0s[0], q0.noonSwapAmp0)
#         start += q0.noonSwapLen0s[0]-q0['piLen']/2
#         # reset q0

         q0.z = env.rect(start, q0.noonSwapLenCs[0]+q0.noonSwapLen0s[0]+delay+currSwap+q1['piLen']/2+q1['piLen']/2+currLen, q0.noonSwapAmp0)-env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmp0-q0.noonSwapAmpC)-env.rect(start+q0.noonSwapLenCs[0]+q0.noonSwapLen0s[0], delay+currSwap+q1['piLen']/2+q1['piLen']/2+currLen, auxAmp)-env.rect(start+q0.noonSwapLenCs[0]+q0.noonSwapLen0s[0]+delay+currSwap+q1['piLen']/2+q1['piLen']/2, currLen, q0.noonSwapAmp0-auxAmp-q0.noonSwapAmpCRead)
         start += q0.noonSwapLenCs[0]+q0.noonSwapLen0s[0]+delay-q0['piLen']/2
         # generate state 0+1 in res C and reset q0

#         q1.z += env.rect(start+q1['piLen']/2, curr, q1.noonSwapAmp0Read)
#
#         start += q1['piLen']/2+curr
#         q1.z += eh.measurePulse(q1, start)
#
#         q1['readout'] = True

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z = env.rect(start, currSwap, q1.noonSwapAmpC21)
         start += currSwap+q1['piLen']/2
         # CPHASE gate

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2
         # pre-measurement pulse

#         q1.z += eh.measurePulse(q1, start)
#
#         q1['readout'] = True

#         q0.z += env.rect(start, curr, q0.noonSwapAmpCRead)
#
#         start += curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

#         q0.z += env.rect(start, currLen, q0.noonSwapAmpCRead)

         start += currLen
         q0.z += eh.measurePulse(q0, start)

         q0['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def QuResCPHASEPreRotArbSwapZero(sample, n=0, probeLen=st.arangePQ(0,155,1,ns), swap21Length=st.arangePQ(0,120,1,ns), CPHASETime = 37.50*ns, measure=1, stats=900L,
                                 delay=0.0*ns, delayA=0.0*ns, phase=np.pi, dfRot=287.45*MHz,
                                 name='Single photon QND variable 21 swap length - rotating frame scan arbitray Fock q0 MQ swaps',
                                 save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[1-measure]
    q1 = qubits[measure]

    axes = [(probeLen, 'swap pulse length'), (swap21Length, '21 swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state', axes, measure=measure, kw=kw)

    sl = q0['noonSwapLenCs']

    def func(server, currLen, currSwap):

         start = 0
         # dfRot=q1['f21']-q1['fRes0']
         ph = phase + 2*np.pi*dfRot[GHz]*currLen[ns]
         # do FFT and define dfRot
         # ph = phase

#         q1.z += env.rect(start+q1['piLen']/2, curr, q1.noonSwapAmp0Read)
#
#         start += q1['piLen']/2+curr
#         q1.z += eh.measurePulse(q1, start)
#
#         q1['readout'] = True

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay

         # curr = 2.0*q1.noonSwapLenC21s[0]+delayA
         q1.z = env.rect(start, currSwap, q1.noonSwapAmpC21)
         start += currSwap+q1['piLen']/2
         # CPHASE gate

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2+2.0*ns
         # pre-measurement pulse

#         q1.z += eh.measurePulse(q1, start)
#
#         q1['readout'] = True

#         q0.z += env.rect(start, curr, q0.noonSwapAmpCRead)
#
#         start += curr
#         q0.z += eh.measurePulse(q0, start)
#
#         q0['readout'] = True

         q0.z = env.rect(start, currLen, q0.noonSwapAmpCRead)

         start += currLen
         q0.z += eh.measurePulse(q0, start)

         q0['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def nightTomoCPHASE(sample):
    #alpha = array([1.1,1.45])[:,None] * exp(1.0j*linspace(0,2*pi,30,endpoint=False))[None,:]
    #alpha = reshape(alpha,size(alpha))
    #plot(real(alpha),imag(alpha))
    alpha = np.linspace(-2.0,2.0,25)
    alpha = alpha[:,None]+1.0j*alpha[None,:]
    alpha = np.reshape(alpha,np.size(alpha))

    CPHASEPreRotArbSQ_TOMO(sample, disp=alpha, phase=np.pi, dfRot=99.1*MHz, stats=1500L, CPHASETime = 42.203*ns, measure=1, save=True)
    CPHASEPreRotArbSQ_TOMO(sample, disp=alpha, phase=np.pi, dfRot=99.1*MHz, stats=1500L, CPHASETime = 21.1015*ns, measure=1, save=True)

def singlePhotonQNDPreFit(sample, swap21Length=st.arangePQ(0,20,0.1,ns), measure=0, stats=1800L, delay=0.0*ns, delayA=0.0*ns, phase=0,
                          name='Single photon QND variable 21 swap length - rotating frame fit MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(swap21Length, '21 swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, curr):
         start = 0

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay

         q1.z = env.rect(start, curr, q1.noonSwapAmpC21)
         start += curr+q1['piLen']/2

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase))
         start += q1['piLen']/2

         q1.z += eh.measurePulse(q1, start)

         q1['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    swap21Len = result[:,0]
    P1 = result[:,1]

    def fitfunc(t, p):
        return p[0]+p[1]*np.cos(2*np.pi*p[2]*(t-p[3]))*np.exp(-p[4]*t)
    def errfunc(p):
        return P1-fitfunc(swap21Len,p)

    p, ok = leastsq(errfunc, [0.5,1.0,0.185,0,0.005])
    plt.figure()
    plt.plot(swap21Len,P1,'b.')
    tfit = np.linspace(swap21Len[0],swap21Len[-1],1000)
    plt.plot(tfit,fitfunc(tfit,p),'r-')
    print 'Freq is %g GHz' %(p[2])
    print 'Decay rate is %g MHz' %(p[4]*1000.0)
    dfRot = p[2]
    return dfRot

def singlePhotonQNDPreRot(sample, swap21Length=st.arangePQ(0,120,0.1,ns), measure=0, stats=1800L, delay=0.0*ns, delayA=0.0*ns, phase=0, dfRot=0.0*GHz,
                          name='Single photon QND variable 21 swap length - rotating frame MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    dfRot = singlePhotonQNDPreFit(sample, swap21Length=st.arangePQ(0,20,0.1,ns), measure=measure, stats=stats, delay=delay, delayA=delayA,
                                  phase=phase, save=False)

    singlePhotonQNDPreRotScan(sample, swap21Length=swap21Length, measure=measure, stats=stats, delay=delay, delayA=delayA, phase=phase,
                              dfRot=dfRot*GHz, save=save, collect=collect, noisy=noisy)

def singlePhotonQNDPre2D(sample, swap21Length=st.arangePQ(0,150,0.1,ns), measure=0, stats=1200L, phase=np.arange(0,2*np.pi,np.pi/50),
                         delay=0.0*ns, delayA=0.0*ns, dfRot=201.6*MHz,
                         name='Single photon QND variable 21 swap length and phase MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    # nameEx = [' q1->q0', ' q0->q1']

    axes = [(swap21Length, '21 swap pulse length'), (phase, 'Extra pi-half pulse phase')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw) # +nameEx[measure]

    def func(server, currLen, currPh):
         start = 0
         ph = currPh + 2*np.pi*dfRot[GHz]*currLen[ns]

         q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
         start += q0['piLen']/2
         q0.z = env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
         start += q0.noonSwapLen0s[0]-q0['piLen']/2+delay
         # 1 photon in resC

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay

         q1.z = env.rect(start, currLen, q1.noonSwapAmpC21)
         start += currLen+q1['piLen']/2

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2

         q1.z += eh.measurePulse(q1, start)

         q1['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def singlePhotonQNDPreRotArb2D(sample, n=1, swap21Length=st.arangePQ(0,150,0.1,ns), measure=0, stats=1200L, phase=np.arange(0,2*np.pi,np.pi/50),
                               delay=0.0*ns, delayA=0.0*ns, dfRot=206.3*MHz,
                               name='Single photon QND variable 21 swap length and phase arb Fock MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    # nameEx = [' q1->q0', ' q0->q1']

    axes = [(swap21Length, '21 swap pulse length'), (phase, 'Extra pi-half pulse phase')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+' '+str(n)+' Fock state', axes, measure=measure, kw=kw) # +nameEx[measure]

    sl = q0['noonSwapLenCs']

    def func(server, currLen, currPh):
         start = 0
         # dfRot=q1['f21']-q1['fResC']
         ph = currPh + 2*np.pi*dfRot[GHz]*currLen[ns]
         # do FFT and define dfRot
         # ph = phase

         q0.xy = env.NOTHING
         q0.z = env.NOTHING

         for i in range(n-1):

             q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
             start += q0['piLen']/2
             q0.z += env.rect(start, sl[i], q0.noonSwapAmpC)
             start += sl[i]+q0['piLen']/2+delay

         q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
         start += q0['piLen']/2
         q0.z += env.rect(start, sl[n-1], q0.noonSwapAmpC)
         start += sl[n-1]-q0['piLen']/2+delay
         # generate Fock states in res C

         q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
         start += q1['piLen']/2+delay

         q1.z = env.rect(start, currLen, q1.noonSwapAmpC21)
         start += currLen+q1['piLen']/2

         q1.xy += eh.mix(q1, eh.piHalfPulse(q1, start, phase=ph))
         start += q1['piLen']/2

         q1.z += eh.measurePulse(q1, start)

         q1['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def singlePhotonQNDRamsey(sample, phiAmp=st.r[0.0:0.5:0.0001], measure=0, stats=1200L, delay=0.0*ns, delayA=0.0*ns, extraAmp=0.0, phase=0, df=50*MHz,
                          name='Single photon QND Ramsey MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    # nameEx = [' q1->q0', ' q0->q1']

    axes = [(phiAmp, 'Compensation phi amplitude')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw) # +nameEx[measure]

    def func(server, curr):

        start = 0

        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2
        q0.z = env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
        start += q0.noonSwapLenCs[0]-q0['piLen']/2+delay
        # 1 photon in resC

        q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
        start += q1['piLen']/2+delay+q1['piLen']/2
        # Ramsey start

        q1.z = env.rect(start, 2.0*q1.noonSwapLenC21s[0]+delayA, q1.noonSwapAmpC21+extraAmp)
        start += 2.0*q1.noonSwapLenC21s[0]+delayA+delay

        q1.z += env.rect(start, 10.0*ns, curr)
        start += 10.0*ns

        ph = phase - 2*np.pi*df[GHz]*delay[ns]
        q1.xy += eh.mix(q1, eh.piHalfPulse(q1, -2.0*q1.noonSwapLenC21s[0]-q1['piLen']/2-10.0*ns) + eh.piHalfPulse(q1, start, phase=ph))

        start += q1['piLen']/2
        q1.z += eh.measurePulse(q1, start)

        q1['readout'] = True

        return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def singlePhotonQND1Correct(sample, repetition=10, measure=[0,1], stats=1500L, delay=0*ns, delayA=0.0*ns, extraAmp=0.0,
                            alpha=1.0, pDeltaPhi1 = 0.35462394,
                            name='Single photon QND 1 TOMO phase corrected MQ', save=True, collect=False, noisy=True, extraDelay=extraDelay):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    measurement = pyle.dataking.measurement.Tomo(2, [1])

    nameEx = [' q0->q1',' q1->q0']
    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx[measure[0]], axes, measure=measurement, kw=kw)

    def func(server, curr):

        start = 0

        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2
        q0.z = env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
        start += q0.noonSwapLen0s[0]-q0['piLen']/2+delay
        # 1 photon in resC

        q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
        start += q1['piLen']/2+delay

        q1.z = env.rect(start, alpha*(2.0*q1.noonSwapLenC21s[0]+delayA), q1.noonSwapAmpC21+extraAmp)
        start += alpha*(2.0*q1.noonSwapLenC21s[0]+delayA)+delay

        return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    # rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)
    Qk = np.reshape(result[1:],(3,2))
    # Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    rho_cal = tomo.qst(Qk,'tomo')
    plotRhoSingle(rho_cal, figNo=100)
    pylab.title('Exp.')

    rhoCoherence = rho_cal[0,1]
    rhoPhi = np.arctan2(np.imag(rhoCoherence), np.real(rhoCoherence))
    errorPhi = 2*np.pi*pDeltaPhi1*(alpha*(2.0*q1.noonSwapLenC21s[0]+delayA))
    rhoPhiCorrect = rhoPhi-errorPhi

    rhoAmp = np.sqrt(np.real(rhoCoherence)**2+np.imag(rhoCoherence)**2)
    aPrime = rhoAmp*np.cos(rhoPhiCorrect)
    bPrime = rhoAmp*np.sin(rhoPhiCorrect)
    rhoCoherenceCorrect = aPrime+1.0j*bPrime

#    rho_cal[0,1] = 1.0*rhoCoherenceCorrect
#    rho_cal[1,0] = -1.0*rhoCoherenceCorrect
    rho_cal[0,1] = aPrime+1.0j*bPrime
    rho_cal[1,0] = aPrime-1.0j*bPrime

    plotRhoSingle(rho_cal, figNo=101)
    pylab.title('Phase correct')

    return rhoPhi, errorPhi, rhoPhiCorrect

#    rho_caln = rho_cal.copy()
#    rho_caln[1,2] = abs(rho_caln[1,2])
#    rho_caln[2,1] = abs(rho_caln[2,1])
#    print 'Fidelity is %g ' % np.trace(np.dot(rho_caln,rho_ideal))
#
#    Us =tomo._qst_transforms['tomo2'][0]
#    rho_calLike = tomo.qst_mle(Qk,Us,rho0 = rho_ideal)
#    rho_calLiken = rho_calLike.copy()
#    rho_calLiken[1,2] = abs(rho_calLiken[1,2])
#    rho_calLiken[2,1] = abs(rho_calLiken[2,1])
#    plotRhoSingle(rho_calLike,figNo=101)
#    pylab.title('Exp. likely')
#    print 'Fidelity for likelyhood is %g ' % np.trace(np.dot(rho_calLiken,rho_ideal))
#    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal)), rho_calLike, np.trace(np.dot(rho_calLiken,rho_ideal))

def singlePhotonQND1FindPhi(sample, repetition=10, measure=[0,1], stats=1500L, delay=0*ns, delayA=0.0*ns, extraAmp=0.0,
                            alpha=1.0, name='Single photon QND 1 TOMO find phase MQ', save=True, collect=False, noisy=True, extraDelay=extraDelay):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    measurement = pyle.dataking.measurement.Tomo(2, [1])

    nameEx = [' q0->q1',' q1->q0']
    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx[measure[0]], axes, measure=measurement, kw=kw)

    def func(server, curr):

        start = 0

        q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
        start += q1['piLen']/2+delay

        q1.z = env.rect(start, alpha*(2.0*q1.noonSwapLenC21s[0]+delayA), q1.noonSwapAmpC21+extraAmp)
        start += alpha*(2.0*q1.noonSwapLenC21s[0]+delayA)+delay

        return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    # rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)
    Qk = np.reshape(result[1:],(3,2))
    # Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    rho_cal = tomo.qst(Qk,'tomo')
    plotRhoSingle(rho_cal, figNo=100)
    pylab.title('Exp.')

    rhoCoherence = rho_cal[0,1]
    rhoPhi = np.arctan2(np.imag(rhoCoherence), np.real(rhoCoherence))

    return rhoPhi

def singlePhotonQND1PhiCorrect(sample, repetition=10, measure=[0,1], stats=1500L, delay=0*ns, delayA=0.0*ns, extraAmp=0.0, alpha=1.0,
                               name='Single photon QND 1 TOMO phase corrected anc MQ', save=True, collect=False, noisy=True, extraDelay=extraDelay):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    measurement = pyle.dataking.measurement.Tomo(2, [1])

    nameEx = [' q0->q1',' q1->q0']
    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx[measure[0]], axes, measure=measurement, kw=kw)

    rhoPhiErr = singlePhotonQND1FindPhi(sample, repetition=repetition, measure=measure, stats=stats,
                                        delay=delay, delayA=delayA, extraAmp=extraAmp,
                                        alpha=alpha, save=False, collect=False, noisy=False, extraDelay=extraDelay)

    def func(server, curr):

        start = 0

        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2
        q0.z = env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
        start += q0.noonSwapLen0s[0]-q0['piLen']/2+delay
        # 1 photon in resC

        q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
        start += q1['piLen']/2+delay

        q1.z = env.rect(start, alpha*(2.0*q1.noonSwapLenC21s[0]+delayA), q1.noonSwapAmpC21+extraAmp)
        start += alpha*(2.0*q1.noonSwapLenC21s[0]+delayA)+delay

        return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    # rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)
    Qk = np.reshape(result[1:],(3,2))
    # Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    rho_cal = tomo.qst(Qk,'tomo')
    plotRhoSingle(rho_cal, figNo=100)
    pylab.title('Exp.')

    rhoCoherence = rho_cal[0,1]
    rhoPhi = np.arctan2(np.imag(rhoCoherence), np.real(rhoCoherence))
    rhoPhiCorrect = rhoPhi-rhoPhiErr

    rhoAmp = np.sqrt(np.real(rhoCoherence)**2+np.imag(rhoCoherence)**2)
    aPrime = rhoAmp*np.cos(rhoPhiCorrect)
    bPrime = rhoAmp*np.sin(rhoPhiCorrect)
    rhoCoherenceCorrect = aPrime+1.0j*bPrime

#    rho_cal[0,1] = 1.0*rhoCoherenceCorrect
#    rho_cal[1,0] = -1.0*rhoCoherenceCorrect
    rho_cal[0,1] = aPrime+1.0j*bPrime
    rho_cal[1,0] = aPrime-1.0j*bPrime

    plotRhoSingle(rho_cal, figNo=101)
    pylab.title('Phase correct')

    return rhoPhiErr, rhoPhi, rhoPhiCorrect

def singlePhotonResonantQND(sample, repetition=10, measure=[0,1], stats=1500L, delay=0*ns, superDelay=0.0*ns, delayA=0.0*ns, extraAmp=0.0,
                            name='Single photon QND 1 TOMO MQ', save=True, collect=False, noisy=True, extraDelay=extraDelay):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    measurement = pyle.dataking.measurement.Tomo(2, [1])

    nameEx = [' q0->q1',' q1->q0']
    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx[measure[0]], axes, measure=measurement, kw=kw)

    def func(server, curr):

        start = 0

        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2
        q0.z = env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
        start += q0.noonSwapLen0s[0]-q0['piLen']/2+delay+superDelay
        # 1 photon in resC

        q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
        start += q1['piLen']/2+delay

        # q1.z = env.rect(start, 2.0*q1.noonSwapLenC21s[0]+delayA, q1.noonSwapAmpC21+extraAmp)
        # start += 1.0*q1.noonSwapLenC21s[0]+delayA+delay
        start += 64.0*ns+delayA+delay

#        q1.z = env.rect(start, 0.0*q1.noonSwapLenC21s[0]+delayA, q1.noonSwapAmpC21+extraAmp)
#        start += 0.0*q1.noonSwapLenC21s[0]+delayA+delay

        return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    # rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)
    # pdb.set_trace()
    Qk = np.reshape(result[1:],(3,2))
    # Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    rho_cal = tomo.qst(Qk,'tomo')
    plotRhoSingle(rho_cal,figNo=100)
    pylab.title('Exp.')

    rhoCoherence = rho_cal[0,1]

    return np.arctan2(np.imag(rhoCoherence), np.real(rhoCoherence))

def singlePhotonResonantQNDOsc(sample, probeLen=st.arangePQ(0,500,1,ns), measure=0, stats=1500L, delay=0*ns, superDelay=0.0*ns, delayA=0.0*ns, extraAmp=0.0,
                               name='Single photon QND 1 TOMO MQ', save=True, collect=False, noisy=True, extraDelay=extraDelay):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[1+measure]

    axes = [(probeLen, 'swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, curr):

        start = 0

        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2
        q0.z = env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
        start += q0.noonSwapLen0s[0]-q0['piLen']/2+delay+superDelay
        # 1 photon in resC

        q1.xy = eh.mix(q1, eh.piHalfPulse(q1, start))
        start += q1['piLen']/2+delay

        # q1.z = env.rect(start, 2.0*q1.noonSwapLenC21s[0]+delayA, q1.noonSwapAmpC21+extraAmp)
        # start += 1.0*q1.noonSwapLenC21s[0]+delayA+delay

#        q1.z = env.rect(start, 0.0*q1.noonSwapLenC21s[0]+delayA, q1.noonSwapAmpC21+extraAmp)
#        start += 0.0*q1.noonSwapLenC21s[0]+delayA+delay

        start += curr
        q1.z = eh.measurePulse2(q1, start)

        q1['readout'] = True

        return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def noonTomo(sample, probeLen=st.arangePQ(0,300,1,ns), disp0=None, disp1=None, n=1, measure=[0,1], stats=1500L,
         name='noon state Tomo MQ', save=True, collect=False, noisy=True, extraDelay=extraDelay, delay=0*ns):

    start = time.time()
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    r0 = qubits[measure[0]+2]
    q1 = qubits[measure[1]]
    r1 = qubits[measure[1]+2]
    nameEx = [' q0->q1',' q1->q0']

    sweepPara = complexSweep(np.array(disp0)/r0.noonAmpScale.value,np.array(disp1)/r1.noonAmpScale.value,probeLen)

    kw = {'stats': stats,
          'measure': measure}
    dataset = sweeps.prepDataset(sample, 'n='+str(n)+' '+name+nameEx[measure[0]],
             axes = [('r0 displacement', 're'),('r0 displacement', 'im'),
                       ('r1 displacement', 're'),('r1 displacement', 'im'),
                       ('swap pulse length', 'ns')], measure=measure, kw=kw)

    def func(server, curr):
        a0 = curr[0]
        a1 = curr[1]
        currLen = curr[2]
        start = 0
        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2+delay
        q0.z = env.rect(start, q0.noonSwapLenCSQRT, q0.noonSwapAmpC)
        start += q0.noonSwapLenCSQRT+delay
        q1.z = env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)

        start += q1.noonSwapLenC+delay
        for i in range(n-1):
            q0.xy += eh.mix(q0, env.gaussian(start, q0.piFWHM, q0.piAmp21, df = q0.piDf21), freq = 'f21')
            q1.xy  = eh.mix(q1, env.gaussian(start, q1.piFWHM, q1.piAmp21, df = q1.piDf21), freq = 'f21')

            start += np.max([q0.piLen,q1.piLen])+delay
            q0.z += env.rect(start, q0.noonSwapLen1s[i], q0.noonSwapAmp1)
            q1.z += env.rect(start, q1.noonSwapLen1s[i], q1.noonSwapAmp1)
            start += np.max([q0.noonSwapLen1s[i],q1.noonSwapLen1s[i]])+delay

        q0.z += env.rect(start, q0.noonSwapLen0s[n-1], q0.noonSwapAmp0)
        q1.z += env.rect(start, q1.noonSwapLen0s[n-1], q1.noonSwapAmp0)

        start += np.max([q0.noonSwapLen0s[n-1],q1.noonSwapLen0s[n-1]])+extraDelay+delay
        r0.xy = eh.mix(r0, env.gaussian(start+r0.piLen/2, r0.piFWHM,
                                        np.conjugate(a0*r0.noonDrivePhase)), freq = 'fRes0')
        r1.xy = eh.mix(r1, env.gaussian(start+r1.piLen/2, r1.piFWHM,
                                        np.conjugate(a1*r1.noonDrivePhase)), freq = 'fRes0')

        start += np.max([r0.piLen,r1.piLen])+extraDelay+delay
        q0.z += env.rect(start, currLen, q0.noonSwapAmp0Read)
        q1.z += env.rect(start, currLen, q1.noonSwapAmp0Read)

        start += currLen+delay
        q0.z += eh.measurePulse(q0, start)
        q1.z += eh.measurePulse(q1, start)

        q0['readout'] = True
        q1['readout'] = True
        data = yield runQubits(server, qubits, stats=stats)

        if measure[0]==0:
            data = np.hstack(([a0.real, a0.imag, a1.real, a1.imag, currLen], data))
        else:
            data = np.hstack(([a1.real, a1.imag, a0.real, a0.imag, currLen], data))
        returnValue(data)

    results = sweeps.run(func, sweepPara, dataset=save and dataset, noisy=noisy)
    end = time.time()
    print 'Time elapsed %g s' %(end-start)
    return

def noonTomoSchemeA(sample, probeLen=st.arangePQ(0,300,2,ns), disp0=None, disp1=None, n=2, measure=[0,1], stats=1500L,
         name='noon state Tomo SchemeA MQ', save=True, collect=False, noisy=True, extraDelay=extraDelay, delay=0*ns):

    start = time.time()
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    r0 = qubits[measure[0]+2]
    q1 = qubits[measure[1]]
    r1 = qubits[measure[1]+2]
    nameEx = [' q0->q1',' q1->q0']

    sweepPara = complexSweep(np.array(disp0)/r0.noonAmpScale.value,np.array(disp1)/r1.noonAmpScale.value,probeLen)

    kw = {'stats': stats,
          'measure': measure}
    dataset = sweeps.prepDataset(sample, 'n='+str(n)+' '+name+nameEx[measure[0]],
             axes = [('r0 displacement', 're'),('r0 displacement', 'im'),
                       ('r1 displacement', 're'),('r1 displacement', 'im'),
                       ('swap pulse length', 'ns')], measure=measure, kw=kw)

    def func(server, curr):
        a0 = curr[0]
        a1 = curr[1]
        currLen = curr[2]

        start = -q0.piLen/2
        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start+q0.piLen/2))
        start += q0['piLen']+delay
        q0.z = env.rect(start, q0.noonSwapLenCSQRT, q0.noonSwapAmpC)
        start += q0.noonSwapLenCSQRT+delay
        q1.z = env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)
        start += q1.noonSwapLenC+delay

        q0.z += env.rect(start, q0.noonSwapLen0s[0], q0.noonSwapAmp0)
        q1.z += env.rect(start, q1.noonSwapLen0s[0], q1.noonSwapAmp0)
        start += np.max([q0.noonSwapLen0s[0],q1.noonSwapLen0s[0]])+delay

        q0.xy += eh.mix(q0, eh.piPulseHD(q0, start+q0.piLen/2))
        start += q0['piLen']+delay
        q0.z += env.rect(start, q0.noonSwapLenCSQRT, q0.noonSwapAmpC)
        start += q0.noonSwapLenCSQRT+delay
        q1.z += env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)
        start += q1.noonSwapLenC+delay

        q0.z += env.rect(start, q0.noonSwapLen0s[1], q0.noonSwapAmp0)
        q1.z += env.rect(start, q1.noonSwapLen0s[1], q1.noonSwapAmp0)
        start += np.max([q0.noonSwapLen0s[1],q1.noonSwapLen0s[1]])+delay+extraDelay

        r0.xy = eh.mix(r0, env.gaussian(start+r0.piLen/2, r0.piFWHM,
                                        np.conjugate(a0*r0.noonDrivePhase)), freq = 'fRes0')
        r1.xy = eh.mix(r1, env.gaussian(start+r1.piLen/2, r1.piFWHM,
                                        np.conjugate(a1*r1.noonDrivePhase)), freq = 'fRes0')

        start += np.max([r0.piLen,r1.piLen])+extraDelay+delay
        q0.z += env.rect(start, currLen, q0.noonSwapAmp0Read)
        q1.z += env.rect(start, currLen, q1.noonSwapAmp0Read)

        start += currLen+delay
        q0.z += eh.measurePulse(q0, start)
        q1.z += eh.measurePulse(q1, start)

        q0['readout'] = True
        q1['readout'] = True
        data = yield runQubits(server, qubits, stats=stats)

        if measure[0]==0:
            data = np.hstack(([a0.real, a0.imag, a1.real, a1.imag, currLen], data))
        else:
            data = np.hstack(([a1.real, a1.imag, a0.real, a0.imag, currLen], data))
        returnValue(data)

    results = sweeps.run(func, sweepPara, dataset=save and dataset, noisy=noisy)
    end = time.time()
    print 'Time elapsed %g s' %(end-start)
    return

def noonTomoTest(sample, probeLen=st.arangePQ(0,300,1,ns), disp0=None, disp1=None, n=1, measure=[0,1], stats=1500L, delay=0*ns,
         name='noon state Tomo test MQ', save=True, collect=False, noisy=True, extraDelay=extraDelay, qubitTomo=False, qMeasure=0):

    start = time.time()
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    r0 = qubits[measure[0]+2]
    q1 = qubits[measure[1]]
    r1 = qubits[measure[1]+2]
    nameEx = [' q0->q1',' q1->q0']

    a0 = disp0/r0.noonAmpScale.value
    a1 = disp1/r1.noonAmpScale.value
    if qubitTomo:
        axes = [(np.arange(1,10,1),'Trials')]
        name = name+' qubit '+str(qMeasure)+' after prep '
    else:
        axes = [(probeLen, 'Probe length')]

    kw = {'stats': stats,
          'measure': measure,
          'disp0': disp0,
          'disp1': disp1}
    dataset = sweeps.prepDataset(sample, 'n='+str(n)+' '+name+nameEx[measure[0]],
             axes = axes, measure=measure, kw=kw)

    def func(server, curr):
        currLen = curr
        start = 0
        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2+delay
        q0.z = env.rect(start, q0.noonSwapLenCSQRT, q0.noonSwapAmpC)
        start += q0.noonSwapLenCSQRT+delay
        q1.z = env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)

        start += q1.noonSwapLenC+delay
        for i in range(n-1):
            q0.xy += eh.mix(q0, env.gaussian(start, q0.piFWHM, q0.piAmp21, df = q0.piDf21), freq = 'f21')
            q1.xy  = eh.mix(q1, env.gaussian(start, q1.piFWHM, q1.piAmp21, df = q1.piDf21), freq = 'f21')

            start += np.max([q0.piLen,q1.piLen])+delay
            q0.z += env.rect(start, q0.noonSwapLen1s[i], q0.noonSwapAmp1)
            q1.z += env.rect(start, q1.noonSwapLen1s[i], q1.noonSwapAmp1)
            start += np.max([q0.noonSwapLen1s[i],q1.noonSwapLen1s[i]])+delay

        q0.z += env.rect(start, q0.noonSwapLen0s[n-1], q0.noonSwapAmp0)
        q1.z += env.rect(start, q1.noonSwapLen0s[n-1], q1.noonSwapAmp0)

        start += np.max([q0.noonSwapLen0s[n-1],q1.noonSwapLen0s[n-1]])+extraDelay+delay
        r0.xy = eh.mix(r0, env.gaussian(start+r0.piLen/2, r0.piFWHM,
                                        np.conjugate(a0*r0.noonDrivePhase)), freq = 'fRes0')
        r1.xy = eh.mix(r1, env.gaussian(start+r1.piLen/2, r1.piFWHM,
                                        np.conjugate(a1*r1.noonDrivePhase)), freq = 'fRes0')

        start += np.max([r0.piLen,r1.piLen])+extraDelay+delay
        if qubitTomo:
            q0.z += eh.measurePulse(q0, start)
            q1.z += eh.measurePulse(q1, start)
            if not np.iterable(qMeasure):
                qubits[qMeasure].readout = True
            else:
                for i in qMeasure:
                    qubits[i].readout = True
        else:
            q0.z += env.rect(start, currLen, q0.noonSwapAmp0Read)
            q1.z += env.rect(start, currLen, q1.noonSwapAmp0Read)

            start += currLen
            q0.z += eh.measurePulse(q0, start)
            q1.z += eh.measurePulse(q1, start)

            q0['readout'] = True
            q1['readout'] = True

        return runQubits(server, qubits, stats=stats)

    results = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    end = time.time()
    print 'Time elapsed %g s' %(end-start)
    return

def noonTomoMidTransTestQubit(sample, repetition=10, n=2, measure=[0,1], stats=1500L, delay=0*ns,
         name='noon state Tomo MidTrans test MQ', save=True, collect=False, noisy=True, extraDelay=extraDelay):

    start = time.time()
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]

    q1 = qubits[measure[1]]

    nameEx = [' q0->q1',' q1->q0']
    repetition = range(repetition)
    axes = [(repetition,'repetition')]
    name = name+' qubit tomo after prep '
    measurement = pyle.dataking.measurement.Tomo(2)
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, 'n='+str(n)+' '+name+nameEx[measure[0]],
             axes = axes, measure=measurement, kw=kw)

    def func(server, curr):
        currLen = curr
        start = 0
        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2+delay
        q0.z = env.rect(start, q0.noonSwapLenCSQRT, q0.noonSwapAmpC)
        start += q0.noonSwapLenCSQRT+delay
        q1.z = env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)

        start += q1.noonSwapLenC+delay
        for i in range(n-1):
            q0.xy += eh.mix(q0, env.gaussian(start, q0.piFWHM, q0.piAmp21, df = q0.piDf21), freq = 'f21')
            q1.xy  = eh.mix(q1, env.gaussian(start, q1.piFWHM, q1.piAmp21, df = q1.piDf21), freq = 'f21')

            start += np.max([q0.piLen,q1.piLen])+delay
            q0.z += env.rect(start, q0.noonSwapLen1s[i], q0.noonSwapAmp1)
            q1.z += env.rect(start, q1.noonSwapLen1s[i], q1.noonSwapAmp1)
            start += np.max([q0.noonSwapLen1s[i],q1.noonSwapLen1s[i]])+delay

        q0.z += env.rect(start, q0.noonSwapLen0s[n-1], q0.noonSwapAmp0)
        q1.z += env.rect(start, q1.noonSwapLen0s[n-1], q1.noonSwapAmp0)

        start += np.max([q0.noonSwapLen0s[n-1],q1.noonSwapLen0s[n-1]])+delay

        # q0.z += env.rect(start, q0.noonSwapLen0s[0], q0.noonSwapAmp0)
        # q1.z += env.rect(start, q1.noonSwapLen0s[0], q1.noonSwapAmp0)

        return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)
    Qk = np.reshape(result[1:],(9,4))
    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    rho_cal = tomo.qst(Qk,'tomo2')
    plotRhoSingle(rho_cal,figNo=100)
    pylab.title('Exp.')
    rho_caln = rho_cal.copy()
    rho_caln[1,2] = abs(rho_caln[1,2])
    rho_caln[2,1] = abs(rho_caln[2,1])
    print 'Fidelity is %g ' % np.trace(np.dot(rho_caln,rho_ideal))

    Us =tomo._qst_transforms['tomo2'][0]
    rho_calLike = tomo.qst_mle(Qk,Us,rho0 = rho_ideal)
    rho_calLiken = rho_calLike.copy()
    rho_calLiken[1,2] = abs(rho_calLiken[1,2])
    rho_calLiken[2,1] = abs(rho_calLiken[2,1])
    plotRhoSingle(rho_calLike,figNo=101)
    pylab.title('Exp. likely')
    print 'Fidelity for likelyhood is %g ' % np.trace(np.dot(rho_calLiken,rho_ideal))
    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal)), rho_calLike, np.trace(np.dot(rho_calLiken,rho_ideal))

def noonTomoPostTransTestQubit(sample, repetition=10, n=2, measure=[0,1], stats=1500L, delay=0*ns,
         name='noon state Tomo PostTrans test MQ', save=True, collect=False, noisy=True, extraDelay=extraDelay):

    start = time.time()
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]

    q1 = qubits[measure[1]]

    nameEx = [' q0->q1',' q1->q0']
    repetition = range(repetition)
    axes = [(repetition,'repetition')]
    name = name+' qubit tomo after prep '
    measurement = pyle.dataking.measurement.Tomo(2)
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, 'n='+str(n)+' '+name+nameEx[measure[0]],
             axes = axes, measure=measurement, kw=kw)

    def func(server, curr):
        start = 0
        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2+delay
        q0.z = env.rect(start, q0.noonSwapLenCSQRT, q0.noonSwapAmpC)
        start += q0.noonSwapLenCSQRT+delay
        q1.z = env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)

        start += q1.noonSwapLenC+delay
        for i in range(n-1):
            q0.xy += eh.mix(q0, env.gaussian(start, q0.piFWHM, q0.piAmp21, df = q0.piDf21), freq = 'f21')
            q1.xy  = eh.mix(q1, env.gaussian(start, q1.piFWHM, q1.piAmp21, df = q1.piDf21), freq = 'f21')

            start += np.max([q0.piLen,q1.piLen])+delay
            q0.z += env.rect(start, q0.noonSwapLen1s[i], q0.noonSwapAmp1)
            q1.z += env.rect(start, q1.noonSwapLen1s[i], q1.noonSwapAmp1)
            start += np.max([q0.noonSwapLen1s[i],q1.noonSwapLen1s[i]])+delay

        q0.z += env.rect(start, q0.noonSwapLen0s[n-1], q0.noonSwapAmp0)
        q1.z += env.rect(start, q1.noonSwapLen0s[n-1], q1.noonSwapAmp0)

        start += np.max([q0.noonSwapLen0s[n-1],q1.noonSwapLen0s[n-1]])+extraDelay+delay

        q0.z += env.rect(start, q0.noonSwapLen0s[n-1], q0.noonSwapAmp0)
        q1.z += env.rect(start, q1.noonSwapLen0s[n-1], q1.noonSwapAmp0)

        start += np.max([q0.noonSwapLen0s[n-1],q1.noonSwapLen0s[n-1]])+delay
        return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)
    Qk = np.reshape(result[1:],(9,4))
    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    rho_cal = tomo.qst(Qk,'tomo2')
    plotRhoSingle(rho_cal,figNo=100)
    pylab.title('Exp.')
    rho_caln = rho_cal.copy()
    rho_caln[1,2] = abs(rho_caln[1,2])
    rho_caln[2,1] = abs(rho_caln[2,1])
    print 'Fidelity is %g ' % np.trace(np.dot(rho_caln,rho_ideal))

    Us =tomo._qst_transforms['tomo2'][0]
    rho_calLike = tomo.qst_mle(Qk,Us,rho0 = rho_ideal)
    rho_calLiken = rho_calLike.copy()
    rho_calLiken[1,2] = abs(rho_calLiken[1,2])
    rho_calLiken[2,1] = abs(rho_calLiken[2,1])
    plotRhoSingle(rho_calLike,figNo=101)
    pylab.title('Exp. likely')
    print 'Fidelity for likelyhood is %g ' % np.trace(np.dot(rho_calLiken,rho_ideal))
    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal)), rho_calLike, np.trace(np.dot(rho_calLiken,rho_ideal))

def noonTomoPostTransTestQubitSchemeA(sample, repetition=10, n=2, measure=[0,1], stats=1500L, delay=0*ns,
         name='noon state Tomo PostTransSchemeA test MQ', save=True, collect=False, noisy=True, extraDelay=extraDelay):

    start = time.time()
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]

    q1 = qubits[measure[1]]

    nameEx = [' q0->q1',' q1->q0']
    repetition = range(repetition)
    axes = [(repetition,'repetition')]
    name = name+' qubit tomo after prep '
    measurement = pyle.dataking.measurement.Tomo(2)
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, 'n='+str(n)+' '+name+nameEx[measure[0]],
             axes = axes, measure=measurement, kw=kw)

    def func(server, curr):
        start = -q0.piLen/2
        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start+q0.piLen/2))
        start += q0['piLen']+delay
        q0.z = env.rect(start, q0.noonSwapLenCSQRT, q0.noonSwapAmpC)
        start += q0.noonSwapLenCSQRT+delay
        q1.z = env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)
        start += q1.noonSwapLenC+delay

        q0.z += env.rect(start, q0.noonSwapLen0s[0], q0.noonSwapAmp0)
        q1.z += env.rect(start, q1.noonSwapLen0s[0], q1.noonSwapAmp0)
        start += np.max([q0.noonSwapLen0s[0],q1.noonSwapLen0s[0]])+delay

        q0.xy += eh.mix(q0, eh.piPulseHD(q0, start+q0.piLen/2))
        start += q0['piLen']+delay
        q0.z += env.rect(start, q0.noonSwapLenCSQRT, q0.noonSwapAmpC)
        start += q0.noonSwapLenCSQRT+delay
        q1.z += env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)
        start += q1.noonSwapLenC+delay

        q0.z += env.rect(start, q0.noonSwapLen0s[1], q0.noonSwapAmp0)
        q1.z += env.rect(start, q1.noonSwapLen0s[1], q1.noonSwapAmp0)
        start += np.max([q0.noonSwapLen0s[1],q1.noonSwapLen0s[1]])+delay+extraDelay

        q0.z += env.rect(start, q0.noonSwapLen0s[1], q0.noonSwapAmp0)
        q1.z += env.rect(start, q1.noonSwapLen0s[1], q1.noonSwapAmp0)

        start += np.max([q0.noonSwapLen0s[1],q1.noonSwapLen0s[1]])+delay
        return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    result = np.sum(result,axis=0)/len(repetition)
    Qk = np.reshape(result[1:],(9,4))
    Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])

    rho_cal = tomo.qst(Qk,'tomo2')
    plotRhoSingle(rho_cal,figNo=100)
    pylab.title('Exp.')
    rho_caln = rho_cal.copy()
    rho_caln[1,2] = abs(rho_caln[1,2])
    rho_caln[2,1] = abs(rho_caln[2,1])
    print 'Fidelity is %g ' % np.trace(np.dot(rho_caln,rho_ideal))

    Us =tomo._qst_transforms['tomo2'][0]
    rho_calLike = tomo.qst_mle(Qk,Us,rho0 = rho_ideal)
    rho_calLiken = rho_calLike.copy()
    rho_calLiken[1,2] = abs(rho_calLiken[1,2])
    rho_calLiken[2,1] = abs(rho_calLiken[2,1])
    plotRhoSingle(rho_calLike,figNo=101)
    pylab.title('Exp. likely')
    print 'Fidelity for likelyhood is %g ' % np.trace(np.dot(rho_calLiken,rho_ideal))
    return rho_cal, np.trace(np.dot(rho_caln,rho_ideal)), rho_calLike, np.trace(np.dot(rho_calLiken,rho_ideal))

def noonTomoTestSchemeA(sample, probeLen=st.arangePQ(0,300,2,ns), disp0=None, disp1=None, n=1, measure=[0,1], stats=1500L, delay=0*ns,
         name='noon state Tomo SchemeA test MQ', save=True, collect=False, noisy=True, extraDelay=extraDelay, qubitTomo=False, qMeasure=0):

    start = time.time()
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    r0 = qubits[measure[0]+2]
    q1 = qubits[measure[1]]
    r1 = qubits[measure[1]+2]
    nameEx = [' q0->q1',' q1->q0']

    a0 = disp0/r0.noonAmpScale.value
    a1 = disp1/r1.noonAmpScale.value
    if qubitTomo:
        axes = [(np.arange(1,10,1),'Trials')]
        name = name+' qubit '+str(qMeasure)+' after prep '
    else:
        axes = [(probeLen, 'Probe length')]

    kw = {'stats': stats,
          'measure': measure,
          'disp0': disp0,
          'disp1': disp1}
    dataset = sweeps.prepDataset(sample, 'n='+str(n)+' '+name+nameEx[measure[0]],
             axes = axes, measure=measure, kw=kw)

    def func(server, curr):
        currLen = curr

        start = -q0.piLen/2
        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start+q0.piLen/2))
        start += q0['piLen']+delay
        q0.z = env.rect(start, q0.noonSwapLenCSQRT, q0.noonSwapAmpC)
        start += q0.noonSwapLenCSQRT+delay
        q1.z = env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)
        start += q1.noonSwapLenC+delay

        q0.z += env.rect(start, q0.noonSwapLen0s[0], q0.noonSwapAmp0)
        q1.z += env.rect(start, q1.noonSwapLen0s[0], q1.noonSwapAmp0)
        start += np.max([q0.noonSwapLen0s[0],q1.noonSwapLen0s[0]])+delay

        q0.xy += eh.mix(q0, eh.piPulseHD(q0, start+q0.piLen/2))
        start += q0['piLen']+delay
        q0.z += env.rect(start, q0.noonSwapLenCSQRT, q0.noonSwapAmpC)
        start += q0.noonSwapLenCSQRT+delay
        q1.z += env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)
        start += q1.noonSwapLenC+delay

        q0.z += env.rect(start, q0.noonSwapLen0s[1], q0.noonSwapAmp0)
        q1.z += env.rect(start, q1.noonSwapLen0s[1], q1.noonSwapAmp0)
        start += np.max([q0.noonSwapLen0s[1],q1.noonSwapLen0s[1]])+delay+extraDelay

        r0.xy = eh.mix(r0, env.gaussian(start+r0.piLen/2, r0.piFWHM,
                                        np.conjugate(a0*r0.noonDrivePhase)), freq = 'fRes0')
        r1.xy = eh.mix(r1, env.gaussian(start+r1.piLen/2, r1.piFWHM,
                                        np.conjugate(a1*r1.noonDrivePhase)), freq = 'fRes0')

        start += np.max([r0.piLen,r1.piLen])+extraDelay+delay
        if qubitTomo:
            q0.z += eh.measurePulse(q0, start)
            q1.z += eh.measurePulse(q1, start)
            if not np.iterable(qMeasure):
                qubits[qMeasure].readout = True
            else:
                for i in qMeasure:
                    qubits[i].readout = True
        else:
            q0.z += env.rect(start, currLen, q0.noonSwapAmp0Read)
            q1.z += env.rect(start, currLen, q1.noonSwapAmp0Read)

            start += currLen
            q0.z += eh.measurePulse(q0, start)
            q1.z += eh.measurePulse(q1, start)

            q0['readout'] = True
            q1['readout'] = True

        return runQubits(server, qubits, stats=stats)

    results = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    end = time.time()
    print 'Time elapsed %g s' %(end-start)
    return

def resdrivephase2D(sample, points=200, stats=1500, unitAmpl=0.25, measure=0, extraDelay=extraDelay,
       name='resonator drive phase', save=True, collect=True, noisy=True):

    sample, qubits, Qubits = util.loadQubits(sample,write_access=True)
    q = qubits[measure]
    r = qubits[measure+2]
    R = Qubits[measure+2]

    angle = np.linspace(0,2*np.pi, points, endpoint=False)
    displacement=0.3*unitAmpl*np.exp(1j*angle)

    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes=[('displacement','re'),('displacement','im')], measure=measure, kw=kw)

    def func(server, curr):
        q.xy = eh.mix(r, env.gaussian(0, q.piFWHM, amp = q.piAmp/2), freq = 'f10')
        q.z = env.rect(q.piLen/2, q.noonSwapLen0, q.noonSwapAmp0)
        r.xy = eh.mix(r, env.gaussian(q.piLen/2+q.noonSwapLen0+extraDelay+r.piLen/2,r.piFWHM,
                  amp = np.conjugate(curr*r.noonDrivePhase)), freq = 'fRes0')
        q.z += env.rect(q.piLen/2+q.noonSwapLen0+extraDelay+r.piLen+extraDelay, q.noonSwapLen0, q.noonSwapAmp0)
        q.z += eh.measurePulse(q, q.piLen/2+q.noonSwapLen0+extraDelay+r.piLen+extraDelay+q.noonSwapLen0)
        q['readout'] = True
        data = yield runQubits(server, qubits, stats, probs=[1])
        data = np.hstack(([curr.real, curr.imag], data))
        returnValue(data)

    result = sweeps.run(func, displacement, dataset=save and dataset, noisy=noisy)

    result = result[:, [0,2]]
    result[:,0] = angle
    def fitfunc(angle,p):
        return p[0]+p[1]*np.cos(angle-p[2])
    def errfunc(p):
        return result[:,1]-fitfunc(result[:,0],p)
    p,ok = leastsq(errfunc, [0.0,100.0,0.0])
    if p[1] < 0:
        p[1] = -p[1]
        p[2] = p[2]+np.pi
    p[2] = (p[2]+np.pi)%(2*np.pi)-np.pi
    plt.plot(result[:,0],result[:,1])
    plt.plot(angle, fitfunc(angle,p))
    a = r.noonDrivePhase*np.exp(1.0j*p[2])
    print 'Resonator drive Phase correction: %g' % p[2]
    R.noonDrivePhase = a/abs(a)
    return

def visibility(sample, mpa=st.r[0:2:0.05], stats=300, measure=0, level=1,
               save=True, name='Visibility MQ', collect=True, update=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    axes = [(mpa, 'Measure pulse amplitude')]
    if level==1:
        deps = [('Probability', '|0>', ''),
                ('Probability', '|1>', ''),
                ('Visibility', '|1> - |0>', ''),
                ]
    elif level==2:
        deps = [('Probability', '|0>', ''),
                ('Probability', '|1>', ''),
                ('Visibility', '|1> - |0>', ''),
                ('Probability', '|2>', ''),
                ('Visibility', '|2> - |1>', '')
                ]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, mpa):
        t_pi = 0
        t_meas = q['piLen']/2.0

        # without pi-pulse
        q['readout'] = True
        q['measureAmp'] = mpa
        q.xy = env.NOTHING
        q.z = eh.measurePulse(q, t_meas)
        req0 = runQubits(server, qubits, stats, probs=[1])

        # with pi-pulse
        q['readout'] = True
        q['measureAmp'] = mpa
        q.xy = eh.mix(q, eh.piPulseHD(q, t_pi))
        q.z = eh.measurePulse(q, t_meas)
        req1 = runQubits(server, qubits, stats, probs=[1])

        if level == 2:
            # |2> with pi-pulse
            q['readout'] = True
            q['measureAmp'] = mpa
            q.xy = eh.mix(q, eh.piPulseHD(q, t_pi-q.piLen))+eh.mix(q, env.gaussian(t_pi, q.piFWHM, q.piAmp21, df=q.piDf21), freq = 'f21')
            q.z = eh.measurePulse(q, t_meas)
            req2 = runQubits(server, qubits, stats, probs=[1])

            probs = yield FutureList([req0, req1, req2])
            p0, p1, p2 = [p[0] for p in probs]

            returnValue([p0, p1, p1-p0, p2, p2-p1])
        elif level == 1:
            probs = yield FutureList([req0, req1])
            p0, p1 = [p[0] for p in probs]

            returnValue([p0, p1, p1-p0])

    return sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)

def zPulse2FluxBias(sample,FBchange=None, stats=60, measure=0):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]
    if FBchange is None:
        FBchange = -10*np.sign(q.measureAmp)*mV
        print FBchange
    if measure == 0:
        sample.q0.biasOperate += FBchange
    elif measure == 1:
        sample.q1.biasOperate += FBchange
    mpa1=multiqubitPQ.find_mpa(sample, target=0.5, stats=stats, measure=measure, update=False)
    if measure == 0:
        sample.q0.biasOperate -= FBchange
    elif measure == 1:
        sample.q1.biasOperate -= FBchange
    mpa2=multiqubitPQ.find_mpa(sample, target=0.5, stats=stats, measure=measure, update=False)
    ratio = FBchange/(mpa1-mpa2)
    print 'unit measure pulse amplitude corresponds to %g mV flux bias.' % ratio['mV']
    Q.calUnitMPA2FBmV = ratio
    return ratio

def coherent(sample, probeLen=st.r[0:100:1,ns], drive=st.r[0:1:0.05], stats=600L, measure=0,
       name='Coherent state', save=True, collect=True, noisy=True, extraDelay=extraDelay):
    """A single pi-pulse on one qubit, with other qubits also operated."""
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    r = qubits[measure+2]

    axes = [(probeLen, 'Measure pulse length'),(drive, 'Resonator uwave drive Amp')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, currLen, currDri):
        r.xy = eh.mix(r, env.gaussian(0, r.piFWHM, currDri), freq = 'fRes0')
        q.z = env.rect(r.piLen/2+extraDelay, currLen, q.noonSwapAmp0)+eh.measurePulse(q, r.piLen/2+extraDelay+currLen)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

def coherentC(sample, probeLen=st.r[0:300:3,ns], drive=st.r[0:2:0.1], stats=600L, measure=0,
       name='Coherent state', save=True, collect=True, noisy=True, extraDelay=extraDelay):
    """A single pi-pulse on one qubit, with other qubits also operated."""
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[1-measure]
    q1 = qubits[measure]

    axes = [(probeLen, 'Measure pulse length'),(drive, 'Resonator uwave drive Amp')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)
    q0.piFWHM *= 1
    q0.piLen *= 1
    def func(server, currLen, currDri):
        q0.xy = eh.mix(q0, env.gaussian(0, q0.piFWHM, currDri), freq = 'fResC')
        q1.z = env.rect(q0.piLen/2+extraDelay, currLen, q1.noonSwapAmpC)+eh.measurePulse(q1, q0.piLen/2+extraDelay+currLen)
        q0.z = env.rect(-q0.piLen/2, q0.piLen, q0.noonSwapAmpC)
        q1['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

def resonatorSpectroscopy(sample, freqScan=None, swapTime=300*ns, stats=600L, measure=0,
       name='Resonator spectroscopy', save=True, collect=True, noisy=True):
    """A single pi-pulse on one qubit, with other qubits also operated."""
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    r = qubits[measure+2]

    if freqScan is None:
        freqScan = st.r[r.fRes0['GHz']-0.002:r.fRes0['GHz']+0.002:0.00002,GHz]
    axes = [(freqScan, 'Resonator frequency')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, curr):
        sb_freq = curr-r.fc
        r.xy = eh.spectroscopyPulse(r, 0, sb_freq)
        q.z = env.rect(r.spectroscopyLen+20*ns, swapTime, q.noonSwapAmp0)+eh.measurePulse(q, r.spectroscopyLen+30*ns+swapTime)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

def testResDelay(sample, startTime=st.r[-50:50:0.5,ns], pulseLength=8*ns, amp = 0.5, stats=600L, measure=0,
       name='Resonator test delay', save=True, collect=True, noisy=True, plot=False, update=True, extraDelay=extraDelay):
    """A single pi-pulse on one qubit, with other qubits also operated."""
    sample, qubits, Qubits = util.loadQubits(sample,write_access=True)
    q = qubits[measure]
    r = qubits[measure+2]
    R = Qubits[measure+2]

    axes = [(startTime, 'Uwave start time')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, curr):
        start = curr
        r.xy = eh.mix(r, env.gaussian(start-20, pulseLength, amp = amp), freq = 'fRes0')+eh.mix(r, env.gaussian(start+20, pulseLength, amp = -amp), freq = 'fRes0')
        q.z = env.rect(-q.noonSwapLen0/2, q.noonSwapLen0, q.noonSwapAmp0)+eh.measurePulse(q, q.noonSwapLen0/2+extraDelay)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    topLen = q.noonSwapLen0['ns']
    translength = r['piFWHM'][ns]
    def fitfunc(x, p):
        return (p[1] +
                p[2] * 0.5*erfc((x - (p[0] - topLen/2.0)) / translength) +
                p[3] * 0.5*erf((x - (p[0] + topLen/2.0)) / translength))
    x, y = result.T
    fit, _ok = leastsq(lambda p: fitfunc(x, p) - y, [0.0, 0.05, 0.85, 0.85])
    if plot:
        plt.figure()
        plt.plot(x, y, '.')
        plt.plot(x, fitfunc(x, fit))
    print 'uwave lag:', -fit[0]
    if update:
        print 'uwave lag corrected by %g ns' % fit[0]
        R['timingLagUwave'] += fit[0]*ns

def testQubResDelayCmp(sample, delay=st.r[-20:20:0.5,ns], stats=1500L,
         name='delay between qubits', save=False, collect=True, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[0]
    q1 = qubits[1]
    aaa=testQubResDelay(sample, startTime=delay, measure=0, save=save, collect=collect, noisy=noisy)
    bbb=testQubResDelay(sample, startTime=delay, measure=1, save=save, collect=collect, noisy=noisy)

    plt.figure(101)
    plt.plot(aaa[:,0],(aaa[:,1]-(1-q0.measureF0))/(q0.measureF0+q0.measureF1-1))
    plt.plot(bbb[:,0],(bbb[:,1]-(1-q1.measureF0))/(q1.measureF0+q1.measureF1-1))
    # if measure=0 is right to measure=1, add a positive value to measure=0 qubit
    return


def testQubResDelay(sample, startTime=st.r[-100:100:2,ns], pulseLength=8*ns, amp = 0.5, stats=600L, measure=0,
       name='Resonator test delay', save=True, collect=True, noisy=True, plot=False, update=True, extraDelay=extraDelay):
    """A single pi-pulse on one qubit, with other qubits also operated."""
    sample, qubits, Qubits = util.loadQubits(sample,write_access=True)
    q0 = qubits[1-measure]
    q1 = qubits[measure]

    axes = [(startTime, 'Uwave start time')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, curr):
        q0.xy = eh.mix(q0, eh.piPulseHD(q0, -q0.noonSwapLenC-5-q0.piLen/2))
        q0.z = env.rect(-q0.noonSwapLenC-5, q0.noonSwapLenC, q0.noonSwapAmpC)+ env.rect(5, q0.noonSwapLenC, q0.noonSwapAmpC)
        q1.z = env.rect(curr, q1.noonSwapLenC, q1.noonSwapAmpC)+eh.measurePulse(q1, q1.noonSwapLenC+curr)
        q1['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return result

    #===========================================================================
    # topLen = q0.noonSwapLen0['ns']
    # translength = q0['piFWHM'][ns]
    # def fitfunc(x, p):
    #    return (p[1] +
    #            p[2] * 0.5*erfc((x - (p[0] - topLen/2.0)) / translength) +
    #            p[3] * 0.5*erf((x - (p[0] + topLen/2.0)) / translength))
    # x, y = result.T
    # fit, _ok = leastsq(lambda p: fitfunc(x, p) - y, [0.0, 0.05, 0.85, 0.85])
    # if plot:
    #    plt.figure()
    #    plt.plot(x, y, '.')
    #    plt.plot(x, fitfunc(x, fit))
    # print 'uwave lag:', -fit[0]
    # if update:
    #    print 'uwave lag corrected by %g ns' % fit[0]
    #    R['timingLagUwave'] += fit[0]*ns
    #===========================================================================

def FockScan(sample, n=1, scanLen=0.0*ns, scanOS=0.0, tuneOS=False, probeFlag=False, paraName='0',stats=1500L, measure=0, delay=0*ns,
       name='Fock state swap length scan MQ', save=False, collect=True, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    axes = [(scanLen, 'Swap length adjust'),(scanOS, 'Amplitude overshoot')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, 'res '+paraName+' '+name+' '+str(n)+' Fock state', axes, measure=measure, kw=kw)

    sl = q['noonSwapLen'+paraName+'s']
    print 'Optizmizing n=%g for the swap length = %g ns...' %(n,sl[n-1])
    sa = q['noonSwapAmp'+paraName+'Read']

    if not tuneOS:
        so = np.array([0.0]*n)
    else:
        so = q['noonSwapAmpOS'+paraName+'s']

    def func(server, currLen, currOS):
        q.xy = env.NOTHING
        q.z = env.NOTHING
        start = -q.piLen/2
        for i in range(n-1):
            q.xy += eh.mix(q, eh.piPulseHD(q, start+q.piLen/2))
            start += q.piLen+delay
            q.z += env.rect(start, sl[i], sa, overshoot=so[i])
            start += sl[i]+delay
        q.xy += eh.mix(q, eh.piPulseHD(q, start+q.piLen/2))
        start += q.piLen+delay
        if not probeFlag:
            q.z += env.rect(start, sl[n-1]+currLen, sa, overshoot=so[n-1]+currOS)
            start += sl[n-1]+currLen+delay
            q.z += eh.measurePulse(q, start)
        else:
            q.z += env.rect(start, sl[n-1], sa, overshoot=so[n-1]+currOS)
            start += sl[n-1]+delay
            q.z += env.rect(start, currLen, sa)
            start += currLen+delay
            q.z += eh.measurePulse(q, start)

        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)

def FockTuner(sample, n=1, iteration=3, tuneOS=False, paraName='0',stats=1500L, measure=0, delay=0*ns,
       save=False, collect=True, noisy=True, update=True):
    sample, qubits, Qubits = util.loadQubits(sample,write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]

    if len(q['noonSwapLen'+paraName+'s'])<n:
        for i in np.arange(len(q['noonSwapLen'+paraName+'s']),n,1):
            sample['q'+str(measure)]['noonSwapLen'+paraName+'s'].append(q['noonSwapLen'+paraName+'s'][0]/np.sqrt(i+1))
    if tuneOS:
        AmpOverShoot = [q['noonSwapAmp'+paraName]]*n
    for i in np.arange(1,n+1,1):
        for iter in range(iteration):
            rf = 2**iter
            print 'iteration %g...' % iter
            sl = sample['q'+str(measure)]['noonSwapLen'+paraName+'s'][i-1]
            results = FockScan(sample, n=i, scanLen=st.PQlinspace(-max([0.3*sl['ns']/rf,1]),max([0.3*sl['ns']/rf,1]),21,'ns'),
                                    paraName=paraName,stats=stats,measure=measure,probeFlag=False,delay=delay,
                                    save=False, collect=collect, noisy=noisy)
            new, percent = datasetMinimum(results, 0, -max([0.3*sl['ns']/rf,1]), max([0.3*sl['ns']/rf,1]))
            sample['q'+str(measure)]['noonSwapLen'+paraName+'s'][i-1] += new

            if tuneOS:
                os = sample['q'+str(measure)]['noonSwapAmp'+paraName+'OSs'][i-1]
                results = FockScan(sample, n=i, scanLen=0.0*ns, tuneOS=tuneOS, scanOS=np.linspace(os*(1-0.5/rf),os*(1+0.5/rf),21),
                                        paraName=paraName,stats=stats,measure=measure,probeFlag=False,delay=delay,
                                        save=False, collect=collect, noisy=noisy)
                new, percent = datasetMinimum(results, 0, -max([0.3*sl['ns']/rf,1]), max([0.3*sl['ns']/rf,1]))
                sample['q'+str(measure)]['noonSwapLen'+paraName+'s'][i-1] += new

        if save:
            FockScan(sample, n=i, scanLen=st.arangePQ(0,100,2,'ns'),
                        paraName=paraName,stats=stats,measure=measure,probeFlag=True,delay=delay,
                        save=save, collect=collect, noisy=noisy)
    if update:
        Q['noonSwapLen'+paraName+'s'] = sample['q'+str(measure)]['noonSwapLen'+paraName+'s']
    return sample['q'+str(measure)]['noonSwapLen'+paraName+'s']

def FockScan21(sample, n=1, scanLen=0.0*ns, scanOS=0.0, tuneOS=False, probeFlag=False, paraName='1',stats=1500L, measure=0, delay=0*ns,
       name='Fock state 21 swap length scan MQ', save=False, collect=True, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    axes = [(scanLen, 'Swap length adjust'),(scanOS, 'Amplitude overshoot')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, 'res '+paraName+' '+name+' '+str(n)+' Fock state', axes, measure=measure, kw=kw)

    sl = q['noonSwapLen'+paraName+'s']
    print 'Optizmizing n=%g for the swap length = %g ns...' %(n,sl[n-1])
    sa = q['noonSwapAmp'+paraName+'Read']

    if not tuneOS:
        so = np.array([0.0]*n)
    else:
        so = q['noonSwapAmpOS'+paraName+'s']

    def func(server, currLen, currSO):
        q.xy = eh.mix(q, eh.piPulseHD(q, -q.piLen/2))
        q.z = env.NOTHING
        start = -q.piLen/2
        for i in range(n-1):
            q.xy += eh.mix(q, env.gaussian(start+q.piLen/2, q.piFWHM, q.piAmp21, df=q.piDf21), freq = 'f21')
            start += q.piLen+delay
            q.z += env.rect(start, sl[i], sa, overshoot=so[i])
            start += sl[i]+delay
        q.xy += eh.mix(q, env.gaussian(start+q.piLen/2, q.piFWHM, q.piAmp21, df=q.piDf21), freq = 'f21')
        start += q.piLen+delay
        if not probeFlag:
            q.z += env.rect(start, sl[n-1]+currLen, sa, overshoot=so[n-1])
            start += sl[n-1]+currLen+delay
            q.z += eh.measurePulse2(q, start)
        else:
            q.z += env.rect(start, sl[n-1], sa, overshoot=so[n-1])
            start += sl[n-1]+delay
            q.z += env.rect(start, currLen, sa)
            start += currLen+delay
            q.z += eh.measurePulse2(q, start)

        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)

def FockTuner21(sample, n=1, iteration=3, tuneOS=False, paraName='1',stats=1500L, measure=0, delay=0*ns,
       save=False, collect=True, noisy=True, update=True):
    sample, qubits, Qubits = util.loadQubits(sample,write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]

    if len(q['noonSwapLen'+paraName+'s'])<n:
        for i in np.arange(len(q['noonSwapLen'+paraName+'s']),n,1):
            sample['q'+str(measure)]['noonSwapLen'+paraName+'s'].append(q['noonSwapLen'+paraName+'s'][0]/np.sqrt(i+1/2.0))
    if tuneOS:
        AmpOverShoot = [q['noonSwapAmp'+paraName]]*n
    for i in np.arange(1,n+1,1):
        for iter in range(iteration):
            rf = 2**iter
            print 'iteration %g...' % iter
            sl = sample['q'+str(measure)]['noonSwapLen'+paraName+'s'][i-1]
            results = FockScan21(sample, n=i, scanLen=st.PQlinspace(-max([0.3*sl['ns']/rf,1]),max([0.3*sl['ns']/rf,1]),21,'ns'),
                                    paraName=paraName,stats=stats,measure=measure,probeFlag=False,delay=delay,
                                    save=False, collect=collect, noisy=noisy)
            new, percent = datasetMinimum(results, 0, -max([0.3*sl['ns']/rf,1]), max([0.3*sl['ns']/rf,1]))
            sample['q'+str(measure)]['noonSwapLen'+paraName+'s'][i-1] += new

        if save:
            FockScan21(sample, n=i, scanLen=st.arangePQ(0,100,1,'ns'),
                        paraName=paraName,stats=stats,measure=measure,probeFlag=True,delay=delay,
                        save=save, collect=collect, noisy=noisy)
    if update:
        Q['noonSwapLen'+paraName+'s'] = sample['q'+str(measure)]['noonSwapLen'+paraName+'s']
    return sample['q'+str(measure)]['noonSwapLen'+paraName+'s']

def plotRhoSingle(rho, scale=1.0, color=None, width=0.05, headwidth=0.1, headlength=0.1, chopN=None, amp=1.0, figNo=100):
    pylab.figure(figNo)
    pylab.clf()
    rho=rho.copy()
    s=numpy.shape(rho)
    if chopN!=None:
        rho = rho[:chopN,:chopN]
    s=numpy.shape(rho)
    rho = rho*amp
    ax = pylab.gca()
    ax.set_aspect(1.0)
    pos = ax.get_position()
    r = numpy.real(rho)
    i = numpy.imag(rho)
    x = numpy.arange(s[0])[None,:] + 0*r
    y = numpy.arange(s[1])[:,None] + 0*i
    pylab.quiver(x,y,r,i,units='x',scale=1.0/scale, width=width, headwidth=headwidth, headlength=headlength, color=color)
    pylab.xticks(numpy.arange(s[1]))
    pylab.yticks(numpy.arange(s[0]))
    pylab.xlim(-0.9999,s[1]-0.0001)
    pylab.ylim(-0.9999,s[0]-0.0001)
    return rho

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


def nightCPHASEwithRecovery1(sample):

    alpha = np.linspace(-2.0,2.0,25)
    alpha = alpha[:,None]+1.0j*alpha[None,:]
    alpha = np.reshape(alpha,np.size(alpha))

    run2DWithRecover_v2(sample, CPHASEPreRotArbSQ_TOMO, 'probeLen', st.arangePQ(0,200,2,ns),
                        disp=alpha, phase=np.pi, dfRot=102.53*MHz, stats=1500L, CPHASETime = 21.459*ns, measure=1, save=True)

    run2DWithRecover_v2(sample, CPHASEPreRotArbSQ0p1_TOMO, 'probeLen', st.arangePQ(0,200,2,ns),
                        disp=alpha, phase=np.pi, dfRot=88.20*MHz, stats=1500L, CPHASETime = 76.373*ns, measure=1, save=True)

    run2DWithRecover_v2(sample, CPHASEPreRotArbSQSwap, 'probeLen', st.arangePQ(0,500,1,ns),
                        swap21Length=st.arangePQ(0,300,0.1,ns), CPHASETime = 42.203*ns, stats=1800L,
                        delay=0.0*ns, delayA=0.0*ns, phase=np.pi, dfRot=102.53*MHz, measure=1, save=True)

def nightCPHASEwithRecovery2(sample):

    alpha1 = np.linspace(-2.5,2.5,25)
    alpha1 = alpha1[:,None]+1.0j*alpha1[None,:]
    alpha1 = np.reshape(alpha1,np.size(alpha1))

    run2DWithRecover_v2(sample, CPHASEPreRotArbSQ_TOMOReset, 'probeLen', st.arangePQ(0,200,2,ns),
                        n=2, disp=alpha1, phase=np.pi, dfRot=105.0*MHz, stats=1500L, CPHASETime = 28.636*ns, measure=1, save=True)

    run2DWithRecover_v2(sample, CPHASEPreRotArbSQ_TOMOReset, 'probeLen', st.arangePQ(0,200,2,ns),
                        n=2, disp=alpha1, phase=np.pi, dfRot=105.0*MHz, stats=1500L, CPHASETime = 15.653*ns, measure=1, save=True)

    alpha2 = np.linspace(-2.0,2.0,25)
    alpha2 = alpha2[:,None]+1.0j*alpha2[None,:]
    alpha2 = np.reshape(alpha2,np.size(alpha2))

    run2DWithRecover_v2(sample, CPHASEPreRotArbSQ0p1_TOMO, 'probeLen', st.arangePQ(0,200,2,ns),
                        disp=alpha2, phase=np.pi, dfRot=87.70*MHz, stats=1500L, CPHASETime = 38.184*ns, measure=1, save=True)

    alpha3 = np.linspace(-1.5,1.5,15)
    alpha3 = alpha3[:,None]+1.0j*alpha3[None,:]
    alpha3 = np.reshape(alpha3,np.size(alpha3))

    run2DWithRecover_v2(sample, CPHASEPreRotArbSQ_TOMOZero, 'probeLen', st.arangePQ(0,200,2,ns),
                        disp=alpha3, phase=np.pi, dfRot=99.1*MHz, stats=1500L, CPHASETime = 42.082*ns , measure=1, save=True)

def nightCPHASEwithRecovery3(sample):

    run2DWithRecover_v2(sample, QuResCPHASEPreRotArbP01P12TwoD, 'swap21Length', st.arangePQ(0,300,0.1,ns),
                        phase=np.arange(0,2*np.pi,np.pi/50), n=1, measure=0, stats=1200L, delay=0.0*ns, delayA=0.0*ns, dfRot=287.45*MHz, save=True)

def nightCPHASEwithRecovery4(sample):

    run2DWithRecover_v2(sample, QuResCPHASEPreRotArbP01P12TwoD, 'swap21Length', st.arangePQ(0,121,0.2,ns),
                        phase=np.arange(0,2*np.pi,np.pi/50), n=2, measure=0, stats=900L, delay=0.0*ns, delayA=0.0*ns, dfRot=284.62*MHz, save=True)

    run2DWithRecover_v2(sample, QuResCPHASEPreRotArbP01P12ZeroTwoD, 'swap21Length', st.arangePQ(0,150,0.2,ns),
                        phase=np.arange(0,2*np.pi,np.pi/50), n=0, measure=0, stats=900L, delay=0.0*ns, delayA=0.0*ns, dfRot=287.64*MHz, save=True)

def nightCPHASEwithRecovery5(sample):

    run2DWithRecover_v2(sample, QuResCPHASEPreRotArbP01P12TwoD, 'swap21Length', st.arangePQ(0,121,0.2,ns),
                        phase=np.arange(0,2*np.pi,np.pi/50), n=3, measure=0, stats=900L, delay=0.0*ns, delayA=0.0*ns, dfRot=282.39*MHz, save=True)

    run2DWithRecover_v2(sample, QuResCPHASEPreRotArb0p1P01P12TwoD, 'swap21Length', st.arangePQ(0,121,0.2,ns),
                        phase=np.arange(0,2*np.pi,np.pi/50), measure=0, stats=900L, delay=0.0*ns, delayA=0.0*ns, dfRot=287.09*MHz, save=True)

    run2DWithRecover_v2(sample, repeatQuResCPHASEPreRotArbTOMO, 'CPHASETime', st.arangePQ(0,121.0,0.2,ns),
                        n=1, stats=600L, nameEx='runFock1', dfRot=0.0*MHz)

    run2DWithRecover_v2(sample, repeatQuResCPHASEPreRotArbTOMO, 'CPHASETime', st.arangePQ(0,121.0,0.2,ns),
                        n=2, stats=600L, nameEx='runFock2', dfRot=0.0*MHz)

    run2DWithRecover_v2(sample, repeatQuResCPHASEPreRotArbTOMO, 'CPHASETime', st.arangePQ(0,121.0,0.2,ns),
                        n=3, stats=600L, nameEx='runFock3', dfRot=0.0*MHz)

    run2DWithRecover_v2(sample, repeatQuResCPHASEPreRotArbZeroTOMO, 'CPHASETime', st.arangePQ(0,121.0,0.2,ns),
                        n=0, stats=600L, nameEx='runFock0', dfRot=0.0*MHz)

    run2DWithRecover_v2(sample, repeatQuResCPHASEPreRotArb0p1TOMO, 'CPHASETime', st.arangePQ(0,121.0,0.2,ns),
                        stats=600L, nameEx='runFock0p1', dfRot=0.0*MHz)

def nightCPHASEwithoutRecovery1(sample):

    QuResCPHASEPreRotArbP01P12TwoD(sample, swap21Length=st.arangePQ(0,121,0.2,ns),
                                   phase=np.arange(0,2*np.pi,np.pi/50), n=3, measure=0, stats=900L,
                                   delay=0.0*ns, delayA=0.0*ns, dfRot=273.00*MHz, save=True)

    QuResCPHASEPreRotArb0p1P01P12TwoD(sample, swap21Length=st.arangePQ(0,121,0.2,ns),
                                      phase=np.arange(0,2*np.pi,np.pi/50), measure=0, stats=900L,
                                      delay=0.0*ns, delayA=0.0*ns, dfRot=277.57*MHz, save=True)

    repeatQuResCPHASEPreRotArbTOMO(sample, CPHASETime=st.arangePQ(0,121.0,0.2,ns),
                                   n=1, stats=600L, nameEx='runFock1', dfRot=0.0*MHz)

    repeatQuResCPHASEPreRotArbTOMO2(sample, CPHASETime=st.arangePQ(0,121.0,0.2,ns),
                                    n=1, stats=600L, nameEx='runFock12', dfRot=0.0*MHz)

#    repeatQuResCPHASEPreRotArbTOMO(sample, CPHASETime=st.arangePQ(0,121.0,0.2,ns),
#                                   n=2, stats=600L, nameEx='runFock2', dfRot=0.0*MHz)
#
#    repeatQuResCPHASEPreRotArbTOMO2(sample, CPHASETime=st.arangePQ(0,121.0,0.2,ns),
#                                    n=2, stats=600L, nameEx='runFock22', dfRot=0.0*MHz)

def nightCPHASEwithoutRecovery2(sample):

    QuResCPHASEPreRotArbSwap(sample, n=1, probeLen=st.arangePQ(0,500,1,ns), swap21Length=st.arangePQ(0,121,1,ns),
                             CPHASETime = 37.50*ns, measure=1, stats=900L,
                             delay=0.0*ns, delayA=0.0*ns, phase=np.pi, dfRot=287.45*MHz, save=True)

    QuResCPHASEPreRotArbSwapZero(sample, n=0, probeLen=st.arangePQ(0,500,1,ns), swap21Length=st.arangePQ(0,121,1,ns),
                                 CPHASETime = 37.50*ns, measure=1, stats=900L,
                                 delay=0.0*ns, delayA=0.0*ns, phase=np.pi, dfRot=287.45*MHz, save=True)

    repeatQuResCPHASEPreRotArbTOMO(sample, CPHASETime=st.arangePQ(0,121.0,0.2,ns),
                                   n=2, stats=600L, nameEx='runFock2', dfRot=0.0*MHz)

    repeatQuResCPHASEPreRotArbTOMO2(sample, CPHASETime=st.arangePQ(0,121.0,0.2,ns),
                                    n=2, stats=600L, nameEx='runFock22', dfRot=0.0*MHz)

    repeatQuResCPHASEPreRotArbTOMO(sample, CPHASETime=st.arangePQ(0,121.0,0.2,ns),
                                   n=3, stats=600L, nameEx='runFock3', dfRot=0.0*MHz)

    repeatQuResCPHASEPreRotArbTOMO2(sample, CPHASETime=st.arangePQ(0,121.0,0.2,ns),
                                    n=3, stats=600L, nameEx='runFock32', dfRot=0.0*MHz)

    repeatQuResCPHASEPreRotArbZeroTOMO(sample, CPHASETime=st.arangePQ(0,121.0,0.2,ns),
                                       n=0, stats=600L, nameEx='runFock0', dfRot=0.0*MHz)

    repeatQuResCPHASEPreRotArbZeroTOMO2(sample, CPHASETime=st.arangePQ(0,121.0,0.2,ns),
                                        n=0, stats=600L, nameEx='runFock02', dfRot=0.0*MHz)

    repeatQuResCPHASEPreRotArb0p1TOMO(sample, CPHASETime=st.arangePQ(0,121.0,0.2,ns),
                                      stats=600L, nameEx='runFock0p1', dfRot=0.0*MHz)

    repeatQuResCPHASEPreRotArb0p1TOMO2(sample, CPHASETime=st.arangePQ(0,121.0,0.2,ns),
                                       stats=600L, nameEx='runFock0p12', dfRot=0.0*MHz)

def nightCPHASEwithoutRecovery3(sample):

    QuResCPHASEPreRotArbSwap(sample, n=1, probeLen=st.arangePQ(0,500,1,ns), swap21Length=st.arangePQ(0,121,1,ns),
                             CPHASETime = 37.06*ns, measure=1, stats=900L,
                             delay=0.0*ns, delayA=0.0*ns, phase=np.pi, dfRot=287.45*MHz, save=True)

    QuResCPHASEPreRotArbSwap(sample, n=2, probeLen=st.arangePQ(0,500,1,ns), swap21Length=st.arangePQ(0,121,1,ns),
                             CPHASETime = 26.21*ns, measure=1, stats=900L,
                             delay=0.0*ns, delayA=0.0*ns, phase=np.pi, dfRot=285.60*MHz, save=True)

    QuResCPHASEPreRotArbSwap(sample, n=3, probeLen=st.arangePQ(0,500,1,ns), swap21Length=st.arangePQ(0,121,1,ns),
                             CPHASETime = 21.40*ns, measure=1, stats=900L,
                             delay=0.0*ns, delayA=0.0*ns, phase=np.pi, dfRot=282.78*MHz, save=True)

    QuResCPHASEPreRot0p1Swap(sample, probeLen=st.arangePQ(0,500,1,ns), swap21Length=st.arangePQ(0,121,1,ns),
                             CPHASETime = 37.06*ns, measure=1, stats=900L,
                             delay=0.0*ns, delayA=0.0*ns, phase=np.pi, dfRot=277.57*MHz, save=True)

    repeatQuResCPHASEPreRotArbZeroTOMO(sample, CPHASETime=st.arangePQ(0,121.0,0.2,ns),
                                       n=0, stats=600L, nameEx='runFock0', dfRot=0.0*MHz)

    repeatQuResCPHASEPreRotArbZeroTOMO2(sample, CPHASETime=st.arangePQ(0,121.0,0.2,ns),
                                        n=0, stats=600L, nameEx='runFock02', dfRot=0.0*MHz)

    repeatQuResCPHASEPreRotArb0p1TOMO(sample, CPHASETime=st.arangePQ(0,121.0,0.2,ns),
                                      stats=600L, nameEx='runFock0p1', dfRot=0.0*MHz)

    repeatQuResCPHASEPreRotArb0p1TOMO2(sample, CPHASETime=st.arangePQ(0,121.0,0.2,ns),
                                       stats=600L, nameEx='runFock0p12', dfRot=0.0*MHz)

def nightCPHASEwithoutRecovery4(sample):

    alpha1 = np.linspace(-2.5,2.5,25)
    alpha1 = alpha1[:,None]+1.0j*alpha1[None,:]
    alpha1 = np.reshape(alpha1,np.size(alpha1))

    CPHASEPreRotArbSQ_TOMOReset(sample, probeLen=st.arangePQ(0,200,2,ns), n=3, disp=alpha1, phase=np.pi, dfRot=105.99*MHz,
                                stats=1500L, CPHASETime = 24.95*ns, measure=1, save=True)

    CPHASEPreRotArbSQ_TOMOReset(sample, probeLen=st.arangePQ(0,200,2,ns), n=3, disp=alpha1, phase=np.pi, dfRot=105.99*MHz,
                                stats=1500L, CPHASETime = 12.48*ns, measure=1, save=True)

    CPHASEPreRotArbSQSwap(sample, probeLen=st.arangePQ(0,500,1,ns), swap21Length=st.arangePQ(0,121,1,ns),
                          CPHASETime = 42.2*ns, stats=900L, delay=0.0*ns, delayA=0.0*ns, phase=np.pi, dfRot=102.53*MHz, measure=1, save=True)

def nightCPHASEwithoutRecovery5(sample):

    QuResCPHASEPreRotArbSwap(sample, n=1, probeLen=st.arangePQ(0,500,1,ns), swap21Length=st.arangePQ(0,121,1,ns),
                             CPHASETime = 36.0*ns, measure=1, stats=900L,
                             delay=0.0*ns, delayA=0.0*ns, phase=np.pi, dfRot=300.0*MHz, save=True)

    QuResCPHASEPreRotArbSwap(sample, n=2, probeLen=st.arangePQ(0,500,1,ns), swap21Length=st.arangePQ(0,121,1,ns),
                             CPHASETime = 25.46*ns, measure=1, stats=900L,
                             delay=0.0*ns, delayA=0.0*ns, phase=np.pi, dfRot=300.0*MHz, save=True)

    QuResCPHASEPreRotArbSwap(sample, n=3, probeLen=st.arangePQ(0,500,1,ns), swap21Length=st.arangePQ(0,121,1,ns),
                             CPHASETime = 20.78*ns, measure=1, stats=900L,
                             delay=0.0*ns, delayA=0.0*ns, phase=np.pi, dfRot=300.0*MHz, save=True)

    QuResCPHASEPreRot0p1Swap(sample, probeLen=st.arangePQ(0,500,1,ns), swap21Length=st.arangePQ(0,121,1,ns),
                             CPHASETime = 36.0*ns, measure=1, stats=900L,
                             delay=0.0*ns, delayA=0.0*ns, phase=np.pi, dfRot=300.0*MHz, save=True)

#    repeatQuResCPHASEPreRotArb0p1TOMO(sample, CPHASETime=st.arangePQ(0,121.0,0.2,ns),
#                                      stats=600L, nameEx='runFock0p1', dfRot=0.0*MHz)
#
#    repeatQuResCPHASEPreRotArb0p1TOMO2(sample, CPHASETime=st.arangePQ(0,121.0,0.2,ns),
#                                       stats=600L, nameEx='runFock0p12', dfRot=0.0*MHz)

def run2DWithRecover_v2(sample,function,sweepName=None,sweep=None,**kw):
    """run2DWithRecover_v2(s, singlePhotonQNDPreRotScanArb, 'swap21Length', st.arangePQ(start,stop,step,unit), anything else)"""

    if sweepName is None:
        raise Exception('Hey, Matteo, you have to specify the name of the sweep variable!')
    if sweep is None:
        raise Exception('You must enter swap21Length')
    unit = sweep[0].units
    sweepStart = sweep[0]
    sweepEnd = sweep[-1]
    sweepStep = sweep[1]-sweep[0]
    finished = False
    hasFailed = False

    while finished is False:

        if hasFailed is True:
            print 'Running the failure block'
            with labrad.connection() as cxn:
                dv = cxn.data_vault
                #Figure out where to start the sweep
                print 'Reading datavault'
                pathToData = sample.__dict__['_dir']
                dv.cd(pathToData)
                dv.open(dv.dir()[1][-1])
                data = np.asarray(dv.get())
                swapAmpStart = np.max(data[:,0])
        try:
            print 'Entered try statement'
            sweep = st.arangePQ(sweepStart,sweepEnd,sweepStep,unit)
            #############
            # Make sure sweepName is correct for the fuction you're running
            #############
            kw[sweepName]=sweep
            print 'Running main function...'
            function(sample,**kw)
            finished = True
        except:
            print 'There has been an exception'
            hasFailed = True
            #Restart relevant servers
            with labrad.connection() as cxn:
                dr = cxn.node_dr
                jingle = cxn.node_jingle
                ghz = cxn.ghz_dacs
                print 'Restarting DR Direct Ethernet'
                dr.restart('DR Direct Ethernet')
                print 'Restarting GHz DACs'
                jingle.restart('GHz DACs')
                print 'Restarting Qubit sequencer'
                jingle.restart('Qubit Sequencer')

                bringupFailure = True
                while bringupFailure is True:
                    bringupFailure = False
                    print 'Running bringup script'
                    bringupAll(cxn)
                    print 'Checking PLL  unlocks'
                    for device in ghz.list_devices()[0]:
                        ghz.select_device(device)
                        bringupFailure = bringupFailure or ghz.pll_query()


#def run2DWithRecover(sample,function,swap21Length=None,**kw):
#    if swap21Length is None:
#        raise Exception('You must enter swap21Length')
#    unit = swap21Length[0].units
#    swapAmpStart = swap21Length[0]
#    swapAmpEnd = swap21Length[-1]
#    swapAmpStep = swap21Length[1]-swap21Length[0]
#    finished = False
#    hasFailed = False
#
#    while finished is False:
#
#        if hasFailed is True:
#            print 'Running the failure block'
#            with labrad.connection() as cxn:
#                dv = cxn.data_vault
#                #Figure out where to start the sweep
#                print 'Reading datavault'
#                pathToData = sample.__dict__['_dir']
#                dv.cd(pathToData)
#                dv.open(dv.dir()[1][-1])
#                data = dv.get().asarray
#                swapAmpStart = np.max(data[:,0])
#        try:
#            testCounter = testCounter+1
#            print 'Entered try statement'
#            swap12Length = st.arangePQ(swapAmpStart,swapAmpEnd,swapAmpStep,unit)
#            kw['swap21Length']=swap21Length
#            print 'Running main function...'
#            function(sample,**kw)
#            finished = True
#        except:
#            print 'There has been an exception'
#            hasFailed = True
#            #Restart relevant servers
#            with labrad.connection() as cxn:
#                dr = cxn.node_dr
#                jingle = cxn.node_jingle
#                ghz = cxn.ghz_dacs
#                print 'Restarting DR Direct Ethernet'
#                dr.restart('DR Direct Ethernet')
#                print 'Restarting GHz DACs'
#                jingle.restart('GHz DACs')
#                print 'Restarting Qubit sequencer'
#                jingle.restart('Qubit Sequencer')
#
#                bringupFailure = True
#                while bringupFailure is True:
#                    bringupFailure = False
#                    print 'Running bringup script'
#                    bringupAll(cxn)
#                    print 'Checking PLL  unlocks'
#                    for device in ghz.list_devices()[0]:
#                        ghz.select_device(device)
#                        bringupFailure = bringupFailure or ghz.pll_query()

def bringupAll(cxn):
    print 'Bringup script: started bringup'
    if True: #Didn't want to unindent
        fpga = cxn.ghz_dacs

        boardList = [b[1] for b in fpga.list_devices()]
        successList = [True]*len(boardList)

        for i, board in enumerate(boardList):
            print 'Connecting to %s...' % board
            fpga.select_device(board)

            print 'Initializing PLL...'
            fpga.pll_reset()
            time.sleep(0.100)
            fpga.pll_init()

            for dac in ['A', 'B']:
                print 'Initializing DAC %s...' % dac
                fpga.dac_init(dac, True)

                print 'Setting DAC %s LVDS Offset...' % dac
                lvds = fpga.dac_lvds(dac)
                print '  SD: %d' % lvds[2]
                y = '  y:  '
                z = '  z:  '

                y += ''.join('_-'[ind[0]] for ind in lvds[3])
                z += ''.join('_-'[ind[1]] for ind in lvds[3])

                print y
                print z
                print 'Setting DAC %s FIFO Offset...' % dac
                fifo = fpga.dac_fifo(dac)
                print '  Operating SD: %2d' %  fifo[0]
                print '  Stable SD:    %2d' %  fifo[1]
                print '  Clk Polarity:  %1d' % fifo[2]
                print '  FIFO Offset:   %1d' % fifo[3]
                print '  FIFO Counter:  %1d (should be 3)' % fifo[4]
                successList[i] = successList[i] and (fifo[4] == 3)
                print 'Running DAC %s BIST...' % dac,
                bistdata = [random.randint(0, 0x3FFF) for j in range(1000)]
                success, thy, lvds, fifo = fpga.dac_bist(dac, bistdata)
                print 'success' if success else 'failure!'
                print '  Theory: %08X, %08X' % thy
                print '  LDVS:   %08X, %08X' % lvds
                print '  FIFO:   %08X, %08X' % fifo
                successList[i] = successList[i] and success
            print

        for board, success in zip(boardList, successList):
            print '%s: %s' % (board, 'ok' if success else 'failure!')

def saveArray(fname,data,complexFlag=False,format="%.6f,"):
    if complexFlag==False:
        if sum(data.imag)>=1e-3:
            print 'Warning, complex value detected, no data being recorded'
            return
    f = file(fname,'w')
    sd = shape(data)
    if len(sd)>2:
        print 'Error! Only 2D or 1D array excepted'
    if len(sd)==1:
        sdt=zeros(2,dtype=int)
        sdt[1]=sd[0]
        sdt[0]=1
        sd=sdt
        data=reshape(data,(1,sd[1]))
    for ir in range(sd[0]):
        s = ''
        for ic in range(sd[1]):
            if complexFlag:
                s += format  %real(data[ir,ic])
                s += format  %imag(data[ir,ic])
            else:
                s += format  %data[ir,ic]
        f.write(s[:-1] + "\n")
    f.close()

def readArray(fname,complexFlag=False):
    f = file(fname)
    lines = f.readlines()
    f.close()

    data = asarray([[float(n) for n in line.strip().split(',')] for line in lines])
    sd = shape(data)
    if complexFlag:
        data0 = zeros((sd[0],sd[1]/2))
        data0 = data[:,0:-1:2]+data[:,1::2]*1j
    else:
        data0 = zeros((sd[0],sd[1]))
        data0 = data

    return data0
