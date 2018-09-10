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
from pyle.dataking import measurement
from pyle.dataking import multiqubitPQ
from pyle.dataking import sweeps
from pyle.dataking.fpgaseq import runQubits
from pyle.util import sweeptools as st
from pyle.plotting import dstools
from pyle.dataking.multiqubitPQ import freqtuner, find_mpa
import pdb
#from fourierplot import fitswap

def test(sample):
    swap1 = swap21(sample, measure=0, swapLen=st.r[0:250:4,ns], swapAmp=st.r[-0.3:0.5:0.008], collect=True)
    swap0 = swap(sample, measure=0, swapLen=st.r[0:500:5,ns], swapAmp=st.r[0.1:0.6:0.008], collect=True)
    return swap0, swap1


def swap(sample, swapLen=None, swapAmp=None, measure=0, stats=1500L,
         name='coupling resonator swap MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    if swapLen is None:
        swapLen = q.swapLen
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

def swap21(sample, swapLen=None, swapAmp=None, measure=0, stats=1500L,
         name='coupling resonator swap MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    if swapLen is None:
        swapLen = q.swapLen
    if swapAmp is None:
        swapAmp = q.swapAmp

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

        Q['f10'] = freqtuner(sample, iterations=1, tEnd=100*ns, timeRes=1*ns, nfftpoints=4000, stats=1200, df=50*MHz,
              measure=measure, save=False, plot=False, noisy=noisy)
    # save updated values
    if update:
        Q['piAmp'] = amp
    return amp

def rabihigh10(sample, amplitude=st.r[0.0:1.5:0.05], measureDelay=None, measure=0, stats=1500L,
                name='Rabi-pulse height MQ', save=False, collect=False, noisy=True):
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
        q.xy = eh.mix(q, eh.piPulseHD(q, 0))
        q.z = eh.measurePulse(q, delay)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)

def pituner21(sample, measure=0, iterations=2, npoints=21, stats=1500L, save=False, update=True, noisy=True, findMPA=True):

    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    amp = q['piAmp21']
    df = (q['piDf21'])['MHz']
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

def t2(sample, delay=st.r[-10:750:2,ns], stats=600L, measure=0,
       name='T1 level2 MQ', save=True, collect=True, noisy=True):
    """A single pi-pulse on one qubit, with other qubits also operated."""
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    axes = [(delay, 'Measure pulse delay')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, delay):
        q.xy = eh.mix(q, eh.piPulseHD(q, 0))+eh.mix(q, env.gaussian(q.piLen, q.piFWHM, q.piAmp21, df = q.piDf21), freq = 'f21')
        q.z = eh.measurePulse2(q, delay)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)


def resonatorT1(sample, delay=st.arangePQ(0,1,0.01,'us')+st.arangePQ(1,8,0.1,'us'),stats=600L, measure=0,
       name='resonator T1 MQ', save=True, collect=True, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    axes = [(delay, 'Measure pulse delay')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    sl = q.noonSwapLenC
    sa = q.noonSwapAmpC

    sl = q.noonSwapLen0
    sa = q.noonSwapAmp0

    def func(server, delay):
        q.xy = eh.mix(q, eh.piPulseHD(q, 0))
        q.z = env.rect(q.piLen/2, sl, sa)
        q.z += env.rect(q.piLen/2+sl+delay, sl, sa)
        q.z += eh.measurePulse(q, q.piLen/2+sl+delay+sl)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)


def noon(sample, probeLen=st.arangePQ(0,100,1,ns), n=1, measure=[0,1], stats=1500L,
         name='noon state MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure[0]]
    q1 = qubits[measure[1]]

    axes = [(probeLen, 'swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, curr):
        start = 0
        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2
        q0.z = env.rect(start, q0.noonSwapLenC/2, q0.noonSwapAmpC)
        start += q0.noonSwapLenC/2
        q1.z = env.rect(start, q1.noonSwapLenC, q1.noonSwapAmpC)

        start += q1.noonSwapLenC
        for i in range(n-1):
            q0.xy += eh.mix(q0, env.gaussian(start, q0.piFWHM, q0.piAmp21, df = q0.piDf21), freq = 'f21')
            q1.xy  = eh.mix(q1, env.gaussian(start, q1.piFWHM, q1.piAmp21, df = q1.piDf21), freq = 'f21')

            start += np.max([q0.piLen,q1.piLen])
            q0.z += env.rect(start, q0.noonSwapLen1/np.sqrt(i+1), q0.noonSwapAmp1)
            q1.z += env.rect(start, q1.noonSwapLen1/np.sqrt(i+1), q1.noonSwapAmp1)
            start += np.max([q0.noonSwapLen1/np.sqrt(i+1),q1.noonSwapLen1/np.sqrt(i+1)])

        q0.z += env.rect(start, q0.noonSwapLen0/np.sqrt(n), q0.noonSwapAmp0)
        q1.z += env.rect(start, q1.noonSwapLen0/np.sqrt(n), q1.noonSwapAmp0)

        start += np.max([q0.noonSwapLen0,q1.noonSwapLen0])
        q0.z += env.rect(start, curr, q0.noonSwapAmp0)
        q1.z += env.rect(start, curr, q1.noonSwapAmp0)

        start += curr
        q0.z += eh.measurePulse(q0, start)
        q1.z += eh.measurePulse(q1, start)

        q0['readout'] = True
        q1['readout'] = True
        return runQubits(server, qubits, stats=stats)
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

def visibility(sample, mpa=st.r[0:2:0.05], stats=300, measure=0,
               save=True, name='Visibility MQ', collect=True, update=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    axes = [(mpa, 'Measure pulse amplitude')]
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

        # |2> with pi-pulse
        q['readout'] = True
        q['measureAmp'] = mpa
        q.xy = eh.mix(q, eh.piPulseHD(q, t_pi-q.piLen))+eh.mix(q, env.gaussian(t_pi, q.piFWHM, q.piAmp21, df=q.piDf21), freq = 'f21')
        q.z = eh.measurePulse(q, t_meas)
        req2 = runQubits(server, qubits, stats, probs=[1])

        probs = yield FutureList([req0, req1, req2])
        p0, p1, p2 = [p[0] for p in probs]

        returnValue([p0, p1, p1-p0, p2, p2-p1])
    return sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)

def zPulse2FluxBias(sample,FBchange=None, stats=60, measure=0):
    sample, qubit, Qubit = util.loadQubits(sample, write_access=True)
    q = qubit[measure]
    Q = Qubit[measure]
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
       name='Coherent state', save=True, collect=True, noisy=True):
    """A single pi-pulse on one qubit, with other qubits also operated."""
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    r = qubits[measure+2]

    axes = [(probeLen, 'Measure pulse length'),(drive, 'Resonator uwave drive Amp')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, currLen, currDri):
        r.xy = eh.mix(r, env.gaussian(0, r.piFWHM, currDri), freq = 'fRes0')
        q.z = env.rect(r.piLen/2, currLen, q.noonSwapAmp0)+eh.measurePulse(q, r.piLen/2+currLen)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

def testResDelay(sample, startTime=st.r[-50:50:0.5,ns], pulseLength=8*ns, amp = 0.5, stats=600L, measure=0,
       name='Resonator test delay', save=True, collect=True, noisy=True, plot=False, update=True):
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
        q.z = env.rect(-q.noonSwapLen0/2, q.noonSwapLen0, q.noonSwapAmp0)+eh.measurePulse(q, q.noonSwapLen0/2+4)
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

