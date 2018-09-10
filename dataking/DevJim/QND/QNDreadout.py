import numpy as np
from numpy import *
from scipy.optimize import leastsq
from scipy.special import erf, erfc
import pylab as plt
import pylab
import numpy
import labrad
from labrad.units import ns, GHz, MHz, mV, V, dBm, us,rad
import random

from math import atan2,asin
from scipy import optimize, interpolate


import pyle
from pyle.pipeline import returnValue, FutureList
from pyle.dataking import util
from pyle import envelopes as env
from pyle.dataking import envelopehelpers as eh
from pyle.dataking import measurement
from pyle.dataking import multiqubitPQ as mq
from pyle.dataking import sweeps
from pyle.dataking.QND.fpgaseq_QND import runQubits
from pyle.util import sweeptools as st
from pyle.dataking import utilMultilevels as ml
from pyle.plotting import dstools
from pyle.fitting import fitting
from pyle.dataking.multiqubitPQ import freqtuner, find_mpa, rabihigh
import pdb
from fourierplot import fitswap
import time
from pyle import tomo
from pyle.dataking.multiqubitPQ import testdelay, pulseshape
from pyle.dataking import adjust
from pyle.analysis import FluxNoiseAnalysis as fna
from pyle.dataking.QND import fpgaseq_QND

from newarbitrarystate import sequence

fpgaseq_QND.PREPAD = 250

extraDelay = 0*ns

cxn=labrad.connect('jingle')
#import labrad
#cxn=labrad.connect()


def longscan(sample, measure=0 ):
    #sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    #q, Q = qubits[measure], Qubits[measure]
    # QubitSpecWithPhotons(sample, qLen=5000*ns, resLen = 5000*ns, photonNum = np.linspace(10,100,101), freq = st.r[6.03:5.975:-0.001, GHz], stats=900, Resfreq = 6.5664*GHz, sb_freq = -200*MHz, squid=True,SZ=0.1 )

    session=['','Yu','QND','wfr20110330','r4c3','120320']
    ampl=np.sqrt(10)/73.0
    pulse_qstate_multi(sample,cxn,coherent=True,amp=ampl,SZ=0.115,SZ2=0.115,stats=3000,reps=300,delay=200*ns,releaseLen=200*ns,sb_freq=115*MHz,releaseMode=1,release2=False,session=session)
    pulse_qstate_multi(sample,cxn,coherent=True,amp=ampl,SZ=0.115,SZ2=0.115,stats=3000,reps=300,delay=200*ns,releaseLen=200*ns,sb_freq=115*MHz,releaseMode=1,release2=True,session=session)
    pulse_qstate_multi(sample,cxn,coherent=True,amp=ampl,SZ=0.115,SZ2=0.155,stats=3000,reps=300,delay=200*ns,releaseLen=200*ns,sb_freq=115*MHz,releaseMode=1,release2=True,session=session)
    pulse_qstate_multi(sample,cxn,coherent=True,amp=ampl,SZ=0.115,SZ2=0.195,stats=3000,reps=300,delay=200*ns,releaseLen=200*ns,sb_freq=115*MHz,releaseMode=1,release2=True,session=session)
    pulse_qstate_multi(sample,cxn,coherent=True,amp=ampl,SZ=0.115,SZ2=0.225,stats=3000,reps=300,delay=200*ns,releaseLen=200*ns,sb_freq=115*MHz,releaseMode=1,release2=True,session=session)
    pulse_qstate_multi(sample,cxn,coherent=True,amp=ampl,SZ=0.115,SZ2=0.320,stats=3000,reps=300,delay=200*ns,releaseLen=200*ns,sb_freq=115*MHz,releaseMode=1,release2=True,session=session)


def pituner10(sample, measure=0, iterations=2, npoints=21, stats=1200, save=False, update=True, noisy=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    amp = q['piAmp']
    for _ in xrange(iterations):
        # optimize amplitude

        data = rabihigh10(sample, amplitude=np.linspace(0.6*amp, 1.4*amp, npoints),
                           measure=measure, stats=stats, save=save,collect=True, noisy=noisy)
        amp_fit = np.polyfit(data[:,0], data[:,1], 2)
        amp = -0.5 * amp_fit[1] / amp_fit[0]
        print 'Amplitude: %g' % amp

        #freqtuner with ramsey
        #freq = freqtuner(sample, iterations=1, tEnd=100*ns, timeRes=1*ns, nfftpoints=4000, stats=1200, df=50*MHz,
        #      measure=measure, save=False, plot=False, noisy=noisy)
        #q.f10 = freq
    # save updated values
    if update:
        Q.piAmp = amp
        #Q.f10 = freq
    return amp

def rabihigh10(sample, amplitude=st.r[0.0:2.0:0.05], measureDelay=None, measure=0, stats=1500L,
                name='Rabi-pulse height MQ', save=True, collect=False, noisy=True):
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

def rabihigh21(sample, amplitude=st.r[0.0:2.5:0.05], detuning=0*MHz, measure=0, stats=1500L,
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

def visibility(sample, mpa=st.r[0:2:0.005], stats=300, measure=0, level=1,
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





def datasetMinimum(data, default, llim, rlim, dataset=None):
    coeffs = np.polyfit(data[:,0],data[:,1],2)
    if coeffs[0] <= 0:
        print 'No minimum found, keeping value'
        return default, np.polyval(coeffs, default)
    result = np.clip(-0.5*coeffs[1]/coeffs[0],llim,rlim)
    return result, np.polyval(coeffs, result)

def swap10(sample, swapLen=st.r[0:400:2,ns], swapAmp=st.r[-0.05:0.05:0.002,None], measure=0,measureC=1, stats=600L, single=False,SZ=0.0,
         name='Q10-resonator swap MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    #c = qubits[measureC]
    if single==True:
        name += 'with swapAmp '+str(swapAmp)

    if swapAmp is None:
        swapAmp = q.swapAmp
    elif np.size(swapAmp) is 1:
        swapAmp = float(swapAmp)
    else: #sweep from middle of range working progressively outward.
        swapAmp = [swapAmp[idx] for idx in np.argsort(np.abs(swapAmp))]

    axes = [(swapAmp, 'swap pulse amplitude'), (swapLen, 'swap pulse length')]
    kw = {'SZ': SZ, 'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, currAmp, currLen):
        q.xy = eh.mix(q, eh.piPulseHD(q, 0))
        q.z = env.rect(q['piLen']/2, currLen, currAmp) + eh.measurePulse(q, q['piLen']/2  + currLen)
        q.cz = env.rect(q['piLen']/2, currLen, SZ)
        q['readout'] = True
        return runQubits(server, qubits, stats=stats, probs=[1])

    return sweeps.grid(func, axes, save=save, dataset=dataset, collect=collect, noisy=noisy)



def swap10tuner(sample, swapLen=None, swapLenBND=6*ns, swapAmp=None, swapAmpBND=0.007, paraName='0',SZ=0.0,
                iteration=3, measure=0, stats=1200L,
         name='Q10-resonator swap tuner MQ', save=False, noisy=True, update=True):
    sample, qubits, Qubits= util.loadQubits(sample,write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]

    if swapAmp is None:
        swapAmp = q['swapAmp'+paraName]
    if swapLen is None:
        swapLen = q['swapLen'+paraName]

    for i in range(iteration):
        rf = 2**i
        swapLenOld = swapLen
        swapAmpOld = swapAmp

        print 'Tuning the swap amplitude'
        results = swap10(sample, swapLen=swapLen,
                        swapAmp=np.linspace(swapAmp-swapAmpBND/rf,swapAmp+swapAmpBND/rf,21),SZ=SZ,
                        measure=measure, stats=600L,
                        name='Q10-resonator swap MQ', save=save, collect=True, noisy=noisy)

        new, percent = datasetMinimum(results, swapAmpOld, swapAmpOld-swapAmpBND/rf, swapAmpOld+swapAmpBND/rf)
        swapAmp = new
        print 'New swap amplitude is %g' % swapAmp

        print 'Tuning the swap length'
        results = swap10(sample, swapLen=st.PQlinspace(max([swapLen['ns']*(1-0.2/rf),swapLen['ns']-swapLenBND['ns']]),
                                                       min([swapLen['ns']*(1+0.2/rf),swapLen['ns']+swapLenBND['ns']]),21,ns),
                        swapAmp=swapAmp, measure=measure, stats=600L, SZ=SZ,
                        name='Q10-resonator swap MQ', save=save, collect=True, noisy=noisy)

        new, percent = datasetMinimum(results, swapLenOld['ns'], max([swapLen['ns']*(1-0.2/rf),swapLen['ns']-swapLenBND['ns']]),
                                      min([swapLen['ns']*(1+0.2/rf),swapLen['ns']+swapLenBND['ns']]))
        swapLen = new*ns
        print 'New swap length is %g ns' % swapLen['ns']

    if update:
        Q['swapAmp'+paraName] = swapAmp
        Q['swapAmp'+paraName+'OS'] = swapAmp
        Q['swapLen'+paraName] = swapLen

    return swapLen, swapAmp




def FockScan(sample, n=1, scanLen=st.arangePQ(0,100,1,'ns'), scanOS=0.0, tuneOS=False, probeFlag=False, paraName='0',stats=1500L, measure=0, delay=0*ns,
       name='Fock state swap length scan MQ', save=False, collect=True, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    axes = [(scanLen, 'Swap length adjust'),(scanOS, 'Amplitude overshoot')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, 'res '+paraName+' '+name+' '+str(n)+' Fock state', axes, measure=measure, kw=kw)

    sl = q['swapLen'+paraName+'s']
    print 'Optizmizing n=%g for the swap length = %g ns...' %(n,sl[n-1])
    sa = q['swapAmp'+paraName]

    #if not tuneOS:
       # so = np.array([0.0]*n)
    #else:
    so = q['swapAmp'+paraName+'OSs']

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
            start += sl[n-1]+currLen+delay+5.0*ns
            q.z += eh.measurePulse(q, start)
        else:
            q.z += env.rect(start, sl[n-1], sa, overshoot=so[n-1]+currOS)
            start += sl[n-1]+delay+5.0*ns
            q.z += env.rect(start, currLen, sa)
            start += currLen+delay
            q.z += eh.measurePulse(q, start)

        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)



def FockTuner(sample, ntuned=0,n=1, iteration=3, tuneOS=False, paraName='0',stats=1500L, measure=0, delay=0*ns,
       save=True, collect=True, noisy=True, update=True):
    sample, qubits, Qubits = util.loadQubits(sample,write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]

    swapAmpOSs=[]
    swapLens=[]
    if ('swapAmp'+paraName+'OSs' in q) and ('swapLen'+paraName+'s' in q):
        if ntuned > len(q['swapAmp'+paraName+'OSs']):
            ntuned = len(q['swapAmp'+paraName+'OSs'])
        swapAmpOSs = q['swapAmp'+paraName+'OSs'][0:ntuned]
        swapLens = q['swapLen'+paraName+'s'][0:ntuned]
        print 'Using existing tuneup up to %d' % ntuned
    else:
        ntuned = 0

    print 'Starting tuneup at %d photons' % (ntuned+1)
    if ntuned < 0:
        ntuned = 0




    for i in np.arange(ntuned+1,n+1,1):
        if i == 1:
            swapAmpOSs = [q['swapAmp'+paraName+'OS']]
            swapLens = [q['swapLen'+paraName]]
        else:
            swapAmpOSs += [swapAmpOSs[-1]]
            swapLens += [swapLens[-1]*float(numpy.sqrt(i-1)/numpy.sqrt(i))]
        q['swapLen'+paraName+'s']=swapLens
        q['swapAmp'+paraName+'OSs'] = swapAmpOSs
        for iter in range(iteration):
            rf = 2**iter
            print 'iteration %g...' % iter
            old = q['swapLen'+paraName+'s'][i-1]
            #results = FockScan(sample, n=i, scanLen=st.PQlinspace(-max([0.3*sl['ns']/rf,1]),max([0.3*sl['ns']/rf,1]),21,'ns'),paraName=paraName,stats=stats,measure=measure,probeFlag=False,delay=delay,tuneOS=tuneOS,save=False, collect=collect, noisy=noisy)
            #new, percent = datasetMinimum(results, 0, -max([0.3*sl['ns']/rf,1]), max([0.3*sl['ns']/rf,1]))
            #swapLens[-1] += new
            results = FockScan(sample, n=i, scanLen=st.PQlinspace(old*(-0.4/rf),old*(+0.4/rf),21,'ns'),scanOS=0.0,paraName=paraName,stats=stats,measure=measure,probeFlag=False,delay=delay,tuneOS=tuneOS,save=False, collect=collect, noisy=noisy)
            new, percent = datasetMinimum(results, 0, -old*0.4/rf, +old*0.4/rf)
            swapLens[-1] += new
            print 'new swaptime: %g..' %swapLens[-1]
            q['swapLen'+paraName+'s'] = swapLens

            if tuneOS:
                old = swapAmpOSs[-1]
                results = FockScan(sample, n=i, scanLen=0.0*ns, tuneOS=tuneOS, scanOS=np.linspace(old*(-0.2/rf), old*(+0.2/rf), 21),paraName=paraName,stats=stats,measure=measure,probeFlag=False,delay=delay,save=False, collect=collect, noisy=noisy)
                new, percent = datasetMinimum(results,  0, -old/rf, +old/rf)
                swapAmpOSs[-1] += new
                print 'new OS: %g..' %swapAmpOSs[-1]

            q['swapAmp'+paraName+'OSs'] = swapAmpOSs

        if save:
            FockScan(sample, n=i, scanLen=st.arangePQ(0,100,1,'ns'),
                        paraName=paraName,stats=stats,measure=measure,probeFlag=True,delay=delay,tuneOS=tuneOS,
                        save=save, collect=collect, noisy=noisy)
    if update:
        Q['swapLen'+paraName+'s'] = q['swapLen'+paraName+'s']
        Q['swapAmp'+paraName+'OSs'] = q['swapAmp'+paraName+'OSs']

    return q['swapLen'+paraName+'s']

def resonatorSpectroscopy(sample, freqScan=None, swapTime = None, stats=600L, measure=0,SZ_drive=0.0, SZ_hold=0.0,
                          paraName='0',amplitude=None,sb_freq=-0*MHz, excited = False, name='Resonator spectroscopy',
                          driveType = 'gaussian', save=True, collect=True, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    if swapTime is None: swapTime = q['swapLen'] + paraName
    if freqScan is None: freqScan = st.r[q['readout frequency']['GHz']-0.002:q['readout frequency']['GHz']+0.002:0.00002,GHz]
    if amplitude is not None: q['readout amplitude'] = amplitude
    if excited:
        deps = [('Probability', 'without pi pulse', ''), ('Probability', 'with pi pulse', '')]
    else:
        deps = [('Probability', '|1>', '')]
    axes = [(freqScan, 'Resonator frequency')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, curr):
        t0 = 0

        #open coupler and drive resonator
        sb_freq = curr-q['readout fc']
        q['readout frequency']=curr
        if driveType == 'gaussian':
            q.cz = env.rect(t0-q['piLen']/2,q['piLen'], SZ_drive)
            q.rr = env.mix(env.gaussian(t0, q['piFWHM'], amp = currDri, phase=0), df= sb_freq)
            t0 = q['piFWHM']
        elif driveType == 'flattop':
            q.cz = env.rect(t0,q['piLen'], SZ_drive)
            eh.ResSpectroscopyPulse(q, 0, sb_freq)
            t0  = q['readout length']
        q.z=  env.rect(q.piLen/2, swapTime, q['swapAmp'+paraName])
        if mod==1:
            q.rr = eh.ResSpectroscopyPulse(q, 0, sb_freq)
            q.cz =env.rect(0,q['readout length']+0.0*ns,SZ)
        else:
            q.rr = eh.ResSpectroscopyPulse2(q, 0, sb_freq)
        if mod==1:
            q.z = env.rect(q['readout length']+20*ns, swapTime, q['swapAmp'+paraName])
            q.cz += env.rect(q['readout length']+20*ns, swapTime, SZ0)
            q.z +=eh.measurePulse(q, q['readout length']+30*ns+swapTime)
        else:
            q.z=  env.rect(q.piLen/2+20*ns, swapTime, q['swapAmp'+paraName])
            q.cz=  env.rect(q.piLen/2+20*ns, swapTime, SZ0)
            q.z +=eh.measurePulse(q, q.piLen/2+30*ns+swapTime)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

def resonatorT1(sample, delay=st.arangePQ(-0.01,1,0.01,'us')+st.arangePQ(1,7.5,0.05,'us'),paraName='0',stats=1200L, measure=0,
       name='resonator T1 MQ', save=True, collect=True, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    axes = [(delay, 'Measure pulse delay')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, paraName+' '+name, axes, measure=measure, kw=kw)


    sl = q['swapLen'+paraName]
    sa = q['swapAmp'+paraName]

    def func(server, delay):
        q.xy = eh.mix(q, eh.piPulseHD(q, 0))
        q.z = env.rect(q.piLen/2, sl, sa)
        q.z += env.rect(q.piLen/2+sl+delay, sl, sa)
        q.z += eh.measurePulse(q, q.piLen/2+sl+delay+sl)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

def testResDelay(sample, startTime=st.r[-80:80:0.5,ns], pulseLength=8*ns, amp = 0.2, stats=600L, measure=0, paraName='0',SZ=0.0,
       name='Resonator test delay', save=True, collect=True, noisy=True, plot=True, update=True):
    """A single pi-pulse on one qubit, with other qubits also operated."""
    """This is the test delay between resonator uwave drive and qubit z pulse line"""
    sample, qubits, Qubits = util.loadQubits(sample,write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]



    axes = [(startTime, 'Uwave start time')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, curr):
        start = curr
        #q.rr = eh.mix(q, env.gaussian(100*ns+start-20, pulseLength, amp = amp), freq = 'readout frequency')+eh.mix(q, env.gaussian(100*ns+start+20, pulseLength, amp = -amp), freq = 'readout frequency')
        q['readout length'] = pulseLength
        q['readout amplitude'] = amp
        sb_freq = q['readout frequency'] - q['readout fc']
        q.rr = env.mix(env.gaussian(100*ns+start-20, pulseLength, amp=amp), sb_freq)+env.mix(env.gaussian(100*ns+start+20, pulseLength, amp=-amp), sb_freq)
        q.cz = env.rect(100*ns+start-20-pulseLength,pulseLength*2, SZ)+env.rect(100*ns+start+20-pulseLength,pulseLength*2, SZ)
        q.z = env.rect(100*ns-q['swapLen'+paraName]/2, q['swapLen'+paraName], q['swapAmp'+paraName])+eh.measurePulse(q, q['swapLen'+paraName]/2+200*ns)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    topLen = q['swapLen'+paraName]['ns']*2*0.8
    translength = q['piFWHM'][ns]
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
        Q['timingLagRRUwave'] += fit[0]*ns


def coherent(sample, probeLen=st.r[0:200:2,ns], drive=st.r[0:1:0.05], stats=600L, measure=0,sb_freq=-50*MHz,paraName='0',SZ=0.0,SZ0=None,Len=16*ns,mode=0,
       name='Coherent state', save=True, collect=True, noisy=True):
    """This is a program used for calibrating photon numbers."""
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    #c = qubits[measureC]

    if SZ0 is None: SZ0 = q['close_sz']
    q['readout length']=Len
    q['readout fc'] =  q['readout frequency']- sb_freq
    axes = [(probeLen, 'Measure pulse length'),(drive, 'Resonator uwave drive Amp')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, currLen, currDri):
        q['readout amplitude']=currDri
        if mode==0:
            q.rr = env.mix(env.gaussian(q['piFWHM'], q['piFWHM'], amp = currDri, phase=0), df=0.0)
        elif mode==1:
            q.rr = eh.ResSpectroscopyPulse(q, 0, sb_freq)
        q.cz = env.rect(0, Len, SZ)

        q.z = env.rect(Len, currLen, q['swapAmp'+paraName])
        q.cz += env.rect(Len, currLen, SZ0)

        q.z +=eh.measurePulse(q, Len+currLen+5*ns)

        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

def coherentWithZ(sample, probeLen=st.r[0:200:2,ns], drive=st.r[0:1:0.05], zpa = 0.0, stats=600L, measure=0,sb_freq=-50*MHz,paraName='0',SZ=None,SZ0=None,Len=None,
       name='Coherent state', save=True, collect=True, noisy=True):
    """This is a program used for calibrating photon numbers."""
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    #c = qubits[measureC]
    if Len is not None:
        q['readout length']=Len
    else:
        Len = q['readout length']
    if SZ is None: SZ = 0.0
    if SZ0 is None: SZ0 = q['close_sz']
    q['readout fc'] =  q['readout frequency']- sb_freq
    axes = [(probeLen, 'Measure pulse length'),(drive, 'Resonator uwave drive Amp')]
    kw = {'zpa': zpa, 'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, currLen, currDri):
        q.z = env.rect(0, Len, zpa)
        q['readout amplitude']=currDri
        q.rr = eh.ResSpectroscopyPulse(q, 0, sb_freq)
        q.cz = env.rect(0, Len, SZ)
        q.z += env.rect(Len, currLen, q['swapAmp'+paraName])
        q.cz += env.rect(Len, currLen, SZ0)

        q.z +=eh.measurePulse(q, Len+currLen+5*ns)

        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

def pulsetimefunc(swaptimes, order=2):
    """Determines a function of t which gives the programmed
    length of a swappulse needed to give an effective length of t
    swapfreq is the swapping frequeny, and swaptimes the pulse lengths
    for photon transfers from qubit to resonator with 0, 1, 2, ... photons
    already in the resonator
    """
    swaptimes = numpy.array([s['ns'] for s in swaptimes])
    idealtimes = 1.0/numpy.sqrt(numpy.arange(1,numpy.alen(swaptimes)+1))
    poly = numpy.polyfit(idealtimes,swaptimes, order)
    func = lambda t: numpy.polyval(poly,t)
    plt.plot(idealtimes, swaptimes,'+')
    t = numpy.linspace(0,1,101)
    plt.plot(t,func(t))
    print poly
    return func

def offrestimefunc(sample, ds, dataset=None, session=None,measure=0,measureR=2):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    r = qubits[measureR]

    data = dstools.getDataset(ds, dataset=dataset, session=session)
    detuning = abs(q.f10 - r.fRes0)
    d = q.piLen
    def fitfunc(p,t):
        return p[0] - p[1]*numpy.cos(numpy.polyval(p[2:],t))

    def errfunc(p):
        return  fitfunc(p,data[:,0]) - data[:,1]

    plt.plot(data[:,0],data[:,1],'+')
    p,ok = leastsq(errfunc, [50.0,50.0,2*numpy.pi*detuning,0.0])
    #if the fit flipped the sign of p[1] we have to shift the phase by pi:
    p[3] = (p[3] + numpy.pi*(p[1]<0)) % (2*numpy.pi)
    p[1] = abs(p[1])
    plt.plot(data[:,0],fitfunc(p,data[:,0]))
    p = -p[2:]*numpy.sign(detuning)*numpy.sign(p[2])
    if abs((-p[0]/(2*numpy.pi)-detuning)/detuning)>0.05:
        raise Exception('Discrepancy between detuning in qubit definition (%g MHz) and fitted detuning (%g MHz) too big. Cowardly refusing to go on.' % (detuning*1000,p[0]*500/numpy.pi))
    return lambda zpulses: d + ((numpy.sign(p[0])*(zpulses - numpy.polyval(p,d))) % (2*numpy.pi))/abs(p[0])

def dephasing(sample, swapLength,
              detuningLength=st.r[10:20:0.1,ns], stats=300, measure=0, paraName='0',tuneOS=False,
              save=True, collect=False,noisy=True,
              name='Detuning time calibration'):
    """pi-pulse on qubit, pi/2 swap with resonator, wait for detuningLength,
    pi/2 swap with resonator. Measure qubit. If the phase accumulated between
    resonator and qubit while the the qubit is detuned is a multiple of 2pi,
    the photon will be transferred to the resonator."""

    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    #r = qubits[measureR]

    if tuneOS:
        os = q['swapAmp'+paraName+'OS']
    else:
        os = 0.0


    axes = [(detuningLength, 'pi/2 swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)
    def func(server, currLen):
        start = 0
        q.xy = eh.mix(q, eh.piPulseHD(q, start))
        start += q.piLen['ns']/2
        q.z = env.rect(start, swapLength, q['swapAmp'+paraName],overshoot=os)
        start += swapLength+currLen
        q.z += env.rect(start, swapLength, q['swapAmp'+paraName],overshoot=os)
        start += swapLength+10
        q.z += eh.measurePulse(q, start)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)


def resdrivephase(sample, tf,points=400, stats=1500, unitAmpl=0.3, measure=0, measureR=2,paraName='0', tuneOS=False,
       name='resonator drive phase', save=True, collect=True, noisy=True):

    sample, qubits, Qubits = util.loadQubits(sample,write_access=True)
    q = qubits[measure]
    r = qubits[measureR]
    R = Qubits[measureR]

    angle = np.linspace(0,2*np.pi, points, endpoint=False)
    displacement=unitAmpl*np.exp(1j*angle)
    if tuneOS:
        os = q['swapAmp'+paraName+'OS']
    else:
        os = 0.0

    # kw = {'stats': stats}
    # dataset = sweeps.prepDataset(sample, name, axes=[('displacement','re'),('displacement','im')], measure=measure, kw=kw)

    # def func(server, curr):
        # start = 0
        # q.xy = eh.mix(q, env.gaussian(start, q.piFWHM, amp = q.piAmp/2), freq = 'f10')
        # start += q.piLen/2
        # q.z = env.rect(start, q['swapLen'+paraName], q['swapAmp'+paraName],overshoot = os)
        # start += q['swapLen'+paraName]+r.piLen/2
        # r.xy = eh.mix(r, env.gaussian(start,r.piFWHM, amp = np.conjugate(curr*r.drivePhase)), freq = 'fRes0')
        # start += r.piLen/2
        # q.z += env.rect(start, q['swapLen'+paraName], q['swapAmp'+paraName],overshoot = os)
        # start += q['swapLen'+paraName]
        # q.z += eh.measurePulse(q, start)
        # q['readout'] = True
        # data = yield runQubits(server, qubits, stats, probs=[1])
        # data = np.hstack(([curr.real, curr.imag], data))
        # returnValue(data)

    # result = sweeps.run(func, displacement, dataset=save and dataset, noisy=noisy)
    result = arbitraryWigner(sample, np.array([tf(1.0)])*ns, pulseAmplitudes=[0.5*q.piAmp],
              probePulseLength = st.r[tf(1.0):tf(1.0):1,ns], alpha=displacement,
              stats=stats, save=False, collect=True, noisy=False, name='Resonator drive phase')
    result = result[:, [0,3]]
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
    a = r.drivePhase*np.exp(1.0j*p[2])
    print 'Resonator drive Phase correction: %g' % p[2]
    R.drivePhase = a/abs(a)
    return



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
    mpa1=multiqubit.find_mpa(sample, target=0.5, stats=stats, measure=measure, update=False)
    if measure == 0:
        sample.q0.biasOperate -= FBchange
    elif measure == 1:
        sample.q1.biasOperate -= FBchange
    mpa2=multiqubit.find_mpa(sample, target=0.5, stats=stats, measure=measure, update=False)
    ratio = FBchange/(mpa1-mpa2)
    print 'unit measure pulse amplitude corresponds to %g mV flux bias.' % ratio['mV']
    Q.calUnitMPA2FBmV = ratio
    return ratio





def testICCdelaySZ(sample, t0=st.r[-30:30:0.25,ns],measure=0,measureC=1, measureR=2,paraName='0',SZ=st.r[0.8:-0.8:-0.05],
               stats=1200, update=True,
              save=True, name='Test Delay', plot=True, noisy=True):
    """ This is to calibrate the test delay between qubit z pulse and adjustable coupler z pulse"""
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    #c = qubits[measureC]
    #C = Qubits[measureC]
    #r = qubits[measureR]

    axes = [(SZ,'squid z'),(t0, 'Detuning pulse center')]
    kw = {
        'stats': stats,
    }
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, currSZ,t0):
        q.xy = eh.mix(q, eh.piPulse(q, -q['piLen']/2-q['swapLen'+paraName]/2))
        q.cz = env.rect(t0-q['swapLen'+paraName]*1.0/2,q['swapLen'+paraName]*1.0,currSZ)
        q.z = env.rect(-q['swapLen'+paraName]/2, q['swapLen'+paraName], q['swapAmp'+paraName]) + eh.measurePulse(q, q['swapLen'+paraName]*2)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    zpl = q['swapLen'+paraName][ns]*0.8
    translength = 0.8*q['piFWHM'][ns]
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
    print 'ICC squid z lag:', fit[0]
    if update:
        print 'ICC squid z corrected by %g ns' % fit[0]
        Q['timingLagCZ'] += fit[0]*ns


def PitunerTable(sample,SB=st.r[-100:50:5,mV],measure=0,measureC=1):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    #c = qubits[measureC]
    table1 = range(np.size(SB))
    table2 = range(np.size(SB))
    table3 = range(np.size(SB))
    table4 = range(np.size(SB))
    i=0
    for squidtune in SB:
        q['couplerfluxBias']=squidtune
        pituner10(sample)
        swap10tuner(sample)
        table1[i]=q['swapAmp0']
        table2[i]=q['swapLen0']
        table3[i]=q['piAmp']
        table4[i]=q['f10']
        i=i+1
    return SB, table1,table2,table3,table4


def ResT1WithSZ(sample, delay=st.arangePQ(0,1,0.01,'us')+st.arangePQ(1,7.5,0.05,'us'),SZ=st.r[0:0.2:0.02,None],SZ0=0.0,paraName='0',stats=600L, measure=0,
       measureC=1,measureR=2,name='resonator T1 with dynamic tuning', save=True, collect=True, noisy=True):
    """Measure resonator T1 with dynamic z pulse tuning on the adjustable coupler"""
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    #c= qubits[measureC]
    #r = qubits[measureR]


    axes = [(SZ,'squid Z bias'),(delay, 'Measure pulse delay')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)


    sl = q['swapLen'+paraName]
    sa = q['swapAmp'+paraName]


    def func(server, squidz,delay):
        q.xy = eh.mix(q, eh.piPulseHD(q, 0))
        q.z = env.rect(q.piLen/2, sl, sa)
        q.cz = env.rect(q.piLen/2, sl, SZ0)
        q.cz += env.rect(q.piLen/2+sl,delay,squidz)
        #r.z =env.rect(q.piLen/2+sl,delay,RZ)
        q.z += env.rect(q.piLen/2+sl+delay, sl, sa)
        q.cz += env.rect(q.piLen/2+sl+delay, sl, SZ0)
        q.z += eh.measurePulse(q, q.piLen/2+sl+delay+sl)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)


def rfreqSZ(sample, freqScan=st.r[6.52:6.586:0.0005,GHz], swapTime=300*ns, SZ=st.r[-0.9:0.9:0.0025,None],RZ=0.0,stats=600L, measure=0,paraName='0',amplitude=0.04,
       name='Resonator spectroscopy', save=True, collect=True, noisy=True):
    """2-D spectroscopy on adjustable resonator(drive&zpulse)"""
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]



    if freqScan is None:
        freqScan = st.r[q['readout frequency']['GHz']-0.002:q['readout frequency']['GHz']+0.002:0.00002,GHz]
    axes = [(SZ,'squid z pulse amp'),(freqScan, 'Resonator frequency')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)


    def func(server, currSZ,curr):
        sb_freq = curr-q['readout fc']
        q['readout amplitude'] = amplitude
        q['readout length'] = 2000*ns
        q.rr = eh.ResSpectroscopyPulse(q, 0, sb_freq)
        q.cz =env.rect(0,q['readout length'],currSZ)
        #r.z =env.rect(0,r.spectroscopyLen,RZ)
        q.z = env.rect(q['readout length']+20*ns, swapTime, q['swapAmp'+paraName])+eh.measurePulse(q, q['readout length']+30*ns+swapTime)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)



def qfreqSZ(sample, freqScan=st.r[6.845:6.882:0.001,GHz], SZ=st.r[-0.3:0.3:0.0025,None],stats=1200L, measure=0,measureC=1,measureR=2,paraName='0',amplitude=0.04,
       name='Resonator spectroscopy', save=True, collect=True, noisy=True):
    """X-talk measurement qubit and adjustable coupler z pulse"""
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    c = qubits[measureC]



    if freqScan is None:
        f = st.nearest(q['f10'][GHz], 0.001)
        freq = st.r[f-0.04:f+0.04:0.001, GHz]
    axes = [(SZ,'squid z pulse amp'),(freqScan, 'qubit frequency')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)


    def func(server, currSZ,currFreq):
        sb_freq = currFreq-r.fc
        q['spectroscopyAmp'] = amplitude
        q.xy = eh.spectroscopyPulse(q, 0, sb_freq)
        q.z  = eh.measurePulse(q, q['spectroscopyLen'])
        c.z =env.rect(0,q.spectroscopyLen,currSZ)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)


def dyTuneResT1(sample, delay=st.arangePQ(-0.01,1,0.01,'us')+st.arangePQ(1,7.5,0.05,'us'),delay2=st.r[0:1000:50,'ns'],paraName='0',stats=1200L, measure=0,measureC=1,SZ1=0.0,SZ2=0.05,
       name='resonator T1 with dynamic tuning', save=True, collect=True, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    c= qubits[measureC]

    axes = [(delay2,'Delay before dynamic tuning'),(delay, 'Measure pulse delay')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)


    sl = q['swapLen'+paraName]
    sa = q['swapAmp'+paraName]

    def func(server, delay2,delay):
        q.xy = eh.mix(q, eh.piPulseHD(q, 0))
        q.z = env.rect(q.piLen/2, sl, sa)
        q.z += env.rect(q.piLen/2+sl+delay, sl, sa)
        q.z += eh.measurePulse(q, q.piLen/2+sl+delay+sl)
        d1=delay['ns']
        d2=delay2['ns']
        if d1>(d2+2):
            c.z =env.rect(q.piLen/2+sl,delay2,SZ1)+env.rect(q.piLen/2+sl+delay2,delay-delay2,SZ2)
        else:
            c.z =env.rect(q.piLen/2+sl,delay,SZ1)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

def FockScanWithSZ(sample, n=1, scanLen=st.arangePQ(0,200,1,'ns'), scanOS=0.0, tuneOS=False, probeFlag=False, paraName='0',stats=1500L, measure=0,measureC=1, delay=0*ns,SZ=np.arange(0,0.7,0.01),criLen=80*ns,
       name='Fock state swap length scan MQ', save=True, collect=True, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    c= qubits[measureC]

    axes = [(SZ,'SquidZ amp'),(scanLen, 'Swap length adjust'),(scanOS, 'Amplitude overshoot')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, 'res '+paraName+' '+name+' '+str(n)+' Fock state', axes, measure=measure, kw=kw)

    sl = q['swapLen'+paraName+'s']
    print 'Optizmizing n=%g for the swap length = %g ns...' %(n,sl[n-1])
    sa = q['swapAmp'+paraName]

    if not tuneOS:
        so = np.array([0.0]*n)
    else:
        so = q['swapAmp'+paraName+'OSs']

    def func(server,currSZ, currLen, currOS):
        q.xy = env.NOTHING
        q.z = env.NOTHING
        start = -q.piLen/2
        for i in range(n-1):
            q.xy += eh.mix(q, eh.piPulseHD(q, start+q.piLen/2))
            start += q.piLen
            q.z += env.rect(start, sl[i], sa, overshoot=so[i])
            start += sl[i]
        q.xy += eh.mix(q, eh.piPulseHD(q, start+q.piLen/2))
        start += q.piLen
        if not probeFlag:
            q.z += env.rect(start, sl[n-1]+currLen, sa, overshoot=so[n-1]+currOS)
            if currLen>criLen:
                c.z =env.rect(start + sl[n-1]+criLen,currLen-criLen,currSZ)
            else:
                c.z =env.rect(start + sl[n-1]+criLen,currLen-criLen,0)
            start += sl[n-1]+currLen
            q.z += eh.measurePulse(q, start)
        else:
            q.z += env.rect(start, sl[n-1], sa, overshoot=so[n-1]+currOS)
            start += sl[n-1]
            q.z += env.rect(start, currLen, sa)
            start += currLen
            q.z += eh.measurePulse(q, start)

        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)

 ##############################################
 # QND measurement with resonator reflection  #
 ##############################################

from pyle.plotting import datareader as dr
from pyle.plotting import dstools as ds
from numpy import *
from pylab import *
import functools

def countcalls(fn):
    "decorator function count function calls "

    @functools.wraps(fn)
    def wrapped(*args):
        wrapped.ncalls +=1
        return fn(*args)

    wrapped.ncalls = 0
    return wrapped

def qstate_analysis(start=1,num=10,session=['','Yi','QND','wfr20110330','r4c3','110827n']):
    data = ds.getDataset(dr,dataset=start,session=session)
    index = len(data[:,0])
    data_stat = np.empty([num,index],dtype=complex)
    data_ave = np.empty([index,5])
    data_ave[:,0] = data[:,0]

    for i in range (num):
        data = ds.getDataset(dr,dataset=start+i,session=session)
        freq, phase, amp = data.T
        ref = amp*np.exp(1j*phase)
        data_stat[i,:]=ref
        #x0,y0,R = circle_cal(num=start+i,session=session,x_m=0,y_m=0)

    data_ave[:,1] = np.abs(data_stat.mean(axis=0))
    data_ave[:,2] = np.angle(data_stat.mean(axis=0))
    data_ave[:,3] = np.sqrt(data_stat.var(axis=0))
    data_ave[:,4] = data_ave[:,1]/data_ave[:,3]
    return data_ave



def circle_cal(num=1,session=['','Yi','QND','wfr20110330','r4c3','110827n'],data = None, x_m=0,y_m=0):

    if data is not None:
        freq, phase, amp = data.T
        ref = amp*np.exp(1j*phase)
        index = len(ref)
        x = ref.real
        y = ref.imag

    elif session != None:
        data = ds.getDataset(dr,dataset=num,session=session)
        index = len(data[:,0])
        x=asarray([data[i,2]*cos(data[i,1]) for i in range(index)])
        y=asarray([data[i,2]*sin(data[i,1]) for i in range(index)])

    def calc_R(xc, yc):
        """ http://www.scipy.org/Cookbook/Least_Squares_Circle         :calculate the distance of each 2D points from the center (xc, yc) """
        return sqrt((x-xc)**2 + (y-yc)**2)

    @countcalls
    def f_2(c):
        """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = x_m, y_m
    center_2, ier = leastsq(f_2, center_estimate)

    xc_2, yc_2 = center_2
    Ri_2       = calc_R(xc_2, yc_2)
    R_2        = Ri_2.mean()
    residu_2   = sum((Ri_2 - R_2)**2)
    residu2_2  = sum((Ri_2**2-R_2**2)**2)
    ncalls_2   = f_2.ncalls

    x_new = asarray([x[i]-xc_2 for i in range(index)])
    y_new = asarray([y[i]-yc_2 for i in range(index)])
    phase_new = asarray([atan2(y_new[i],x_new[i]) for i in range(index)])
    for i in range(index-1):
        if phase_new[i+1]>phase_new[i]:
            multi = (phase_new[i+1]-phase_new[i]+0.5)//np.pi
            phase_new[i+1]-=np.pi*multi
    figure(1)
    plot(data[:,0],phase_new[:],'-')
    figure(2)
    plot(x_new[:],y_new[:],'o-')

    def fitfunc(freq,p):
        return p[0]+p[1]*freq
    def errfunc(p):
        return phase_new-fitfunc(data[:,0],p)
    p,ok = leastsq(errfunc, [0.0,300.0])
    figure(1)
    plot(data[:,0],p[0]+p[1]*data[:,0])


    return xc_2, yc_2, R_2, p


def s_scanning(sample, freq=st.r[6.562:6.59:0.00005, GHz], photonNum=1, zpa=0.0, SZ=None, pulseLen=None, sb_freq = -50*MHz, measure=0,
               stats=150, mode=0,tail=2800*ns, cali_update = False, amp_rescale = False,phase_rescale=False,
               save=True, name='S parameter scanning', collect=True, noisy=True):
    """ This is the original s_scaning with sz"""
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    if pulseLen is None: pulseLen = q['readout length']
    #if amp is None: amp = q['readout amplitude']
    if SZ is None: SZ = 0.0
    if cali_update: SZ = q['cali_sz']
    if cali_update: name = 'Cali ' + name


    axes = [(freq, 'Frequency'),(photonNum,'photon number')]
    deps = [('Phase', 'S11 for %s'%q.__name__, rad) for q in qubits]+ [('Amplitude','S11 for %s'%q.__name__,'') for q in qubits]
    kw = {'stats': stats, 'zpa': zpa}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, freqCurr,ampCurr):
        amp =[]
        phase =[]
        ampCurr = np.sqrt(photonNum)/q['photonNum2Amp']
        q['biasReadout'] = q['biasOperate']
        q['readout amplitude'] = ampCurr
        q['adc mode'] = 'demodulate'
        q['readout frequency']=freqCurr
        q['readout length']=pulseLen
        q['readout fc'] = q['readout frequency'] - sb_freq
        q['adc demod frequency'] = q['readout frequency']-q['readout fc']
        q['readout'] = True
        q['readoutType'] = 'resonator'
        q['fc'] = 7.5*GHz
        if mode==0:
            q.rr = eh.ResSpectroscopyPulse(q, 0*ns, sb_freq)
        elif mode==1:
            q.rr = eh.ResSpectroscopyPulse2(q, 0*ns, sb_freq)

        q.z = env.rect(0, q['readout length']+tail, zpa)
        q.cz =env.rect(0,q['readout length']+tail,SZ)


        #q.xy = eh.mix(q, eh.piPulseHD(q, -q['piLen']/2))
        if noisy: print freqCurr, ampCurr

        data = yield FutureList([runQubits(server, qubits, stats, raw=True)])
        I_mean = np.mean(data[0][0][0])
        Q_mean = np.mean(data[0][0][1])
        ref = I_mean + 1j*Q_mean - (Q['Cal_x0']+1j*Q['Cal_y0'])


        phase_mod = mod(freqCurr['GHz']*q['Cal_p1'],2*pi)
        if freqCurr==freq[0]:
            if amp_rescale:
                Q['adc amp offset'] = abs(ref)
            else:
                Q['adc amp offset'] = 1.0
            if phase_rescale:
                Q['adc phase offset'] = np.angle(ref) - phase_mod
        if cali_update:
            amp.append(abs(I_mean+1j*Q_mean))
            phase.append(atan2(Q_mean,I_mean))
        else:
            amp.append(abs(ref/Q['adc amp offset']))
            phase.append(mod(np.angle(ref)-phase_mod-Q['adc phase offset'],2*pi))

        returnValue(phase+amp)

    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=False)

    if cali_update:
        x0,y0,r0,p=circle_cal(data=data)
        Q['Cal_x0']=x0
        Q['Cal_y0']=y0
        Q['Cal_p0']=p[0]
        Q['Cal_p1']=p[1]
    if collect:
        return data


def autocoherentscan(sample, zpas, measure=0):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    for zpa in zpas:
        data = s_scanning(sample, freq = st.r[6.5655:6.5625:1e-5, GHz], photonNum=1, phase_rescale=True, amp_rescale=True, zpa=zpa, stats=9000 )
        data = data[data[:, 1].argsort()]
        freq, phase, amp = data.T
        phase_interp = interpolate.interp1d(phase, freq)
        f0 = phase_interp(np.pi)
        Q['readout frequency'] = f0
        coherentWithZ(sample, zpa=zpa, drive = st.r[0:0.005:0.0002])


def autoQubitscan(sample, zpas):
    for zpa in zpas:
        data = mq.spectroscopy(sample, freq = st.r[6.6:6.9:5e-4, GHz], Zpa = zpa, collect = True, update=False)
        f, p1 = data.T
        f0 = f[np.argmax(p1)]
        numbers = [0.1,0.5,1,2,5]
        for num in numbers:
            QubitSpecWithPhotons(sample, freq = f0*GHz, Resfreq = st.r[6.5615:6.564:1e-5, GHz],zpa=zpa, stats=9000,
                                 qLen=5000*ns, resLen=5000*ns, photonNum=num, squid=False)

def QubitSpecWithPhotons(sample, freq=None, Resfreq = None, photonNum = 1, tail = 1000*ns, zpa = 0.0, squid = True, stats=300L, measure=0, amp  =None, sb_freq=-200*MHz,
                         qLen = None, resLen = None, zLen = None, qStart = 0*ns, resStart = 0*ns, res_sb_freq = -50*MHz, save=True, cali = True,  SZ=0.0,
                         name = 'Spectroscopy with photons', collect = False, noisy = True):
    """Qubit spectroscopy with photons in the resonator"""
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    if freq is None:
        f = st.nearest(q['f10'][GHz], 0.001)
        freq = st.r[f-0.04:f+0.04:0.001, GHz]
    if Resfreq is None:
        Resf = st.nearest(q['readout frequency'][GHz], 1e-5)
        Resfreq = st.r[Resf-0.001:Resf+0.001:1e-5, GHz]
    if qLen is None:
        qLen = q['spectroscopyLen']
    if resLen is None:
        resLen = q['readout length']
    if zLen is None:
        zLen = resLen+tail
    if amp is None:
        amp = q['spectroscopyAmp']


    axes = [(Resfreq, 'resonator frequency'), (freq, 'qubit frequency'), (photonNum, 'resontor photon number')]
    if squid:
        name = name + '(Squid)'
        deps = [('Probability', '|1>', '')]
    else:
        name = name + '(Resonator)'
        deps = [('S11 amp','no pi pulse',''), ('S11 phase', 'no pi pulse', 'rad'),('S11 amp','pi pulse','rad'),
                ('S11 phase', 'pi pulse', 'rad'), ('Signal to noise ratio', '', '')]
    kw = {'stats':stats, 'qubit pulse length': qLen, 'resonator pulse length': resLen, 'spectroscopy amp': amp, 'zpa': zpa}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)


    def func(server, ResfreqCurr, freqCurr, num):
        """Global settings"""
        q['readout'] = True
        q['readout amplitude']= np.sqrt(num)/q['photonNum2Amp']
        q['readout length'] = resLen
        q['readout frequency'] = ResfreqCurr
        q['readout fc'] = q['readout frequency'] - res_sb_freq
        q['adc demod frequency'] = q['readout frequency']-q['readout fc']
        q.rr = eh.ResSpectroscopyPulse(q, resStart, res_sb_freq)
        q['fc'] = freqCurr - sb_freq
        q['spectroscopyAmp'] = amp
        q['spectroscopyLen'] = qLen
        q.z = env.rect(qStart, zLen, zpa)
        q.xy = eh.spectroscopyPulse(q, qStart, sb_freq)
        q.cz =env.rect(qStart,q['readout length'],SZ)



        """squid readout qubit spectroscopy"""
        result = []
        if squid:
            q.z += eh.measurePulse(q, q['readout length']+q['releaseLen'])
            q.cz += env.rect(q['readout length'], q['releaseLen'], q['cali_sz'])
            q['readoutType'] = 'squid'
            prob = yield FutureList([runQubits(server, qubits, stats, probs=[1])])
            result.append(prob[0])
        else:
            q['readoutType'] = 'resonator'
            phase_mod = mod(ResfreqCurr['GHz']*q['Cal_p1'],2*pi)

            """excited-state qubit measurement"""
            data1 = yield FutureList([runQubits(server, qubits, stats, raw=True)])
            I1 = np.asarray(data1[0][0][0])
            Q1 = np.asarray(data1[0][0][1])
            if cali: ref1 = (I1 + 1j*Q1 - (q['Cal_x0']+1j*q['Cal_y0']))*exp(-1j*(phase_mod+q['adc phase offset']))
            else: ref1 = I1 + 1j*Q1
            sig1 = ref1.mean()

            """ground-state qubit measure"""
            q['readout'] = True
            q['readoutType'] = 'resonator'
            q['biasReadout'] = q['biasOperate']
            q['readout amplitude']= np.sqrt(num)/q['photonNum2Amp']
            q['readout length'] = resLen
            q['readout frequency'] = ResfreqCurr
            q['readout fc'] = q['readout frequency'] - res_sb_freq
            q['adc demod frequency'] = q['readout frequency']-q['readout fc']
            q.rr = eh.ResSpectroscopyPulse(q, resStart, res_sb_freq)
            q['spectroscopyAmp'] = amp
            q['spectroscopyLen'] = qLen
            q['fc'] = 7.5*GHz
            q.xy = env.NOTHING
            data0 = yield FutureList([runQubits(server, qubits, stats, raw=True)])
            I0 = np.asarray(data0[0][0][0])
            Q0 = np.asarray(data0[0][0][1])
            ref0 = (I0 + 1j*Q0 - (q['Cal_x0']+1j*q['Cal_y0']))*exp(-1j*(phase_mod+q['adc phase offset']))
            sig0 = ref0.mean()

            sg_ratio = abs(sig0-sig1)/(ref0.std()+ref1.std())

            result.append(abs(sig0))
            result.append(mod(np.angle(sig0), 2*pi))
            result.append(abs(sig1))
            result.append(mod(np.angle(sig1), 2*pi))
            result.append(sg_ratio)
        returnValue(result)

    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if collect:
        return data



def spectroscopy_two_stateWithPhotons(sample, freq=None, stats=300L, measure=0, sb_freq=0*GHz, detunings=None, uwave_amps=None,
                           photonNum=1, Resfreq = None, pulseLen=None,save=True, name='Two-state spectroscopy with ', SZ=0.0,
                           collect=False, noisy=True, update=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    q['readout'] = True
    if Resfreq is None: Resfreq = q['readout frequency']
    if pulseLen is None: pulseLen = q['readout length']
    if freq is None:
        f = st.nearest(q['f10'][GHz], 0.001)
        freq = st.r[f-0.20:f+0.04:0.002, GHz]
    if uwave_amps is None:
        uwave_amps = q['spectroscopyAmp'], q['spectroscopyAmp']*10, q['spectroscopyAmp']*15
    if detunings is None:
        zpas = [0.0] * len(qubits)
    else:
        zpas = []
        for i, (q, df) in enumerate(zip(qubits, detunings)):
            print 'qubit %d will be detuned by %s' % (i, df)
            zpafunc = get_zpa_func(q)
            zpa = zpafunc(q['f10'] + df)
            zpas.append(zpa)

    name = name  + str(photonNum) + ' photons'
    axes = [(freq, 'Frequency')]
    deps = [('Probability', '|1>, uwa=%g' % amp, '') for amp in uwave_amps]
    kw = {
        'stats': stats,
        'photon number': photonNum,
        'sideband freq': sb_freq
    }
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, f):
        reqs = []
        for amp in uwave_amps:
            for i, (q, zpa) in enumerate(zip(qubits, zpas)):
                q['fc'] = f - sb_freq
                if zpa:
                    q.z = env.rect(-100, qubits[measure]['spectroscopyLen'] + 100, zpa)
                else:
                    q.z = env.NOTHING
                if i == measure:
                    q['readoutType'] = 'squid'
                    q['readout'] = True

                    q['spectroscopyLen'] = pulseLen
                    q['spectroscopyAmp'] = amp
                    q.xy = eh.spectroscopyPulse(q, 0, sb_freq)
                    q['readout amplitude']= np.sqrt(photonNum)/q['photonNum2Amp']
                    q['readout length'] = pulseLen
                    q['readout frequency'] = Resfreq
                    q['adc demod frequency'] = q['readout frequency']-q['readout fc']
                    q.rr = eh.ResSpectroscopyPulse(q, 0*ns, q['readout frequency']-q['readout fc'])
                    q.z += eh.measurePulse(q, q['spectroscopyLen']+q['releaseLen'])
                    q.cz = env.rect(0*ns,q['readout length'],SZ)
                    q.cz += env.rect(q['readout length'], q['releaseLen'], q['cali_sz'])
            reqs.append(runQubits(server, qubits, stats, probs=[1]))
        probs = yield FutureList(reqs)
        returnValue([p[0] for p in probs])
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if update:
        adjust.adjust_frequency_02(Q, data)
    if collect:
        return data


def rabihighWithPhotons(sample, amplitude=st.r[0.0:1.5:0.05], driveLen=5*us, Resfreq = None, photonNum=1, measure=0, stats=1500L,
                        state=1, measstate=None, name='Rabi-pulse height ', save=True, collect=False, noisy=True, update=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    ml.setMultiKeys(q,max(state,measstate))
    name='|'+str(state)+'> '+name

    if Resfreq is None: Resfreq = q['readout frequency']
    if amplitude is None: amplitude = ml.getMultiLevels(q,'piAmp',state)
    if measstate is None: measstate=state

    name = name  + str(photonNum) + ' photons'
    axes = [(amplitude, 'pulse height')]
    deps = [('Probability', '|'+str(measstate)+'>', '')]
    kw = {'stats': stats,
          'photon number': photonNum}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, amp):
        ml.setMultiLevels(q,'piAmp',amp,state)
        q.xy = eh.boostState(q, driveLen, state-1) + eh.mix(q, eh.piPulseHD(q, driveLen+(state-1)*q['piLen'], state=state), state=state)
        q['readout amplitude']= np.sqrt([num])/q['photonNum2Amp']
        q['readout frequency'] = Resfreq
        q['readout length'] = driveLen+(state-0.5)*q['piLen']
        q['adc demod frequency'] = q['readout frequency']-q['readout fc']
        q.rr = eh.ResSpectroscopyPulse(q, 0*ns, q['readout frequency']-q['readout fc'])
        q.z = eh.measurePulse(q, q['readout length']+q['releaseLen'], state=measstate)
        q.cz = env.rect(q['readout length'], q['releaseLen'], q['cali_sz'])
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, collect=(collect or update), noisy=noisy)

    if update:
        adjust.adjust_rabihigh(Q, data, state=state)
    if collect:
        return data




def RamseyWithPhotons(sample, measure=0, delay=st.r[0:200:1,ns], driveLen = 5*us, photonNum=1, Resfreq=None,
                      fringeFreq = 50*MHz, stats=600L, power = None, name='Ramsey', save = True, noisy=True,
           collect = False, randomize=False, averages = 1, tomo=True, state=1,SZ=0.0,
           plot=True,update=True):
    """Ramsey sequence on one qubit. Can be single phase or 4-axis tomo, and
    can have randomized time axis and/or averaging over the time axis multiple
    times

    PARAMETERS
    sample: object defining qubits to measure, loaded from registry
    measure - scalar: number of qubit to measure. Only one qubit allowed.
    delay - iterable: time axis
    fringeFreq - value [Mhz]: Desired frequency of Ramsey fringes
    stats - scalar: number of times a point will be measured per iteration over
            the time axis. That the actual number of times a point will be
            measured is stats*averages
    name - string: Name of dataset.
    save - bool: Whether or not to save data to the datavault
    noisy - bool: Whether or not to print out probabilities while the scan runs
    collect - bool: Whether or not to return data to the local scope.
    randomize - bool: Whether or not to randomize the time axis.
    averages - scalar: Number of times to iterate over the time axis.
    tomo - bool: Set True if you want to measure all four tomo axes, False if
                 you only want the X axis (normal Ramsey fringes).
    """
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]
    q['readout'] = True

    if Resfreq is None: Resfreq = q['readout frequency']
    if (update or plot) and not(q.has_key('calT1')):
        raise Exception("Cannot plot and fit until you do a T1 scan")
    ml.setMultiKeys(q,state) #Creates q['multiKeys']
    #Randomize time axis if you want
    if randomize:
        delay = st.shuffle(delay)

    #Generator that produces time delays. Iterates over the list of delays as many times as specified by averages.
    delay_gen = st.averagedScan(delay,averages,noisy=noisy)

    axes = [(delay_gen(), 'Delay')]
    #If you want XY state tomography then we use all four pi/2 pulse phases
    if tomo:
        deps = [('Probability', '+X', ''),('Probability', '+Y', ''),
                ('Probability', '-X', ''),('Probability', '-Y', '')]
        tomoPhases = {'+X': 0.0, '+Y': 0.25, '-X': -0.5, '-Y': -0.25} #[+X, +Y, -X, -Y] in CYCLES
    #Otherwise we only do a final pi/2 pulse about the +X axis.
    else:
        deps = [('Probability', '|'+str(state)+'>', '')]
        tomoPhases = {'+X': 0.0}

    if Resfreq is None: Resfreq = q['readout frequency']
    name = name  + str(photonNum) + ' photons'
    kw = {'averages': averages,
           'stats': stats,
           'fringeFrequency': fringeFreq,
           'photon number': photonNum}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)
    timeUpstate = (state-1)*q['piLen']+driveLen
    dt = q['piFWHM']

    #Pump pulse is at time=0 (after the time to reach the initial state) with phase=0
    pump = eh.piHalfPulse(q, timeUpstate, phase=0.0, state=state)
    #Probe is at variable time with variable phase
    def probe(time, tomoPhase):
        return eh.piHalfPulse(q, timeUpstate+time, phase = 2*np.pi*(fringeFreq['GHz']*time['ns']+tomoPhases[tomoPhase]), state=state)

    def func(server, delay):
        reqs = []
        for tomoPhase in tomoPhases.keys():
            q.xy = eh.boostState(q, driveLen, state-1) + eh.mix(q, pump + probe(delay, tomoPhase = tomoPhase), state=state)
            q['readout amplitude']= np.sqrt([photonNum])/q['photonNum2Amp']
            q['readout frequency'] = Resfreq
            q['readout length'] = timeUpstate+dt+delay
            q['adc demod frequency'] = q['readout frequency']-q['readout fc']
            q.rr = eh.ResSpectroscopyPulse(q, 0*ns, q['readout frequency']-q['readout fc'])
            q.z = eh.measurePulse(q, q['readout length']+q['releaseLen'], state=state)
            q.cz = env.rect(0*ns,q['readout length'],SZ)
            q.cz += env.rect(q['readout length'], q['releaseLen'], q['cali_sz'])
            reqs.append(runQubits(server, qubits, stats, probs=[1]))
        probs = yield FutureList(reqs)
        data = [p[0] for p in probs]
        returnValue(data)
    #Run the scan and save data
    data = sweeps.grid(func, axes, dataset = save and dataset, collect=collect, noisy=noisy)
    #Fit the data. Plot if desired. Update the registered value of T2
    #Fit. Must first retrieve dataset from datavault
    if plot or update:
        T1 = q['calT1']
        with labrad.connect() as cxn:
            dv = cxn.data_vault
            dataset = dstools.getOneDeviceDataset(dv, datasetNumber=-1,session=sample._dir,
                                                           deviceName=None, averaged=averages>1)
        if tomo:
            result = fna.ramseyTomo_noLog(dataset, T1=T1, timeRange=(10,delay[-1]),
                                                                 plot=plot)
        else:
            raise Exception('Cannot do plotting or fitting without tomo. It would be easy to fix this')
        if update:
            Q['calT2']=result['T2']
    return result['T2']


def T1WithPhotons(sample, delay=st.r[-10:1000:2,ns], stats=600L, measure=0,photonNum=1,Resfreq=None,driveLen=5*us,
       name='T1', save=True, collect=True, noisy=True, state=1,SZ=0.0,
       update=True, plot=False):
    """A single pi-pulse on one qubit, with other qubits also operated."""
    if plot and (not save):
        raise Exception("Cannot plot without saving. This is Dan's fault, bother him")
    if update and (state>1):
        raise Exception('updating with states above |1> not yet implemented')
    sample, qubits, Qubits = util.loadQubits(sample,write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]
    ml.setMultiKeys(q,state)
    if state>1: name=name+' for |'+str(state)+'>'

    if Resfreq is None: Resfreq = q['readout frequency']
    name = name  + str(photonNum) + ' photons'
    axes = [(delay, 'Measure pulse delay')]
    kw = {'stats': stats,
          'photon number': photonNum}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, delay):
        q.xy = eh.boostState(q, driveLen, state)
        q['readout amplitude']= np.sqrt([photonNum])/q['photonNum2Amp']
        q['readout frequency'] = Resfreq
        q['readout length'] = driveLen+(state-0.5)*q['piLen']+delay
        q['adc demod frequency'] = q['readout frequency']-q['readout fc']
        q.rr = eh.ResSpectroscopyPulse(q, 0*ns, q['readout frequency']-q['readout fc'])
        q.z = eh.measurePulse(q, q['readout length']+q['releaseLen'], state=state)
        q.cz = env.rect(0*ns,q['readout length'],SZ)
        q.cz += env.rect(q['readout length'], q['releaseLen'], q['cali_sz'])
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if plot or update:
        with labrad.connect() as cxn:
            dv = cxn.data_vault
            dataset = dstools.getOneDeviceDataset(dv,-1,session=sample._dir,
                                                  deviceName=None,averaged=False)
        result = fitting.t1(dataset,timeRange=(10*ns,delay[-1]))
    if plot:
        fig = dstools.plotDataset1D(dataset.data,
                                    dataset.variables[0],dataset.variables[1],
                                    marker='.',markersize=15)
        ax = fig.get_axes()[0]
        ax.plot(dataset.data[:,0],result['fitFunc'](dataset.data[:,0],result['fit']),'r')
        ax.grid()
        fig.show()
    if update:
        Q['calT1']=result['T1']
    return result['T1']

def ramsey(sample, measure=0, delay=st.r[0:200:1,ns], fringeFreq = 50*MHz,
           stats=600L, name='Ramsey', save = True, noisy=True,
           collect = False, randomize=False, averages = 1, tomo=True, state=1,
           plot=True,update=True):
    """Ramsey sequence on one qubit. Can be single phase or 4-axis tomo, and
    can have randomized time axis and/or averaging over the time axis multiple
    times

    PARAMETERS
    sample: object defining qubits to measure, loaded from registry
    measure - scalar: number of qubit to measure. Only one qubit allowed.
    delay - iterable: time axis
    fringeFreq - value [Mhz]: Desired frequency of Ramsey fringes
    stats - scalar: number of times a point will be measured per iteration over
            the time axis. That the actual number of times a point will be
            measured is stats*averages
    name - string: Name of dataset.
    save - bool: Whether or not to save data to the datavault
    noisy - bool: Whether or not to print out probabilities while the scan runs
    collect - bool: Whether or not to return data to the local scope.
    randomize - bool: Whether or not to randomize the time axis.
    averages - scalar: Number of times to iterate over the time axis.
    tomo - bool: Set True if you want to measure all four tomo axes, False if
                 you only want the X axis (normal Ramsey fringes).
    """
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]
    q['readout'] = True
    if (update or plot) and not(q.has_key('calT1')):
        raise Exception("Cannot plot and fit until you do a T1 scan")
    ml.setMultiKeys(q,state) #Creates q['multiKeys']
    #Randomize time axis if you want
    if randomize:
        delay = st.shuffle(delay)

    #Generator that produces time delays. Iterates over the list of delays as many times as specified by averages.
    delay_gen = st.averagedScan(delay,averages,noisy=noisy)

    axes = [(delay_gen(), 'Delay')]
    #If you want XY state tomography then we use all four pi/2 pulse phases
    if tomo:
        deps = [('Probability', '+X', ''),('Probability', '+Y', ''),
                ('Probability', '-X', ''),('Probability', '-Y', '')]
        tomoPhases = {'+X': 0.0, '+Y': 0.25, '-X': -0.5, '-Y': -0.25} #[+X, +Y, -X, -Y] in CYCLES
    #Otherwise we only do a final pi/2 pulse about the +X axis.
    else:
        deps = [('Probability', '|'+str(state)+'>', '')]
        tomoPhases = {'+X': 0.0}

    kw = {'averages': averages, 'stats': stats, 'fringeFrequency': fringeFreq}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)
    timeUpstate = (state-1)*q['piLen']
    dt = q['piFWHM']

    #Pump pulse is at time=0 (after the time to reach the initial state) with phase=0
    pump = eh.piHalfPulse(q, timeUpstate, phase=0.0, state=state)
    #Probe is at variable time with variable phase
    def probe(time, tomoPhase):
        return eh.piHalfPulse(q, timeUpstate+time, phase = 2*np.pi*(fringeFreq['GHz']*time['ns']+tomoPhases[tomoPhase]), state=state)

    def func(server, delay):
        reqs = []
        for tomoPhase in tomoPhases.keys():
            q.xy = eh.boostState(q, 0, state-1) + eh.mix(q, pump + probe(delay, tomoPhase = tomoPhase), state=state)
            q.z = eh.measurePulse(q,timeUpstate+dt+delay, state=state)
            reqs.append(runQubits(server, qubits, stats, probs=[1]))
        probs = yield FutureList(reqs)
        data = [p[0] for p in probs]
        returnValue(data)
    #Run the scan and save data
    data = sweeps.grid(func, axes, dataset = save and dataset, collect=collect, noisy=noisy)
    #Fit the data. Plot if desired. Update the registered value of T2
    #Fit. Must first retrieve dataset from datavault
    if plot or update:
        T1 = q['calT1']
        with labrad.connect() as cxn:
            dv = cxn.data_vault
            dataset = dstools.getOneDeviceDataset(dv, datasetNumber=-1,session=sample._dir,
                                                           deviceName=None, averaged=averages>1)
        if tomo:
            result = fna.ramseyTomo_noLog(dataset, T1=T1, timeRange=(10,delay[-1]),
                                                                 plot=plot)
        else:
            raise Exception('Cannot do plotting or fitting without tomo. It would be easy to fix this')
        if update:
            Q['calT2']=result['T2']
    return data


def DecoherenceWithPhotons(sample,reps=10, measure=0, SZ=0.0,numbers = np.linspace(0,100,21)):
    outfile1 = open('U:\\Yu\\Decoherence with photon raw.txt','w')
    outfile2 = open('U:\\Yu\\Decoherence with photon.txt','w')

    numList = [0,20,40,60,80,100]
    f10List = np.asarray([6.0874, 6.0830,6.0780,6.0741,6.0700,6.0640])
    f21List = np.asarray([5.8441,5.8453, 5.8460,5.8459,5.842,5.842])
    f10_interp = interpolate.interp1d(numList, f10List)
    f21_interp = interpolate.interp1d(numList, f21List)
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]
    for num in numbers:
        t1List = []
        t2List = []
        f10 = f10_interp(num)*GHz
        f21 = f21_interp(num)*GHz
        fc = f10 + 200*MHz
        Q['f10'] = f10
        Q['f21'] = f21
        Q['fc'] = fc
        for i in range(reps):
            t1 = T1WithPhotons(sample, photonNum=num, delay = st.r[0:1000:10, ns], stats=1500, plot=False, update=True,SZ=SZ)
            Q['calT1'] = t1
            t2 = RamseyWithPhotons(sample, photonNum=num, SZ=SZ,plot=False, update=True)
            Q['calT2'] = t2
            data = str(num)+ ',' + str(t1) + ',' + str(t2)
            print data
            print>>outfile1, data
            t1List.append(t1)
            t2List.append(t2)
        t1List = np.asarray(t1List)
        t2List = np.asarray(t2List)
        data_final = str(num) + ',' + str(t1List.mean()) + ',' + str(t1List.std())+ ',' + str(t2List.mean()) + ',' + str(t2List.std())
        print data_final
        print>>outfile2, data_final

    outfile1.close()
    outfile2.close()
