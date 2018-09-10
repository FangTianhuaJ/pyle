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










def channels(sample, measure=0,channel=0):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    Q['channels']=[]
    q['channels']=[]
    if channel==0: #squid readout
        for i in q['channelsObs']:
            if (i[0]=='readout ADC'):
                continue
            else:
                q['channels'].append(i)
    elif channel==1: #resonator readout
        for i in q['channelsObs']:
            if (i[0]=='timing') or (i[0]=='squid') :
               continue
            else:
               q['channels'].append(i)
    Q['channels']=q['channels']

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

def resonatorSpectroscopy(sample, freqScan=None, swapTime=300*ns, stats=600L, measure=0,SZ=0.0,SZ0=0.0,paraName='0',amplitude=None,sb_freq=-0*MHz,mod=1,
       name='Resonator spectroscopy', save=True, collect=True, noisy=True):
    """A single pi-pulse on one qubit, with other qubits also operated."""
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    if freqScan is None:
        freqScan = st.r[q['readout frequency']['GHz']-0.002:q['readout frequency']['GHz']+0.002:0.00002,GHz]
    axes = [(freqScan, 'Resonator frequency')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, curr):
        sb_freq = curr-q['readout fc']
        #q['readout fc']=curr-sb_freq
        q['readout frequency']=curr
        if amplitude is not None:
            q['readout amplitude'] = amplitude
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
            q.z=  env.rect(q.piLen/2+20*ns, swapTime, SZ0)
            q.z +=eh.measurePulse(q, q.piLen/2+30*ns+swapTime)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

def resonatorT1(sample, delay=st.arangePQ(-0.01,1,0.01,'us')+st.arangePQ(1,7.5,0.05,'us'),paraName='0',stats=1200L,zpa=0.0,SZ=0.0,
                measure=0,name='resonator T1 MQ', excited = False, adi = False, save=True, collect=True, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    if excited: name = name + ' (with qubit excitation)'
    axes = [(zpa, 'z pulse amp'),(delay, 'Measure pulse delay')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, paraName+' '+name, axes, measure=measure, kw=kw)


    sl = q['swapLen'+paraName]
    sa = q['swapAmp'+paraName]


    def func(server, currZpa, delay):
        q['z'] = env.NOTHING
        q.xy = eh.mix(q, eh.piPulseHD(q, 0))
        q.z += env.rect(q.piLen/2, sl, sa)
        if excited:
            q.xy += eh.mix(q, eh.piPulseHD(q,1.0*q.piLen+sl))
            q.cz = env.rect(q.piLen*1.5+sl, delay, SZ)
            if adi:
                q.z += env.trapezoid(q.piLen*1.5+sl, q['adiRampLen'], delay, q['adiRampLen'], currZpa)
                q.z += env.rect(q.piLen*1.5+sl+delay+2*q['adiRampLen'], q['swapLen1'], q['swapAmp1'])
                q.z += env.rect(q.piLen*1.5+sl+delay+q['swapLen1']+2*q['adiRampLen'], sl, sa)
                q.z += eh.measurePulse(q, q.piLen*1.5+sl+delay+q['swapLen1']+sl+2*q['adiRampLen'])
            else:
                q.z += env.rect(q.piLen*1.5+sl, delay, currZpa)
                q.z += env.rect(q.piLen*1.5+sl+delay, q['swapLen1'], q['swapAmp1'])
                q.z += env.rect(q.piLen*1.5+sl+delay+q['swapLen1'], sl, sa)
                q.z += eh.measurePulse(q, q.piLen*1.5+sl+delay+q['swapLen1']+sl)
        else:
            q.cz = env.rect(q.piLen/2+sl, delay, SZ)
            if adi:
                q.z += env.trapezoid(q.piLen/2+sl, q['adiRampLen'], delay, q['adiRampLen'], currZpa)
                q.z += env.rect(q.piLen/2+sl+delay+2*q['adiRampLen'], sl, sa)
                q.z += eh.measurePulse(q, q.piLen/2+sl+delay+sl+2*q['adiRampLen'])
            else:
                q.z += env.rect(q.piLen/2+sl, delay, currZpa)
                q.z += env.rect(q.piLen/2+sl+delay, sl, sa)
                q.z += eh.measurePulse(q, q.piLen/2+sl+delay+sl)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
        return
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


def coherent(sample, probeLen=st.r[0:200:2,ns], drive=st.r[0:1:0.05], stats=600L, measure=0,sb_freq=-50*MHz,paraName='0',SZ=None,SZ0=None, mode='rect',
       name='Coherent state', save=True, collect=True, noisy=True):
    """This is a program used for calibrating photon numbers."""
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    #c = qubits[measureC]

    if mode == 'gaussian': name+' (Gaussian)'
    if SZ is None: SZ = 0.0
    if SZ0 is None: SZ0 = q['close_sz']
    q['readout fc'] =  q['readout frequency']- sb_freq
    axes = [(probeLen, 'Measure pulse length'),(drive, 'Resonator uwave drive Amp')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, currLen, currDri):
        q['readout amplitude']=currDri
        if mode== 'gaussian':
            q['resGaussianAmp'] = currDri
            q.rr = eh.ResPulseGaussian(q, q['resGaussianLen']/2, sb_freq)
            Len = q['resGaussianLen']
            q.cz = env.rect(0, Len, SZ)
        elif mode=='rect':
            q['readout amplitude'] = currDri
            q.rr = eh.ResSpectroscopyPulse(q, 0, sb_freq)
            Len = q['readout length']
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


def measureFid(sample, repetition=100, stats=1500, measure=0, level=1,
               save=True, name='measure fidelity MQ', collect=True, update=False, noisy=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]
    repetition = range(repetition)
    axes = [(repetition, 'repetition times')]
    if level==1:
        deps = [('Probability', '|0>', ''),
                ('Probability', '|1>', ''),
                ('Visibility', '|1> - |0>', ''),
                ]
    elif level==2:
        deps = [('Probability', '|1>', ''),
                ('Probability', '|2>', ''),
                ('Visibility', '|2> - |1>', '')
                ]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, 'level'+str(level)+str(level-1)+' '+name, axes, deps, measure=measure, kw=kw)

    def func(server, curr):
        t_pi = 0
        t_meas = q['piLen']/2.0

        if level == 2:
            # with pi-pulse
            q['readout'] = True
            q.xy = eh.mix(q, eh.piPulseHD(q, t_pi))
            q.z = eh.measurePulse2(q, t_meas)
            req1 = runQubits(server, qubits, stats, probs=[1])

            # |2> with pi-pulse
            q['readout'] = True
            q.xy = eh.mix(q, eh.piPulseHD(q, t_pi-q.piLen))+eh.mix(q, env.gaussian(t_pi, q.piFWHM, q.piAmp21, df=q.piDf21), freq = 'f21')
            q.z = eh.measurePulse2(q, t_meas)
            req2 = runQubits(server, qubits, stats, probs=[1])

            probs = yield FutureList([req1, req2])
            p1, p2 = [p[0] for p in probs]

            returnValue([p1, p2, p2-p1])

        elif level == 1:
            # without pi-pulse
            q['readout'] = True
            q.xy = env.NOTHING
            q.z = eh.measurePulse(q, t_meas)
            req0 = runQubits(server, qubits, stats, probs=[1])

            # with pi-pulse
            q['readout'] = True
            q.xy = eh.mix(q, eh.piPulseHD(q, t_pi))
            q.z = eh.measurePulse(q, t_meas)
            req1 = runQubits(server, qubits, stats, probs=[1])

            probs = yield FutureList([req0, req1])
            p0, p1 = [p[0] for p in probs]

            returnValue([p0, p1, p1-p0])

    result = sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)
    if level==1:
        measFidVal = np.sum(result,axis=0)/len(repetition)
        Q.measureF0 = 1-measFidVal[1]
        Q.measureF1 = measFidVal[2]
    return result


def baretomo(sample, repetition=10, stats=3000L, measure=0, name='|0> state tomography', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    measurement = pyle.dataking.measurement.Tomo(3, [0])

    repetition = range(repetition)
    axes = [(repetition, 'repetition')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measurement, kw=kw)

    def func(server, curr):
        print 'baretomo'
        start = 0
        return measurement(server, qubits, start, **kw)

    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    # rho_ideal = np.array([[0.0,0.0j,0.0j,0.0j],[0.0j,0.5,0.5,0.0j],[0j,0.5,0.5,0j],[0j,0j,0j,0]])
    resultAvg = np.sum(result,axis=0)/len(repetition)
    Qk = np.reshape(resultAvg[1:],(3,2))

    #Qk = readoutFidCorr(Qk, [sample.q0.measureF0,sample.q0.measureF1,sample.q1.measureF0,sample.q1.measureF1])
    rho_cal = tomo.qst(Qk,'tomo')
    plotRhoSingle(rho_cal,figNo=100)
    pylab.title('Exp.')
    if collect:
        return result

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

def arbitraryState(sample,state, pulseTimeFunc, offresTimeFunc, alpha=None,probePulseLength=st.r[0:300:2,ns],
                     repeat=10,dumpPulseLength=None, dumpPulseAmplitude=None, probePulseDelay=0*ns, SZ=0.0,
                     measure=0,measureR=2,paraName='0', stats=1200L,
                     name='arbitrarystate', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    r = qubits[measureR]

    if alpha==None:
#alpha = array([1.1,1.45])[:,None] * exp(1.0j*linspace(0,2*pi,30,endpoint=False))[None,:]
#alpha = reshape(alpha,size(alpha))
#plot(real(alpha),imag(alpha))
        alpha = np.linspace(-2.0,2.0,25)
        alpha = alpha[:,None]+1.0j*alpha[None,:]
        alpha = np.reshape(alpha,np.size(alpha))

    state = numpy.array(state)
    drives, swapTimes, zPulses = sequence(state,verbose=True)
    drives = drives * float(q.piAmp)/numpy.pi
    offresTimes = np.arange(numpy.size(zPulses))
    swapTimesV = np.arange(numpy.size(zPulses))
    for i in np.arange(numpy.size(zPulses)):
        offresTimes[i] = offresTimeFunc(zPulses[i])*ns
        swapTimesV[i]= pulseTimeFunc(swapTimes[i]/numpy.pi)*ns

    print type(offresTimes)

    arbitraryTomo(sample, swapTimesV, offresTimes = offresTimes, pulseAmplitudes = drives, repeat=5,
              stats=stats, measure=measure,measureR=measureR,paraName=paraName,
              dumpPulseAmplitude = dumpPulseAmplitude, dumpPulseLength=dumpPulseLength,probePulseDelay=probePulseDelay,
              name = name+' tomo', save=save,collect=collect,noisy=noisy)
    arbitraryWigner(sample, swapTimesV,offresTimes = offresTimes, pulseAmplitudes = drives, alpha=alpha,probePulseLength=probePulseLength,
              stats=stats, measure=measure,measureR=measureR,paraName=paraName,SZ=SZ,
              dumpPulseAmplitude = dumpPulseAmplitude, dumpPulseLength=dumpPulseLength, probePulseDelay=probePulseDelay,
              name = name+' wigner', save=save,collect=collect,noisy=noisy)
    #arbitraryTest(sample, swapTimesV,offresTimes = offresTimes, pulseAmplitudes = drives, probePulseLength=probePulseLength,
              #stats=stats, measure=measure,measureR=measureR,paraName=paraName,
              #dumpPulseAmplitude = dumpPulseAmplitude, dumpPulseLength=dumpPulseLength, probePulseDelay=probePulseDelay,
              #name = name+' wigner', save=save,collect=collect,noisy=noisy)

def arbitraryTomo(sample, swapTimes, offresTimes=None, pulseAmplitudes=None, repeat=5,stats=1200L,measure=0,measureR=2,paraName='0',dumpPulseAmplitude = None, dumpPulseLength=None, probePulseDelay=0*ns,
                name='arbitrary Wigner', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    r = qubits[measureR]

    nPulses = len(swapTimes)
    delay = q.piLen/2
    rdelay = r.piLen/2
    pilength = q.piLen
    qf = q.f10
    rf = r.fRes0
    af = q.fc
    sa = float(q['swapAmp'+paraName])
    os = float(q['swapAmp'+paraName+'OS'])+numpy.zeros(nPulses)

    measurement = pyle.dataking.measurement.Tomo(3, [0])
    kw = {'stats': stats}


    if offresTimes is None:
        offresTimes = numpy.resize(2*delay,nPulses)
    # else:
       # kw['times off resonance'] = offresTimes
    if pulseAmplitudes is None:
        pulseAmplitudes = numpy.resize(float(q.piAmp),nPulses)
    # else:
        # kw['pulse amplitudes'] = pulseAmplitudes
    if dumpPulseAmplitude is None and dumpPulseLength is None:
        # kw['dumpPulseAmplitude'] = dumpPulseAmplitude
        # kw['dumpPulseLength'] = dumpPulseLength


        dumpPulseLength = 0*ns
        dumpPulseAmplitude = 0
    if probePulseDelay is None:
        # kw['probe swap pulse delay'] = probePulseDelay

        probePulseDelay = 0*ns

    repetition = range(repeat)
    axes = [(repetition, 'repetition')]

    dataset = sweeps.prepDataset(sample, name, axes, measure=measurement, kw=kw)


    def func(server, curr):
        start = 0.0
        q.xy = env.NOTHING
        q.z=env.NOTHING
        for i in numpy.arange(len(swapTimes)):
            start += offresTimes[i]
            q.piAmp = pulseAmplitudes[i]
            q.xy += eh.mix(q, env.gaussian(start - delay, q.piFWHM, numpy.conjugate(pulseAmplitudes[i]) * numpy.exp(2.0j*numpy.pi*(qf-rf)*start)))    #No HD
            q.z += env.rect(start, swapTimes[i], sa,overshoot = os[i])
            start += swapTimes[i]
        resstart = start + probePulseDelay + 2*rdelay
        cut2 = resstart
        if dumpPulseLength is not None:
            start += 4
            q.z += env.rect(start, dumpPulseLength, dumpPulseAmplitude)
        cut1 = start

        start += 4
        start = max(start,resstart)
        if start < cut2:
            cut2 = start
        start += delay
        tomostart = start
        return measurement(server, qubits, tomostart, **kw)
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)


    return

def arbitraryWigner(sample, swapTimes, offresTimes=None, pulseAmplitudes=None, alpha=[0.5], probePulseLength=st.r[0:300:2,ns],
              stats=1200L,measure=0,measureR=2,measureC=1,SZ=0.0,paraName='0',dumpPulseAmplitude = None, dumpPulseLength=None, probePulseDelay=0*ns,
              name='arbitrary Wigner', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    r = qubits[measureR]
    c = qubits[measureC]

    nPulses = len(swapTimes)
    delay = q.piLen/2
    rdelay = r.piLen/2
    pilength = q.piLen
    qf = q.f10
    rf = r.fRes0
    af = q.fc
    sa = float(q['swapAmp'+paraName])
    os = float(q['swapAmp'+paraName+'OS'])+numpy.zeros(nPulses)

    sweepPara = complexSweep(np.array(alpha),probePulseLength)

    kw = {'stats': stats,'measure': measure}

    if offresTimes is None:
        offresTimes = numpy.resize(2*delay,nPulses)
    else:
        kw['times off resonance'] = offresTimes
    if pulseAmplitudes is None:
        pulseAmplitudes = numpy.resize(float(q.piAmp),nPulses)
    else:
        kw['pulse amplitudes'] = pulseAmplitudes
    if dumpPulseAmplitude is not None and dumpPulseLength is not None:
        kw['dumpPulseAmplitude'] = dumpPulseAmplitude
        kw['dumpPulseLength'] = dumpPulseLength
    else:
        dumpPulseLength = 0*ns
        dumpPulseAmplitude = 0
    if probePulseDelay is not None:
        kw['probe swap pulse delay'] = probePulseDelay
    else:
        probePulseDelay = 0*ns

    dataset = sweeps.prepDataset(sample, name, axes = [('rm displacement', 're'),('rm displacement', 'im'),
               ('swap pulse length', 'ns')], measure=measure, kw=kw)

    def func(server, curr):
        am = curr[0]
        currLen = curr[1]

        start = 0.0
        q.xy = env.NOTHING
        q.z=env.NOTHING
        for i in numpy.arange(len(swapTimes)):
            start += offresTimes[i]
            q.piAmp = pulseAmplitudes[i]
            q.xy += eh.mix(q, env.gaussian(start - delay, q.piFWHM, numpy.conjugate(pulseAmplitudes[i]) * numpy.exp(2.0j*numpy.pi*(qf-rf)*start)))    #No HD
            q.z += env.rect(start, swapTimes[i], sa,overshoot = os[i])
            start += swapTimes[i]
        resstart = start + probePulseDelay + 2*rdelay
        cut2 = resstart

        if dumpPulseLength is not None:
            start += 4
            q.z += env.rect(start, dumpPulseLength, dumpPulseAmplitude)
        cut1 = start

        r.xy = eh.mix(r, env.gaussian(resstart - rdelay, r.piFWHM, numpy.conjugate(am*r.drivePhase)),freq='fRes0')
        c.z =env.rect(resstart-rdelay-r.piLen/2,r.piLen+12,SZ)
        start += 4
        start = max(start, resstart)

        q.z += env.rect(start+12, currLen, sa, overshoot=q['swapAmp'+paraName+'OS'])
        start += currLen + 4+12

        q.z += eh.measurePulse(q, start)

        q['readout'] = True
        data = yield runQubits(server, qubits, stats=stats, probs=[1])
        data = np.hstack(([am.real, am.imag, currLen], data))
        returnValue(data)
    result = sweeps.run(func, sweepPara, dataset=save and dataset, noisy=noisy)
    return result


def arbitraryTest(sample, swapTimes, offresTimes=None, pulseAmplitudes=None,  probePulseLength=st.r[0:300:2,ns],
              stats=1200L,measure=0,measureR=2,paraName='0',dumpPulseAmplitude = None, dumpPulseLength=None, probePulseDelay=0*ns,
              name='arbitrary Wigner', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    r = qubits[measureR]

    nPulses = len(swapTimes)
    delay = q.piLen/2
    rdelay = r.piLen/2
    pilength = q.piLen
    qf = q.f10
    rf = r.fRes0
    af = q.fc
    sa = float(q['swapAmp'+paraName])
    os = float(q['swapAmp'+paraName+'OS'])+numpy.zeros(nPulses)

    axes = [(probePulseLength, 'test length')]


    kw = {'stats': stats,'measure': measure}

    if offresTimes is None:
        offresTimes = numpy.resize(2*delay,nPulses)
    else:
        kw['times off resonance'] = offresTimes
    if pulseAmplitudes is None:
        pulseAmplitudes = numpy.resize(float(q.piAmp),nPulses)
    else:
        kw['pulse amplitudes'] = pulseAmplitudes
    if dumpPulseAmplitude is not None and dumpPulseLength is not None:
        kw['dumpPulseAmplitude'] = dumpPulseAmplitude
        kw['dumpPulseLength'] = dumpPulseLength
    else:
        dumpPulseLength = 0*ns
        dumpPulseAmplitude = 0
    if probePulseDelay is not None:
        kw['probe swap pulse delay'] = probePulseDelay
    else:
        probePulseDelay = 0*ns

    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, currLen):


        start = 0.0
        q.xy = env.NOTHING
        q.z=env.NOTHING
        for i in numpy.arange(len(swapTimes)):
            start += offresTimes[i]
            q.piAmp = pulseAmplitudes[i]
            q.xy += eh.mix(q, env.gaussian(start - delay, q.piFWHM, numpy.conjugate(pulseAmplitudes[i]) * numpy.exp(2.0j*numpy.pi*(qf-rf)*start)))    #No HD
            q.z += env.rect(start, swapTimes[i], sa,overshoot = os[i])
            start += swapTimes[i]
        resstart = start + probePulseDelay + 2*rdelay
        cut2 = resstart

        if dumpPulseLength is not None:
            start += 4
            q.z += env.rect(start, dumpPulseLength, dumpPulseAmplitude)
        cut1 = start


        start += 4
        start = max(start, resstart)

        q.z += env.rect(start, currLen, sa, overshoot=q['swapAmp'+paraName+'OS'])
        start += currLen + 4

        q.z += eh.measurePulse(q, start)

        q['readout'] = True
        return runQubits(server, qubits, stats=stats, probs=[1])

    result = sweeps.grid(func, axes, save=save,dataset=save and dataset, noisy=noisy)
    return

def complexSweep(displacement, sweepTime):
    return [[d,sT] for d in displacement for sT in sweepTime]


def adiTest(sample, testLen=st.arangePQ(20,100,0.2,ns), detuningratio=st.r[1.0:0.8:-0.001], measure=1, n=1,qubit=0,paraName='C',stats=600L,Delay=8*ns,
         name='adiabatic process testing', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    sa = q['swapAmp'+paraName]
    axes = [(detuningratio, 'detuingratio'),(testLen, 'test length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server,currRatio, currLen):
        #q.xy = eh.mix(q, eh.piPulseHD(q, 0))
        #q.z = env.trapezoid(0+q['piLen']/2, 0, 0, currLen, testAmp)
        start = 0
        #if pulseshape==0:
            #q.z = rampPulse(start, k*detuningratio, dispT-2*curr[scank]*detuningratio, sa*detuningratio)
        #elif pulseshape==1:
        q.xy = eh.mix(q, eh.piPulseHD(q, start+q.piLen/2))
        start += q.piLen+Delay
        q.z = env.rect(start, q['swapLen'+paraName+'s'][0], q['swapAmp'+paraName],overshoot=q['swapAmp'+paraName+'OSs'][0])
        start += q['swapLen'+paraName+'s'][0]+Delay
        if n==2:
            q.xy += eh.mix(q, eh.piPulseHD(q, start+q.piLen/2))
            start += q.piLen+Delay
            q.z += env.rect(start, q['swapLen'+paraName+'s'][1], q['swapAmp'+paraName],overshoot=q['swapAmp'+paraName+'OSs'][1])
            start += q['swapLen'+paraName+'s'][1]+Delay
        elif n==3:
            q.xy += eh.mix(q, eh.piPulseHD(q, start+q.piLen/2))
            start += q.piLen+Delay
            q.z += env.rect(start, q['swapLen'+paraName+'s'][1], q['swapAmp'+paraName],overshoot=q['swapAmp'+paraName+'OSs'][1])
            start += q['swapLen'+paraName+'s'][1]+Delay
            q.xy += eh.mix(q, eh.piPulseHD(q, start+q.piLen/2))
            start += q.piLen+Delay
            q.z += env.rect(start, q['swapLen'+paraName+'s'][2], q['swapAmp'+paraName],overshoot=q['swapAmp'+paraName+'OSs'][2])
            start += q['swapLen'+paraName+'s'][2]+Delay
        if qubit==1:
            q.xy += eh.mix(q, eh.piPulseHD(q, start+q.piLen/2))
        #start += q.piLen+8*ns

        #q.z += env.parabola(start, currLen,sa*currRatio)

        #elif pulseshape==2:
            #q.z = flattop(start+k*detuningratio*1.5,dispT-2*curr[scank]*detuningratio,k*detuningratio,sa*detuningratio,sbfreq=0)
        #elif pulseshape==3:
            #q.z = gaussPulse(start+k*detuningratio*1.5,k,amplitude=sa*detuningratio,sbfreq=0)
        start += currLen+Delay
        q.z += eh.measurePulse(q, start)
        q['readout'] = True
        return runQubits(server, qubits, stats=stats, probs=[1])

    return sweeps.grid(func, axes, save=save, dataset=dataset, collect=collect, noisy=noisy)

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


def testICCdelayRZ(sample, t0=st.r[-30:30:0.25,ns], measure=0,measureC=1,measureR=2,paraName='0',RZ=st.r[0.04:-0.03:0.001,None],
               stats=1200, update=True,
              save=True, name='Test Delay', plot=True, noisy=True):
    """ This is to calibrate the test delay between qubit z pulse and resonator z pulse"""
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    c = qubits[measureC]
    C = Qubits[measureC]
    r = qubits[measureR]
    R = Qubits[measureR]

    axes = [(RZ,'squid z'),(t0, 'Detuning pulse center')]
    kw = {
        'stats': stats,
    }
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, currRZ,t0):
        q.xy = eh.mix(q, eh.piPulse(q, -q['piLen']/2-q['swapLen'+paraName]/2))
        r.z = env.rect(t0-q['swapLen'+paraName]*1.0/2,q['swapLen'+paraName]*1.0,currRZ)
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
    print 'ICC r z lag:', fit[0]
    if update:
        print 'ICC r z corrected by %g ns' % fit[0]
        R['timingLagMeas'] += fit[0]*ns

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

def ResT1WithSB(sample, delayLen=st.arangePQ(0,1,0.01,'us')+st.arangePQ(1,6.0,0.05,'us'),SB=st.r[20:-600:-5,mV],paraName='0',stats=1200L, measure=0,measureC=1,
       name='resonator T1 with dynamic tuning', save=True, collect=True, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    #c= qubits[measureC]

    xaxis=range(np.size(SB))
    axes = [(xaxis,'number of squid bias'),(delayLen, 'Measure pulse delay')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)


    sb,TableSwapAmp,TableSwapLen, TablePiAmp, TableQfreq = PitunerTable(sample,SB=SB,measure=measure,measureC=measureC)


    def func(server, xaxisnumber,currLen):
        q['couplerfluxBias']=SB[xaxisnumber]
        sl=TableSwapLen[xaxisnumber]
        sa=TableSwapAmp[xaxisnumber]
        q.piAmp=TablePiAmp[xaxisnumber]
        q.f10=TableQfreq[xaxisnumber]
        q.xy = eh.mix(q, eh.piPulseHD(q, 0))
        q.z = env.rect(q.piLen/2, sl, sa)
        q.z += env.rect(q.piLen/2+sl+currLen, sl, sa)
        q.z += eh.measurePulse(q, q.piLen/2+sl+currLen+sl)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

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


def ResT1WithRZ(sample, delay=st.arangePQ(0,1,0.01,'us')+st.arangePQ(1,5,0.05,'us'),RZ=st.r[0:0.2:0.05,None],paraName='0',stats=1800L, measure=0,
       measureC=1,measureR=2,name='resonator T1 with dynamic tuning', save=True, collect=True, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    c= qubits[measureC]
    r = qubits[measureR]


    axes = [(RZ,'squid Z bias'),(delay, 'Measure pulse delay')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)


    sl = q['swapLen'+paraName]
    sa = q['swapAmp'+paraName]


    def func(server, rSZ,delay):
        q.xy = eh.mix(q, eh.piPulseHD(q, 0))
        q.z = env.rect(q.piLen/2, sl, sa)
        r.z =env.rect(q.piLen/2+sl,delay,rSZ)
        q.z += env.rect(q.piLen/2+sl+delay, sl, sa)
        q.z += eh.measurePulse(q, q.piLen/2+sl+delay+sl)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)


def dycoherent(sample, probeLen=st.r[0:400:2,ns], drive=st.r[0:0.15:0.005], SZ=st.r[0:0.2:0.02,None],
       freq=6.5637*GHz,mode=0,readLen=20*ns,stats=600L, measure=0,paraName='0',
       name='Coherent state', save=True, collect=True, noisy=True):
    """A single pi-pulse on one qubit, with other qubits also operated."""
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    axes = [(probeLen, 'Measure pulse length'),(drive, 'Resonator uwave drive Amp'),(SZ,'Squid Z bias')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, currLen, currDri,currSZ):
        q['readout frequency']=freq
        q['readout length']=readLen
        q.cz =env.rect(0,q['readout length'],currSZ)
        q['readout amplitude']=currDri
        if mode==0:   #flattop
            q.rr = eh.ResSpectroscopyPulse(q, 0*ns, q['readout frequency'] -q['readout fc'])
        elif mode==1:   #gaussian
            q.rr = eh.ResSpectroscopyPulse2(q, 0*ns, q['readout frequency'] -q['readout fc'])
        q.z = env.rect(q['readout length'], currLen, q['swapAmp'+paraName])+eh.measurePulse(q, q['readout length']+currLen)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

def zerophotonDet(sample, probeLen=st.r[0:200:0.5,ns], SZ=st.r[-0.3:0.3:0.005,None],stats=1200L, measure=0,measureC=1,measureR=2,paraName='0', measDelay=20.0*ns,amp=0.0,
       name='Coherent state', save=True, collect=True, noisy=True):
    """A single pi-pulse on one qubit, with other qubits also operated."""
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    #c = qubits[measureC]
    #r = qubits[measureR]

    axes = [(probeLen, 'Measure pulse length'),(SZ,'squid z')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, currLen,currSZ):
        q['couplerfluxBias'] =-245*mV-16.618*mV*currSZ/0.02
        q.cz =env.rect(0,7500*ns,currSZ)
        q.z = env.rect(measDelay,currLen, q['swapAmp'+paraName])+eh.measurePulse(q, measDelay+currLen)
        #r.xy = eh.mix(r, env.gaussian(r.piLen/2, r.piFWHM, amp), freq = 'fRes0')
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

def qfreqSB(sample, freqScan=st.r[6.840:6.882:0.001,GHz], SB=st.r[-200:230:1.5,mV],stats=1200L, measure=0,measureC=1,measureR=2,paraName='0',amplitude=0.04,
       name='Resonator spectroscopy', save=True, collect=True, noisy=True):
    """A single pi-pulse on one qubit, with other qubits also operated."""
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    c = qubits[measureC]



    if freqScan is None:
        f = st.nearest(q['f10'][GHz], 0.001)
        freq = st.r[f-0.04:f+0.04:0.001, GHz]
    axes = [(SB,'squid bias'),(freqScan, 'qubit frequency')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)


    def func(server, currSB,currFreq):
        sb_freq = currFreq-r.fc
        q['spectroscopyAmp'] = amplitude
        q.xy = eh.spectroscopyPulse(q, 0, sb_freq)
        q.z  = eh.measurePulse(q, q['spectroscopyLen'])
        c['biasOperate']=currSB
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


def plotcomplex(num=1,num2=0,scale=1.0,index=None,session=['','Yi','QND','wfr20110330','r4c3','110827n']):

    if num2==0:
        data=ds.getDataset(dr,dataset=num,session=session)
        if index==None:
            index=len(data[:,0])
        else:
            index=min(len(data[:,0]),index)
        s11=[complex(scale*data[i,2]*cos(data[i,1]),scale*data[i,2]*sin(data[i,1])) for i in range(index)]

    else:
        data1=ds.getDataset(dr,dataset=num,session=session)
        data2=ds.getDataset(dr,dataset=num2,session=session)
        x0,y0,R = circle_cal(num=num2,session=session,x_m=0,y_m=0)
        if index==None:
            index=len(data1[:,0])
        else:
            index=min(len(data1[:,0]),index)
        I0_new = [(data1[i,2]*cos(data1[i,1])-x0)   for i in range(index)]
        Q0_new = [(data1[i,2]*sin(data1[i,1])-y0)   for i in range(index)]
        amp0_new = [abs(I0_new[i]+1j*Q0_new[i]) for i in range(index)]
        phase0_new = [atan2(Q0_new[i],I0_new[i]) for i in range(index)]
        I1_new = [(data2[i,2]*cos(data2[i,1])-x0)   for i in range(index)]
        Q1_new = [(data2[i,2]*sin(data2[i,1])-y0)   for i in range(index)]
        amp1_new = [abs(I1_new[i]+1j*Q1_new[i]) for i in range(index)]
        phase1_new = [atan2(Q1_new[i],I1_new[i]) for i in range(index)]

        s11=[complex(amp0_new[i]/amp1_new[i]*cos(phase0_new[i]-phase1_new[i]),amp0_new[i]/amp1_new[i]*sin(phase0_new[i]-phase1_new[i])) for i in range(index)]
    figure(100)
    plot(real(s11),imag(s11),'o-')

    outfile=open('test.txt','w')
    for i in range(index):
        print>>outfile,real(s11)[i],',',imag(s11)[i]
    outfile.close()

def plotcomplex_ref(num=1,scale=1.0,index=None,session=['','Yi','QND','wfr20110330','r4c3','110827n']):


    data=ds.getDataset(dr,dataset=num,session=session)
    if index==None:
        index=len(data[:,0])
    else:
        index=min(len(data[:,0]),index)
    #s11=[complex(scale*data[i,2]*cos(data[i,1]),scale*data[i,2]*sin(data[i,1])) for i in range(index)]
    amp0 = data[:,2]
    phase0 = data[:,1]
    amp1 = data[:,4]
    phase1 = data[:,3]

    x0,y0,R = circle_cal(num=num,session=None,amp=amp1,phase=phase1,x_m=0,y_m=0)

    I0_new = [(amp0[i]*cos(phase0[i])-x0)   for i in range(index)]
    Q0_new = [(amp0[i]*sin(phase0[i])-y0)   for i in range(index)]
    amp0_new = [abs(I0_new[i]+1j*Q0_new[i]) for i in range(index)]
    phase0_new = [atan2(Q0_new[i],I0_new[i]) for i in range(index)]
    I1_new = [(amp1[i]*cos(phase1[i])-x0)   for i in range(index)]
    Q1_new = [(amp2[i]*sin(phase1[i])-y0)   for i in range(index)]
    amp1_new = [abs(I1_new[i]+1j*Q1_new[i]) for i in range(index)]
    phase1_new = [atan2(Q1_new[i],I1_new[i]) for i in range(index)]

    s11=[complex(amp0_new[i]/amp1_new[i]*cos(phase0_new[i]-phase1_new[i]),amp0_new[i]/amp1_new[i]*sin(phase0_new[i]-phase1_new[i])) for i in range(index)]
    figure(100)
    plot(real(s11),imag(s11),'o-')

    outfile=open('test.txt','w')
    for i in range(index):
        print>>outfile,real(s11)[i],',',imag(s11)[i]
    outfile.close()

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


def pulse_test(sample, cxn, amp=0.05,freq = 6.56508*GHz, mode=1, SZ=0.0,readLen=100*ns,SZ2=0.0,release=True,
               delay = 200*ns,release2=False,delay2=0*ns,releaseLen2=5000*ns,
               tail=2000*ns,sb_freq = 50*MHz, save=False, stats=150, measure=0, releaseMode=0,releaseLen=2000*ns,
               session=['','Yi','QND','wfr20110330','r4r3']):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    q['readout'] = True
    q['readoutType'] = 'resonator'
    q['biasReadout'] = q['biasOperate']
    q['adc_start_delay'] = 0*ns
    q['readout DAC start delay'] = 0*ns
    q['adc mode'] = 'average'
    q['readout amplitude']=amp
    q['readout frequency'] = freq
    q['adc demod frequency'] = sb_freq
    q['readout fc'] = q['readout frequency'] - sb_freq
    q['readout length'] = readLen
    if mode==0:
        q.rr = eh.ResSpectroscopyPulse(q, 0*ns, sb_freq)
    elif mode==1:
        q.rr = eh.ResSpectroscopyPulse2(q, 0*ns, sb_freq,phase=0)
    if release==True:
        if releaseMode==0:
            q.cz = env.rect(0,q['readout length']+000.0*ns,SZ)+env.rect(q['readout length']+delay,releaseLen,SZ2)
        elif releaseMode==1:
            q.cz = env.rect(0,q['readout length']+000.0*ns,SZ)+env.gaussian(q['readout length']+delay, releaseLen/2.0, amp=SZ2, phase=0.0, df=0.0)
            if release2:
                q. cz += env.rect(q['readout length']+delay+delay2,releaseLen2,SZ2)
        elif releaseMode==2:
            q.cz = env.rect(0,q['readout length']+000.0*ns,SZ)+env.gaussian(q['readout length']+delay, releaseLen/2.0, amp=SZ2, phase=0.0, df=0.0)
            q.cz+= env.gaussian(q['readout length']+delay+delay, releaseLen/2.0, amp=SZ2, phase=0.0, df=0.0)
            q.cz+= env.gaussian(q['readout length']+delay+delay*2, releaseLen/2.0, amp=SZ2, phase=0.0, df=0.0)
    else:
        q.cz =env.rect(0,q['readout length']+tail,SZ)

    qseq = cxn.qubit_sequencer
    ans = runQubits(qseq, qubits, stats, raw=True).wait()
    Is = np.asarray(ans[0][0])
    Qs = np.asarray(ans[0][1])


    data_output = np.empty([len(Is),3])
    for i in range(len(Is)):
        data_output[i,0]=i*2
    data_output[:,1] = Is
    data_output[:,2] = Qs

    if save:
        dv = cxn.data_vault
        dv.cd(session)
        dv.new('pulse qstate', ['time [ns]'],['I', 'Q'])
        all = [[i*2,Is[i], Qs[i]] for i in range(len(Is))]
        dv.add(all)
    return data_output

def pulse_qstate_multi(sample,cxn,coherent=True,amp=0.043,SZ=0.0,SZ2=0.0,stats=150,reps=10,releaseLen=2000*ns,releaseMode=0,sb_freq=50*MHz,delay=200*ns,release2=False,save=True,session=['','Yi','QND','wfr20110330','r4r3']):
    data0 = pulse_test(sample,cxn,amp=amp,session=session,SZ=SZ,SZ2=SZ2,stats=stats,save=False)
    data0[:,1]=0
    data0[:,2]=0


    for i in range(reps):
        if coherent:
            data_temp = pulse_test(sample,cxn,session=session,amp=amp,SZ=SZ,SZ2=SZ2,stats=stats,sb_freq=sb_freq,delay=delay,releaseMode=releaseMode,releaseLen=releaseLen,release2=release2,save=False)
        else:
            data_temp = pulse_qstate(sample,cxn,session=session,SZ=SZ,SZ2=SZ2,stats=stats,sb_freq=sb_freq,releaseMode=releaseMode,releaseLen=releaseLen,release2=release2,save=False)
        data0[:,1] +=data_temp[:,1]/stats
        data0[:,2] +=data_temp[:,2]/stats


    data0[:,1] /=reps
    data0[:,2] /=reps

    if save:
        dv = cxn.data_vault
        dv.cd(session)
        dv.new('pulse qstate', ['time [ns]'],['I', 'Q'])
        dv.add(data0)
    return data0


def pulse_qstate(sample, cxn,measure=0, SZ=0.0,SZ2=0.0,qstate=True,stats=150, sb_freq=50*MHz, releaseLen=2000*ns,releaseMode=0,release2=False,
              save=True,session=['','Yi','QND','wfr20110330','r4r3']):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    q['readout'] = True
    q['readoutType'] = 'resonator'
    q['biasReadout'] = q['biasOperate']
    q['adc_start_delay'] = 0*ns
    q['readout DAC start delay'] = 0*ns
    q['adc mode'] = 'average'
    #sb_freq = q['readout frequency']-q['readout fc']
    q['readout fc'] = q['readout frequency']-sb_freq
    q['adc demod frequency'] = sb_freq

    start=0.0
    if qstate:
        q.xy = eh.mix(q, eh.piHalfPulseHD(q, start,phase=0))
    start += q['piLen']/2.0
    q.z = env.rect(start, q['swapLen0'],q['swapAmp0'])
    start += q['swapLen0']
    if releaseMode==0:
        q.cz = env.rect(start,releaseLen,SZ)
    elif releaseMode==1:
        start += releaseLen/2.0
        q.cz = env.gaussian(start, releaseLen/2.0, amp=SZ)
        if release2:
            start += 200*ns
            q.cz += env.rect(start, 2000*ns,SZ2)
    elif releaseMode==2:
        start += releaseLen/2.0
        q.cz = env.flattop(start,releaseLen,w=100*ns,amp=SZ)
    elif releaseMode==3:
        start += releaseLen/2.0
        q.cz = env.halfgaussian(start, releaseLen/2.0, amp=SZ)
        if release2:
            start += 200*ns
            q.cz += env.rect(start, 2000*ns,SZ2)
    elif releaseMode==4:
        q.cz = env.parabola(start, releaseLen, amp=SZ)

    qseq = cxn.qubit_sequencer
    ans = runQubits(qseq, qubits, stats, raw=True).wait()
    Is = np.asarray(ans[0][0])
    Qs = np.asarray(ans[0][1])



    data_output = np.empty([len(Is),3])
    for i in range(len(Is)):
        data_output[i,0]=i*2
    data_output[:,1] = Is
    data_output[:,2] = Qs


    if save:
        dv = cxn.data_vault
        dv.cd(session)
        dv.new('pulse qstate', ['time [ns]'],['I', 'Q'])
        all = [[i*2,Is[i], Qs[i]] for i in range(len(Is))]
        dv.add(all)
    return data_output



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



def s_scanning(sample, freq=None, photonNum=1, zpa=0.0, SZ=None, pulseLen=None, sb_freq = -50*MHz, measure=0,
               stats=150, mode=0,tail=2800*ns, cali_update = False, amp_rescale = False,phase_rescale=False,
               save=True, name='S parameter scanning', collect=True, noisy=True):
    """ This is the original s_scaning with sz"""
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    if freq is None:
        f = st.nearest(q['readout frequency'][GHz], 1e-5)
        freq = st.r[f-0.0015:f+0.0015:1e-5, GHz]
    if pulseLen is None: pulseLen = q['readout length']
    #if amp is None: amp = q['readout amplitude']
    if SZ is None: SZ = 0.0
    if cali_update: SZ = q['cali_sz']
    if cali_update: name = 'Cali ' + name


    axes = [(photonNum,'photon number'), (freq, 'Frequency')]
    deps = [('Phase', 'S11 for %s'%q.__name__, rad) for q in qubits]+ [('Amplitude','S11 for %s'%q.__name__,'') for q in qubits]
    kw = {'stats': stats, 'zpa': zpa, 'SZ': SZ}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, number, freqCurr):
        amp =[]
        phase =[]
        ampCurr = np.sqrt(number)/q['photonNum2Amp']
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


def singlePhotonDetection(sample, freq=6.56478*GHz, SZ=0.0, measure=0, rabiAngle=st.r[0:2*pi:2*pi/100.0,None],period=1,releaseLen=2000*ns,
               filStart=380*ns,stats=150, save=True, name='singlePhotonDetection', collect=True, noisy=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    rabiHigh=st.r[0:q['piAmp']:q['piAmp']/100.0,None]
    axes = [(rabiHigh, 'rabiHigh')]
    deps = [('Phase', 'S11 for %s'%q.__name__, rad) for q in qubits]+ [('Amplitude','S11 for %s'%q.__name__,'') for q in qubits]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, raCurr):
        amp =[]
        phase =[]
        q['biasReadout'] = q['biasOperate']
        q['adc mode'] = 'demodulate'
        q['readout frequency']=freq
        sb_freq = q['readout frequency']-q['readout fc']
        q['adc demod frequency'] = q['readout frequency']-q['readout fc']
        q['readout'] = True
        q['readoutType'] = 'resonator'

        # q['readout length']=200*ns
        # q['readout amplitude']=0.0056
        # q.rr = eh.ResSpectroscopyPulse2(q, 0*ns, sb_freq,phase=raCurr)
        # q.cz = env.rect(0,q['readout length'],SZ)+env.rect(q['readout length']+200*ns,3000*ns,SZ)
        # q['adc filterWindowStart']=filStart+q['readout length']+200*ns

        start=0.0
        q.z = env.NOTHING
        q.cz = env.NOTHING
        q.xy = env.NOTHING
        for i in range(period):
            start = (q['piLen']+q['swapLen0']+releaseLen)*i
            q['piAmp']=raCurr
            q.xy += eh.mix(q, eh.piPulseHD(q, start,phase=0))
            start += q['piLen']/2.0
            q.z +=  env.rect(start, q['swapLen0'],q['swapAmp0'])
            start += q['swapLen0']
            q.cz +=env.rect(start,releaseLen,SZ)

        if noisy: print raCurr

        data = yield FutureList([runQubits(server, qubits, stats, raw=True)])
        I_mean = np.mean(data[0][0][0])
        Q_mean = np.mean(data[0][0][1])

        ref = I_mean + 1j*Q_mean - (Q['Cal_x0']+1j*Q['Cal_y0'])
        phase_mod = mod(freq['GHz']*q['Cal_p1'],2*pi)

        amp.append(abs(ref/Q['adc amp offset']))
        phase.append(mod(np.angle(ref)-phase_mod-Q['adc phase offset'],2*pi))

        returnValue(phase+amp)

    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=False)

    if collect:
        return data

def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = sqrt(abs((arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = sqrt(abs((arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y) http://www.scipy.org/Cookbook/FittingData
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: ravel(gaussian(*p)(*indices(data.shape)) -data)
    p, success = optimize.leastsq(errorfunction, params)
    return p

def dist_cal(num=1,session=['','Yi','QND','wfr20110330','r4c3','110827n'],data = None,edge=None,bins=100):
    if data is None:
        data=ds.getDataset(dr,num,session=session)

    I1 = data[:,1]
    Q1 = data[:,2]
    I2 = data[:,3]
    Q2 = data[:,4]

    H1,xe1,ye1=np.histogram2d(I1,Q1,bins=bins)#,range=[[-50,50],[-50,50]])
    H2,xe2,ye2=np.histogram2d(I2,Q2,bins=bins)#,range=[[-50,50],[-50,50]])

    if edge is None:
        extent1 = [ye1[0], ye1[-1], xe1[-1],xe1[0]]
        extent2 = [ye2[0], ye2[-1], xe2[-1],xe2[0]]
    else:
        extent1 = [-edge, edge, edge, -edge]
        extent2 = [-edge, edge, edge, -edge]
    # plt.figure()
    # plt.imshow(H1,extent=extent,interpolation='nearest',cmap=cm.gist_earth_r/cm.copper)
    # plt.colorbar()
    plt.figure()
    plt.imshow(H1, extent=extent1,interpolation='nearest')
    plt.colorbar()
    params1 = fitgaussian(H1)
    x1_cen=xe1[0]+(xe1[-1]-xe1[0])*(params1[1]+0.5)/bins
    y1_cen=ye1[0]+(ye1[-1]-ye1[0])*(params1[2]+0.5)/bins
    xwid1=params1[3]*(xe1[-1]-xe1[0])/bins
    ywid1=params1[4]*(ye1[-1]-ye1[0])/bins
    print params1,np.shape(params1)
    print x1_cen,y1_cen,xwid1,ywid1
    fit1 = gaussian(*params1)
    #plt.contour(fit1(*indices(H1.shape)), cmap=cm.copper)
    plt.figure()
    plt.imshow(H2, extent=extent2,interpolation='nearest')
    plt.colorbar()
    params2 = fitgaussian(H2)
    x2_cen=xe2[0]+(xe2[-1]-xe2[0])*(params2[1]+0.5)/bins
    y2_cen=ye2[0]+(ye2[-1]-ye2[0])*(params2[2]+0.5)/bins
    xwid2=params2[3]*(xe2[-1]-xe2[0])/bins
    ywid2=params2[4]*(ye2[-1]-ye2[0])/bins
    print params2
    print x2_cen,y2_cen,xwid2,ywid2
    fit2 = gaussian(*params2)
    #plt.contour(fit2(*indices(H2.shape)), cmap=cm.copper)
    H3=H2-H1
    plt.figure()
    plt.imshow(H3, extent=extent2,interpolation='nearest')
    plt.colorbar()


def singlePhotonDistCoreCompare(sample, freq=6.56455*GHz, SZ=0.0, sb_freq = -50*MHz, measure=0, rep=10,period=1,amp=0.0,
               filStart=380*ns,stats=150, releaseLen=3000*ns,save=True, name='SinglePhotonDistCompare', collect=True, noisy=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    reps=range(rep)
    axes = [(reps, 'index')]
    deps = [('I1','s11',''), ('Q1','s11',''),('I2','s11',''), ('Q2','s11','')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, raCurr):
        result=[]
        q['biasReadout'] = q['biasOperate']
        q['adc mode'] = 'demodulate'
        q['readout frequency']=freq
        if sb_freq is None:
            sb_freq = q['readout frequency']-q['readout fc']
        else:
            q['readout fc'] = q['readout frequency'] - sb_freq
        q['fc'] = q['readout fc']
        q['adc demod frequency'] = q['readout frequency']-q['readout fc']
        q['readout'] = True
        q['readoutType'] = 'resonator'

        # start=0.0
        # q.z = env.NOTHING
        # q.cz = env.NOTHING
        # for i in range(period):
            # start = (q['piLen']+q['swapLen0']+releaseLen)*i
            # start += q['piLen']/2.0
            # q.z +=  env.rect(start, q['swapLen0'],q['swapAmp0'])
            # start += q['swapLen0']
            # q.cz +=env.rect(start,releaseLen,SZ)

        q['readout length']=200*ns
        q['readout amplitude']=0.00
        q.xy = eh.ResSpectroscopyPulse2(q, 0*ns, sb_freq,phase=0)
        q.cz = env.rect(0,q['readout length'],SZ)+env.rect(q['readout length']+200*ns,3000*ns,SZ)
        q['adc filterWindowStart']=filStart+q['readout length']+200*ns

        if noisy: print raCurr

        data = yield FutureList([runQubits(server, qubits, stats, raw=True)])

        Is = np.asarray(data[0][0][0])
        Qs = np.asarray(data[0][0][1])
        phase_mod = mod(freq['GHz']*q['Cal_p1'],2*pi)

        ref = Is + 1j*Qs - (Q['Cal_x0']+1j*Q['Cal_y0'])
        ref = ref*np.exp(-1j*(phase_mod+Q['adc phase offset']))

        result.append(ref.real)
        result.append(ref.imag)

        q['biasReadout'] = q['biasOperate']
        q['adc mode'] = 'demodulate'
        q['readout frequency']=freq
        if sb_freq is None:
            sb_freq = q['readout frequency']-q['readout fc']
        else:
            q['readout fc'] = q['readout frequency'] - sb_freq

        q['fc'] = q['readout fc']
        q['adc demod frequency'] = q['readout frequency']-q['readout fc']
        q['readout'] = True
        q['readoutType'] = 'resonator'

        # start=0.0
        # q.z = env.NOTHING
        # q.cz = env.NOTHING
        # q.xy = env.NOTHING
        # for i in range(period):
            # start = (q['piLen']+q['swapLen0']+releaseLen)*i
            # q.xy += eh.mix(q, eh.piHalfPulseHD(q, start,phase=0))
            # start += q['piLen']/2.0
            # q.z +=  env.rect(start, q['swapLen0'],q['swapAmp0'])
            # start += q['swapLen0']
            # q.cz +=env.rect(start,releaseLen,SZ)


        q['readout length']=200*ns
        q['readout amplitude']=amp
        q.xy = eh.ResSpectroscopyPulse2(q, 0*ns, sb_freq,phase=pi)
        q.cz = env.rect(0,q['readout length'],SZ)+env.rect(q['readout length']+200*ns,3000*ns,SZ)
        q['adc filterWindowStart']=filStart+q['readout length']+200*ns

        # if noisy: print raCurr

        data2 = yield FutureList([runQubits(server, qubits, stats, raw=True)])

        Is2 = np.asarray(data2[0][0][0])
        Qs2 = np.asarray(data2[0][0][1])

        ref2 = Is2 + 1j*Qs2 - (Q['Cal_x0']+1j*Q['Cal_y0'])
        ref2 = ref2*np.exp(-1j*(phase_mod+Q['adc phase offset']))

        result.append(ref2.real)
        result.append(ref2.imag)
        returnValue(np.vstack(result).T)
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=False)
    if collect:
        return data

def autoBringup(s):
    for i in range(2):
        mq.pituner(s)
        mq.freqtuner(s)
    swap10tuner(s, paraName = '0')
    swap10tuner(s, paraName = '1')

def autoQubitscan(sample, SZ = 0.0):
    zpas3 = np.arange(-0.38,-0.46,-0.01)
    for zpa in zpas3:
        data = spectroscopy(sample, freq = st.r[6.4:7.4:5e-4, GHz], Zpa = zpa, collect = True, update=False)
        f, p1 = data.T
        f0 = f[np.argmax(p1)]
        numbers = [1]
        for num in numbers:
            QubitSpecWithPhotons(sample, freq = f0*GHz, Resfreq = st.r[6.5670:6.5685:2e-5, GHz],zpa=zpa, stats=9000,
                                 qLen=5000*ns, resLen=5000*ns, photonNum=num, squid=False, SZ=SZ)


def dynamicQND(sample, zpa = st.r[0:-0.5:0.005], pulseLen = st.r[0:100:1, ns], photonNum = None, fDrive = None, ramp = None,
                   squid=False, measure=0, name = 'dynamic QND readout',sb_freq = -50*MHz, cali=True, save=True,
                   stats = 300, collect=False, noisy=True):
    """Qubit spectroscopy with photons in the resonator"""
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    if photonNum is None: photonNum = 1
    if fDrive is not None: q['readout frequency'] = fDrive
    if ramp is not None: q['adiRampLen'] = ramp

    axes = [(zpa, 'ZPA'), (pulseLen, 'detune pulse length')]
    if squid:
        name = name + '(Squid)'
        deps = [('P1', '|1>', ''), ('P1', '|0>', '')]
    else:
        name = name + '(Resonator)'
        deps = [('S11 amp','no pi pulse',''), ('S11 phase', 'no pi pulse', 'rad'),('S11 amp','pi pulse','rad'),
                ('S11 phase', 'pi pulse', 'rad'), ('Signal to noise ratio', '', '')]
    kw = {'stats':stats, 'photon number': photonNum}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)


    def func(server, currZpa, currLen):
        """Global settings"""
        q['readout'] = True
        #q['readout amplitude']= np.sqrt(photonNum)/q['photonNum2Amp']
        q['resGaussianAmp'] = np.sqrt(photonNum)/q['photonNum2AmpG']
        q['readout fc'] = q['readout frequency'] - sb_freq
        q['adc demod frequency'] = q['readout frequency']-q['readout fc']

        result = []

        "Excited state"
        start = 0
        q.cz = env.trapezoid(start, q['resAdiRampLen'], q['resGaussianLen'], q['resAdiRampLen'], q['open_sz'])
        start += 2*q['resAdiRampLen'] + q['resGaussianLen']
        q.rr = eh.ResPulseGaussian(q, start/2, sb_freq)
        q.xy = eh.mix(q, eh.piPulseHD(q, start))
        start += q['piLen']
        q.z = env.trapezoid(start, q['adiRampLen'], currLen, q['adiRampLen'], currZpa)
        start += 2*q['adiRampLen'] + currLen
        q['adc filterWindowStart'] = q['timingLagADC'] + start
        q.cz += env.trapezoid(start, q['resAdiRampLen'], q['resReleaseLen'], q['resAdiRampLen'], q['open_sz'])
        start += 2*q['resAdiRampLen'] + q['resReleaseLen']
        if squid:
            q['readoutType'] = 'squid'
            q.z += eh.measurePulse(q, start)
            p1_e = yield FutureList([runQubits(server, qubits, stats, probs=[1])])
            result.append(p1_e[0][0])
        else:
            q['readoutType'] = 'resonator'
            phase_mod = mod(q['readout frequency']['GHz']*q['Cal_p1'],2*pi)
            data1 = yield FutureList([runQubits(server, qubits, stats, raw=True)])
            I1 = np.asarray(data1[0][0][0])
            Q1 = np.asarray(data1[0][0][1])
            if cali: ref_e = (I1 + 1j*Q1 - (q['Cal_x0']+1j*q['Cal_y0']))*exp(-1j*(phase_mod+q['adc phase offset']))
            else: ref_e = I1 + 1j*Q1
            sig_e = ref_e.mean()

        "Ground state"
        start = 0
        q.xy = env.NOTHING
        q.cz = env.trapezoid(start, q['resAdiRampLen'], q['resGaussianLen'], q['resAdiRampLen'], q['open_sz'])
        start += 2*q['resAdiRampLen'] + q['resGaussianLen']
        q.rr = eh.ResPulseGaussian(q, start/2, sb_freq)
        q.z = env.trapezoid(start, q['adiRampLen'], currLen, q['adiRampLen'], currZpa)
        start += 2*q['adiRampLen'] + currLen
        q['adc filterWindowStart'] = q['timingLagADC'] + start
        q.cz += env.trapezoid(start, q['resAdiRampLen'], q['resReleaseLen'], q['resAdiRampLen'], q['open_sz'])
        start += 2*q['resAdiRampLen'] + q['resReleaseLen']
        if squid:
            q['readouType'] = 'squid'
            q.z += eh.measurePulse(q, start)
            p1_g = yield FutureList([runQubits(server, qubits, stats, probs=[1])])
            result.append(p1_g[0][0])
        else:
            q['readoutType'] = 'resonator'
            phase_mod = mod(q['readout frequency']['GHz']*q['Cal_p1'],2*pi)
            data0 = yield FutureList([runQubits(server, qubits, stats, raw=True)])
            I0 = np.asarray(data0[0][0][0])
            Q0 = np.asarray(data0[0][0][1])
            if cali: ref_g = (I0 + 1j*Q0 - (q['Cal_x0']+1j*q['Cal_y0']))*exp(-1j*(phase_mod+q['adc phase offset']))
            else: ref_g = I0 + 1j*Q0
            sig_g = ref_g.mean()

            sg_ratio = abs(sig_g-sig_e)/(ref_g.std()+ref_e.std())

            result.append(abs(sig_g))
            result.append(mod(np.angle(sig_g), 2*pi))
            result.append(abs(sig_e))
            result.append(mod(np.angle(sig_e), 2*pi))
            result.append(sg_ratio)

        returnValue(result)

    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if collect:
        return data


def spectroscopy(sample, freq=None, stats=300L, measure=0, sb_freq=0*GHz, detunings=None, Zpa = None, SZ = None, uwave_amp=None,
                 save=True, name='Spectroscopy MQ', collect=False, noisy=True, update=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    q['readout'] = True

    if SZ is None: SZ=0.0
    if freq is None:
        f = st.nearest(q['f10'][GHz], 0.001)
        freq = st.r[f-0.04:f+0.04:0.001, GHz]
    if uwave_amp is None:
        uwave_amp = q['spectroscopyAmp']
    if detunings is None:
        zpas = [0.0] * len(qubits)
    else:
        zpas = []
        for i, (q, df) in enumerate(zip(qubits, detunings)):
            print 'qubit %d will be detuned by %s' % (i, df)
            zpafunc = get_zpa_func(q)
            zpa = zpafunc(q['f10'] + df)
            zpas.append(zpa)

    axes = [(uwave_amp, 'Microwave Amplitude'), (freq, 'Frequency')]
    deps = [('Probability', '|1>', '')]
    kw = {'ZPA': Zpa,
        'stats': stats,
        'sideband': sb_freq
    }
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, amp, f):
        for i, (q, zpa) in enumerate(zip(qubits, zpas)):
            q['fc'] = f - sb_freq
            q.cz = env.rect(-100, qubits[measure]['spectroscopyLen']+100, SZ)
            if zpa:
                q.z = env.rect(-100, qubits[measure]['spectroscopyLen'] + 100, zpa)
            else:
                if Zpa is not None:
                    q.z = env.rect(-100, qubits[measure]['spectroscopyLen'] + 100, Zpa)
                else:
                    q.z = env.NOTHING
            if i == measure:
                q['spectroscopyAmp'] = amp
                q.xy = eh.spectroscopyPulse(q, 0, sb_freq)
                q.z += eh.measurePulse(q, q['spectroscopyLen'])
        #eh.correctCrosstalkZ(qubits)
        return runQubits(server, qubits, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if update:
        adjust.adjust_frequency(Q, data)
    if collect:
        return data



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


def ResSpecWithQexcitation(sample, freq=None, Resfreq = None, photonNum = 1, tail = 1000*ns, zpa = 0.0, squid = True, stats=300L, measure=0, amp  =None, sb_freq=-200*MHz,
                         qLen = None, resLen = None, zLen = None, qStart = 0*ns, resStart = 0*ns, res_sb_freq = -50*MHz, save=True, cali = True,  SZ=0.0,
                         name = 'ResScan with Q', collect = False, noisy = True):
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


    axes = [(Resfreq, 'resonator frequency'), (photonNum, 'resontor photon number')]
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
        q.xy = eh.mix(q, eh.piPulseHD(q, start))
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




 ########################################
 ##### old code
 ###################



def s_scanning_ref3_old(sample, freq=st.r[6.562:6.59:0.00005, GHz], amp=st.r[0:0.1:0.02,None], readLen=3000*ns,sb_freq = -50*MHz,detuning=-865.7*MHz, delay=0*ns,qstate=0,
               measure=0, stats=150,SZ=0.0,mode=0,tail=0*ns,x0=0,y0=0,p0=0,p1=0,amp_scale=True,phase_scale=True,filter=False,filStart=0*ns,filEnd=8*us,Zamp=0.0,ramp=0,
               save=True, name='S parameter scanning', collect=True, noisy=True):
    """This is a s_scanning with calibration"""
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Qubit = qubits[measure], Qubits[measure]

    axes = [(freq, 'Frequency'),(amp,'amp')]
    deps = [('Phase', 'S11 for %s'%q.__name__, rad) for q in qubits]+ [('Amplitude','S11 for %s'%q.__name__,'') for q in qubits]

    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    resFreq = q['readout frequency']

    def func(server, freqCurr,ampCurr):
        amp =[]
        phase =[]
        q['biasReadout'] = q['biasOperate']
        q['readout'] = True
        q['readout amplitude'] = ampCurr
        q['adc mode'] = 'demodulate'
        q['readout frequency']=freqCurr
        q['readout length']=readLen
        q['readout fc'] = q['readout frequency'] - sb_freq
        q['adc demod frequency'] = q['readout frequency']-q['readout fc']
        if mode==0:
            q.rr = eh.ResSpectroscopyPulse(q, 0*ns, sb_freq)
        elif mode==1:
            q.rr = eh.ResSpectroscopyPulse2(q, 0*ns, sb_freq)
        #q['couplerfluxBias']=SB

        #q.cz =env.rect(0,q['readout length'],SZ) +env.rect(q['readout length']+delay,tail,SZ)

        if qstate == 0:
            q.xy = env.NOTHING
        elif qstate == 1:
            q.xy = eh.mix(q, eh.piPulseHD(q, readLen-q['piLen']/2))




        #Zamp = q['swapAmp0']*((q['f10']['GHz']-resFreq['GHz'])-detuning['GHz'])/(q['f10']['GHz']-resFreq['GHz'])
        if ramp==1:
            q.z = env.rect(readLen,delay+tail,Zamp)
        elif ramp==2:
            q.z = env.trapezoid(q['readout length'], 20*ns, 20*ns, 20*ns, Zamp) +env.rect(q['readout length']+delay,tail,-1.4)

        if filter:
            q['adc filterWindowStart']=400*ns+filStart
            #q['adc filterWindowEnd']=400*ns+filEnd

        if noisy: print freqCurr, Zamp

        data = yield FutureList([runQubits(server, qubits, stats, raw=True)])

        I = np.mean(data[0][0][0])
        Q = np.mean(data[0][0][1])

        phase_mod = mod(freqCurr['GHz']*p1,2*pi)
        if freqCurr==freq[0]:
            if phase_scale:
                Qubit['adc phase offset'] = atan2((Q-y0),(I-x0))-phase_mod
            if amp_scale:
                Qubit['adc amp offset'] = abs((I-x0)+1j*(Q-y0))
            else:
                Qubit['adc amp offset'] = 1.0

        amp.append(abs((I-x0)+1j*(Q-y0))/Qubit['adc amp offset'])
        phase.append(mod(atan2((Q-y0),(I-x0))-phase_mod-Qubit['adc phase offset'],2*pi))
        returnValue(phase+amp)
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=False)
    if collect:
        return data


def QubitSpecWithPhotons_old(sample, freq=None, stats=300L, measure=0, sb_freq=0*GHz, detunings=None, uwave_amp=None,qLen=5000*ns,resdrive_amp=0.0,SZ_release=0.0,
                 save=True, name='Spectroscopy MQ', collect=False, noisy=True, update=True):
    """Qubit spectroscopy measured in presence of photons in the resonator"""
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    q['readout'] = True
    if freq is None:
        f = st.nearest(q['f10'][GHz], 0.001)
        freq = st.r[f-0.04:f+0.04:0.001, GHz]
    if uwave_amp is None:
        uwave_amp = q['spectroscopyAmp']
    if detunings is None:
        zpas = [0.0] * len(qubits)
    else:
        zpas = []
        for i, (q, df) in enumerate(zip(qubits, detunings)):
            print 'qubit %d will be detuned by %s' % (i, df)
            zpafunc = get_zpa_func(q)
            zpa = zpafunc(q['f10'] + df)
            zpas.append(zpa)

    axes = [(uwave_amp, 'Microwave Amplitude'), (freq, 'Frequency')]
    deps = [('Probability', '|1>', '')]
    kw = {
        'stats': stats,
        'sideband': sb_freq
    }
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    q['readout length']=5000*ns
    sb_freq_res = q['readout frequency']-q['readout fc']
    q['spectroscopyLen']=qLen

    def func(server, amp, f):
        for i, (q, zpa) in enumerate(zip(qubits, zpas)):
            q['fc'] = f - sb_freq
            if zpa:
                q.z = env.rect(-100, qubits[measure]['spectroscopyLen'] + 100, zpa)
            else:
                q.z = env.NOTHING
            if i == measure:
                q['spectroscopyAmp'] = amp
                q.xy = eh.spectroscopyPulse(q, -(qLen-2500*ns), sb_freq)
                q.z += eh.measurePulse(q, q['spectroscopyLen']+10*ns)
                q['readout amplitude']=resdrive_amp
                q.rr = eh.ResSpectroscopyPulse(q, -2500*ns, sb_freq_res)
                q.cz = env.rect(q['spectroscopyLen'], 10*ns, q['cali_sz'])
        return runQubits(server, qubits, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    if collect:
        return data



def QubitSpecByS11_old(sample, freq=st.r[6.0:6.12:0.0005, GHz], ResAmp=0.001, ResFreq=st.r[6.57:6.565:-0.0005, GHz],QubitAmp=0.02,readLen=5000*ns,sb_freq = -50*MHz,
               measure=0, stats=150,SZ=0.0,x0=0,y0=0,p0=0,p1=0,save=True, name='S parameter scanning', collect=True, noisy=True):
    """measure with s11"""
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Qubit = qubits[measure], Qubits[measure]

    axes = [(freq, 'Frequency'),(ResFreq,'ResFreq')]
    deps = [('S11 amp','no pi pulse',''), ('S11 phase', 'no pi pulse', 'rad'),('S11 amp','pi pulse','rad'), ('S11 phase', 'pi pulse', 'rad')]
            #('Signal to noise ratio', '', '')]

    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)


    def func(server, freqCurr,ResFreqCurr):
        result=[]
        q['biasReadout'] = q['biasOperate']
        q['readout'] = True
        q['readout amplitude'] = ResAmp
        q['adc mode'] = 'demodulate'
        q['readout frequency']=ResFreqCurr
        q['readout length']=readLen
        q['readout fc'] = q['readout frequency'] - sb_freq
        q['adc demod frequency'] = q['readout frequency']-q['readout fc']
        q.rr = eh.ResSpectroscopyPulse(q, 0*ns, sb_freq)

        q['fc'] = 7.5*GHz
        q.xy = env.NOTHING

        data = yield FutureList([runQubits(server, qubits, stats, raw=True)])
        I1 = np.mean(data[0][0][0])
        Q1 = np.mean(data[0][0][1])
        phase_mod = mod(ResFreqCurr['GHz']*p1,2*pi)
        amp1=abs((I1-x0)+1j*(Q1-y0))
        phase1=mod(atan2((Q1-y0),(I1-x0))-phase_mod-Qubit['adc phase offset'],2*pi)

        if noisy: print 'groud:',freqCurr, amp1, phase1

        q['biasReadout'] = q['biasOperate']
        q['readout'] = True
        q['readout amplitude'] = ResAmp
        q['adc mode'] = 'demodulate'
        q['readout frequency']=ResFreqCurr
        q['readout length']=readLen
        q['readout fc'] = q['readout frequency'] - sb_freq
        q['adc demod frequency'] = q['readout frequency']-q['readout fc']
        q.rr = eh.ResSpectroscopyPulse(q, 0*ns, sb_freq)

        q['fc'] = freqCurr
        q['spectroscopyAmp'] = QubitAmp
        q['spectroscopyLen']= readLen
        q.xy = eh.spectroscopyPulse(q, 0, 0)

        data = yield FutureList([runQubits(server, qubits, stats, raw=True)])
        I2 = np.mean(data[0][0][0])
        Q2 = np.mean(data[0][0][1])
        phase_mod = mod(ResFreqCurr['GHz']*p1,2*pi)
        amp2=abs((I2-x0)+1j*(Q2-y0))
        phase2=mod(atan2((Q2-y0),(I2-x0))-phase_mod-Qubit['adc phase offset'],2*pi)

        if noisy: print 'excited:',freqCurr, amp2, phase2


        result.append(amp1)
        result.append(phase1)
        result.append(amp2)
        result.append(phase2)

        returnValue(result)
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=False)
    if collect:
        return data
