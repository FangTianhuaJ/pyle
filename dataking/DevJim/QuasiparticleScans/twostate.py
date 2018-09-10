import numpy as np

from labrad.units import Unit
mK, V, mV, us, ns, GHz, MHz = [Unit(s) for s in ('mK', 'V', 'mV', 'us', 'ns', 'GHz', 'MHz')]

import pyle.envelopes as env
from pyle.pipeline import returnValue, FutureList
from pyle.dataking import envelopehelpers as eh
from pyle.dataking import adjust
from pyle.dataking.fpgaseq import runQubits
from pyle.util import sweeptools as st
from pyle.dataking import utilMultilevels as ml
from pyle.dataking import multiqubitPQ as mq
from pyle.dataking import dephasingSweeps
from pyle.dataking import sweeps
from pyle.dataking import util
from pyle.plotting import dstools
import labrad

def getMixTemp(cxn, sample, measure=0):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    lr = cxn.lakeshore_ruox
    lr.select_device('Vince GPIB Bus - GPIB0::12')
    temp = lr.temperatures()[0][0][mK]
    Q['temperature'] = temp
    return temp


def getFreq(sample, measure=0, state=1):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    fkey = 'f'+str(state)+str(state-1)
    return q[fkey]


def measure2(sample, paramName, stats=3000L, measstate=2, zeroing=st.r[0:1:1], pi=st.r[0:1:1], measure=0, save=True, collect=True, noisy=True, name=''):
    """A single pi-pulse on one qubit, with other qubits also operated."""
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    ml.setMultiKeys(q,max([2,measstate]))

    axes = [(pi, '|1>-to-|2> Pi Pulse'), (zeroing, 'Swap to Resonator')]
    kw = {'stats': stats}
    name='Measure |1) with |'+str(measstate)+')'+name
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, pi, zeroing):
        q.z = env.NOTHING
        if zeroing==0:
            startPi = q['piFWHM']
        elif zeroing==1:
            startPi = q[paramName+'SwapTime']+q['piFWHM']
            q.z +=  env.rect(0, q[paramName+'SwapTime'], q[paramName+'SwapAmp'])
        if pi:
            q.xy = eh.mix(q, eh.piPulse(q, startPi, state=2), state=2)
        q.z += eh.measurePulse(q, startPi+2*q['piFWHM'], state=measstate)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if collect:
        return data


def tuneup(cxn, sample, target=0.05, measure=0, maxiter=10, noisy=False):

    # Set squid heating values
    getMixTemp(cxn, sample, measure=measure)
    mq.find_mpa(sample,stats=300,state=1,resolution=0.005,target=target,measure=measure,noisy=noisy)
    mq.spectroscopy_two_state(sample, measure=measure, noisy=noisy)

    #Tune up |1) state: MPA, Pi-Pulse, Frequency
    freq1=[getFreq(sample, measure=measure, state=1)]
    freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
    mq.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
    freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
    mq.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
    count = 2
    while (count<maxiter) and ((abs(freq1[-1]-freq1[-2])>0.0006*GHz) or (abs(freq1[-2]-freq1[-3])>0.0006*GHz)):
        # Iteratively optimize Pi-Pulse and Frequency
        print ('f01: '+str([freq['GHz'] for freq in freq1]))
        freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
        mq.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
        count = count+1
    print ('f01: '+str([freq['GHz'] for freq in freq1]))
    mpa1 = mq.find_final_mpa(sample,stats=300,state=1,resolution=0.005,target=target,measure=measure,noisy=noisy)

    #Tune up |2) state: MPA, Pi-Pulse, Frequency
    mq.find_mpa(sample,stats=300,state=2,resolution=0.005,target=target,measure=measure,noisy=noisy)
    freq2=[getFreq(sample, measure=measure, state=2)]
    freq2.append(mq.freqtuner(sample,state=2,measure=measure,noisy=noisy))
    mq.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
    freq2.append(mq.freqtuner(sample,state=2,measure=measure,noisy=noisy))
    mq.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
    count = 2
    while (count<maxiter) and ((abs(freq2[-1]-freq2[-2])>0.0006*GHz) or (abs(freq2[-2]-freq2[-3])>0.0006*GHz)):
        # Iteratively optimize Pi-Pulse and Frequency
        print ('f12: '+str([freq['GHz'] for freq in freq2]))
        freq2.append(mq.freqtuner(sample,state=2,measure=measure,noisy=noisy))
        mq.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
        count = count+1
    print ('f12: '+str([freq['GHz'] for freq in freq2]))
    mpa2 = mq.find_final_mpa(sample,stats=300,state=2,resolution=0.005,target=target,measure=measure,noisy=noisy)

    # Get swap spectroscopy and calibration functions
    freqRange = st.r[(freq2[-1]['GHz']-.05):(freq1[-1]['GHz']+.05):.005,GHz]
    mq.find_zpa_func(sample, stats=120L, freqScan=freqRange, measure=measure, noisy=noisy)
    zpa1 = mq.freq2zpa(sample,0,freq1[-1])
    zpa2 = mq.freq2zpa(sample,0,freq2[-1])
    mq.swapSpectroscopy(sample, swapLen=st.arangePQ(10,500,10,ns), swapAmp=np.arange((min(zpa1,zpa2)-0.02),(max(zpa1,zpa2)+0.02),0.002), measure=measure, noisy=noisy)
    mq.find_mpa_func(sample, measure=measure, noisy=noisy)
    mq.find_flux_func(sample, stats=120L, freqScan=freqRange, measure=measure, noisy=noisy)

    # Measure T1, T2, Visibility
    mq.t1(sample,delay=st.r[20:3000:20,ns],stats=3000L,measure=measure,noisy=noisy, plot=False)
    dephasingSweeps.ramsey(sample, delay=st.r[0:300:1, ns], plot=False, update=False, measure=measure, noisy=noisy)
    mq.visibility(sample, states=[1,2], mpa=st.r[max(mpa2-0.15,0):min(mpa1+0.1,2):0.001], calstats=12000, update=False, measure=measure, noisy=noisy)

    # Tune up swap to TLS
    swapLen, swapAmp = mq.swap10tuner(sample, noisy=noisy, measure=measure)
    mq.swapSpectroscopy(sample, state=1, swapLen=st.arangePQ(0,2.5*swapLen,2,ns), swapAmp=np.arange(swapAmp-0.01,swapAmp+0.01,0.0005), measure=measure, noisy=noisy)
    mq.swap10tuner(sample, stats=1800L, noisy=noisy, measure=measure)

    # Measure P1
    measure2(sample,measstate=1,stats=18000L,measure=measure,noisy=noisy)
    measure2(sample,measstate=2,stats=18000L,measure=measure,noisy=noisy)

def squidcutoff(sample, resets=st.r[-2.5:2.5:5,V], measure=0, stats=3000,
               save=False, name='Squid Heating Steps', collect=False, update=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    axes = [(resets, 'Squid Reset Bias')]
    deps = [('Switching Time','', us)]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, reset):
        reqs=[]
        q['biasReset'] = [reset]
        q['readout'] = True
        reqs.append(runQubits(server, qubits, stats, dataFormat='raw_microseconds'))
        data = yield FutureList(reqs)
        returnValue(np.vstack(data).T)
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=False)

    if update: #Find time where Gaussians representing R-well and L-well intersect.
        flux=data[:,0]
        times=data[:,1]
        meantime = np.mean(times)
        fluxvals=np.unique(flux)
        lowtimes=times[flux==fluxvals[0]]
        lm=np.mean(lowtimes)
        ls=np.std(lowtimes)
        hightimes=times[flux==fluxvals[1]]
        hm=np.mean(hightimes)
        hs=np.std(hightimes)
#        if abs(ls-hs)<0.05: cutoff=meantime
#        else:
#            cutoffvals0=np.array([-1,1])
#            cutoffvals=(-hs**2*lm+ls**2*hm+cutoffvals0*hs*ls*np.sqrt((lm-hm)**2+2*(ls**2-hs**2)*(np.log(ls)-np.log(hs))))/(ls**2-hs**2)
#            diffs=cutoffvals-meantime
#            cutoff=meantime+float(diffs[abs(diffs)==np.min(abs(diffs))])
#            Q['squidSwitchIntervals']=[(cutoff*us,100*us)]
        Q['squidSwitchIntervals']=[((0.5*(lm+ls+hm-hs))*us,100*us)]
    if collect:
        return data

def adjust_squid_heat_cutoff(qubit, data):
    heat, low, high = data.T
    traces = [{'x': heat, 'y': low, 'args': ('b.',)},
              {'x': heat, 'y': high, 'args': ('r.',)}]
    params = [{'name': 'timing0', 'val': qubit['squidSwitchIntervals'][0][0][us], 'range': (0,40), 'axis': 'y', 'color': 'k'},
              {'name': 'timing1', 'val': qubit['squidSwitchIntervals'][0][1][us], 'range': (0,40), 'axis': 'y', 'color': 'gray'}]
    result = adjust.adjust(params, traces)
    if result is not None:
        qubit['squidSwitchIntervals'] = [(result['timing0']*us, result['timing1']*us)]


def squidheatingsteps(sample, squidheat=st.r[-2.5:2.5:0.05, V], resets=(-2.5*V, 2.5*V), measure=0, stats=1200,
               save=True, name='Squid Heating Steps', collect=False, noisy=False, update=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    axes = [(squidheat, 'Squid Heating Bias')]
    deps = [('Switching Time', 'Reset: %s' % (reset,), us) for reset in resets]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, heat):
        reqs = []
        for reset in resets:
            q['squidheatBias'] = heat
            q['biasReset'] = [reset]
            q['readout'] = True
            reqs.append(runQubits(server, qubits, stats, dataFormat='raw_microseconds'))
        data = yield FutureList(reqs)
        if noisy: print fb
        returnValue(np.vstack(data).T)
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=False)

    if update:
        adjust_squid_heat_cutoff(Q, data)
    if collect:
        return data
