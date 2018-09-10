'''
Created on May 27, 2010
author: Daniel Sank
'''

#CHANGELOG


import math
import time

import numpy as np
import matplotlib.pyplot as plt

from labrad.units import Unit,Value
V, mV, us, ns, GHz, MHz, sec = [Unit(s) for s in ('V', 'mV', 'us', 'ns', 'GHz', 'MHz', 's')]

from pyle import envelopes as env
from pyle.dataking import envelopehelpers as eh
from pyle.dataking import util
from pyle.pipeline import returnValue, FutureList
from pyle.util import sweeptools as st
from pyle.dataking import sweeps
from pyle.dataking.fpgaseq import runInterlacedSRAM, runQubits
from pyle.plotting import dstools
from pyle.analysis import signalProcessing as sp
from pyle.analysis import FluxNoiseAnalysis as fna
from pyle.dataking import utilMultilevels as ml

import labrad

def do_queue(s, measure):
    """Use this function to queue up multiple experiments"""
    rabilong_averaged(s, length=st.r[0:1200:1,ns], measure=measure, spread=0*ns, amplitude=0.6, stats=300, averages=20, check2State=False)
    rabilong_averaged(s, length=st.r[0:1200:1,ns], measure=measure, spread=0*ns, amplitude=0.7, stats=300, averages=20, check2State=False)
    rabilong_averaged(s, length=st.r[0:1200:1,ns], measure=measure, spread=0*ns, amplitude=0.8, stats=300, averages=20, check2State=False)
    rabilong_averaged(s, length=st.r[0:1200:1,ns], measure=measure, spread=0*ns, amplitude=0.9, stats=300, averages=20, check2State=False)

def ramsey(sample, measure=0, delay=st.r[0:200:1,ns], fringeFreq = 50*MHz,
           stats=600L, name='Ramsey MQ', save = True, noisy=True,
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
    swap = q['cZControlLen']+q['piLen']/2.0
    timeUpstate = swap+(state-1)*q['piLen']
    dt = 2.0*q['piFWHM']

    #Pump pulse is at time=0 (after the time to reach the initial state) with phase=0
    pump = eh.piHalfPulse(q, timeUpstate, phase=0.0, state=state)
    #Probe is at variable time with variable phase
    def probe(time, tomoPhase):
        return eh.piHalfPulse(q, timeUpstate+time, phase = 2*np.pi*(fringeFreq['GHz']*time['ns']+tomoPhases[tomoPhase]), state=state)
    
    def func(server, delay):
        reqs = []
        for tomoPhase in tomoPhases.keys():
            q.xy = eh.boostState(q, swap, state-1) + eh.mix(q, pump + probe(delay, tomoPhase = tomoPhase), state=state)
            q.z = env.rect(0, q['cZControlLen'], q['cZControlAmp'])+eh.measurePulse(q,timeUpstate+dt+delay, state=state)
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

def spinEcho(sample, measure=0, delay=st.r[0:1000:10,ns], df=50*MHz,
                   stats=300L, name='Spin Echo MQ', save=True,
                   collect=True, noisy=True, randomize=False, averages=1,
                   tomo=True):
    """Spin echo sequence on one qubit. Can be single phase or 4-axis tomo, and
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
 
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    q['readout']=True
    
    #Randomize time axis
    if randomize:
        delay=st.shuffle(delay)
    def delayGen():
        for _ in range(averages):
            for d in delay:
                yield d
                
    axes = [(delayGen(), 'Delay')]
    if tomo:
        deps = [('Probability', '+X', ''), ('Probability', '+Y', ''),
                ('Probability', '-X', ''), ('Probability', '-Y', '')]
        tomoPhases = {'+X': 0.0, '+Y':0.25, '-X': -0.5, '-Y':-0.25}
    else:
        deps = [('Probability','','')]
        tomoPhases={'+X':0.0}
        
    kw={'averages':averages, 'stats':stats, 'fringeFrequency':df}
    dataset=sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)
    
    #Pump pulse is at time=0 with phase=0
    pump = eh.piHalfPulse(q, 0, phase=0.0)
    #Probe is at variable time with variable phase
    def probe(time, tomoPhase):
        return eh.piHalfPulse(q, time, phase=2.0*np.pi*tomoPhases[tomoPhase])

    def func(server, delay):
        reqs=[]
        dt=q['piLen']
        tpi = dt/2.0 + delay/2.0
        tProbe = dt/2.0 + delay + dt/2.0
        tMeas = tProbe + dt/2.0
        piPhase = 2*np.pi*df[GHz]*delay[ns]/2.0
        for tomoPhase in tomoPhases.keys():
            q.xy = eh.mix(q, pump +
                            eh.piPulse(q, tpi, phase=piPhase) + 
                            probe(tProbe, tomoPhase))
            q.z = eh.measurePulse(q, tMeas)
            reqs.append(runQubits(server, qubits, stats, probs=[1]))
        probs = yield FutureList(reqs)
        data = [p[0] for p in probs]
        returnValue(data)
    return sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)

####################################################
#SEQUENCES BELOW HERE HAVE NOT BEEN COMITTED TO PYLE
####################################################

def rabilong_averaged(sample, length=st.r[0:1200:2,ns], spread=0*ns, amplitude=None, detuning=None,
                      measureDelay=None, randomize = False, measure=0, stats=300L,
                      name='Rabi DP', save=True, collect=False, noisy=True, useHd=True,
                      averages=1, check2State=False, turnOnWidth=None):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    
    if amplitude is None: amplitude = q['piAmp']
    if detuning is None: detuning = 0
    if turnOnWidth is None:
        turnOnWidth = q['piFWHM']
        measureDelay = q['piLen']/2.0
    else:
        measureDelay=turnOnWidth
    if randomize:
        length = st.shuffle(length)

        
    if spread['ns']>0:
        def length_gen():
            for _ in range(averages):
                for l in length:
                    for dt in st.r[-spread.value:spread.value:1,spread.units]:
                        yield l+dt
    else:
        def length_gen():
            for _ in range(averages):
                for l in length:
                    yield l
    
    axes = [(length_gen(), 'pulse length'),
            (amplitude, 'pulse height'),
            (measureDelay, 'measure delay')]
    kw = {'stats': stats, 'amplitude':amplitude, 'averages': averages}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)
    
    def func(server, len, amp, delay):
        if useHd:
            q.xy = eh.mix(q, eh.rabiPulseHD(q, 0, len, amp=amp, width=turnOnWidth))
        else:
            q.xy = eh.mix(q, env.flattop(0, len, w=turnOnWidth, amp=amp))
        if not check2State:
            q.z = eh.measurePulse(q, delay+len)
        else:
            q.z = eh.measurePulse2(q, delay+len)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)

def rabiT1(sample, measure, amplitude, turnOnWidth=None, length=st.r[0:1200:2,ns], spread=0*ns, stats=300L, name='RabiT1', save=True,
           collect=False, noisy=True, averages=1, randomize=False):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    if turnOnWidth is None:
        turnOnWidth = q['piFWHM']
        measureDelay=q['piLen']/2.0
    else:
        measureDelay=turnOnWidth
    if randomize:
        length=st.shuffle(length)
    if spread['ns']>0:
        def length_gen():
            for _ in range(averages):
                for l in length:
                    for dt in st.r[-spread.value:spread.value:1,spread.units]:
                        yield l+dt
    else:
        def length_gen():
            for _ in range(averages):
                for l in length:
                    yield l
    axes = [(length_gen(), 'pulse length'),
            (amplitude, 'pulse height')]
    deps = [('Probability', 'Rabi', ''),
            ('Probability', 'T1', '')]
    kw={'stats':stats, 'amplitude':amplitude, 'averages':averages}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)
    
    def func(server, len, amp):
        q.xy = eh.mix(q, eh.rabiPulseHD(q, 0, len, amp=amp, width=turnOnWidth))
        q.z = eh.measurePulse(q, measureDelay+len)
        q['readout']=True
        reqRabi = runQubits(server, qubits, stats, probs=[1])
        
        q.xy = eh.mix(q, eh.piPulse(q,0))
        q.z = eh.measurePulse(q, len)
        q['readout']=True
        reqT1 = runQubits(server, qubits, stats, probs=[1])
        
        probs = yield FutureList([reqRabi, reqT1])
        pRabi, pT1 = [p[0] for p in probs]
        returnValue([pRabi,pT1])
    data = sweeps.grid(func, axes, dataset = save and dataset, noisy=noisy)
    if collect:
        return data

def rabiT1_v1(sample, measure, amplitude, turnOnWidth=None, length=st.r[0:1200:2,ns], spread=0*ns, stats=300L, name='RabiT1', save=True,
           collect=False, noisy=True, averages=1, randomize=False):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    if turnOnWidth is None:
        turnOnWidth = q['piFWHM']
        measureDelay=q['piLen']/2.0
    else:
        measureDelay=turnOnWidth
    if randomize:
        length=st.shuffle(length)
    if spread['ns']>0:
        def length_gen():
            for l in length:
                for dt in st.r[-spread.value:spread.value:1,spread.units]:
                    yield l+dt
    else:
        def length_gen():
            for l in length:
                yield l
    def iterations():
        for iter in range(averages):
            yield iter
    axes = [(iterations(), 'iteration'),
            (length, 'pulse length'),
            (amplitude, 'pulse height')]
    deps = [('Probability', 'Rabi', ''),
            ('Probability', 'T1', '')]
    kw={'stats':stats, 'amplitude':amplitude, 'averages':averages}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)
    
    def func(server, len, iter, amp):
        q.xy = eh.mix(q, eh.rabiPulseHD(q, 0, len, amp=amp, width=turnOnWidth))
        q.z = eh.measurePulse(q, measureDelay+len)
        q['readout']=True
        reqRabi = runQubits(server, qubits, stats, probs=[1])
        
        q.xy = eh.mix(q, eh.piPulse(q,0))
        q.z = eh.measurePulse(q, len)
        q['readout']=True
        reqT1 = runQubits(server, qubits, stats, probs=[1])
        
        probs = yield FutureList([reqRabi, reqT1])
        pRabi, pT1 = [p[0] for p in probs]
        returnValue([pRabi,pT1])
    data = sweeps.grid(func, axes, dataset = save and dataset, noisy=noisy)
    if collect:
        return data
    
RABI_PARAMETERS = [(0.2, 8*ns),
                   (0.3, 6*ns),
                   (0.4, 5*ns),
                   (0.5, 5*ns),
                   (0.6, 4*ns),
                   (0.7, 3*ns)
                   ]

def rabiAuto(sample, measure=0, parameters=RABI_PARAMETERS, randomize=False, stats=300, averages=10):
    for amp, spread in parameters:
        #Get frequency at this amplitude
        result = rabilong_averaged(sample, length=st.r[0:700:1,ns], spread=0*ns, amplitude=amp, detuning=None,
                                   measure=measure, stats=300, save=False, collect=True, averages=1)
        #Compute Fourier transform and find the frequency
        S = sp.DFT(result[:,1]-np.mean(result[:,1]),result[-1,0])
        index = np.argmax(S.S)
        freq = S.frequencies[index] #GHz, because times are in nanoseconds
        print 'Frequency in GHz: ',freq
        dt = (1.0/freq)/2.0 # Divide by two because you want to measure the max and min of the oscillations
        length = st.r[0:1200:dt,ns]
        rabilong_averaged(sample, length=length, spread=spread, amplitude=amp, measure=measure, stats=stats, save=True, averages=averages)
        
def rabi_set(s, measure=0, parameters=RABI_PARAMETERS, randomize=False, stats=300, averages=50):
    periods=[]
    for amp, spread in parameters:
        result = rabilong_averaged(s, measure=measure, length=st.r[0:1200:2,ns], spread=0*ns, amplitude=amp, stats=300, save=False, collect=True, averages=1)
        plt.figure()
        plt.plot(result[:,0],result[:,1],'.')
        period = float(raw_input('period [ns]: '))*ns
        periods.append(period)
    amps, spreads = zip(*parameters)
    for amp, spread, period in zip(amps, spreads, periods):
        length=st.r[0:1200:period.value/2.0,ns]
        rabilong_averaged(s, measure=measure, length=length, spread=spread, amplitude = amp, stats=stats,
                          averages=averages, randomize=randomize)

def ramsey_oscilloscope(sample, measure=0, holdTime = 100*ns, fringeFreq = 50*MHz, timeStep = 1*sec,
                        stats=600, name='RamseyScope', save = True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    
    #Set up datavault
    axes = [('iteration','')]
    deps = [('Probability', '+X', ''),('Probability', '+Y', ''),
            ('Probability', '-X', ''),('Probability', '-Y', ''),('time', 't', 'sec')]
    kw={'stats': stats,'holdTime': holdTime,'fringeFrequency': fringeFreq,'timeStep':timeStep, 'zAttenuator': q['zAttenuator'],
        'fluxDivider': q['fluxDivider']}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)
    def iterations():
        i = 0
        while True:
            yield i
            i += 1
    tomoPhases = {'+X': 0.0, '+Y': 0.25, '-X': -0.5, '-Y': -0.25} #[+X, +Y, -X, -Y] in CYCLES
    pump = eh.piHalfPulseHD(q,0, phase=0.0)
    def probe(time, tomoPhase):
        return eh.piHalfPulse(q, time, phase = 2.0*np.pi*(fringeFreq['GHz']*time['ns']+tomoPhases[tomoPhase]))
    start = [None]
    skips = [0] # number of data points that have been skipped
    waits = [0, 0]
    ts = timeStep['s']
    def func(server, iteration):
        q.xy = []
        q.z = []
        for tomoPhase in ['+X','+Y','-X','-Y']:
            q.xy.append(eh.mix(q, pump + probe(holdTime, tomoPhase=tomoPhase)))
            q.z.append(eh.measurePulse(q, holdTime+20*ns))
        q['readout'] = True
        
        #Arrange timing
        iteration += skips[0]
        if start[0] is None:
            start[0] = time.clock()
            nextTime = 0
        else:
            elapsed = time.clock() - start[0]
            nextIteration = int(math.ceil(elapsed / ts))
            if nextIteration > iteration + 1: #Should be nextIteration > iteration + 1
                # oops, we missed some datapoints :-(
                skips[0] += nextIteration - (iteration + 1)
                print 'skip!  skips = %d/%d (%g%%)' % (skips[0], iteration, 100*skips[0]/iteration)
            #Hang out until it's time to fire off the next iteration
            nextTime = nextIteration * ts
            wait = nextTime - (time.clock() - start[0])
            if wait > 0:
                waits[0] += wait
                waits[1] += 1
                avg = waits[0] / waits[1]
                pct = avg / ts
                print 'average wait: %g = %g%% of timeStep' % (avg, 100*pct)
                time.sleep(wait)

        pX, pY, pmX, pmY = yield runInterlacedSRAM(server, qubits, stats, probs=[1])
        data = [iteration] + [pX[0], pY[0], pmX[0], pmY[0]] + [nextTime]
        returnValue(data)
    sweeps.run(func, iterations(), save, dataset, pipesize=2)

def ramsey_oscilloscope_hack(sample, measure=0, holdTime = 100*ns, fringeFreq = 50*MHz, timeStep = 1*sec,
                             stats=600, name='RamseyScope', save = True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    
    #Set up datavault
    axes = [('iteration','')]
    deps = [('Probability', '+X', ''),('Probability', '+Y', ''),
            ('Probability', '-X', ''),('Probability', '-Y', ''),('time', 't', 'sec')]
    kw={'stats': stats,'holdTime': holdTime,'fringeFrequency': fringeFreq,'timeStep':timeStep, 'zAttenuator': q['zAttenuator'],
        'fluxDivider': q['fluxDivider']}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)
    def iterations():
        i = 0
        while True:
            yield i
            i += 1
    tomoPhases = {'+X': 0.0, '+Y': 0.25, '-X': -0.5, '-Y': -0.25} #[+X, +Y, -X, -Y] in CYCLES
    pump = eh.piHalfPulseHD(q,0, phase=0.0)
    def probe(time, tomoPhase):
        return eh.piHalfPulse(q, time, phase = 2.0*np.pi*(fringeFreq['GHz']*time['ns']+tomoPhases[tomoPhase]))
    start = [None]
    skips = [0] # number of data points that have been skipped
    waits = [0, 0]
    ts = timeStep['s']
    def func(server, iteration):
        reqs=[]
        for tomoPhase in ['+X','+Y','-X','-Y']:
            q.xy = eh.mix(q, pump + probe(holdTime, tomoPhase=tomoPhase))
            q.z = eh.measurePulse(q, holdTime+20*ns)
            q['readout']=True
            reqs.append(runQubits(server, qubits, stats, probs=[1]))
        #Arrange timing
        iteration += skips[0]
        if start[0] is None:
            start[0] = time.clock()
            nextTime = 0
        else:
            elapsed = time.clock() - start[0]
            nextIteration = int(math.ceil(elapsed / ts))
            if nextIteration > iteration + 1: #Should be nextIteration > iteration + 1
                # oops, we missed some datapoints :-(
                skips[0] += nextIteration - (iteration + 1)
                print 'skip!  skips = %d/%d (%g%%)' % (skips[0], iteration, 100*skips[0]/iteration)
            #Hang out until it's time to fire off the next iteration
            nextTime = nextIteration * ts
            wait = nextTime - (time.clock() - start[0])
            if wait > 0:
                waits[0] += wait
                waits[1] += 1
                avg = waits[0] / waits[1]
                pct = avg / ts
                print 'average wait: %g = %g%% of timeStep' % (avg, 100*pct)
                time.sleep(wait)
        probs = yield FutureList(reqs)
        data = [p[0] for p in probs]
        data = [iteration] + data + [nextTime]
        returnValue(data)
    sweeps.run(func, iterations(), save, dataset, pipesize=2)

#################################################################################################
#################################################################################################
###THINGS BELOW THIS LINE NEED TO BE UPDATED FOR THE LATEST REVISIONS IN PYLE
def ramsey_correlator(sample, holdTime = 100*ns, fringeFreq = 50*MHz, timeStep = 1*sec,
                        fluxDivider = 0, zAttenuator=0, stats=600, measure=None, name='RamseyCorrelate', save = True):
    if measure is None:
        raise Exception('Choose qubits to correlate')
    
    sample, qubits = util.loadQubits(sample)
    for q in qubits:
        q._readout = True

    def iterations():
        i=0
        while True:
            yield i
            i+=1
    
    params = [('iteration', '')]
    dependents = [('Probability', '+X0', ''),('Probability', '+Y0', ''),
                  ('Probability', '-X0', ''),('Probability', '-Y0', ''),
                  ('Probability', '+X1', ''),('Probability', '+Y1', ''),
                  ('Probability', '-X1', ''),('Probability', '-Y1', ''),('time', 't', 'sec')]
    ds_info = build_info(sample, name, params, dependents=dependents, kw={'stats': stats,'holdTime': holdTime,
                                                                          'fringeFrequency': fringeFreq,
                                                                          'timeStep':timeStep, 'zAttenuator': zAttenuator,
                                                                          'fluxDivider': fluxDivider})
    tomoPhases = {'+X': 0.0, '+Y': 0.25, '-X': -0.5, '-Y': -0.25} #[+X, +Y, -X, -Y] in CYCLES
    
    def pump(qubit):
        return pi_half_pulse(qubit,0,phase=0.0)
    def probe(qubit, time, tomoPhase):
        return pi_half_pulse(qubit, time, phase = 2*pi*(fringeFreq['GHz']*time['ns']+tomoPhases[tomoPhase]))
    
    ts = timeStep['s']
    
    def func(iteration):
        seqs = []
        for tomoPhase in ['+X','+Y','-X','-Y']: #Run through tomo axes
            for m in measure: #Set up the sequence for each qubit for this tomography axis
                qubits[m]._xy = mix(qubits[m], pump(qubits[m]) + probe(qubits[m], holdTime, tomoPhase = tomoPhase))
                qubits[m]._z  = measure_pulse(qubits[m],holdTime+20*ns)
            seqs += [Seq(qubits,stats,separate=True)] #Add that axis to the list of sequences to run
        #Timing. Wait until next sequence should be run.
        nextIteration = iteration+1
        while time.clock()>=nextIteration*ts:
            nextIteration+=2
        #Hang out until it's time to fire off the next iteration
        t = time.clock()
        nextTime = nextIteration * ts
        time.sleep(nextTime - t)

        tomoAxes = yield seqs  #tomoAxes is a list of four lists, one for each tomo axis (ie. sequence)
                            #Each of these four lists has three elements [Pq0, Pq1, Pq2], for the prob that each qubit
                            #switched (ie. was in the |1> state).
        data = [iteration] + [tomoAxes[axis][qubit] for qubit in measure for axis in range(4)] + [nextTime]
        #data = [iteration] + [p[0] for p in probs]+[nextTime]
        returnValue(data)
    run(func, iterations(), save, ds_info, pipesize=2)

def ramsey_correlator_shuffled(sample, holdTime = 100*ns, fringeFreq = 50*MHz, timeStep = 1*sec,
                        fluxDivider = 0, zAttenuator=0, stats=600, measure=None, name='RamseyCorrelateShuffled', save = True):
    if measure is None:
        raise Exception('Choose qubits to correlate')
    M = len(measure)
    sample, qubits = util.loadQubits(sample)
    for m in measure:
        qubits[m]._readout = True
    def iterations():
        i=0
        while True:
            yield i
            i+=1
    params = [('iteration', '')]
    dependents = [('Probability', '+X0', ''),('Probability', '+Y0', ''),
                  ('Probability', '-X0', ''),('Probability', '-Y0', ''),
                  ('Probability', '+X1', ''),('Probability', '+Y1', ''),
                  ('Probability', '-X1', ''),('Probability', '-Y1', ''),('time', 't', 'sec')]
    ds_info = build_info(sample, name, params, dependents=dependents, kw={'stats': stats,'holdTime': holdTime,
                                                                          'fringeFrequency': fringeFreq,
                                                                          'timeStep':timeStep, 'zAttenuator': zAttenuator,
                                                                          'fluxDivider': fluxDivider})
    tomoPhases = {'+X': 0.0, '+Y': 0.25, '-X': -0.5, '-Y': -0.25} #[+X, +Y, -X, -Y] in CYCLES
    def pump(qubit):
        return pi_half_pulse(qubit,0,phase=0.0)
    def probe(qubit, time, tomoPhase):
        return pi_half_pulse(qubit, time, phase = 2*pi*(fringeFreq['GHz']*time['ns']+tomoPhases[tomoPhase]))
    
    ts = timeStep['s']
    
    def func(iteration):
        seqs = []
        qubitPhases = [shuffleable(['+X','+Y','-X','-Y']) for i in range(M)]
        for i in range(M): #Shuffle the order of the tomo phases
            qubitPhases[i].shuffle()
        for tomoPhase in range(4):
            for i,m in enumerate(measure):
                qubits[m]._xy = mix(qubits[m], pump(qubits[m]) + probe(qubits[m], holdTime, tomoPhase = qubitPhases[i].table[tomoPhase][1]))
                qubits[m]._z  = measure_pulse(qubits[m],holdTime+20*ns)
            seqs += [Seq(qubits,stats,separate=True)]
        
        ###TIMING. Wait until next sequence should be run.#################
        nextIteration = iteration+1
        while time.clock()>=nextIteration*ts:
            nextIteration+=2
        #Hang out until it's time to fire off the next iteration
        t = time.clock()
        nextTime = nextIteration * ts
        time.sleep(nextTime - t)
        ###END TIMING#######################################################

        response = yield seqs   #Returns a list of 4 numpy arrays, one array for each tomo axis. Each array has two elements, one for each
                                #qubit measured
        data=[]
        #Sort data
        for i,m in enumerate(measure):
            L = zip([qubitPhases[i].table[j][0] for j in range(4)],[response[j][i] for j in range(4)])
            L.sort()
            L = [L[j][1] for j in range(4)]
            data.extend(L)
        data = [iteration] + data + [nextTime]
        returnValue(data)

    run(func, iterations(), save, ds_info, pipesize=2)



_tomoOps = [
    ('I', 0, 0),
    ('+X', 0.5, 0),
    ('+Y', 0.5, 0.5),
    ('-X', -0.5, 0),
    ('-Y', -0.5, 0.5),
    ('pi', 1, 0)
]
