import numpy as np
from scipy.optimize import leastsq, curve_fit
from scipy.special import erf, erfc
import matplotlib.pyplot as plt

from labrad.units import Unit,Value
from pyle import tomo, fidelity

import pyle.envelopes as env
from pyle.pipeline import returnValue, FutureList
from pyle.dataking import envelopehelpers as eh
from pyle.dataking import measurement
from pyle.dataking import adjust
from pyle.dataking.fpgaseq import runQubits as runQubits
from pyle.dataking import dephasingSweeps
from pyle.util import sweeptools as st
from math import atan2
from pyle.dataking import utilMultilevels as ml
from pyle.plotting import dstools
from pyle.fitting import fitting
from pyle.util import structures

import scipy.optimize as opt

import qubitpulsecal as qpc
import sweeps
import util
import labrad

V, mV, us, ns, GHz, MHz, dBm, rad = [Unit(s) for s in ('V', 'mV', 'us', 'ns', 'GHz', 'MHz', 'dBm', 'rad')]

def squidsteps(sample, measure=0, stats=150,
               bias=st.r[-2.5:2.5:0.05, V], resets=(-2.5*V, 2.5*V),
               save=True, name='SquidSteps MQ', collect=False, noisy=False,
               update=True, dataFormat=None):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    if 'squidBiasLimits' in q:
        default = (-2.5*V, 2.5*V)
        bias_lim = q['squidBiasLimits']
        if bias_lim != default:
            print 'limiting bias range to (%s, %s)' % tuple(bias_lim)
        resets = max(resets[0], bias_lim[0]), min(resets[1], bias_lim[1])
        bias = st.r[bias_lim[0]:bias_lim[1]:bias.range.step, V]

    axes = [(bias, 'Flux Bias')]
    deps = [('Switching Time', 'Reset: %s' % (reset,), us) for reset in resets]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)
    if dataFormat is None:
        dataFormat = 'raw_microseconds'

    def func(server, fb):
        reqs = []
        for reset in resets:
            q['biasOperate'] = fb
            q['biasReadout'] = fb
            q['biasReset'] = [reset]
            q['readout'] = True
            reqs.append(runQubits(server, qubits, stats, dataFormat=dataFormat))
        data = yield FutureList(reqs)
        if noisy: print fb
        returnValue(np.vstack(data).T)
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=False)

    if update:
        adjust.adjust_squid_steps(Q, data)
    if collect:
        return data


def s_scanning(sample, freq=None, power=None, reset = None, bias=0*V, sb_freq = -50*MHz, measure=0, stats=150,
               save=True, name='S parameter scanning', collect=False, noisy=True, update=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    if freq is None:
        f = st.nearest(q['readout frequency'][GHz], 1e-5)
        freq = st.r[f-0.001:f+0.001:1e-5, GHz]
    axes = [(freq, 'Frequency')]
    deps = [('Phase', 'S11 for %s'%q.__name__, rad), ('Amplitude','S11 for %s'%q.__name__,'')]
    kw = {'stats': stats, 'power':power}
    if reset is not None:
        name = name + 'bias:' + str(bias) + 'reset:' + str(reset)
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    for qubit in qubits:
        qubit['biasOperate'] = bias
        qubit['biasReadout'] = bias
        if reset is not None:
            qubit['biasReset'] = reset
        else:
            qubit['biasResetSettling']=5*us
            qubit['biasOperateSettling']=5*us
            qubit['biasReadoutSettling']=10*us
            qubit['biasReset'] = bias

    q['readout'] = True
    if power is not None:
        q['readout power'] = power

    def func(server, f):
        for qubit in qubits:
            qubit['readout frequency']=f
            qubit['readout fc'] = qubit['readout frequency'] - sb_freq
        if noisy: print f
        data = yield FutureList([runQubits(server, qubits, stats, dataFormat='iq')])
        I = np.mean(data[0][0][0])
        Q = np.mean(data[0][0][1])
        amp = abs(I+1j*Q)
        phase = (atan2(Q,I))
        returnValue([phase, amp])

    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=False)
    if update:
        phase = np.asarray([[row[0], row[1]] for row in data])
        adjust.adjust_s_scanning(Q, phase)
    if collect:
        return data

def readoutIq(sample, freq=None, power=None, reset = None, bias=0*V, sb_freq=-50.0*MHz, measure=0, stats=150,
              save=True, name='Readout IQ', collect=False, noisy=True, plot=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    # this allows us to look at the resonance circle in the IQ plane
    # update readoutIqOffset with the additive translation to move circle to the origin

    # since s_scanning operates at -50*MHz sideband,
    # after this scan, set readout fc to be 50*MHz above readout frequency

    if freq is None:
        f = st.nearest(q['readout frequency'][GHz], 1e-5)
        freq = st.r[f-0.001:f+0.001:1e-5, GHz]
    axes = [(freq, 'Frequency')]
    deps = [('Clicks', 'I for %s' %q.__name__, ''), ('Clicks', 'Q for %s' %q.__name__, '')]
    kw = {'stats': stats, 'power':power}
    if reset is not None:
        name = name + 'bias:' + str(bias) + 'reset:' + str(reset)
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    for qubit in qubits:
        qubit['biasOperate'] = bias
        qubit['biasReadout'] = bias
        if reset is not None:
            qubit['biasReset'] = reset
        else:
            #qubit['biasResetSettling']=5*us
            #qubit['biasOperateSettling']=5*us
            #qubit['biasReadoutSettling']=10*us
            qubit['biasReset'] = bias

    q['readout'] = True
    if power is not None:
        q['readout power'] = power

    def func(server, f):
        for qubit in qubits:
            qubit['readout frequency']=f
            qubit['readout fc'] = qubit['readout frequency'] - sb_freq
        if noisy: print f
        data = yield FutureList([runQubits(server, qubits, stats, dataFormat='iq')])
        I = np.mean(data[0][0][0])
        Q = np.mean(data[0][0][1])
        returnValue([I,Q])

    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=False)
    if plot:
        plt.figure()
        plt.plot(data[:,1],data[:,2],'.')
        plt.grid()
        for i,freq in enumerate(data[:,0]):
            plt.text(data[i,1],data[i,2],str(data[i,0]))
        plt.xlabel('I')
        plt.ylabel('Q')
    if collect:
        return data

def phase_arc(sample, bias=None, resets=(-2.5*V, 2.5*V), power=None, measure=0, stats=None,
               save=True, name='Phase Arc', collect=False, noisy=True, update=True, rough=False, average=False):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    if bias is None:
        bias = st.r[-2.5:2.5:0.1,V] if rough else st.r[-2.5:2.5:0.05,V]
    if stats is None:
        stats = 30 if rough else 150

    if 'qubitBiasLimits' in q:
        default = (-2.5*V, 2.5*V)
        bias_lim = q['squidBiasLimits']
        if bias_lim != default:
            print 'limiting bias range to (%s, %s)' % tuple(bias_lim)
        resets = max(resets[0], bias_lim[0]), min(resets[1], bias_lim[1])
        bias = st.r[bias_lim[0]:bias_lim[1]:bias.range.step, V]

    if power is not None:
        name = name + 'power:' + str(power)
    axes = [(bias, 'Flux Bias')]
    deps = []
    deps = deps + [('S11 Phase', 'Reset: %s Qubit: %s' % (reset, q.__name__), rad) for reset in resets]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, fb):
        reqs = []
        if noisy: print "fb = %s" % fb
        for reset in resets:
            q['biasOperate'] = fb
            q['biasReadout'] = fb
            q['biasReset'] = [reset]
            if power is not None:
                q['readout power'] = power
            q['readout'] = True
            reqs.append(runQubits(server, qubits, stats, dataFormat='phases'))
        result = yield FutureList(reqs)
        returnValue(np.vstack(result).T)
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=False)

    if update:
        if rough:
            print Q['adc adjusted phase']
            adjust.adjust_phase(Q, data)
        else:
            adjust.adjust_phase_arc(Q, data)
    if collect:
        return data

def stepedge(sample, bias=None, stats=300L, measure=0,
             save=True, name='StepEdge MQ', collect=False, noisy=True, update=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    if bias is None:
        stepedge = q['biasOperate'][mV]
        stepedge = st.nearest(stepedge, 2.0)
        bias = st.r[stepedge-100:stepedge+100:2, mV]

    axes = [(bias, 'Operating bias')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, fb):
        q['biasOperate'] = fb
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    if update:
        adjust.adjust_operate_bias(Q, data)
    if collect:
        return data


def findSquidEdges(Sample, measure, iterations=4,
                   save=False, collect=False, noisy=False, update=True, plot=False):
    """Find the edges of two adjacent SQUID steps to get mutual inductance between bias line and qubit"""
    sample, qubits, Qubits = util.loadDeviceType(Sample, 'phaseQubit', write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]
    #Store operating conditions
    biasOperate = q['biasOperate']
    biasReadout = q['biasReadout']
    biasReset = q['biasReset']
    switchIntervals = q['squidSwitchIntervals']
    oldEdges = q['squidEdges']
    oldStepEdge = q['biasStepEdge']
    try:
        #Get a SQUID step scan
        data = squidsteps(sample, measure=measure, save=False, name='SquidSteps MQ', collect=True,
                          noisy=False, update=False, dataFormat=None, stats=60)
        #Find edge near current operating point
        firstEdge  =  findStepEdge_Test(Sample, measure, iterations=iterations, save=save, collect=True, noisy=noisy, update=False,  plot=plot)
        #Adjust parameters for a different operating point
        print 'Adjust parameters for the second step edge. Do not worry, your old parameters will be reset when we are done here.'
        adjust.adjust_squid_steps(Q, data)
        Q['biasStepEdge'] = Q['biasOperate']
        #Find edge near other operating point
        secondEdge  = findStepEdge_Test(Sample, measure, iterations=iterations, save=save, collect=True, noisy=noisy, update=False,  plot=plot)
        print "Old left edge: %s mV"  %(str(oldEdges[0]['mV'])[0:5])
        print "New left edge: %s mV"  %(str(firstEdge['mV'])[0:5])
        print "Old right edge: %s mV" %(str(oldEdges[1]['mV'])[0:5])
        print "New right edge: %s mV" %(str(secondEdge['mV'])[0:5])
        #Store the newly found squid edges
    finally:
        if update:
            Q['squidEdges']=[firstEdge,secondEdge]
            Q['biasStepEdge'] = firstEdge
        else:
            Q['biasStepEdge']=oldStepEdge
        #Restore normal parameters
        Q['biasOperate'] = biasOperate
        Q['biasReadout'] = biasReadout
        Q['biasReset'] = biasReset
        Q['squidSwitchIntervals'] = switchIntervals


def findStepEdge_Test(Sample, measure, biasBound=100*mV, iterations=4,
                      stats=150, save=False, collect=False, noisy=False, update=False, plot=False):
    """Find the step edge by doing a careful fit"""
    sample, qubits, Qubits = util.loadDeviceType(Sample, 'phaseQubit', write_access=True)
    q=qubits[measure]
    Q=Qubits[measure]
    stepEdge = q['biasStepEdge']
    fit = [stepEdge['mV'], 1.0, 0.0, 0.5]
    for i in range(iterations):
        dBias = biasBound/(2**i)
        dataStats = stats*(2**i)
        bias = structures.ValueArray(np.linspace(stepEdge['mV']-dBias['mV'],stepEdge['mV']+dBias['mV'],40),mV)
        data = stepedge(Sample, bias=bias, stats=dataStats, measure=measure, save=save, collect=True, noisy=noisy, update=False)
        if i==0:
            if data[-1,1]>data[0,1]:
                flip=1
            else:
                flip=-1
        def fitFunc(x, p):
            return 1.0+flip*p[3]*erf((x-p[0])/p[1])+p[2]
        fit, _ok  = leastsq(lambda p: fitFunc(data[:,0], p) - data[:,1],  fit)
        if plot and (i==iterations-1):
            plt.figure()
            plt.plot(data[:,0],data[:,1],'.',markersize=15)
            plt.plot(data[:,0],fitFunc(data[:,0],fit),'r')
        stepEdge = fit[0]*mV
        print 'New step edge %f mV'%(stepEdge['mV'])
    if update:
        Q['biasStepEdge']=stepEdge
    if collect:
        return stepEdge


def maxVisibility(Sample, measure, whichMpa, mpaRange=None, mpaFactor=None,
                  maxFuncEvals=10000, stats=1500, statsIncrease = 1.5,
                  maxStats = 10000, noisy=True, update=True):

    sample, qubits, Qubits = util.loadQubits(Sample, write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]
    ml.setMultiKeys(q,whichMpa)

    if mpaRange is None:
        if mpaFactor is None:
            mpaMin = float(raw_input('minMpa: '))
            mpaMax = float(raw_input('maxMpa: '))
            mpaRange = (mpaMin,mpaMax)
        else:
            mpaGuess = ml.getMultiLevels(q,'measureAmp',whichMpa)
            mpaRange = (mpaGuess*(1-mpaFactor),mpaGuess*(1+mpaFactor))

    #Maximize this function
    def func(mpa):
        data = scurve(Sample, mpa=[mpa], measure=measure, visibility=True,
                      stats=stats, states=[whichMpa-1,whichMpa],
                      save=False, collect=True, noisy=noisy, update=False)
        #TODO; change stats based on the results
        #We want to maximize, hence the factor of -1.
        #Return the -1th element because that's the visibility
        return -1*data[0][-1]

    result = opt.fminbound(func, mpaRange[0], mpaRange[1], maxfun=maxFuncEvals)
    print 'Best amplitude found: %f' %result
    if update:
        key = ml.saveKeyNumber('measureAmp',whichMpa)
        print 'Old %s: %f' %(key,q[key])
        print 'New %s: %f' %(key,result)
        Q[key] = result


def scurve(sample, mpa=None, visibility=False, stats=300, measure=0, states=0,
           save=True, name='SCurve MQ', collect=True, noisy=True, update=True):
    """S-Curve sequence on one qubit. Displays S-Curves for states given in state.
    If update is selected, will enable changing mpa.

    INPUT PARAMETERS
    sample: Object defining qubits to measure, loaded from registry.
    mpa - iterable: Measure pulse amplitude
    visibility - bool: Whether to calculate visibilities.
    stats - scalar: Number of times a point will be measured.
    measure - scalar: Number of qubit to measure. Only one qubit allowed.
    states - list or int: State(s) for which to measure S-curves. If
        visibility=True, must be a list where adjacent entries differ by 1.
    save - bool: Whether to save data to the datavault.
    name - string: Name of dataset.
    collect - bool: Whether or not to return data.
    noisy - bool: Whether to print out probabilities while the scan runs
    update - bool: Whether to display a popup to update the mpas in registry. If
        visibility=False, will update mpas for states one higher than those
        given by the parameter states. If visibility=True, will update mpas for
        states given by the parameter states.
    """
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    # Set mpa to [0:2:0.05] if stepEdge>biasOperate or to [0:-2:-0.05] if stepEdge<biasOperate.
    direction = -1 if (q['biasStepEdge']['mV']<q['biasOperate']['mV']) else 1
    if mpa is None:
        mpa=st.r[0:direction*2:direction*0.05]

    if isinstance(states,int):
        states = [states]
    if not (isinstance(states,list) or isinstance(states,int)):
        raise Exception('States must be a list or int.')
    states = sorted(states)
    ml.setMultiKeys(q,max(states[-1],1))

    if visibility:
        if len(states)<2:
            raise Exception('Visibility requires at least 2 S-curves.')
        if min([(states[i+1]-states[i]) for i in range(len(states)-1)])<1:
            raise Exception('States must not be identical for visibility.')
        if name=='SCurve MQ':
            name = 'Visibility MQ'

    axes = [(mpa, 'Measure pulse amplitude')]
    deps =[('Probability', '|'+str(state)+'>', '') for state in states]
    if visibility:
        deps = deps + [('Visibility', '|'+str(states[elem+1])+'>-|'+str(states[elem]), '') for elem in range(len(states)-1)]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, mpa):
        reqs =[]
        for state in states:
            ml.setMultiLevels(q,'measureAmp',mpa,state=1)
            q['readout'] = True
            q.xy = eh.boostState(q,0,state)
            t_meas = (state-0.5)*q['piLen'] if state>0.5 else 0
            q.z = eh.measurePulse(q, t_meas)
            reqs.append(runQubits(server, qubits, stats, probs=[1]))
        probs = yield FutureList(reqs)
        problist = [p[0] for p in probs]
        if visibility:
            problist = problist + list(np.array(problist[1:])-np.array(problist[:-1]))
        returnValue(problist)
    data = sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)
    if update:
        if visibility:
            if max([(states[i+1]-states[i]) for i in range(len(states)-1)])>1:
                print 'Can not update mpa for non-adjacent states.'
            else:
                adjust.adjust_visibility(Q, data, states)
        else:
            adjust.adjust_scurve(Q, data, states)
    if collect:
        return data


def calscurve(sample, stats=30000, measure=0, state=1,
           save=True, name='Cal SCurve MQ', collect=True, noisy=True, update=True):
    """Get S-Curve calibration on one qubit. If update is selected, will produce
        registry keys giving all state probs for each mpa.

    FIXED PARAMETERS
    sample: Object defining qubits to measure, loaded from registry.
    stats - scalar: Number of times a point will be measured.
    measure - scalar: Number of qubit to measure. Only one qubit allowed.
    save - bool: Whether to save data to the datavault.
    name - string: Name of dataset.
    collect - bool: Whether or not to return data.
    noisy - bool: Whether to print out probabilities while the scan runs
    update - bool: Add calibration keys (calScurve) to registry. calScurveN is
        list of [P(|0>),P(|1>),...P(|N>)] given that used measurePulseN.
    state - int: highest state to measure
    """
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    ml.setMultiKeys(q,state)

    # If measuring s-curve calibrations, make sure that measure probs for
    # states=[0,1,...,state].
    states = range(state+1)
    whichScurves=states[1:]

    axes = [(whichScurves, 'Measured pulse state')]
    deps =[('Probability', '|'+str(state)+'>', '') for state in states]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, whichScurve):
        reqs =[]
        for state in states:
            q['readout'] = True
            q.xy = eh.boostState(q,0,state)
            q.z = eh.measurePulse(q, (state-0.5)*q['piLen'], state=whichScurve)
            reqs.append(runQubits(server, qubits, stats, probs=[1]))
        probs = yield FutureList(reqs)
        problist = [p[0] for p in probs]
        returnValue(problist)
    data = sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)
    if update:
        for whichScurve in whichScurves:
            Q['calScurve'+str(whichScurve)]=data[whichScurve-1,1:]
    if collect:
        return data


def visibility(sample, mpa=st.r[0:2:0.05], states=1, stats=300, calstats=30000, measure=0,
           save=True, name='Visibility MQ', collect=True, noisy=True, update=True):
    """S-Curve sequence on one qubit. Displays S-Curves and visibilities for all mpas
    given in state. If update is selected, will enable changing mpa. At end, will
    produce registry keys (calScurve) giving all state probs for given mpa.

    PARAMETERS
    sample: Object defining qubits to measure, loaded from registry.
    mpa - iterable: Measure pulse amplitude.
    states - list or int: State(s) for whose measure pulse amplitude(s) to measure
        visibilities.
    stats - scalar: Number of times a point will be measured.
    calstats - scalar: Number of times a point will be measured for calibrations.
    measure - scalar: Number of qubit to measure. Only one qubit allowed.
    save - bool: Whether to save data to the datavault.
    name - string: Name of dataset.
    collect - bool: Whether or not to return data.
    noisy - bool: Whether to print out probabilities while the scan runs
    update - bool: Whether to display a popup to update the mpas in registry.
    """
    if isinstance(states,int):
        states = [states]
    if not (isinstance(states,list) or isinstance(states,int)):
        raise Exception('States must be a list or int.')
    minstate = min(states)
    maxstate = max(states)
    # set state to [minstate-1,minstate...,maxstate]
    # Use maxstate+1 since np.arange(m,n+1) gives [m,...,n]
    states = list(np.arange(max(minstate-1,0),maxstate+1))

    data = scurve(sample, states=states, mpa=mpa, measure=measure,visibility=True,
                  stats=stats, save=save, name=name, collect=True, noisy=noisy, update=update)
    data2 = calscurve(sample, state=maxstate, stats=calstats, measure=measure,
                    save=save, name=name+' Cals', collect=True, noisy=noisy)

    if collect:
        return data, data2


def find_step_edge(sample, stats=60, target=0.5, bias_range=None,
                   measure=0, resolution=0.1, blowup=0.05,
                   edgeOnLeft=None, statsinc=1.25,
                   save=False, name='StepEdge Search MQ', collect=False, update=True, noisy=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    axes = [('Flux Bias', 'mV')]
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure)

    if edgeOnLeft is None:
        edgeOnLeft = q['biasOperate'][V] > q['biasStepEdge'][V]
    if bias_range is None:
        stepedge = q['biasOperate'][mV]
        stepedge = st.nearest(stepedge, 2.0)
        bias_range = (stepedge-100, stepedge+100)
    interval = list(bias_range)

    def sweep(stats=stats):
        yield 0.5*(interval[0]+interval[1]), stats
        lower = True
        coeffs = 0.25, 0.75
        while interval[1] - interval[0] > resolution:
            stats *= statsinc
            fb = coeffs[lower]*interval[0] + coeffs[not lower]*interval[1]
            fb = st.nearest(fb, 0.2*resolution)
            yield fb, min(int((stats+29)/30)*30, 30000)
            lower = not lower

    def func(server, args):
        #Reset the qubit, operate at fb, and measure
        fb, stats = args
        q['biasOperate'] = fb*mV
        q['readout'] = True
        prob = yield runQubits(server, qubits, stats, probs=[1])

        #Bring the upper limit down if we overshot an edge to the right
        #or undershot an edge to the left
        if (prob[0] > target) ^ edgeOnLeft:    # ^ is XOR in python
            interval[1] = min(fb, interval[1])
        #Otherwise raise the lower limit
        else:
            interval[0] = max(fb, interval[0])
        inc = blowup * (interval[1] - interval[0])
        interval[0] -= inc
        interval[1] += inc
        if noisy:
            print fb, prob[0]
        returnValue([fb, prob[0]])
    sweeps.run(func, sweep(), save, dataset, pipesize=2, noisy=False)

    fb = 0.5 * (interval[0] + interval[1])*mV
    if 'biasStepEdge' in q:
        print 'Old bias_step_edge: %.3f' % Q['biasStepEdge']
    print 'New biasStepEdge: %.3f' % fb
    if update:
        Q['biasStepEdge'] = fb
    return fb


def find_mpa(sample, stats=60, target=0.05, mpa_range=(-2.0, 2.0), state=1,
             measure=0, pulseFunc=None, resolution=0.005, blowup=0.05,
             falling=None, statsinc=1.25,
             save=False, name='SCurve Search MQ', collect=True, update=True, noisy=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    ml.setMultiKeys(q,state)

    axes = [('Measure Pulse Amplitude', '')]
    deps = [('Probability', '|'+str(state)+'>', '')]
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure)

    if falling is None:
        falling = q['biasOperate'][V] > q['biasStepEdge'][V]
    interval = [min(mpa_range), max(mpa_range)]

    def sweep(stats=stats):
        mpa = 0.5 * (interval[0] + interval[1])
        yield mpa, min(int((stats+29)/30)*30, 30000)
        lower = True
        coeffs = 0.25, 0.75
        while interval[1] - interval[0] > resolution:
            stats *= statsinc
            mpa = coeffs[lower]*interval[0] + coeffs[not lower]*interval[1]
            mpa = st.nearest(mpa, 0.2*resolution)
            yield mpa, min(int((stats+29)/30)*30, 30000)
            lower = not lower

    def func(server, args):
        mpa, stats = args
        ml.setMultiLevels(q,'measureAmp',mpa,state)
        q.xy = eh.boostState(q, 0, state-1)
        q.z = eh.measurePulse(q, max(0.0,state-1.5)*q['piLen'], state=state)
        q['readout'] = True
        probs = yield runQubits(server, qubits, stats, probs=[1])

        prob = probs[0]
        if (prob > target) ^ falling:
            interval[1] = min(mpa, interval[1])
        else:
            interval[0] = max(mpa, interval[0])
        inc = blowup * (interval[1] - interval[0])
        interval[0] -= inc
        interval[1] += inc

        if noisy:
            print mpa, prob
        returnValue([mpa, prob])
    sweeps.run(func, sweep(), save, dataset, pipesize=2, noisy=False)

    mpa = 0.5 * (interval[0] + interval[1])
    key = ml.saveKeyNumber('measureAmp',state)
    if (key in q) and noisy:
        print 'Old %s: %.3f' % (key, Q[key])
    if noisy: print 'New %s: %.3f' % (key, mpa)
    if update:
        Q[key] = mpa
    return mpa


def find_mpa_func(sample, target=0.05, measure=0, biaspoints=None, order=1,
                  steps=5, noisy=True, update=True, plot=False):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    if biaspoints is None:
        fb0 = q['biasOperate'][mV]
        mpa = q['measureAmp']
        fb1 = fb0 + 20 * np.sign(mpa) * np.sign(abs(mpa)-1)
        biaspoints = [fb*mV for fb in np.linspace(fb0, fb1, steps)]
        if noisy: print 'Biaspoints:', ', '.join(str(b) for b in biaspoints)
    mpas = []
    for fb in biaspoints:
        q['biasOperate'] = fb
        if noisy: print 'Flux bias:'+str(fb)
        mpas.append(find_mpa(sample, target=target, measure=measure, noisy=noisy))
    biaspoints = np.array([fb[V] - q['biasStepEdge'][V] for fb in biaspoints])
    mpas = np.array(mpas)
    p = np.polyfit(biaspoints, mpas, order)
    if plot:
        data = np.vstack((biaspoints,mpas)).T
        fig=dstools.plotDataset1D(data,[('Bias Flux referenced to Step Edge','V')],[('Measure Pulse Amplitude','','')],
                                  style='.',markersize=20,legendLocation=None,show=False,title='MPA vs Flux bias')
        fig.get_axes()[0].plot(biaspoints,np.polyval(p,biaspoints),'r',linewidth=4)
        fig.show()
    if update:
        Q['calMpaFunc'] = p
    return get_mpa_func(q, p)


def get_mpa_func(qubit, p=None):
    if p is None:
        p = qubit['calMpaFunc']
    return lambda fb: np.polyval(p, fb[V] - qubit['biasStepEdge'][V])


def find_flux_func(sample, freqScan=None, measAmplFunc=None, measure=0,
                   fluxBelow=2*mV, fluxAbove=2*mV, fluxStep=0.1*mV, sb_freq=0*GHz, stats=300L,plot=False,
                   save=True, name='Flux func search MQ', collect=False, update=True, noisy=True):
    """Find the qubit frequency vs. flux bias voltage. A fit to the data is
    stored in the registry for the measured qubit.

    EXPLANATION OF FITTING:
    The plasma frequency of the flux biased phase qubit is give as
    f = f0 (1-x)^{1/4} where f0 is a constant and
    where x is the bias flux divided by the critical bias flux. The actual flux
    put into the system may have an offset, and has a scaling with the bias
    voltage determined by mutual inductance, etc, so, the parameter x
    depends on voltage linearly as x=aV+x0. Therefore,
    f^4 = f0^4(1-x0-aV)
    We reference V to the step edge value by defining V'=V-Vstep and get
    f^4 = f0^4(1-x0-aVstep) - f0^4 a V'
    find_flux_func takes the frequency to the fourth power as the independent
    variable so we re-arrange the formula to
    V' = [(1-x0-aVstep)/a] - (1/a)(f/f0)^4
    This is the equation fitted by find_flux_func. The returned first order
    polynomial fit, p, is defined by
    V' = p[1] + p[0]f^4
    """
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    qubit, Qubit = qubits[measure], Qubits[measure]

    if measAmplFunc is None:
        measAmplFunc = get_mpa_func(qubit)
    #Set up the frequency scan
    if freqScan is None:
        freq = st.nearest(qubit['f10'][GHz], 0.001)
        dfs = np.logspace(-3, -0.3, 25)
        freqScan = freq + np.hstack(([0], dfs, -dfs))
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
    #Set up flux scan. For each frequency we scan over a range of fluxes.
    #The scan is adaptive, so the flux range will change as the scan goes on.
    fluxBelow = fluxBelow[V]
    fluxAbove = fluxAbove[V]
    fluxStep = fluxStep[V]
    fluxScan = np.arange(-fluxBelow, fluxAbove, fluxStep)
    fluxScan = fluxScan[np.argsort(abs(fluxScan))]
    fluxPoints = len(fluxScan)
    step_edge = qubit['biasStepEdge'][V]
    #This is the running table defining the flux scan. This is changed as the
    #sweep progresses.
    sweepData = {
        'fluxFunc': np.array([st.nearest(qubit['biasOperate'][V], fluxStep) - step_edge]),
        'fluxIndex': 0,
        'freqIndex': 0,
        'flux': 0*fluxScan,
        'prob': 0*fluxScan,
        'maxima': 0*freqScan,
    }
    #Set up dataset
    axes = [('Flux Bias', 'V'), ('Frequency', 'GHz')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)
    #Object that defines the
    def sweep():
        for f in freqScan:
            center = np.polyval(sweepData['fluxFunc'], f**4) + step_edge
            center = st.nearest(center, fluxStep)
            for flx in center + fluxScan:
                yield flx*V, f*GHz

    def func(server, args):
        flux, freq = args
        for q in qubits:
            q['fc'] = freq - sb_freq # set all frequencies since they share a common microwave source
        qubit['biasOperate'] = flux
        qubit['measureAmp'] = measAmplFunc(flux)
        qubit.xy = eh.spectroscopyPulse(qubit, 0, sb_freq)
        qubit.z = eh.measurePulse(qubit, qubit['spectroscopyLen'] + qubit['piLen'])
        qubit['readout'] = True
        prob = yield runQubits(server, qubits, stats, probs=[1])

        flux_idx = sweepData['fluxIndex']
        sweepData['flux'][flux_idx] = flux[V]
        sweepData['prob'][flux_idx] = prob[0]
        if flux_idx + 1 == fluxPoints:
            if noisy: print 'row %d done' %sweepData['freqIndex']
            # one row is done.  find the maximum and update the spectroscopy fit
            freq_idx = sweepData['freqIndex']
            sweepData['maxima'][freq_idx] = sweepData['flux'][np.argmax(sweepData['prob'])]
            sweepData['fluxFunc'] = np.polyfit(freqScan[:freq_idx+1]**4,
                                                  sweepData['maxima'][:freq_idx+1] - step_edge,
                                                  (freq_idx > 5))
            sweepData['fluxIndex'] = 0
            sweepData['freqIndex'] += 1
        else:
            # just go to the next point
            sweepData['fluxIndex'] = flux_idx + 1
        returnValue([flux, freq, prob[0]])
    sweeps.run(func, sweep(), dataset=save and dataset, collect=collect, noisy=noisy)

    # create a flux function and return it
    p = sweepData['fluxFunc']
    freq_idx = sweepData['freqIndex']
    #return (freqScan[:freq_idx],sweepData['maxima'][:freq_idx],p)
    if plot:
        indices = np.argsort(sweepData['maxima'][:freq_idx])
        data = np.vstack((sweepData['maxima'][:freq_idx][indices],freqScan[0:freq_idx][indices])).T
        fig=dstools.plotDataset1D(data,[('Bias Flux','V')],[('Frequency','','GHz')],style='.',legendLocation=None,show=False,
                                  markersize=15,title='')
        fig.get_axes()[0].plot(np.polyval(p,data[:,1]**4)+qubit['biasStepEdge']['V'],data[:,1],'r',linewidth=3)
        fig.show()
    if update:
        Qubit['calFluxFunc'] = p
    return get_flux_func(Qubit, p, step_edge*V)


def get_flux_func(qubit, p=None, step_edge=None):
    """Returns a function that gives bias voltage as a function of frequency

    Remember that qubit['calFluxFunc'] takes a frequency in GHz and returns
    a bias voltage referenced to the step edge. Look at find_flux_func doc
    string for more information
    """
    if p is None:
        p = qubit['calFluxFunc']
    if step_edge is None:
        step_edge = qubit['biasStepEdge']
    return lambda f: np.polyval(p, f[GHz]**4)*V + step_edge #V here is a labrad unit 'Volt'


def freq2bias(sample, qubitNum, freq):
    sample, qubits = util.loadQubits(sample)
    qubit = qubits[qubitNum]
    p = qubit['calFluxFunc']
    bias = p[1] + p[0]*(freq['GHz']**4)
    bias = bias*V

    return bias + qubit['biasStepEdge']


def find_zpa_func(sample, freqScan=None, measure=0, spectroscopyAmp = None,
                  fluxBelow=0.01, fluxAbove=0.01, fluxStep=0.0005, sb_freq=0*GHz, stats=300L,
                  name='ZPA func search MQ', save=True, collect=False, noisy=True, update=True,
                  plot=False):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    qubit = qubits[measure]
    if spectroscopyAmp is not None:
        qubit['spectroscopyAmp'] = spectroscopyAmp
    if freqScan is None:
        freq = st.nearest(qubit['f10'][GHz], 0.001)
        dfs = np.logspace(-3, -0.3, 25)
        freqScan = freq + np.hstack(([0], dfs, -dfs))
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
    fluxPoints = len(fluxScan)

    sweepData = {
        'fluxFunc': np.array([0]),
        'fluxIndex': 0,
        'freqIndex': 0,
        'flux': 0*fluxScan,
        'prob': 0*fluxScan,
        'maxima': 0*freqScan,
    }

    axes = [('Z-pulse amplitude', ''), ('Frequency', 'GHz')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def sweep():
        for f in freqScan:
            center = np.polyval(sweepData['fluxFunc'], f**4)
            center = st.nearest(center, fluxStep)
            for zpa in center + fluxScan:
                yield zpa, f*GHz

    def func(server, args):
        zpa, freq = args
        for q in qubits:
            q['fc'] = freq - sb_freq # set all frequencies since they share a common microwave source
        dt = qubit['spectroscopyLen']
        qubit.xy = eh.spectroscopyPulse(qubit, 0, sb_freq)
        qubit.z = env.rect(0, dt, zpa) + eh.measurePulse(qubit, dt)
        qubit['readout'] = True
        prob = yield runQubits(server, qubits, stats, probs=[1])

        flux_idx = sweepData['fluxIndex']
        sweepData['flux'][flux_idx] = zpa
        sweepData['prob'][flux_idx] = prob[0]
        if flux_idx + 1 == fluxPoints:
            # one row is done.  find the maximum and update the spectroscopy fit
            freq_idx = sweepData['freqIndex']
            sweepData['maxima'][freq_idx] = sweepData['flux'][np.argmax(sweepData['prob'])]
            sweepData['fluxFunc'] = np.polyfit(freqScan[:freq_idx+1]**4,
                                               sweepData['maxima'][:freq_idx+1],
                                               freq_idx > 5)
            sweepData['fluxIndex'] = 0
            sweepData['freqIndex'] += 1
        else:
            # just go to the next point
            sweepData['fluxIndex'] = flux_idx + 1
        returnValue([zpa, freq, prob[0]])
    sweeps.run(func, sweep(), dataset=save and dataset, collect=collect, noisy=noisy)

    # create a flux function and return it
    poly = sweepData['fluxFunc']
    freq_idx = sweepData['freqIndex']
    if plot:
        indices = np.argsort(sweepData['maxima'][:freq_idx])
        data = np.vstack((sweepData['maxima'][:freq_idx][indices],freqScan[0:freq_idx][indices])).T
        fig=dstools.plotDataset1D(data,[('Z pulse amplitude','')],[('Frequency','','GHz')],style='.',legendLocation=None,show=False,
                                  markersize=15,title='')
        fig.get_axes()[0].plot(np.polyval(poly,data[:,1]**4),data[:,1],'r',linewidth=3)
        fig.show()
    if update:
        Qubits[measure]['calZpaFunc'] = poly
    return get_zpa_func(Qubits[measure], poly)


def get_zpa_func(qubit, p=None):
    if p is None:
        p = qubit['calZpaFunc']
    return lambda f: np.polyval(p, f[GHz]**4)


def freq2zpa(sample, qubitNum, freq):
    sample, qubits = util.loadQubits(sample)
    qubit = qubits[qubitNum]
    p = qubit['calZpaFunc']
    zpa = p[1] + p[0]*(freq['GHz']**4)
    return zpa


def zpa2freq(sample, qubitNum, zpa):
    sample, qubits = util.loadQubits(sample)
    qubit = qubits[qubitNum]
    p = qubit['calZpaFunc']
    freq = ((zpa-p[1])/p[0])**(0.25)
    return freq


def findDfDv(sample, measure, freqScan=None, measAmplFunc=None,
             fluxBelow=2*mV, fluxAbove=2*mV, fluxStep=0.1*mV, sb_freq=0*GHz, stats=300L,
             plot=False, save=True, name='Find dfreq/dV', collect=False, update=True, noisy=True):
    """Find the qubit frequency sensitivity to bias voltage"""
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    qubit, Qubit = qubits[measure], Qubits[measure]

    if measAmplFunc is None:
        measAmplFunc = get_mpa_func(qubit)
    #Set up the frequency scan
    if freqScan is None:
        freq = st.nearest(qubit['f10'][GHz], 0.001)
        dfs = np.linspace(-0.05,0.05,30)
        freqScan = freq + np.hstack(([0], dfs))
    else:
        freqScan = np.array([f[GHz] for f in freqScan])
    freqScan = freqScan[np.argsort(abs(freqScan-qubit['f10'][GHz]))]
    #Set up flux scan. For each frequency we scan over a range of fluxes.
    #The scan is adaptive, so the flux range will change as the scan goes on.
    fluxBelow = fluxBelow[V]
    fluxAbove = fluxAbove[V]
    fluxStep = fluxStep[V]
    fluxScan = np.arange(-fluxBelow, fluxAbove, fluxStep)
    fluxScan = fluxScan[np.argsort(abs(fluxScan))]
    fluxPoints = len(fluxScan)
    step_edge = qubit['biasStepEdge'][V]
    #This is the running table defining the flux scan. This is changed as the
    #sweep progresses.
    sweepData = {
        'fluxFunc': np.array([st.nearest(qubit['biasOperate'][V], fluxStep) - step_edge]),
        'fluxIndex': 0,
        'freqIndex': 0,
        'flux': 0*fluxScan,
        'prob': 0*fluxScan,
        'maxima': 0*freqScan,
    }
    #Set up dataset
    axes = [('Flux Bias', 'V'), ('Frequency', 'GHz')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)
    #Object that defines the
    def sweep():
        for f in freqScan:
            center = np.polyval(sweepData['fluxFunc'], f) + step_edge
            center = st.nearest(center, fluxStep)
            for flx in center + fluxScan:
                yield flx*V, f*GHz

    def func(server, args):
        flux, freq = args
        for q in qubits:
            q['fc'] = freq - sb_freq # set all frequencies since they share a common microwave source
        qubit['biasOperate'] = flux
        qubit['measureAmp'] = measAmplFunc(flux)
        qubit.xy = eh.spectroscopyPulse(qubit, 0, sb_freq)
        qubit.z = eh.measurePulse(qubit, qubit['spectroscopyLen'] + qubit['piLen'])
        qubit['readout'] = True
        prob = yield runQubits(server, qubits, stats, probs=[1])

        flux_idx = sweepData['fluxIndex']
        sweepData['flux'][flux_idx] = flux[V]
        sweepData['prob'][flux_idx] = prob[0]
        if flux_idx + 1 == fluxPoints:
            if noisy: print 'row %d done' %sweepData['freqIndex']
            # one row is done.  find the maximum and update the spectroscopy fit
            freq_idx = sweepData['freqIndex']
            sweepData['maxima'][freq_idx] = sweepData['flux'][np.argmax(sweepData['prob'])]
            sweepData['fluxFunc'] = np.polyfit(freqScan[:freq_idx+1],
                                                  sweepData['maxima'][:freq_idx+1] - step_edge,
                                                  (freq_idx > 5))
            sweepData['fluxIndex'] = 0
            sweepData['freqIndex'] += 1
        else:
            # just go to the next point
            sweepData['fluxIndex'] = flux_idx + 1
        returnValue([flux, freq, prob[0]])
    sweeps.run(func, sweep(), dataset=save and dataset, collect=collect, noisy=noisy)

    # create a flux function and return it
    p = sweepData['fluxFunc']
    freq_idx = sweepData['freqIndex']
    if plot:
        indices = np.argsort(sweepData['maxima'][:freq_idx])
        data = np.vstack((sweepData['maxima'][:freq_idx][indices],freqScan[0:freq_idx][indices])).T
        fig=dstools.plotDataset1D(data,[('Bias Flux','V')],[('Frequency','','GHz')],style='.',legendLocation=None,show=False,
                                  markersize=15,title='')
        fig.get_axes()[0].plot(np.polyval(p,data[:,1])+qubit['biasStepEdge']['V'],data[:,1],'r',linewidth=3)
        fig.show()
    if update:
        slope = (1.0/p[0])*(GHz/V)
        Qubit['calDfDV'] = slope


def operateQubitAt(qubit, f):
    """Update qubit parameters to operate at the specified frequency."""
    fb = get_flux_func(qubit)(f)
    mpa = get_mpa_func(qubit)(fb)
    zpa = get_zpa_func(qubit)(f)
    print 'f: %s, fb: %s, mpa: %s, zpa: %s' % (f, fb, mpa, zpa)
    qubit['f10'] = f
    qubit['biasOperate'] = fb
    qubit['measureAmp'] = mpa


def operateAt(sample, f, measure=0):
    """Update qubit parameters to operate at the specified frequency."""
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    Q = Qubits[measure]
    operateQubitAt(Q,f)


def find_xypa_func(sample, amps=np.linspace(0.1, 1.2, 30), length=st.r[0:200:1, ns],
                 measure=0, stats=600, save=True, noisy=True, collect=True,
                 nfftpoints=4096, update=True, plot=True, forceZero=True):
    """
    get rabi amp vs rabi frequency
    freq = a * amp^2 + b*amp + c, and registry key calXYpaFunc=[a, b, c]
    and registry key calXYpaFunc=[a,b,c]

    get rabi frequency vs rabi amp
    amp = a * freq^2 + b * freq + c
    and registry key calRabiFreqFunc=[a,b,c]

    fitting these functions by 2-order is due to the experimental results.
    if forceZero is True, then the fitting function will be a*x^2 + b*x
    """
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    data = rabilongHD(sample, length=length, amplitude=amps, detuning=None,
                         measureDelay=None, measure=measure, stats=stats, save=save,
                         collect=True, noisy=noisy, randomize=False, name='xypa func search MQ')

    def quadratic(x, a, b):
        return a*x**2 + b*x
    data = np.asarray(data)
    data = dstools.arrange2Dscan(data)[2]
    rabi_freqs = np.array([fitting.maxFreq(np.array(zip(length,d)), nfftpoints,
                            plot=False)['GHz'] for d in data])
    if forceZero:
        amps = np.array(amps)
        func, _ = curve_fit(quadratic, amps, rabi_freqs)
        func_I, _ = curve_fit(quadratic, rabi_freqs, amps)
        func = np.hstack([func, 0.0])
        func_I = np.hstack([func_I, 0.0])
    else:
        func = np.polyfit(amps, rabi_freqs, 2)
        func_I = np.polyfit(rabi_freqs, amps, 2)
    if update:
        Q['calXYpaFunc'] = list(func)
        Q['calRabiFreqFunc'] = list(func_I)
    if plot or update:
        plt.figure()
        plt.plot(amps, rabi_freqs,'ro')
        plt.plot(amps, np.poly1d(func)(amps),'b-')
        plt.xlabel('rabi amp')
        plt.ylabel('rabi freq')
        plt.figure()
        plt.plot(rabi_freqs, amps, 'ro')
        plt.plot(rabi_freqs, np.poly1d(func_I)(rabi_freqs), 'b-')
        plt.xlabel('rabi freq')
        plt.ylabel('rabi amp')
    if collect:
        return data, func, func_I

def get_xypa_func(qubit, p=None):
    if p is None:
        p = qubit['calXYpaFunc']
    func = lambda amp: np.polyval(p, amp)
    return func

def get_rabifreq_func(qubit, p=None):
    if p is None:
        p = qubit['calRabiFreqFunc']
    func = lambda f: np.polyval(p, f)
    return func

def xypa2rabifreq(qubit, amp):
    func = get_xypa_func(qubit)
    freq = func(amp)
    return freq

def rabifreq2xypa(qubit, freq):
    assert isinstance(freq, Value) is True
    freq = freq["GHz"]
    func = get_rabifreq_func(qubit)
    xypa = func(freq)
    return xypa

def pulseshape(sample, height=-1.0, pulseLen=2000*ns, useOldRatesAmps = False,
               time=None, mpaStep=0.005, minPeak=0.4, stopAt=0.3, minCount=2,
               stats=300, measure=0, save=True, name='Pulse shape measurement',
               plot=True, noisy=True, update=True):
    if useOldRatesAmps and update:
        raise Exception('Using settling correction for the experiment and then updating will lead to incorrect settling parameters!')
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    if time is None:
        time = range(3, 25) + [round(10**(0.02*i)) for i in range(70, 110)]
        time = [t*ns for t in time]

    axes = [('time after step', time[0].units), ('z offset', '')]
    kw = {'step height': height, 'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    scanInfo = {'peakFound': False, 'lowCount': 0, 'highCount': 0}
    def sweep():
        center = 0
        for t in time:
            scanInfo['time'] = t
            scanInfo['center'] = center
            yield t, center
            low = center
            high = center
            scanInfo['lowCount'] = 0
            scanInfo['highCount'] = 0
            scanInfo['peakFound'] = False
            while ((scanInfo['lowCount'] < minCount) or
                   (scanInfo['highCount'] < minCount) or
                   (not scanInfo['peakFound'])):
                if (scanInfo['lowCount'] < minCount) or (not scanInfo['peakFound']):
                    low -= mpaStep
                    yield t, low
                if (scanInfo['highCount'] < minCount) or (not scanInfo['peakFound']):
                    high += mpaStep
                    yield t, high
            center = round(0.5*(low+high)/mpaStep)*mpaStep

    def func(server, args):
        t, ofs = args
        delay = q['piLen']
        if not useOldRatesAmps:
            q['settlingAmplitudes'] = []
            q['settlingRates'] = []
        q.xy = eh.mix(q, eh.piPulse(q, pulseLen + t))
        q.z = (env.rect(0, pulseLen + t + delay,  height + ofs) +
                env.rect(pulseLen, t + delay, -height) +
                eh.measurePulse(q, pulseLen + t + delay*2))
        q['readout'] = True
        probData = yield runQubits(server, qubits, stats, probs=[1])
        prob = probData[0]
        if t == scanInfo['time']:
            if prob >= minPeak:
                scanInfo['peakFound'] = True
            side = 'highCount' if ofs > scanInfo['center'] else 'lowCount'
            if prob < stopAt:
                scanInfo[side] += 1
            else:
                scanInfo[side] = 0
        returnValue([t, ofs, prob])
    pulsedata = sweeps.run(func, sweep(), dataset=save and dataset, noisy=noisy)

    p, func = qpc._getstepfunc(pulsedata, height, plot=plot,
                               ind=dataset.independents, dep=dataset.dependents)

    if update:
        Q['settlingRates'] = p[2::2]
        Q['settlingAmplitudes'] = p[1::2]/float(height)


def rabihigh(sample, amplitude=st.r[0.0:1.75:0.05], measureDelay=None, measure=0, state=1, numPulses=1, tBuf=0.0*ns,
             stats=1500L, name='Rabi-pulse height MQ', save=True, collect=False, noisy=True, update=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    ml.setMultiKeys(q,state)
    name='|'+str(state)+'> '+name

    if amplitude is None:
        amp = ml.getMultiLevels(q,'piAmp',state)
        amplitude = np.linspace(0.8*amp,1.2*amp,50)
    if measureDelay is None: measureDelay = q['piLen']/2.0

    axes = [(amplitude, 'pulse height'),
            (measureDelay, 'measure delay')]
    deps = [('Probability', '|'+str(state)+'>', '')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, amp, delay):
        ml.setMultiLevels(q,'piAmp',amp,state)
        q.xy = eh.boostState(q, 0, state-1)
        t = (state-1)*q['piLen']
        for _ in range(numPulses):
            q.xy += eh.mix(q, eh.piPulseHD(q, t, state=state), state=state)
            t += q['piLen']+tBuf
        t -= (tBuf+q['piLen'])
        q.z = eh.measurePulse(q, t+delay, state=state)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, collect=(collect or update), noisy=noisy)

    if update:
        result = adjust.adjust_rabihigh(Q, data, state=state)
        if result is not None:
            key = ml.saveKeyNumber('piAmp', state)
            print 'Old %s: %f' %(key, Q[key])
            print 'Selected %s: %f' %(key, result)
            # Fit parabola to part of rabihigh centered about chosen point.
            rabikeep = (abs(data[:,0]-result) < result*0.4/numPulses)
            datachoose = np.array(data[rabikeep])
            amp,maxrabiprob = fitting.findMinimum(datachoose)
            Q[key] = amp
            print 'New (fitted) %s: %f' %(key, amp)
    if collect:
        return data


def rabilong(sample, length=st.r[0:500:2,ns], amplitude=None, detuning=None, measureDelay=None, measure=0, stats=1500L,
             name='Rabi-pulse length MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    if amplitude is None: amplitude = q['piAmp']
    if detuning is None: detuning = 0
    if measureDelay is None: measureDelay = q['piLen']/2.0

    axes = [(length, 'pulse length'),
            (amplitude, 'pulse height'),
            (measureDelay, 'measure delay')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, len, amp, delay):
        q.xy = eh.mix(q, env.flattop(0, len, w=q['piFWHM'], amp=amp))
        q.z = eh.measurePulse(q, delay+len)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)
    if collect:
        return data


def rabilongHD(sample, length=st.r[0:500:2,ns], amplitude=None, detuning=None, measureDelay=None,
               randomize = True, measure=0, stats=1500L, name='Rabi MQ', save=True,
               collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    if amplitude is None: amplitude = q['piAmp']
    if detuning is None: detuning = 0
    if measureDelay is None: measureDelay = q['piLen']/2.0
    if randomize:
        length = st.shuffle(length)

    axes = [(length, 'pulse length'),
            (amplitude, 'pulse height'),
            (measureDelay, 'measure delay')]
    kw = {'stats': stats, 'amplitude':amplitude}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, len, amp, delay):
        q.xy = eh.mix(q, eh.rabiPulseHD(q, 0, len, amp=amp))
        q.z = eh.measurePulse(q, delay+len)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)
    if collect:
        return data


def pituner(sample, diff=0.4, iterations=1, numPulses=3, tBuf=0.0*ns, npoints=101,
            measure=0, stats=1200, save=False, update=True, noisy=True, state=1):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    ml.setMultiKeys(q,state)

    amp = ml.getMultiLevels(q,'piAmp',state)
    ampstart = amp
    damp = diff/numPulses
    for _ in xrange(iterations):
        # optimize amplitude
        data = rabihigh(sample, amplitude=np.linspace((1-damp)*amp, (1+damp)*amp, npoints),
                        numPulses=numPulses, tBuf=tBuf, measure=measure, stats=stats,
                        collect=True, noisy=noisy, save=save, state=state, update=False)
        amp_fit = np.polyfit(data[:,0], data[:,1], 2)
        amp,maxrabiprob = fitting.findMinimum(data)
        if noisy: print 'Amplitude: %g' % amp
    # save updated values
    if noisy: print 'Old Amplitude: %g' % ampstart
    if update:
        key = ml.saveKeyNumber('piAmp',state)
        Q[key] = amp
        Q['calRabiProb'+str(state)] = maxrabiprob
    return amp


def pitunerZ(sample, measure=0, npoints=100, nfftpoints=2048, stats=1200, save=False, update=True, noisy=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    # optimize amplitude
    data = ramseyZPulse(sample, zAmp=st.r[-0.1:0.1:0.002],
                       measure=measure, stats=stats, collect=True, noisy=noisy)
    x = data[:,0]
    y = data[:,1] - np.polyval(np.polyfit(data[:,0], data[:,1], 2), data[:,0])
    zstep = x[1]-x[0]
    fourier = np.fft.fftshift(abs(np.fft.fft(y,nfftpoints)))
    freq = np.fft.fftshift(np.fft.fftfreq(nfftpoints,zstep))
    ztwopi = 1.0/(abs(freq[np.argmax(fourier)]))
    zpi = ztwopi / 2.0
    print 'zAmplitude in DAC units that give a Pi rotation: %g' %zpi
    # save updated values
    q['piAmpZ'] = zpi
    if update:
        Q['piAmpZ'] = zpi
    return zpi


def freqtuner(sample, iterations=1, tEnd=100*ns, timeRes=1*ns, nfftpoints=4000, stats=1200, df=50*MHz,
              measure=0, save=False, plot=False, noisy=True, update=True, state=1, spinEcho=False):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    ml.setMultiKeys(q,state)
    # New Dec 2009: Automatically finds best f10, using ramsey can plot the FFT of the Ramsey fringe to extract true f10
    # works for sweeps with time steps that are all equivalent (i.e. not concatenated sweeps with diff time steps)

    # TODO make the choice of steps more physical/meaningful, currently set up for speed.
    # Time resolution should be at least at the Nyquist frequency, but better to oversample
    # nyfreq=float(fringeFreq)*2*10e6
    # timeRes = (1.0/float(nyfreq))*1e9
    # TODO make successive iterations finer resolution
    # TODO better way to detect frequency than just maximum value (peak fitting)
    for i in xrange(iterations):
        if not spinEcho:
            data = dephasingSweeps.ramsey(sample, measure=measure, delay=st.r[0:tEnd:timeRes, ns], fringeFreq = df,
                                          stats=stats, name='Ramsey Freq Tuner MQ', save = save, noisy=noisy, state=state,
                                          collect = True, randomize=False, averages = 1, tomo=False, plot=False, update=False)
        else:
            data = dephasingSweeps.spinEcho(sample, measure=measure, delay=st.r[0:tEnd:timeRes, ns], df=df,
                                            stats=stats, name='spinEcho Freq Tuner MQ', save=save, collect=True,
                                            randomize=False, averages=1, tomo=False, plot=False, update=False)
        fringe = fitting.maxFreq(data, nfftpoints, plot=plot)
        delta_freq = df - fringe
        if noisy:
            print 'Desired Fringe Frequency: %s' % df
            print 'Actual Fringe Frequency: %s' % fringe
            print 'Qubit frequency adjusted by %s' % delta_freq

        fkey = 'f'+str(state)+str(state-1)
        if update:
            ml.setMultiLevels(q,'frequency',ml.getMultiLevels(q,'frequency',state)-delta_freq,state)
            if noisy: print 'new resonance frequency: %g' % ml.getMultiLevels(q,'frequency',state)
            Q[fkey] = st.nearest(ml.getMultiLevels(q,'frequency',state)[GHz], 0.0001)*GHz
        else:
            if noisy: print 'new resonance frequency: %g' % (ml.getMultiLevels(q,'frequency',state)+delta_freq)

    return Q[fkey]

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

def nonlintuner(sample, iterations=1, tEnd=100*ns, timeRes=0.5*ns, nfftpoints=4000, stats=2400,
                measure=0, save=False, plot=False, noisy=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    # New Dec 2009: Automatically finds best f10, using ramsey can plot the FFT of the Ramsey fringe to extract true f10
    # works for sweeps with time steps that are all equivalent (i.e. not concatenated sweeps with diff time steps)

    # TODO make the choice of steps more physical/meaningful, currently set up for speed.
    # Time resolution should be at least at the Nyquist frequency, but better to oversample
    # nyfreq=float(fringeFreq)*2*10e6
    # timeRes = (1.0/float(nyfreq))*1e9
    # TODO make successive iterations finer resolution
    # TODO better way to detect frequency than just maximum value (peak fitting)
    if plot:
        fig = plt.figure()
    for i in xrange(iterations):
        data = ramseyFilter(sample, st.r[0:tEnd:timeRes,ns], stats=stats,
                            measure=measure, name='Nonlin Tuner MQ', collect=True, noisy=noisy, save=save)
        ts, ps = data.T
        y = ps - np.polyval(np.polyfit(ts, ps, 1), ts) # detrend
        timestep = ts[1] - ts[0]
        freq = np.fft.fftfreq(nfftpoints, timestep)
        fourier = abs(np.fft.fft(y, nfftpoints))
        fringe = abs(freq[np.argmax(fourier)])*1e3*MHz
        if plot:
            ax = fig.add_subplot(iterations,1,i)
            ax.plot(np.fft.fftshift(freq), np.fft.fftshift(fourier))
        print 'Actual Nonlinearity: %s' % -fringe
        print 'Current f21 %s' % q['f21']

        q['f21'] = q['f10'] - fringe
        print 'New f21: %g' % q['f21']

    Q['f21'] = st.nearest(q['f21'][GHz], 0.0001)*GHz
    return Q['f21']


def spectroscopy(sample, freq=None, stats=300L, measure=0, sb_freq=0*GHz, detunings=None, uwave_amp=None,
                 save=True, name='Spectroscopy MQ', collect=False, noisy=True, update=True):
    sample, qubits, Qubits = util.loadDeviceType(sample, 'phaseQubit', write_access=True)
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

    def func(server, amp, f):
        for i, (q, zpa) in enumerate(zip(qubits, zpas)):
            q['fc'] = f - sb_freq
            if zpa:
                q.z = env.rect(-100, qubits[measure]['spectroscopyLen'] + 100, zpa)
            else:
                q.z = env.NOTHING
            if i == measure:
                q['spectroscopyAmp'] = amp
                q.xy = eh.spectroscopyPulse(q, 0, sb_freq)
                q.z += eh.measurePulse(q, q['spectroscopyLen'])
        eh.correctCrosstalkZ(qubits)
        return runQubits(server, qubits, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if update:
        adjust.adjust_frequency(Q, data)
    if collect:
        return data


def spectroscopy_two_state(sample, freq=None, stats=300L, measure=0, sb_freq=0*GHz, detunings=None, uwave_amps=None,
                           save=True, name='Two-state finder spectroscopy MQ', collect=False, noisy=True, update=True):
    sample, qubits, Qubits = util.loadDeviceType(sample, 'phaseQubit', write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    q['readout'] = True
    if freq is None:
        f = st.nearest(q['f10'][GHz], 0.001)
        freq = st.r[f-0.20:f+0.05:0.0005, GHz]
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

    axes = [(freq, 'Frequency')]
    deps = [('Probability', '|1>, uwa=%g' % amp, '') for amp in uwave_amps]
    kw = {
        'stats': stats,
        'sideband': sb_freq
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
                    q['spectroscopyAmp'] = amp
                    q.xy = eh.spectroscopyPulse(q, 0, sb_freq)
                    q.z += eh.measurePulse(q, q['spectroscopyLen'])
            eh.correctCrosstalkZ(qubits)
            reqs.append(runQubits(server, qubits, stats, probs=[1]))
        probs = yield FutureList(reqs)
        returnValue([p[0] for p in probs])
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if update:
        adjust.adjust_frequency_02(Q, data)
        if len(sample['config']) is 1:
            adjust.adjust_fc(Q, data)
    if collect:
        return data


def spectroscopy2Dxtalk(sample, freqScan=None, measure=0, bias=0, stats=300L,
                        fluxBelow=2*mV, fluxAbove=2*mV, fluxStep=0.1*mV, sb_freq=0*GHz,
                        save=True, name='2D spectroscopy MQ', collect=False, noisy=True):
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
    return sweeps.run(func, sweep(), dataset=save and dataset, collect=collect, noisy=noisy)


def spectroscopy2DZxtalk(sample, freqScan=None, measAmplFunc=None, measure=0, z_pulse=1,
                         fluxBelow=0.5, fluxAbove=0.5, fluxStep=0.025, sb_freq=0*GHz, stats=300L, plot=False,
                         name='2D Z-pulse xtalk spectroscopy MQ', save=True, collect=False, update=True, noisy=True):
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
    sweeps.run(func, sweep(), dataset=save and dataset, collect=collect, noisy=noisy)

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

def amplify_phase_error(sample, phase=np.arange(0,2,0.01)*np.pi, numAmp=(0,1,3,5),
                        measure=0,stats=1200L, name='Amplify Phase Error', save=True,
                        collect=True, noisy=True, alpha=0.5, tBuf=4.0*ns, plot=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    state = 1
    name = '|%d> %s' %(state, name)

    if not np.iterable(numAmp):
        numAmp = [numAmp]
    axes = [(phase, 'ramsey phase'), (alpha, 'alpha')]
    deps = [('Probability', '|%d> (n=%d)' %(state, n), '') for n in numAmp]
    kw = {'stats': stats, 'alpha': alpha, 'tBuf': tBuf}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def ape(qubit, t0, n):
        qubit.xy = env.NOTHING
        for i in range(n):
            qubit.xy += eh.piHalfPulse(qubit, t0, phase=0, alpha=alpha)
            t0 += qubit.piLen + tBuf
            qubit.xy += eh.piHalfPulse(qubit, t0, phase=np.pi, alpha=alpha)
            t0 += qubit.piLen + tBuf
        return qubit.xy

    def func(server, phi, alpha):
        reqs =[]
        for n in numAmp:
            q.xy = eh.boostState(q, 0, state-1, alpha=alpha)
            t = (state-1)*q.piLen
            q.xy += eh.mix(q, eh.piHalfPulse(q, t, state=1, alpha=alpha))
            t = t+q.piLen
            q.xy += eh.mix(q, ape(q, t, n))
            t += 2*n*q.piLen + 2*n*tBuf
            q.xy += eh.mix(q, eh.piHalfPulse(q, t, phase=phi, alpha=alpha))
            q.z  = eh.measurePulse(q, t+q.piLen/2, state=state)
            q['readout'] = True
            reqs.append(runQubits(server, qubits, stats, probs=[1]))
        probs = yield FutureList(reqs)
        data = [p[0] for p in probs]
        if noisy: print "phase fraction: ", phi/np.pi
        returnValue(data)
    data = sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)

    def fit_cos(x, w, x0, amp, offset):
        return np.cos(w*(x-x0)) * amp + offset

    if plot:
        colors = 'rgbcmyk'
        x = data[:,0]
        xp = np.linspace(np.min(x), np.max(x), 10*len(x))
        new_data = data[:,1:].T
        ps = []
        plt.figure()
        for y, c in zip(new_data, colors):
            y_max, y_min = np.max(y), np.min(y)
            p0 = [1.0, 0.0, 0.5*(y_max-y_min), 0.5*(y_max+y_min)]
            p, cov = curve_fit(fit_cos, x, y, p0=p0)
            plt.plot(x/np.pi, y, c+'o')
            plt.plot(xp/np.pi, fit_cos(xp, *p), c)
            ps.append(p)
        plt.xlabel("phase [pi]")
        plt.ylabel("P1")
        plt.grid()
        return data, ps
    if collect:
        return data

def fidelity_piPulse(sample, alpha=np.linspace(-0.6,0.6,60, endpoint=True), measure=0, stats=1800L, name='piPulse fidelity',
        useAPE=False, numAPE=2, save=True, collect=True, plot=False, noisy=True, tBuf=0*ns):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    axes = [(alpha, 'alpha')]
    kw = {'stats': stats, 'useAPE': useAPE, 'numAPE': numAPE}

    measureFunc = measurement.Octomo(len(sample['config']), measure, tBuf=tBuf)
    dataset = sweeps.prepDataset(sample, name, axes, measure=measureFunc, kw=kw)

    def ape(qubit, t0, n, alpha):
        if not useAPE:
            return env.NOTHING
        qubit.xy = env.NOTHING
        for i in range(n):
            qubit.xy += eh.piHalfPulse(qubit, t0, phase=0, alpha=alpha)
            t0 += qubit.piLen
            qubit.xy += eh.piHalfPulse(qubit, t0, phase=np.pi, alpha=alpha)
            t0 += qubit.piLen
        return qubit.xy * useAPE

    def func(server, a):
        q['alpha'] = a
        q.xy = eh.boostState(q, 0, 1)
        t = q.piLen
        q.xy += eh.mix(q, ape(q, t, numAPE, alpha=a))
        t += numAPE*q.piLen*2*useAPE
        return measureFunc(server, qubits, t-q.piLen/2, stats=stats)

    data = sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)
    rho0 = np.array([[0,0],[0,1]],dtype=np.complex)
    alphas = data[:,0]
    probs = data[:,1:].reshape((-1,6,2))
    rhos = [tomo.qst(diags, 'octomo') for diags in probs]
    Fs =  np.array([fidelity(rho, rho0) for rho in rhos])

    if plot:
        fig = plt.figure()
        plt.subplot(2,1,1)
        octomoKeys = ['I','+X','+Y','-X','-Y','XPi']
        for i, key in enumerate(octomoKeys):
            plt.plot(alphas, probs[:,i,0])
        plt.legend(octomoKeys)
        plt.xlabel('alpha')
        plt.ylabel('Prob|1>')
        plt.subplot(212)
        plt.plot(alphas, Fs.reshape((Fs.size, 1)))
        plt.xlabel('alpha')
        plt.ylabel('Fidelity')

    if collect:
        return np.hstack((data, Fs.reshape((Fs.size, 1))))

def swapSpectroscopy(sample,
                     swapLen=st.r[0:500:5,ns], swapAmp=np.arange(-0.05,0.05,0.002),
                     overshoot = 0.0,
                     measure=0, stats=600L, name='Swap Spectroscopy',
                     save=True, collect=False, noisy=True, state=1, piPulse=True,
                     username=None, paramName=''):
    """Measures T1 vs z-pulse amplitude"""
    sample, qubits = util.loadDeviceType(sample,'phaseQubit')
    q = qubits[measure]
    ml.setMultiKeys(q, state)
    if swapAmp is None:
        swapAmp = q['swapAmp'+paramName]
    elif np.size(swapAmp) is 1:
        swapAmp = float(swapAmp)
    else:
        swapAmp = swapAmp[np.argsort(np.abs(swapAmp))]
    if state >1:
        name='|'+str(state)+'> '+name

    axes = [(swapAmp, 'swap pulse amplitude'), (swapLen, 'swap pulse length'), (overshoot, 'overshoot')]
    deps = [('Probability', '|'+str(state)+'>', '')]
    kw = {'stats': stats, 'state': state}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, currAmp, currLen, currOvershoot):
        t=0.0*ns
        #It can be useful to do swap spectroscopy without exciting the qubit to see
        #if the microwave carrier is leaking into your system.
        if piPulse:
            #q.xy = eh.boostState(q, 0, state=state)
            q.xy = eh.boostState(q, q['piLen']/2, state=state)
            t+=q['piLen']*state
        #q.z = env.rect(q['piLen']*(state-0.5), currLen, currAmp) + eh.measurePulse(q, q['piLen']*(state-0.5) + currLen + 10*ns, state=state)
        q.z = env.rect(t,currLen,currAmp,overshoot=currOvershoot)
        t += currLen#+10*ns
        q.z += eh.measurePulse(q, t, state=state)
        q['readout'] = True
        return runQubits(server, qubits, stats=stats, probs=[1])

    data = sweeps.grid(func, axes, save=save, dataset=dataset, collect=collect, noisy=noisy)
    if username is not None:
        with labrad.connect() as cxn:
            try:
                cxn.telecomm_server.send_sms('Scan complete','Swap spectroscopy is complete',username)
            except:
                print 'Failed to send text message'
    if collect:
        return data


def datasetMinimum(data, default, llim, rlim, dataset=None):
    coeffs = np.polyfit(data[:,0],data[:,1],2)
    if coeffs[0] <= 0:
        print 'No minimum found, keeping value'
        return default, np.polyval(coeffs, default)
    result = np.clip(-0.5*coeffs[1]/coeffs[0],llim,rlim)
    return result, np.polyval(coeffs, result)


def swapTuner(sample, paramName, measure=0, swapAmp=None, swapTime=None, iterations=3,
                  ampBound=0.025,timeBound=10.0*ns, state=1,
                  save=False,update=True,stats=1200,noisy=False):
    """Finds the best qubit z-pulse amplitude and pulse length to do a swap into a resonator"""
    name = 'Swap Tuner |%d>' %state
    sample, qubits, Qubits = util.loadDeviceType(sample, 'phaseQubit', write_access=True)
    qubit=qubits[measure]
    Qubit=Qubits[measure]
    paramName += str(state) if state>1 else ''
    if swapAmp is None:
        swapAmp = qubit['swapAmp'+paramName]
    if swapTime is None:
        swapTime= qubit['swapTime'+paramName]
    if noisy:
        print 'Original swap amplitude: %f' %swapAmp
        print 'Original swap time: %f ns' %swapTime['ns']
    for iteration in range(iterations):
        dataStats=stats*(2**iteration)
        dSwapAmp = ampBound/(2.0**iteration)
        dSwapTime = (timeBound/(2.0**iteration))['ns']
        data = swapSpectroscopy(sample,
                                swapLen=swapTime,
                                swapAmp=np.linspace(swapAmp-dSwapAmp,swapAmp+dSwapAmp,21),
                                measure=measure, state=state, collect=True, save=save,
                                stats=dataStats, noisy=noisy, name=name)
        swapAmp,probability = fitting.findMinimum(np.array(data),fit=True)
        if noisy: print 'Best swap amplitude: %f' %swapAmp
        data = swapSpectroscopy(sample,
                                swapLen=structures.ValueArray(np.linspace(swapTime-dSwapTime,swapTime+dSwapTime,21),'ns'),
                                swapAmp=swapAmp,
                                measure=measure, state=state, collect=True, save=save,
                                stats=dataStats, noisy=noisy, name=name)
        swapTime,probability = fitting.findMinimum(np.array(data),fit=True)
        swapTime = swapTime*ns
        if noisy: print 'Best swap time: %f ns' %swapTime[ns]
    if update:
        Qubit['swapAmp'+paramName]=swapAmp
        Qubit['swapTime'+paramName]=swapTime
    return swapTime,swapAmp


def fockScan(sample, n=1, scanLen=0.0*ns, probeFlag=False,stats=1500L, measure=0, delay=0*ns,
       name='Fock state swap length scan MQ', save=False, collect=True, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    axes = [(scanLen, 'Swap Control length adjust')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample,name,axes, measure=measure, kw=kw)

    def func(server, currLen):
        q.xy = env.NOTHING
        q.z = env.NOTHING
        start = -q.piLen/2

        #Excite qubit
        q.xy += eh.mix(q, eh.piPulseHD(q, start+q.piLen/2))
        start += q.piLen+delay

        if not probeFlag:
            q.z += env.rect(start, q['cZControlLen']+currLen,  q['cZControlAmp'])
            start += q['cZControlLen']+currLen+delay
            q.z += eh.measurePulse(q, start)
        else:
            q.z += env.rect(start, q['cZControlLen']/2, q['cZControlAmp'])
            start += q['cZControlLen']/2+delay
            q.z += env.rect(start, currLen, q['cZControlAmp'])
            start += currLen+delay
            q.z += eh.measurePulse(q, start)

        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)


def fockScan21(sample, n=1, scanLen=0.0*ns, scanOS=0.0, tuneOS=False, probeFlag=False, stats=1500L, measure=0, delay=0*ns,
       name='Fock state 21 swap length scan MQ', save=False, collect=True, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    sl = q['cZTargetLen']/2 #Target len is for iSWAP^2
    sa = q['cZTargetAmp']
#
#    if not tuneOS:
#        so = np.array([0.0]*n)
#        scanOS = 0.0
#    else:
#        so = q['noonSwapAmp'+paraName+'OSs']

    axes = [(scanLen, 'Target Swap length adjust')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, 'name', axes, measure=measure, kw=kw)

    def func(server, currLen):
        q.xy = eh.mix(q, eh.piPulseHD(q, -q.piLen/2))
        q.z = env.NOTHING
        start = 0
        q.xy += eh.mix(q, env.gaussian(start+q.piLen/2, q.piFWHM, q.piAmp21, df=q.piDf21), freq = 'f21')
        start += q.piLen+delay
        if not probeFlag:
            q.z += env.rect(start, sl+currLen, sa)
            start += sl+currLen+delay
            q.z += eh.measurePulse2(q, start)
        else:
            q.z += env.rect(start, sl, sa)
            start += sl+delay
            q.z += env.rect(start, currLen, sa)
            start += currLen+delay
            q.z += eh.measurePulse2(q, start)

        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)


def fockTunerLevel(sample, n=1, scanLen=st.arangePQ(0,300,2,'ns'), iteration=3, tuneOS=False, stats=1500L, measure=0, delay=0*ns,
       save=False, collect=True, noisy=True, update=True, level=1):
    sample, qubits, Qubits = util.loadQubits(sample,write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]



#    if tuneOS:
#        if sample['q'+str(measure)]['noonSwapAmp'+paraName+'OSs'][0] == 0.0:
#            sample['q'+str(measure)]['noonSwapAmp'+paraName+'OSs'] = [abs(q['noonSwapAmp'+paraName])]*n
#        print 'Use preset overshoot values'

    for iter in range(iteration):
        rf = 2**iter
        print 'iteration %g...' % iter
        if level==1:
            sl = q['cZControlLen']
            results = fockScan(sample, n=1, scanLen=st.PQlinspace(-max([0.3*sl['ns']/rf,1]),max([0.3*sl['ns']/rf,1]),21,'ns'),
                                    stats=stats,measure=measure,probeFlag=False,delay=delay,
                                    save=False, collect=collect, noisy=noisy)
            newLen, percent = datasetMinimum(results, 0, -max([0.3*sl['ns']/rf,1]), max([0.3*sl['ns']/rf,1]))
            q['cZControlLen'] += newLen

            print 'New swap length is %g' % q['cZControlLen']
            print '.................'
            if update:
                Q['cZControlLen'] = q['cZControlLen']
        elif level==2:
            sl = q['cZTargetLen']
            results = fockScan21(sample, n=1, scanLen=st.PQlinspace(-max([0.3*sl['ns']/rf,1]),max([0.3*sl['ns']/rf,1]),21,'ns'),
                                    stats=stats,measure=measure,probeFlag=False,delay=delay,
                                    save=False, collect=collect, noisy=noisy)
            newLen, percent = datasetMinimum(results, 0, -max([0.3*sl['ns']/rf,1]), max([0.3*sl['ns']/rf,1]))
            q['cZTargetLen'] += newLen

            print 'New swap length is %g' % q['cZTargetLen']
            print '.................'
            if update:
                Q['cZTargetLen'] = q['cZTargetLen']

#        if tuneOS:
#            os = sample['q'+str(measure)]['noonSwapAmp'+paraName+'OSs'][i-1]
#            if level==1:
#                results = FockScan(sample, n=i, scanLen=0.0*ns, tuneOS=tuneOS, scanOS=np.linspace(os*(1-0.5/rf),os*(1+0.5/rf),21),
#                                        paraName=paraName,stats=stats,measure=measure,probeFlag=False,delay=delay,
#                                        save=False, collect=collect, noisy=noisy)
#            elif level==2:
#                results = FockScan21(sample, n=i, scanLen=0.0*ns, tuneOS=tuneOS, scanOS=np.linspace(os*(1-0.5/rf),os*(1+0.5/rf),21),
#                                        paraName=paraName,stats=stats,measure=measure,probeFlag=False,delay=delay,
#                                        save=False, collect=collect, noisy=noisy)
#            new, percent = datasetMinimum(results, os, os*(1-0.5/rf), os*(1+0.5/rf))
#            sample['q'+str(measure)]['noonSwapAmp'+paraName+'OSs'][i-1] = new
#
#            print 'New swap overshoot is %g' % sample['q'+str(measure)]['noonSwapAmp'+paraName+'OSs'][i-1]
#            print '.................'

        if save:
            if level==1:
                fockScan(sample, n=1, scanLen=scanLen,
                            stats=stats,measure=measure,probeFlag=True,delay=delay,
                            save=save, collect=collect, noisy=noisy)
            elif level==2:
                fockScan21(sample, n=1, scanLen=scanLen,
                            stats=stats,measure=measure,probeFlag=True,delay=delay,
                            save=save, collect=collect, noisy=noisy)


def fockTuner(sample, n=1, iteration=3, tuneOS=False,stats=1500L, measure=0, delay=0*ns,
       save=False, paramName='', collect=True, noisy=True, update=True):
    sample, qubits, Qubits = util.loadQubits(sample,write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]

#    if len(q['cZControlLen'])<n:
#        for i in np.arange(len(q['noonSwapLen'+paraName+'s']),n,1):
#            sample['q'+str(measure)]['noonSwapLen'+paraName+'s'].append(q['noonSwapLen'+paraName+'s'][0]/np.sqrt(i+1))

    for iter in range(iteration):
        rf = 2**iter
        print 'iteration %g...' % iter
        sl = q[paramName+'SwapTime']
        results = fockScan(sample, n=1, scanLen=st.PQlinspace(-max([0.3*sl['ns']/rf,1]),max([0.3*sl['ns']/rf,1]),21,'ns'),
                                stats=stats,measure=measure,probeFlag=False,delay=delay,
                                save=False, collect=collect, noisy=noisy)
        newLen, percent = datasetMinimum(results, 0, -max([0.3*sl['ns']/rf,1]), max([0.3*sl['ns']/rf,1]))
        q['cZControlLen'] += newLen

#        if tuneOS:
#            os = sample['q'+str(measure)]['noonSwapAmp'+'OSs'][i-1]
#            results = fockScan(sample, n=1, scanLen=0.0*ns, tuneOS=tuneOS, scanOS=np.linspace(os*(1-0.5/rf),os*(1+0.5/rf),21),
#                                    stats=stats,measure=measure,probeFlag=False,delay=delay,
#                                    save=False, collect=collect, noisy=noisy)
#            newLen, percent = datasetMinimum(results, 0, -max([0.3*sl['ns']/rf,1]), max([0.3*sl['ns']/rf,1]))
#            q['cZControlLen'] += newLen

        if save:
            fockScan(sample, n=1, scanLen=st.arangePQ(0,300,2,'ns'),
                        stats=stats,measure=measure,probeFlag=True,delay=delay,
                        save=save, collect=collect, noisy=noisy)
    if update:
        Q['cZControlLen'] = q['cZControlLen']
    return q['cZControlLen']


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
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if collect:
        return result


def testdelay(sample, t0=st.r[-40:40:0.5,ns], zpa=-0.1, zpl=20*ns, tm=65*ns,
              measure=0, stats=3000, update=True,
              save=True, name='Test Delay', plot=False, noisy=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    axes = [(t0, 'Detuning pulse center')]
    kw = {
        'stats': stats,
        'detuning pulse length': zpl,
        'detuning pulse height': zpa,
        'start of measurement pulse': tm
    }
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, t0):
        q.xy = eh.mix(q, eh.piPulse(q, 0))
        q.z = env.rect(t0-zpl/2.0, zpl, zpa) + eh.measurePulse(q, tm)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

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
    print 'uwave lag:', -fit[0]
    if update:
        print 'uwave lag corrected by %g ns' % -fit[0]
        Q['timingLagUwave'] -= fit[0]*ns


def testdelay_x(sample, t0=st.r[-40:40:0.5,ns], zpa=-1.5, zpl=20*ns, tm=65*ns,
                measure=0, z_pulse=1, stats=1200, update=True,
                save=True, name='Test Delay', plot=False, noisy=True):
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
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

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


def insensitive(sample, bias=st.r[-500:500:50, mV], detuning=st.r[-50:50:2,MHz], sb_freq=0*GHz,
                stats=300L, measure=0, save=True, name='Insensitive Points', noisy=True, collect=False):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    axes = [(bias, 'Squid Bias'), (detuning, 'Detuning')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, bias, df):
        for qubit in qubits:
            qubit['fc'] = q['f10'] + df - sb_freq # qubits share same microwave generator
        q['squidBias'] = bias
        q.xy = eh.spectroscopyPulse(q, 0, sb_freq)
        q.z = eh.measurePulse(q, q['spectroscopyLen'])
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if collect:
        return data


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
    return sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)


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
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    if update:
        t = adjust.adjust_time(data)
        if t is not None:
            print 'timing lag corrected by %g ns' % t
            Qa['timingLagWrtMaster'] += t*ns


def t1(sample, delay=st.r[-10:1000:2,ns], stats=600L, measure=0,
       name='T1 MQ', save=True, collect=True, noisy=True, state=1,
       update=True, plot=True):
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

    axes = [(delay, 'Measure pulse delay')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, delay):
        q.xy = eh.boostState(q, 0, state)
        q.z = eh.measurePulse(q, delay, state=state)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if (plot or update) and save:
        with labrad.connect() as cxn:
            dv = cxn.data_vault
            dataset = dstools.getOneDeviceDataset(dv,-1,session=sample._dir,
                                                  deviceName=None,averaged=False,
                                                  correctVisibility=True)
        fitResult = fitting.t1(dataset,timeRange=(10*ns,delay[-1]))
    if plot:
        fig = dstools.plotDataset1D(dataset.data,
                                    dataset.variables[0],dataset.variables[1],
                                    marker='.',markersize=15)
        ax = fig.get_axes()[0]
        ax.plot(dataset.data[fitResult.indexList,0],dataset.data[fitResult.indexList,1],'r.', markersize=15)
        ax.plot(dataset.data[:,0],fitResult['fitFunc'](dataset.data[:,0],*fitResult['fitParams']),'k')
        ax.grid()
        ax.set_title(dataset['path'])
        fig.show()
    if update:
        Q['calT1']=fitResult['T1']
    return data


def ramseyFilter(sample, delay=st.r[5:50:1,ns], theta=0, measure=0, stats=900L,
                 name='Ramsey Error Filter', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    ml.setMultiKeys(q, 2)

    axes = [(delay, 'Pulse Separation Delay'), (theta, 'Phase of 2nd Pulse')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, delay, theta):
        q.xy = eh.mix(q, eh.piPulse(q, 0) + eh.piPulse(q, delay))
        #q.z = eh.measurePulse(q, delay+q['piLen']/2.0, state=2)
        q.z = eh.measurePulse2(q, delay + q['piLen']/2.0) # measure 2-state population
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if collect:
        return data


def ramseyZPulse(sample, zAmp=None, measure=0, stats=900L,
                 measureDelay = 2.0*ns, numPulses=1, tBuf=0.0*ns,
                 name='Z-Pulse Ramsey Fringe', save=True, collect=False,
                 update=False, noisy=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]
    q['readout'] = True

    if zAmp is None:
        if numPulses > 1:
            zAmpMin = 0.85*q['piAmpZ']
            zAmpMax = 1.15*q['piAmpZ']
            zAmp = np.linspace(zAmpMin,zAmpMax,50)
        else:
            zAmp = np.linspace(0,0.1,50)

    axes = [(zAmp, 'Z-Pulse Amplitude')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, zAmp):
        q['piAmpZ'] = zAmp
        q['xy'] = env.NOTHING
        q['z'] = env.NOTHING
        t = 0.0*ns
        #Put qubit in equator of Bloch sphere
        q['xy'] += eh.piHalfPulse(q,t)
        t += q['piLen']/2.0
        #Do z pulses. Nominally pi pulses, but may not be, as we are scanning
        #the amplitude needed for a z pi pulse
        t += tBuf
        for _ in range(numPulses):
            q['z'] += eh.rotPulseZ(q, t+q['piLenZ']/2.0, angle=np.pi)
            t += q['piLenZ']
        t += tBuf
        #Second Ramsey pulse
        q['xy'] += eh.piHalfPulse(q, t+q['piLen']/2.0)
        t += q['piLen']
        #Measure the qubit
        t += tBuf
        q['z'] += eh.measurePulse(q, t+measureDelay)
        #Apply sideband mixing
        q['xy'] = eh.mix(q, q['xy'])
        return runQubits(server, qubits, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if update:
        adjust.adjust_piZ(Q, data)
    if collect:
        return data
