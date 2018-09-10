import numpy as np
from scipy.optimize import leastsq
from scipy.special import erf, erfc
import matplotlib.pyplot as plt

from labrad.units import Unit,Value
V, mV, us, ns, GHz, MHz, dBm, rad = [Unit(s) for s in ('V', 'mV', 'us', 'ns', 'GHz', 'MHz', 'dBm', 'rad')]

import pyle.envelopes as env
from pyle.pipeline import returnValue, FutureList
from pyle.dataking import envelopehelpers as eh
from pyle.dataking import measurement
from pyle.dataking import adjust
from pyle.dataking.fpgaseq import runQubits
from pyle.dataking.quasiparticle import dephasingSweeps_swap
from pyle.util import sweeptools as st
from math import atan2
from pyle.dataking import utilMultilevels as ml
from pyle.plotting import dstools
from pyle.fitting import fitting

import qubitpulsecal as qpc
import sweeps
import util
import labrad

def squidsteps(sample, bias=st.r[-2.5:2.5:0.05, V], resets=(-2.5*V, 2.5*V), measure=0, stats=150,
               save=True, name='SquidSteps MQ', collect=False, noisy=False, update=True):
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

    def func(server, fb):
        reqs = []
        for reset in resets:
            q['biasOperate'] = fb
            q['biasReadout'] = fb
            q['biasReset'] = [reset]
            q['readout'] = True
            reqs.append(runQubits(server, qubits, stats, raw=True))
        data = yield FutureList(reqs)
        if noisy: print fb
        returnValue(np.vstack(data).T)
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=False)

    if update:
        adjust.adjust_squid_steps(Q, data)
    if collect:
        return data


def s_scanning(sample, freq_range=st.r[4.382:4.386:0.00001, GHz], power=-60*dBm, bias=0*V, sb_freq = -50*MHz, measure=0, stats=150,
               save=True, name='S parameter scanning', collect=False, noisy=True, update=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    #q, Q = qubits[measure], Qubits[measure]

    axes = [(freq_range, 'Frequency')]
    deps = [('Phase', 'S11 for %s'%q.__name__, rad) for q in qubits]+ [('Amplitude','S11 for %s'%q.__name__,'') for q in qubits]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    for q in qubits:
        q['biasResetSettling']=5*us
        q['biasOperateSettling']=5*us
        q['biasReadoutSettling']=10*us
        q['biasOperate'] = bias
        q['biasReadout'] = bias
        q['biasReset'] = bias
        q['readout'] = True
        q['readout power'] = power


    def func(server, freq):
        amp = []
        phase = []
        for q in qubits:
            q['readout frequency']=freq
            q['readout fc'] = q['readout frequency'] - sb_freq
        if noisy: print freq
        data = yield FutureList([runQubits(server, qubits, stats, raw=True)])
        for q in qubits:
            I = np.mean(data[0][0][0][q['adc channel']::11])
            Q = np.mean(data[0][0][1][q['adc channel']::11])
            amp.append(abs(I+1j*Q))
            phase.append(atan2(Q,I))
        returnValue(phase + amp)
        #returnValue(runQubits(server, qubits, stats, raw=True, debug=True))


    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=False)
    if update:
        for i in range(len(Qubits)):
            phase = np.asarray([[row[0], row[i+1]] for row in data])
            adjust.adjust_s_scanning(Qubits[i], phase, sb_freq)
    if collect:
        return data


def phase_arc(sample, bias=st.r[-2.5:2.5:0.05, V], resets=(-2.5*V, 2.5*V), measure=0, stats=150,
               save=True, name='Phase Arc', collect=False, noisy=False, update=True, average=False):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    if 'qubitBiasLimits' in q:
        default = (-2.5*V, 2.5*V)
        bias_lim = q['squidBiasLimits']
        if bias_lim != default:
            print 'limiting bias range to (%s, %s)' % tuple(bias_lim)
        resets = max(resets[0], bias_lim[0]), min(resets[1], bias_lim[1])
        bias = st.r[bias_lim[0]:bias_lim[1]:bias.range.step, V]

    axes = [(bias, 'Flux Bias')]
    deps = []
    for q in qubits:
        deps = deps + [('S11 Phase', 'Reset: %s Qubit: %s' % (reset, q.__name__), rad) for reset in resets]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, fb):
        reqs = []
        if noisy: print "fb = %s" % fb
        for reset in resets:
            for q in qubits:
                q['biasOperate'] = fb
                q['biasReadout'] = fb
                q['biasReset'] = [reset]
                q['readout'] = True
            data = yield FutureList([runQubits(server, qubits, stats, raw=True)])
            for q in qubits:
                Is = data[0][0][0][q['adc channel']::11]
                Qs = data[0][0][1][q['adc channel']::11]
                if average:
                    Is = [np.mean(Is)]
                    Qs = [np.mean(Qs)]
            reqs.append([atan2(Qs[i], Is[i]) for i in range(len(Is))])
        returnValue(np.vstack(reqs).T)
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=False)

    if update:
        for i in range(len(Qubits)):
            print i
            data_q = np.asarray([[row[0], row[2*i+1], row[2*i+2]] for row in data])
            adjust.adjust_phase_arc(Q, data_q)
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


def scurve(sample, mpa=st.r[0:2:0.05], stats=300, measure=0, states=0,
           visibility=False, calstates=None,
           save=True, name='SCurve MQ', collect=True, noisy=True, update=True):
    """S-Curve sequence on one qubit. Displays S-Curves for states given in state.
    If update is selected, will enable changing mpa. If calstate is selected, will
    produce registry keys giving all state probs for given mpa.

    PARAMETERS
    sample: Object defining qubits to measure, loaded from registry.
    mpa - iterable: Measure pulse amplitude.
    stats - scalar: Number of times a point will be measured.
    measure - scalar: Number of qubit to measure. Only one qubit allowed.
    states - list or int: State(s) for which to measure S-curve.
    visibility - bool: Whether to calculate visibilities.
    calstates - list or int: State(s) for whose measure pulse amplitude(s) to create
        calibration keys in registry.
    save - bool: Whether to save data to the datavault.
    name - string: Name of dataset.
    collect - bool: Whether or not to return data.
    noisy - bool: Whether to print out probabilities while the scan runs
    update - bool: Whether to display a popup to update the mpas in registry. This
        option is set to False if calstate=True.
    """
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    def makelist(vals,name):
        if isinstance(vals,list):
            return vals
        elif isinstance(vals,tuple):
            return list(vals)
        elif isinstance(vals,int) or isinstance(vals,float):
            return [int(vals)]
        else:
            raise Exception('What the heck data type did you use? Try making '+name+' a list or int.')

    states=makelist(states,'states')
    minstate = min(states)
    maxstate = max(states)
    ml.setMultiKeys(q,max([maxstate,1]))
    if visibility:
        if len(states)>1:
            # set state to [minstate-1,minstate...,maxstate]
            # Use maxstate+1 since np.arange(m,n+1) gives [m,...,n]
            states = list(np.arange(minstate,maxstate+1))
        elif states[0]>0:
            states = [states[0]-1,states[0]]
        else:
            states = [0,1]
        if name=='SCurve MQ':
            name = 'Visibility MQ'
    if calstates is not None:
        calstates=makelist(calstates,'calstates')
        if min(calstates)<1:
            raise Exception('All calstates must be at least 1.')
        # Set state to [0,1,...,max], where max is max of maxstate or highest calstate.
        # Use maxstate+1 since range(n+1) goes up to n
        states = range(max(max(calstates),maxstate)+1)
    ml.setMultiKeys(q,max(max(states),1))
    if calstates is not None:
        mpa = [ml.getMultiLevels(q,'measureAmp',calstate) for calstate in calstates]
        update = False

    axes = [(mpa, 'Measure pulse amplitude')]
    deps =[('Probability', '|'+str(state)+'>', '') for state in states]
    if visibility:
        deps = deps + [('Visibility', '|'+str(state)+'>-|'+str(state-1), '') for state in states[1:]]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, mpa):
        t_meas = max(states)*q['piLen']
        reqs =[]
        for state in states:
            # Measure each state up to state
            ml.setMultiLevels(q,'measureAmp',mpa,state=1)
            q['readout'] = True
            swap = q['cZControlLen']+q['piLen']/2.0
            q.xy = eh.boostState(q,swap,state)
            q.z = env.rect(0, q['cZControlLen'], q['cZControlAmp'])+eh.measurePulse(q, swap+t_meas)
            reqs.append(runQubits(server, qubits, stats, probs=[1]))
        probs = yield FutureList(reqs)
        problist = [p[0] for p in probs]
        if visibility:
            problist = problist + list(np.array(problist[1:])-np.array(problist[:-1]))
        returnValue(problist)
    data = sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)
    if update:
        if visibility:
            adjust.adjust_visibility2(Q, data, states)
        else:
            adjust.adjust_scurve2(Q, data, states)
    if calstates is not None:
        for num,calstate in enumerate(calstates):
            Q['calScurve'+str(calstate)]=data[num,1:(2+states[-1])]
    if collect:
        return data


def visibility(sample, mpa=st.r[0:2:0.05], stats=300, calstats=1500, measure=0, states=1,
           save=True, name='Visibility MQ', collect=True, noisy=True, update=True):
    """S-Curve sequence on one qubit. Displays S-Curves and visibilities for all mpas
    given in state. If update is selected, will enable changing mpa. At end, will
    produce registry keys giving all state probs for given mpa.

    PARAMETERS
    sample: Object defining qubits to measure, loaded from registry.
    states - list or int: State(s) for whose measure pulse amplitude(s) to measure
        visibilities.
    mpa - iterable: Measure pulse amplitude.
    stats - scalar: Number of times a point will be measured.
    stats - scalar: Number of times a point will be measured for calibrations.
    measure - scalar: Number of qubit to measure. Only one qubit allowed.
    save - bool: Whether to save data to the datavault.
    name - string: Name of dataset.
    collect - bool: Whether or not to return data.
    noisy - bool: Whether to print out probabilities while the scan runs
    update - bool: Whether to display a popup to update the mpas in registry.
    """
    def makelist(vals,name):
        if isinstance(vals,list):
            return vals
        elif isinstance(vals,tuple):
            return list(vals)
        elif isinstance(vals,int) or isinstance(vals,float):
            return [int(vals)]
        else:
            raise Exception('What the heck data type did you use? Try making '+name+' a list or int.')

    states=makelist(states,'states')
    minstate = min(states)
    maxstate = max(states)
    # set state to [minstate-1,minstate...,maxstate]
    # Use maxstate+1 since np.arange(m,n+1) gives [m,...,n]
    states = list(np.arange(max(minstate-1,0),maxstate+1))
    calstates = list(np.arange(1,maxstate+1))

    data = scurve(sample,states=states, mpa=mpa,stats=stats,measure=measure,visibility=True,
                  calstates=None,save=save,name=name,collect=True,noisy=noisy,update=update)
    data2 = scurve(sample,states=states, calstates=calstates, mpa=mpa,stats=calstats,measure=measure,
                   visibility=False,save=save,name=name+'Cals',collect=True,noisy=noisy)

    if collect:
        return data, data2


def find_step_edge(sample, stats=60, target=0.5, bias_range=None,
                   measure=0, resolution=0.1, blowup=0.05,
                   falling=None, statsinc=1.25,
                   save=False, name='StepEdge Search MQ', collect=False, update=True, noisy=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    axes = [('Flux Bias', 'mV')]
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure)

    if falling is None:
        falling = q['biasOperate'][V] > q['biasStepEdge'][V]
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
        fb, stats = args
        q['biasOperate'] = fb*mV
        q['readout'] = True
        prob = yield runQubits(server, qubits, stats, probs=[1])
        if (prob[0] > target) ^ falling:
            interval[1] = min(fb, interval[1])
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
        swap = q['cZControlLen']+q['piLen']/2.0
        q.xy = eh.boostState(q, swap, state-1)
        q.z = env.rect(0, q['cZControlLen'], q['cZControlAmp'])+eh.measurePulse(q, swap+max(0.0,state-1)*q['piLen'], state=state)
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


def find_final_mpa(sample, stats=60, mpa_range=None, find_center=True, state=1, initBlowup = 0.1,
             measure=0, resolution=0.005, blowup=0.05, statsinc=1.25, midpoints=5, centerCut=2, target=0.05,
             save=False, name='SCurve Search MQ', collect=True, update=True, noisy=True):
    """Revises mpa based on maximizing visibility. After each iteration, finds points with
    visibilities at least the median and reduces range of mpas measured to these median values.

    PARAMETERS
    sample: Object defining qubits to measure, loaded from registry.
    stats - scalar: Number of times the first set of points will be measured.
            Will increase with each iteration.
    mparange - scalar list: initial range of mpas to look at, in form [min,max].
    find_center - bool: Whether initial guess for mpa comes from find_mpa. If
                  not, comes from registry.
    state - scalar: State for which to find mpa.
    initBlowup - scalar: Sets width of initial mpa range compared to initial guess.
    measure - scalar: Number of qubit to measure. Only one qubit allowed.
    state - scalar: State for which to find mpa.
    resolution - scalar: Maximum width of mpa range for which will accept final mpa.
    blowup - scalar: How much to increase reduced mpa range before next iteration.
    statsinc - scalar: How much to increase stats before next iteration.
    midpoints - scalar: How many points to use between the mpa range endpoints for
                measuring visibility.
    centerCut - scalar: How many times to have all mpas with visibilities above the
                median before setting mpa range to be between mpas on either side of
                mpa with peak visibility.
    target - scalar: Value of target probability for find_mpa, used for initial guess.
    save - bool: Whether or not to save data to the datavault.
    name - string: Name of dataset.
    collect - bool: Whether or not to return data.
    noisy - bool: Whether or not to print out probabilities while the scan runs.
    update - bool: Whether or not to update the registry.
    """
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    ml.setMultiKeys(q,state)

    axes = [('Measure Pulse Amplitude', '')]
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure)

    # Set initial range of mpas for which to measure visibility.
    if mpa_range is None:
        # Use find_mpa to determine inital guess for mpa.
        mpamp = find_mpa(sample,state=state,measure=measure,noisy=noisy,target=target) if find_center else ml.getMultiLevels(q,'measureAmp',state)
        mpa_range = [(1-initBlowup)*mpamp,(1+initBlowup)*mpamp]
    interval = [min(mpa_range),max(mpa_range)]
    centerlow = 0

    if noisy: print 'Interval: ' + str(interval)

    while interval[1]-interval[0]>resolution:
        #Continue measuring visibilities until width of mpa range is less than the desired resolution.

        #Increase number of stats from previous iteration and define points for measuring visibility.
        stats = min(int((stats+29)/30)*30, 30000)

        #Use visibility to measure visibilities at desired points in current mpa range.
        intstep = (interval[1]-interval[0])/float(midpoints+1)
        visresults = scurve(sample, mpa=st.r[interval[0]:interval[1]:intstep], visibility=True, update=False, stats=stats, measure=measure, states=state, save=False, noisy=noisy, collect=True)
        mpas = visresults[:,0]
        visibilities = visresults[:,3]

        #Determine which points in mpa range have visibilities at least the median. If both endpoints
        #satisfy this condition, increase the centerlow counter and proceed with the same mpa range.
        #Otherwise, reduce the mpa range to just the high-visibility points
        indices = visibilities>=np.median(visibilities)
        elems = np.nonzero(indices)[0]
        if (min(elems)==0) and (max(elems)==(midpoints+1)):
            centerlow += 1
            if noisy: print 'centerlow='+str(centerlow)
        else:
            mpacut = mpas[indices]
            interval = [min(mpacut),max(mpacut)]
            centerlow = 0
        if centerlow == centerCut:
            maxmpa = mpas[visibilities == max(visibilities)][0]
            interval = [maxmpa-intstep,maxmpa+intstep]
            centerlow = 0

        inc = blowup * (interval[1] - interval[0])
        interval[0] -= inc
        interval[1] += inc
        stats = statsinc * stats
        if noisy: print 'New interval: ' + str(interval)

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
    f = f0 (1-x)^{1/4} where c is a constant and
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
    V' = p[0] + p[1]f^4
    """
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    qubit, Qubit = qubits[measure], Qubits[measure]

    if measAmplFunc is None:
        measAmplFunc = get_mpa_func(qubit)
    #Set up the frequency scan
    if freqScan is None:
        freq = st.nearest(qubit['f10'][GHz], 0.001)
        dfs = np.logspace(-3, 0, 25)
        freqScan = freq + np.hstack(([0], dfs, -dfs))
    elif isinstance(freqScan, tuple) and len(freqScan) == 2:
        freq = st.nearest(qubit['f10'][GHz], 0.001)
        rng = freqScan
        dfs = np.logspace(-3, 0, 25)
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
        returnValue([flux, freq, prob])
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


def find_zpa_func(sample, freqScan=None, measure=0,
                  fluxBelow=0.01, fluxAbove=0.01, fluxStep=0.0005, sb_freq=0*GHz, stats=300L,
                  name='ZPA func search MQ', save=True, collect=False, noisy=True, update=True,
                  plot=False):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    qubit = qubits[measure]
    if freqScan is None:
        freq = st.nearest(qubit['f10'][GHz], 0.001)
        dfs = np.logspace(-3, 0, 25)
        freqScan = freq + np.hstack(([0], dfs, -dfs))
    elif isinstance(freqScan, tuple) and len(freqScan) == 2:
        freq = st.nearest(qubit['f10'][GHz], 0.001)
        rng = freqScan
        dfs = np.logspace(-3, 0, 25)
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
        returnValue([zpa, freq, prob])
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


def findDfDV(sample, freqScan=None, measAmplFunc=None, measure=0,
                   fluxBelow=2*mV, fluxAbove=2*mV, fluxStep=0.1*mV, sb_freq=0*GHz, stats=300L,
                   save=True, name='Find DfDV', collect=False, update=True, noisy=True, plot=True):
    """Find a linear fit to the 2D spectroscopy curve near the current operation point.
    This will update qubit['calDfDVbias'] with the current qubit frequency sensitivity
    to bias voltage.
    """
    if plot is True:
        raise Exception('Plotting code does not work, fix it!')
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    qubit, Qubit = qubits[measure], Qubits[measure]

    if measAmplFunc is None:
        measAmplFunc = get_mpa_func(qubit)

    if freqScan is None:
        freq = st.nearest(qubit['f10'][GHz], 0.001)
        freqScan = np.arange(freq-0.05, freq+0.05, 0.002)
    else:
        freqScan = np.array(f[GHz] for f in freqScan)
    freqScan = freqScan[np.argsort(abs(freqScan-qubit['f10'][GHz]))]

    fluxBelow = fluxBelow[V]
    fluxAbove = fluxAbove[V]
    fluxStep = fluxStep[V]
    fluxScan = np.arange(-fluxBelow, fluxAbove, fluxStep)
    fluxScan = fluxScan[np.argsort(abs(fluxScan))] #Orders fluxScan from smallest to largest absolute values
    fluxPoints = len(fluxScan)
    step_edge = qubit['biasStepEdge'][V]

    sweepData = {
        'fluxFunc': np.array([st.nearest(qubit['biasOperate'][V], fluxStep) - step_edge]),
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
            # one row is done.  find the maximum and update the spectroscopy fit
            freq_idx = sweepData['freqIndex']
            sweepData['maxima'][freq_idx] = sweepData['flux'][np.argmax(sweepData['prob'])]
            #when freq_idx<=5 get an order 0 polynomial, when freq_idx>5 get an order=1 polynomial.
            sweepData['fluxFunc'] = np.polyfit(freqScan[:freq_idx+1],
                                               sweepData['maxima'][:freq_idx+1] - step_edge,
                                               (freq_idx > 5))
            sweepData['fluxIndex'] = 0
            sweepData['freqIndex'] += 1
        else:
            # just go to the next point
            sweepData['fluxIndex'] = flux_idx + 1
        returnValue([flux, freq, prob])
    sweeps.run(func, sweep(), dataset=save and dataset, collect=collect, noisy=noisy)

    p = sweepData['fluxFunc']
    if update:
        Qubit['calDfDVbias'] = Value((1.0/p[0]),(GHz/V))
    if plot is True:
        #TODO: fix plotting!
        pass
        #plt.xlabel('Bias Voltage [V]')
        #plt.ylabel('Frequency [GHz]')
        #plt.plot(sweepData['maxima'], freqScan, '.')
        #plt.plot(np.polyval(p, freqScan), freqScan, 'g')

    return (sweepData['maxima'],freqScan)


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


def pulseshape(sample, height=-1.0, time=None, pulseLen=2000*ns,
               mpaStep=0.005, minPeak=0.4, stopAt=0.3, minCount=2,
               stats=300, measure=0, save=True, name='Pulse shape measurement',
               plot=True, noisy=True, update=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    if time is None:
        time = range(3, 25) + [round(10**(0.02*i)) for i in range(70, 151)]
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
        q['settlingAmplitudes'] = []
        q['settlingRates'] = []
        q.xy = eh.mix(q, eh.piPulse(q, t))
        q.z = (env.rect(-pulseLen, pulseLen + t + delay,  height + ofs) +
                env.rect(0.0, t + delay, -height) +
                eh.measurePulse(q, t + delay*2))
        q['readout'] = True
        prob = yield runQubits(server, qubits, stats, probs=[1])
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


def rabihigh(sample, amplitude=st.r[0.0:1.5:0.05], measureDelay=None, measure=0, stats=1500L, state=1, measstate=None,
             name='Rabi-pulse height MQ', save=True, collect=False, noisy=True, update=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    ml.setMultiKeys(q,max(state,measstate))
    name='|'+str(state)+'> '+name

    if amplitude is None: amplitude = ml.getMultiLevels(q,'piAmp',state)
    if measureDelay is None: measureDelay = state*q['piLen'] # /2.0
    if measstate is None: measstate=state

    axes = [(amplitude, 'pulse height'),
            (measureDelay, 'measure delay')]
    deps = [('Probability', '|'+str(measstate)+'>', '')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, amp, delay):
        ml.setMultiLevels(q,'piAmp',amp,state)
        swap = q['cZControlLen']+q['piLen']/2.0
        q.xy = eh.boostState(q, swap, state-1) + eh.mix(q, eh.piPulseHD(q, swap+(state-1)*q['piLen'], state=state), state=state)
        q.z = env.rect(0, q['cZControlLen'], q['cZControlAmp'])+eh.measurePulse(q, swap+delay, state=measstate)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, collect=(collect or update), noisy=noisy)

    if update:
        adjust.adjust_rabihigh(Q, data, state=state)
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
    return sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)


def rabilongHD(sample, length=st.r[0:500:2,ns], amplitude=None, detuning=None, measureDelay=None,
               randomize = True, measure=0, stats=1500L, name='Rabi MQ', save=True,
               collect=False, noisy=True):
    raise Exception('RabilongHD was removed from envelopeHelpers by some idiot. Now you have to rewrite it to make this work')
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
    return sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)


def pituner(sample, diff=0.4, measure=0, iterations=1, npoints=101, stats=1200, save=False, update=True, noisy=True, state=1):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    ml.setMultiKeys(q,state)

    amp = ml.getMultiLevels(q,'piAmp',state)
    ampstart = amp
    for _ in xrange(iterations):
        # optimize amplitude
        data = rabihigh(sample, amplitude=np.linspace((1-diff)*amp, (1+diff)*amp, npoints),
                        measure=measure, stats=stats, collect=True, noisy=noisy, save=save, state=state, update=False)
        amp_fit = np.polyfit(data[:,0], data[:,1], 2)
        amp,maxrabiprob = fitting.rabimax(data)
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
              measure=0, save=False, plot=False, noisy=True, update=True, state=1):
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
    if plot:
        fig = plt.figure()
    for i in xrange(iterations):
        data = dephasingSweeps_swap.ramsey(sample, measure=measure, delay=st.r[0:tEnd:timeRes,ns], fringeFreq = df,
                                      stats=stats, name='Ramsey Freq Tuner MQ', save = save, noisy=noisy, state=state,
                                      collect = True, randomize=False, averages = 1, tomo=False, plot=False,update=False)
        ts, ps = data.T
        y = ps - np.polyval(np.polyfit(ts, ps, 1), ts) # detrend
        timestep = ts[1] - ts[0]
        freq = np.fft.fftfreq(nfftpoints, timestep)
        fourier = abs(np.fft.fft(y, nfftpoints))
        fringe = abs(freq[np.argmax(fourier)])*1e3*MHz
        delta_freq = df - fringe
        if plot:
            ax = fig.add_subplot(iterations,1,i)
            ax.plot(np.fft.fftshift(freq), np.fft.fftshift(fourier))
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
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
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


def spectroscopy2DZxtalk(sample, freqScan=None, measAmplFunc=None, uwaveAmplSweep=None, measure=0, z_pulse=1,
                         fluxBelow=0.5, fluxAbove=0.5, fluxStep=0.025, sb_freq=0*GHz, stats=300L,
                         name='2D Z-pulse xtalk spectroscopy MQ', save=True, collect=False, update=True, noisy=True):
    sample, qubits = util.loadQubits(sample)
    qubit = qubits[measure]

    if measAmplFunc is None:
        measAmplFunc = get_mpa_func(qubit)

    if freqScan is None:
        freq = st.nearest(qubit['f10'][GHz], 0.001)
        freqScan = np.arange(freq-0.01, freq+0.01, 0.0005)
    elif isinstance(freqScan, tuple) and len(freqScan) == 2:
        freq = st.nearest(qubit['f10'][GHz], 0.001)
        rng = freqScan
        dfs = np.logspace(-3, 0, 25)
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
                if abs(zpa) < 2.0: # skip if zpa is too large
                    yield zpa, f*GHz

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
        returnValue([zpa, freq, prob])
    sweeps.run(func, sweep(), dataset=save and dataset, collect=collect, noisy=noisy)
    return sweepData['fluxFunc']


def swapSpectroscopy(sample, swapLen=st.arangePQ(0,200,4,ns), swapAmp=np.arange(-0.05,0.05,0.002), measure=0, stats=600L,
         name='Swap Spectroscopy', save=True, collect=False, noisy=True, state=1, piPulse=True):
    """Measures T1 vs z-pulse amplitude"""

    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    if swapAmp is None:
        swapAmp = q.swapAmp
    elif np.size(swapAmp) is 1:
        swapAmp = float(swapAmp)
    else:
        swapAmp = swapAmp[np.argsort(np.abs(swapAmp))]

    axes = [(swapAmp, 'swap pulse amplitude'), (swapLen, 'swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, currAmp, currLen):
        #It can be useful to do swap spectroscopy without exciting the qubit to see
        #if the microwave carrier is leaking into your system.
        if piPulse:
            q.xy = eh.boostState(q, 0, state=state)
        q.z = env.rect(q['piLen']*(state-0.5), currLen, currAmp) + eh.measurePulse(q, q['piLen']*(state-0.5) + currLen, state=state)
        q['readout'] = True
        return runQubits(server, qubits, stats=stats, probs=[1])

    return sweeps.grid(func, axes, save=save, dataset=dataset, collect=collect, noisy=noisy)


def datasetMinimum(data, default, llim, rlim, dataset=None):
    coeffs = np.polyfit(data[:,0],data[:,1],2)
    if coeffs[0] <= 0:
        print 'No minimum found, keeping value'
        return default, np.polyval(coeffs, default)
    result = np.clip(-0.5*coeffs[1]/coeffs[0],llim,rlim)
    return result, np.polyval(coeffs, result)


def swap10tuner(sample, swapLen=None, swapAmp=None, swapAmpBND=0.01, iteration=3, measure=0, stats=600L,
         name='Qeg-R10 swap tuner MQ', save=False, noisy=True, update=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    if swapAmp is None:
        swapAmp = q['cZControlAmp']
    if swapLen is None:
        swapLen = q['cZControlLen'][ns]
    for i in range(iteration):
        rf = 2**i
        swapLenOld = swapLen
        swapAmpOld = swapAmp
        if noisy:
            print 'Tuning the swap length'
        results = swapSpectroscopy(sample, swapLen=st.PQlinspace(swapLen*(1-0.3/rf),swapLen*(1+0.3/rf),21,ns),
                        swapAmp=swapAmp, state=1, measure=measure, stats=stats,
                        name='Qeg-R10 swap MQ', save=save, collect=True, noisy=noisy)
        newLen, percent = datasetMinimum(results, swapLenOld, swapLenOld-4/rf, swapLenOld+4/rf)
        swapLen = newLen
        if (noisy or (i is 0)):
            print 'Old swap length was ',q['cZControlLen']
        if update:
            Q['cZControlLen'] = swapLen*ns
        if (noisy or (i is (iteration-1))):
            print 'New Control swap length is ', swapLen, 'ns'
        if noisy:
            print 'Tuning the swap amplitude'
        results = swapSpectroscopy(sample, state=1, swapLen=swapLen,
                        swapAmp=np.linspace(swapAmp-swapAmpBND/rf,swapAmp+swapAmpBND/rf,21), measure=measure, stats=stats,
                        name='Qeg-R10 swap MQ', save=save, collect=True, noisy=noisy)
        newAmp, percent = datasetMinimum(results, swapAmpOld, swapAmpOld-4/rf, swapAmpOld+4/rf)
        swapAmp = newAmp
        if (noisy or (i is 0)):
            print 'Old Control swap amplitude was ',q['cZControlAmp']
        if update:
            Q['cZControlAmp']= swapAmp
        if (noisy or (i is (iteration-1))):
            print 'New swap amplitude is ', swapAmp

    return swapLen, swapAmp


def swap21tuner(sample, swapLen=None, swapAmp=None, iteration=3, measure=0, stats=600L,
         name='Qfe-R10 swap tuner MQ', save=False, noisy=True, update=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    if swapAmp is None:
        swapAmp = q['cZTargetAmp']
    if swapLen is None:
        swapLen = q['cZTargetLen'][ns]
        swapLen = swapLen/2 #cZTargetLen is the time for a iswap^2 so we divide by two for this iSWAP experiment.

    for m in range(iteration):
        rf = 2**m #not sure what m is stepping over, I think it is a dummy variable, but used in the linspace calc"
        swapLenOld = swapLen
        swapAmpOld = swapAmp
        print 'Tuning the swap length'
        results = swapSpectroscopy(sample, swapLen=st.PQlinspace(swapLen*(1-0.3/rf),swapLen*(1+0.3/rf),21,ns),
                        swapAmp=swapAmp, state=2, measure=measure, stats=stats,
                        name='Qfe-R10 swap MQ', save=save, collect=True, noisy=noisy)
        newLen, percent = datasetMinimum(results, swapLenOld, swapLenOld-4/rf, swapLenOld+4/rf)
        swapLen=newLen
        print 'Old swap length was ',q['cZTargetLen'], 'ns'
        if update:
            Q['cZTargetLen'] = (2*swapLen)*ns #cZControlLen is the time for a iswap^2.
        print 'New Target swap length is ', 2*swapLen, 'ns'
        print 'Tuning the swap amplitude'
        swapLen = newLen
        results = swapSpectroscopy(sample, swapLen=swapLen,
                        swapAmp=np.linspace(swapAmp*(1-0.3/rf),swapAmp*(1+0.3/rf),21), state=2, measure=measure, stats=stats,
                        name='Qfe-R10 swap MQ', save=save, collect=True, noisy=noisy)
        newAmp, percent = datasetMinimum(results, swapAmpOld, swapAmpOld-4/rf, swapAmpOld+4/rf)
        swapAmp = newAmp
        print 'Old swap amplitude was ',q['cZTargetAmp']
        if update:
            Q['cZTargetAmp']= swapAmp
        print 'New Target swap amplitude is ', swapAmp

    return swapLen, swapAmp


def swap21tunerMax(sample, swapLen=None, swapAmp=None, iteration=3, measure=0, stats=600L,
         name='Qfe-R10 swap tuner MQ', save=False, noisy=True, update=True):
    '''Finds a maximum on the 21 chevron'''
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    if swapAmp is None:
        swapAmp = q['cZTargetAmp']
    if swapLen is None:
        swapLen = q['cZTargetLen'][ns]


    for m in range(iteration):
        rf = 2**m #not sure what m is stepping over, I think it is a dummy variable, but used in the linspace calc"
        swapLenOld = swapLen
        swapAmpOld = swapAmp
        print 'Tuning the swap length'
        results = swapSpectroscopy(sample, swapLen=st.PQlinspace(swapLen*(1-0.3/rf),swapLen*(1+0.3/rf),21,ns),
                        swapAmp=swapAmp, state=2, measure=measure, stats=stats,
                        name='Qfe-R10 swap MQ', save=save, collect=True, noisy=noisy)
        newLen, percent = datasetMinimum(results, swapLenOld, swapLenOld-4/rf, swapLenOld+4/rf)
        swapLen=newLen
        print 'Old swap length was ',q['cZTargetLen'], 'ns'
        if update:
            Q['cZTargetLen'] = (swapLen)*ns #cZControlLen is the time for a iswap^2.
        print 'New Target swap length is ', swapLen, 'ns'
        print 'Tuning the swap amplitude'
        swapLen = newLen
        results = swapSpectroscopy(sample, state=2, swapLen=swapLen,
                        swapAmp=np.linspace(swapAmp*(1-0.3/rf),swapAmp*(1+0.3/rf),21), measure=measure, stats=stats,
                        name='Qfe-R10 swap MQ', save=save, collect=True, noisy=noisy)
        newAmp, percent = datasetMinimum(results, swapAmpOld, swapAmpOld-4/rf, swapAmpOld+4/rf)
        swapAmp = newAmp
        print 'Old swap amplitude was ',q['cZTargetAmp']
        if update:
            Q['cZTargetAmp']= swapAmp
        print 'New Target swap amplitude is ', swapAmp

    return swapLen, swapAmp


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
       save=False, collect=True, noisy=True, update=True):
    sample, qubits, Qubits = util.loadQubits(sample,write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]

#    if len(q['cZControlLen'])<n:
#        for i in np.arange(len(q['noonSwapLen'+paraName+'s']),n,1):
#            sample['q'+str(measure)]['noonSwapLen'+paraName+'s'].append(q['noonSwapLen'+paraName+'s'][0]/np.sqrt(i+1))

    for iter in range(iteration):
        rf = 2**iter
        print 'iteration %g...' % iter
        sl = q['cZControlLen']
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


def testQubResDelayCmp(sample, delay=st.r[-80:50:2,ns], stats=1500L, measureC=0, measureT=1,pulseDelay=5*ns,
         name='delay between qubits', save=False, collect=True, noisy=True, update=True):
    sample, qubits, Qubits = util.loadQubits(sample,write_access=True)
    qc = qubits[measureC]
    qt = qubits[measureT]
    Q0 = Qubits[measureC]

    result0 = testQubResDelay(sample, startTime=delay, measureC=measureC, measureT=measureT, save=save, collect=collect, noisy=noisy)
    result1 = testQubResDelay(sample, startTime=delay, measureC=measureC, measureT=measureT, save=save, collect=collect, noisy=noisy)


    topLenT = qt.cZTargetLen['ns']
    transLenT = pulseDelay
    def fitfunc0(x, p):
        return (p[1] +
                p[2] * 0.5*erfc((x - (p[0] - topLenT/2.0)) / transLenT) +
                p[3] * 0.5*erf((x - (p[0] + topLenT/2.0)) / transLenT))
    x0, y0 = result0.T
    fit0, _ok0 = leastsq(lambda p: fitfunc0(x0, p) - y0, [0.0, 0.05, 0.9, 0.9])

    topLenC = qc.cZTargetLen['ns']
    transLenC = pulseDelay
    def fitfunc1(x, p):
        return (p[1] +
                p[2] * 0.5*erfc((x - (p[0] - topLenC/2.0)) / transLenC) +
                p[3] * 0.5*erf((x - (p[0] + topLenC/2.0)) / transLenC))
    x1, y1 = result1.T
    fit1, _ok1 = leastsq(lambda p: fitfunc1(x1, p) - y1, [0.0, 0.05, 0.9, 0.9])

    plt.figure()
    plt.plot(x0, y0, 'b.')
    plt.plot(x0, fitfunc0(x0, fit0), 'b-')
    plt.plot(x1, y1, 'r.')
    plt.plot(x1, fitfunc1(x1, fit1), 'r-')

    print fit0[0], fit1[0]
    if update:
        print 'correct lag between two qubits by adding %g ns to q0' % ((fit0[0]-fit1[0])/2.0)
        Q0['timingLagUwave'] += ((fit0[0]-fit1[0])/2.0)*ns
        Q0['timingLagMeas']  += ((fit0[0]-fit1[0])/2.0)*ns

    return


def testQubResDelay(sample, startTime=st.r[-100:100:2,ns], pulseDelay=5*ns, stats=600L, measureC=0 , measureT=1,
       name='Qubit-Qubit delay using Resonator "test delay" ', save=True, collect=True, noisy=True, plot=False, update=True):
    """A single pi-pulse on the Target qubit, iSWAP with R, delay, iSwap. Control sweeps iSWAP through sequence."""
    sample, qubits, Qubits = util.loadQubits(sample,write_access=True)
    qc = qubits[measureC]
    qt = qubits[measureT]

    axes = [(startTime, 'Control iSWAP start time')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measureC, kw=kw)

    def func(server, curr):
        qt.xy = eh.mix(qt, eh.piPulseHD(qt, -qt.cZControlLen-pulseDelay-qt.piLen/2))
        qt.z = env.rect(-qt.cZControlLen-pulseDelay, qt.cZControlLen, qt.cZControlAmp)
        qc.z = env.rect(curr, qc.cZControlLen, qc.cZControlAmp)
        qc.z += eh.measurePulse(qc, qc.cZControlLen+curr)
        qc['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return result


def testdelay(sample, t0=st.r[-40:40:0.5,ns], zpa=-0.1, zpl=20*ns, tm=65*ns,
              measure=0, stats=1200, update=True,
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
        Other['timingLagMeas'] += fit[0]*ns
        Other['timingLagUwave'] += fit[0]*ns


def insensitive(sample, bias=st.r[-500:500:50, mV], detuning=st.r[-50:50:2,MHz], sb_freq=0*GHz,
                stats=300L, measure=0, save=True, name='Insensitive Points', noisy=True):
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
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)


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
            return runQubits(server, qubits, stats, separate=True)
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
                     df=None, stats=600L, drive=0, simult=True,
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
        qa.z = eh.measurePulse(qa, t)
        qr.z = eh.measurePulse(qr, 0)
        qa['readout'] = True
        qr['readout'] = True
        return runQubits(server, qubits, stats)
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    if update:
        t = adjust.adjust_time(data)
        if t is not None:
            print 'timing lag corrected by %g ns' % t
            Qa['timingLagMeas'] += t*ns
            Qa['timingLagUwave'] += t*ns


def t1(sample, delay=st.r[-10:1000:2,ns], stats=600L, measure=0,
       name='T1 MQ', save=True, collect=True, noisy=True, state=1,
       update=True, plot=True):
    """A single pi-pulse on one qubit, with other qubits also operated."""
    if update and (state>1):
        raise Exception('updating with states above |1> not yet implemented')
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]
    ml.setMultiKeys(q,state)
    if state>1: name=name+' for |'+str(state)+'>'

    axes = [(delay, 'Measure pulse delay')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, delay):
        swap = q['cZControlLen']+q['piLen']/2.0
        q.xy = eh.boostState(q, swap, state)
        q.z = env.rect(0, q['cZControlLen'], q['cZControlAmp'])+eh.measurePulse(q, swap+delay, state=state)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if plot or update:
        with labrad.connect() as cxn:
            dv = cxn.data_vault
            dataset = dstools.getOneDeviceDataset(dv,-1,session=sample._dir,
                                                  deviceName=None,averaged=False)
        result = fitting.t1(dataset,timeRange=(10,delay[-1]['ns']),plot=plot)
        if update:
            Q['calT1']=result['T1']
    return data


def ramseyFilter(sample, delay=st.r[5:50:1,ns], theta=0, measure=0, stats=900L,
                 name='Ramsey Error Filter', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    axes = [(delay, 'Pulse Separation Delay'), (theta, 'Phase of 2nd Pulse')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, delay, theta):
        q.xy = eh.mix(q, eh.piPulse(q, 0) + eh.piPulse(q, delay))
        q.z = eh.measurePulse2(q, delay + q['piLen']/2.0) # measure 2-state population
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)


def ramseyZPulse(sample, zAmp=st.r[0:0.1:0.005], measure=0, stats=900L,
                 name='Z-Pulse Ramsey Fringe', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    axes = [(zAmp, 'Z-Pulse Amplitude')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, zAmp):
        dt = q['piLen']
        q.xy = eh.mix(q, eh.piHalfPulse(q, 0) + eh.piHalfPulse(q, 2*dt))
        q['piAmpZ'] = zAmp
        q.z = eh.rotPulseZ(q, dt) + eh.measurePulse(q, 2*dt + dt/2)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

