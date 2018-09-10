import numpy as np
from scipy.optimize import leastsq
from scipy.special import erf, erfc
import matplotlib.pyplot as plt
import pylab
import numpy
import pdb
from labrad.units import Unit
V, mV, us, ns, GHz, MHz = [Unit(s) for s in ('V', 'mV', 'us', 'ns', 'GHz', 'MHz')]

import pyle.envelopes as env
from pyle.pipeline import returnValue, FutureList
from pyle.dataking import envelopehelpers as eh
from pyle.dataking import measurement
from pyle.dataking import adjust
from pyle.dataking.fpgaseq import runQubits
from pyle.util import sweeptools as st


import pyle.dataking.qubitpulsecal as qpc
from pyle.dataking import sweeps
from pyle.dataking import util

from pyle.dataking import fpgaseq
fpgaseq.PREPAD = 300

import time


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


def scurve(sample, mpa=st.r[0:2:0.05], stats=300, measure=0,
           save=True, name='SCurve MQ', collect=True, noisy=True, update=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    axes = [(mpa, 'Measure pulse amplitude')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, mpa):
        q['measureAmp'] = mpa
        q.z = eh.measurePulse(q, 0)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    if update:
        adjust.adjust_scurve(Q, data)
    if collect:
        return data


def visibility(sample, mpa=st.r[0:2:0.05], stats=300, measure=0,
               save=True, name='Visibility MQ', collect=True, update=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    axes = [(mpa, 'Measure pulse amplitude')]
    deps = [('Probability', '|0>', ''),
            ('Probability', '|1>', ''),
            ('Visibility', '|1> - |0>', '')]
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
        q.xy = eh.mix(q, eh.piPulse(q, t_pi))
        q.z = eh.measurePulse(q, t_meas)
        req1 = runQubits(server, qubits, stats, probs=[1])

        probs = yield FutureList([req0, req1])
        p0, p1 = [p[0] for p in probs]
        returnValue([p0, p1, p1-p0])
    return sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)


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


def find_mpa(sample, stats=60, target=0.05, mpa_range=(-2.0, 2.0), pi_pulse=False,
             measure=0, pulseFunc=None, resolution=0.005, blowup=0.05,
             falling=None, statsinc=1.25,
             save=False, name='SCurve Search MQ', collect=True, update=True, noisy=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    axes = [('Measure Pulse Amplitude', '')]
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure)

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
        q['measureAmp'] = mpa
        if pi_pulse:
            q.xy = eh.mix(q, eh.piPulse(q, 0))
            q.z = eh.measurePulse(q, q['piLen']/2.0)
        else:
            q.xy = env.NOTHING
            q.z = eh.measurePulse(q, 0)
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
    key = 'measureAmp' if not pi_pulse else 'measureAmp2'
    if key in q:
        print 'Old %s: %.3f' % (key, Q[key])
    print 'New %s: %.3f' % (key, mpa)
    if update:
        Q[key] = mpa
    return mpa


def find_mpa_func(sample, measure=0, biaspoints=None, order=1, steps=5, noisy=True, update=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    if biaspoints is None:
        fb0 = q['biasOperate'][mV]
        mpa = q['measureAmp']
        fb1 = fb0 + 20 * np.sign(mpa) * np.sign(abs(mpa)-1)
        biaspoints = [fb*mV for fb in np.linspace(fb0, fb1, steps)]
        print 'Biaspoints:', ', '.join(str(b) for b in biaspoints)
    mpas = []
    for fb in biaspoints:
        q['biasOperate'] = fb
        mpas.append(find_mpa(sample, measure=measure, noisy=noisy))
    biaspoints = np.array([fb[V] - q['biasStepEdge'][V] for fb in biaspoints])
    mpas = np.array(mpas)
    p = np.polyfit(biaspoints, mpas, order)
    if update:
        Q['calMpaFunc'] = p
    return get_mpa_func(q, p)


def get_mpa_func(qubit, p=None):
    if p is None:
        p = qubit['calMpaFunc']
    return lambda fb: np.polyval(p, fb[V] - qubit['biasStepEdge'][V])


def find_flux_func(sample, freqScan=None, measAmplFunc=None, measure=0,
                   fluxBelow=2*mV, fluxAbove=2*mV, fluxStep=0.1*mV, sb_freq=0*GHz, stats=300L,
                   save=True, name='Flux func search MQ', collect=False, update=True, noisy=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    qubit, Qubit = qubits[measure], Qubits[measure]

    if measAmplFunc is None:
        measAmplFunc = get_mpa_func(qubit)

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

    fluxBelow = fluxBelow[V]
    fluxAbove = fluxAbove[V]
    fluxStep = fluxStep[V]
    fluxScan = np.arange(-fluxBelow, fluxAbove, fluxStep)
    fluxScan = fluxScan[np.argsort(abs(fluxScan))]
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
    if update:
        Qubit['calFluxFunc'] = p
    return get_flux_func(Qubit, p, step_edge*V)


def get_flux_func(qubit, p=None, step_edge=None):
    if p is None:
        p = qubit['calFluxFunc']
    if step_edge is None:
        step_edge = qubit['biasStepEdge']
    return lambda f: np.polyval(p, f[GHz]**4)*V + step_edge


def find_zpa_func(sample, freqScan=None, measure=0,
                  fluxBelow=0.01, fluxAbove=0.01, fluxStep=0.0005, sb_freq=0*GHz, stats=300L,
                  name='ZPA func search MQ', save=True, collect=False, noisy=True, update=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    qubit = qubits[measure]

    if freqScan is None:
        freq = st.nearest(qubit['f10'][GHz], 0.001)
        freqScan = np.arange(freq-0.1, freq+1.0, 0.005)
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
    if update:
        Qubits[measure]['calZpaFunc'] = poly
    return get_zpa_func(Qubits[measure], poly)


def get_zpa_func(qubit, p=None):
    if p is None:
        p = qubit['calZpaFunc']
    return lambda f: np.polyval(p, f[GHz]**4)


def operate_at(qubit, f):
    """Update qubit parameters to operate at the specified frequency."""
    fb = get_flux_func(qubit)(f)
    mpa = get_mpa_func(qubit)(fb)
    print 'f: %s, fb: %s, mpa: %s' % (f, fb, mpa)
    qubit['f10'] = f
    qubit['biasOperate'] = fb
    qubit['measureAmp'] = mpa


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


def rabihigh(sample, amplitude=st.r[0.0:1.5:0.05], detuning=None, measureDelay=None, measure=0, stats=1500L,
             name='Rabi-pulse height MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    if amplitude is None: amplitude = q['piAmp']
    if detuning is None: detuning = 0
    if measureDelay is None: measureDelay = q['piLen']/2.0

    axes = [(amplitude, 'pulse height'),
            (detuning, 'detuning'),
            (measureDelay, 'measure delay')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, amp, df, delay):
        q.xy = eh.mix(q, env.gaussian(0, q['piFWHM'], amp=amp, df=df))
        q.z = eh.measurePulse(q, delay)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)


def rabihigh_hd(sample, amplitude=st.r[0.0:1.5:0.05], measureDelay=None, measure=0, stats=1500L,
                name='Rabi-pulse height HD MQ', save=True, collect=False, noisy=True):
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


def rabilong(sample, length=st.r[0:500:2,ns], amplitude=None, detuning=None, measureDelay=None, measure=0, stats=1500L,
             name='Rabi-pulse length MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    if amplitude is None: amplitude = q['piAmp']
    if detuning is None: detuning = 0
    if measureDelay is None: measureDelay = q['piLen']/2.0

    axes = [(length, 'pulse length'),
            (amplitude, 'pulse height'),
            (detuning, 'detuning'),
            (measureDelay, 'measure delay')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, len, amp, df, delay):
        q.xy = eh.mix(q, env.flattop(0, len, w=q['piFWHM'], amp=amp, df=df))
        q.z = eh.measurePulse(q, delay)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)


def pituner(sample, measure=0, iterations=2, npoints=21, stats=1200, save=False, update=True, noisy=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    amp = q['piAmp']
    df = q['piDf'][MHz]
    for _ in xrange(iterations):
        # optimize amplitude
        data = rabihigh(sample, amplitude=np.linspace(0.75*amp, 1.25*amp, npoints),
                        measure=measure, stats=stats, collect=True, noisy=noisy)
        amp_fit = np.polyfit(data[:,0], data[:,1], 2)
        amp = -0.5 * amp_fit[1] / amp_fit[0]
        print 'Amplitude: %g' % amp
        # optimize detuning
        data = rabihigh(sample, amplitude=None, detuning=st.PQlinspace(df-20, df+20, npoints, MHz),
                        measure=measure, stats=stats, collect=True, noisy=noisy)
        df_fit = np.polyfit(data[:,0], data[:,1], 2)
        df = -0.5 * df_fit[1] / df_fit[0]
        print 'Detuning: %g MHz' % df
    # save updated values
    if update:
        Q['piAmp'] = amp
        Q['piDf'] = df*MHz
    return amp, df*MHz


def pitunerHD(sample, measure=0, iterations=2, npoints=21, stats=1200, save=False, update=True, noisy=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    amp = q['piAmp']
    for _ in xrange(iterations):
        # optimize amplitude
        data = rabihigh_hd(sample, amplitude=np.linspace(0.6*amp, 1.4*amp, npoints),
                           measure=measure, stats=stats, collect=True, noisy=noisy)
        amp_fit = np.polyfit(data[:,0], data[:,1], 2)
        amp = -0.5 * amp_fit[1] / amp_fit[0]
        print 'Amplitude: %g' % amp
    # save updated values
    if update:
        Q['piAmp'] = amp
    return amp


def pitunerZ(sample, measure=0, npoints=100, nfftpoints=2048, stats=1200, save=False, update=True, noisy=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    # optimize amplitude
    data = ramseyScope(sample, zAmp=st.r[-0.1:0.1:0.002],
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
        data = ramsey(sample, st.r[0:tEnd:timeRes,ns], df=df, stats=stats,
                      measure=measure, name='Ramsey Freq Tuner MQ', collect=True, noisy=noisy, save=save)
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
        print 'Desired Fringe Frequency: %s' % df
        print 'Actual Fringe Frequency: %s' % fringe
        print 'Qubit frequency adjusted by %s' % delta_freq

        q['f10'] += delta_freq
        print 'new resonance frequency: %g' % q['f10']

    Q['f10'] = st.nearest(q['f10'][GHz], 0.0001)*GHz
    return Q['f10']


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


def swaptuner(sample, measure=0, pi_pulse_on=1, iterations=3, npoints=41, stats=1200,
              tune_overshoot=True,
              save=False, update=True, noisy=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    zpas = [q['swapAmp'] if i == measure else 0.0 for i in range(len(qubits))]
    overshoots = [0.0] * len(qubits)
    overshoot = q['swapOvershoot']
    plen = q['swapLen'][ns]
    for i in xrange(iterations):
        ratio = 1.0 / 2**i

        # optimize pulse length
        llim = plen*(1-ratio)
        ulim = plen*(1+ratio)
        overshoots[measure] = overshoot
        plens = np.linspace(llim, ulim, npoints)
        data = w_state(sample, pi_pulse_on=pi_pulse_on, measure=[measure], t_couple=1000*ns,
                       delay=plens, zpas=zpas, overshoots=overshoots, stats=stats)
        fit = np.polyfit(data[:,0], data[:,1], 2)
        if fit[0] < 0: # found a maximum
            plen = np.clip(-0.5 * fit[1] / fit[0], llim, ulim)
            print 'Pulse Length: %g ns' % plen
        else:
            print 'No maximum found versus pulse length.'

        if tune_overshoot:
            # optimize overshoot
            llim = np.clip(overshoot*(1-ratio), 0, 1)
            ulim = np.clip(overshoot*(1+ratio), 0, 1)
            overshoots[measure] = np.linspace(llim, ulim, npoints)
            data = w_state(sample, pi_pulse_on=pi_pulse_on, measure=[measure], t_couple=1000*ns,
                           delay=plen, zpas=zpas, overshoots=overshoots, stats=stats*4)
            fit = np.polyfit(data[:,0], data[:,1], 2)
            if fit[0] < 0: # found a maximum
                overshoot = np.clip(-0.5 * fit[1] / fit[0], llim, ulim)
                print 'Overshoot: %g' % overshoot
            else:
                print 'No maximum found versus overshoot.'
    # save updated values
    if update:
        Q['swapOvershoot'] = overshoot
        Q['swapLen'] = plen*ns
    return overshoot, plen*ns


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
    if collect:
        return data


def spectroscopy2Dauto(sample, freqScan=None, measAmplFunc=None, measure=0, stats=300L,
                       fluxBelow=2*mV, fluxAbove=2*mV, fluxStep=0.1*mV, sb_freq=0*GHz,
                       save=True, name='2D spectroscopy MQ', collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    q['readout'] = True

    if measAmplFunc is None:
        measAmplFunc = get_mpa_func(q)

    if freqScan is None:
        freq = st.nearest(q['f10'][GHz], 0.001)
        freqScan = np.arange(freq-0.1, freq+1.0, 0.005)
    else:
        freqScan = np.array([f[GHz] for f in freqScan])
    freqScan = freqScan[np.argsort(abs(freqScan-q['f10'][GHz]))]

    fluxBelow = fluxBelow[V]
    fluxAbove = fluxAbove[V]
    fluxStep = fluxStep[V]
    fluxScan = np.arange(-fluxBelow, fluxAbove, fluxStep)
    fluxScan = fluxScan[np.argsort(abs(fluxScan))]
    fluxPoints = len(fluxScan)

    sweepData = {
        'fluxFunc': np.array([st.nearest(q['biasOperate'][V], fluxStep) - q['biasStepEdge']]),
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
            center = np.polyval(sweepData['fluxFunc'], f**4) + q['biasStepEdge']
            center = st.nearest(center, fluxStep)
            for flx in center + fluxScan:
                yield flx*V, f*GHz

    def func(server, args):
        flux, freq = args
        for qubit in qubits:
            qubit['fc'] = freq - sb_freq # set all frequencies since they share a common microwave source
        q['biasOperate'] = flux
        q['measureAmp'] = measAmplFunc(flux)
        q.xy = eh.spectroscopyPulse(q, 0, sb_freq)
        q.z = eh.measurePulse(q, q['spectroscopyLen'])
        prob = yield runQubits(server, qubits, stats, probs=[1])

        flux_idx = sweepData['fluxIndex']
        sweepData['flux'][flux_idx] = flux[V]
        sweepData['prob'][flux_idx] = prob[0]
        if flux_idx + 1 == fluxPoints:
            # one row is done.  find the maximum and update the spectroscopy fit
            freq_idx = sweepData['freqIndex']
            sweepData['maxima'][freq_idx] = sweepData['flux'][np.argmax(sweepData['prob'])]
            sweepData['fluxFunc'] = np.polyfit(freqScan[:freq_idx+1]**4,
                                               sweepData['maxima'][:freq_idx+1] - q['biasStepEdge'],
                                               freq_idx > 5)
            sweepData['fluxIndex'] = 0
            sweepData['freqIndex'] += 1
        else:
            # just go to the next point
            sweepData['fluxIndex'] = flux_idx + 1
        returnValue([flux, freq, prob])
    sweeps.run(func, sweep(), dataset=save and dataset, collect=collect, noisy=noisy)

    # create a flux function and return it
    step = q['biasStepEdge']
    poly = sweepData['fluxFunc']
    return lambda f: np.polyval(poly, st.inUnits(f, GHz)**4) + step


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


def spectroscopy2DZauto(sample, freqScan=None, measure=0,
                        fluxBelow=0.01, fluxAbove=0.01, fluxStep=0.0005, sb_freq=0*GHz, stats=300L,
                        name='2D Z-pulse spectroscopy MQ', save=True, collect=False, noisy=True, update=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    qubit = qubits[measure]

    if freqScan is None:
        freq = st.nearest(qubit['f10'][GHz], 0.001)
        freqScan = np.arange(freq-0.1, freq+1.0, 0.005)
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
    if update:
        Qubits[measure]['calZpaFunc'] = poly
    return lambda f: np.polyval(poly, f[GHz]**4)


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
    print 'uwave lag:', fit[0]
    if update:
        print 'uwave lag corrected by %g ns' % fit[0]
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


def insensitive(sample, bias=st.r[-500:500:50, mV], detuning=st.r[-50:50:2, MHz], sb_freq=0*GHz,
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
       name='T1 MQ', save=True, collect=True, noisy=True):
    """A single pi-pulse on one qubit, with other qubits also operated."""
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    axes = [(delay, 'Measure pulse delay')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, delay):
        q.xy = eh.mix(q, eh.piPulse(q, 0))
        q.z = eh.measurePulse(q, delay)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)


def ramsey(sample, delay=st.r[0:500:1,ns], phase=0, df=50*MHz, stats=600L, measure=0,
           name='Ramsey MQ', save=True, collect=True, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    axes = [(delay, 'Ramsey pulse delay'),
            (phase, 'Ramsey phase')]
    kw = {
        'stats': stats,
        'ramsey_freq': df,
    }
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, delay, phase):
        dt = q['piLen']
        tp = dt/2.0 + delay + dt/2.0
        tm = tp + dt/2.0
        ph = phase - 2*np.pi*df[GHz]*delay[ns]
        q.xy = eh.mix(q, eh.piHalfPulse(q, 0) + eh.piHalfPulse(q, tp, phase=ph))
        q.z = eh.measurePulse(q, tm)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)


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


def ramseyScope(sample, zAmp=st.r[0:0.1:0.005], measure=0, stats=900L,
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


def spin_echo(sample, delay=st.r[0:500:1,ns], df=50*MHz, stats=600L, measure=0,
              name='Spin Echo MQ', save=True, collect=True, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    axes = [(delay, 'Ramsey pulse delay')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, delay):
        dt = q['piLen']
        tpi = dt/2.0 + delay/2.0
        tp = dt/2.0 + delay + dt/2.0
        tm = tp + dt/2.0
        phasepi = 2*np.pi*df[GHz]*delay[ns]/2.0
        q.xy = eh.mix(q, eh.piHalfPulse(q, 0) +
                         eh.piPulse(q, tpi, phase=phasepi) +
                         eh.piHalfPulse(q, tp))
        q.z = eh.measurePulse(q, tm)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)


def uwave_phase_adjust(sample, phase=0, t0=None, t_couple=st.r[0:200:1,ns], adjust=0, ref=1, zpas=None, stats=3000L,
                       name='uwave-phase cal', save=True, collect=True, noisy=True):
    sample, qubits = util.loadQubits(sample)

    axes = [(phase, 'uw phase %d' % adjust), (t_couple, 'Coupling time')]
    measure = measurement.Null(len(qubits), [adjust, ref]) # do xtalk-free measurement
    kw = {
        'stats': stats,
        'adjust phase': adjust,
        'reference phase': ref,
        'zpas': zpas,
    }
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, phase, t_couple):
        dt = max(q['piLen'] for q in qubits)
        if t0 is None:
            tp0 = 0
            tz = dt/2
        else:
            tp0 = t0 - dt/2
            tz = t0
        tm = tz + t_couple + dt/2

        for i, (q, zpa) in enumerate(zip(qubits, zpas)):
            q['uwavePhase'] = 0 # turn off automatic phase correction

            if i == adjust:
                q.xy = eh.piHalfPulse(q, tp0) * np.exp(1j*phase)
            elif i == ref:
                q.xy = eh.piHalfPulse(q, tp0)
            else:
                q.xy = env.NOTHING

            q.z = env.rect(tz, t_couple, zpa, overshoot=q['wZpulseOvershoot'])
        return measure(server, qubits, tm, stats=stats)
    return sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)


def swap_dphase_adjust(sample, dphase, stats=600L, adjust=0, ref=1,
                       name='Swap phase adjust', save=True, collect=True, noisy=True):
    sample, qubits = util.loadQubits(sample)

    axes = [(dphase, 'dphase')]
    measure = measurement.Simult(len(qubits), adjust)
    kw = {'stats': stats, 'ref qubit': ref}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, dphase):
        q = qubits[adjust]
        r = qubits[ref]

        dt = max(qubit['piLen'] for qubit in qubits)
        tcouple = q['swapLen']

        tz = dt/2
        tp = dt/2 + tcouple + dt/2
        tm = dt/2 + tcouple + 2*dt/2

        q.xy = eh.piHalfPulse(q, 0) + eh.piHalfPulse(q, tp, phase=dphase)
        q.z = env.rect(tz, tcouple, q['swapAmp'], overshoot=q['swapOvershoot'])

        zpafunc = get_zpa_func(r)
        zpa = zpafunc(r['f10'] - (q['f10'] - r['f10'])) # move the ref qubit out of the way
        r.z = env.rect(tz, tcouple, zpa)

        return measure(server, qubits, tm, stats=stats)
    return sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)


def w_dphase_adjust(sample, dphase, stats=600L, adjust=0, ref=1,
                   name='W dphase adjust', save=True, collect=True, noisy=True):
    sample, qubits = util.loadQubits(sample)

    axes = [(dphase, 'dphase')]
    measure = measurement.Simult(len(qubits), adjust)
    kw = {'stats': stats, 'ref qubit': ref}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, dphase):
        for q in qubits:
            q.xy = env.NOTHING
            q.z = env.NOTHING

        q = qubits[adjust]
        r = qubits[ref]

        dt = max(qubit['piLen'] for qubit in qubits)
        tcouple = q['wZpulseLen']

        tz = dt/2
        tp = dt/2 + tcouple + dt/2
        tm = dt/2 + tcouple + 2*dt/2

        q.xy = eh.piHalfPulse(q, 0) + eh.piHalfPulse(q, tp, phase=dphase)
        q.z = env.rect(tz, tcouple, q['wZpulseAmp'], overshoot=q['wZpulseOvershoot'])

        zpafunc = get_zpa_func(r)
        zpa = zpafunc(r['f10'] - (q['f10'] - r['f10'])) # move the ref qubit out of the way
        r.z = env.rect(tz, tcouple, zpa)

        return measure(server, qubits, tm, stats=stats)
    return sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)


#def w_dphase_adjust_z(sample, dphase_amp=None, dphase_len=None, stats=600L, adjust=0, ref=1,
#                     name='W dphase adjust Z', save=True, collect=True, noisy=True):
#    sample, qubits = util.loadQubits(sample)
#    if dphase_amp is None:
#        dphase_amp = qubits[measure]['w_dphase_amp']
#    if dphase_len is None:
#        dphase_len = qubits[measure]['w_dphase_len']
#
#    axes = [(dphase_amp, 'dphase_amp'), (dphase_len, 'dphase_len')]
#    measure = measurement.Simult(len(qubits), adjust)
#    kw = {'stats': stats, 'ref qubit': ref}
#    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)
#
#    def func(dphase_amp, dphase_len):
#        for q in qubits:
#            q.xy = env.NOTHING
#            q.z = env.NOTHING
#
#        q = qubits[adjust]
#        r = qubits[ref]
#
#        dt = max(qubit['piLen'] for qubit in qubits)
#        tcouple = q['wZpulseLen']
#        tcorr = dphase_len
#
#        tz = dt
#        tc = dt + tcouple
#        tp = dt + tcouple + tcorr + dt
#        tm = dt + tcouple + tcorr + 2*dt
#
#        q.xy = eh.piHalfPulse(q, 0) + eh.piHalfPulse(q, tp)
#        q.z = env.rect(tz, tcouple, q['wZpulseAmp']) + env.rect(tc, tcorr, dphase_amp)
#
#        zpafunc = get_zpa_func(r)
#        zpa = zpafunc(r['f10'] - (q['f10'] - r['f10'])) # move the ref qubit out of the way
#        r.z = env.rect(tz, tcouple, zpa)
#
#        return measure(qubits, tm, stats=stats)
#    return sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)


def w_state(sample, t_couple=0, delay=0*ns, zpas=None, overshoots=None, pi_pulse_on=0, stats=600L,
            measure=None, phase_fit=True,
            name='W-state MQ', **kwargs):
    sample, qubits = util.loadQubits(sample)

    if zpas is None: zpas = [q['wZpulseAmp'] for q in qubits]
    if overshoots is None: overshoots = [q['wZpulseOvershoot'] for q in qubits]

    axes = ([(t_couple, 'Coupling time'), (delay, 'Delay')] +
            [(zpa, 'Z-pulse amplitude %i' % i) for i, zpa in enumerate(zpas)] +
            [(overshoot, 'Overshoot %i' % i) for i, overshoot in enumerate(overshoots)])
    kw = {'stats': stats,
          'pi_pulse_on': pi_pulse_on}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, t_couple, delay, zpa0, zpa1, zpa2, overshoot0, overshoot1, overshoot2):
        dt = max(q['piLen'] for q in qubits)
        zp_len = min(t_couple, delay)
        tp0 = 0
        tz = dt/2
        tm = dt/2 + delay + dt/2

        zpas = (zpa0, zpa1, zpa2)
        overshoots = (overshoot0, overshoot1, overshoot2)
        for i, (q, zpa, overshoot) in enumerate(zip(qubits, zpas, overshoots)):
            if i == pi_pulse_on:
                q.xy = eh.piPulse(q, tp0)
            else:
                q.xy = env.NOTHING

            q.z = env.rect(tz, zp_len, zpa, overshoot=overshoot)

            # adjust phase of tomography pulses
            if phase_fit:
                q['tomoPhase'] = np.polyval(q['wDphaseFit'].asarray, zp_len)
            else: # use slope only
                q['tomoPhase'] = 2*np.pi * zp_len * q['wDphaseSlope']
        return measurement.do(measure, server, qubits, tm, stats=stats)
    return sweeps.grid(func, axes, dataset=dataset, **kwargs)

def complexSweep(displacement, sweepTime):
    return [[d,sT] for d in displacement for sT in sweepTime]

def quanTrans1D(sample, probeLen=st.arangePQ(0,500,1,ns), measure=1, stats=1200L, storageTime=10*ns,
                name='Quantum transducer MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    qg = qubits[1-measure]
    qm = qubits[measure]
    nameEx = [' q1->q0', ' q0->q1']

    axes = [(probeLen, 'swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, 'n=1 '+str(storageTime['ns'])+str(storageTime.unit)+name+nameEx[measure], axes, measure=measure, kw=kw)
    #t0 = time.time()

    def func(server, curr):
         start = 0
         qg.xy = eh.mix(qg, eh.piPulseHD(qg, start))
         start += qg['piLen']/2
         qg.z = env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
         start += qg.noonSwapLen0s[0]+storageTime
         # 1 photon in res0 --> TOMO
         qg.z += env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
         # TOMO
         start += qg.noonSwapLen0s[0]
         qg.z += env.rect(start, qg.noonSwapLenCs[0], qg.noonSwapAmpC)
         # 1 photon in resC
         start += qg.noonSwapLenCs[0]+storageTime
         qm.z = env.rect(start, qm.noonSwapLenCs[0], qm.noonSwapAmpC)
         start += qm.noonSwapLenCs[0]
         qm.z += env.rect(start, qm.noonSwapLen0s[0], qm.noonSwapAmp0)
         start += qm.noonSwapLen0s[0]+storageTime
         # 1 photon in res1 --> TOMO

         qm.z += env.rect(start, curr, qm.noonSwapAmp0Read)

         start += curr
         qm.z += eh.measurePulse(qm, start)

         qm['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    #t1 = time.time()
    #print t1-t0
    return

def quanTrans1DRabi(sample, probeLen=st.arangePQ(0,115,1,ns), disp0=None, disp1=None, n=1, measure=[0,1], stats=1200L, storageTime=10*ns,
                    name='Quantum transducer 1D Rabi MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
#    qg = qubits[1-measure]
#    qm = qubits[measure]
#    nameEx = [' q1->q0', ' q0->q1']

    q0 = qubits[measure[0]]
    r0 = qubits[measure[0]+2]
    q1 = qubits[measure[1]]
    r1 = qubits[measure[1]+2]
    nameEx = [' q0->q1',' q1->q0']

    axes = [(probeLen, 'swap pulse length')]

    kw = {'stats': stats,
          'measure': measure,
          'disp0': disp0,
          'disp1': disp1}
    #kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, 'n=1 '+str(storageTime['ns'])+str(storageTime.unit)+name+nameEx[measure[0]], axes, measure=measure, kw=kw)
    #t0 = time.time()

    def func(server, curr):
         start = 0
         q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
         start += q0['piLen']/2
         q0.z = env.rect(start, q0.noonSwapLen0s[0], q0.noonSwapAmp0)
         start += q0.noonSwapLen0s[0]+storageTime
         # 1 photon in r0
         q0.z += env.rect(start, 20.0*q0.noonSwapLen0s[0]/10.0, q0.noonSwapAmp0)
         start += 20.0*q0.noonSwapLen0s[0]/10.0
         q0.z += env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
         start += q0.noonSwapLenCs[0]+storageTime
         # 1 partial photon in rC

         q0.z += env.rect(start, curr, q0.noonSwapAmp0Read)
         q1.z = env.rect(start, curr, q1.noonSwapAmpCRead)

         start += curr
         q0.z += eh.measurePulse(q0, start)
         q1.z += eh.measurePulse(q1, start)

         q0['readout'] = True
         q1['readout'] = True

         return runQubits(server, qubits, stats=stats)

    results = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    #t1 = time.time()
    #print t1-t0
    return

def quanTrans2DRabi(sample, probeLen=st.arangePQ(0,70,1,ns), QTRabiSwapLen0=st.arangePQ(0,240.0,3.0,ns), measure=1, stats=6000L, storageTime=2*ns,
                    name='Quantum transducer 2D Rabi MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[1-measure]
    q1 = qubits[measure]
    # QTRabiSwapLen0 = st.arangePQ(0,8.0*q0.noonSwapLen0s[0],q0.noonSwapLen0s[0]/10.0,ns)
    nameEx = [' q1->q0', ' q0->q1']

    axes = [(probeLen, 'swap pulse length'), (QTRabiSwapLen0, 'QTRabiSwapLen0')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, 'n=1 '+name+nameEx[measure], axes, measure=measure, kw=kw)

    def func(server, curr, curr1):
        start = 0
        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2
        q0.z = env.rect(start, q0.noonSwapLen0s[0], q0.noonSwapAmp0)
        start += q0.noonSwapLen0s[0]+storageTime
        # 1 photon in r0
        q0.z += env.rect(start, curr1, q0.noonSwapAmp0)
        start += curr1
        q0.z += env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
        start += q0.noonSwapLenCs[0]+storageTime
        # 1 partial photon in rC

#        q0.z += env.rect(start, curr, q0.noonSwapAmp0Read)
#
#        start += curr
#        q0.z += eh.measurePulse(q0, start)
#
#        q0['readout'] = True

        q1.z = env.rect(start, curr, q1.noonSwapAmpCRead)

        start += curr
        q1.z += eh.measurePulse(q1, start)

        q1['readout'] = True

        return runQubits(server, qubits, stats=stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

def quanTrans1Tomo13(sample, probeLen=st.arangePQ(0,200,2,ns), dispm = [0.5], measure=1, stats=1200L, storageTime=10*ns,
                     name='Quantum transducer MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    qg = qubits[1-measure]
    qm = qubits[measure]
    rm = qubits[measure+2]
    rg = qubits[measure+1]
    nameEx = [' q1->q0', ' q0->q1']

    sweepPara = complexSweep(np.array(dispm)/rm.noonAmpScale.value,probeLen)

    kw = {'stats': stats,
    'measure': measure}
    dataset = sweeps.prepDataset(sample, 'n=1 '+str(storageTime['ns'])+str(storageTime.unit)+name+nameEx[measure],
       axes = [('rm displacement', 're'),('rm displacement', 'im'),
               ('swap pulse length', 'ns')], measure=measure, kw=kw)

    #t0 = time.time()

    def func(server, curr):
        am = curr[0]
        currLen = curr[1]

        start = 0
        qg.xy = eh.mix(qg, eh.piPulseHD(qg, start))
        start += qg['piLen']/2
        qg.z = env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
        start += qg.noonSwapLen0s[0]+storageTime+8.0*ns
        # 1 photon in res0 --> TOMO
        qg.z += env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
        # TOMO
        start += qg.noonSwapLen0s[0]
        qg.z += env.rect(start, qg.noonSwapLenCs[0], qg.noonSwapAmpC)
        # 1 photon in resC
        start += qg.noonSwapLenCs[0]+storageTime
        qm.z = env.rect(start, qm.noonSwapLenCs[0], qm.noonSwapAmpC)
        start += qm.noonSwapLenCs[0]
        qm.z += env.rect(start, qm.noonSwapLen0s[0], qm.noonSwapAmp0)
        start += qm.noonSwapLen0s[0]+storageTime+8.0*ns
        # 1 photon in res1 --> TOMO

        rm.xy = eh.mix(rm, env.gaussian(start+rm.piLen/2, rm.piFWHM,
                                        np.conjugate(am*rm.noonDrivePhase)), freq = 'fRes0')
        start += rm.piLen+8.0*ns

        qm.z += env.rect(start, currLen, qm.noonSwapAmp0Read)

        start += currLen
        qm.z += eh.measurePulse(qm, start)

        qm['readout'] = True

        data = yield runQubits(server, qubits, stats=stats, probs=[1])

        data = np.hstack(([am.real, am.imag, currLen], data))
        returnValue(data)
    result = sweeps.run(func, sweepPara, dataset=save and dataset, noisy=noisy)
    #t1 = time.time()
    #print t1-t0
    return

def quanTrans1Tomo11(sample, probeLen=st.arangePQ(0,200,2,ns), dispm = [0.5], measure=1, stats=1200L, storageTime=10*ns,
                     name='Quantum transducer MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    qg = qubits[1-measure]
    qm = qubits[measure]
    rm = qubits[measure+2]
    rg = qubits[measure+1]
    nameEx = [' q1->q0', ' q0->q1']

    sweepPara = complexSweep(np.array(dispm)/rg.noonAmpScale.value,probeLen)

    kw = {'stats': stats,
    'measure': measure}
    dataset = sweeps.prepDataset(sample, 'n=1 '+str(storageTime['ns'])+str(storageTime.unit)+name+nameEx[measure],
       axes = [('rg displacement', 're'),('rg displacement', 'im'),
               ('swap pulse length', 'ns')], measure=measure, kw=kw)

    #t0 = time.time()

    def func(server, curr):
        ag = curr[0]
        currLen = curr[1]

        start = 0
        qg.xy = eh.mix(qg, eh.piPulseHD(qg, start))
        start += qg['piLen']/2
        qg.z = env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
        start += qg.noonSwapLen0s[0]+storageTime+8.0*ns
        # 1 photon in res0 --> TOMO
#        qg.z += env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
#        # TOMO
#        start += qg.noonSwapLen0s[0]
#        qg.z += env.rect(start, qg.noonSwapLenCs[0], qg.noonSwapAmpC)
#        # 1 photon in resC
#        start += qg.noonSwapLenCs[0]+storageTime
#        qm.z = env.rect(start, qm.noonSwapLenCs[0], qm.noonSwapAmpC)
#        start += qm.noonSwapLenCs[0]
#        qm.z += env.rect(start, qm.noonSwapLen0s[0], qm.noonSwapAmp0)
#        start += qm.noonSwapLen0s[0]+storageTime+8.0*ns
#        # 1 photon in res1 --> TOMO

        rg.xy = eh.mix(rg, env.gaussian(start+rg.piLen/2, rg.piFWHM,
                                        np.conjugate(ag*rg.noonDrivePhase)), freq = 'fRes0')
        start += rg.piLen+8.0*ns

        qg.z += env.rect(start, currLen, qg.noonSwapAmp0Read)

        start += currLen
        qg.z += eh.measurePulse(qg, start)

        qg['readout'] = True

        data = yield runQubits(server, qubits, stats=stats, probs=[1])

        data = np.hstack(([ag.real, ag.imag, currLen], data))
        returnValue(data)
    result = sweeps.run(func, sweepPara, dataset=save and dataset, noisy=noisy)
    #t1 = time.time()
    #print t1-t0
    return

def nightTomoQT(sample):
    #alpha = array([1.1,1.45])[:,None] * exp(1.0j*linspace(0,2*pi,30,endpoint=False))[None,:]
    #alpha = reshape(alpha,size(alpha))
    #plot(real(alpha),imag(alpha))
    alpha = np.linspace(-2.0,2.0,25)
    alpha = alpha[:,None]+1.0j*alpha[None,:]
    alpha = np.reshape(alpha,np.size(alpha))

    quanTrans1Tomo13(sample, dispm=alpha, stats=1200L, save=True)
    quanTrans1Tomo11(sample, dispm=alpha, stats=1200L, save=True)

def HanoiTomoEnd(sample, probeLen=st.arangePQ(0,200,2,ns), dispm = [0.5], measure=1, stats=1200L, storageTime=10*ns, hanoiDelay=0.0*ns,
                 name='Hanoi TOMO end MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    qg = qubits[1-measure]
    qm = qubits[measure]
    rm = qubits[measure+2]
    rg = qubits[measure+1]
    nameEx = [' q1->q0', ' q0->q1']

    sweepPara = complexSweep(np.array(dispm)/rm.noonAmpScale.value,probeLen)

    kw = {'stats': stats,
    'measure': measure}
    dataset = sweeps.prepDataset(sample, 'n=1 '+str(storageTime['ns'])+str(storageTime.unit)+name+nameEx[measure],
       axes = [('rm displacement', 're'),('rm displacement', 'im'),
               ('swap pulse length', 'ns')], measure=measure, kw=kw)

    #t0 = time.time()

    def func(server, curr):
        am = curr[0]
        currLen = curr[1]

        start = 0
        qg.xy = eh.mix(qg, eh.piPulseHD(qg, start))
        start += qg['piLen']/2
        qg.z = env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
        start += qg.noonSwapLen0s[0]+hanoiDelay+qg['piLen']/2
        # 1 photon in res0
        qg.xy += eh.mix(qg, eh.piPulseHD(qg, start))
        start += qg['piLen']/2
        qg.z += env.rect(start, qg.noonSwapLen0s[1], qg.noonSwapAmp0)
        start += qg.noonSwapLen0s[1]+hanoiDelay+storageTime
        # 2 photons in res0

        qg.z += env.rect(start, qg.noonSwapLen0s[1], qg.noonSwapAmp0)
        start += qg.noonSwapLen0s[1]+hanoiDelay
        qg.z += env.rect(start, qg.noonSwapLenCs[0], qg.noonSwapAmpC)
        start += qg.noonSwapLenCs[0]+hanoiDelay
        # 1 photon in res0 and 1 photon in resC
        qg.z += env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
        start += qg.noonSwapLen0s[0]+hanoiDelay
        qg.z += env.rect(start, qg.noonSwapLenCs[1], qg.noonSwapAmpC)
        start += qg.noonSwapLenCs[1]+hanoiDelay+storageTime
        # 2 photons in resC

        qm.z = env.rect(start, qm.noonSwapLenCs[1], qm.noonSwapAmpC)
        start += qm.noonSwapLenCs[1]+hanoiDelay
        qm.z += env.rect(start, qm.noonSwapLen0s[0], qm.noonSwapAmp0)
        start += qm.noonSwapLen0s[0]+hanoiDelay
        # 1 photon in resC and 1 photon in res1
        qm.z += env.rect(start, qm.noonSwapLenCs[0], qm.noonSwapAmpC)
        start += qm.noonSwapLenCs[0]+hanoiDelay
        qm.z += env.rect(start, qm.noonSwapLen0s[1], qm.noonSwapAmp0)
        start += qm.noonSwapLen0s[1]+hanoiDelay+storageTime+2.0*ns
        # 2 photons in res1

        qm.z += env.rect(start, qm.noonSwapLenR1s[0], qm.noonSwapAmpR1)
        start += qm.noonSwapLenR1s[0]+4.0*ns

        rm.xy = eh.mix(rm, env.gaussian(start+rm.piLen/2, rm.piFWHM,
                                        np.conjugate(am*rm.noonDrivePhase)), freq = 'fRes0')
        start += rm.piLen+8.0*ns

        qm.z += env.rect(start, currLen, qm.noonSwapAmp0Read)

        start += currLen
        qm.z += eh.measurePulse(qm, start)

        qm['readout'] = True

        data = yield runQubits(server, qubits, stats=stats, probs=[1])

        data = np.hstack(([am.real, am.imag, currLen], data))
        returnValue(data)
    result = sweeps.run(func, sweepPara, dataset=save and dataset, noisy=noisy)
    #t1 = time.time()
    #print t1-t0
    return

def HanoiTomoStart(sample, probeLen=st.arangePQ(0,200,2,ns), dispm = [0.5], measure=1, stats=1200L, storageTime=10*ns, hanoiDelay=0.0*ns,
                   name='Hanoi TOMO start MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    qg = qubits[1-measure]
    qm = qubits[measure]
    rm = qubits[measure+2]
    rg = qubits[measure+1]
    nameEx = [' q1->q0', ' q0->q1']

    sweepPara = complexSweep(np.array(dispm)/rg.noonAmpScale.value,probeLen)

    kw = {'stats': stats,
    'measure': measure}
    dataset = sweeps.prepDataset(sample, 'n=1 '+str(storageTime['ns'])+str(storageTime.unit)+name+nameEx[measure],
       axes = [('rg displacement', 're'),('rg displacement', 'im'),
               ('swap pulse length', 'ns')], measure=measure, kw=kw)

    #t0 = time.time()

    def func(server, curr):
        ag = curr[0]
        currLen = curr[1]

        start = 0
        qg.xy = eh.mix(qg, eh.piPulseHD(qg, start))
        start += qg['piLen']/2
        qg.z = env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
        start += qg.noonSwapLen0s[0]+hanoiDelay+qg['piLen']/2
        # 1 photon in res0
        qg.xy += eh.mix(qg, eh.piPulseHD(qg, start))
        start += qg['piLen']/2
        qg.z += env.rect(start, qg.noonSwapLen0s[1], qg.noonSwapAmp0)
        start += qg.noonSwapLen0s[1]+hanoiDelay+storageTime+2.0*ns
        # 2 photons in res0

#        qg.z += env.rect(start, qg.noonSwapLen0s[1], qg.noonSwapAmp0)
#        start += qg.noonSwapLen0s[1]+hanoiDelay
#        qg.z += env.rect(start, qg.noonSwapLenCs[0], qg.noonSwapAmpC)
#        start += qg.noonSwapLenCs[0]+hanoiDelay
#        # 1 photon in res0 and 1 photon in resC
#        qg.z += env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
#        start += qg.noonSwapLen0s[0]+hanoiDelay
#        qg.z += env.rect(start, qg.noonSwapLenCs[1], qg.noonSwapAmpC)
#        start += qg.noonSwapLenCs[1]+hanoiDelay+storageTime
#        # 2 photons in resC
#
#        qm.z = env.rect(start, qm.noonSwapLenCs[1], qm.noonSwapAmpC)
#        start += qm.noonSwapLenCs[1]+hanoiDelay
#        qm.z += env.rect(start, qm.noonSwapLen0s[0], qm.noonSwapAmp0)
#        start += qm.noonSwapLen0s[0]+hanoiDelay
#        # 1 photon in resC and 1 photon in res1
#        qm.z += env.rect(start, qm.noonSwapLenCs[0], qm.noonSwapAmpC)
#        start += qm.noonSwapLenCs[0]+hanoiDelay
#        qm.z += env.rect(start, qm.noonSwapLen0s[1], qm.noonSwapAmp0)
#        start += qm.noonSwapLen0s[1]+hanoiDelay+storageTime+2.0*ns
#        # 2 photons in res1

        qg.z += env.rect(start, qg.noonSwapLenCs[0], qg.noonSwapAmpC)
        start += qg.noonSwapLenCs[0]+4.0*ns

        rg.xy = eh.mix(rg, env.gaussian(start+rg.piLen/2, rg.piFWHM,
                                        np.conjugate(ag*rg.noonDrivePhase)), freq = 'fRes0')
        start += rg.piLen+8.0*ns

        qg.z += env.rect(start, currLen, qg.noonSwapAmp0Read)

        start += currLen
        qg.z += eh.measurePulse(qg, start)

        qg['readout'] = True

        data = yield runQubits(server, qubits, stats=stats, probs=[1])

        data = np.hstack(([ag.real, ag.imag, currLen], data))
        returnValue(data)
    result = sweeps.run(func, sweepPara, dataset=save and dataset, noisy=noisy)
    #t1 = time.time()
    #print t1-t0
    return

def nightTomoStorageHanoi19052010Wed(sample):
    #alpha = array([1.1,1.45])[:,None] * exp(1.0j*linspace(0,2*pi,30,endpoint=False))[None,:]
    #alpha = reshape(alpha,size(alpha))
    #plot(real(alpha),imag(alpha))
    alpha = np.linspace(-2.5,2.5,25)
    alpha = alpha[:,None]+1.0j*alpha[None,:]
    alpha = np.reshape(alpha,np.size(alpha))

    HanoiTomoEnd(sample, dispm=alpha, stats=1200L, save=True)
    HanoiTomoStart(sample, dispm=alpha, stats=1200L, save=True)
    quanHanoiTowers2D(sample, probeLen=st.arangePQ(0,250,2,ns), measure=1, stats=1200L, storageTime=st.arangePQ(10,1330,20,ns), save=True)
    quanTrans1_2D(sample, probeLen=st.arangePQ(0,250,2,ns), measure=1, stats=1200L, storageTime=st.arangePQ(10,1670,20,ns), save=True)

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

def spectroscopy2DZauto(sample, freqScan=None, measure=0,
                        fluxBelow=0.01, fluxAbove=0.01, fluxStep=0.0005, sb_freq=0*GHz, stats=300L,
                        name='2D Z-pulse spectroscopy MQ', save=True, collect=False, noisy=True, update=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    qubit = qubits[measure]

    if freqScan is None:
        freq = st.nearest(qubit['f10'][GHz], 0.001)
        freqScan = np.arange(freq-0.1, freq+1.0, 0.005)
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
    if update:
        Qubits[measure]['calZpaFunc'] = poly
    return lambda f: np.polyval(poly, f[GHz]**4)

def nightHoleBurningSacns(sample):

    swap10(sample, swapLen=st.arangePQ(0,600,1,ns), swapAmp=np.arange(-0.2,0.3,0.001), save=True, measure=0)
    spectroscopy2DZauto(s, st.arangePQ(5.9,7.4,0.005,'GHz'), fluxBelow=0.03, fluxAbove=0.03, fluxStep=0.0005, save=True, measure=0)

def quanTrans1Hack(sample, probeLen=st.arangePQ(0,100,1,ns), measure=1, stats=600L, storageTime=10*ns,
                   name='Quantum transducer MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    qg = qubits[1-measure]
    qm = qubits[measure]
    nameEx = [' q1->q0', ' q0->q1']

    axes = [(probeLen, 'swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, 'n=1 '+str(storageTime['ns'])+str(storageTime.unit)+name+nameEx[measure], axes, measure=measure, kw=kw)
    #t0 = time.time()

    def func(server, curr):
         start = 0
#        qg.z = env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
#
#        start += qg.noonSwapLen0s[0]/2+storageTime
#         qg.z = env.rect(start, curr, qg.noonSwapAmp0Read)
#
#         start += curr
#         qg.z += eh.measurePulse(qg, start)
#
#         qg['readout'] = True

#        start = 0
#        qg.z = env.rect(start, qg.noonSwapLenCs[0], qg.noonSwapAmpC)
#
#        start += qg.noonSwapLenCs[0]/2+storageTime
#         qg.z = env.rect(start, curr, qg.noonSwapAmpC)
#
#         start += curr
#         qg.z += eh.measurePulse(qg, start)
#
#         qg['readout'] = True

#         qm.z = env.rect(start, curr, qm.noonSwapAmpCRead)
#
#         start += curr
#         qm.z += eh.measurePulse(qm, start)
#
#         qm['readout'] = True

#        start = 0
#        qm.z = env.rect(start, qm.noonSwapLen0s[0], qm.noonSwapAmp0)
#
#        start += qm.noonSwapLen0s[0]/2+storageTime
#         qm.z = env.rect(start, curr, qm.noonSwapAmp0Read)
#
#         start += curr
#         qm.z += eh.measurePulse(qm, start)
#
#         qm['readout'] = True

#        start = 0
#         qg.xy = eh.mix(qg, eh.piPulseHD(qg, start))
#         start += qg['piLen']/2
#         qg.z = env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
#         start += qg.noonSwapLen0s[0]+storageTime
         #1 photon in res0 --> TOMO

#         qg.z += env.rect(start, curr, qg.noonSwapAmp0Read)
#
#         start += curr
#         qg.z += eh.measurePulse(qg, start)
#
#         qg['readout'] = True

#         qg.z += env.rect(start, curr, qg.noonSwapAmpC)
#
#         start += curr
#         qg.z += eh.measurePulse(qg, start)
#
#         qg['readout'] = True

#         qm.z = env.rect(start, curr, qm.noonSwapAmpC)
#
#         start += curr
#         qm.z += eh.measurePulse(qm, start)
#
#         qm['readout'] = True

#         qm.z = env.rect(start, curr, qm.noonSwapAmp0)
#
#         start += curr
#         qm.z += eh.measurePulse(qm, start)
#
#         qm['readout'] = True

#        qg.xy = eh.mix(qg, eh.piPulseHD(qg, start))
#        start += qg['piLen']/2
#        qg.z = env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
#        start += qg.noonSwapLen0s[0]+storageTime
#        # 1 photon in res0 --> TOMO
#        qg.z += env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
#        # TOMO
#        start += qg.noonSwapLen0s[0]
#        qg.z += env.rect(start, qg.noonSwapLenCs[0], qg.noonSwapAmpC)
#        # 1 photon in resC
#        start += qg.noonSwapLenCs[0]+storageTime
#        qm.z = env.rect(start, qm.noonSwapLenCs[0], qm.noonSwapAmpC)
#        start += qm.noonSwapLenCs[0]
#        qm.z += env.rect(start, qm.noonSwapLen0s[0], qm.noonSwapAmp0)
#        start += qm.noonSwapLen0s[0]+storageTime
#        # 1 photon in res1 --> TOMO
#
#        qm.z += env.rect(start, curr, qm.noonSwapAmp0Read)
#
#        start += curr
#        qm.z += eh.measurePulse(qm, start)
#
#        qm['readout'] = True

#         start = 0
#         qg.xy = eh.mix(qg, eh.piPulseHD(qg, start))
#         start += qg['piLen']/2
#         qg.z = env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
#         start += qg.noonSwapLen0s[0]+storageTime
#         # 1 photon in res0 --> TOMO
#         qg.z += env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
#         # TOMO
#         start += qg.noonSwapLen0s[0]
#         qg.z += env.rect(start, qg.noonSwapLenCs[0], qg.noonSwapAmpC)
#         # 1 photon in resC
#         start += qg.noonSwapLenCs[0]+storageTime

#         qg.z += env.rect(start, curr, qg.noonSwapAmp0Read)
#
#         start += curr
#         qg.z += eh.measurePulse(qg, start)
#
#         qg['readout'] = True

#         qg.z += env.rect(start, curr, qg.noonSwapAmpC)
#
#         start += curr
#         qg.z += eh.measurePulse(qg, start)
#
#         qg['readout'] = True


#         qm.z = env.rect(start, curr, qm.noonSwapAmpCRead)
#
#         start += curr
#         qm.z += eh.measurePulse(qm, start)
#
#         qm['readout'] = True

#         qm.z = env.rect(start, curr, qm.noonSwapAmp0Read)
#
#         start += curr
#         qm.z += eh.measurePulse(qm, start)
#
#         qm['readout'] = True

#         start = 0
#         qg.xy = eh.mix(qg, eh.piPulseHD(qg, start))
#         start += qg['piLen']/2
#         qg.z = env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
#         start += qg.noonSwapLen0s[0]+storageTime
#         # 1 photon in res0 --> TOMO
#         qg.z += env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
#         # TOMO
#         start += qg.noonSwapLen0s[0]
#         qg.z += env.rect(start, qg.noonSwapLenCs[0], qg.noonSwapAmpC)
#         # 1 photon in resC
#         start += qg.noonSwapLenCs[0]+storageTime
#         qm.z = env.rect(start, qm.noonSwapLenCs[0], qm.noonSwapAmpC)
#         start += qm.noonSwapLenCs[0]
#         qm.z += env.rect(start, qm.noonSwapLen0s[0], qm.noonSwapAmp0)
#         start += qm.noonSwapLen0s[0]+storageTime
#         #1 photon in res1 --> TOMO

#         qg.z += env.rect(start, curr, qg.noonSwapAmp0Read)
#
#         start += curr
#         qg.z += eh.measurePulse(qg, start)
#
#         qg['readout'] = True

#         qg.z += env.rect(start, curr, qg.noonSwapAmpC)
#
#         start += curr
#         qg.z += eh.measurePulse(qg, start)
#
#         qg['readout'] = True

#         qm.z += env.rect(start, curr, qm.noonSwapAmpCRead)
#
#         start += curr
#         qm.z += eh.measurePulse(qm, start)
#
#         qm['readout'] = True

#         qm.z += env.rect(start, curr, qm.noonSwapAmp0Read)
#
#         start += curr
#         qm.z += eh.measurePulse(qm, start)
#
#         qm['readout'] = True

#         start = 0
         qg.xy = eh.mix(qg, eh.piPulseHD(qg, start))
         start += qg['piLen']/2
         qg.z = env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
         start += qg.noonSwapLen0s[0]+storageTime
         # 1 photon in res0 --> TOMO
         qg.z += env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
         # TOMO
         start += qg.noonSwapLen0s[0]
         qg.z += env.rect(start, qg.noonSwapLenCs[0], qg.noonSwapAmpC)
         # 1 photon in resC
         start += qg.noonSwapLenCs[0]+storageTime
         qm.z = env.rect(start, qm.noonSwapLenCs[0], qm.noonSwapAmpC)
         start += qm.noonSwapLenCs[0]
         qm.z += env.rect(start, qm.noonSwapLen0s[0], qm.noonSwapAmp0)
         start += qm.noonSwapLen0s[0]+storageTime
         # 1 photon in res1 --> TOMO
         qm.z += env.rect(start, qm.noonSwapLen0s[0], qm.noonSwapAmp0)
         start += qm.noonSwapLen0s[0]
         qm.z += env.rect(start, qm.noonSwapLenCs[0], qm.noonSwapAmpC)
         start += qm.noonSwapLenCs[0]+storageTime

#         qg.z += env.rect(start, curr, qg.noonSwapAmpCRead)
#
#         start += curr
#         qg.z += eh.measurePulse(qg, start)
#
#         qg['readout'] = True

#         qg.z += env.rect(start, curr, qg.noonSwapAmpC)
#
#         start += curr
#         qg.z += eh.measurePulse(qg, start)
#
#         qg['readout'] = True

         qm.z += env.rect(start, curr, qm.noonSwapAmp0Read)

         start += curr
         qm.z += eh.measurePulse(qm, start)

         qm['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    #t1 = time.time()
    #print t1-t0
    return

def quanTransR0R1(sample, probeLen=st.arangePQ(0,100,1,ns), measure=1, stats=600L, storageTime=10*ns,
                  name='Quantum transducer MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    qg = qubits[1-measure]
    qm = qubits[measure]
    nameEx = [' q1->q0', ' q0->q1']

    axes = [(probeLen, 'swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, 'n=1 '+str(storageTime['ns'])+str(storageTime.unit)+name+nameEx[measure], axes, measure=measure, kw=kw)
    #t0 = time.time()

    def func(server, curr):
         start = 0
         qg.xy = eh.mix(qg, eh.piPulseHD(qg, start))
         start += qg['piLen']/2
         qg.z = env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
         start += qg.noonSwapLen0s[0]+storageTime

         qg.z += env.rect(start, curr, qg.noonSwapAmp0Read)

         start += curr
         qg.z += eh.measurePulse(qg, start)

         qg['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    #t1 = time.time()
    #print t1-t0
    return

def quanHanoiTowers(sample, probeLen=st.arangePQ(0,500,1,ns), measure=1, stats=600L, storageTime=10*ns, hanoiDelay=0*ns,
                    name='Quantum Hanoi Towers MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    qg = qubits[1-measure]
    qm = qubits[measure]
    nameEx = [' q1->q0', ' q0->q1']

    axes = [(probeLen, 'swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, 'n=2 '+str(storageTime['ns'])+str(storageTime.unit)+name+nameEx[measure], axes, measure=measure, kw=kw)
    t0 = time.time()

    def func(server, curr):
        start = 0
        qg.xy = eh.mix(qg, eh.piPulseHD(qg, start))
        start += qg['piLen']/2
        qg.z = env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
        start += qg.noonSwapLen0s[0]+hanoiDelay+qg['piLen']/2
        # 1 photon in res0
        qg.xy += eh.mix(qg, eh.piPulseHD(qg, start))
        start += qg['piLen']/2
        qg.z += env.rect(start, qg.noonSwapLen0s[1], qg.noonSwapAmp0)
        start += qg.noonSwapLen0s[1]+hanoiDelay+storageTime
        # 2 photons in res0

        qg.z += env.rect(start, qg.noonSwapLen0s[1], qg.noonSwapAmp0)
        start += qg.noonSwapLen0s[1]+hanoiDelay
        qg.z += env.rect(start, qg.noonSwapLenCs[0], qg.noonSwapAmpC)
        start += qg.noonSwapLenCs[0]+hanoiDelay
        # 1 photon in res0 and 1 photon in resC
        qg.z += env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
        start += qg.noonSwapLen0s[0]+hanoiDelay
        qg.z += env.rect(start, qg.noonSwapLenCs[1], qg.noonSwapAmpC)
        start += qg.noonSwapLenCs[1]+hanoiDelay+storageTime
        # 2 photons in resC

        qm.z = env.rect(start, qm.noonSwapLenCs[1], qm.noonSwapAmpC)
        start += qm.noonSwapLenCs[1]+hanoiDelay
        qm.z += env.rect(start, qm.noonSwapLen0s[0], qm.noonSwapAmp0)
        start += qm.noonSwapLen0s[0]+hanoiDelay
        # 1 photon in resC and 1 photon in res1
        qm.z += env.rect(start, qm.noonSwapLenCs[0], qm.noonSwapAmpC)
        start += qm.noonSwapLenCs[0]+hanoiDelay
        qm.z += env.rect(start, qm.noonSwapLen0s[1], qm.noonSwapAmp0)
        start += qm.noonSwapLen0s[1]+hanoiDelay+storageTime+2.0*ns
        # 2 photons in res1

        qm.z += env.rect(start, qm.noonSwapLenR1s[0], qm.noonSwapAmpR1)
        start += qm.noonSwapLenR1s[0]+2.0*ns
#
        # qg.z += env.rect(start, qg.resetLens[0], qg.resetAmps[0])
        # qm.z += env.rect(start, qm.resetLens[2], qm.resetAmps[2])
        # start += qg.resetLens[0]+4.0*ns
        # start += qm.resetLens[2]+4.0*ns

#        qg.z += env.rect(start, curr, qg.noonSwapAmpCRead)
#
#        start += curr
#        qg.z += eh.measurePulse(qg, start)
#
#        qg['readout'] = True

        qm.z += env.rect(start, curr, qm.noonSwapAmp0Read)

        start += curr
        qm.z += eh.measurePulse(qm, start)

        qm['readout'] = True

#        qm.z += env.rect(start, curr, qm.noonSwapAmp0Read)
#
#        start += curr
#        qm.z += eh.measurePulse(qm, start)
#
#        qm['readout'] = True

        return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    t1 = time.time()
    print t1-t0
    return

def quanTrans1_2D(sample, probeLen=st.arangePQ(0,240,2,ns), measure=1, stats=600L, storageTime=st.arangePQ(10,1630,20,ns),
                  name='Quantum transducer 2D MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    qg = qubits[1-measure]
    qm = qubits[measure]
    nameEx = [' q1->q0', ' q0->q1']

    axes = [(probeLen, 'swap pulse length'), (storageTime, 'storage time')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, 'n=1 '+name+nameEx[measure], axes, measure=measure, kw=kw)

    def func(server, curr, curr1):
        start = 0
        qg.xy = eh.mix(qg, eh.piPulseHD(qg, start))
        start += qg['piLen']/2
        qg.z = env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
        start += qg.noonSwapLen0s[0]+curr1
        # 1 photon in res0 --> TOMO
        qg.z += env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
        # TOMO
        start += qg.noonSwapLen0s[0]
        qg.z += env.rect(start, qg.noonSwapLenCs[0], qg.noonSwapAmpC)
        # 1 photon in resC
        start += qg.noonSwapLenCs[0]+curr1
        qm.z = env.rect(start, qm.noonSwapLenCs[0], qm.noonSwapAmpC)
        start += qm.noonSwapLenCs[0]
        qm.z += env.rect(start, qm.noonSwapLen0s[0], qm.noonSwapAmp0)
        start += qm.noonSwapLen0s[0]+curr1
        # 1 photon in res1 --> TOMO

        qm.z += env.rect(start, curr, qm.noonSwapAmp0Read)

        start += curr
        qm.z += eh.measurePulse(qm, start)

        qm['readout'] = True

        return runQubits(server, qubits, stats=stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

def quanHanoiTowers2D(sample, probeLen=st.arangePQ(0,250,2,ns), measure=1, stats=600L, storageTime=st.arangePQ(10,1110,20,ns), hanoiDelay=0*ns,
                      name='Quantum Hanoi Towers 2D MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    qg = qubits[1-measure]
    qm = qubits[measure]
    nameEx = [' q1->q0', ' q0->q1']

    axes = [(probeLen, 'swap pulse length'), (storageTime, 'storage time')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, 'n=2 '+name+nameEx[measure], axes, measure=measure, kw=kw)
#    t0 = time.time()

    def func(server, curr, curr1):
        start = 0
        qg.xy = eh.mix(qg, eh.piPulseHD(qg, start))
        start += qg['piLen']/2
        qg.z = env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
        start += qg.noonSwapLen0s[0]+hanoiDelay+qg['piLen']/2
        # 1 photon in res0
        qg.xy += eh.mix(qg, eh.piPulseHD(qg, start))
        start += qg['piLen']/2
        qg.z += env.rect(start, qg.noonSwapLen0s[1], qg.noonSwapAmp0)
        start += qg.noonSwapLen0s[1]+hanoiDelay+curr1
        # 2 photons in res0

        qg.z += env.rect(start, qg.noonSwapLen0s[1], qg.noonSwapAmp0)
        start += qg.noonSwapLen0s[1]+hanoiDelay
        qg.z += env.rect(start, qg.noonSwapLenCs[0], qg.noonSwapAmpC)
        start += qg.noonSwapLenCs[0]+hanoiDelay
        # 1 photon in res0 and 1 photon in resC
        qg.z += env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
        start += qg.noonSwapLen0s[0]+hanoiDelay
        qg.z += env.rect(start, qg.noonSwapLenCs[1], qg.noonSwapAmpC)
        start += qg.noonSwapLenCs[1]+hanoiDelay+curr1
        # 2 photons in resC

        qm.z = env.rect(start, qm.noonSwapLenCs[1], qm.noonSwapAmpC)
        start += qm.noonSwapLenCs[1]+hanoiDelay
        qm.z += env.rect(start, qm.noonSwapLen0s[0], qm.noonSwapAmp0)
        start += qm.noonSwapLen0s[0]+hanoiDelay
        # 1 photon in resC and 1 photon in res1
        qm.z += env.rect(start, qm.noonSwapLenCs[0], qm.noonSwapAmpC)
        start += qm.noonSwapLenCs[0]+hanoiDelay
        qm.z += env.rect(start, qm.noonSwapLen0s[1], qm.noonSwapAmp0)
        start += qm.noonSwapLen0s[1]+hanoiDelay+curr1
        # 2 photons in res1

        qm.z += env.rect(start, curr, qm.noonSwapAmp0Read)

        start += curr
        qm.z += eh.measurePulse(qm, start)

        qm['readout'] = True

        return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    #t1 = time.time()
    #print t1-t0
    return

def whereIsVacuum(sample, probeLen=st.arangePQ(0,100,1,ns), measure=1, stats=600L, storageTime=10*ns,
                  name='Where is the vacuum MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    qg = qubits[1-measure]
    qm = qubits[measure]
    nameEx = [' q1->q0', ' q0->q1']

    axes = [(probeLen, 'swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, 'n=1 '+str(storageTime['ns'])+str(storageTime.unit)+name+nameEx[measure], axes, measure=measure, kw=kw)
    #t0 = time.time()

    def func(server, curr):
#         start = 0
#        qg.z = env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
#
#        start += qg.noonSwapLen0s[0]/2+storageTime
#         qg.z = env.rect(start, curr, qg.noonSwapAmp0Read)
#
#         start += curr
#         qg.z += eh.measurePulse(qg, start)
#
#         qg['readout'] = True

#        start = 0
#        qg.z = env.rect(start, qg.noonSwapLenCs[0], qg.noonSwapAmpC)
#
#        start += qg.noonSwapLenCs[0]/2+storageTime
#         qg.z = env.rect(start, curr, qg.noonSwapAmpC)
#
#         start += curr
#         qg.z += eh.measurePulse(qg, start)
#
#         qg['readout'] = True

#        start = 0
#        qm.z = env.rect(start, qm.noonSwapLen0s[0], qm.noonSwapAmp0)
#
#        start += qm.noonSwapLen0s[0]/2+storageTime
#         qm.z = env.rect(start, curr, qm.noonSwapAmp0Read)
#
#         start += curr
#         qm.z += eh.measurePulse(qm, start)
#
#         qm['readout'] = True

#        start = 0
#        qg.xy = eh.mix(qg, eh.piPulseHD(qg, start))
#        start += qg['piLen']/2
#        qg.z = env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
#        start += qg.noonSwapLen0s[0]+storageTime
#        # 1 photon in res0 --> TOMO

#        qg.z += env.rect(start, curr, qg.noonSwapAmp0Read)
#
#        start += curr
#        qg.z += eh.measurePulse(qg, start)
#
#        qg['readout'] = True

#        qg.z += env.rect(start, curr, qg.noonSwapAmpC)
#
#        start += curr
#        qg.z += eh.measurePulse(qg, start)
#
#        qg['readout'] = True

#        qm.z = env.rect(start, curr, qm.noonSwapAmp0)
#
#        start += curr
#        qm.z += eh.measurePulse(qm, start)
#
#        qm['readout'] = True

#        qg.xy = eh.mix(qg, eh.piPulseHD(qg, start))
#        start += qg['piLen']/2
#        qg.z = env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
#        start += qg.noonSwapLen0s[0]+storageTime
#        # 1 photon in res0 --> TOMO
#        qg.z += env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
#        # TOMO
#        start += qg.noonSwapLen0s[0]
#        qg.z += env.rect(start, qg.noonSwapLenCs[0], qg.noonSwapAmpC)
#        # 1 photon in resC
#        start += qg.noonSwapLenCs[0]+storageTime
#        qm.z = env.rect(start, qm.noonSwapLenCs[0], qm.noonSwapAmpC)
#        start += qm.noonSwapLenCs[0]
#        qm.z += env.rect(start, qm.noonSwapLen0s[0], qm.noonSwapAmp0)
#        start += qm.noonSwapLen0s[0]+storageTime
#        # 1 photon in res1 --> TOMO
#
#        qm.z += env.rect(start, curr, qm.noonSwapAmp0Read)
#
#        start += curr
#        qm.z += eh.measurePulse(qm, start)
#
#        qm['readout'] = True

#         start = 0
#         qg.xy = eh.mix(qg, eh.piPulseHD(qg, start))
#         start += qg['piLen']/2
#         qg.z = env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
#         start += qg.noonSwapLen0s[0]+storageTime
#         # 1 photon in res0 --> TOMO
#         qg.z += env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
#         # TOMO
#         start += qg.noonSwapLen0s[0]
#         qg.z += env.rect(start, qg.noonSwapLenCs[0], qg.noonSwapAmpC)
#         # 1 photon in resC
#         start += qg.noonSwapLenCs[0]+storageTime

#         qg.z += env.rect(start, curr, qg.noonSwapAmp0Read)
#
#         start += curr
#         qg.z += eh.measurePulse(qg, start)
#
#         qg['readout'] = True

#         qg.z += env.rect(start, curr, qg.noonSwapAmpC)
#
#         start += curr
#         qg.z += eh.measurePulse(qg, start)
#
#         qg['readout'] = True

#         qm.z = env.rect(start, curr, qm.noonSwapAmp0Read)
#
#         start += curr
#         qm.z += eh.measurePulse(qm, start)
#
#         qm['readout'] = True

#         start = 0
#         qg.xy = eh.mix(qg, eh.piPulseHD(qg, start))
#         start += qg['piLen']/2
#         qg.z = env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
#         start += qg.noonSwapLen0s[0]+storageTime
#         # 1 photon in res0 --> TOMO
#         qg.z += env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
#         # TOMO
#         start += qg.noonSwapLen0s[0]
#         qg.z += env.rect(start, qg.noonSwapLenCs[0], qg.noonSwapAmpC)
#         # 1 photon in resC
#         start += qg.noonSwapLenCs[0]+storageTime
#         qm.z = env.rect(start, qm.noonSwapLenCs[0], qm.noonSwapAmpC)
#         start += qm.noonSwapLenCs[0]
#         qm.z += env.rect(start, qm.noonSwapLen0s[0], qm.noonSwapAmp0)
#         start += qm.noonSwapLen0s[0]+storageTime
#         # 1 photon in res1 --> TOMO

#         qg.z += env.rect(start, curr, qg.noonSwapAmp0Read)
#
#         start += curr
#         qg.z += eh.measurePulse(qg, start)
#
#         qg['readout'] = True

#         qg.z += env.rect(start, curr, qg.noonSwapAmpC)
#
#         start += curr
#         qg.z += eh.measurePulse(qg, start)
#
#         qg['readout'] = True

         start = 0
         qg.xy = eh.mix(qg, eh.piPulseHD(qg, start))
         start += qg['piLen']/2
         qg.z = env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
         start += qg.noonSwapLen0s[0]+storageTime+qg['piLen']/2
         # 1 photon in res0

         qm.xy = eh.mix(qm, eh.piPulseHD(qm, start))
         start += qm['piLen']/2
         qm.z = env.rect(start, qm.noonSwapLen0s[0], qm.noonSwapAmp0)
         start += qm.noonSwapLen0s[0]+storageTime
         # 1 photon in res1

#         qg.z += env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
#         # TOMO
#         start += qg.noonSwapLen0s[0]
#         qg.z += env.rect(start, qg.noonSwapLenCs[0], qg.noonSwapAmpC)
#         # 1 photon in resC
#         start += qg.noonSwapLenCs[0]+storageTime
#         qm.z = env.rect(start, qm.noonSwapLenCs[0], qm.noonSwapAmpC)
#         start += qm.noonSwapLenCs[0]
#         qm.z += env.rect(start, qm.noonSwapLen0s[0], qm.noonSwapAmp0)
#         start += qm.noonSwapLen0s[0]+storageTime
#         # 1 photon in res1 --> TOMO
#         qm.z += env.rect(start, qm.noonSwapLen0s[0], qm.noonSwapAmp0)
#         start += qm.noonSwapLen0s[0]
#         qm.z += env.rect(start, qm.noonSwapLenCs[0], qm.noonSwapAmpC)
#         start += qm.noonSwapLenCs[0]+storageTime
#
#         qg.z += env.rect(start, curr, qg.noonSwapAmp0Read)
#
#         start += curr
#         qg.z += eh.measurePulse(qg, start)
#
#         qg['readout'] = True
#
#         qg.z += env.rect(start, curr, qg.noonSwapAmpCRead)
#
#         start += curr
#         qg.z += eh.measurePulse(qg, start)
#
#         qg['readout'] = True
#
         qm.z += env.rect(start, curr, qm.noonSwapAmp0Read)

         start += curr
         qm.z += eh.measurePulse(qm, start)

         qm['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    #t1 = time.time()
    #print t1-t0
    return

# zpam=-0.036
def zPulseSpectroscopy(sample, freq=None, stats=300L, measure=0, sb_freq=0*GHz, zpam=None, detunings=None, uwave_amp=None,
                       save=True, name='z-pulse spectroscopy MQ', collect=False, noisy=True, update=False):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    q['readout'] = True

    # added
    if zpam is None: zpam = q['noonSwapAmpCRead']

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
        'sideband': sb_freq,
        'zpam': zpam
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
                dt = q['spectroscopyLen']
                q.xy = eh.spectroscopyPulse(q, 0, sb_freq)
                q.z = env.rect(0, dt, zpam) + eh.measurePulse(q, dt)
        eh.correctCrosstalkZ(qubits)
        return runQubits(server, qubits, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if update:
        adjust.adjust_frequency(Q, data)
    if collect:
        return data

# holeBurningFreq = 6.66*GHz,

def holeBurning(sample, delay=st.arangePQ(-10,20,1,ns)+st.arangePQ(20,100,2,ns)+st.arangePQ(100,500,4,ns)+st.arangePQ(500,1500,10,ns),
                zpa=-0.036, measure=0, stats=300L, name='Hole burning MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    axes = [(delay, 'Measure pulse delay')]
    kw = {'stats': stats,
          'zpa': zpa}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)
    # q0.piAmp*1.7
    # q0.piFWHM

    def func(server, delay):
        start = 0
        q0.xy = env.NOTHING
        for i in range(15):
            q0.xy += eh.mix(q0, env.gaussian(start, q0.piFWHM, 2.5, df = 0), freq = 'holeBurningFreq')
            # q0.xy += eh.mix(q0, env.gaussian(start, q0.piFWHM, q0.piAmp*1.0, df = 0), freq = holeBurningFreq)
            start += q0['piLen']

        start += q0['piLen']-2.0*ns
        q0.xy += eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2
        q0.z = env.rect(start, delay, zpa)
        start += delay
        q0.z += eh.measurePulse(q0, start)
        q0['readout'] = True

        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

def holeBurningRabi(sample, length=st.r[0:250:1,ns], amplitude=None, detuning=None, measureDelay=None, zpa=-0.036, measure=0, stats=1500L,
                    name='Hole burning - Rabi MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]

    if amplitude is None: amplitude = q0['piAmp']
    if detuning is None: detuning = 0
    if measureDelay is None: measureDelay = q0['piLen']/2.0

    axes = [(length, 'pulse length'),
            (amplitude, 'pulse height'),
            (detuning, 'detuning'),
            (measureDelay, 'measure delay')]

    #axes = [(delay, 'Measure pulse delay')]
    kw = {'stats': stats,
          'zpa': zpa}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)
    # q0.piAmp*1.7
    # q0.piFWHM

    def func(server, len, amp, df, delay):
        start = 0
        q0.xy = env.NOTHING
        for i in range(15):
            q0.xy += eh.mix(q0, env.gaussian(start, q0.piFWHM, 0.0, df = 0), freq = 'holeBurningFreq')
            # q0.xy += eh.mix(q0, env.gaussian(start, q0.piFWHM, q0.piAmp*1.0, df = 0), freq = holeBurningFreq)
            start += q0['piLen']

        start += q0['piLen']/2+0.0*ns
        q0.z = env.rect(start, len, zpa)
        q0.xy += eh.mix(q0, env.flattop(start, len, w=q0['piFWHM'], amp=amp), freq='holeBurningFreq')
        start += len
        q0.z += eh.measurePulse(q0, delay+start)
        q0['readout'] = True

        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

def JCAntiJC(sample, probeLen=st.arangePQ(0,250,1,ns), measure=0, stats=600L, amp=2.4,
             name='JC and anti JC', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[1+measure]
    nameEx = [' q0', ' q1']

    axes = [(probeLen, 'swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name+nameEx[measure], axes, measure=measure, kw=kw)

    def func(server, curr):
         start = 0
         # q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
         q0.xy = eh.mix(q0, eh.piHalfPulse(q0, start))
         start += q0['piLen']/2

         q0.xy += eh.mix(q0, env.flattop(start-100.0*ns, curr, w=q0['piFWHM'], amp=amp), freq='fRes0')
         # 300 ns
         q0.z = env.rect(start, curr, q0.noonSwapAmp0Read)

         start += curr
         q0.z += eh.measurePulse(q0, start)

         q0['readout'] = True

#         rm.xy = eh.mix(rm, env.gaussian(start+rm.piLen/2, rm.piFWHM,
#                                        np.conjugate(am*rm.noonDrivePhase)), freq = 'fRes0')
#        start += rm.piLen+8.0*ns
#
#        qm.z += env.rect(start, currLen, qm.noonSwapAmp0Read)
#
#        start += currLen
#        qm.z += eh.measurePulse(qm, start)
#
#        qm['readout'] = True
#
#        data = yield runQubits(server, qubits, stats=stats, probs=[1])
#
#        data = np.hstack(([am.real, am.imag, currLen], data))
#        returnValue(data)

         # vacuum Rabi SWAP between q0 and rC
#         qg.z += env.rect(start, qg.noonSwapLen0s[0], qg.noonSwapAmp0)
#         # TOMO
#         start += qg.noonSwapLen0s[0]
#         qg.z += env.rect(start, qg.noonSwapLenCs[0], qg.noonSwapAmpC)
#         # 1 photon in resC
#         start += qg.noonSwapLenCs[0]+storageTime
#         qm.z = env.rect(start, qm.noonSwapLenCs[0], qm.noonSwapAmpC)
#         start += qm.noonSwapLenCs[0]
#         qm.z += env.rect(start, qm.noonSwapLen0s[0], qm.noonSwapAmp0)
#         start += qm.noonSwapLen0s[0]+storageTime
#         # 1 photon in res1 --> TOMO

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def JCAntiJCTomo(sample, probeLen=st.arangePQ(0,750,1,ns), measure=0, stats=1200L, amp=2.4, disp0=None, npoints=30,
                 name='JC and anti JC', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[1+measure]
    r0 = qubits[2+measure]
    r1 = qubits[3+measure]
    nameEx = [' q0', ' q1']

    if disp0 is None: disp0 = np.array([1.1,1.45])[:,None] * np.exp(1.0j*np.linspace(0,2*np.pi,npoints,endpoint=False))[None,:]
    disp0 = np.reshape(disp0, np.size(disp0))

    sweepPara = complexSweep(np.array(disp0)/r0.noonAmpScale.value,probeLen)

    kw = {'stats': stats,
          'measure': measure}
    dataset = sweeps.prepDataset(sample, name+nameEx[measure],
                                 axes = [('r0 displacement', 're'),('r0 displacement', 'im'),
                                         ('swap pulse length', 'ns')], measure=measure, kw=kw)

    def func(server, curr):
        a0 = curr[0]
        currLen = curr[1]

        start = 0
        q0.xy = eh.mix(q0, eh.piHalfPulse(q0, start))
        start += q0['piLen']/2

        q0.xy += eh.mix(q0, env.flattop(start+10.0*ns, currLen, w=q0['piFWHM'], amp=amp), freq='fRes0')
        start += currLen

        r0.xy = eh.mix(r0, env.gaussian(start+r0.piLen/2, r0.piFWHM,
                                        np.conjugate(a0*r0.noonDrivePhase)), freq = 'fRes0')
        start += r0.piLen+4.0*ns

        q0.z = env.rect(start, currLen, q0.noonSwapAmp0Read)

        start += currLen
        q0.z += eh.measurePulse(q0, start)

        q0['readout'] = True

        data = yield runQubits(server, qubits, stats=stats, probs=[1])

        data = np.hstack(([a0.real, a0.imag, currLen], data))
        returnValue(data)

    result = sweeps.run(func, sweepPara, dataset=save and dataset, noisy=noisy)
    return

def quantumResetRes(sample, probeLen=st.arangePQ(0,500,1,ns), measure=0, stats=600L, storageTime=50*ns, zpa=-0.036, measureDelay=15.0*ns,
                    name='Quantum reset resonator MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]
    nameEx = [' q0', ' q1']

    axes = [(probeLen, 'swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, 'n=1 '+str(storageTime['ns'])+str(storageTime.unit)+name+nameEx[measure], axes, measure=measure, kw=kw)

    def func(server, curr):
         start = 0
         q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
         start += q0['piLen']/2
         q0.z = env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
         start += q0.noonSwapLenCs[0]+storageTime
         # 1 photon in rC

         q0.z += env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
         start += q0.noonSwapLenCs[0]
         q0.z += env.rect(start, curr, zpa)
         start += measureDelay

         q1.z = env.rect(start, curr, q1.noonSwapAmpCRead)

         start += curr
         q1.z += eh.measurePulse(q1, start)

         q1['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

def quantumResetQ(sample, probeLen=st.arangePQ(0,500,1,ns), measure=0, stats=600L, storageTime=10.0*ns, zpa=-0.036, measureDelay=15.0*ns,
                  resetTime=50.0*ns, name='Quantum reset qubit MQ', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[measure]
    q1 = qubits[measure+1]
    nameEx = [' q0', ' q1']

    axes = [(probeLen, 'swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, 'n=1 '+str(storageTime['ns'])+str(storageTime.unit)+name+nameEx[measure], axes, measure=measure, kw=kw)

    def func(server, curr):
         start = 0
         q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
         start += q0['piLen']/2
         q0.z = env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
         start += q0.noonSwapLenCs[0]+storageTime
         # 1 photon in rC

         q0.z += env.rect(start, q0.noonSwapLenCs[0], q0.noonSwapAmpC)
         start += q0.noonSwapLenCs[0]
         q0.z += env.rect(start, resetTime, zpa)
         start += resetTime

         q0.z += env.rect(start, curr, q0.noonSwapAmpCRead)

         start += curr
         q0.z += eh.measurePulse(q0, start)

         q0['readout'] = True

         return runQubits(server, qubits, stats=stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    return

# st.arangePQ(-10,20,1,ns)+st.arangePQ(20,
def qubitReset(sample, delay=st.arangePQ(0,100,2,ns)+st.arangePQ(100,500,4,ns)+st.arangePQ(500,1500,10,ns),
               stats=600L, measure=0, name='Qubit reset MQ', save=True, collect=True, noisy=True):
    """To be written."""
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    axes = [(delay, 'Measure pulse delay')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, delay):
        start = 0

#        q.xy = eh.mix(q, eh.piPulseHD(q, start))
#        start += q['piLen']/2

        q.z = env.rect(start, q.resetLens1[0], q.resetAmps1[0])
        start += q.resetLens1[0]

        start += delay
        q.z += eh.measurePulse(q, start)
        q['readout'] = True

        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
