import numpy as np

from labrad.units import Unit
ns = Unit('ns')

import pyle.envelopes as env
from pyle.dataking import envelopehelpers as eh
from pyle.dataking import measurement
from pyle.dataking import multiqubitPQ as mq
from pyle.dataking import sweeps
from pyle.dataking import util
import pyle.util.sweeptools as st


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
                xy = eh.piHalfPulse(q, tp0) * np.exp(1j*phase)
            elif i == ref:
                xy = eh.piHalfPulse(q, tp0)
            else:
                xy = env.NOTHING

            q.xy = eh.mix(q, xy)
            q.z = env.rect(tz, t_couple, zpa, overshoot=q['wZpulseOvershoot'])

        eh.correctCrosstalkZ(qubits)
        return measure(server, qubits, tm, stats=stats)
    return sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)


def swap_dphase_adjust(sample, dphase, stats=600L, adjust=0, ref=1,
                       name='Swap phase adjust', save=True, collect=True, noisy=True):
    sample, qubits = util.loadQubits(sample)

    axes = [(dphase, 'dphase')]
    measure = measurement.Simult(len(qubits), adjust)
    kw = {'stats': stats, 'adjust qubit': adjust, 'ref qubit': ref}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, dphase):
        qa = qubits[adjust]
        qr = qubits[ref]

        dt = max(qubit['piLen'] for qubit in qubits)
        tcouple = qa['swapLen']

        tz = dt/2
        tp = dt/2 + tcouple + dt/2
        tm = dt/2 + tcouple + 2*dt/2

        for i, q in enumerate(qubits):
            if i == adjust:
                q.xy = eh.mix(q, eh.piHalfPulse(q, 0) + eh.piHalfPulse(q, tp, phase=dphase))
                q.z = env.rect(tz, tcouple, q['swapAmp'], overshoot=q['swapOvershoot'])
            elif i == ref:
                zpafunc = mq.get_zpa_func(q)
                zpa = zpafunc(qr['f10'] - (qa['f10'] - qr['f10'])) # move the ref qubit out of the way
                q.z = env.rect(tz, tcouple, zpa)
            else:
                q.z = env.NOTHING
        eh.correctCrosstalkZ(qubits)
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

        qa = qubits[adjust]
        qr = qubits[ref]

        dt = max(qubit['piLen'] for qubit in qubits)
        tcouple = qa['wZpulseLen']

        tz = dt/2
        tp = dt/2 + tcouple + dt/2
        tm = dt/2 + tcouple + 2*dt/2

        for i, q in enumerate(qubits):
            if i == adjust:
                q.xy = eh.mix(q, eh.piHalfPulse(q, 0) + eh.piHalfPulse(q, tp, phase=dphase))
                q.z = env.rect(tz, tcouple, q['wZpulseAmp'], overshoot=q['wZpulseOvershoot'])
            elif i == ref:
                zpafunc = mq.get_zpa_func(q)
                zpa = zpafunc(qr['f10'] - (qa['f10'] - qr['f10'])) # move the ref qubit out of the way
                q.z = env.rect(tz, tcouple, zpa)
            else:
                q.z = env.NOTHING

        eh.correctCrosstalkZ(qubits)
        return measure(server, qubits, tm, stats=stats)
    return sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)


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
                q.xy = eh.mix(q, eh.piPulse(q, tp0))
            else:
                q.xy = env.NOTHING

            q.z = env.rect(tz, zp_len, zpa, overshoot=overshoot)

            # adjust phase of tomography pulses
            if phase_fit:
                q['tomoPhase'] = np.polyval(np.asarray(q['wDphaseFit']), zp_len)
            else: # use slope only
                q['tomoPhase'] = 2*np.pi * zp_len * q['wDphaseSlope']
        eh.correctCrosstalkZ(qubits)
        return measurement.do(measure, server, qubits, tm, stats=stats)
    return sweeps.grid(func, axes, dataset=dataset, **kwargs)








# adjust the microwave phase for coupling

#def seq_phase_adjust(qubits, adjust, ref, phase, t_couple):
#    dt = max(q['pi_len'] for q in qubits)
#    tp0 = 0
#    tz = dt
#    tm = dt + t_couple + dt
#
#    def f(i, q):
#        if i == adjust:
#            xy = eh.piHalfPulse(q, tp0) * np.exp(-1j*phase)
#        elif i == ref:
#            xy = eh.piHalfPulse(q, tp0)
#        else:
#            xy = env.NOTHING
#        z = env.rect(tz, t_couple, q['w_zpulse_amp'], overshoot=q['w_zpulse_overshoot'])
#        return q.where(xy=xy, z=z, uwave_phase=0) # turn off phase correction
#    return tuple(f(i, q) for i, q in enumerate(qubits)), tm
#
#
#def uwave_phase_adjust(sample, phase=0, t_couple=st.r[0:200:1,ns], adjust=0, ref=1, zpas=None, stats=3000L,
#                       name='uwave-phase cal', save=True, collect=True, noisy=True):
#    sample, qubits = util.loadQubits(sample)
#    measure = measurement.Simult(len(qubits), [adjust, ref])
#    axes = [(phase, 'uw phase %d' % adjust), (t_couple, 'Coupling time')]
#    kw = {
#        'stats': stats,
#        'adjust phase': adjust,
#        'reference phase': ref,
#        'zpas': zpas,
#    }
#    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)
#    def func(server, phase, t_couple):
#        qs, tm = seq_phase_adjust(qubits, adjust, ref, phase, t_couple)
#        return measure(server, qs, tm, stats=stats)
#    return sweeps.grid(func, axes, dataset=dataset if save else None, collect=collect, noisy=noisy)
#
#
#
#
## calibrate the phase accumulated during z-pulse
#
#def seq_tomo_phase_adjust(qubits, adjust, ref, phase, t_couple):
#    dt = max(q['pi_len'] for q in qubits)
#    tp0 = 0
#    tz = dt
#    tm = dt + t_couple + dt
#
#    def f(i, q):
#        if i == adjust:
#            xy = eh.piHalfPulse(q, tp0) * np.exp(-1j*phase)
#        elif i == ref:
#            xy = eh.piHalfPulse(q, tp0)
#        else:
#            xy = env.NOTHING
#        z = env.rect(tz, t_couple, q['w_zpulse_amp'], overshoot=q['w_zpulse_overshoot'])
#        return q.where(xy=xy, z=z, uwave_phase=0) # turn off phase correction
#    return tuple(f(i, q) for i, q in enumerate(qubits)), tm
#
#
#def tomo_phase_adjust(sample, dphase, stats=600L, adjust=0, ref=1,
#                      name='tomo-phase cal', save=True, collect=True, noisy=True):
#    sample, qubits = util.loadQubits(sample)
#
#    axes = [(dphase, 'dphase')]
#    measure = measurement.Simult(len(qubits), adjust)
#    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw={'stats': stats, 'ref_qubit': ref})
#    def func(server, dphase):
#        for q in qubits:
#            q.xy = env.NOTHING
#            q.z = env.NOTHING
#
#        q = qubits[adjust]
#        r = qubits[ref]
#
#        dt = max(qubit['pi_len'] for qubit in qubits)
#        tcouple = q['w_zpulse_len']
#
#        tz = dt
#        tp = dt + tcouple + dt
#        tm = dt + tcouple + 2*dt
#
#        q.xy = eh.piHalfPulse(q, 0) + eh.piHalfPulse(q, tp, phase=dphase)
#        q.z = env.rect(tz, tcouple, q['w_zpulse_amp'])
#
#        zpafunc = mq.get_zpa_func(r)
#        zpa = zpafunc(r['f_10'] - (q['f_10'] - r['f_10'])) # move the ref qubit out of the way
#        r.z = env.rect(tz, tcouple, zpa)
#
#        return measure(server, qubits, tm, stats=stats)
#    return sweeps.grid(func, axes, dataset=dataset if save else None, collect=collect, noisy=noisy)
#
#
#
## generate the w-state
#
#def seq_basic_w(qubits, pi_pulse_on, t_couple, delay):
#    dt = max(q['pi_len'] for q in qubits)
#    zp_len = min(t_couple, delay)
#    tp0 = 0
#    tz = dt
#    tm = dt + delay + dt
#
#    def f(i, q):
#        if i == pi_pulse_on:
#            xy = eh.piPulse(q, tp0)
#        else:
#            xy = env.NOTHING
#        z = env.rect(tz, zp_len, q['w_zpulse_amp'], overshoot=q['w_zpulse_overshoot'])
#        return q.where(xy=xy, z=z, tomo_phase=q['w_dphase'])
#    return tuple(f(i, q) for i, q in enumerate(qubits)), tm
#
#
#def w_state(sample, pi_pulse_on=1, t_couple=32*ns, delay=32*ns, stats=600L,
#            measure=measurement.Null(3), name='W-state MQ', **kwargs):
#    sample, qubits = util.loadQubits(sample)
#
#    axes = [(t_couple, 'Coupling time'), (delay, 'Delay')]
#    kw = {
#        'stats': stats,
#        'pi_pulse_on': pi_pulse_on,
#    }
#    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)
#
#    def func(server, t_couple, delay):
#        qs, tm = seq_basic_w(qubits, pi_pulse_on, t_couple)
#        return measure(server, qs, tm, stats=stats)
#    return sweeps.grid(func, axes, dataset=dataset, **kwargs)
#
#
#
#
##def w_dphase_adjust_z(sample, dphase_amp=None, dphase_len=None, stats=600L, adjust=0, ref=1,
##                     correct_xtalk=True, correct_uwave=False,
##                     name='W dphase adjust Z', save=True, collect=True, noisy=True):
##    sample, qubits = util.loadQubits(sample)
##    if dphase_amp is None:
##        dphase_amp = qubits[adjust]['w_dphase_amp']
##    if dphase_len is None:
##        dphase_len = qubits[adjust]['w_dphase_len']
##    axes = [(dphase_amp, 'dphase_amp'), (dphase_len, 'dphase_len')]
##    measure = measurement.Simult(len(qubits), adjust)
##    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw={'stats': stats, 'ref qubit': ref})
##    def func(dphase_amp, dphase_len):
##        for q in qubits:
##            q.xy = env.NOTHING
##            q.z = env.NOTHING
##
##        q = qubits[adjust]
##        r = qubits[ref]
##
##        dt = max(qubit['pi_len'] for qubit in qubits)
##        tcouple = q['w_zpulse_len']
##        tcorr = dphase_len
##
##        tz = dt
##        tc = dt + tcouple
##        tp = dt + tcouple + tcorr + dt
##        tm = dt + tcouple + tcorr + 2*dt
##
##        q.xy = eh.piHalfPulse(q, 0) + eh.piHalfPulse(q, tp)
##        q.z = env.rect(tz, tcouple, q['w_zpulse_amp']) + env.rect(tc, tcorr, dphase_amp)
##
##        zpafunc = mq.get_zpa_func(r)
##        zpa = zpafunc(r['f_10'] - (q['f_10'] - r['f_10'])) # move the ref qubit out of the way
##        r.z = env.rect(tz, tcouple, zpa)
##
##        return measure(qubits, tm, correct_xtalk=correct_xtalk, stats=stats)
##    return sweeps.grid(func, axes, dataset=dataset if save else None, collect=collect, noisy=noisy)
#
#
##def w_state(sample, pi_pulse_on=0, t_couple=0, delay=0*ns, zpas=None, overshoots=None, stats=600L,
##            measure=None, correct_xtalk=False, correct_uwave=False,
##            name='W-state MQ', save=True, collect=True, noisy=True, pipesize=10):
##    sample, qubits = util.loadQubits(sample)
##    if zpas is None:
##        zpas = [0.0] * len(qubits)
##    if overshoots is None:
##        #overshoots = [0.0] * len(qubits)
##        overshoots = [q['w_zpulse_overshoot'] for q in qubits]
##    axes = ([(t_couple, 'Coupling time'), (delay, 'Delay')] +
##            [(zpa, 'Z-pulse amplitude %i' % i) for i, zpa in enumerate(zpas)] +
##            [(overshoot, 'Overshoot %i' % i) for i, overshoot in enumerate(overshoots)])
##    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw={
##        'stats': stats,
##        'pi_pulse_on': pi_pulse_on,
##    })
##    def func(t_couple, delay, zpa0, zpa1, zpa2, overshoot0, overshoot1, overshoot2):
##        dt = max(q['pi_len'] for q in qubits)
##        zp_len = min(t_couple, delay)
##        tp0 = 0
##        tz = dt
##        tm = dt + delay + dt
##
##        zpas = (zpa0, zpa1, zpa2)
##        overshoots = (overshoot0, overshoot1, overshoot2)
##        for i, (q, zpa, overshoot) in enumerate(zip(qubits, zpas, overshoots)):
##            if i == pi_pulse_on:
##                q.xy = eh.piPulse(q, tp0)
##            else:
##                q.xy = env.NOTHING
##
##            q.z = env.rect(tz, zp_len, zpa, overshoot=overshoot)
##
##            # adjust phase of tomography pulses
##            # TODO use experimental calibration here, rather than theory
##            q['tomo_phase'] = 2*np.pi*zp_len*(qubits[pi_pulse_on]['f_10'] - q['f_10'])[GHz]
##        return measurement.do(measure, qubits, tm, stats=stats)
##    return sweeps.grid(func, axes, dataset=dataset if save else None, collect=collect, noisy=noisy, pipesize=pipesize)
#
#
##def seq_basic_w1(qubits, pi_pulse_on, t_couple, delay):
##    q0, q1, q2 = qubits
##
##    dt = max(q['pi_len'] for q in qubits)
##    zp_len = min(t_couple, delay)
##    tp0 = 0
##    tz = dt
##    tm = dt + delay + dt
##
##    return (
##        q0.where(
##            xy = env.NOTHING,
##            z  = env.rect(tz, zp_len, q0['w_zpulse_amp'], overshoot=q0['w_zpulse_overshoot'])
##        ),
##
##        q1.where(
##            xy = eh.piPulse(q1, tp0),
##            z  = env.rect(tz, zp_len, q1['w_zpulse_amp'], overshoot=q1['w_zpulse_overshoot'])
##        ),
##
##        q2.where(
##            xy = env.NOTHING,
##            z  = env.rect(tz, zp_len, q2['w_zpulse_amp'], overshoot=q2['w_zpulse_overshoot'])
##        ),
##
##    ), tm
#
#
##def seq_basic_w2(sample, pi_pulse_on, t_couple, delay):
##
##    dt = max(q['pi_len'] for q in qubits)
##    zp_len = min(t_couple, delay)
##    tp0 = 0
##    tz = dt
##    tm = dt + delay + dt
##
##    return sample.where(
##        q0__xy = env.NOTHING,
##        q0__z  = env.rect(tz, zp_len, q0['w_zpulse_amp'], overshoot=q0['w_zpulse_overshoot']),
##
##        q1__xy = eh.piPulse(q1, tp0),
##        q1__z  = env.rect(tz, zp_len, q1['w_zpulse_amp'], overshoot=q1['w_zpulse_overshoot']),
##
##        q2__xy = env.NOTHING,
##        q2__z  = env.rect(tz, zp_len, q2['w_zpulse_amp'], overshoot=q2['w_zpulse_overshoot']),
##
##        tm = tm,
##    )
#
#


