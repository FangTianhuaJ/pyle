import numpy as np
from scipy.linalg import expm

from labrad.units import Unit
ns, GHz = [Unit(s) for s in ('ns', 'GHz')]

import pyle
from pyle import envelopes as env
from pyle.pipeline import returnValue, FutureList
from pyle.dataking import envelopehelpers as eh
from pyle.dataking import measurement
from pyle.dataking import multiqubitPQ as mq
from pyle.dataking import sweeps
from pyle.dataking import util

from pyle.analysis import wernerAnalysis as werner


class GHZ(measurement.Measurer):
    def __init__(self, basisA=[(-np.pi/2, np.pi/2), (np.pi/2, 0)],
                       basisB=[(-np.pi/2, np.pi/2), (np.pi/2, 0)],
                       basisC=[(-np.pi/2, np.pi/2), (np.pi/2, 0)], null=False):
        """Measurement of anticorrelation parameter for GHZ state.

        Bases are specified as rotations to perform for the primed and
        unprimed settings, respectively.  Each rotation is specified as
        a tuple of (angle, axis).  The default for the primed setting
        is an X measurement, for which we rotate by -pi/2 about Y.  The
        default for the unprimed setting is a Y measurement, for which
        we rotate by pi/2 about X.  These settings are appropriate
        for the canonical |GHZ> state, namely |000> + |111>.

        Classically, one can show that G <= 0, while the pure GHZ state
        gives G = 1.  Any result showing G > 0 indicates nonclassicality.
        """
        measurement.Measurer.__init__(self, 3, measure=[0,1,2])
        self.basisA = basisA
        self.basisB = basisB
        self.basisC = basisC
        self.null = null

    def dependents(self):
        """Dependent variables for the data vault."""
        states = [bin(i)[2:].rjust(3,'0') for i in xrange(2**3)]
        bases = ['XXX', 'YYX', 'YXY', 'XYY']
        deps = [('Probability', 'III ' + state, '') for state in states]
        for basis in bases:
            deps += [('Probability', basis + ' ' + state, '') for state in states]
        deps += [('Anticorrelation', basis, '') for basis in bases]
        deps += [('Violation Param', 'G', '')]
        return deps

    def params(self):
        """Extra parameters for the data vault."""
        return {'measure': self.measure,
                'basisA': self.basisA,
                'basisB': self.basisB,
                'basisC': self.basisC}

    def __call__(self, server, qubits, t, **kw):
        """Tomographic measurement."""
        dt = max(q['piLen'] for q in qubits)
        tp = t + dt # TODO this timing might be overly conservative
        tm = t + 2*dt

        measureFunc = measurement.null if self.null else measurement.simult

        def addRotation(q, rot):
            """Add rotation pulse to a single qubit."""
            if rot is None:
                return q
            angle, axis = rot
            phase = axis + q['tomoPhase']
            pulse = eh.mix(q, eh.rotPulse(q, tp, angle=angle, phase=phase))
            return q.where(xy=q.get('xy', env.NOTHING) + pulse)

        def rotateAndMeasure(rotations):
            """Rotate all qubits and then measure them."""
            rotated = [addRotation(q, rot) for q, rot in zip(qubits, rotations)]
            return measureFunc(server, rotated, tm, self.measure, **kw)

        I = None # no rotation
        XA, YA = self.basisA
        XB, YB = self.basisB
        XC, YC = self.basisC

        rots = [(I, I, I), (XA, XB, XC), (YA, YB, XC), (YA, XB, YC), (XA, YB, YC)]

        futures = [rotateAndMeasure(rot) for rot in rots]
        Piii, Pxxx, Pyyx, Pyxy, Pxyy = yield FutureList(futures)

        # compute anticorrelations
        indices = [int(i,2) for i in ('000', '011', '101', '110')]
        Axxx = sum(Pxxx[indices])
        Ayyx = sum(Pyyx[indices])
        Ayxy = sum(Pyxy[indices])
        Axyy = sum(Pxyy[indices])

        G = Axxx - Ayyx - Ayxy - Axyy

        returnValue(np.hstack((Piii, Pxxx, Pyyx, Pyxy, Pxyy, [Axxx, Ayyx, Ayxy, Axyy, G])))


def ghz_simult_tomo_optimizer(sample, stats=1200, retune=None):
    sample, qubits = util.loadQubits(sample)

    axes = [('iteration', '')]
    measure = measurement.TomoNull(3)
    deps = [('initial phase', '0', ''), # phase of first pulse
            ('initial phase', '1', ''),
            ('initial phase', '2', ''),
            ('swap dphase', '0', ''), # delta phase change
            ('swap dphase', '1', ''),
            ('swap dphase', '2', ''),
            ('final phase', '0', ''), # phase of final pulse
            ('final phase', '1', ''),
            ('final phase', '2', '')]
    deps += measure.dependents()
    deps += [('fidelity', 'GHZ', '')]
    kw = {
        'stats': stats,
    }
    dataset = sweeps.prepDataset(sample, 'Tomo optimization', axes, dependents=deps, measure=measure, kw=kw)

    def func(server, stage, delay, i0, i1, i2, d0, d1, d2, f0, f1, f2):
        if retune == 'freq':
            for i in range(3):
                mq.freqtuner(sample, measure=i, noisy=False)

        qubits[0]['uwavePhase'] = None #phase0
        qubits[2]['uwavePhase'] = None #phase2
        qubits[0]['swapDphase'] = None #dphase0
        qubits[2]['swapDphase'] = None #dphase2
        qs, tm = seq_ghz_simult(qubits, delay, stage)
        return measurement.do(measure, server, qs, tm, stats=stats)

    with pyle.QubitSequencer() as server:
        with dataset as ds:
            def optimfunc(x):
                print 'measuring tomography...'
                data = func(server, 3, 0*ns, *x)
                print 'done.'
                print

                print 'calculating physical density matrix...'
                f0s = [q['measureF0'] for q in qubits]
                f1s = [q['measureF1'] for q in qubits]
                Us = pyle.tomo._qst_transforms['tomo3'][0]
                F = werner.Fxtf(3, f0s, f1s)
                rho = pyle.tomo.qst_mle(data, Us, F)
                overlap = pyle.overlap(rho, werner.rhoG)
                print 'done.'
                print 'overlap =', overlap
                print


                ds.add(np.hstack((x, data, [overlap])))



def psopipe(vars, n_particles):
    x = np.random.random((n_particles, len(vars))) # current position
    v = np.zeros((n_particles, len(vars))) # current velocity (per particle)
    lx = np.zeros((n_particles, len(vars))) # best position (per particle)
    lf = np.zeros(n_particles) # best fidelity (per particle)
    gx = np.zeros(len(vars)) # global best position
    gf = 0

    def func(curr):
        i, iteration = curr



def ghz_simult(sample, stage=3, phase0=None, phase2=None, dphase0=None, dphase2=None, delay=0*ns,
               retune=False,
               measure=None, stats=600L,
               name='GHZ simult MQ', **kwargs):
    sample, qubits = util.loadQubits(sample)

    # by default, we compute initial microwave phase from the fit
    dt = max(q['piLen'] for q in qubits)
    if phase0 is None:
        t0 = dt/2
        phase0 = np.polyval(qubits[0]['uwavePhaseFit'], t0)
        #phase0 = qubits[0]['uwave_phase']

    if phase2 is None:
        t0 = dt/2
        phase2 = np.polyval(qubits[2]['uwavePhaseFit'], t0)
        #phase2 = qubits[2]['uwave_phase']

    if dphase0 is None: dphase0 = qubits[0]['swapDphase']
    if dphase2 is None: dphase2 = qubits[2]['swapDphase']

    axes = [(stage, 'Sequence Progress'),
            (delay, 'Measure Delay'),
            (phase0, 'Microwave phase 0'),
            (phase2, 'Microwave phase 2'),
            (dphase0, 'Swap dphase 0'),
            (dphase2, 'Swap dphase 2')]
    kw = {
        'stats': stats,
    }
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, stage, delay, phase0, phase2, dphase0, dphase2):
        if retune == 'freq':
            for i in range(3):
                mq.freqtuner(sample, measure=i, noisy=False)

        qubits[0]['uwavePhase'] = phase0
        qubits[2]['uwavePhase'] = phase2
        qubits[0]['swapDphase'] = dphase0
        qubits[2]['swapDphase'] = dphase2
        qs, tm = seq_ghz_simult(qubits, delay, stage)
        return measurement.do(measure, server, qs, tm, stats=stats)
    return sweeps.grid(func, axes, dataset=dataset, **kwargs)


def seq_ghz_simult(qubits, delay=0*ns, stage=3, ref=1):
    """GHZ sequence using simultaneous coupling."""
    dt = max(q['piLen'] for q in qubits)
    tswap = sum(q['swapLen'] for q in qubits) / 2

    # compute times for various stages
    tp0 = 0
    tz = dt/2
    tp1 = tz + tswap + dt/2

    # compute measurement time based on number of stages
    if stage < 0:
        raise Exception('stage must be >= 0')
    if stage == 0:
        tm = 0
    elif stage <= 1:
        tm = tz
    elif stage <= 2:
        fraction = np.clip(stage-1, 0, 1)
        tm = tz + tswap * fraction
    elif stage <= 3:
        tm = tp1 + dt/2
    else:
        raise Exception('stage must be between 0 and 3')
    tm += delay # add measure delay

    # build xy and z sequences for each qubit
    def f(i, q):
        xy = env.NOTHING
        z = env.NOTHING
        ph = 0

        df = (qubits[ref]['f10'] - q['f10'])[GHz]

        if stage >= 0: # first microwave pulse
            fraction = np.clip(stage-0, 0, 1)
            xy += eh.rotPulse(q, tp0, angle=np.pi/2 * fraction, phase=np.pi/2) # Y

        if stage >= 1: # first swap
            fraction = np.clip(stage-1, 0, 1)
            if i != ref:
                tsw = tswap * fraction
                z += env.rect(tz, tsw, q['swapAmp'], overshoot=q['swapOvershoot'])
                #ph += q['swapDphase']*fraction
                ph += -2*np.pi*df*tsw

        if stage >= 3: # second microwave pulse
            fraction = np.clip(stage-3, 0, 1)
            xy += eh.rotPulse(q, tp1, angle=np.pi/2 * fraction, phase=ph) # X

        return q.where(xy=eh.mix(q, xy), z=z, tomoPhase=ph)
    qubits = [f(i, q) for i, q in enumerate(qubits)]
    eh.correctCrosstalkZ(qubits)
    return qubits, tm


def seq_ghz_simult_phase_tune(qubits, dphases0, dphases1, dphases2, delay=0*ns, stage=3, ref=1):
    """GHZ sequence using simultaneous coupling."""
    dt = max(q['piLen'] for q in qubits)
    tswap = sum(q['swapLen'] for q in qubits) / 2

    # compute times for various stages
    tp0 = 0
    tz = dt/2
    tp1 = tz + tswap + dt/2

    # compute measurement time based on number of stages
    if stage == 0:
        raise Exception('stage must be >= 0')
    elif stage == 1:
        tm = tz
    elif stage == 2:
        tm = tz + tswap
    elif stage == 3:
        tm = tp1 + dt/2
    else:
        raise Exception('stage must be be one of 1, 2 and 3')
    tm += delay # add measure delay

    # build xy and z sequences for each qubit
    def f(i, q, dph0, dph1, dph2):
        xy = env.NOTHING
        z = env.NOTHING

        ph0 = np.polyval(q['uwavePhaseFit'], tz)
        ph1 = ph0 + q['swapDphase']

        if stage > 0: # first microwave pulse
            xy += eh.rotPulse(q, tp0, angle=np.pi/2, phase=np.pi/2 + ph0 + dph0) # Y
            ph2 = ph0

        if stage > 1: # first swap
            if i != ref:
                z += env.rect(tz, tswap, q['swapAmp'], overshoot=q['swapOvershoot'])
            ph2 = ph1

        if stage > 2: # second microwave pulse
            xy += eh.rotPulse(q, tp1, angle=np.pi/2, phase=ph1 + dph1) # X

        return q.where(xy=eh.mix(q, xy), z=z, tomoPhase=ph2 + dph2)
    qubits = [f(i, *rest) for i, rest in enumerate(zip(qubits, dphases0, dphases1, dphases2))]
    eh.correctCrosstalkZ(qubits)
    return qubits, tm



def ghz_iswap(sample, stage=4, phase0=None, phase2=None, dphase0=None, dphase2=None, delay=0*ns, averages=1,
              swap_first=0, swap_second=2, swap_buffer=0*ns, echo_first=False, echo_second=False, retune=False,
              measure=None, stats=600L,
              name='GHZ iswap MQ', **kwargs):
    sample, qubits = util.loadQubits(sample)

    # by default, we compute initial microwave phase from the fit
    dt = max(q['piLen'] for q in qubits)
    if phase0 is None:
        if swap_first == 0:
            t0 = dt/2
        else:
            t0 = dt/2 + qubits[2]['swapLen'] + swap_buffer
        phase0 = np.polyval(qubits[0]['uwavePhaseFit'], t0)
        #phase0 = qubits[0]['uwave_phase']

    if phase2 is None:
        if swap_first == 2:
            t0 = dt/2
        else:
            t0 = dt/2 + qubits[0]['swapLen'] + swap_buffer
        phase2 = np.polyval(qubits[2]['uwavePhaseFit'], t0)
        #phase2 = qubits[2]['uwave_phase']

    if dphase0 is None: dphase0 = qubits[0]['swapDphase']
    if dphase2 is None: dphase2 = qubits[2]['swapDphase']

    if averages > 1:
        averages = range(averages)

    axes = [(averages, 'Averages'),
            (stage, 'Sequence Progress'),
            (delay, 'Measure Delay'),
            (phase0, 'Microwave phase 0'),
            (phase2, 'Microwave phase 2'),
            (dphase0, 'Swap dphase 0'),
            (dphase2, 'Swap dphase 2')]
    kw = {
        'stats': stats,
        'swap_first': swap_first,
        'swap_second': swap_second,
        'swap_buffer': swap_buffer,
        'echo_first': echo_first,
        'echo_second': echo_second,
    }
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, iteration, stage, delay, phase0, phase2, dphase0, dphase2):
        if retune == 'freq':
            for i in range(3):
                mq.freqtuner(sample, measure=i, noisy=False)

        qubits[0]['uwavePhase'] = phase0
        qubits[2]['uwavePhase'] = phase2
        qubits[0]['swapDphase'] = dphase0
        qubits[2]['swapDphase'] = dphase2
        qs, tm = seq_ghz_iswap(qubits, swap_first, swap_second, swap_buffer, delay, stage, echo_first=echo_first, echo_second=echo_second)
        return measurement.do(measure, server, qs, tm, stats=stats)
    return sweeps.grid(func, axes, dataset=dataset, **kwargs)


def seq_ghz_iswap(qubits, swap_first, swap_second, swap_buffer=0*ns, delay=0*ns, stage=4, ref=1, echo_first=False, echo_second=False):
    """GHZ sequence built using iSWAP gates, rather than simultaneous coupling."""
    dt = max(q['piLen'] for q in qubits)
    tswap0 = qubits[swap_first]['swapLen']
    tswap1 = qubits[swap_second]['swapLen']

    # compute times for various stages
    tp0 = 0
    tz0 = dt/2
    tz1 = tz0 + tswap0 + swap_buffer
    tp1 = tz1 + tswap1 + dt/2

    # compute measurement time based on number of stages
    if stage < 0:
        raise Exception('stage must be >= 0')
    if stage == 0:
        tm = 0
    elif stage <= 1:
        tm = tz0
    elif stage <= 2:
        fraction = np.clip(stage-1, 0, 1)
        tm = tz0 + tswap0 * fraction
    elif stage <= 3:
        fraction = np.clip(stage-2, 0, 1)
        tm = tz1 + tswap1 * fraction
    elif stage <= 4:
        tm = tp1 + dt/2
    else:
        raise Exception('stage must be between 0 and 4')
    tm += delay # add measure delay

    # build xy and z sequences for each qubit
    def f(i, q):
        xy = env.NOTHING
        z = env.NOTHING
        ph = 0

        df = (qubits[ref]['f10'] - q['f10'])[GHz]

        if stage >= 0: # first microwave pulse
            fraction = np.clip(stage-0, 0, 1)
            xy += eh.rotPulse(q, tp0, angle=np.pi/2 * fraction, phase=np.pi/2) # Y

        if stage >= 1: # first swap
            fraction = np.clip(stage-1, 0, 1)
            if i == swap_first:
                tsw = q['swapLen']*fraction
                z += env.rect(tz0, tsw, q['swapAmp'], overshoot=q['swapOvershoot'])
                #ph += q['swapDphase']*fraction
                ph += -2*np.pi*df*tsw

        if stage >= 2: # second swap
            fraction = np.clip(stage-2, 0, 1)
            if i == swap_second:
                tsw = q['swapLen']*fraction
                z += env.rect(tz1, tsw, q['swapAmp'], overshoot=q['swapOvershoot'])
                #ph += q['swapDphase']*fraction
                ph += -2*np.pi*df*tsw

                if echo_first:
                    # add echo pulses during first swap
                    print 'adding first-swap echos'
                    tpad = (tswap0 - 2*q['piLen']) / 3.0
                    techo0 = tz0 + tpad + q['piLen'] / 2.0
                    techo1 = tz0 + tpad + q['piLen'] + tpad + q['piLen'] / 2.0
                    xy += eh.piPulse(q, techo0, phase=0) # do x-pulses for first echo
                    xy += eh.piPulse(q, techo1, phase=0)

        if stage >= 3: # second microwave pulse
            fraction = np.clip(stage-3, 0, 1)
            if i != swap_second:
                xy += eh.rotPulse(q, tp1, angle=-np.pi/2 * fraction, phase=ph) # -X
            if (i == swap_first) and echo_second:
                # add echo pulses during second swap
                print 'adding second-swap echos'
                tpad = (tswap1 - 2*q['piLen']) / 3.0
                techo0 = tz1 + tpad + q['piLen'] / 2.0
                techo1 = tz1 + tpad + q['piLen'] + tpad + q['piLen'] / 2.0
                xy += eh.piPulse(q, techo0, phase=ph + np.pi/2) # do y-pulses for second echo
                xy += eh.piPulse(q, techo1, phase=ph + np.pi/2)

        return q.where(xy=eh.mix(q, xy), z=z, tomoPhase=ph)
    qubits = [f(i, q) for i, q in enumerate(qubits)]
    eh.correctCrosstalkZ(qubits)
    return qubits, tm


def ghz_iswap_tight(sample, stage=4, phase0=None, phase2=None, dphase0=None, dphase2=None, delay=0*ns, averages=1,
                    swap_first=0, swap_second=2, swap_buffer=0*ns, retune=False, echoA=None,
                    measure=None, stats=600L,
                    name='GHZ iswap tight MQ', **kwargs):
    sample, qubits = util.loadQubits(sample)

    # by default, we compute initial microwave phase from the fit
    dt = max(q['piLen'] for q in qubits)
    if phase0 is None:
        if swap_first == 0:
            t0 = dt/2
        else:
            t0 = dt/2 + qubits[2]['swapLen'] + swap_buffer
        phase0 = np.polyval(qubits[0]['uwavePhaseFit'], t0)
        #phase0 = qubits[0]['uwave_phase']

    if phase2 is None:
        if swap_first == 2:
            t0 = dt/2
        else:
            t0 = dt/2 + qubits[0]['swapLen'] + swap_buffer
        phase2 = np.polyval(qubits[2]['uwavePhaseFit'], t0)
        #phase2 = qubits[2]['uwave_phase']

    if dphase0 is None: dphase0 = qubits[0]['swapDphase']
    if dphase2 is None: dphase2 = qubits[2]['swapDphase']

    if averages > 1:
        averages = range(averages)

    axes = [(averages, 'Averages'),
            (stage, 'Sequence Progress'),
            (delay, 'Measure Delay'),
            (phase0, 'Microwave phase 0'),
            (phase2, 'Microwave phase 2'),
            (dphase0, 'Swap dphase 0'),
            (dphase2, 'Swap dphase 2')]
    kw = {
        'stats': stats,
        'swap_first': swap_first,
        'swap_second': swap_second,
        'swap_buffer': swap_buffer,
        'echoA': str(echoA),
    }
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, iteration, stage, delay, phase0, phase2, dphase0, dphase2):
        if retune == 'freq':
            for i in range(3):
                mq.freqtuner(sample, measure=i, noisy=False)

        qubits[0]['uwavePhase'] = phase0
        qubits[2]['uwavePhase'] = phase2
        qubits[0]['swapDphase'] = dphase0
        qubits[2]['swapDphase'] = dphase2
        qs, tm = seq_ghz_iswap_tight(qubits, swap_first, swap_second, swap_buffer, delay, stage, echoA=echoA)
        return measurement.do(measure, server, qs, tm, stats=stats)
    return sweeps.grid(func, axes, dataset=dataset, **kwargs)


def seq_ghz_iswap_tight(qubits, swap_first, swap_second, swap_buffer=0*ns, delay=0*ns, stage=4, ref=1, echoA=None):
    """GHZ sequence built using iSWAP gates, rather than simultaneous coupling."""
    dt = max(q['piLen'] for q in qubits)
    tswap0 = qubits[swap_first]['swapLen']
    tswap1 = qubits[swap_second]['swapLen']

    # compute times for various stages
    tp0 = 0
    tz0 = dt/2
    tz1 = tz0 + tswap0 + swap_buffer
    tp1 = tz1 + tswap1 + dt/2

    # compute measurement time based on number of stages
    if stage < 0:
        raise Exception('stage must be >= 0')
    if stage == 0:
        tm = 0
    elif stage == 1:
        tm = tz0
    elif stage == 2:
        tm = tz0 + tswap0
    elif stage == 3:
        tm = tz1 + tswap1
    elif stage == 4:
        tm = tp1 + dt/2
    else:
        raise Exception('stage must be between 0 and 4')
    tm += delay # add measure delay

    # build xy and z sequences for each qubit
    def f(i, q):
        xy = env.NOTHING
        z = env.NOTHING
        ph = 0

        df = (qubits[ref]['f10'] - q['f10'])[GHz]

        if stage >= 0: # first microwave pulse
            if i == swap_second:
                t = tz1 - swap_buffer - dt/2 # place right before second swap
            else:
                t = tp0
            xy += eh.rotPulse(q, t, angle=np.pi/2, phase=np.pi/2) # Y

        if stage >= 1: # first swap
            if i == swap_first:
                tsw = q['swapLen']
                z += env.rect(tz0, tsw, q['swapAmp'], overshoot=q['swapOvershoot'])
                #ph += q['swapDphase']*fraction
                ph += -2*np.pi*df*tsw

        if stage >= 2: # second swap
            if i == swap_second:
                tsw = q['swapLen']
                z += env.rect(tz1, tsw, q['swapAmp'], overshoot=q['swapOvershoot'])
                #ph += q['swapDphase']*fraction
                ph += -2*np.pi*df*tsw

        if stage >= 3: # second microwave pulse
            if i == swap_first:
                tpad = (tswap1 - 2*q['piLen']) / 3.0
                t0 = tz1 + tpad + q['piLen'] / 2.0
                t1 = tz1 + tpad + q['piLen'] + tpad + q['piLen'] / 2.0
                t2 = tp1
                X = ph
                Y = ph + np.pi/2
                if echoA == 'pipi_before':
                    xy += eh.rotPulse(q, t0, angle=np.pi, phase=Y) # Ypi
                    xy += eh.rotPulse(q, t1, angle=np.pi, phase=Y) # Ypi
                    xy += eh.rotPulse(q, t2, angle=-np.pi/2, phase=X) # -X
                elif echoA == 'pipi_after':
                    xy += eh.rotPulse(q, t0, angle=-np.pi/2, phase=X) # -X
                    xy += eh.rotPulse(q, t1, angle=np.pi, phase=Y) # Ypi
                    xy += eh.rotPulse(q, t2, angle=np.pi, phase=Y) # Ypi

                elif echoA == 'pimpi_before':
                    xy += eh.rotPulse(q, t0, angle=np.pi, phase=Y) # Ypi
                    xy += eh.rotPulse(q, t1, angle=-np.pi, phase=Y) # -Ypi
                    xy += eh.rotPulse(q, t2, angle=-np.pi/2, phase=X) # -X
                elif echoA == 'pimpi_after':
                    xy += eh.rotPulse(q, t0, angle=-np.pi/2, phase=X) # -X
                    xy += eh.rotPulse(q, t1, angle=np.pi, phase=Y) # Ypi
                    xy += eh.rotPulse(q, t2, angle=-np.pi, phase=Y) # -Ypi

                elif echoA == 'halfpi_before':
                    xy += eh.rotPulse(q, t0, angle=np.pi/2, phase=Y) # Y
                    xy += eh.rotPulse(q, t1, angle=-np.pi/2, phase=Y) # -Y
                    xy += eh.rotPulse(q, t2, angle=-np.pi/2, phase=X) # -X
                elif echoA == 'halfpi_after':
                    xy += eh.rotPulse(q, t0, angle=-np.pi/2, phase=X) # -X
                    xy += eh.rotPulse(q, t1, angle=np.pi/2, phase=Y) # Ypi
                    xy += eh.rotPulse(q, t2, angle=-np.pi/2, phase=Y) # -Ypi

                elif echoA == None:
                    #t = tz0 + tswap0 + swap_buffer + dt/2 # place right after first swap
                    xy += eh.rotPulse(q, t0, angle=-np.pi/2, phase=ph) # -X
                else:
                    raise Exception('unknown option for echoA: %s' % (echoA,))
            if i == ref:
                xy += eh.rotPulse(q, tp1, angle=-np.pi/2, phase=ph) # -X

        return q.where(xy=eh.mix(q, xy), z=z, tomoPhase=ph)
    qubits = [f(i, q) for i, q in enumerate(qubits)]
    eh.correctCrosstalkZ(qubits)
    return qubits, tm


def ghz_tomo_tester(sample, dphases0, dphases1, dphases2, stage=3, stats=600L):
    sample, qubits = util.loadQubits(sample)

    with pyle.QubitSequencer() as server:
        meas = measurement.TomoNull(3)
        qs, tm = seq_ghz_simult_phase_tune(qubits, dphases0, dphases1, dphases2, stage=stage)
        data = meas(server, qs, tm, stats=stats)

    import time

    print 'computing linear tomography...',
    start = time.time()
    rho0 = werner.doTomo(data, 'tomo3')
    end = time.time()
    print 'done. elapsed =', end - start

    print 'computing maximum-likelihood tomography...'
    pms = data.reshape((-1, 8))
    pxms = np.array([np.dot(werner.sierp(3), p) for p in pms])
    f0s = [q['measureF0'] for q in qubits]
    f1s = [q['measureF1'] for q in qubits]
    F = werner.Fxtf(3, f0s, f1s)
    Us = pyle.tomo._qst_transforms['tomo3'][0]

    start = time.time()
    rho = pyle.tomo.qst_mle(pxms[:,:-1], Us, F[:-1,:], rho0)
    end = time.time()
    print 'done. elapsed =', end - start
    print

    psi0 = np.array([1,0,0,0,0,0,0,0], dtype=complex)
    U = ghz_simult_trajectory([stage])[0]
    psi_th = np.dot(U, psi0)
    rho_th = pyle.ket2rho(psi_th)

    print 'theory:'
    print rho_th.real
    print rho_th.imag
    print

    print 'experiment:'
    print rho.real
    print rho.imag
    print

    print 'plotting...',
    start = time.time()
    fig = werner.plotRho(rho, rho_th)
    fig.suptitle('maximum-likelihood tomography')

    #fig = werner.plotRho(rho0, rho_th)
    #fig.suptitle('linear tomography')
    end = time.time()
    print 'done. elapsed =', end - start
    print 'overlap:', pyle.fidelity(rho, rho_th)
    print 'G:', werner.ghz_G(rho)

    return rho, fig




def tomo_tester(sample, ops, stats=600L):
    sample, qubits = util.loadQubits(sample)

    def op_matrix(ops):
        def m(op):
            if op is None:
                angle = 0
                phase = 0
            else:
                axis, angle = op
                phase = {
                    'X': 0,
                    'Y': np.pi/2,
                }[axis]
            sigma = np.cos(phase) * pyle.tomo.sigmaX + np.sin(phase) * pyle.tomo.sigmaY
            return pyle.tomo.Rmat(sigma, angle)
        return pyle.tensor(m(op) for op in ops)


    def op_sequence(ops):
        dt = max(q['piLen'] for q in qubits)
        tm = dt/2
        def f(q, op):
            if op is None:
                angle = 0
                phase = 0
            else:
                axis, angle = op
                phase = {
                    'X': 0,
                    'Y': np.pi/2,
                }[axis]
            xy = eh.rotPulse(q, 0, angle, phase)
            return q.where(xy=xy)
        return tuple(f(q, op) for q, op in zip(qubits, ops)), tm

    with pyle.QubitSequencer() as server:
        meas = measurement.TomoNull(3)
        qs, tm = op_sequence(ops)
        data = meas(server, qs, tm, stats=stats)

    import time

    print 'computing linear tomography...',
    start = time.time()
    rho0 = werner.doTomo(data, 'tomo3')
    end = time.time()
    print 'done. elapsed =', end - start

    print 'computing maximum-likelihood tomography...'
    pms = data.reshape((-1, 8))
    pxms = np.array([np.dot(werner.sierp(3), p) for p in pms])
    f0s = [0.9487, 0.9526, 0.9479]
    f1s = [0.9224, 0.9511, 0.9557]
    F = werner.Fxtf(3, f0s, f1s)
    Us = pyle.tomo._qst_transforms['tomo3'][0]

    start = time.time()
    rho = pyle.tomo.qst_mle(pxms[:,:-1], Us, F[:-1,:], rho0)
    end = time.time()
    print 'done. elapsed =', end - start
    print

    psi0 = np.array([1,0,0,0,0,0,0,0], dtype=complex)
    U = op_matrix(ops)
    psi_th = np.dot(U, psi0)
    rho_th = pyle.ket2rho(psi_th)

    print 'theory:'
    print rho_th.real
    print rho_th.imag
    print

    print 'experiment:'
    print rho.real
    print rho.imag
    print

    print 'plotting...',
    start = time.time()
    fig = werner.plotRho(rho, rho_th)
    fig.suptitle('maximum-likelihood tomography')

    #fig = werner.plotRho(rho0, rho_th)
    #fig.suptitle('linear tomography')
    end = time.time()
    print 'done. elapsed =', end - start
    print 'overlap:', pyle.fidelity(rho, rho_th)
    print 'G:', werner.ghz_G(rho)

    return rho, fig


def ghz_simult_trajectory(stages):
    rhos = []
    for stage in stages:
        U = np.eye(8, dtype=complex)
        I = pyle.tomo.sigmaI
        X = pyle.tomo.sigmaX
        Y = pyle.tomo.sigmaY
        couple = (pyle.tensor((X,X,I)) + pyle.tensor((Y,Y,I)) +
                  pyle.tensor((X,I,X)) + pyle.tensor((Y,I,Y)) +
                  pyle.tensor((I,X,X)) + pyle.tensor((I,Y,Y))) / 2.0

        if stage > 0:
            fraction = np.clip(stage-0, 0, 1)
            H = -1j * (np.pi/2)/2 * (pyle.tensor((Y,I,I)) + pyle.tensor((I,Y,I)) + pyle.tensor((I,I,Y)))
            U = np.dot(expm(fraction * H), U)

        if stage > 1:
            fraction = np.clip(stage-1, 0, 1)
            H = -1j * np.pi/2 * couple
            U = np.dot(expm(fraction * H), U)

        if stage > 2:
            fraction = np.clip(stage-2, 0, 1)
            H = -1j * (np.pi/2)/2 * (pyle.tensor((X,I,I)) + pyle.tensor((I,X,I)) + pyle.tensor((I,I,X)))
            U = np.dot(expm(fraction * H), U)

        psi0 = np.array([1,0,0,0,0,0,0,0], dtype=complex)
        psi_th = np.dot(U, psi0)
        rho_th = pyle.ket2rho(psi_th)
        rhos.append(rho_th)
    return np.array(rhos)



