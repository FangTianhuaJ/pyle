import itertools

import numpy as np

from labrad.units import Unit
ns = Unit('ns')

from pyle import envelopes as env
from pyle.pipeline import pmap, FutureList
from pyle.dataking import envelopehelpers as eh
from pyle.dataking.fpgaseq import runQubits
from pyle.util import sierpinv


def do(measure, server, qubits, *a, **kw):
    """Do the specified measurement or a default.

    If measure is a callable Measurer object, we just call it.
    Otherwise we fall back to the default simultaneous measurement,
    interpreting measure as a list of which qubits to measure.
    This is mainly provided for backward compatibility so that
    sweep functions can still accept ints or lists of ints for
    the measure parameter.
    """
    if callable(measure):
        return measure(server, qubits, *a, **kw)
    else:
        mfunc = Simult(len(qubits), measure)
        return mfunc(server, qubits, *a, **kw)


def simult(server, qubits, t, measure=None, correctCrosstalkZ=True, **kw):
    """Create a new sequence to measure some qubits at time t.

    This is the last operation that gets applied to the qubits, so
    we also (optionally) apply any crosstalk corrections and microwave
    mixing to the qubits before creating the sequence.

    This is the base upon which all of the other measurement schemes
    are built, since they all call this function in the end.
    """
    readouts = _makeReadoutList(len(qubits), measure)

    # create measure pulses for measured qubits
    pulses = [eh.measurePulse(q, t) if readout else env.NOTHING
              for q, readout in zip(qubits, readouts)]

    # optionally correct the measure pulses for Z-crosstalk
    if correctCrosstalkZ:
        def correct(pulses, coefs):
            return sum(float(c) * pulse for c, pulse in zip(coefs, pulses))
        pulses = [correct(pulses, q['calZpaXtalkInv']) for q in qubits]

    # add measure pulses to qubits and set readout flag
    qubits = [q.where(z=q.get('z', env.NOTHING)+pulse, readout=readout)
              for q, pulse, readout in zip(qubits, pulses, readouts)]

    return runQubits(server, qubits, **kw)


def null(server, qubits, t, measure=None, **kw):
    """Crosstalk-free measurement of some qubits using subset measurements.

    We perform a series of measurements of various subsets of qubits, discarding
    all probabilities but the null result when nothing switches.  From these
    probabilities, we use the 'sierpinski' matrix to retrieve the actual state
    occupation probabilities.
    """
    measure = _fixMeasureList(len(qubits), measure)
    n = len(measure)

    def measureNulls(state):
        nulls = [measure[i] for i in range(n) if state[i] == 0]
        return simult(server, qubits, t, nulls, probs=[0], **kw)
    states = list(itertools.product(range(2), repeat=n))[:-1] # drop 11..11 state
    req = FutureList([measureNulls(state) for state in states])
    def process(probs):
        return np.dot(sierpinv(n), [p[0] for p in probs] + [1])
    req.addCallback(process)
    return req


def nullpi(server, qubits, t, measure=None, **kw):
    """Crosstalk-free measurement using pi-pulses on selected qubits.

    We perform a series of measurements in which various qubits are flipped
    with a pi-pulse to map each state to the ground state before measuring.
    For each measurement, only the ground-state occupation probability is recorded.
    """
    measure = _fixMeasureList(len(qubits), measure)
    n = len(measure)

    dt = max(q['piLen'] for q in qubits)
    tp = t + dt
    tm = t + 2*dt
    def flipAndMeasure(state):
        def maybeFlip(i, q):
            if i in measure and state[measure.index(i)]:
                xy = q.get('xy', env.NOTHING) + eh.mix(q, eh.piPulse(q, tp))
                return q.where(xy=xy)
            else:
                return q
        flipped = [maybeFlip(i, q) for i, q in enumerate(qubits)]
        return simult(server, flipped, tm, measure=measure, probs=[0], **kw)
    states = list(itertools.product(range(2), repeat=n))
    req = FutureList([flipAndMeasure(state) for state in states])
    def process(probs):
        return np.array([p[0] for p in probs])
    req.addCallback(process)
    return req


def _fixMeasureList(N, measure):
    """Create a list of qubits to be measured out of N total.

    If measure is None, we assume all qubits are to be measured.
    If measure is an int, then only one qubit will be measured.
    The output list of indices for qubits to measure will always
    be sorted in ascending order.
    """
    if measure is None:
        measure = range(N)
    elif isinstance(measure, (int, long)):
        measure = [measure]
    return sorted(measure)


def _makeReadoutList(N, measure):
    measure = _fixMeasureList(N, measure)
    return [(i in measure) for i in range(N)]


# tomography rotations (name, angle, axis)
_identity = [
    ('I', 0, 0),
]

_tomoOps = [
    ('I', 0, 0),
    ('X', np.pi/2, 0),
    ('Y', np.pi/2, np.pi/2),
]

# octomography rotations (name, angle, axis)
_octomoOps = [
    ('I', 0, 0),
    ('+X', np.pi/2, 0),
    ('+Y', np.pi/2, np.pi/2),
    ('-X', -np.pi/2, 0),
    ('-Y', -np.pi/2, np.pi/2),
    ('pi', np.pi, 0)
]


# classes implementing different measurement protocols

class Measurer(object):
    """Represents a scheme for measuring one or more qubits.

    When applied to a set of qubit objects (as loaded from the registry,
    for example), the Measurer will return a new sequence with measure
    pulses included that is suitable for running with a pyq.QubitSequencer.
    Note that some more complicated measurement protocols need to do post-
    processing of the returned data, so they return generators that should
    be used with the functions in the sweeps module.

    This class also allows you to determine what quantities result from applying
    this measurement, so that datasets can be built properly, for example.
    In addition, it gives a dictionary of parameters describing the measurement
    that can also be added to datasets to give more complete information.
    """

    def __init__(self, N, measure=None, tBuf=0*ns):
        """Initialize for the given qubits where the indices in measure are to be measured."""
        if hasattr(N, '__len__'):
            N = len(N)
        self.N = N
        self.measure = _fixMeasureList(N, measure)
        self.n = len(self.measure)
        self.tBuf = tBuf

    def dependents(self):
        """Dependent variables for the data vault."""
        if self.n == 1:
            labels = ['|1>']
        else:
            labels = ['|%s>' % bin(i)[2:].rjust(self.n,'0') for i in xrange(2**self.n)]
        return [('Probability', s, '') for s in labels]

    def params(self):
        """Extra parameters for the data vault."""
        return {'measure': self.measure,
                'measureType': self.__class__.__name__}


class Simult(Measurer):
    def __call__(self, server, qubits, t, **kw):
        probs = [1] if self.n == 1 else None
        return simult(server, qubits, t + self.tBuf, self.measure, probs=probs, **kw)


class Null(Measurer):
    def __call__(self, server, qubits, t, **kw):
        if self.n == 1:
            return simult(server, qubits, t, self.measure, probs=[1], **kw)
        else:
            return null(server, qubits, t + self.tBuf, self.measure, **kw)


class NullPi(Measurer):
    def __call__(self, server, qubits, t, **kw):
        if self.n == 1:
            return simult(server, qubits, t + self.tBuf, self.measure, probs=[1], **kw)
        else:
            return nullpi(server, qubits, t + self.tBuf, self.measure, **kw)


class Tomo(Measurer):
    _ops = _tomoOps

    def dependents(self):
        opLabel = lambda ops: ','.join(op[0] for op in ops)
        stateLabel = lambda state: bin(state)[2:].rjust(self.n,'0')
        return [('Probability', opLabel(ops) + ',' + stateLabel(state), '')
                for ops in self.ops() for state in range(2**self.n)]

    def ops(self):
        rotations = [self._ops if i in self.measure else _identity for i in range(self.N)]
        return itertools.product(*rotations)

    def _measureFunc(self, *a, **kw):
        return simult(*a, **kw)

    def __call__(self, server, qubits, t, **kw):
        """Tomographic measurement."""

        dt = max(q['piLen'] for q in qubits)
        tp = t + dt/2.0
        tm = t + dt + self.tBuf

        def addTomoPulse(q, rot):
            """Add a tomography pulse to the microwave sequence for one qubit."""
            _name, angle, axis = rot
            phase = axis + q['tomoPhase'] + q.get('xy_phase', 0.0)
            pulse = eh.mix(q, eh.rotPulse(q, tp, angle=angle, phase=phase))
            xy = q.get('xy', env.NOTHING) + pulse
            return q.where(xy=xy)

        def func(op):
            """Add tomography pulses to each qubit and then measure."""
            print 'tomo:', ','.join(rot[0] for rot in op)
            rotated = [addTomoPulse(q, rot) for q, rot in zip(qubits, op)]
            return self._measureFunc(server, rotated, tm, self.measure, **kw)

        # do our own pipelining of tomography rotations
        return np.hstack(pmap(func, self.ops()))


class TomoNull(Tomo):
    def _measureFunc(self, *a, **kw):
        return null(*a, **kw)


class Octomo(Tomo):
    _ops = _octomoOps


class OctomoNull(Octomo):
    def _measureFunc(self, *a, **kw):
        return null(*a, **kw)

