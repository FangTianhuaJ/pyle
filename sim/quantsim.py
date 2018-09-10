from math import pi
import numpy as np
from scipy.linalg import expm

from pyle.math import ket2rho, tensor, commutator, lindblad, dot3
from pyle.sim import integrate


## Property helper for simulations
#
# this allows things like microwave drive or detuning to be set to a
# constant or a function of time, and then accessed as a function of time

def makeCallable(f):
    """Turn a value f into a function.
    
    If f is callable, it is returned unchanged.  Otherwise, we create a new
    function of time that will return the constant value f.
    """
    if callable(f):
        return f
    else:
        return lambda t: np.kron(np.ones_like(t), f) if isinstance(t, np.ndarray) else f


def callableProperty(name):
    """A property that can be assigned a function or value, but always returns a function"""
    mangled = '_' + name # the actual value will be stored under this new name
    def _get(self):
        return getattr(self, mangled)
    def _set(self, val):
        setattr(self, mangled, makeCallable(val))
    return property(_get, _set)


## Subsystems
# 
# subsystems must define:
# - n: the number of energy levels in the subsystem
# - I: identity matrix
# - Ad, A: raising (Ad) and lowering (A) operators
# - Ls: Lindblad operators (assumed independent of time)
#
# - H(t): the Hamiltonian as a function of time

class Qubit2(object):
    """An ideal Qubit with just two levels."""
    def __init__(self, T1, T2, df=0., uw=0.):
        self.T1 = float(T1)
        self.T2 = float(T2)
        
        self.n = 2 # number of levels
        self.I = np.eye(self.n)
        self.N = np.diag([0., 1.])  # number operator
        self.A = np.diag([1.], 1)   # lowering operator [[0,1],[0,0]]
        self.Ad = np.diag([1.], -1) # raising operator  [[0,0],[1,0]]
        
        # Lindblad operators
        L1 = np.sqrt(1./T1) * self.A
        L2 = np.sqrt(2./T2) * self.N
        self.Ls = [L1, L2]
        
        # control signals
        self.df = df
        self.uw = uw
    
    df = callableProperty('df')
    uw = callableProperty('uw')
    
    def H(self, t):
        a = 2*pi*self.uw(t)/2.0
        ad = np.conj(a)
        dw = 2*pi*self.df(t)
        #return a*self.Ad + ad*self.A + dw*self.N
        return np.array([[0, ad],
                         [a, dw]])


class Qubit3(object):
    """A Qubits with 3 levels and finite anharmonicity"""
    def __init__(self, T1, T2, nonlin=-0.2, lam=np.sqrt(2), df=0., uw=0.):
        self.T1 = float(T1)
        self.T2 = float(T2)
        self.nonlin = float(nonlin)
        self.lam = float(lam)
        
        self.n = 3 # number of levels
        self.I = np.eye(self.n)
        self.N = np.diag([0., 1., 2.]) # number operator
        self.A = np.diag([1., lam], 1) # lowering operator
        self.Ad = np.diag([1., lam], -1) # raising operator
        self.P2 = np.diag([0., 0., 1.]) # projector onto the 2-state
        
        # Lindblad operators
        L1 = np.sqrt(1./T1) * self.A
        L2 = np.sqrt(2./T2) * self.N
        self.Ls = [L1, L2]
        
        # control signals
        self.df = df
        self.uw = uw
    
    df = callableProperty('df')
    uw = callableProperty('uw')
    
    def H(self, t):
        a = 2*pi*self.uw(t)/2.0
        ad = np.conj(a)
        lam = self.lam
        dw = 2*pi*self.df(t)
        dw2 = 2*pi*self.nonlin
        #return a*self.Ad + ad*self.A + dw*self.N + dw2*self.P2
        return np.array([[0,    ad,        0],
                         [a,    dw,   lam*ad],
                         [0, lam*a, 2*dw+dw2]])


class Resonator(object):
    """A linear resonator, truncated to the specified number of levels"""
    def __init__(self, T1, T2, n=5, df=0., uw=0.):
        self.T1 = float(T1)
        self.T2 = float(T2)
        self.df = float(df)

        self.n = n
        self.I = np.eye(n)
        self.N = np.diag(range(n)) # number operator
        off_diags = [np.sqrt(i) for i in range(1,n)]
        self.A = np.diag(off_diags, 1) # lowering operator
        self.Ad = np.diag(off_diags, -1) # raising operator
        
        # Lindblad operators
        L1 = np.sqrt(1./T1) * self.A
        L2 = np.sqrt(2./T2) * self.N
        self.Ls = [L1, L2]
        
        self.uw = uw
    
    uw = callableProperty('uw')
    
    def H(self, t):
        a = 2*pi*self.uw(t)/2.0
        ad = np.conj(a)
        dw = 2*pi*self.df
        return a*self.Ad + ad*self.A + dw*self.N


## Couplers
#
# couplers must define
# - Cs: a list of coupling operators of the form {q1: A, q2: B,...}
# - s(t): the splitting (in GHz) as a function of time

class Coupler(object):
    """Simple transverse (swap) coupling.
    
    For qubits, this is the usual XX + YY coupling.  Here though, we instead
    write this in terms of raising and lowering operators so that it is applicable
    also to multilevel qubits or to qubit + resonator. 
    """
    def __init__(self, q0, q1, s=0):
        self.s = s
        self.Cs = [{q0: q0.A, q1: q1.Ad},
                   {q0: q0.Ad, q1: q1.A}]
    s = callableProperty('s')
    

## Quantum System
#
# has a few important attributes
# - m: number of component subsystems (e.g. qubits)
# - n: number of levels = size of hilbert space (product of n for each subsystem)

class QuantumSystem(object):
    """A set of qubits and couplers.
    
    From the parts, we can compute the full system Hamiltonian and
    decoherence operators and thus simulate the system evolution.
    """
    def __init__(self, qubits, couplers=[]):
        self.qubits = qubits
        self.couplers = couplers
        
        # number of subsystems
        self.m = len(qubits)
        
        # shape of hilbert space
        self.shape = tuple(q.n for q in qubits)
        
        # size of hilbert space
        self.n = 1
        for q in qubits:
            self.n *= q.n
        
        # precompute coupling matrices
        #
        # we compute one matrix for each coupler, consisting of a sum of terms
        # contributed by that coupler to the Hamiltonian.  Each such term is a tensor
        # product of single-qubit terms with most of them being the identity, but
        # two or more being operators on various qubits (e.g. A_i x Ad_j)
        # Later, when we actually compute the Hamiltonian, we multiply each matrix
        # by the coupling strength, which may be a function of time, so we store
        # a list of pairs of couplers (to get the coupling strength) and coupling matrices.
        Cs = [sum(tensor(C.get(q, q.I) for q in qubits) for C in coupler.Cs)
              for coupler in couplers]
        self.Cs = zip(couplers, Cs)
        
        # precompute Lindblad terms
        #
        # each term is a tensor product of identity on all qubits but one, and each
        # qubit can contribute multiple lindblad terms.  We store a list of pairs of
        # Lindblad terms and their complex conjugates.
        Ls = [tensor(L if q == qubit else q.I for q in qubits)
              for qubit in qubits for L in qubit.Ls]
        self.Ls = [(L, L.conj().T) for L in Ls]
    
    def H(self, t):
        """Calculate the hamiltonian as a function of time."""
        
        # qubits
        #
        # each term is a tensor product of the Hamiltonian of one qubit
        # and identities of all the other qubits, with the Hamiltonian
        # evaluated at the current time, of course.
        Hq = sum(tensor(q.H(t) if q == qubit else q.I for q in self.qubits)
                 for qubit in self.qubits)
        
        # couplers
        #
        # here C is the coupling matrix (precomputed above), s(t) is the
        # splitting in GHz evaluated at the current time, and the factor
        # of pi converts to radians, as usually defined by the coupling
        # strength g.  Note that the factor is pi, not 2*pi, because the
        # splitting is by _probability_ oscillation frequency, and so is
        # equal to 2 * g/(2*pi)
        Hc = sum(pi*coupler.s(t) * C for coupler, C in self.Cs)
        
        # full Hamiltonian
        return Hq + Hc
        
    def dlindblad(self, rho):
        """Lindblad contribution to the derivative for density matrix rho."""
        return sum(lindblad(rho, L, Ld) for L, Ld in self.Ls)
    
    def dfunc(self, rho, t):
        """Calculate derivative of rho as a function of time."""
        return -1j * commutator(self.H(t), rho) + self.dlindblad(rho)
    
    def propagate(self, rho, t, dt):
        """Propagate rho forward in time from t to t+dt for fast (but less accurate) integration."""
        U = expm(-1j*dt*self.H(t))
        return dot3(U, rho, U.conj().T) + self.dlindblad(rho) * dt
    
    def ket(self, state):
        """Create a ket corresponding to a particular basis state.

        This ket is a vector of zeros with a single one in the
        position corresponding to the desired basis state.        
        """
        psi = np.zeros(self.shape)
        indices = tuple(int(digit) for digit in state)
        psi[indices] = 1
        return psi.flatten()
        
        psi = np.zeros(self.n, dtype=complex)
        psi[self.index(state)] = 1
        return psi
        
    def rho(self, state):
        return ket2rho(self.ket(state))
        
    def index(self, state):
        """Calculate the index of a particular state in our hilbert space (or density matrix)
        
        This is complicated by the fact that the various subsystems do not necessarily have
        the same size.  States are ordered lexically, e.g. 000, 001, 002, 010, 011, 012,...
        where each digit runs from zero up to the size of that particular subsystem.
        """
        idx = 0
        mult = 1
        for q, digit in reversed(zip(self.qubits, state)):
            idx += int(digit) * mult
            mult *= q.n
        return idx
        
    def simulate(self, rho0, T, method='fast', **opts):
        """Simulate the evolution of this quantum system, starting with
        the given state and evolving for the specified time.
        """
        # if we got a pure state input, convert to density matrix
        rho0 = np.asarray(rho0)
        if len(rho0.shape) == 1:
            rho0 = np.outer(rho0.conj(), rho0)

        # integrate the master equation and save the result
        if method == 'fast':
            # use fast propagation
            rho = rho0
            rhos = [rho]
            for ti, tf in zip(T[:-1], T[1:]):
                rho = self.propagate(rho, ti, tf-ti)
                rhos.append(rho)
            self.rhos = np.array(rhos)
        else:
            # use rhodeint
            self.rhos, self.ode_info = integrate.rhodeint(self.dfunc, rho0, T, full_output=True, **opts)
        return self.rhos
        
    def partial(self, subsystems, rhos=None):
        """Take partial trace of the density matrix (or matrices) rho.
        
        This returns the reduced density matrix of just the
        indicated subsystems, tracing over all other subsystems.
        Subsystems are specified by their indices into the array of
        subsystems (e.g. qubits) passed in when this quantum system
        was created.
        
        Note that the reduced density matrix preserves the original
        order of the subsystems, regardless of the order passed in to
        this method, that is, partial(rho, (2, 1)) actually gives
        the same thing as partial(rho, (1, 2)).
        """
        if rhos is None:
            rhos = self.rhos
        rhos = np.asarray(rhos)
        preshape = rhos.shape[:-2]
        rhos = rhos.reshape(preshape + self.shape + self.shape)
        
        # subsystems can be passed in as objects, if desired
        if not np.iterable(subsystems):
            subsystems = [subsystems]
        subsystems = list(subsystems) # copy the input list
        for i, q in enumerate(subsystems):
            if q in self.qubits:
                subsystems[i] = self.qubits.index(q)

        # trace over all axes
        n = self.n
        m = self.m
        ofs = len(preshape)
        trace_axes = [i for i in range(m) if i not in subsystems]
        for i in reversed(trace_axes):
            rhos = np.trace(rhos, axis1=ofs+i, axis2=ofs+i+m)
            n /= self.qubits[i].n
            m -= 1
        
        return rhos.reshape(preshape + (n, n))

    @property
    def probs(self):
        """Get probabilities (diagonal elements of density matrix) from last simulation run."""
        i = range(self.n)
        return self.rhos[:,i,i].real

    def partial_probs(self, subsystems, rhos=None):
        """Get probabilities after taking a partial trace."""
        rhos = self.partial(subsystems, rhos)
        preshape = rhos.shape[:-2]
        
        i = range(rhos.shape[-1])
        if len(preshape) == 0:
            return rhos[i,i].real
        else:
            return rhos[:,i,i].real


