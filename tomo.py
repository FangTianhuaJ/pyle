import math # local module, not stdlib
import itertools

import numpy as np
from scipy import optimize
from scipy.linalg import expm

from pyle.math import tensor, dot3


# define some useful matrices
def Rmat(axis, angle):
    return expm(-1j*angle/2.0*axis)

sigmaI = np.eye(2, dtype=complex)
sigmaX = np.array([[0, 1], [1, 0]], dtype=complex)
sigmaY = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigmaZ = np.array([[1, 0], [0, -1]], dtype=complex)

sigmaP = (sigmaX - 1j*sigmaY)/2
sigmaM = (sigmaX + 1j*sigmaY)/2

Xpi2 = Rmat(sigmaX, np.pi/2)
Ypi2 = Rmat(sigmaY, np.pi/2)
Zpi2 = Rmat(sigmaZ, np.pi/2)

Xpi = Rmat(sigmaX, np.pi)
Ypi = Rmat(sigmaY, np.pi)
Zpi = Rmat(sigmaZ, np.pi)

Xmpi2 = Rmat(sigmaX, -np.pi/2)
Ympi2 = Rmat(sigmaY, -np.pi/2)
Zmpi2 = Rmat(sigmaZ, -np.pi/2)

Xmpi = Rmat(sigmaX, -np.pi)
Ympi = Rmat(sigmaY, -np.pi)
Zmpi = Rmat(sigmaZ, -np.pi)


# 3-level operators
sigmaI3=np.eye(3,dtype=np.complex)
I3 = np.eye(3, dtype=np.complex)
sigmaX01 = np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=complex)
sigmaX12 = np.array([[0,0,0],[0,0,1],[0,1,0]], dtype=complex)
sigmaY01 = np.array([[0,-1j,0],[1j,0,0],[0,0,0]], dtype=complex)
sigmaY12 = np.array([[0,0,0],[0,0,-1j],[0,1j,0]], dtype=complex)

Xpi01 = Rmat(sigmaX01, np.pi)
Xpi12 = Rmat(sigmaX12, np.pi)
Ypi01 = Rmat(sigmaY01, np.pi)
Ypi12 = Rmat(sigmaY12, np.pi)

Xpi2_01 = Rmat(sigmaX01, np.pi/2)
Ypi2_01 = Rmat(sigmaY01, np.pi/2)
Xpi2_12 = Rmat(sigmaX12, np.pi/2)
Ypi2_12 = Rmat(sigmaY12, np.pi/2)

Xmpi2_01 = Rmat(sigmaX01, -np.pi/2)
Ympi2_01 = Rmat(sigmaY01, -np.pi/2)
Xmpi2_12 = Rmat(sigmaX12, -np.pi/2)
Ympi2_12 = Rmat(sigmaY12, -np.pi/2)

# tomo operations for 3-level operators
r0 = sigmaI3
r1 = Xpi2_01
r2 = Ypi2_01
r3 = Xpi01
r4 = Xpi2_12
r5 = Ypi2_12
r6 = np.dot(Xpi2_12, Xpi01)
r7 = np.dot(Ypi2_12, Xpi01)
r8 = np.dot(Xpi12, Xpi01)
tomo3_ops0 = [r0, r1, r2, r3, r4, r5, r6, r7, r8]

# octomo operations for 3-level operators
R0 = sigmaI3
R1 = Xpi2_01
R2 = Ypi2_01
R3 = Xmpi2_01
R4 = Ympi2_01
R5 = np.dot(Xpi12, Xpi2_01)
R6 = np.dot(Xpi12, Ypi2_01)
R7 = np.dot(Xpi12, Xmpi2_01)
R8 = np.dot(Xpi12, Ympi2_01)
R9 = Xpi01
R10 = Xpi2_12
R11 = Ypi2_12
R12 = Xmpi2_12
R13 = Ympi2_12
R14 = Xpi12
R15 = np.dot(Xpi2_12, Xpi01)
R16 = np.dot(Ypi2_12, Xpi01)
R17 = np.dot(Xmpi2_12, Xpi01)
R18 = np.dot(Ympi2_12, Xpi01)
R19 = np.dot(Xpi12, Xpi01)
octomo3_ops0 = [R0, R1, R2, R3, R4, R5, R6, R7, R8,
                R9, R10, R11, R12, R13, R14, R15,
                R16, R17, R18, R19]

# gell-mann basis
gellmann_basis = [
    np.eye(3, dtype=np.complex),
    np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=np.complex),
    np.array([[0,-1j,0],[1j,0,0],[0,0,0]], dtype=np.complex),
    np.array([[1,0,0],[0,-1,0],[0,0,0]], dtype=np.complex),
    np.array([[0,0,1],[0,0,0],[1,0,0]], dtype=np.complex),
    np.array([[0,0,-1j],[0,0,0],[1j,0,0]], dtype=np.complex),
    np.array([[0,0,0],[0,0,1],[1,0,0]], dtype=np.complex),
    np.array([[0,0,0],[0,0,-1j],[0,1j,0]], dtype=np.complex),
    np.array([[1,0,0],[0,1,0],[0,0,-2]], dtype=np.complex)/np.sqrt(3)
]
sigma3_basis = [
    np.array([[1,0,0],[0,0,0],[0,0,1]], dtype=np.complex), # I02
    np.array([[0,0,1],[0,0,0],[1,0,0]], dtype=np.complex), # sigmaX02
    np.array([[0,0,-1j],[0,0,0],[1j,0,0]], dtype=np.complex), # sigmaY02
    np.array([[1,0,0],[0,0,0],[0,0,-1]], dtype=np.complex), # sigmaZ02
    np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=np.complex), # sigmaX01
    np.array([[0,-1j,0],[1j,0,0],[0,0,0]], dtype=np.complex), # sigmaY01
    np.array([[0,0,0],[0,0,1],[0,1,0]], dtype=np.complex), # sigmaX12
    np.array([[0,0,0],[0,0,-1j],[0,1j,0]], dtype=np.complex), # sigmaY12
    np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.complex), # |1><1|
]

# store all initialized tomography protocols
_qst_transforms = {}
_qpt_transforms = {}


def init_qst(Us, key=None):
    """Initialize quantum state tomography for a set of unitaries.

    Us - a list of unitary operations that will be applied to the
        state before measuring the diagonal elements.  These unitaries
        should form a 'complete' set to allow the full density matrix
        to be determined, though this is not enforced.

    key - (optional) a dictionary key under which this tomography
        protocol will be stored so it can be referred to without
        recomputing the transformation matrix.

    Returns a transformation matrix that should be passed to qst along
    with measurement data to perform the state tomography.
    """

    Us = np.asarray(Us)

    #Number of different rotations applied to the system prior
    #to measurement
    numRotations = len(Us)

    #Number of states. Normally this will be 2**n where n is the number
    #of qubits measured in the experiment. More generally this is the
    #number of diagonal elements that you measured.
    #For example, imagine that for a given set of rotations on a three qubit
    #system, we measure the following probabilities:
    #|ggg>, |geg>, |egg>, |eeg>
    #then numStates=4.
    #Note that in this example the measurements DO NOT completely determine
    #the three qubit density matrix, but they do determine the density
    #matrix for the subsystem consisting of the first two qubits.
    numStates = len(Us[0])

    # we have to be a bit careful here, because things blow up
    # exponentially with the number of qubits.  The first method
    # uses direct indexing to generate the entire transform matrix
    # in one shot.  This is elegant and much faster than for-loop
    # iteration, but uses more memory and so only works for
    # smaller qubit numbers.
    if numStates <= 16:
        # 1-4 qubits
        # For explanation of this code see Matthew Neeley's thesis
        # page 144-146.
        # j = which rotation (U)
        # k = which state (diagonal element)
        def transform(K, L):
            j, k = divmod(K, numStates)
            m, n = divmod(L, numStates)
            return Us[j, k, m] * Us[j, k, n].conj()
        U = np.fromfunction(transform, (numRotations*numStates, numStates**2), dtype=int)
    else:
        # 5+ qubits
        U = np.zeros((numRotations*numStates, numStates**2), dtype=complex)
        for K in range(numRotations*numStates):
            for L in range(numStates**2):
                j, k = divmod(K, numStates)
                m, n = divmod(L, numStates)
                U[K, L] = Us[j, k, m] * Us[j, k, n].conj()

    # save this transform if a key was provided
    if key is not None:
        _qst_transforms[key] = (Us, U)

    return U


def init_qpt(As, key=None):
    """Initialize quantum process tomography for an operator basis.

    As - a list of matrices giving the basis in which to compute
        the chi matrix for process tomography.  These matrices
        should form a 'complete' set to allow the full chi matrix
        to be represented, though this is not enforced.

    key - (optional) a dictionary key under which this tomography
        protocol will be stored so it can be referred to without
        recomputing the transformation matrix.

    Returns a transformation matrix that should be passed to qpt along
    with input and output density matrices to perform the process tomography.
    """

    As = np.asarray(As, dtype=complex)

    Dout, Din = As[0].shape
    chiSize = Dout*Din

    # we have to be a bit careful here, because things blow up
    # exponentially with the number of qubits.  The first method
    # uses direct indexing to generate the entire transform matrix
    # in one shot.  This is elegant and much faster than for-loop
    # iteration, but uses more memory and so only works for
    # smaller qubit numbers.
    if chiSize <= 16:
        # one or two qubits
        def transform(alpha, beta):
            L, J = divmod(alpha, chiSize)
            M, N = divmod(beta, chiSize)
            i, j = divmod(J, Dout)
            k, l = divmod(L, Din)
            return As[M, i, k] * As[N, j, l].conj()
        T = np.fromfunction(transform, (chiSize**2, chiSize**2), dtype=int)
    else:
        # three or more qubits
        T = np.zeros((chiSize**2, chiSize**2), dtype=complex)
        for alpha in range(chiSize**2):
            for beta in range(chiSize**2):
                L, J = divmod(alpha, chiSize)
                M, N = divmod(beta, chiSize)
                i, j = divmod(J, Dout)
                k, l = divmod(L, Din)
                T[alpha, beta] = As[M, i, k] * As[N, j, l].conj()

    if key is not None:
        _qpt_transforms[key] = (As, T)

    return T


def qst(diags, U, return_all=False):
    """Convert a set of diagonal measurements into a density matrix.

    diags - measured probabilities (diagonal elements) after acting
        on the state with each of the unitaries from the qst protocol

    U - transformation matrix from init_qst for this protocol, or
        key passed to init_qst under which the transformation was saved
    """
    if isinstance(U, str) and U in _qst_transforms:
        U = _qst_transforms[U][1]

    diags = np.asarray(diags)
    N = diags.shape[1]
    rhoFlat, resids, rank, s = np.linalg.lstsq(U, diags.flatten())
    if return_all:
        return rhoFlat.reshape((N, N)), resids, rank, s
    else:
        return rhoFlat.reshape((N, N))


def qst_mle(pxms, Us, F=None, rho0=None):
    """State tomography with maximum-likelihood estimation.

    pxms - a 2D array of measured probabilites.  The first index indicates which
           operation from Us was applied, while the second index tells which measurement
           result this was (e.g. 000, 001, etc.).

    Us - the unitary operations that were applied to the system before measuring.
    F - a 'fidelity' matrix, relating the actual or 'intrinsic' probabilities to the
        measured probabilites, via pms = dot(F, pis).  If no fidelity matrix is given,
        the identity will be used.
    rho0 - an initial guess for the density matrix, e.g. from linear tomography.
    """
    if isinstance(Us, str) and Us in _qst_transforms:
        Us = _qst_transforms[Us][0]
    N = len(Us[0]) # size of density matrix

    if F is None:
        F = np.eye(N)

    try:
        indices_re = np.tril_indices(N)
        indices_im = np.tril_indices(N, -1)
    except AttributeError:
        # tril_indices is new in numpy 1.4.0
        indices_re = (np.hstack([[k]*(k+1) for k in range(N)]),
                      np.hstack([range(k+1) for k in range(N)]))
        indices_im = (np.hstack([[k+1]*(k+1) for k in range(N-1)]),
                      np.hstack([range(k+1) for k in range(N-1)]))
    n_re = len(indices_re[0]) # N*(N+1)/2
    n_im = len(indices_im[0]) # N*(N-1)/2

    def make_T(tis):
        T = np.zeros((N,N), dtype=complex)
        T[indices_re] = tis[:n_re]
        T[indices_im] += 1j*tis[n_re:]
        return T

    def unmake_T(T):
        return np.hstack((T[indices_re].real, T[indices_im].imag))

    def make_rho(ts):
        T = make_T(ts)
        TT = np.dot(T, T.conj().T)
        return TT / np.trace(TT)

    def rho2T(rho):
        d, V = np.linalg.eigh(rho)
        d = d.real
        d[d<.01] =  0.01
        d /= np.sum(d)
        rho0 = dot3(V, np.diag(d), V.conj().T)
        T0 = np.linalg.cholesky(rho0)
        tis_guess = unmake_T(T0)
        return tis_guess

    # make an initial guess using linear tomography
    if rho0 is None:
        T = init_qst(Us)
        Finv = np.linalg.inv(F)
        pis_guess = np.array([np.dot(Finv, p) for p in pxms])
        rho0 = qst(pis_guess, T)

    # convert the initial guess into t vector
    # to do this we use a cholesky decomposition, which
    # only works if the matrix is positive and hermitian.
    # so, we diagonalize and fix up the eigenvalues before
    # attempting the cholesky decomp.
    tis_guess = rho2T(rho0)

    # precompute conjugate transposes of matrices
    # UUds = [(U, U.conj().T) for U in Us]

    Utensor = np.array([U for U in Us])
    Udtensor = np.array([U.conj().T for U in Us])

    def log(x):
        """Safe version of log that returns -Inf when x < 0, rather than NaN.

        This is good for our purposes since negative probabilities are infinitely unlikely.
        """
        # return np.log(x.real * (x.real > 0))
        return np.log(np.maximum(x.real, 1e-100))

    array = np.array
    dot = np.dot
    diag = np.diag

    def unlikelihood(tis): # negative of likelihood function
        rho = make_rho(tis)
        # pxis = array([dot(F, diag(dot3(U, rho, Ud))) for U, Ud in UUds])
        # using einsum, faster than dot(F, diag(...)
        pxis = np.dot(np.einsum('aij,jk,aki->ai', Utensor, rho, Udtensor).real, F.T)
        terms = pxms * log(pxis) + (1-pxms) * log(1-pxis)
        # terms = pxms * log(pxis)
        # I comment the second term, It sum(pxms) of each measure is 1,
        # the second term is the same with the first after sum.
        # In this scheme, data should not be corrected, and F should be inputed.
        # Also, 3-level tomo_mle can be supported. --ZZX, 2018.05.12
        return -np.mean(terms.flat)

    #minimize
    tis = optimize.fmin(unlikelihood, tis_guess)
    #tis = optimize.fmin_bfgs(unlikelihood, tis_guess)
    return make_rho(tis)


def qpt(rhos, Erhos, T, return_all=False):
    """Calculate the chi matrix of a quantum process.

    rhos - array of input density matrices
    Erhos - array of output density matrices

    T - transformation matrix from init_qpt for the desired operator
        basis, or key passed to init_qpt under which this basis was saved
    """
    chi_pointer = qpt_pointer(rhos, Erhos)
    return transform_chi_pointer(chi_pointer, T, return_all)


def transform_chi_pointer(chi_pointer, T, return_all=False):
    """Convert a chi matrix from the pointer basis into a different basis.

    transform_chi_pointer(chi_pointer, As) will transform the chi_pointer matrix
    from the pointer basis (as produced by qpt_pointer, for example) into the
    basis specified by operator elements in the cell array As.
    """
    if T in _qpt_transforms:
        T = _qpt_transforms[T][1]

    _Din, Dout = chi_pointer.shape
    chi_flat, resids, rank, s = np.linalg.lstsq(T, chi_pointer.flatten())
    chi = chi_flat.reshape((Dout, Dout))
    if return_all:
        return chi, resids, rank, s
    else:
        return chi


def qpt_pointer(rhos, Erhos, return_all=False):
    """Calculates the pointer-basis chi-matrix.

    rhos - array of input density matrices
    Erhos - array of output density matrices.

    Uses linalg.lstsq to calculate the closest fit
    when the chi-matrix is overdetermined by the data.
    The return_all flag specifies whether to return
    all the parameters returned from linalg.lstsq, such
    as the residuals and the rank of the chi-matrix.  By
    default (return_all=False) only chi is returned.
    """

    # the input and output density matrices can have different
    # dimensions, although this will rarely be the case for us.
    Din = rhos[0].size
    Dout = Erhos[0].size
    n = len(rhos)

    # reshape the input and output density matrices
    # each row of the resulting matrix has a flattened
    # density matrix (in or out, respectively)
    rhos_mat = np.asarray(rhos).reshape((n, Din))
    Erhos_mat = np.asarray(Erhos).reshape((n, Dout))

    chi, resids, rank, s = np.linalg.lstsq(rhos_mat, Erhos_mat)
    if return_all:
        return chi, resids, rank, s
    else:
        return chi


def tensor_combinations(matrices, repeat):
    return [tensor(ms) for ms in itertools.product(matrices, repeat=repeat)]


def tensor_combinations_phases(matrices, repeat, phases):
    products = itertools.product(matrices, repeat=repeat)
    tensorProducts = []
    for ms in products:
        tensorProduct = []
        for num,qubitMatrix in enumerate(ms):
            tensorProduct.append(dot3(Rmat(sigmaZ,-phases[num]),qubitMatrix,Rmat(sigmaZ,phases[num])))
        tensorProducts.append(tensor(tensorProduct))
    return tensorProducts


def gen_qst_tomo_ops(tomoNames, repeat):
    return itertools.product(tomoNames, repeat=repeat)

gen_qptPrep_tomo_ops = gen_qst_tomo_ops
gen_qptPost_tomo_ops = gen_qst_tomo_ops

def gen_qpt_tomo_ops(tomoNames, repeat):
    names = itertools.product(tomoNames, repeat=repeat)
    names = [n for n in names]
    for n in names:
        for m in names:
            yield n, m


def parse_qpt_probs(probs, tomoNames, qst_name):
    # This guy takes the tomo names, and turns all of our probs into
    # rhos. We have to take the probs, and go in sets of how many it takes
    # to get a density matrix out. If N is the # of tomo rotations, for
    # 1 qubit its sets of N**2 probs, for 2 qubits its N**4, so N**(2*#) of qubits.
    # We take this list of probs and parse it into groupings of N**#, as each
    # set of those constitutes a density matrix rho. Then after we go through
    # all of the groupings, we have all of our rhos.

    # len(probs) = N**(2*#)

    # probs (list) is formatted s.t. probs[0] returns [0,1] probabilities for 1q
    # or [00, 01, 10, 11] probs for 2q etc.

    if isinstance(probs, np.ndarray):
        # if this is true, then we have the raw data coming out
        # of the data vault formatted as
        # array([[tomoIndex, prob0, prob1],[tomoIndex,prob0,prob1],...])
        probs = [prob[1:] for prob in probs]

    numTomo = len(tomoNames)
    numQubits = int(np.log(len(probs)) / np.log(numTomo) / 2)

    opsPerRho = numTomo ** numQubits

    rhos = []
    while len(probs) > 0:
        currProbs = [probs.pop(0) for k in range(opsPerRho)]

        rho = qst(currProbs, qst_name)

        rhos.append(rho)

    return rhos


def gen_ideal_chi_matrix(unitary, qpt_name, tomo_ops):
    numQubits = int(np.log2(len(unitary)))

    mainDiag = [1] + (2 ** numQubits - 1) * [0]
    rho0 = np.diag(mainDiag)

    inRhos = []
    outRhos = []

    operations = tensor_combinations(tomo_ops, repeat=numQubits)

    for op in operations:
        inRho = dot3(op, rho0, op.conjugate().transpose())
        outRho = dot3(unitary, inRho, unitary.conjugate().transpose())

        inRhos.append(inRho)
        outRhos.append(outRho)

    chi = qpt(inRhos, outRhos, qpt_name)

    return chi, inRhos, outRhos


# standard single-qubit QST protocols

tomo_ops = [np.eye(2), Xpi2, Ypi2]
tomo_names = ['I', 'X/2', 'Y/2']
octomo_ops = [np.eye(2), Xpi2, Ypi2, Xmpi2, Ympi2, Xpi]
octomo_names = ['I', 'X/2', 'Y/2', '-X/2', '-Y/2', 'X']

# 3-level tomo name
tomo3l_ops = tomo3_ops0
tomo3l_names = [("I", "I"), ("X/2", "I"), ("Y/2", "I"), ("X", "I"),
               ("I", "X/2"), ("I", "Y/2"), ("X", "X/2"), ("X", "Y/2"), ("X", "X")]

octomo3l_ops = octomo3_ops0
octomo3l_names = [("I", "I"),
                  ("X/2", "I"), ("Y/2", "I"), ("-X/2", "I"), ("-Y/2", "I"),
                  ("X/2", "X"), ("Y/2", "X"), ("-X/2", "X"), ("-Y/2", "X"),
                  ("X", "I"),
                  ("I", "X/2"), ("I", "Y/2"), ("I", "-X/2"), ("I", "-Y/2"),
                  ("I", "X"),
                  ("X", "X/2"), ("X", "Y/2"), ("X", "-X/2"), ("X", "-Y/2"),
                  ("X", "X")]

init_qst(tomo_ops, 'tomo')
init_qst(octomo_ops, 'octomo')

init_qst(tensor_combinations(tomo_ops, 2), 'tomo2')
init_qst(tensor_combinations(octomo_ops, 2), 'octomo2')

init_qst(tensor_combinations(tomo_ops, 3), 'tomo3')
init_qst(tensor_combinations(octomo_ops, 3), 'octomo3')

init_qst(tensor_combinations(tomo_ops, 4), 'tomo4')
init_qst(tensor_combinations(octomo_ops, 4), 'octomo4')

#init_qst([tensor(ops) for ops in itertools.product(tomo_ops, repeat=4)], 'tomo4')
#init_qst([tensor(ops) for ops in itertools.product(octomo_ops, repeat=4)], 'octomo4')

# 3-level
init_qst(tomo3l_ops, 'tomo3l')
init_qst(octomo3l_ops, 'octomo3l')

# standard QPT protocols

sigma_basis = [np.eye(2), sigmaX, sigmaY, sigmaZ]
raise_lower_basis = [np.eye(2), sigmaP, sigmaM, sigmaZ]

init_qpt(sigma_basis, 'sigma')
init_qpt(raise_lower_basis, 'raise-lower')

init_qpt(tensor_combinations(sigma_basis, 2), 'sigma2')
init_qpt(tensor_combinations(raise_lower_basis, 2), 'raise-lower2')

# takes A LOT of memory!
#init_qpt(tensor_combinations(sigma_basis, 3), 'sigma3')
#init_qpt(tensor_combinations(raise_lower_basis, 3), 'raise-lower3')

init_qpt(gellmann_basis, key='gellmann')
init_qpt(sigma3_basis, key='3l-sigma')

## tests

def test_qst(n=100):
    """Generate a bunch of random states, and check that
    we recover them from state tomography.
    """

    def test_qst_protocol(proto):
        Us = _qst_transforms[proto][0]
        rho = (np.random.uniform(-1, 1, Us[0].shape) +
            1j*np.random.uniform(-1, 1, Us[0].shape))
        diags = np.vstack(np.diag(dot3(U, rho, U.conj().T)) for U in Us)
        rhoCalc = qst(diags, proto)
        return np.max(np.abs(rho - rhoCalc))

    # 1 qubit
    et1 = [test_qst_protocol('tomo') for _ in range(n)]
    eo1 = [test_qst_protocol('octomo') for _ in range(n)]
    print '1 qubit max error: tomo=%g, octomo=%g' % (max(et1), max(eo1))

    # 2 qubits
    et2 = [test_qst_protocol('tomo2') for _ in range(n/2)]
    eo2 = [test_qst_protocol('octomo2') for _ in range(n/2)]
    print '2 qubits max error: tomo2=%g, octomo2=%g' % (max(et2), max(eo2))

    # 3 qubits
    et3 = [test_qst_protocol('tomo3') for _ in range(n/10)]
    eo3 = [test_qst_protocol('octomo3') for _ in range(n/10)]
    print '3 qubits max error: tomo3=%g, octomo3=%g' % (max(et3), max(eo3))

    # 4 qubits
    #et4 = [testQstProtocol('tomo4') for _ in range(2)]
    #eo4 = [testQstProtocol('octomo4') for _ in range(2)]
    #print '4 qubits max error: tomo4=%g, octomo4=%g' % (max(et4), max(eo4))


def test_qpt(n=1):
    """Generate a random chi matrix, and check that we
    recover it from process tomography.
    """
    def operate(rho, chi, As):
        return sum(chi[m, n] * dot3(As[m], rho, As[n].conj().T)
                   for m in range(len(As)) for n in range(len(As)))

    def test_qpt_protocol(proto):
        As = _qpt_transforms[proto][0]
        s = As.shape[1]
        N = len(As)
        chi = (np.random.uniform(-1, 1, (N, N)) +
            1j*np.random.uniform(-1, 1, (N, N)))

        # create input density matrices from a bunch of rotations
        ops = [np.eye(2), Xpi2, Ypi2, Xmpi2]
        Nqubits = int(np.log2(s))
        Us = tensor_combinations(ops, Nqubits)
        rho = np.zeros((s, s))
        rho[0, 0] = 1
        rhos = [dot3(U, rho, U.conj().T) for U in Us]

        # apply operation to all inputs
        Erhos = [operate(rho, chi, As) for rho in rhos]

        # calculate chi matrix and compare to actual
        chiCalc = qpt(rhos, Erhos, proto)
        return np.max(np.abs(chi - chiCalc))

    # 1 qubit
    errs = [test_qpt_protocol('sigma') for _ in range(n)]
    print 'sigma max error:', max(errs)

    errs = [test_qpt_protocol('raise-lower') for _ in range(n)]
    print 'raise-lower max error:', max(errs)

    # 2 qubits
    errs = [test_qpt_protocol('sigma2') for _ in range(n)]
    print 'sigma2 max error:', max(errs)

    errs = [test_qpt_protocol('raise-lower2') for _ in range(n)]
    print 'raise-lower2 max error:', max(errs)

    # 3 qubits
    #from datetime import datetime
    #start = datetime.now()
    #errs = [test_qpt_protocol('sigma3') for _ in range(n)]
    #print 'sigma3 max error:', max(errs)
    #print 'elapsed:', datetime.now() - start

    #errs = [test_qpt_protocol('raise-lower3') for _ in range(n)]
    #print 'raise-lower3 max error:', max(errs)


if __name__ == '__main__':
    print 'Testing state tomography...'
    test_qst(10)

    print 'Testing process tomography...'
    test_qpt()

