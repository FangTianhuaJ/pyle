import numpy as np

# see http://qwiki.stanford.edu/wiki/Entanglement_of_Formation

dotall = lambda a: reduce(np.dot, a)

def concurrence(rho):
    """Concurrence of a two-qubit density matrix."""
    yy = np.array([[0,0,0,-1], [0,0,1,0], [0,1,0,0], [-1,0,0,0]], dtype=complex)
    m = dotall([rho, yy, rho.conj(), yy])
    eigs = [np.abs(e) for e in np.linalg.eig(m)[0]]
    e = [np.sqrt(x) for x in sorted(eigs, reverse=True)]
    return max(0, e[0] - e[1] - e[2] - e[3])

def eof(rho):
    """Entanglement of formation of a two-qubit density matrix."""
    def h(x):
        if x <= 0 or x >= 1:
            return 0
        return -x*np.log2(x) - (1-x)*np.log2(1-x)
    C = concurrence(rho)
    arg = max(0, 1-C**2)
    return h((1 + arg)/2.)
