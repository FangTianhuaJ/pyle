import numpy as np
from numpy import linalg

# some helper functions for dealing with matrices and computing fidelity

def tensor(matrices):
    """Compute the tensor product of a list (or array) of matrices"""
    return reduce(np.kron, matrices)

def dots(matrices):
    """Compute the dot product of a list (or array) of matrices"""
    return reduce(np.dot, matrices)

def dot3(A, B, C):
    """Compute the dot product of three matrices"""
    return np.dot(np.dot(A, B), C)
    
def sqrtm(A):
    """Compute the matrix square root of a matrix"""
    d, U = linalg.eig(A)
    s = np.sqrt(d.astype(complex))
    return dot3(U, np.diag(s), U.conj().T)

def trace_distance(rho, sigma):
    """Compute the trace distance between matrices rho and sigma
    See Nielsen and Chuang, p. 403
    """
    A = rho - sigma
    abs = sqrtm(np.dot(A.conj().T, A))
    return np.real(np.trace(abs)) / 2.0

def fidelity(rho, sigma):
    """Compute the fidelity between matrices rho and sigma
    See Nielsen and Chuang, p. 409
    """
    rhosqrt = sqrtm(rho)
    return np.real(np.trace(sqrtm(dot3(rhosqrt, sigma, rhosqrt))))
    

