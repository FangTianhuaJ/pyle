import numpy as np
from numpy import abs, exp, dot, size, pi, real, imag, diag, min, max
from numpy.linalg.linalg import svd
from scipy.misc import factorial
from scipy.special import laguerre
from pylab import imshow, xlabel, ylabel, ion, ioff, draw, clf

from myPlotUtils import mycolormap


def complexgrid(r, n):
    alpha = np.linspace(-r, r, n, endpoint=True)
    return alpha[np.newaxis,:]+ 1.0j * alpha[:,np.newaxis]


def leastsqinv(m):
    u,s,vh = svd(m, full_matrices=0)
    s = np.conjugate(1/s)
    u = dot(u * s[np.newaxis,:], vh)
    return np.transpose(np.conjugate(u)), u

def packmatrix(N):
    """Helper function for Hpack and Hunpack"""
    d = np.arange(N)
    d = d[:,None] - d[None,:]
    return 1.0 * (d==0) + np.sqrt(2) * (d>0) + 1j*np.sqrt(2) * (d<0)

def Hconjugate(m):
    """Hermitian conjugate. If m has more than 2 dimensions, transpose the
       last 2."""
    perm = np.arange(len(np.shape(m)))
    perm[-2:] = perm[-1], perm[-2]
    return np.transpose(np.conjugate(m),perm)

def Hpack(rho):
    """Transform a hermitian matrix into a real matrix"""
    s = np.shape(rho)
    l = size(rho)
    n = np.arange(s[-2])
    m = np.arange(s[-1])
    rho = np.reshape(rho,(l/s[-2]/s[-1],s[-2],s[-1]))
    if len(s) == 2:
        ns = (l,)
    else:
        ns = (l/s[-2]/s[-1],s[-2]*s[-1])
    return np.reshape((rho*np.conjugate(packmatrix(s[-1])[None,:,:])).real, ns)

def Hunpack(rho):
    """Transform a real matrix into a hermition matrix,
    undoes what Hpack does"""
    s = np.shape(rho)
    l = size(rho)
    N = int(round(np.sqrt(s[-1])))
    ns = s[:-1] + (N,N)
    rho = np.reshape(rho,(l/(N**2),N,N))
    n = np.arange(N)
    rho = rho * packmatrix(N)[None,:,:]
    return np.reshape(0.5*(rho + Hconjugate(rho)),ns)

def displacement(alpha, dims=(10,10)):
    """Calculate the displacement operator in the Fock basis"""
    alphaS = -abs(1.0*alpha)**2

    m,n = np.meshgrid(range(dims[1]), range(dims[0]))
    nsm = n<m
    s = n*nsm + m * (1-nsm)
    d = abs(m-n)
    iterator = 1.0/factorial(s)/factorial(d)
    result = 1.0*iterator
    for j in range(1, min(dims)):
        iterator *= (s+1-j)*alphaS/j/(j+d)
        result += iterator

    return result * alpha**(d*(1-nsm)) * (-np.conjugate(alpha))**(d*nsm) * exp(0.5*alphaS) * np.sqrt(factorial(n)*factorial(m))


def wignermatrixold(alpha, N=10, K=100):
    D = displacement(alpha, (N,K))
    return dot(D * (1-2*(np.arange(K)%2))[np.newaxis,:], np.conjugate(np.transpose(D)))


def wignermatrix(alpha, N=10):
    """calculate the Matrix Displacement(alpha)*Parity*Displacement(-alpha)
    = Displacement(2*alpha)*Parity (see write-up"""
    alpha = np.asarray(alpha)
    salpha = np.shape(alpha)
    alpha = np.reshape(2.0*alpha,size(alpha))
    alphaS = -abs(alpha)**2
    m,n = np.meshgrid(range(N), range(N))
    s = min([n,m],axis=0)
    d = abs(n-m)
    iterator = (1.0/factorial(s)/factorial(d))[np.newaxis,:,:] \
        + 0.0 * alphaS[:,np.newaxis,np.newaxis]
    result = 1.0*iterator
    for x in range(1, N):
        iterator *= ((s+1-x) / float(x) / (x+d))[np.newaxis,:,:] \
            * alphaS[:,np.newaxis,np.newaxis]
        result += iterator
    return np.reshape(result * alpha[:,np.newaxis,np.newaxis]**(n-s)[np.newaxis,:,:] \
       * np.conjugate(alpha)[:,np.newaxis,np.newaxis]**(m-s)[np.newaxis,:,:] \
       * exp(0.5*alphaS)[:,np.newaxis,np.newaxis] \
       * (np.sqrt(factorial(n)*factorial(m)) * (1-2*(s%2)))[np.newaxis,:,:],
                   salpha + (N,N))


def wignermatrix1(alpha, N=10):
    alpha = 2.0*alpha
    alphaS = -abs(alpha)**2
    m,n = np.meshgrid(range(N), range(N))
    s = min([n,m],axis=0)
    d = abs(m-n)
    iterator = 1.0/factorial(s)/factorial(d)
    result = 1.0*iterator
    for x in range(1, N):
        iterator *= (s+1-x)*alphaS/x/(x+d)
        result += iterator
    return result * alpha**(n-s) * np.conjugate(alpha)**(m-s) \
        * exp(0.5*alphaS) * np.sqrt(factorial(n)*factorial(m)) * (1-2*(s%2))

def Qkmatrix(alpha, K=10, N=10):
    """Calculate <m|D(alpha)|k><k|D(-alpha)|n>. k=0...K-1, n,m=0...N-1.
    The return value has the following shape:
    [shape(alpha), K, N, N]"""

    s = np.shape(alpha)
    alpha = np.reshape(alpha,size(alpha))
    M = displacement(-alpha,N=(K,N))
    return M[:,:,None,:] * np.conjugate(M)[:,:,:,None]

def wigner(rho, alpha, checkimag=True):

    sia = size(alpha)
    sir = size(rho)
    sizelim = 1<<20
    sha = np.shape(alpha)
    alpha = np.resize(alpha, sia)
    blocksize = sizelim / sir
    blocksize += (blocksize == 0)
    result = np.zeros(sia, dtype=complex)
    for n in range((sia+blocksize-1)/blocksize):
        start = n*blocksize
        stop = min(start+blocksize,sia)
        result[start:stop] = np.trace(dot(wignermatrix(alpha[start:stop],
                                                  N=np.shape(rho)[0]),rho),
                                 axis1=-2,axis2=-1)
    if max(abs(imag(result))) > 1e-12:
        print 'Wigner function has imaginary part. Is your density matrix hermitian?'
    else:
        result = real(result)
    result *= 2.0/pi
    result = np.reshape(result,sha)
    return result


def wigner1(rho, alpha):
    alpha = np.asarray(alpha)
    s = np.shape(alpha)
    alpha = np.reshape(alpha, size(alpha))
    result = 0.0j * alpha
    N = np.shape(rho)[1]
    for n, a in enumerate(alpha):
        result[n] = np.trace(dot(rho, wignermatrix(a, N=N)))
    if max(abs(imag(result))) > 1e-4:
        print 'Wigner function contains imaginary part. Is your density matrix hermitian?'
    else:
        result=real(result)
    return np.reshape(2.0/pi*result,s)


def wignerN(n, alpha):
    alpha = abs(alpha)**2
    return 2.0/pi * (-1)**n * exp(-2*alpha) * np.polyval(laguerre(n),4*alpha)


def plotW(rho, r=2, points=101, interpolation='bicubic'):
    ioff()
    clf()
    alpha = complexgrid(r,points)
    w = wigner(rho,alpha)
    r = r + 0.5*r/(points-1)
    imshow(w, cmap = mycolormap('rkb'), interpolation = interpolation, aspect = 'equal', extent = (-r,r,-r,r), vmin=-2.0/pi, vmax=2.0/pi, origin='lower')
    xlabel('Real')
    ylabel('Imag')
    draw()
    ion()


def coherent(alpha, K=100):
    """return the first K elements in the Fock basis of a coherent state with
     aplitude alpha"""
    n = np.arange(K)
    return exp(-0.5*abs(alpha)**2)*alpha**n/np.sqrt(factorial(n))


def pure(psi):
    psi = 1.0*np.asarray(psi)
    cpsi = np.conjugate(psi)
    return np.outer(psi, cpsi) / np.inner(psi, cpsi)


def WtoRho(alpha, W, sigma=1.0, N=5, K=100, returnCovar=False):
    if np.shape(alpha) != np.shape(W):
        print 'alpha and W must have the same shape'
        return
    sigma = sigma + W * 0.0
    s = size(alpha)
    W = np.reshape(W,s)
    sigma = np.reshape(sigma,s)
    alpha = np.reshape(alpha,s)
    N2 = N*N
    NU = N*(N-1)/2
    n = np.arange(N)
    mm,nm = np.meshgrid(n,n)
    nm = np.reshape(nm, size(nm))
    mm = np.reshape(mm, size(mm))
    i = np.nonzero(nm < mm)[0] # upper diagonal
    nm = nm[i]
    mm = mm[i]
    print '.',
    upper = 2/pi*wignermatrix(alpha, N=N)
    matrix = np.zeros((s,N2), dtype=complex)
    matrix[:,2*NU:N2] = upper[:,n,n]
    upper = np.reshape(upper, (s,N2))[:,i]
    matrix[:,0:NU] = 2*real(upper)
    matrix[:,NU:2*NU] = 2*imag(upper)

    print '.',
    linv, rinv = leastsqinv(matrix)
    print '.',
    fit = dot(linv,W)
    print '.',
    covar = dot(linv * (sigma**2)[np.newaxis,:], rinv)
    print '.',
    sigmaRaw = np.sqrt(diag(covar))

    i = np.arange(NU)
    i,_ = np.meshgrid(i,i)

    rho = np.zeros((N,N), dtype=complex)
    sigma = np.zeros((N,N), dtype=complex)

    rho[nm,mm] = fit[0:NU] + 1.0j * fit[NU:2*NU]
    rho[mm,nm] = fit[0:NU] - 1.0j * fit[NU:2*NU]
    sigma[nm,mm] = sigmaRaw[0:NU] + 1.0j * sigmaRaw[NU:2*NU]
    sigma[mm,nm] = sigmaRaw[0:NU] + 1.0j * sigmaRaw[NU:2*NU]
    n = np.arange(N)
    rho[n,n] = fit[2*NU:N2]
    sigma[n,n] = sigmaRaw[2*NU:N2]
    print '.'
    if returnCovar:
        return rho, sigma, covar, nm, mm
    else:
        return rho, sigma


def QktoRho(alpha, Qk, sigma=1.0, N=5, returnSigma=False, returnCovar=False):
    """Calculate the NxN density matrix corresponding to the photon number
    Probabilities Qk, measured after a displacements alpha. The shape
    of Qk has to be the same as alpha but with and added last
    dimension giving the photon number.  If returnSigma=True, return
    also the uncertainty in the density matrix (NxN matrix). If
    returnCovar=True return also the covariance of the density matrix
    (NxNxNxN matrix)."""

    sigma = sigma + Qk * 0.0
    s = size(alpha)
    K = np.shape(Qk)[-1]
    Qk = np.reshape(Qk,s*K)
    sigma = np.reshape(sigma,s*K)
    alpha = np.reshape(alpha,s)
    linv,rinv = leastsqinv(Hpack(Qkmatrix(alpha,K=K,N=N)))
    rho = Hunpack(dot(linv,Qk))
    if returnSigma or returnCovar:
        covar = dot(linv * (sigma**2)[np.newaxis,:], rinv)
        rho = [rho]
    if returnSigma:
        rho += [np.reshape(np.sqrt(diag(covar)),(N,N))]
    if returnCovar:
        rho += [np.reshape(covar,(N,N,N,N))]
    return rho


def jinjang(r, n):
    alpha = complexgrid(r,n)
    result = 1.0*np.sign(real(alpha))*(abs(alpha)<=r)
    outer = abs(alpha-0.5j*r)<0.5*r
    inner = abs(alpha-0.5j*r)<0.2*r
    result = result * (1-outer) + outer - 2*inner
    outer = abs(alpha+0.5j*r)<0.5*r
    inner = abs(alpha+0.5j*r)<0.2*r
    result = result * (1-outer) - outer + 2*inner
    return result


