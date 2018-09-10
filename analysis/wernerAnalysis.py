import functools
import itertools

import numpy as np
from scipy import optimize
from scipy.linalg import expm

import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3

#import pyq
#import pyle


def memoize(f):
    _cache = {}
    @functools.wraps(f)
    def wrapped(a):
        try:
            return _cache[a]
        except KeyError:
            val = _cache[a] = f(a)
            return val
    wrapped._cache = _cache
    return wrapped

@memoize
def sierp(n):
    if n == 0:
        return np.array([[1.0]])
    s = sierp(n-1)
    z = np.zeros_like(s)
    return np.vstack([np.hstack([s, z]),
                      np.hstack([s, s])])

@memoize
def sierpinv(n):
    return np.linalg.inv(sierp(n))


def Fmat(f0, f1):
    return np.array([[f0, 1-f1],
                     [1-f0, f1]])


def Fxtf(n, f0s, f1s):
    """Get fidelity matrix for null-result crosstalk-free measurement.
    
    n - number of qubits.  Resulting matrix is 2**n x 2**n
    f0s - measurement fidelities for 0-state on each qubit
    f1s - measurement fidelities for 1-state on each qubit
    """
    assert len(f0s) == n
    assert len(f1s) == n
    return pyq.tensor(np.array([[f0, 1-f1], [1, 1]]) for f0, f1 in zip(f0s, f1s))


def Fxtfinv(n, f0s, f1s):
    """Get inverse of fidelity matrix for null-result crosstalk-free measurement.
    """
    return np.linalg.inv(Fxtf(n, f0s, f1s))


# compute theoretical density matrices
psiW = np.array([0,1,1,0,1,0,0,0], dtype=complex) / np.sqrt(3)
rhoW = np.outer(psiW, psiW.conj())

psiW4 = np.array([0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0], dtype=complex) / np.sqrt(4)
rhoW4 = np.outer(psiW4, psiW4.conj())

psiG = np.array([1,0,0,0,0,0,0,1], dtype=complex) / np.sqrt(2)
rhoG = np.outer(psiG, psiG.conj())

psiG2 = np.array([1,0,0,1], dtype=complex) / np.sqrt(2)
rhoG2 = np.outer(psiG2, psiG2.conj())

psiG2a = np.array([1,0,0,np.exp(-1j*np.pi*0.60)], dtype=complex) / np.sqrt(2)
rhoG2a = np.outer(psiG2a, psiG2a.conj())

psiG4 = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], dtype=complex) / np.sqrt(2)
rhoG4 = np.outer(psiG4, psiG4.conj())


def doTomo(probs, proto, theory=None, correct_vis=False, correct_ml=False, plot=False, e0=0.05, e1=0.05):
    diags = probs.reshape(-1,8)
    if correct_vis:
        def do_correction(diag):
            pxx = np.dot(sierp(3), diag)
            pxxcorr = (pxx - e0) / (1 - e0 - e1)
            return np.dot(sierpinv(3), pxxcorr)
        diags = np.array([do_correction(diag) for diag in diags])
    if correct_ml:
        def do_ml_corr(diag):
            pml = diag * (diag > 0)
            pml = pml / sum(pml)
            return pml
        diags = np.array([do_ml_corr(diag) for diag in diags])
    rho = pyq.tomo.qst(diags, proto)
    if plot:
        plotRho(rho, theory)
    return rho


def max_liklihood(p):
    pos = p * (p > 0) # clear negative entries
    norm = pos / sum(pos) # normalize so sum is 1
    return norm


def correct_xtf(p, f0s, f1s, n=3):
    Finv = Fxtfinv(n, f0s, f1s)
    return pyq.dot3(Finv, sierp(n), p)


def xtf_mle(pms, f0s, f1s, n=3, **fmin_opts):
    pxms = np.dot(sierp(n), pms)
    M = Fxtf(n, f0s, f1s) # matrix from intrinsic to measured probs
    Minv = np.linalg.inv(M)
    
    def make_ps(ts):
        tt = ts*ts
        return tt / sum(tt)
    
    def log(x): # safe version that returns -Inf for negative numbers, rather than NaN
        return np.log(x.real * (x.real > 0))
    
    def unlikelihood(tis): # negative of likelihood function
        pis = make_ps(tis)
        pxis = np.dot(M, pis)
        terms = -pxms * log(pxis) - (1-pxms) * log(1-pxis)
        return sum(terms)
    
    # make a guess
    pis_guess = np.dot(Minv, pxms)
    pis_guess = pis_guess * (pis_guess > 0)
    pis_guess /= sum(pis_guess)
    tis_guess = np.sqrt(pis_guess)
    
    # minimize
    tis = optimize.fmin(unlikelihood, tis_guess, **fmin_opts)
    return make_ps(tis)


def correct_zero_tunneling(p, e0=0.05, n=3):
    pxx = np.dot(sierp(n), p)
    pxxr = pxx.reshape((2,)*n)
    for indices in itertools.product(range(2), repeat=n):
        num_zeros = n - sum(indices)
        #print 'dividing %s by (1 - e0)**%d' % (indices, num_zeros)
        pxxr[tuple(indices)] /= (1 - e0)**num_zeros
    pxxc = pxxr.reshape(pxx.shape)
    pc = np.dot(sierpinv(n), pxxc)
    return pc


def hermiticity(rho):
    """A measure of how Hermitian a matrix is."""
    rhoabs = np.abs(rho)
    absdiff = np.abs(rho - rho.conj().T)
    anhermiticity = np.max(absdiff) / np.max(rhoabs)
    return 1 - 0.5*anhermiticity # anhermiticity is between 0 and 2


def plotRho(rho, theory, cmap=None):
    fig = plt.figure(figsize=(11,6))
    axabs_ex = fig.add_subplot(2,3,1)
    axre_ex = fig.add_subplot(2,3,2)
    axim_ex = fig.add_subplot(2,3,3)
    axabs_th = fig.add_subplot(2,3,4)
    axre_th = fig.add_subplot(2,3,5)
    axim_th = fig.add_subplot(2,3,6)
    plt.subplots_adjust(right=0.8)
    
    def set_ticklabels(ax, n=3):
        ticks = [i+0.5 for i in range(2**n)]
        labels = [bin(i)[2:].rjust(n,'0') for i in range(2**n)]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    for ax in [axabs_ex, axre_ex, axim_ex, axabs_th, axre_th, axim_th]:
        set_ticklabels(ax)
    
    absmax = lambda a: max(np.max(np.max(np.abs(a.real))), np.max(np.max(np.abs(a.imag))))
    
    vmax_ex = absmax(rho)
    vmax_th = absmax(theory)

    if cmap is None:
        cmap = cm.get_cmap('RdYlBu')
    opts_ex = {'vmin': -vmax_ex, 'vmax': vmax_ex, 'cmap': cmap}
    opts_th = {'vmin': -vmax_th, 'vmax': vmax_th, 'cmap': cmap}

    axabs_ex.pcolor(np.abs(rho), **opts_ex)
    axre_ex.pcolor(rho.real, **opts_ex)
    axim_ex.pcolor(rho.imag, **opts_ex)
    
    axabs_th.pcolor(np.abs(theory), **opts_th)
    axre_th.pcolor(theory.real, **opts_th)
    axim_th.pcolor(theory.imag, **opts_th)
    
    axabs_ex.set_title('abs')
    axre_ex.set_title('real')
    axim_ex.set_title('imag')
    
    axabs_ex.set_ylabel('experiment')
    axabs_th.set_ylabel('theory')
    
    norm_ex = matplotlib.colors.Normalize(vmin=-vmax_ex, vmax=vmax_ex)
    norm_th = matplotlib.colors.Normalize(vmin=-vmax_th, vmax=vmax_th)
    
    rect_ex = axim_ex.get_position().get_points()
    axcb_ex = fig.add_axes([0.85, rect_ex[0,1], 0.05, rect_ex[1,1]-rect_ex[0,1]])
    matplotlib.colorbar.ColorbarBase(axcb_ex, norm=norm_ex, cmap=cmap)
    
    rect_th = axim_th.get_position().get_points()
    axcb_th = fig.add_axes([0.85, rect_th[0,1], 0.05, rect_th[1,1]-rect_th[0,1]])
    matplotlib.colorbar.ColorbarBase(axcb_th, norm=norm_th, cmap=cmap)
    
    return fig


def plot3d(rho, d=0.1, cmap=None, vmin=-1, vmax=1, zmin=-1, zmax=1):
    if cmap is None:
        cmap = cm.RdYlBu
    norm = matplotlib.colors.Normalize(vmin, vmax)
    
    fig = plt.figure(figsize=(6,3))
    ax = plt3.Axes3D(fig)
    
    w, h = rho.shape
    for x in range(w):
        for y in range(h):
            ax.bar3d([x+d/2], [y+d/2], [0], 1-d, 1-d, rho[x,y], cmap(norm(rho[x,y])))
            
#            xs = np.array([x+d/2, x+1-d/2])
#            ys = np.array([y+d/2, y+1-d/2])
#            zs = np.array([0, rho[x,y]])
#            
#            c = cmap(norm(zs[1]))
#            
#            # front / back
#            Z, X = np.meshgrid(zs, xs)
#            Y = np.ones((2,2)) * ys[0]
#            ax.plot_surface(X, Y, Z, color=c)
#            
#            Y = np.ones((2,2)) * ys[1]
#            ax.plot_surface(X, Y, Z, color=c)
#            
#            # left / right
#            Y, Z = np.meshgrid(ys, zs)
#            X = np.ones((2,2)) * xs[0]
#            ax.plot_surface(X, Y, Z, color=c)
#            
#            X = np.ones((2,2)) * xs[1]
#            ax.plot_surface(X, Y, Z, color=c)
#            
#            # bottom / top
#            X, Y = np.meshgrid(xs, ys)
#            Z = np.ones((2,2)) * zs[0]
#            ax.plot_surface(X, Y, Z, color=c)
#            
#            X, Y = np.meshgrid(xs, ys)
#            ax.plot_surface(X, Y, Z, color=c)
    ax.set_zlim3d((zmin, zmax))
    
    #ax.w_xaxis.set_ticks(np.arange(w)+0.5)
    #ax.w_xaxis.set_ticklabels([bin(i)[2:].rjust(3,'0') for i in range(w)])
    
    #ax.w_yaxis.set_ticks(np.arange(w)+0.5)
    #ax.w_yaxis.set_ticklabels([bin(i)[2:].rjust(3,'0') for i in range(w)])

    return fig


# set up operator bases
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
dot3 = lambda a, b, c: np.dot(np.dot(a, b), c)

basis1 = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

labels2 = ['XI', 'YI', 'ZI',
           'IX', 'IY', 'IZ',
           'XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']
basis2 = dict((label, reduce(np.kron, [basis1[c] for c in label])) for label in labels2)

labels3 = ['XII', 'YII', 'ZII',
           'IXI', 'IYI', 'IZI',
           'IIX', 'IIY', 'IIZ',
            
           'XXI', 'XYI', 'XZI', 'YXI', 'YYI', 'YZI', 'ZXI', 'ZYI', 'ZZI',
           'XIX', 'XIY', 'XIZ', 'YIX', 'YIY', 'YIZ', 'ZIX', 'ZIY', 'ZIZ',
           'IXX', 'IXY', 'IXZ', 'IYX', 'IYY', 'IYZ', 'IZX', 'IZY', 'IZZ',
            
           'XXX', 'XXY', 'XXZ', 'XYX', 'XYY', 'XYZ', 'XZX', 'XZY', 'XZZ',
           'YXX', 'YXY', 'YXZ', 'YYX', 'YYY', 'YYZ', 'YZX', 'YZY', 'YZZ',
           'ZXX', 'ZXY', 'ZXZ', 'ZYX', 'ZYY', 'ZYZ', 'ZZX', 'ZZY', 'ZZZ']
basis3 = dict((label, reduce(np.kron, [basis1[c] for c in label])) for label in labels3)

#basis4 = {}
#for key in sorted(basis1.keys()):
#    basis4[key] = reduce(np.kron, [basis1[c] for c in key])


def E(op, rho):
    """Find the expectation value of an operator for a given density matrix."""
    return np.trace(np.dot(op, rho))


def plotPauli(rho, rho_th=None, d=0.1, phases=None, dpi=200, map=True):
    num_qubits = int(np.log2(rho.shape[0]))
    N = 4**num_qubits-1
    #basis = {2: basis2, 3: basis3, 4: basis4}[num_qubits]
    basis, labels = {2: (basis2, labels2),
                     3: (basis3, labels3)}[num_qubits]

    if phases is not None:
        z = pyq.tensor(expm(-1j*phase*Z) for phase in phases)
        rho = dot3(z, rho, z.conj().T)

    w = 1 - 2*d
    x = np.array(range(N)) + d
    xticks = np.array(range(N)) + 0.5
    #labels = sorted(basis.keys())
    if rho_th is not None:
        vals_th = [E(basis[op], rho_th) for op in labels]
    vals = [E(basis[op], rho) for op in labels]

    cmap = cm.get_cmap('RdYlBu')
    norm = matplotlib.colors.Normalize(-1, 1)

    fig = plt.figure(figsize=(10,3), dpi=dpi)
    ax = fig.add_subplot(111)
    
    ax.add_artist(matplotlib.patches.Rectangle((-0.5, -1.1), 3.5, 2.2, color=(0.85, 0.85, 0.85), zorder=-10))
    ax.add_artist(matplotlib.patches.Rectangle((6, -1.1), 3, 2.2, color=(0.85, 0.85, 0.85), zorder=-10))
    ax.add_artist(matplotlib.patches.Rectangle((18, -1.1), 9, 2.2, color=(0.85, 0.85, 0.85), zorder=-10))
    ax.add_artist(matplotlib.patches.Rectangle((36, -1.1), 27.5, 2.2, color=(0.85, 0.85, 0.85), zorder=-10))
    
    #ax.add_artist(matplotlib.patches.Rectangle((3, -1.1), 3, 2.2, color=(0.85, 0.85, 0.85), zorder=-10))
    #ax.add_artist(matplotlib.patches.Rectangle((9, -1.1), 9, 2.2, color=(0.85, 0.85, 0.85), zorder=-10))
    #ax.add_artist(matplotlib.patches.Rectangle((27, -1.1), 9, 2.2, color=(0.85, 0.85, 0.85), zorder=-10))
    
    if rho_th is not None:
        if map:
            for xi, val in zip(x, vals_th):
                c = cmap(norm(val))
                c = (0.75, 0.75, 0.75)
                ax.bar([xi], [val], width=w, fc=c, ec=(0.3, 0.3, 0.3))
        else:
            ax.bar(x, vals_th, width=w, fc=(0.7, 0.7, 1), ec=(0.4, 0.4, 0.4))
    if map:
        for xi, val in zip(x, vals):
            c = cmap(norm(val))
            ax.bar([xi], [val], width=w, fc=c, ec=(0, 0, 0))
    else:
        ax.bar(x, vals, width=w, fc=(0, 0, 1), ec=(0, 0, 0))
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, N+0.5)
    ax.set_ylim(-1.1, 1.1)
    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.12)
    
    rect = ax.get_position().get_points()
    axcb = fig.add_axes([0.93, rect[0,1], 0.02, rect[1,1]-rect[0,1]])
    matplotlib.colorbar.ColorbarBase(axcb, norm=norm, cmap=cmap)
    
    return fig


#fig1 = plot_pauli(rho_w, rhoW)
#fig1.suptitle('W Experiment')
#fig1.savefig('pauli_W_exp.png')
#
#fig2 = plot_pauli(rho_ghz2, rhoG2)
#fig2.suptitle('GHZ2 Experiment')
#fig2.savefig('pauli_GHZ2_exp.png')
#
#fig3 = plot_pauli(rho_ghz2, rhoG2a)
#fig3.suptitle('GHZ2 Experiment (+ phase correction)')
#fig3.savefig('pauli_GHZ2_exp_phase.png')
#
#fig4 = plot_pauli(rhoG)
#fig4.suptitle('GHZ3 Theory')
#fig4.savefig('pauli_GHZ3_th.png')
#
#fig5 = plot_pauli(rhoG4)
#fig5.suptitle('GHZ4 Theory')
#fig5.savefig('pauli_GHZ4_th.png')
#
#fig6 = plot_pauli(rhoW4)
#fig6.suptitle('W4 Theory')
#fig6.savefig('pauli_W4_th.png')
#
#
#plt.show()

def ghz_G(rho):
    Mx = pyq.tomo.Ympi2 # to measure about X, rotate about Y by -pi/2
    My = pyq.tomo.Xpi2 # to measure about Y, rotate about X by pi/2
    
    Uxxx = pyq.tensor((Mx, Mx, Mx))
    Uyyx = pyq.tensor((My, My, Mx))
    Uyxy = pyq.tensor((My, Mx, My))
    Uxyy = pyq.tensor((Mx, My, My))
    
    Pxxx = np.diag(pyq.dot3(Uxxx, rho, Uxxx.conj().T)).real
    Pyyx = np.diag(pyq.dot3(Uyyx, rho, Uyyx.conj().T)).real
    Pyxy = np.diag(pyq.dot3(Uyxy, rho, Uyxy.conj().T)).real
    Pxyy = np.diag(pyq.dot3(Uxyy, rho, Uxyy.conj().T)).real
    
    indices = [int(i,2) for i in ('000', '011', '101', '110')]
    Axxx = sum(Pxxx[indices])
    Ayyx = sum(Pyyx[indices])
    Ayxy = sum(Pyxy[indices])
    Axyy = sum(Pxyy[indices])
    
    G = Axxx - Ayyx - Ayxy - Axyy
    
    return G
    
    




