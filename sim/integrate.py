import numpy as np
from scipy.integrate import odeint

from pyle.util import processPriority


def codeint(func, z0, t, args=(), full_output=0, reducePriority=1, **opts):
    """Integrator for complex vector equations.
    
    The initial value is specified as a complex vector z0, and the
    derivative function should accept a vector z and time t, and return
    the time derivative at time t as a vector of the same shape.
    """
    m = len(z0)
    def flatfunc(y, t, *args):
        z = y[:m] + 1j*y[m:]
        dz = func(z, t, *args)
        return np.hstack((dz.real, dz.imag))
    y0 = np.hstack((z0.real, z0.imag))
    with processPriority(1 if reducePriority else 2):
        ys = odeint(flatfunc, y0, t, args, full_output=full_output, **opts)
    if full_output:
        ys, res = ys
    psis = ys[:,:m] + 1j*ys[:,m:]
    if full_output:
        return psis, res
    else:
        return psis


def modeint(func, m0, t, args=(), full_output=0, reducePriority=1, **opts):
    """Integrator for real matrix equations.
    
    The initial value is specified as a real matrix m0, and the
    derivative function should accept a matrix m and time t, and return
    the time derivative at time t as a matrix of the same shape.
    """
    h, w = m0.shape
    def flatfunc(y, t, *args):
        m = y.reshape((h,w))
        dm = func(m, t, *args)
        return dm.real.flatten()
    y0 = m0.real.flatten()
    with processPriority(1 if reducePriority else 2):
        ys = odeint(flatfunc, y0, t, args, full_output=full_output, **opts)
    if full_output:
        ys, res = ys
    ms = ys.reshape((-1,h,w))
    if full_output:
        return ms, res
    else:
        return ms


def rhodeint(func, rho0, t, args=(), full_output=0, reducePriority=1, **opts):
    """Integrator for complex matrix equations.
    
    The initial value is specified as a complex matrix rho0, and the
    derivative function should accept a matrix rho and time t, and return
    the time derivative at time t as a matrix of the same shape.
    """
    h, w = rho0.shape
    hw = h*w
    def flatfunc(y, t, *args):
        rho = y[:hw].reshape((h,w)) + 1j*y[hw:].reshape((h,w))
        drho = func(rho, t, *args)
        return np.hstack((drho.real.flatten(), drho.imag.flatten()))
    y0 = np.hstack((rho0.real.flatten(), rho0.imag.flatten()))
    with processPriority(1 if reducePriority else 2):
        ys = odeint(flatfunc, y0, t, args, full_output=full_output, **opts)
    if full_output:
        ys, res = ys
    rhos = (ys[:,:hw].reshape((-1,h,w)) +
         1j*ys[:,hw:].reshape((-1,h,w)))
    if full_output:
        return rhos, res
    else:
        return rhos

def psideint(func, psi0, t, args=(), full_output=0, reducePriority=1, **opts):
    """Wrapper around odeint that allows it to integrate complex vector equations.
    
    The initial value is specified as a (complex) vector psi0, and the
    derivative function should accept a vector psi and time t, and return
    the time derivative at time t as a vector of the same shape.
    """
    m = len(psi0)
    def flatfunc(y, t, *args):
        psi = y[:m] + 1j*y[m:]
        dpsi = func(psi, t, *args)
        return np.r_[dpsi.real, dpsi.imag]
    y0 = np.r_[psi0.real, psi0.imag]
    with processPriority(1 if reducePriority else 2):
        Y = odeint(flatfunc, y0, t, args, full_output=full_output, **opts)
    if full_output:
        Y, res = Y
    psis = Y[:,:m] + 1j*Y[:,m:]
    if full_output:
        return psis, res
    else:
        return psis

# the following helper just allows us to dynamically reduce the priority
# of this running process.  We use it to decrease the priority while running
# the simulation, so as not to lock up our machine while the simulation
# is in progress.  If you have a multi-processor machine, this is probably
# not necessary.  Also, since I'm using the windows API to do this, I capture
# the import error and make a dummy priority manager for other platforms.
