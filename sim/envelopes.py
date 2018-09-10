import numpy as np
import scipy as sp
from scipy.special import erf
import matplotlib.pyplot as plt

# sequences in pyle.sim is not compatible with pyle.envelopes
# and not convenient to add pulses with the same time range
# so I write this module to replace pyle.sim.sequence
# as we do not need the frequency domain when we simulate,
# I focus on the time domain.

class Envelope(object):

    def __init__(self, timeFunc, start=None, end=None):
        self.timeFunc = timeFunc
        self.start = start
        self.end = end

    def __call__(self, x):
        return self.timeFunc(x) # * (x>self.start) * (x<=self.end)

    def __add__(self, other):
        if isinstance(other, Envelope):
            start, end = timeRange((self, other))
            def timeFunc(t):
                return self.timeFunc(t)+other.timeFunc(t)
            return Envelope(timeFunc, start=start, end=end)
        else:
            if other==0:
                return self
            raise Exception("can not add a constant to envelopes")
    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Envelope):
            start, end = timeRange((self, other))
            def timeFunc(t):
                return self.timeFunc(t) - other.timeFunc(t)
            return Envelope(timeFunc, start=start, end=end)
        else:
            if other==0:
                return -self
            raise Exception("Cannot substract a constant from envelopes")

    def __rsub__(self, other):
        if isinstance(other, Envelope):
            start, end = timeRange((self, other))
            def timeFunc(t):
                return other.timeFunc(t) - self.timeFunc(t)
            return Envelope(timeFunc, start=start, end=end)
        else:
            if other==0:
                return self
            raise Exception("Cannot substract a constant from envelopes")

    def __mul__(self, other):
        if isinstance(other, Envelope):
            raise Exception('envelopes can only be multilied by constants')
        else:
            def timeFunc(t):
                return self.timeFunc(t) * other
            return Envelope(timeFunc, start=self.start, end=self.end)
    __rmul__ = __mul__

    def __div__(self, other):
        if isinstance(other, Envelope):
            raise Exception("Envelopes can only be divided by constants")
        else:
            def timeFunc(t):
                return self.timeFunc(t)/ other
            return Envelope(timeFunc, start=self.start, end=self.end)
    def __rdiv__(self, other):
        if isinstance(other, Envelope):
            raise Exception("Envelopes can only be divided by constants")
        else:
            def timeFunc(t):
                return other / self.timeFunc(t)
            return Envelope(timeFunc, start=self.start, end=self.end)

    def __neg__(self):
        return -1 * self

    def __pos__(self):
        return self

_zero = lambda x: 0*x

NOTHING = Envelope(_zero, start=None, end=None)

def gaussian(t0, w, length, amp=1.0, phase=0.0, df=0.0):
    sigma = w/np.sqrt(8*np.log(2))
    def timeFunc(t):
        return amp * np.exp(-(t-t0)**2/(2*sigma**2) - 2j*np.pi*df*(t-t0) + 1j*phase)
    return Envelope(timeFunc, start=t0-length/2.0, end=t0+length/2.0)

def gaussianHD(t0, w, length, amp=1.0, phase=0.0, df=0.0,
               Delta=-0.2*2*np.pi, alpha=0.5):
    x = gaussian(t0, w, length, amp, phase, df)
    y = - deriv(x) / Delta * alpha
    return x+1j*y

def Cos(w, amp=1.0, phase=0.0, start=0.0, end=1.0):
    def timeFunc(t):
        return amp*np.cos(w*(t-start)+phase) * (t>=start) * (t<end)
    return Envelope(timeFunc, start, end)

def Sin(w, amp=1.0, phase=0.0, start=0.0, end=1.0):
    def timeFunc(t):
        return amp*np.sin(w*(t-start)+phase) * (t>=start) * (t<end)
    return Envelope(timeFunc, start, end)

def iExp(w, amp=1.0, phase=0.0, start=0.0, end=1.0):
    c = Cos(w, amp, phase, start, end)
    s = Sin(w, amp, phase, start, end)
    return c + 1j*s

def triangle(t0, len, amp, fall=True):
    if not fall:
        return triangle(t0+len, -len, -amp, fall=True)

    tmin = min(t0, t0+len)
    tmax = max(t0, t0+len)

    if len==0 or amp==0:
        return Envelope(_zero, start=tmin, end=tmax)

    def timeFunc(t):
        return amp * (t>=tmin) * (t<tmax) * (1-(t-t0)/len)

    return Envelope(timeFunc, start=tmin, end=tmax)

def rect(t0, len, amp, overshoot=0.0, overshoot_w=1.0):
    tmin = min(t0, t0+len)
    tmax = max(t0, t0+len)
    tmid = (tmin + tmax) / 2.0

    overshoot *= np.sign(amp) # overshoot will be zero if amp is zero

    # to add overshoots in time, we create an envelope with two gaussians
    if overshoot:
        o_w = overshoot_w
        o_amp = 2*np.sqrt(np.log(2)/np.pi) / o_w # total area == 1
        o_env = gaussian(tmin, o_w, o_amp) + gaussian(tmax, o_w, o_amp)
    else:
        o_env = NOTHING
    def timeFunc(t):
        return (amp * (t >= tmin) * (t < tmax) +
                overshoot * o_env(t))
    return Envelope(timeFunc, start=tmin, end=tmax)

def flattop(t0, len, w, amp=1.0, overshoot=0.0, overshoot_w=1.0):
    """A rectangular pulse convolved with a gaussian to have smooth rise and fall."""
    tmin = min(t0, t0+len)
    tmax = max(t0, t0+len)
    overshoot *= np.sign(amp)
    a = 2*np.sqrt(np.log(2)) / w
    if overshoot:
        o_w = overshoot_w
        o_amp = 2*np.sqrt(np.log(2)/np.pi) / o_w # total area == 1
        o_env = gaussian(tmin, o_w, o_amp) + gaussian(tmax, o_w, o_amp)
    else:
        o_env = NOTHING
    def timeFunc(t):
        return (amp * (erf(a*(tmax - t)) - erf(a*(tmin - t)))/2.0 +
                overshoot * o_env(t))

    return Envelope(timeFunc, start=tmin, end=tmax)

def wait(t0, len):
    tmin = min(t0, t0+len)
    tmax = min(t0, t0+len)
    return Envelope(_zero, start=tmin, end=tmax)

delay = wait

def lorentzian(t0, amp, gamma, start=None, end=None):
    """
    lorentzian function
    amp/( (t-t0)^2 + gamma^2 )
    if start is None, start = t0 - 4*abs(gamma)
    if end is None: end = t0 + 4*abs(gamma)
    """
    g = 4*abs(gamma)
    start = (t0 - g) if start is None else start
    end = (t0 + g) if end is None else end
    def timeFunc(t):
         return amp/((t-t0)**2 + gamma**2) * (t>=start) * (t<end)
    return Envelope(timeFunc, start, end)

def linear(b0, slope, start, end):
    """
    linear function
    slope * t + b0
    """
    def timeFunc(t):
        return (slope * t + b0) * (t>=start) * (t<end)
    return Envelope(timeFunc, start, end)

def trapezoid(t0, rise, hold, fall, amp):
    """Create a trapezoidal pulse, built up from triangles and rectangles."""
    return (triangle(t0, rise, amp, fall=False) +
            rect(t0+rise, hold, amp) +
            triangle(t0+rise+hold, fall, amp))

def deriv(env, dt=1e-4):
    def timeFunc(t):
        return (env(t+dt) - env(t-dt))/(2*dt)
    return Envelope(timeFunc, start=env.start, end=env.end)

def timeRange(envelopes):
    starts = [env.start for env in envelopes if env.start is not None]
    start = min(starts) if len(starts) else None
    ends = [env.end for env in envelopes if env.end is not None]
    end = max(ends) if len(ends) else None
    return start, end

def plotEnv(env, start=0, end=100, N=1e3):
    t = np.linspace(start, end, N)
    y = env(t)
    plt.plot(t, np.real(y), t, np.imag(y))
