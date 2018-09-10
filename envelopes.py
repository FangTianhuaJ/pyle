# Control envelopes in time domain and frequency domain
#
# For a pair of functions g(t) <-> h(f) we use the following
# convention for the Fourier Transform:
#
#          / +inf
#         |
# h(f) =  | g(t) * exp(-j*2pi*f*t) dt
#         |
#        / -inf
#
#          / +inf
#         |
# g(t) =  | h(f) * exp(2j*pi*f*t) df
#         |
#        / -inf
#
# Note that we are working with frequency in GHz, rather than
# angular frequency.  Also note that the sign convention is opposite
# to what is normally taken in physics.  But this is the convention
# used here and in the DAC deconvolution code, so you should use it.
#
# To get Mathematica to use this convention, add the option FourierParameters -> {0, -2*Pi}
#
# Also, this convention is better :)
# The physics convention comes from a very unfortunate choice of sign in
# Heisenberg/Schrodinger's equation - DTS

from __future__ import absolute_import # have to do this so we get math std library

import math

import numpy as np
from scipy.special import erf
from scipy.interpolate import interp1d
import scipy.interpolate as interpolate
from scipy.integrate import quad
import matplotlib.pyplot as plt
import types
import traceback

from pyle.util import convertUnits
from pyle.interpol import interp1d_cubic
from labrad.units import Unit

V, mV, us, ns, GHz, MHz, dBm, rad, au = [Unit(s) for s in ('V', 'mV', 'us', 'ns', 'GHz', 'MHz', 'dBm', 'rad', 'au')]

exp = np.exp
pi = np.pi
cos = np.cos
sin = np.sin
sinc = np.sinc

enable_caching = False
cache_hits = 0
cache_misses = 0

class Envelope(object):
    """Represents a control envelope as a function of time or frequency.

    Envelopes can be added to each other or multiplied by constant values.
    Multiplication of two envelopes and addition of a constant value (other
    than zero) are not equivalent in time and fourier domains, so these
    operations are not supported.

    Envelopes keep track of their start and end time, and when added
    together the new envelope will use the earliest start and latest end,
    to cover the entire range of its constituent parts.

    Envelopes can be evaluated as functions of time or frequency using the
    fourier flag.  By default, they are evaluated as a function of time.
    """
    @convertUnits(start='ns', end='ns')
    def __init__(self, start=None, end=None):
        self.start = start
        self.end = end

    def timeFunc(self, t):
        raise NotImplemented('Envelope timeFunc must be overridden')

    def freqFunc(self, f):
        raise NotImplemented('Envelope freqFunc must be overridden')

    @property
    def duration(self):
        return self.end - self.start

    def __call__(self, x, fourier=False):
        if fourier:
            return self.freqFunc(x)
        else:
            return self.timeFunc(x)

    def __add__(self, other):
        if isinstance(other, Envelope):
            return EnvSum(self, other)
        else:
            # if we try to add envelopes with the built in sum() function,
            # the first envelope is added to 0 before adding the rest.  To support
            # this, we add a special case here since adding 0 in time or fourier
            # is equivalent
            if other == 0:
                return self
            return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Envelope):
            return EnvSum(self, -other)
        else:
            # if we try to add envelopes with the built in sum() function,
            # the first envelope is added to 0 before adding the rest.  To support
            # this, we add a special case here since adding 0 in time or fourier
            # is equivalent
            if other == 0:
                return self
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, Envelope):
            return EnvSum(-self, other)
        else:
            # if we try to add envelopes with the built in sum() function,
            # the first envelope is added to 0 before adding the rest.  To support
            # this, we add a special case here since adding 0 in time or fourier
            # is equivalent
            if other == 0:
                return -self
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Envelope):
            return NotImplemented
        else:
            return EnvProd(self, other)
    __rmul__ = __mul__

    def __div__(self, other):
        if isinstance(other, Envelope):
            return NotImplemented
        else:
            return EnvProd(self, 1./other)

    def __rdiv__(self, other):
        if isinstance(other, Envelope):
            return NotImplemented
        else:
            return EnvProd(EnvInv(self), other)

    def __neg__(self):
        return -1 * self

    def __pos__(self):
        return self

class DynamicEnvelope(Envelope):
    """
    for old Envelope class, which took the time/freq func as parameters.
    now deprecated
    """
    def __init__(self, timeFunc, freqFunc, start=None, end=None):
        self.__freqFunc = freqFunc
        self.__timeFunc = timeFunc
        super(DynamicEnvelope, self).__init__(start, end)
    def timeFunc(self, t):
        return self.__timeFunc(t)
    def freqFunc(self, f):
        return self.__freqFunc(f)

class NumericalPulse(Envelope):
    '''
    Base class for numerical pulses.  In a subclass, provide a
    timeFunc method.  freqFunc will interpolate and FFT to get the
    frequency space representation, and convolve with a Gaussian to
    reduce ringing.

    IMPORTANT NOTE: Convolution (width w) with a Gaussian is done only
    for the fourier domain.  Therefore, timeFunc provided should NOT
    contain the Gaussian, and it will not show up when e.g., plotted
    on the computer screen, but it will show up on the scope.

    The interpolation parameters can be overridden by changing the
    class variables NP_dt, NP_padTime, and NP_w.

    This is a more object oriented approach than the numerical_pulse
    decorator below.
    '''
    NP_dt=0.5*ns
    NP_padTime=1000*ns
    NP_w=0.0*ns

    def freqFunc(self, f):
        """
        A numerically defined frequency representation from NumericalPulse
        """
        ### FREQUENCY DOMAIN INTERPOLATING FUNCTION ###
        # Python DFT convention:
        #       N - 1
        #       ----
        #       \            -2*pi*i*n*k/N
        # a_k =  >    a_n * e
        #       /
        #       ----
        #      n = 0
        # But the actual FT relates to the DFT through a(v_k) = DFT(k)*T/N
        # Need to have the Fourier transform of [times,samples]
        # Best way to do this would probably be to pad the GRAPE pulse with starting values
        # (maybe remove the DC componant) and take the DFT

        dt = self.NP_dt['ns']
        padTime = self.NP_padTime['ns']
        w = self.NP_w['ns']

        numpts = 2**int(np.ceil(np.log2((self.end-self.start+2*padTime)/dt)))
        t = np.linspace(self.start-padTime, self.end+padTime, numpts, endpoint=False)
        assert len(t)==numpts
        dt = t[1]-t[0]  # Actual dt may be smaller to ensure power-of-two points

        samples = self.timeFunc(t)
        freqs = np.fft.fftshift(np.fft.fftfreq(numpts,dt))
        sampleSpectrum = np.fft.fft(samples)
        sampleSpectrum = dt*np.fft.fftshift(sampleSpectrum)
        # shift the function to its origin for a slowly evolving phase
        # envelope which is easy to interpolate If the signal is
        # symmetric, the imaginary part should be zero.  We'll subtract
        # this phase later with fast_phase

        sampleSpectrum = sampleSpectrum * np.exp(2.0j * np.pi * (self.end-self.start+2*padTime)/2.0 * freqs)
        interpSpecFuncR = interp1d_cubic(freqs, np.real(sampleSpectrum),fill_value=0)
        interpSpecFuncI = interp1d_cubic(freqs, np.imag(sampleSpectrum),fill_value=0)

        if w>0.0:
            kernel = gaussian(0, w, 2*np.sqrt(np.log(2)/np.pi) / w)(f, fourier=True)
        else:
            kernel = 1.0

        # At this point, the middle of the pulse will be at time zero.  We need to shift it by end-start/2
        #fast_phase = 1
        fast_phase = -2.0 * np.pi * (self.end+self.start)/2  * f

        return (interpSpecFuncR(f) + 1j*interpSpecFuncI(f)) * np.exp(1j*fast_phase) * kernel

def memoize_envelope(cls):
    '''
    Decorator for memoizing envelope classes.
    If all of the parameters to __init__ are the same, and the function
    is evaulated at the same frequency points, return the same data.

    The array is set to be read-only.  This prevents accidental
    corruption of the cache.  We could also just return a copy
    if that turns out to be a problem.
    '''
    if not enable_caching:
        return cls
    old_init = cls.__init__
    old_freqFunc = cls.freqFunc

    def __init__(self, *args, **kwargs):
        # NOTE: Caching now defaults to True
        cache = kwargs.pop('cache', True)
        if cache:
            key = (args, tuple(kwargs.items()))
            self._instance_key = key
        old_init(self, *args, **kwargs)
    def freqFunc(self, f):
        if not enable_caching or not hasattr(self, '_instance_key'):  # Allow global disabling of caching
            return old_freqFunc(self, f)
        data_key = (f[0], f[-1], len(f))
        key = (self._instance_key, data_key)
        try:
            x = self._cache[key]
            global cache_hits
            cache_hits += 1
            return x
        except KeyError:
            global cache_misses
            cache_misses += 1
            x = old_freqFunc(self, f)
            x.setflags(write=False)
            self._cache[key] = x
            return x
        except TypeError:
            print("Warning: unable to hash parameters for envelope of type %s" % (cls.__name__,))
            return old_freqFunc(self, f)
    __init__.__doc__ = old_init.__doc__
    freqFunc.__doc__ = (old_freqFunc.__doc__ or "Envelope Frequency Function") + '\n\nCaching added by decorator memoize_envelope'
    cls._cache={}
    cls.freqFunc = freqFunc
    if not hasattr(cls, '_memoize_manual_key'):
        cls.__init__ = __init__
    return cls

def envelope_factory_t0(cls):
    '''
    Use only on envelopes that take a t0 as their first argument.
    This will construct envelopes at time zero and shift them, allowing
    for better caching.
    '''
    # I would like to rewrite this in a way that it modifies the class rather than
    # wrapping it in a factory, but I didn't see a clean way to do that -- ERJ
    @convertUnits(t0='ns')
    def factory(t0, *args, **kwargs):
        if not enable_caching or t0==0.0:   # If caching is disabled, this doesn't help, so keep it simple
            return cls(t0, *args, **kwargs) # Also, avoid extra shift if t0 is already zero.
        x = cls(0.0, *args, **kwargs)
        y = shift(x, t0)
        return y
    factory.__doc__ = 'Envelope factory for type %s.  Access original class via __klass__ attribute' % (cls.__name__,)
    factory.__klass__ = cls
    return factory

class EnvSum(Envelope):
    '''
    Helper class to support __add__ and __sub__
    '''
    def __init__(self, a, b):
        self.a = a
        self.b = b
        start, end = timeRange((self.a, self.b))
        Envelope.__init__(self, start, end)
    def timeFunc(self, t):
        return self.a.timeFunc(t) + self.b.timeFunc(t)
    def freqFunc(self, f):
        return self.a.freqFunc(f) + self.b.freqFunc(f)

class EnvProd(Envelope):
    '''
    Helper class to support __mul__, __div__, and __neg__.  Represents multiplication by a scalar
    '''
    def __init__(self, envIn, const):
        self.envIn = envIn
        self.const = const
        Envelope.__init__(self, envIn.start, envIn.end)
    def timeFunc(self, t):
        return self.envIn.timeFunc(t) * self.const
    def freqFunc(self, f):
        return self.envIn.freqFunc(f) * self.const

class EnvInv(Envelope):
    '''
    Helper class to support division of a scalar by an
    envelope. (__rdiv__).  I don't know why this is helpful -- ERJ
    '''
    def __init__(self, envIn):
        self.envIn = envIn
        Envelope.__init__(self, envIn.start, envIn.end)
    def timeFunc(self, t):
        return 1.0/self.envIn.timeFunc(t)
    def freqFunc(self, f):
        return 1.0/self.envIn.freqFunc(f)

class EnvConvFOnly(Envelope):
    '''
    Helper class that convolves an envelope with a smoothing function
    **in the frequency domain only**.  Convolution is expensive in the
    time domain, and hard to get right.  so we just pass that data
    through directly -- it is only used for plotting anyway.
    '''
    def __init__(self, env_in, env_filter):
        self.env_in = env_in
        self.env_filter = env_filter
    def timeFunc(self, t):
        return self.env_in.timeFunc(t)
    def freqFunc(self, f):
        return self.env_in.freqFunc(f) * self.env_filter.freqFunc(f)

# empty envelope
class EnvZero(Envelope):
    def timeFunc(self, t):
        return 0*t
    def freqFunc(self, f):
        return 0*f

NOTHING = EnvZero(start=None, end=None)
ZERO = EnvZero(start=0, end=0)

# @envelope_factory_t0
@memoize_envelope
class gaussian(Envelope):

    @convertUnits(t0='ns', w='ns', amp=None, phase=None, df='GHz')
    def __init__(self, t0, w, amp=1.0, phase=0.0, df=0.0):
        """
        A gaussian pulse with specified center and full-width at half max.
        @param t0, center time
        @param w, FWHM
        @param phase
        @param df, additional frequency
        """
        self.sigma = w / np.sqrt(8*np.log(2)) # convert fwhm to std. deviation
        self.t0 = t0
        self.amp = amp
        self.phase = phase
        self.df = df
        Envelope.__init__(self, start=t0-w, end=t0+w)

    def timeFunc(self, t):
        t0 = self.t0
        df = self.df
        return self.amp * np.exp(-(t-t0)**2/(2*self.sigma**2) - 2j*pi*df*(t-t0) + 1j*self.phase)

    def freqFunc(self, f):
        sigmaf = 1 / (2*np.pi*self.sigma) # width in frequency space
        ampf = self.amp * np.sqrt(2*np.pi*self.sigma**2) # amp in frequency space
        return ampf * np.exp(-(f+self.df)**2/(2*sigmaf**2) - 2j*np.pi*f*self.t0 + 1j*self.phase)

# @envelope_factory_t0
@memoize_envelope
class GaussianTrunc(NumericalPulse):
    @convertUnits(t0='ns', w='ns', Len=None, amp=None, phase=None, df='GHz')
    def __init__(self, t0, w, Len=None, amp=1.0, phase=0.0, df=0.0):
        """
        A gaussian pulse with specified center and full-width at half max,
        but truncated outside the envelope, and move to zero
        A * ( exp(-(t-t0)^2/(2 sigma^2)) - exp(-Len^2/(8 sigma^2) )
        @param t0: center time
        @param w: FWHM
        @param Len: the length of the pulse, default is None, means Len=2*w
        @param amp: amplitude of the gaussian pulse
        @param phase: phase of the gaussian pulse
        @param df: additional frequency of the gaussian pulse
        """
        self.sigma = w / np.sqrt(8*np.log(2)) # convert fwhm to std. deviation
        self.t0 = t0
        self.amp = amp
        self.phase = phase
        self.df = df
        if Len is None:
            Len = 2*w
        self.tlen = Len
        self.oft = np.exp(-self.tlen ** 2 / (8 * self.sigma ** 2))
        Envelope.__init__(self, start=t0-Len/2.0, end=t0+Len/2.0)

    def timeFunc(self, t):
        t0 = self.t0
        df = self.df
        tlen = self.tlen
        val = self.amp * (np.exp(-(t-t0)**2/(2*self.sigma**2)) - self.oft) / (1.0 - self.oft) * np.exp(-2j * np.pi * df * (t - t0) + 1j * self.phase)
        return val * (t>=self.start) * (t<self.end)

# @envelope_factory_t0
@memoize_envelope
class cosine(Envelope):
    @convertUnits(t0='ns', w='ns', amp=None, phase=None, df='GHz')
    def __init__(self, t0, w, amp=1.0, phase=0.0, df=0.0):
        """
        A cosine function centered at t0 with FULL WIDTH w
        @param t0, center time
        @param w, FullWidth of pulse, the time range of the envelope is [t0-w/2, t0+w/2)
        @param amp, amplitude
        @param df, additional frequency of cosine pulse
        @param phase
        """
        self.t0 = t0
        self.w = w
        self.amp = amp
        self.phase = phase
        self.df = df
        Envelope.__init__(self, t0-w/2.0, t0+w/2.0)
    def timeFunc(self, t):
        x = t - self.t0
        return self.amp * 0.5 * (1+np.cos(2*np.pi*x/self.w)) * \
               ((x+self.w/2.)>0) * ((-x+self.w/2.)>0) * \
               np.exp(1j*self.phase- 2j*pi*self.df*x)
    def freqFunc(self, f):
        wf = self.w*(f + self.df)
        a = 1.0 - wf
        b = 1.0 + wf
        return self.amp * np.exp(-2j*pi*f*self.t0+1j*self.phase)*self.w/4.0* \
               (2.0*np.sinc(wf)+np.sinc(a)+np.sinc(b))

# @envelope_factory_t0
@memoize_envelope
class cosineLambda(Envelope):
    @convertUnits(t0='ns', w='ns', amp=None, phase=None, lambdas=None)
    def __init__(self, t0, w, amp=1.0, phase=0.0,lambdas=()):
        """
        Cosine-like function: (1-cos(2*pi*n*(t-w/2)/w)/2,
        centered at t0 with FULL WIDTH w.
        Useful for mitigating parasitic rotations on other qubits. RB.
        @param t0, center time
        @param w, full width
        @param amp, amplitude
        @param phase
        @param, lambdas, parameters of the cosine-like function
        """
        self.t0 = t0
        self.w = w
        self.amp = amp
        self.phase = phase
        self.lambdas = (1.-np.sum(lambdas),) + lambdas #initialize first element
        Envelope.__init__(self, t0-w/2.0, t0+w/2.0)
    def timeFunc(self, t):
        envelope=0.
        for idx,lambdaValue in enumerate(self.lambdas):
            n=idx+1.
            envelope += lambdaValue * self.amp * 0.5 * (1-np.cos(2*np.pi*n*(t-self.t0-self.w/2)/self.w)) * (((t-self.t0)+self.w/2.)>0) * ((-(t-self.t0)+self.w/2.)>0) * np.exp(1j*self.phase)
        return envelope
    def freqFunc(self, f):
        envelope=0.
        for idx,lambdaValue in enumerate(self.lambdas):
            n=idx+1.
            wf = self.w*f
            a = n-wf
            b = n+wf
            envelope += lambdaValue*self.w/4.*(2.*np.sinc(wf)+ (-1)**(n-1)*(np.sinc(a)+np.sinc(b)))
        return self.amp*np.exp(-2j*np.pi*f*self.t0+1j*self.phase)*envelope

@memoize_envelope
class CosineHDBase(Envelope):
    """
    cosine pulse with drag, which is more efficient for caching.  Apply time and phase shifts
    separately to avoid caching issue
    """
    @convertUnits(w='ns', amp=None, drag=None)
    def __init__(self, w, amp=1.0, drag=0.0):
        """
        cosine pulse with drag, which is more efficient for caching.
        Apply time and phase shifts separately to avoid caching issue
        @param w: full width
        @param amp: amplitude
        @param drag: drag coefficient
        """
        cosenv = cosine(0.0, w, amp, 0.0)
        dragenv = 1j * drag * deriv(cosenv)
        self.env = cosenv + dragenv
        Envelope.__init__(self, -w/2.0, w/2.0)

    def timeFunc(self, t):
        return self.env.timeFunc(t)
    def freqFunc(self, f):
        return self.env.freqFunc(f)

@convertUnits(t0='ns', w='ns', amp=None, phase=None, drag=None)
def cosineHD(t0, w, amp=1.0, phase=0.0, drag=0.0):
    """
    This wraps the underlying CosineHD class with zero time and phase shifts,
    then does the shift at the end.  This reduces unnecessary redundant caching.
    @param t0
    @param amp
    @param phase
    @param drap
    """
    env = CosineHDBase(w, amp, drag)
    if phase:
        env = env*np.exp(1j*phase)
    if t0 or phase:
        env = shift(env, t0)
    return env

@memoize_envelope
class CosineWWFullBase(Envelope):

    @convertUnits(w='ns', amp=None)
    def __init__(self, w, amp=1.0, freqs=()):
        """
        Creates a cosine pulse with up to 3 notches in the frequency spectrum, at freqs RB.
        More efficient for caching.  Apply time and phase shifts
        separately to avoid caching issue.
        @param w
        @param amp
        """
        cosenv = cosine(0.0, w, amp, 0.0)
        if len(freqs)==0:
            alpha=0
            beta=0
            gamma=0
        elif len(freqs)==1:
            #DRAG
            w0=freqs[0]*(2*np.pi)
            alpha = 1. / w0
            beta=0
            gamma=0
        elif len(freqs)==2:
            #WAHWAH
            w0=freqs[0]*(2*np.pi)
            w1=freqs[1]*(2*np.pi)
            alpha = (w0+w1) / (w0*w1)
            beta  = -1. / (w0*w1)
            gamma = 0
        elif len(freqs)==3:
            w0=freqs[0]*(2*np.pi)
            w1=freqs[1]*(2*np.pi)
            w2=freqs[2]*(2*np.pi)
            alpha = (w0*w1+w0*w2+w1*w2) / (w0*w1*w2)
            beta  = -(w0+w1+w2) / (w0*w1*w2)
            gamma = -1. / (w0*w1*w2)
        else:
            raise Exception('only up to three notch frequencies')
        Xdot=deriv(cosenv)
        Xdotdot=deriv(Xdot)
        Xdotdotdot=deriv(Xdotdot)
        X = cosenv + beta *  Xdotdot
        Y = alpha * Xdot + gamma*Xdotdotdot
        self.env = X + 1j * Y
        Envelope.__init__(self, -w/2.0, w/2.0)
    def timeFunc(self, t):
        return self.env.timeFunc(t)
    def freqFunc(self, f):
        return self.env.freqFunc(f)

@convertUnits(t0='ns', w='ns', amp=None, phase=None)
def cosineWWFull(t0, w, amp=1.0, phase=0.0, freqs=()):
    """
    This wraps the underlying CosineWAHWAH class with zero time and phase shifts,
    then does the shift at the end.  This reduces unnecessary redundant caching.
    @param t0
    @param w
    @param amp
    @param phase
    @param freqs
    """
    env = CosineWWFullBase(w, amp, freqs)
    if phase:
        env = env*np.exp(1j*phase)
    if t0 or phase:
        env = shift(env, t0)
    return env

@memoize_envelope
class CosineLambdaHDBase(Envelope):
    @convertUnits(w='ns', amp=None, drag=None, lambdas=None)
    def __init__(self, w, amp=1.0, drag=0.0,lambdas=()):
        """
        cosine pulse with drag, which is more efficient for caching.  Apply time and phase shifts
        separately to avoid caching issue
        @param w
        @param amp
        @param drag
        @param lambdas
        """
        cosenv = cosineLambda(0.0, w, amp, 0.0,lambdas)
        dragenv = 1j * drag * deriv(cosenv)
        self.env = cosenv + dragenv
        Envelope.__init__(self, -w/2.0, w/2.0)
    def timeFunc(self, t):
        return self.env.timeFunc(t)
    def freqFunc(self, f):
        return self.env.freqFunc(f)

@convertUnits(t0='ns', w='ns', amp=None, phase=None, drag=None, lambdas=None)
def cosineLambdaHD(t0, w, amp=1.0, phase=0.0, drag=0.0,lambdas=()):
    """
    This wraps the underlying CosineHD class with zero time and phase shifts,
    then does the shift at the end.  This reduces unnecessary redundant caching.
    @param t0
    @param w
    @param amp
    @param phase
    @param drag
    @param lambdas
    """
    env = CosineLambdaHDBase(w, amp, drag, lambdas)
    if phase:
        env = env*np.exp(1j*phase)
    if t0 or phase:
        env = shift(env, t0)
    return env

# @envelope_factory_t0
@memoize_envelope
class triangle(Envelope):
    @convertUnits(t0='ns', tlen='ns', amp=None)
    def __init__(self, t0, tlen, amp, fall=True):
        """
        A triangular pulse, either rising or falling.
        @param t0, the start time
        @param tlen, length
        @param amp, amplitude
        @param fall, if True, the triangle is decrease to 0
        """
        self.t0 = t0
        self.tlen = tlen
        self.amp = amp
        if not fall:
            self.t0 = t0+tlen
            self.tlen = -tlen
            self.amp = -amp

        tmin = min(t0, t0+tlen)
        tmax = max(t0, t0+tlen)
        Envelope.__init__(self, tmin, tmax)

    def timeFunc(self, t):
        if self.tlen == 0 or self.amp == 0:
            return 0.0*t
        return self.amp * (t >= self.start) * (t < self.end) * (1 - (t-self.t0)/self.tlen)*np.sign(self.tlen)

    def freqFunc(self, f):
        if self.tlen == 0 or self.amp == 0:
            return 0.0*f
        # this is tricky because the fourier transform has a 1/f term, which blows up for f=0
        # the z array allows us to separate the zero-frequency part from the rest
        z = f == 0
        f = 2j*np.pi*(f + z)
        return self.amp * ((1-z)*np.exp(-f*self.t0)*(1.0/f - (1-np.exp(-f*self.tlen))/(f**2*self.tlen)) + z*self.tlen/2.0)

# @envelope_factory_t0
@memoize_envelope
class linear(Envelope):
    @convertUnits(t0='ns', tlen='ns', init=None, end=None)
    def __init__(self, t0, tlen, init=0.0, end=0.0):
        """
        line (t0, init) -> (t0+len, end)
        @param t0
        @param tlen
        @param init
        @param end
        """
        self.b = end
        self.a = init
        self.tlen = tlen
        self.t0 = t0
        tmin = min(t0, t0+tlen)
        tmax = max(t0, t0+tlen)
        Envelope.__init__(self, tmin, tmax)

    def timeFunc(self, t):
        t0, a, b = self.t0, self.a, self.b
        tlen = self.tlen
        y = ((b-a)/tlen * (t-t0) + a) * (t>=self.start) * (t<(self.end))
        return y

    def freqFunc(self, f):
        """similar trick in triangle"""
        tlen = self.tlen
        z = f == 0
        f = 2j*pi*(f+z)
        t0, a, b = self.t0, self.a, self.b
        a1 = (np.exp(-f*(t0+self.tlen))*(a-b-b*f*self.tlen) - np.exp(-f*t0)*(a-b-a*f*self.tlen))/(f**2*self.tlen)*(1-z)
        a2 = tlen/2*(a+b)*z
        return (a1 + a2)*np.sign(tlen)

# @envelope_factory_t0
@memoize_envelope
class rect(Envelope):
    @convertUnits(t0='ns', tlen='ns', amp=None, overshoot=None)
    def __init__(self, t0, tlen, amp, overshoot=0.0, overshoot_w=1.0):
        """A rectangular pulse with sharp turn on and turn off.

        Note that the overshoot_w parameter, which defines the FWHM of the gaussian overshoot peaks
        is only used when evaluating the envelope in the time domain.  In the fourier domain, as is
        used in the dataking code which uploads sequences to the boards, the overshoots are delta
        functions.
        @param t0
        @param tlen
        @param amp
        @param overshoot
        @param overshoot_w
        """
        self.t0 = t0
        self.amp = amp
        self.overshoot = overshoot * np.sign(amp) # overshoot will be zero if amp is zero
        tmin = min(t0, t0+tlen)
        tmax = max(t0, t0+tlen)
        self.tmid = (tmin + tmax) / 2.0
        self.tlen = tlen

        # to add overshoots in time, we create an envelope with two gaussians
        if overshoot:
            o_w = overshoot_w
            o_amp = 2*np.sqrt(np.log(2)/np.pi) / o_w # total area == 1
            self.o_env = gaussian(tmin, o_w, o_amp) + gaussian(tmax, o_w, o_amp)
        else:
            self.o_env = EnvZero(tmin, tmax)

        Envelope.__init__(self, tmin, tmax)
    def timeFunc(self, t):
        return (self.amp * (t >= self.start) * (t < self.end) +
                self.overshoot * self.o_env(t))

    # to add overshoots in frequency, use delta funcs (smoothed by filters)
    def freqFunc(self, f):
        return (self.amp * abs(self.tlen) * np.sinc(self.tlen*f) * np.exp(-2j*np.pi*f*self.tmid) +
                self.overshoot * (np.exp(-2j*np.pi*f*self.start) + np.exp(-2j*np.pi*f*self.end)))

# @envelope_factory_t0
@memoize_envelope
class flattop(Envelope):
    def _asdf__new__(cls, t0, tlen, w, amp=1.0, overshoot=0.0, overshoot_w=1.0, cache=None):
        """
        __new__ optimizes the case where amp=0 by constructing an EnvZero instance
        instead of a flattop.  This seems to happen unnecessarily often, so this saves memory,
        and maybe a bit of performance
        """
        if amp==0:
            return EnvZero(t0, t0+tlen)
        else:
            # __init__ will be called!
            return Envelope.__new__(cls)

    @convertUnits(t0='ns', tlen='ns', w='ns', amp=None)
    def __init__(self, t0, tlen, w, amp=1.0, overshoot=0.0, overshoot_w=1.0):
        """
        A rectangular pulse convolved with a gaussian to have smooth rise and fall.
        @param t0
        @param tlen
        @param w
        @param amp
        @param overshoot
        @param overshoot_w
        """
        self.t0 = t0
        self.tlen = tlen
        self.amp = amp
        self.w = w
        self.overshoot = overshoot * np.sign(amp) # overshoot will be zero if amp is zero
        tmin = min(t0, t0+tlen)
        tmax = max(t0, t0+tlen)
        Envelope.__init__(self, tmin-w, tmax+w)
    # to add overshoots in time, we create an envelope with two gaussians
        if overshoot:
            o_w = overshoot_w
            o_amp = 2*np.sqrt(np.log(2)/np.pi) / o_w # total area == 1
            self.o_env = gaussian(tmin, o_w, o_amp) + gaussian(tmax, o_w, o_amp)
        else:
            self.o_env = EnvZero(tmin-w, tmax+w)

    def timeFunc(self, t):
        a = 2*np.sqrt(np.log(2)) / self.w
        tmin = min(self.t0, self.t0+self.tlen)
        tmax = max(self.t0, self.t0+self.tlen)
        return (self.amp * (erf(a*(tmax - t)) - erf(a*(tmin - t)))/2.0 +
                self.overshoot * self.o_env(t))

    # to add overshoots in frequency, use delta funcs (smoothed by filters)
    def freqFunc(self, f):
        #
        #  This function calculates a gaussian convolved with a
        #  top-hat, and then adds overshoot delta functions.  I
        #  don't know why it does not convolve the overshoot with
        #  the gaussian.  -- ERJ
        rect_env = rect(self.t0, self.tlen, 1.0)
        kernel = gaussian(0, self.w, 2*np.sqrt(np.log(2)/np.pi) / self.w) # area = 1
        return (self.amp * rect_env(f, fourier=True) * kernel(f, fourier=True) + # convolve with gaussian kernel
                self.overshoot * (np.exp(-2j*np.pi*f*self.t0) + np.exp(-2j*np.pi*f*(self.t0+self.tlen))))

@convertUnits(t0='ns', tlen='ns')
def wait(t0, tlen):
    """
    zero from t0 to t0+tlen
    @param t0:
    @param tlen:
    """
    tmin=min(t0, t0+tlen)
    tmax=max(t0, t0+tlen)
    return EnvZero(start=tmin, end=tmax)

@convertUnits(t0='ns', rise='ns', hold='ns', fall='ns', amp=None)
def trapezoid(t0, rise, hold, fall, amp):
    """
    Create a trapezoidal pulse, built up from triangles and rectangles.
    @param t0
    @param rise
    @param rise
    @param hold
    @param fall
    @param amp
    """
    return (triangle(t0, rise, amp, fall=False) +
            rect(t0+rise, hold, amp) +
            triangle(t0+rise+hold, fall, amp))

# @envelope_factory_t0
@memoize_envelope
class iExp(Envelope):
    @convertUnits(t0='ns', f='GHz', amp=None, tlen='ns', phase='None')
    def __init__(self, t0, f, amp, tlen, phase=0):
        """
        amp*exp(2j*pi*f*(t-t0)+1j*phase) from t0 to t0+tlen
        tlen should >0
        @param t0
        @param f
        @param tlen
        @param phase
        """
        tmin = min(t0, t0+tlen)
        tmax = max(t0, t0+tlen)
        self.t0 = t0
        self.f = f
        self.amp = amp
        self.tlen = tlen
        self.phase = phase
        Envelope.__init__(self, tmin, tmax)

    def timeFunc(self, t):
        amp, f, t0, phase = self.amp, self.f, self.t0, self.phase
        return amp*np.exp(1j*2.0*pi*f*(t-t0)+1j*phase) * (t>=self.start) * (t<self.end)

    def freqFunc(self, x):
        amp, f, t0, phase = self.amp, self.f, self.t0, self.phase
        tlen = self.tlen
        p = -2*pi*x*(t0+tlen/2.0)+pi*f*tlen+phase
        # the sign(tlen) is for tlen<0
        return amp*tlen*sinc(tlen*(x-f))*exp(1j*p)*np.sign(tlen)

# @envelope_factory_t0
@memoize_envelope
class Cos(Envelope):
    @convertUnits(t0='ns', f='GHz', amp=None, tlen='ns', phase=None)
    def __init__(self, t0, f, amp, tlen, phase=0.0):
        """
        amp*cos(2*pi*f*(t-t0)+phase) from t0 to t0+tlen
        tlen > 0
        @param t0:
        @param f:
        @param amp:
        @param tlen:
        @param phase:
        """
        tmin = min(t0, t0+tlen)
        tmax = max(t0, t0+tlen)
        self.t0 = t0
        self.f = f
        self.amp = amp
        self.tlen = tlen
        self.phase = phase
        Envelope.__init__(self, tmin, tmax)

    def timeFunc(self, t):
        amp, f, t0, phase = self.amp, self.f, self.t0, self.phase
        return amp*cos(2*np.pi*f*(t-t0)+phase) * (t>=self.start) * (t<self.end)
    def freqFunc(self, x):
        amp, f, t0, phase = self.amp, self.f, self.t0, self.phase
        tlen = self.tlen
        p = pi*f*tlen+phase
        a = (sinc(tlen*(x-f))*exp(1j*p) + sinc(tlen*(x+f))*exp(-1j*p))
        return amp*tlen*exp(-2j*pi*x*(t0+tlen/2)) * a/2.0

# @envelope_factory_t0
@memoize_envelope
class Sin(Envelope):
    @convertUnits(t0='ns', f='GHz', amp=None, tlen='ns', phase=None)
    def __init__(self, t0, f, amp, tlen, phase=0.0):
        """
        amp*sin(2*pi*f*(t-t0)+phase) from t0 to t0+tlen
        tlen > 0
        @param t0:
        @param f:
        @param amp:
        @param tlen:
        @param phase:
        """
        tmin = min(t0, t0+tlen)
        tmax = max(t0, t0+tlen)
        self.t0 = t0
        self.f = f
        self.amp = amp
        self.tlen = tlen
        self.phase = phase
        Envelope.__init__(self, tmin, tmax)

    def timeFunc(self, t):
        amp, f, t0, phase = self.amp, self.f, self.t0, self.phase
        return amp*sin(2*np.pi*f*(t-t0)+phase) * (t>=self.start) * (t<self.end)
    def freqFunc(self, x):
        amp, f, t0, phase = self.amp, self.f, self.t0, self.phase
        tlen = self.tlen
        p = pi*f*tlen+phase
        a = (sinc(tlen*(x-f))*exp(1j*p) - sinc(tlen*(x+f))*exp(-1j*p))
        return -0.5j*amp*tlen*exp(-2j*pi*x*(t0+tlen/2)) * a

@convertUnits(t0='ns', amp=None, gamma='ns', tlen='ns', start='ns')
def lorentzian(t0, amp, gamma, tlen, start=None):
    """
    return amp*gamma**2 /( (t-t0)^2+gamma^2 )
    t0 is center point
    start is the start time,
    len is the length of pulse
    default is t0-4*gamma
    @param t0
    @param amp
    @param gamma
    @param tlen
    @param start
    """
    gamma = abs(gamma)
    if start is None:
        start = t0 - 4*gamma
    tmin = min(start, start+tlen)
    tmax = max(start, start+tlen)
    amp *= gamma**2
    def timeFunc(t):
        return amp * (t>=tmin) * (t<tmax) / ((t-t0)**2 + gamma**2)
    envL = NumericalPulse(tmin, tmax)
    envL.timeFunc = timeFunc
    return envL

class mix(Envelope):
    @convertUnits(df='GHz')
    def __init__(self, env, df=0.0, phase=0.0):
        """
        Apply sideband mixing at difference frequency df.
        @param env, envelope
        @param df, frequency shift
        @param phase
        """
        if df is None:
            raise Exception
        self.df = df
        self.phase = phase
        self.env = env
        Envelope.__init__(self, env.start, env.end)
    def timeFunc(self, t):
        return self.env(t) * np.exp(-2j*np.pi*self.df*t - 1.0j*self.phase)
    def freqFunc(self, f):
        return self.env(f + self.df, fourier=True)*np.exp(-1.0j*self.phase)

class deriv(Envelope):
    @convertUnits(dt='ns')
    def __init__(self, env, dt=0.01):
        """
        Get the time derivative of a given envelope.
        @param env, envelope
        @param dt
        """
        self.env = env
        self.dt = dt
        Envelope.__init__(self,env.start, env.end)
    def timeFunc(self, t):
        return (self.env(t+self.dt) - self.env(t-self.dt)) / (2*self.dt)
    def freqFunc(self, f):
        return 2j*np.pi*f * self.env(f, fourier=True)

class dragify(Envelope):
    @convertUnits(dt='ns')
    def __init__(self, env, alpha, dt):
        """
        env + alpha * d/dt(env)
        @param env:
        @param alpha:
        @param dt:
        """
        self.env = env
        self.alpha = alpha
        self.dt = dt
        Envelope.__init__(self, env.start, env.end)
    def freqFunc(self, f):
        return (1+2j*np.pi*f*self.alpha) * self.env(f, fourier=True)
    def timeFunc(self, t):
        return self.env(t) + self.alpha*(self.env(t+self.dt) - self.env(t-self.dt))/(2*self.dt)

class shift(Envelope):
    @convertUnits(dt='ns')
    def __init__(self, env, dt):
        """
        shift envelope by dt
        @param env: envelope
        @param dt:
        """
        self.dt = dt
        self.env = env
        Envelope.__init__(self, env.start+dt if env.start is not None else None,
                                env.end+dt if env.end is not None else None)
    def timeFunc(self, t):
        return self.env(t-self.dt)
    def freqFunc(self, f):
        return self.env(f, fourier=True) * exp(-2j*np.pi*f*self.dt)

# utility functions

def timeRange(envelopes):
    """Calculate the earliest start and latest end of a list of envelopes.

    Returns a tuple (start, end) giving the time range.  Note that one or
    both of start and end may be None if the envelopes do not specify limits.
    @param envelopes, a list of envelope
    """
    starts = [env.start for env in envelopes if env.start is not None]
    start = min(starts) if len(starts) else None
    ends = [env.end for env in envelopes if env.end is not None]
    end = max(ends) if len(ends) else None
    return start, end


def fftFreqs(time=1024):
    """Get a list of frequencies for evaluating fourier envelopes.

    The time is rounded up to the nearest power of two, since powers
    of two are best for the fast fourier transform.  Returns a tuple
    of frequencies to be used for complex and for real signals.
    """
    nfft = 2**int(math.ceil(math.log(time, 2)))
    f_complex = np.fft.fftfreq(nfft)
    f_real = f_complex[:nfft/2+1]
    return f_complex, f_real


def ifft(envelope, t0=-200, n=1000):
    f = np.fft.fftfreq(n)
    return np.fft.ifft(envelope(f, fourier=True) * np.exp(2j*np.pi*t0*f))


def fft(envelope, t0=-200, n=1000):
    t = t0 + np.arange(n)
    return np.fft.fft(envelope(t))


def plotFT(envelope, t0=-200, n=1000):
    t = t0 + np.arange(n)
    y = ifft(envelope, t0, n)
    plt.plot(t, np.real(y))
    plt.plot(t, np.imag(y))


def plotTD(envelope, t0=-200, n=1000):
    t = t0 + np.arange(n)
    y = envelope(t)
    plt.plot(t, np.real(y))
    plt.plot(t, np.imag(y))


def test_env(envelope):
    padtime = 200
    start = envelope.start
    end = envelope.end
    numpt = 2**int(np.ceil(np.log2(20*(end-start+2*padtime))))
    T = np.linspace(start-padtime, end+padtime, numpt)
    t0 = start-padtime
    dt = T[1] - T[0]
    f = np.fft.fftfreq(len(T), dt)
    envt = envelope(T)
    envt_fft = np.fft.fft(envt)*np.exp(-2j*np.pi*t0*f)
    envf = envelope(f, fourier=True)
    envf_ifft = np.fft.ifft(envf*np.exp(2j*np.pi*t0*f))
    plt.figure()
    plt.subplot(211)
    plt.plot(T, envt.real, 'b-')
    plt.plot(T, envt.imag, 'r-')
    plt.plot(T, envf_ifft.real/dt, 'b--', linewidth=3)
    plt.plot(T, envf_ifft.imag/dt, 'r--', linewidth=3)
    plt.legend(["Re[seq(t)]", "Im[seq(t)]", "Re[ifft(seq(f))]", "Im[ifft(seq(f))]"])
    plt.subplot(212)
    sf = np.fft.fftshift(f)
    plt.plot(sf, np.fft.fftshift(envf.real), 'b-')
    plt.plot(sf, np.fft.fftshift(envf.imag), 'r-')
    plt.plot(sf, np.fft.fftshift(envt_fft.real*dt), 'b--', linewidth=3)
    plt.plot(sf, np.fft.fftshift(envt_fft.imag*dt), 'r--', linewidth=3)
    plt.legend(["Re[seq(f)]", "Im[seq(f)]", "Re[fft(seq(t))]", "Im[fft(seq(t))]"])

