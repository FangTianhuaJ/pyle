from numpy import exp, log, pi, array, sqrt, ndarray, cos, ones_like, sin
from scipy.signal import slepian

from itertools import chain

class Pulse(object):
    def __init__(self, len=1, ofs=0):
        self.len = len
        self.ofs = ofs
    
    def __radd__(self, prev):
        self.ofs += prev.len
        return PulseSequence(chain(prev, self))
    
    def __ror__(self, prev):
        self.ofs += prev.len
        return PulseSequence(chain(prev, self))
    
    def __iter__(self):
        yield self

class PulseSequence(Pulse):
    def __init__(self, pulses):
        self.pulses = list(pulses)
    
    @property    
    def len(self):
        return sum(pulse.len for pulse in self.pulses)
        
    def __call__(self, t):
        return sum(pulse(t) for pulse in self.pulses)
        
    def __iter__(self):
        return iter(self.pulses)
        
class Delay(Pulse):
    def __call__(self, t):
        return 0 * t
        
class Square(Pulse):
    def __init__(self, amp=0, len=1, ofs=0):
        self.amp = amp
        self.len = len
        self.ofs = ofs
        
    def __call__(self, t):
        t = t - self.ofs
        return (0 <= t) * (t < self.len) * self.amp

class Trapezoid(Pulse):
    def __init__(self, amp=0, hold=1, rise=0, ofs=0):
        self.len = rise + hold + rise
        self.ofs = ofs
        self.amp = amp
        self.rise = rise
        self.hold = hold
        
    def __call__(self, t):
        t = t - self.ofs
        if self.rise == 0:
            return (0 <= t) * (t < self.len) * self.value
        t0 = 0
        t1 = self.rise
        t2 = self.rise + self.hold
        t3 = self.rise + self.hold + self.rise
        return self.amp * ((t0 <= t) * (t < t1) * t / self.rise +
                           (t1 <= t) * (t < t2) +
                           (t2 <= t) * (t < t3) * (t3 - t) / self.rise)
        
class Gaussian(Pulse):
    LOG16 = log(16)
    
    def __init__(self, amp=0, phase=0, len=10, fwhm=5, df=0, ofs=0):
        self.len = len
        self.ofs = ofs
        self.amp = amp * exp(1j*phase)
        self.fwhm = fwhm
        self.sigmasq = fwhm**2 / Gaussian.LOG16
        self.df = df
        
    def __call__(self, t):
        t = t - self.ofs
        t0 = 0
        t1 = self.len/2
        t2 = self.len
        return (t0 <= t) * (t < t2) * self.amp * exp(-(t-t1)**2 / self.sigmasq) * exp(-2j*pi*self.df*t)

class GaussianNormalizedHD(Pulse):
    LOG16 = log(16)
    LOG256 = log(256)
    
    def __init__(self, amp=0, phase=0, len=10, fwhm=5, df=0, ofs=0, Delta=0, alpha=0.5):
        self.len = len
        self.ofs = ofs
        self.amp = amp * exp(1j*phase)
        self.fwhm = fwhm
        self.sigmasq = fwhm**2 / GaussianNormalizedHD.LOG256
        self.df = df
        self.Delta = Delta
        self.alpha = alpha
        
    def __call__(self, t):
        alpha = self.alpha
        t = t - self.ofs
        t0 = 0
        t1 = self.len/2
        t2 = self.len
        return (1.0/sqrt(2*pi*self.sigmasq)) \
              *((t0 <= t) * (t < t2) * self.amp * exp(-(t-t1)**2 / (2.0 * self.sigmasq)) * exp(-2j*pi*self.df*t) \
              + (t0 <= t) * (t < t2) * self.amp * exp(-(t-t1)**2 / (2.0 * self.sigmasq)) * exp(-2j*pi*self.df*t) \
              * (-(t-t1)/self.sigmasq) * alpha*(-1j/self.Delta))
        
class Gaussian_HD(Pulse):
    LOG16 = log(16)
    
    def __init__(self, amp=0, phase=0, len=10, fwhm=5, df=0, ofs=0, Delta=0, alpha=0.5):
        self.len = len#Delta should be given in angular freq and negative.
        self.ofs = ofs
        self.amp = amp * exp(1j*phase)
        self.fwhm = fwhm
        self.sigmasq = fwhm**2 / Gaussian.LOG16
        self.df = df
        self.Delta = Delta
        self.alpha = alpha
        
    def __call__(self, t):
        alpha = self.alpha
        t = t - self.ofs
        t0 = 0
        t1 = self.len/2
        t2 = self.len
        return  ((t0 <= t) * (t < t2) * self.amp * exp(-(t-t1)**2 / self.sigmasq) * exp(-2j*pi*self.df*t) \
              + (t0 <= t) * (t < t2) * self.amp * exp(-(t-t1)**2 / self.sigmasq) * exp(-2j*pi*self.df*t) \
              * (-2*(t-t1)/self.sigmasq) * alpha*(-1j/self.Delta))

class Cos(Pulse):
    LOG16 = log(16)
    
    def __init__(self, amp=0, phase=0, len=10, df=0, ofs=0):
        self.len = len
        self.ofs = ofs
        self.amp = amp * exp(1j*phase)
        self.df = df
        
    def __call__(self, t):
        t = t - self.ofs
        tf = self.len
        return (0 <= t) * (t < tf) * self.amp * (1-cos(2*pi*t/tf))/2 * exp(-2j*pi*self.df*t)

class Cos_HD(Pulse):
    LOG16 = log(16)
    
    def __init__(self, amp=0, phase=0, len=10, df=0, ofs=0, Delta=0):
        self.len = len#Delta should be given in angular freq and negative.
        self.ofs = ofs
        self.amp = amp * exp(1j*phase)
        self.df = df
        self.Delta = Delta
        
    def __call__(self, t):
        t = t - self.ofs
        tf = self.len
        return ((0 <= t) * (t < tf) * self.amp * (1-cos(2*pi*t/tf))/2 * exp(-2j*pi*self.df*t)
                + (0 <= t) * (t < tf) * self.amp * sin(2*pi*t/tf) * exp(-2j*pi*self.df*t) * (pi/tf) * (-1j/2/self.Delta))
#                - (0 <= t) * (t < tf) * self.amp * (1-cos(2*pi*t/tf)) * exp(-2j*pi*self.df*t) * (pi*self.df) * (-1j/2/self.Delta))

class Constant(Pulse):
    def __init__(self, amp=0):
        self.len = 0
        self.ofs = 0
        self.amp = amp
        
    def __call__(self, t):
        return self.amp * ones_like(t)

class Nothing(Pulse):
    def __init__(self):
        self.len = 0
        self.ofs = 0
        self.amp = 0
        
    def __call__(self, t):
        return self.amp * ones_like(t)