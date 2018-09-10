import numpy as np
from labrad.units import Unit, Value

GHz, MHz = Unit("GHz"), Unit("MHz")

fitFunc = lambda x, M, fmax, offset, fc: (fmax + fc) * np.sqrt(np.abs(np.cos(np.pi * M * np.abs(x - offset)))) - fc


def freqDiffToAmp(q, freqDifference):
    """
    returns the amplitude to bring about a frequency difference. freqDifference in labrad units
    """
    freqDifference = freqDifference['GHz']
    ztoffunc = AmpToFrequency(q)
    ftozfunc = FrequencyToAmp(q)
    f10 = ztoffunc(0.0)
    return ftozfunc(freqDifference + f10)


def AmpToFrequency(q, params=None):
    """
    Returns the function z -> f (f in units of GHz, no labrad units)
    """
    if params is None:
        params = q.get('calZpaFunc', None)
    func = lambda x: AmpToFrequencyFunc(params, x)
    return func


def AmpToFrequencyFunc(params, amp):
    M = params[0]
    fmax = params[1]
    offset = params[2]
    fc = params[3]
    freq = fitFunc(amp, M, fmax, offset, fc)
    return freq


def FrequencyToAmp(q, params=None):
    """
    Returns the function f -> z (f in units of GHz, no labrad units)
    """
    if params is None:
        params = q.get('calZpaFunc', None)

    # func = lambda f: (M*offset*np.pi + np.arccos( (f+fc)**2/(fc+fmax)**2 ) ) /(M*np.pi)
    func = lambda f: FrequencyToAmpFunc(params, f)
    return func


def FrequencyToAmpFunc(params, freq):
    if isinstance(freq, Value):
        freq = freq['GHz']
    M = params[0]
    fmax = params[1]
    offset = params[2]
    fc = params[3]
    amp = np.arccos((freq + fc)**2 / (fc + fmax)**2) / (M * np.pi) + offset
    return amp
