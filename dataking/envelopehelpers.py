import numpy as np

import pyle.envelopes as env
from pyle.dataking import utilMultilevels as ml

import util
import labrad.units as U
from math import ceil

# sequence helpers
# these are simple functions for building standard control envelopes by
# pulling out appropriate configuration parameters for a particular qubit.

def power2amp(power):
    """
    Convert readout uwave power in to DAC amplitude

    This function contains several hardcoded values and should be reworked.
    """
    assert power.isCompatible('dBm'), 'Power must be put in dBm.'
    power = power['dBm']
    v = 10**((power-10)/20)
    # dac_amp = 1 corresponds to 400mV
    return  2*v/0.4

def fourNsCeil(t):
    """
    this function convert t to ceil(t/4)*4 to make sure
    the output t is aligned at 4 ns.
    the time resolution of ADC when we define the demodulation windows is 4ns
    @param t: labrad.units, should be a time value
    @return: ceil(t/4)*4
    """
    return ceil(t['ns']/4.0)*4*U.ns

def mix(q, seq, freq=None, state=None):
    """Apply microwave mixing to a sequence.
    This mixes to a particular frequency from the carrier frequency.
    Also, adjusts the microwave phase according to the phase calibration.

    PARAMETERS
    q: Qubit dictionary.
    seq - eh functions: Pulses to mix with microwaves.
    freq - string: Registry key indicating desired frequency of post-mix
           pulse (e.g., 'f10','f21').
    state - scalar: Which qubit frequency is desired for post-mix pulse
            (e.g., 1 gives f10, 2 gives f21).
    """
    if freq is not None and state is not None:
        raise Exception('state and freq are not orthogonal parameters for mixing')
    if isinstance(freq, str):
        freq = q[freq]
    if freq is None:
        if state is None:
            state=1
        freq = ml.getMultiLevels(q,'frequency',state)
    return env.mix(seq, freq - q['fc']) * np.exp(1j*q['uwavePhase'])


# xy rotations with half-derivative term on other quadrature
def piPulseHD(q, t0, phase=0, alpha=None, state=1, length='piLen', df='piDetune'):
    """Pi pulse using a gaussian envelope with half-derivative Y quadrature."""
    return rotPulseHD(q, t0, angle=np.pi, phase=phase, state=state, length=length, alpha=alpha, df=df)

def piHalfPulseHD(q, t0, phase=0, alpha=None, state=1, length='piHalfLen', df='piHalfDetune'):
    """Pi/2 pulse using a gaussian envelope with half-derivative Y quadrature."""
    # if piHalfAmp for state is not exist, use piAmp/2, else use piHalfAmp
    piamp = ml.getMultiLevels(q, 'piAmp', state)
    keyName = ml.multiLevelKeyName('piHalfAmp', state)
    pihalfamp = q.get(keyName, 0.5*piamp) # if piHalfAmp does not exist, default is half of piamp
    r = pihalfamp/piamp
    return rotPulseHD(q, t0, angle=np.pi*r, phase=phase, state=state, length=length, alpha=alpha, df=df)

def rotPulseHD(q, t0, angle=np.pi, phase=0, alpha=None, state=1, length='piLen', df=None):
    """Rotation pulse using a gaussian envelope with half-derivative Y quadrature.

    This also allows for an arbitrary pulse length. The new length must be defined as a key in the registry.
    """
    if alpha is None:
        alpha = float(q['alphaDRAG'])
    # Eliminate DRAG for higher order pulses
    if state>1: alpha = 0
    #Get the pi amplitude. getMultiLevels() ensures that the correct key is read regardless of which state is desired.
    #Note in particular that old code, which does not explicitly set state, and therefore gets the default value of 1,
    #will get 'piAmp', as desired.
    piamp = ml.getMultiLevels(q,'piAmp',state)
    r = angle / np.pi
    if df is None:
        df_val = 0*U.GHz
    else:
        df_val = q.get(df, 0*U.GHz)
    delta = 2*np.pi * (q['f21'] - q['f10'])['GHz'] + 2*np.pi*(df_val['GHz'])
    phase_detune = -2*np.pi*df_val['GHz']*(t0['ns'])
    # x = env.gaussian(t0, w=q[length], amp=piamp*r, phase=phase)
    # x = env.GaussianTrunc(t0, w=q[length], amp=piamp*r, phase=phase)
    x = env.cosine(t0, w=q[length], amp=piamp*r, phase=phase)
    y = -alpha * env.deriv(x) / delta
    return env.mix(x + 1j*y, df_val, phase=phase_detune)
    # return x + 1j*y

def rabiPulseHD(q, t0, len, w=None, amp=None, overshoot=0.0, overshoot_w=1.0, alpha=None, state=1):
    """Rabi pulse using a flattop envelope with half-derivative Y quadrature."""
    if alpha is None:
        alpha = float(q['alphaDRAG'])
    # Eliminate DRAG for higher order pulses
    if state>1: alpha = 0
    #Get the pi amplitude. getMultiLevels() ensures that the correct key is read regardless of which state is desired.
    #Note in particular that old code, which does not explicitly set state, and therefore gets the default value of 1,
    #will get 'piAmp', as desired.
    if amp is None:
        amp = ml.getMultiLevels(q,'piAmp',state)
    if w is None:
        w=q['piFWHM']
    delta = 2*np.pi * (q['f21'] - q['f10'])['GHz']
    x = env.flattop(t0, len, w, amp, overshoot, overshoot_w)
    y = -alpha * env.deriv(x) / delta
    return x + 1j*y

# z rotations
def piPulseZ(q, t0):
    """Pi pulse using a gaussian envelope."""
    return rotPulseZ(q, t0, angle=np.pi)

def piHalfPulseZ(q, t0):
    """Pi/2 pulse using a gaussian envelope."""
    piamp = q['piAmpZ']
    pihalfamp = q.get('piHalfAmpZ', 0.5*piamp)
    r = pihalfamp/piamp
    return rotPulseZ(q, t0, angle=np.pi*r)

def rotPulseZ(q, t0, angle=np.pi):
    """Rotation pulse using a gaussian envelope."""
    r = angle / np.pi
    # return env.gaussian(t0, w=q['piFWHMZ'], amp=q['piAmpZ']*r)
    return env.GaussianTrunc(t0, w=q['piFWHMZ'], amp=q['piAmpZ']*r)

# default pulse type is half-derivative
piPulse = piPulseHD
piHalfPulse = piHalfPulseHD
rotPulse = rotPulseHD
rabiPulse = rabiPulseHD


def spectroscopyPulse(q, t0, df=0):
    dt = q['spectroscopyLen']
    amp = q['spectroscopyAmp']
    return env.mix(env.flattop(t0, dt, w=q['piFWHM'], amp=amp), df)


def measurePulse(q, t0, state=1):
    """Add a measure pulse for the desired state.

    PARAMETERS
    q: Qubit dictionary.
    t0 - value [us]: Time to start the measure pulses.
    state - scalar: Which state's measure pulse to use.
    """
    return env.trapezoid(t0, 0, q['measureLenTop'], q['measureLenFall'], ml.getMultiLevels(q,'measureAmp',state))


def measurePulse2(q, t0):
    return env.trapezoid(t0, 0, q['measureLenTop2'], q['measureLenFall2'], q['measureAmp2'])


def readoutPulse(q, t0):
    tlen = q['readoutLen']
    amp = power2amp(q['readoutPower'])
    df = q['readoutFrequency'] - q['readoutDevice']['carrierFrequency']
    w = q['readoutWidth']
    return env.mix(env.flattop(t0, tlen, w=w, amp=amp), df)

def readoutPulseRingup(q, t0):
    tlen = q['readoutLen']
    amp = power2amp(q['readoutPower'])
    df = q['readoutFrequency'] - q['readoutDevice']['carrierFrequency']
    w = q['readoutWidth']
    emphFactor = float(q['readoutRingupFactor'])
    emphLen = q['readoutRingupLen']
    rr = env.flattop(t0, tlen, w=w, amp=amp) + (emphFactor-1.0)*env.flattop(t0, emphLen, w=w, amp=amp)
    return env.mix(rr, df, q.get('readoutPulsePhase', 0.0))

def readoutRingdown(q, t0, tlen=None, amp=None, phase=None):
    if tlen is None:
        tlen = q['readoutRingdownLen']
    if amp is None:
        amp = q['readoutRingdownAmp']
    if phase is None:
        phase = q['readoutRingdownPhase']
    df = q['readoutFrequency'] - q['readoutDevice']['carrierFrequency']
    rrd = env.flattop(t0, tlen, w=q['readoutWidth'], amp=amp*np.exp(1j*phase))
    return env.mix(rrd, df)

def boostState(q, t0, state, alpha=None):
    """Excite the qubit to the desired state, concatenating pi pulses as needed.

    PARAMETERS
    q: Qubit dictionary.
    t0 - value [ns]: Time to start the pulses (center of first pi pulse).
    state - scalar: State to which qubit should be excited.
    """
    xypulse = env.NOTHING
    for midstate in range(state):
        xypulse = xypulse + mix(q, piPulse(q, t0+midstate*q['piLen'], state=(midstate+1), alpha=alpha), state=(midstate+1))
    return xypulse



# sequence corrections

def correctCrosstalkZ(qubits):
    """Adjust the z-pulse sequences on all qubits for z-xtalk."""
    biases = [q.get('z', env.NOTHING) for q in qubits]
    for q in qubits:
        coefs = list(q['calZpaXtalkInv'])
        q['z'] = sum(float(c) * bias for c, bias in zip(coefs, biases))

    # dual block
    biases = [q.get('z_s')[0] for q in qubits]
    for q in qubits:
        coefs = list(q['calZpaXtalkInv'])
        q['z_s'][0] = sum(float(c) * bias for c, bias in zip(coefs, biases))

    if max( [len(q.get('z_s', [])) for q in qubits]) > 1:
        biases = [q.get('z_s')[1] for q in qubits]
        for q in qubits:
            coefs = list(q['calZpaXtalkInv'])
            q['z_s'][1] = sum(float(c) * bias for c, bias in zip(coefs, biases))
