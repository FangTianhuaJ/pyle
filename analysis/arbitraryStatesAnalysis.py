'''
Created back in the day ('09?)
author: Max Hofheinz and/or probably Matthew Neeley
Recovered by Erik Lucero May 2011

Needs some retouching to get aligned with current pyle release.
Looks useful for Resonator measurement analysis
'''

import numpy as np
from numpy import sin, cos, dot, sqrt, size, pi, exp, sign, log, argmax, sum, asarray
from scipy.linalg import eig
from scipy.optimize import leastsq

from pyle.util.sweeptools import PQ
from pyle.envelopes import gaussian, flattop

from pyle.sim.JaynesCummings import Ujc, plotdensity
import stateAnalysis as states
from wignerAnalysis import pure

class qubit:
    operateBias = PQ(180, 'mV')
    measAmplitude = -1.246
    resonanceFrequency = PQ(6.0, 'GHz')
    resonatorFrequency = PQ(6.5,'GHz')
    swapLength = PQ(17,'ns')

    anritsuFrequency = resonatorFrequency
    #anritsuFrequency = 0.5 * (resonanceFrequency + resonatorFrequency - \
    #                              PQ(sqrt((resonanceFrequency - resonatorFrequency)['GHz']**2 + (0.5/swapLength['ns'])**2),'GHz'))

    measBias      = PQ(500, 'mV')
    measDelay     = PQ(0, 'ns')
    measLength    = PQ(10, 'ns')

    cutoff = PQ(25.3, 'mus')
    is1bigger = True

    reset = dict(
        bias1 = PQ(1.15, 'V'),
        bias2 = PQ(0.9, 'V'),
        count = 1
    )


    piLength = PQ(8,'ns')
    piAmplitude = 0.5/(piLength['ns']/2*sqrt(pi/log(2)))
    delayAfterPi = PQ(12,'ns')
    uwaveLength = PQ(1000,'ns')
    uwaveAmplitude = 1.0
    swapAmplitude = 0.5


def swappulse(state, angle, phase=1.0, conj=False):
    N = np.shape(state)[1]
    matrix = np.zeros((2,N,2,N),dtype=complex)
    n = np.arange(N)
    matrix[1,n,1,n] = cos(0.5*angle*sqrt(n+1))
    matrix[0,n,0,n] = cos(0.5*angle*sqrt(n))
    n = n[:-1]
    matrix[1,n,0,n+1] = -1j * phase * sin(0.5*angle*sqrt(n+1))
    matrix[0,n+1,1,n] = -1j * np.conjugate(phase) * sin(0.5*angle*sqrt(n+1))
    matrix = np.reshape(matrix,(2*N,2*N))
    if conj:
        return np.reshape(np.conjugate(dot(np.conjugate(np.reshape(state, 2*N)), matrix)), (2,N))
    else:
        return np.reshape(dot(matrix, np.reshape(state, 2*N)), (2,N))


def drivepulse(state, angle, phase=1.0, conj=False):
    N = np.shape(state)[1]
    matrix = np.zeros((2,N,2,N), dtype=complex)
    n = np.arange(N)
    phase *= angle/abs(angle)
    angle = abs(angle)
    matrix[0,n,0,n] = cos(0.5*angle)
    matrix[1,n,1,n] = cos(0.5*angle)
    matrix[0,n,1,n] = -1j * np.conjugate(phase) * sin(0.5*angle)
    matrix[1,n,0,n] = -1j * phase * sin(0.5*angle)
    matrix = np.reshape(matrix,(2*N,2*N))
    if conj:
        return np.reshape(np.conjugate(dot(np.conjugate(np.reshape(state, 2*N)), matrix)), (2,N))
    else:
        return np.reshape(dot(matrix,np.reshape(state, 2*N)), (2,N))


def phasepulse(state, angle, conj=False):
    result = 1.0*state
    if conj:
        angle = -angle
    result[1,:] *= exp(1j*angle)
    return result


def sequence_paper(state):
    """Calculate the sequence to produce an arbitrary state using the algorithm from PRL 76 1055"""
    n = size(state)-1
    while state[n] == 0:
        n -= 1
    state = state[np.newaxis,:] * np.array([1,0])[:,np.newaxis]
    swaptimes = np.zeros(n, dtype=float)
    swapphases = 1.0j*swaptimes
    drivetimes = 1.0*swaptimes
    drivephases = 1.0j*drivetimes
    states.printstate(state)
    while n > 0:
        n -= 1
        if state[1,n] == 0:
            swaptimes[n] = pi/sqrt(n+1)
            swapphases[n] = 1
        else:
            ratio = state[0,n+1] / state[1,n]
            swapphases[n] = np.conjugate(1j*ratio/abs(ratio))
            swaptimes[n] = 2*np.arctan(abs(ratio))/sqrt(n+1)
        state = swappulse(state, swaptimes[n], swapphases[n], conj=True)
        states.printstate(state)
        if state[0,n] == 0:
            drivetimes[n] = pi
            drivephases[n] = 1
        else:
            ratio = state[1,n]/state[0,n]
            drivephases[n] = 1j*ratio/abs(ratio)
            drivetimes[n] = 2*np.arctan(abs(ratio))
        state = drivepulse(state, drivetimes[n], drivephases[n], conj=True)
        states.printstate(state)
    return swaptimes, swapphases, drivetimes, drivephases


def sequence(state, verbose=False, qubit=None, driveAdjust=-1.0j, swapAdjust=0,
             visualization=states.printstate, filenameiter=None):
    """Calculate the sequence to produce an arbitrary state of an HO according to PRL 76 1055. The algorithm is adapted to our system: We can't adjust the phase of the qubit resonator coupling but we can dephase ground and excited state of the qubit with a z pulse. If qubit is a qubit object, calculate the excact pulse sequence, given the qubit parameters."""
    def NoneIter():
        while True:
            yield None

    if filenameiter is None:
        filenameiter = NoneIter()
    state = asarray(state).astype(complex)
    state /= sqrt(sum(abs(state**2)))
    n = size(state)-1
    while state[n] == 0:
        n -= 1
    state = state[np.newaxis,:] * np.array([1,0])[:,np.newaxis]
    swaptimes = np.zeros(n,dtype=float)
    zpulses = 1.0*swaptimes
    drives = 1.0j*swaptimes
    if verbose:
        visualization(state,'',filename=filenameiter.next())
    while n > 0:
        n -= 1

        # phase shift between |g,n> and |e,n-1> and swap
        if state[1,n] == 0:
            time = pi/sqrt(n+1)
            phase = 0
        else:
            time = 1j*state[0,n+1]/state[1,n]
            phase = -np.arctan2(np.imag(time), np.real(time))
            oldstate = state
            state = phasepulse(state,phase,conj=True)
            if verbose:
                visualization(state,'phase', phase, oldstate=oldstate,
                              filename=filenameiter.next())
                visualization(state, filename=filenameiter.next())
            time = np.arctan(abs(time))
            time = (time + pi * (time<0)) * 2 / sqrt(n+1)
        oldstate = state
        state = swappulse(state, time, conj=True)
        swaptimes[n] = time
        zpulses[n] = phase

        if verbose:
            visualization(state,'swap', time, oldstate=oldstate,
                          filename=filenameiter.next())
            visualization(state, filename=filenameiter.next())
        #drive
        if state[0,n] == 0:
            if state[1,n] == 0:
                drive = 0
            else:
                drive = pi * 1j*state[1,n]/abs(state[1,n])
        else:
            ratio = state[1,n]/state[0,n]
            drive = np.arctan(abs(ratio))
            drive = (drive + pi * (drive < 0)) * 1j*ratio/abs(ratio)*2
        oldstate = state
        state = drivepulse(state, drive, conj=True)
        drives[n] = drive
        if verbose:
            visualization(state, 'drive', drive, oldstate=oldstate,
                          filename=filenameiter.next())
            visualization(state, filename=filenameiter.next())
    zpulses[1:] = 1.0*zpulses[:-1]
    zpulses[0] = 0
    if qubit is None:
        return drives, swaptimes, zpulses
    else:
        detuning = 2*pi*(qubit.resonanceFrequency['GHz'] - \
            qubit.resonatorFrequency['GHz'])
        sbfreq = 2*pi*(qubit.resonanceFrequency['GHz'] - \
            qubit.anritsuFrequency['GHz'])

        swapTimes = (qubit.swapLength['ns']+swapAdjust)*swaptimes/pi

        offresTimes = 2*qubit.delayAfterPi['ns'] # minimum time off resonance
        offresTimes = offresTimes + \
            ((-sign(detuning)*(zpulses - offresTimes * detuning)) % (2*pi))/abs(detuning)
        print offresTimes
        swapStarts = 1.0 * offresTimes
        swapStarts[1:] += (offresTimes + swapTimes)[:-1].cumsum()
        drivepulseCenter = swapStarts - 0.5*offresTimes
        n = np.arange(size(swapTimes))
        def flux(t):
            result = 0
            for i in n:
                result = result + flattop(swapStarts[i],swapTimes[i],
                                             amplitude=qubit.swapAmplitude,w=0.0)(t)
            return result
        def uwave(t):
            result = 0
            for i in n:
                result += gaussian(drivepulseCenter[i], qubit.piLength['ns'])(t) * \
                                  qubit.piAmplitude/pi*drives[i] * \
                    driveAdjust * exp(-1.0j*sbfreq*(t-swapStarts[i]))
            return result
        return uwave, flux


def simulate(qubit, uwave, flux, t,n=4):
    return Ujc(t, fHO=qubit.resonatorFrequency['GHz'],
               fTLS=qubit.resonanceFrequency['GHz'] + flux(t),
               fDrive=qubit.anritsuFrequency['GHz'],
               driveTLS=uwave(t), coupling=0.5/qubit.swapLength['ns'],
               initial=n, T1HO=5e3, T1TLS=5e3, T2TLS=5e3)


def optimizer(state, qubit, t):
    n = size(state)
    def errfunc(arg):
        print arg
        u, f = sequence(state, qubit=qubit,
                       driveAdjust=arg[0]+1.0j*arg[1],
                       swapAdjust=arg[2])
        result = simulate(qubit, u, f, t, n=n)[-1,:,:]
        result[0:n,0:n] -= pure(state)
        return abs(np.reshape(result, size(result)))
    fit = leastsq(errfunc, np.array([0, -1.0, 3.0]), epsfcn=0.01)
    fit = fit[0]
    print fit
    return sequence(state, qubit=qubit, driveAdjust = fit[0]+1.0j*fit[1],
                       swapAdjust=fit[2])


def optimizer2(state, qubit, t, iterations=10):
    n = size(state)
    corstate = 1.0*state
    for _ in np.arange(iterations):
        corstate /= sqrt(sum(corstate**2))
        print corstate
        u,f = sequence(corstate, qubit=qubit)
        result = simulate(qubit, u, f, t, n=n)
        plotdensity(t,result)
        newstate = closestpure(result[-1,0:n,0:n])
        corstate -= 0.1*(newstate - state * sum(np.conjugate(state)*newstate)/sqrt(sum(state**2)*sum(newstate**2)))
    corstate /= sqrt(sum(corstate**2))
    return corstate, result


def test(swaptimes=None, drivetimes=None):
    if swaptimes == None or drivetimes == None:
        swaptimes = 2*np.array([1.556,0.7283,-0.8322,0.7594,0.6553,-0.6211,-0.5879,0.5554,0.5236,0.4967])
    drivetimes = 2*np.array([1.2874,-1.1748,-1.3744,1.4900,1.2629,-1.2475,-1.2549,-1.1702,1.5708,1.5708])
    state = np.zeros((2,11))
    state[0,0] = 1
    states.printstate(state)
    for n in np.arange(size(swaptimes)):
        print 'drive angle: %g' % drivetimes[n]
        state = drivepulse(state, drivetimes[n], 1.0)
        states.printstate(state)
        print 'swap angle: %g' % swaptimes[n]
        state = swappulse(state, swaptimes[n], 1.0)
        states.printstate(state)


def closestpure(rho):
    val, vec = eig(rho)
    i = argmax(val)
    vec[:,i] /= sqrt(dot(np.conjugate(vec[:,i]), vec[:,i]))
    return vec[:,i]

