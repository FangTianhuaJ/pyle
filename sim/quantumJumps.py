import numpy as np
from numpy import fft, pi
from numpy import random
from scipy import interpolate

from matplotlib import pyplot as plt

import integrate

def generate_noise():
    freqs = np.linspace(1,2**10,2**10)
    ft = 10 * 1/freqs * np.exp(2j*np.pi * random.random(freqs.shape))
    ft = np.r_[[0], ft, ft[:-1][::-1].conj()]
    noise = fft.ifft(ft).real
    noise_func = interpolate.interp1d(np.arange(0, 2**11, 1), noise)
    return noise_func

Sz = np.array([[0, 0], [0, 1]], dtype=complex)
Sx = np.array([[0, 1], [1, 0]], dtype=complex)
Sy = np.array([[0, -1j], [1j, 0]], dtype=complex)

rabi_f = 0.01

def dpsi(psi, t, noise_func):
    f = noise_func(t)
    H = 2*pi*rabi_f * Sx + 2*pi*f * Sz
    return -1j*np.dot(H, psi)

psi0 = np.array([1,0])

allpsis = []
for i in range(10):
    print 'run %d' % i,
    psis = integrate.psideint(dpsi, psi0, np.arange(0, 1024, 1), args=(generate_noise(),))
    allpsis.append(psis)
    print 'done.'
