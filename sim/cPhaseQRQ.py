# hack: setup python path to get local pyle
import os
import sys
sys.path.insert(0,os.path.join('..','pyle'))

from math import pi

import numpy as np
import matplotlib.pyplot as plt

from pyle.sim import quantsim as sim
from pyle.sim import ket

T1 = 100e-9, 500e-9, 500e-9
T2 = 300e-9, 300e-9, 300e-9

coupling = [[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]]
qubit_detuning = 0, 0, 0
uwave_drive = 0, 0, 0

psi0 = ket('100')
T = np.arange(0, 100e-9, 1e-9, dtype=float)

qs = QuantumSystem(T1, T2, coupling, qubit_detuning, uwave_drive)
rho = qs.sim(psi0, T)

from pylab import *
figure()
for i in range(8):
    plot(T*1e9, rho[:,i,i])
    hold(1)

xlabel('time [ns]')
legend(['000', '001', '010', '011', '100', '101', '110', '111'])
title('T1 Decay')

show()