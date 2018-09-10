import numpy as np
from numpy import pi

from matplotlib import pyplot as plt

import quantsim as sim
from qctools import trace_distance, fidelity
from sequences import Trapezoid, Delay, Gaussian, Cos, Square

# qubits
q0 = sim.Qubit3(T1=500, T2=300)
qubits = [q0]

# the full quantum system
system = sim.QuantumSystem(qubits)

# apply a pi-pulse
q0.uw = Gaussian(0.1, len=10)

# run simulation
psi0 = system.ket('0')
T = np.arange(0, 500, 0.5, dtype=float)
rhos = system.simulate(psi0, T)

# plot time traces
plt.figure()
for i in range(system.n):
    plt.plot(T, rhos[:,i,i])
    plt.hold(1)
plt.plot(T, np.trace(rhos, axis1=1, axis2=2), ':')

plt.xlabel('time [ns]')
plt.legend(['%d' % i for i in range(q0.n)])
plt.title('T1 decay')

# plot control sequences
plt.figure(figsize=(6,5))
uw = np.array([qubit.uw(T) for qubit in qubits]).T
df = np.array([qubit.df(T) for qubit in qubits]).T
rng = max(np.amax(abs(uw)), np.amax(abs(df)), 0.01)
for i in range(system.m):
    plt.subplot(system.m,1,i+1)
    plt.plot(T, uw[:,i].real, T, uw[:,i].imag, T, -df[:,i])
    plt.ylabel('q%i' % i)
    plt.ylim(-rng*1.1, rng*1.1)
    plt.legend(('X', 'Y', 'Z'))
plt.xlabel('time [ns]')

plt.show()