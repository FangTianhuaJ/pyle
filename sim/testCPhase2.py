import numpy as np
from numpy import pi

from matplotlib import pyplot as plt

import quantsim as sim
from qctools import trace_distance, fidelity
from sequences import Trapezoid, Delay, Gaussian, Cos, Square, Constant, Cos_HD, Gaussian_HD

# all times in ns, all frequencies in GHz

# qubits
q0 = sim.Qubit3(T1=513, T2=204, nonlin=-0.200)
q1 = sim.Qubit3(T1=504, T2=234, nonlin=-0.200)
qubits = [q0, q1]

# couplers
c = sim.Coupler(q0, q1, s=0.0142)
couplers = [c]

# the full quantum system
system = sim.QuantumSystem(qubits, couplers)

# control signals
dt = 1
#df = 0.200

q0.uw = Gaussian_HD(0.098, fwhm=10, len=20, Delta=2*pi*q0.nonlin)
q0.df = 0

q1.uw = Gaussian_HD(0.098, fwhm=10, len=20, Delta=2*pi*q1.nonlin, df=0.200)
q1.df = Delay(20) + Square(-0.10, len=200) + Constant(0.100)

# run simulation
psi0 = system.ket('00')
T = np.arange(0, 220, 1, dtype=float)
rho = system.simulate(psi0, T)

# compare final state and measured final state to ideal target state
rho_final = rho[20]

psi_ideal = system.ket('11')
rho_ideal = np.outer(psi_ideal.conj(), psi_ideal)

print 'actual trace_distance:', trace_distance(rho_final, rho_ideal)
print 'actual fidelity:', fidelity(rho_final, rho_ideal)

# plot time traces
plt.figure()
for i in range(system.n):
    plt.plot(T, rho[:,i,i])
    plt.hold(1)
plt.plot(T, [np.trace(rho[i]) for i in range(len(rho))], ':')

plt.xlabel('time [ns]')
plt.legend(['%d%d' % (i, j) for i in range(q0.n) for j in range(q1.n)])
plt.title('Simultaneous pi-pulses')

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

# plot density matrices
#plt.figure(figsize=(6,6))
#for i, data in enumerate([rho_final.real, rho_final.imag,
#                          rho_ideal.real, rho_ideal.imag]):
#    plt.subplot(2,2,i+1)
#    plt.pcolor(data)
#    plt.colorbar()

plt.show()