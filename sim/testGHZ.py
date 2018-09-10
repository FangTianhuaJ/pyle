import numpy as np
from numpy import pi

from matplotlib import pyplot as plt

import quantsim as sim
from qctools import trace_distance, fidelity
from sequences import Trapezoid, Delay, Gaussian, Cos, Square

# all times in ns, all frequencies in GHz

# qubits
q0 = sim.Qubit3(T1=500, T2=300)
q1 = sim.Qubit3(T1=500, T2=300)
q2 = sim.Qubit3(T1=500, T2=300)
qubits = [q0, q1, q2]

# couplers
c01 = sim.Coupler(q0, q1, s=0.020)
c02 = sim.Coupler(q0, q2, s=0.020)
c12 = sim.Coupler(q1, q2, s=0.020)
couplers = [c01, c02, c12]

# the full quantum system
system = sim.QuantumSystem(qubits, couplers)

# control signals
dt = 15
df = -0.100

q0.uw = Cos(0.1, phase=pi/2, len=10) + Delay(15) + Cos(0.1, phase=0, len=10)
q0.df = 0

q1.uw = Cos(0.1, phase=pi/2, len=10) + Delay(15) + Cos(0.1, phase=0, len=10)
q1.df = 0

q2.uw = Cos(0.1, phase=pi/2, len=10) + Delay(15) + Cos(0.1, phase=0, len=10)
q2.df = 0

# run simulation
psi0 = system.ket('000')
T = np.arange(0, 100, 0.5, dtype=float)
rhos = system.simulate(psi0, T)

# compare final state and measured final state to ideal target state
rho_final = rhos[70]

psi_ideal = (system.ket('000') + system.ket('111')) / np.sqrt(2)
rho_ideal = sim.ket2rho(psi_ideal) #np.outer(psi_ideal.conj(), psi_ideal)

print 'actual trace_distance:', trace_distance(rho_final, rho_ideal)
print 'actual fidelity:', fidelity(rho_final, rho_ideal)

# plot time traces
plt.figure()
for i in range(system.n):
    plt.plot(T, rhos[:,i,i])
    plt.hold(1)
plt.plot(T, np.trace(rhos, axis1=1, axis2=2), ':')

plt.xlabel('time [ns]')
plt.legend(['%d%d%d' % (i, j, k) for i in range(q0.n) for j in range(q1.n) for k in range(q2.n)])
plt.title('GHZ state generation')

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
plt.figure(figsize=(6,6))
for i, data in enumerate([rho_final.real, rho_final.imag,
                          rho_ideal.real, rho_ideal.imag]):
    plt.subplot(2,2,i+1)
    plt.pcolor(data)
    plt.colorbar()

plt.show()