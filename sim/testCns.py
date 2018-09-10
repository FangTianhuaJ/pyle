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
qubits = [q0, q1]

# couplers
c = sim.Coupler(q0, q1, s=0.020)
couplers = [c]

# the full quantum system
system = sim.QuantumSystem(qubits, couplers)

# control signals
dt = 15
df = -0.100

q0.uw = Cos(0.2, phase=pi/2, len=10, df=df)
q0.df = df

q1.uw = Cos(0.2, phase=pi/2, len=10)
q1.df = 0

# run simulation
psi0 = system.ket('00')
T = np.arange(0, 100, 0.5, dtype=float)
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
plt.figure(figsize=(6,6))
for i, data in enumerate([rho_final.real, rho_final.imag,
                          rho_ideal.real, rho_ideal.imag]):
    plt.subplot(2,2,i+1)
    plt.pcolor(data)
    plt.colorbar()

plt.show()