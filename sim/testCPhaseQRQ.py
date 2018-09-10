import numpy as np
from numpy import pi

from matplotlib import pyplot as plt

import quantsim as sim
from qctools import trace_distance, fidelity
from sequences import Trapezoid, Delay, Gaussian, Cos, Square, Constant, Cos_HD, Gaussian_HD

# all times in ns, all frequencies in GHz

fq0 = 6.8
fq1 = 6.8
fr0 = 6.5

fc = 6.8 #carrier frequency

S = 0.05 # qubit-resonator coupling

# qubits
q0 = sim.Qubit3(T1=500, T2=200, nonlin=-0.200)
q1 = sim.Qubit3(T1=500, T2=200, nonlin=-0.200)
rc = sim.Resonator(T1=3000, T2=10e3, n=3, df=fr0-fc)
qubits = [q0, q1]

# couplers
cc0 = sim.Coupler(q0, rc, s=S)
cc1 = sim.Coupler(q1, rc, s=S)
couplers = [cc0,cc1]

# the full quantum system
#system = sim.QuantumSystem(qubits, couplers)
system = sim.QuantumSystem([q0,q1,rc],[cc0,cc1])
#detuning frequencies (all calculated w.r.t. carrier)
dfr0 = fr0 - fc

# control signals
dt = 1.0
#df = 0.200

q0.uw = Gaussian_HD(0.096, fwhm=10, len=20, Delta=2*pi*q0.nonlin)
#q0.df = 0
q0.df = Delay(20) + Square(-0.30, len =10) + Delay(14) + Square(-0.3, len=10)

q1.uw = Gaussian_HD(0.096, fwhm=10, len=20, Delta=2*pi*q0.nonlin) 
#q1.df = 0
q1.df = Delay(30) + Square(-0.10, len =14) 


# run simulation
psi0 = system.ket('000')
T = np.arange(0, 220, 1, dtype=float)
rho = system.simulate(psi0, T)

# compare final state and measured final state to ideal target state
#rho_final = rho[20]

#psi_ideal = system.ket('11')
#rho_ideal = np.outer(psi_ideal.conj(), psi_ideal)

#print 'actual trace_distance:', trace_distance(rho_final, rho_ideal)
#print 'actual fidelity:', fidelity(rho_final, rho_ideal)

# plot time traces
plt.figure()
for i in range(system.n):
    plt.plot(T, rho[:,i,i])
    plt.hold(1)
plt.plot(T, [np.trace(rho[i]) for i in range(len(rho))], ':')

plt.xlabel('time [ns]')
plt.legend(['%d%d%d' % (i, j, k) for i in range(q0.n) for j in range(q1.n) for k in range(rc.n)])
plt.title('Simul Pi then swaps')

# plot control sequences
plt.figure(figsize=(6,5))
uw = np.array([qubit.uw(T) for qubit in qubits]).T
df = np.array([qubit.df(T) for qubit in qubits]).T
rng = max(np.amax(abs(uw)), np.amax(abs(df)), 0.01)
for i in range(len(qubits)):
    plt.subplot(len(qubits),1,i+1)
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