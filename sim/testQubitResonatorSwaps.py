import numpy as np
from numpy import pi

from matplotlib import pyplot as plt

import quantsim as sim
from qctools import trace_distance, fidelity
from sequences import Trapezoid, Delay, Gaussian, Cos, Square

# all times in ns, all frequencies in GHz

# qubits
q = sim.Qubit2(T1=500, T2=300)
r = sim.Resonator(T1=500, T2=300, n=5, df=0.200)

# couplers
c = sim.Coupler(q, r, s=0.100)

# the full quantum system
system = sim.QuantumSystem([q, r], [c])

# control signals
def interact(df, dt):
    q.uw = Cos(0.2, phase=pi/2, len=10)
    q.df = Delay(10) + Trapezoid(df, hold=dt, rise=2)
    
    psi0 = system.ket('00')
    T = np.arange(0, q.df.len, 0.5)
    rhos = system.simulate(psi0, T)
    rho_qs = system.partial(rhos, [0]) # reduced density matrix of qubit only
    p1 = np.real(rho_qs[-1, 1, 1])
    #rhof = rhos[-1]
    #indices = [system.index('1%d' % i) for i in range(r.n)]
    #p1 = np.real(sum(rhof[i,i] for i in indices))
    print 'df=', df, 'dt=', dt, 'p1=', p1
    return p1

dfmin, dfmax = 0.0, 0.4
dtmin, dtmax = 0, 100

dfs = np.linspace(dfmin, dfmax, 100)
dts = np.linspace(dtmin, dtmax, 100)

fig = plt.figure(figsize=(6,6))

p1 = None
for i, dt in enumerate(dts[:-1]):
    p1row = np.array([interact(df, dt) for df in dfs[:-1]])
    p1 = np.vstack((p1, [p1row])) if p1 is not None else np.array([p1row])
    
    X, Y = np.meshgrid(dfs, dts[:i+2])
    
    fig.clf()
    ax = fig.add_subplot(111)
    im = ax.pcolor(X, Y, p1)
    ax.set_xlim(dfmin, dfmax)
    ax.set_ylim(dtmin, dtmax)
    fig.colorbar(im)
    plt.draw()
    plt.show()
