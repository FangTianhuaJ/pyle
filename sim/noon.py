import time

import numpy as np
from numpy import pi

from matplotlib import pyplot as plt

import quantsim as sim
import qctools as qct
import sequences as seq

# TODO: allow for different coupling strengths (makes timing more difficult)
# TODO: tune up pi-pulses (DRAG?) and swap pulses (overshoots?)


fq0 = 6.5
fr0 = 5.95

fq1 = 6.5
fr1 = 6.05

frc = 6.8

fc = 6.5 # carrier frequency

noon = 2

# all times in ns, all frequencies in GHz
S = 0.03 # qubit-resonator coupling

# qubits
q0 = sim.Qubit3(T1=500, T2=300, nonlin=-0.2)
r0 = sim.Resonator(T1=3000, T2=10e3, n=noon+1, df=fr0 - fc)

q1 = sim.Qubit3(T1=500, T2=300, nonlin=-0.2)
r1 = sim.Resonator(T1=3000, T2=10e3, n=noon+1, df=fr1 - fc)

rc = sim.Resonator(T1=3000, T2=10e3, n=2, df=frc - fc)

# couplers
c0 = sim.Coupler(q0, r0, s=S)
c1 = sim.Coupler(q1, r1, s=S)
cc0 = sim.Coupler(q0, rc, s=S)
cc1 = sim.Coupler(q1, rc, s=S)

# the full quantum system
system = sim.QuantumSystem([q0, q1, r0, r1, rc], [c0, c1, cc0, cc1])
systemc = sim.QuantumSystem([q0, q1, rc], [cc0, cc1])
system0 = sim.QuantumSystem([q0, r0], [c0])
system1 = sim.QuantumSystem([q1, r1], [c1])
system01 = sim.QuantumSystem([q0, r0, q1, r1], [c0, c1])

# detuning frequencies (all calculated w.r.t carrier)
dfc = frc - fc
dfr0 = fr0 - fc
dfr1 = fr1 - fc

dfc0 = frc - fc
dfidle0 = fq0 - fc
dfnoon0 = fr0 - q0.nonlin - fc
dfswap0 = fr0 - fc
dfmeas0 = fr0 - 0.4 - fc

dfc1 = frc - fc
dfidle1 = fq1 - fc
dfnoon1 = fr1 - q1.nonlin - fc
dfswap1 = fr1 - fc
dfmeas1 = fr1 - 0.4 - fc

# fwhm for z sweeps and uw pulses
fwhmsw = 1
fwhm = 5

# pulse time and swap times
dtp = 12 # microwave pulse time (should be a little more than 2*fwhm
dtswap01 = 1/S / 2.0 #+ fwhmsw*4
dtswap12 = dtswap01 / np.sqrt(2) #+ fwhmsw*4
dtswap01n = lambda n: dtswap01 / np.sqrt(n) # swap on 01 with n photons
dtswap12n = lambda n: dtswap12 / np.sqrt(n) # swap on 12 with n photons

# microwave pulse amplitudes
Api01 = np.sqrt(np.log(2)/pi) / fwhm
Api12 = Api01 / np.sqrt(2)

# control signals
params = [
    (dtp,              dfidle0, Api01, 0,           dfidle1, 0,     0,         ), # pi pulse
    (dtswap01/2+0.5,   dfc0,    0,     0,           dfidle1, 0,     0,         ), # entangle q0 with rc
    (dtswap01,         dfnoon0, 0,     0,           dfc1,    0,     0,         ), # swap rc with q1
]

for i in range(1,noon):
    params += [
    (dtp,              dfidle0, Api12, q0.nonlin,   dfidle1, Api12, q1.nonlin, ), # pump 1 -> 2
    (dtswap12n(i),     dfnoon0, 0,     0,           dfnoon1, 0,     0,         ), # swap into target resonators
    ]

params += [
    (dtswap01n(noon),  dfswap0, 0,     0,           dfswap1, 0,     0,         ), # transfer entanglement to resonators
    (dtswap01,         dfmeas0, 0,     0,           dfmeas1, 0,     0,         ), # idle
    (dtswap01,         dfmeas0, 0,     0,           dfmeas1, 0,     0,         ), # one more idle
]

dts, df0, amp0, duw0, df1, amp1, duw1 = zip(*params)

t = [sum(dts[:i]) for i in range(len(dts)+1)]

def dffunc(dfs, dfidle):
    #return dfidle + sum(seq.flattop(ti, tf-ti, fwhmsw, df-dfidle, overshoot=0) for ti, tf, df in zip(t[:-1], t[1:], dfs))
    #return sum(seq.rect(ti, tf-ti, df) for ti, tf, df in zip(t[:-1], t[1:], dfs))
def uwfunc(dfs, amps, duws):
    # hack: added detuning for 01 pi-pulse only.  this should be a separate param
    return sum(seq.gaussian(t+dtp/2.0, fwhm, amp, 0, df+duw+(0.01 * (1 if duw==0 else 0))) for t, df, amp, duw in zip(t, dfs, amps, duws))

q0.df = dffunc(df0, dfidle0)
q1.df = dffunc(df1, dfidle1)

q0.uw = uwfunc(df0, amp0, duw0)
q1.uw = uwfunc(df1, amp1, duw1)


# entangle qubits
rhoc = systemc.ket([0,0,0])
rhosc = []
tsc = range(int(t[3]))
for i in tsc:
    print 'c:', i,
    start = time.time()
    rhoc = systemc.simulate(rhoc, [i,i+1])[-1]
    rhosc.append(rhoc)
    end = time.time()
    print 'elapsed: %g' % (end - start)
rho_q0 = systemc.partial([0], rhosc[-1])
rho_q1 = systemc.partial([1], rhosc[-1])

print 'q0:', rho_q0[1,1]
print 'q1:', rho_q1[1,1]

# pump resonator 0
rho_0 = qct.ket2rho(system0.ket([0,0]))
rho_r0 = system0.partial([1], rho_0)

rho0 = qct.tensor([rho_q0, rho_r0])
rhos0 = []
ts0 = range(int(t[3]), int(t[-2]))
for i in ts0:
    print '0:', i,
    start = time.time()
    rho0 = system0.simulate(rho0, [i,i+1])[-1]
    rhos0.append(rho0)
    end = time.time()
    print 'elapsed: %g' % (end - start)
    
# pump resonator 1
rho_1 = qct.ket2rho(system1.ket([0,0]))
rho_r1 = system1.partial([1], rho_0)

rho1 = qct.tensor([rho_q1, rho_r1])
rhos1 = []
ts1 = range(int(t[3]), int(t[-2]))
for i in ts1:
    print '1:', i,
    start = time.time()
    rho1 = system1.simulate(rho1, [i,i+1])[-1]
    rhos1.append(rho1)
    end = time.time()
    print 'elapsed: %g' % (end - start)


# plot
fmin = min(dfmeas0 + q0.nonlin, dfmeas1 + q1.nonlin) + fc - 0.2
fmax = frc + 0.2

ts = np.linspace(t[0], t[-2], 400)

fig = plt.figure(figsize=(10,6))
fig.subplots_adjust(left=0.03, right=0.98, bottom=0.03, top=0.97)

def add_vlines(ax):
    for tb in t[1:-2]:
        ax.axvline(tb, linestyle=':', color='gray')

def add_title(ax, title):
    ax.text(0, 1, '\n  '+title, ha='left', va='top', transform=ax.transAxes)

# plot detunings
ax1 = fig.add_subplot(7, 1, 1)
ax1.plot(ts, q0.df(ts) + fc, 'b-', label='q0: 01')
ax1.plot(ts, q0.df(ts) + q0.nonlin + fc, 'b:', label='q0: 12')
ax1.plot(ts, q1.df(ts) + fc, 'r-', label='q1: 01')
ax1.plot(ts, q1.df(ts) + q1.nonlin + fc, 'r:', label='q1: 12')
ax1.axhline(fc, linestyle=':', color='gray')
ax1.axhline(frc, linestyle='--', color='gray')
ax1.axhline(fr0, linestyle='--', color='b')
ax1.axhline(fr1, linestyle='--', color='r')
ax1.set_ylim((fmin, fmax))
ax1.legend()
add_title(ax1, 'freq')
add_vlines(ax1)

# plot microwaves
ax2 = fig.add_subplot(7, 1, 2, sharex=ax1)
ax2.plot(ts, q0.uw(ts).real, 'b-', label='q0: I')
ax2.plot(ts, q0.uw(ts).imag, 'b:', label='q0: Q')
ax2.plot(ts, q1.uw(ts).real, 'r-', label='q1: I')
ax2.plot(ts, q1.uw(ts).imag, 'r:', label='q1: Q')
ax2.legend()
add_title(ax2, 'uwave')
yrng = max(abs(y) for y in ax2.get_ylim())
ax2.set_ylim(-yrng, yrng)
add_vlines(ax2)

# calculate reduced density matrices
rhos_rc = systemc.partial([2], rhosc)
diags_rc = np.array([np.diag(rho) for rho in rhos_rc])

rhos_q0ent = systemc.partial([0], rhosc)
diags_q0ent = np.array([np.diag(rho) for rho in rhos_q0ent])
rhos_q0 = system0.partial([0], rhos0)
diags_q0 = np.array([np.diag(rho) for rho in rhos_q0])

rhos_res0 = system0.partial([1], rhos0)
diags_res0 = np.array([np.diag(rho) for rho in rhos_res0])

rhos_q1ent = systemc.partial([1], rhosc)
diags_q1ent = np.array([np.diag(rho) for rho in rhos_q1ent])
rhos_q1 = system1.partial([0], rhos1)
diags_q1 = np.array([np.diag(rho) for rho in rhos_q1])

rhos_res1 = system1.partial([1], rhos1)
diags_res1 = np.array([np.diag(rho) for rho in rhos_res1])

# plot coupling res
ax3 = fig.add_subplot(7, 1, 3, sharex=ax1)
for i in range(rc.n):
    ax3.plot(tsc, diags_rc[:,i], label=str(i))
ax3.legend()
ax3.set_ylim(0,1)
add_title(ax3, 'rc')
add_vlines(ax3)

# plot qubit 0
ax4 = fig.add_subplot(7, 1, 4, sharex=ax1)
for i in range(q0.n):
    x = np.hstack((tsc, ts0))
    y = np.hstack((diags_q0ent[:,i], diags_q0[:,i]))
    ax4.plot(x, y, label=str(i))
ax4.legend()
ax4.set_ylim(0,1)
add_title(ax4, 'q0')
add_vlines(ax4)

# plot resonator 0
ax5 = fig.add_subplot(7, 1, 5, sharex=ax1)
for i in range(r0.n):
    ax5.plot(ts0, diags_res0[:,i], label=str(i))
ax5.legend()
ax5.set_ylim(0,1)
add_title(ax5, 'r0')
add_vlines(ax5)

# plot qubit 1
ax6 = fig.add_subplot(7, 1, 6, sharex=ax1)
for i in range(q1.n):
    x = np.hstack((tsc, ts1))
    y = np.hstack((diags_q1ent[:,i], diags_q1[:,i]))
    ax6.plot(x, y, label=str(i))
ax6.legend()
ax6.set_ylim(0,1)
add_title(ax6, 'q1')
add_vlines(ax6)

# plot resonator 1
ax7 = fig.add_subplot(7, 1, 7, sharex=ax1)
for i in range(r1.n):
    ax7.plot(ts1, diags_res1[:,i], label=str(i))
ax7.legend()
ax7.set_ylim(0,1)
add_title(ax7, 'r1')
add_vlines(ax7)

plt.show()

