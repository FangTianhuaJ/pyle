import numpy as np
from numpy import pi

import pyle.tomo

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import quantsim as sim
from qctools import trace_distance, fidelity
from sequences import Trapezoid, Delay, Gaussian, Cos, Square, Gaussian_HD, GaussianNormalizedHD

from pyle.dataking.benchmarking import danBench as db

def rho2bloch(rho):
    sigmas = [pyle.tomo.sigmaX, pyle.tomo.sigmaY, pyle.tomo.sigmaZ]
    return np.array([np.trace(np.dot(rho, sigma)) for sigma in sigmas])

def plotTrajectory(ax, rhosIn):
    #If these are three level qubits, get rid of the |2> contribution
    if rhosIn[0].shape[0]==3:
        rhos = np.array([rho[:-1,:-1] for rho in rhosIn])
    else:
        rhos = rhosIn[:]
    blochs = np.real(np.array([rho2bloch(rho) for rho in rhos]))
    ax.plot(blochs[:,0],blochs[:,1],blochs[:,2], markersize=10)

def drawBlochSphere(bloch = True, labels = True): # takes a sequence and projects the trajectory of the qubit onto the bloch sphere, works for T1/T2!
    fig = plt.figure()
    ax = Axes3D(fig)
    #draw a bloch sphere
    if bloch:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z,  rstride=5, cstride=5, color = '0.8', alpha = 0.9)#, cmap = cm.PuBu) #, alpha = 1.0)
        ax.set_aspect('equal')
        
    #Axis label
    if labels:
        ax.text(0, 0, 1, '|0>')
        ax.text(0, 0, -1, '|1>')
        ax.text(1, 0, 0, '|0> + |1>')
        ax.text(-1, 0, 0, '|0> - |1>')
        ax.text(0, 1, 0, '|0> + i|1>')
        ax.text(0, -1, 0, '|0> - i|1>')
        
    #Axis
    ax.plot([-1.2,1.2],[0,0],[0,0], color = '0.1', linewidth=2)
    ax.plot([0,0],[-1.2,1.2],[0,0], color = '0.1', linewidth=2)
    ax.plot([0,0],[0,0],[-1.2,1.2], color = '0.1', linewidth=2)
    ax.set_aspect('equal')
    
    return fig,ax

def piPulseGaussian(fwhm, delta, T1, T2):

    # qubits
    q0 = sim.Qubit3(T1=T1, T2=T2, nonlin=delta)
    qubits = [q0]

    # the full quantum system
    system = sim.QuantumSystem(qubits)

    # apply a pi-pulse
    lngth = fwhm*4
    amp = (1.0/(2.0*fwhm))*np.sqrt(np.log(16)/np.pi)
    q0.uw = Gaussian(amp=amp, fwhm=fwhm, len=lngth, ofs=0)

    # run simulation
    psi0 = system.ket('0')
    T = np.arange(0, lngth+10, 0.1, dtype=float)
    rhos = system.simulate(psi0, T, method='other')
    
    return (T,q0.uw,q0.z),rhos

def piPulseHD(fraction, fwhm, delta, T1, T2, df=0):

    # qubits
    q0 = sim.Qubit3(T1=T1, T2=T2, nonlin=delta)
    qubits = [q0]

    # the full quantum system
    system = sim.QuantumSystem(qubits)

    # apply a pi-pulse
    fwhm = 8
    lngth = fwhm*4
    q0.uw = GaussianNormalizedHD(amp=0.5*fraction, fwhm=fwhm, len=lngth, ofs=0, df=df, Delta=2*np.pi*delta)

    # run simulation
    psi0 = system.ket('0')
    T = np.arange(0, lngth+10, 0.1, dtype=float)
    rhos = system.simulate(psi0, T, method='other')
    
    return (T,q0.uw,q0.df),rhos

def plotBlochSphere(rhosIn, ax):
    #Drop |2> components from density matrix
    rhos = np.array([np.array(rho[:-1,:-1]) for rho in rhosIn])
    #Convert to bloch vectors
    blochs = np.array([db.qubitMatrix2Vector(rho) for rho in rhos])
    #plot
    ax.plot(blochs[:,0], blochs[:,1], blochs[:,2],'.')
    
    
# # plot time traces
# plt.figure()
# for i in range(system.n):
    # plt.plot(T, rhos[:,i,i])
    # plt.hold(1)
# plt.plot(T, np.trace(rhos, axis1=1, axis2=2), ':')

# plt.xlabel('time [ns]')
# plt.legend(['%d' % i for i in range(q0.n)])
# plt.title('T1 decay')

# # plot control sequences
# plt.figure(figsize=(6,5))
# uw = np.array([qubit.uw(T) for qubit in qubits]).T
# df = np.array([qubit.df(T) for qubit in qubits]).T
# rng = max(np.amax(abs(uw)), np.amax(abs(df)), 0.01)
# for i in range(system.m):
    # plt.subplot(system.m,1,i+1)
    # plt.plot(T, uw[:,i].real, T, uw[:,i].imag, T, -df[:,i])
    # plt.ylabel('q%i' % i)
    # plt.ylim(-rng*1.1, rng*1.1)
    # plt.legend(('X', 'Y', 'Z'))
# plt.xlabel('time [ns]')

# plt.show()
