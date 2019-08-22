import numpy as np
from numpy import pi
from matplotlib import pyplot as plt
import pylesim.plotting as pyplt
from scipy import optimize
from zzwpylesim import envelopes as env
from zzwpylesim.envelopes import Envelope
from matplotlib.pyplot import figure
import zzwpylesim.quantsim as sim
from scipy.interpolate import interp1d
import time
import math
alpha = 0.7
nonlin = -0.24

q0 = sim.Qubit3(T1=10e9, T2=10e9, nonlin=nonlin)

system = sim.QuantumSystem([q0])

psi0 = np.array([1+0j,0,0])

def cosinex(t0=10,amp=1.0/20,w=20.0,phase=0.0,df=0.0):
    def timeFunc(t):
        x = t - t0
        return amp * 0.5 * (1+np.cos(2*np.pi*x/w)) * ((x+w/2.)>0) * ((-x+w/2.)>0) * \
            np.exp(1j*phase- 2j*np.pi*df*x)
    return Envelope(timeFunc, start=t0-w/2, end=t0+w/2)

deltaf = 0.001507537688442211
T0 = np.arange(0,20,0.1)
cosine = cosinex(t0=10, w=20, amp=1.0/20, phase=0.0, df=0)
q0.uw = cosine-1j*(alpha/(2*np.pi*nonlin))*env.deriv(cosine)
q0.df = 0.001507537688442211
rhos0 = system.simulate(psi0, T0, method='fast')
rhos = rhos0[:,0:2,0:2]
P0 = rhos[:, 0][:, 0]
P1 = rhos[:, 1][:, 1]
P2 = rhos0[:, 2][:, 2]


def findAlpha(init=0.0, end=2, step=0.01, count=0, error=0.001):
    for alpha in np.arange(init,end,step):
        count += 1
        q0.uw = cosine-1j*(alpha/(2*np.pi*nonlin))*env.deriv(cosine) 
        rhos0a = system.simulate(psi0, T0, method='fast')
        P2a = rhos0a[:, 2][:, 2]

        alpha1 = alpha+step
        q0.uw = cosine-1j*(alpha1/(2*np.pi*nonlin))*env.deriv(cosine) 
        rhos0a1 = system.simulate(psi0, T0, method='fast')
        P2a1 = rhos0a1[:, 2][:, 2]

        if np.real(P2a1[-1])-np.real(P2a[-1]) < 0:
            while np.real(P2a[-1]) > error:
                print 'count = ', count
                print 'alpha = ', alpha
                print 'P2a = ', np.real(P2a[-1])
                time.sleep(0.5)
                break
        else:
            break
    print 'alpha_final = ', alpha
    print 'P2a_final = ', np.real(P2a[-1])
    plt.plot(T0,P2a,label='alpha = '+str(alpha))
    plt.show()
    return alpha, P2a

def findz(init=0.008, end=0.012, num=10, sleep=0.5, count=0, f=0.999, Alpha=1*np.ones(1), plot=True, trace=False):
    for loop_i in range(len(Alpha)):
        q0.uw = cosine-1j*(Alpha[loop_i]/(2*np.pi*nonlin))*env.deriv(cosine)
        for z in np.linspace(init,end,num):
            count += 1
            q0.df = z 
            rhos0z = system.simulate(psi0, T0, method='fast')
            rhosz = rhos0z[:,0:2,0:2]
            P1z = rhosz[:, 1][:, 1]

            q0.df = z+(end-init)/(num-1)
            rhos0z1 = system.simulate(psi0, T0, method='fast')
            rhosz1 = rhos0z1[:,0:2,0:2]
            P1z1 = rhosz1[:, 1][:, 1]

            if np.real(P1z1[-1])-np.real(P1z[-1]) > 0:
                while np.real(P1z[-1]) < f:
                    print 'alpha = ', Alpha[loop_i]
                    print 'count = ', count
                    print 'z = ', z
                    print 'P1z = ', np.real(P1z[-1])
                    print '------------------------'
                    time.sleep(sleep)
                    break
            else:
                break
        print 'z_final = ', np.real(z)
        print 'P1z_final = ', np.real(P1z[-1])

        if plot:
            plt.subplot(int(math.ceil(len(Alpha)/5.0)),5,loop_i+1)
            plt.plot(T0,P1z,label='alpha = '+str(Alpha[loop_i]))
            plt.ylim(0.999,1.0)
            plt.grid()
            plt.legend(loc=1)
        if trace:
            pyplt.plotTrajectory(rhosz, state=1, labels=True)
            plt.show()
    plt.show()

    return z, P1z, rhosz

eigval = []
for ti in range(len(T0)):
    Hti = q0.H(T0[ti])[1:3,1:3]
    V, U = np.linalg.eig(Hti)
    eigval.append((abs(V[1]-V[0])-abs(Hti[1][1]))/2)
eigval = np.array(eigval) 
average = sum(eigval/(2*np.pi)*(T0[1]-T0[0]))/T0[-1]

if any(q0.df(T0)) == 0:
    print 'average =', np.real(average)

# plt.plot(T0,eigval/(2*np.pi))
# plt.show()

if __name__ == '__main__':
    print 'P1_final =',np.real(P1[-1])
    #pyplt.plotTrajectory(rhos, state=1, labels=True)
    plt.plot(T0,P0,label='P0')
    plt.plot(T0,P1,label='P1')
    plt.plot(T0,P2,label='P2')
    plt.legend(loc=1)
    plt.show()
    #findAlpha(init=0.0, end=2, step=0.01, count=0, error=1e-6)
    #findz(init=0.0, end=0.02, num=200, sleep=0.1, count=0, f=0.999999, Alpha=np.ones(1)*0.7, plot=True, trace=False)
    
 