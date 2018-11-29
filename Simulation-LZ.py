import numpy as np
import matplotlib.pyplot as plt
import pylesim.quantsim as sim
import pylesim.plotting as pyplt
from scipy import optimize
from pylesim.envelopes import Envelope

T = np.arange(0,150,1)
psi0 = np.array([0,1+0j])
theta = np.arange(0,2*np.pi,0.1)
p1 = []
Rhos0 = []
q = sim.Qubit2(T1=5000,T2=5000)

def lzx(start=0,end=150,k0=None):
    def timeFunc(t):
        if 0<=t<25:
            nx=k0*1e-3*t
        elif 25<=t<=50:
            nx=25*k0*1e-3
        elif 50<t<100:
            nx=-k0*1e-3*(t-75)
        elif 100<=t<=125:
            nx=-25*k0*1e-3
        elif 125<t<=150:
            nx=k0*1e-3*(t-150)
        else:
            nx=0
        return nx
    return Envelope(timeFunc, start, end)

def lzy(start=0,end=150,k1=None):
    def timeFunc(t):
        if 0<=t<25:
            ny=k1*1e-3*t
        elif 25<=t<=50:
            ny=25*k1*1e-3
        elif 50<t<100:
            ny=-k1*1e-3*(t-75)
        elif 100<=t<=125:
            ny=-25*k1*1e-3
        elif 125<t<=150:
            ny=k1*1e-3*(t-150)
        else:
            ny=0
        return ny
    return Envelope(timeFunc, start, end)

def lzz(start=0,end=150,k2=None):
    def timeFunc(t):
        if 25<t<=50:
            m=k2*1e-3*(t-25)
        elif 50<t<100:
            m=0.025*k2
        elif 100<=t<125:
            m=-k2*1e-3*(t-125)
        else:
            m=0
        return m
    return Envelope(timeFunc, start, end)

def pipulse(A=np.pi/2,tau=2,t0=75,phase=0,start=74,end=76):
    def timeFunc(t):
        value = (0.5*A*(1+np.cos(2*np.pi*(t-t0)/tau)) * (((t-t0)+tau/2.)>0) \
                    * ((-(t-t0)+tau/2.)>0) * np.exp(1j*phase))/np.pi
        val=np.real(value)
        return val
    return Envelope(timeFunc, start, end)

def func(theta,s):
    Plz = s
    return 1-4*Plz*(1-Plz)*(np.sin(theta))**2

def residuals(s,y,theta):
    return y-func(theta,s)

for th in theta:
    q.uw=lzx(start=0,end=150,k0=2)+1j*lzy(start=0,end=150,k1=2*np.tan(th))+pipulse(start=74,end=76)
    q.df=lzz(start=0,end=150,k2=4.0)
    qsys=sim.QuantumSystem([q])
    rhos0=qsys.simulate(psi0,T,method='other')
    Rhos0.append(rhos0)

Rhos = np.array(Rhos0)

for j in range(len(theta)):
    rhos = Rhos[j]
    p1.append(rhos[:,0][:,0][-1])

P1 = np.real(np.array(np.array(p1)))

s0 = [0.5]
plsq = optimize.leastsq(residuals,s0,args=(P1,theta))

print u"Landau-Zener Transition Probability P-LZ:",plsq[0]
plt.xlabel('Geometric phase')
plt.ylabel('P1')
plt.scatter(theta,P1,label='real data')

plt.plot(theta,func(theta,plsq[0]),'k',label='fitting data')
plt.legend(loc=1)
plt.show()