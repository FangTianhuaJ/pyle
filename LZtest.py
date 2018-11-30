import numpy as np
import matplotlib.pyplot as plt
import pylesim.quantsim as sim
import pylesim.plotting as pyplt
from scipy import optimize
from pylesim.envelopes import Envelope
from matplotlib.pyplot import figure

T = np.arange(0,200,0.5)
psi0 = np.array([0,1+0j])
theta = np.arange(0,2*np.pi,0.2)
p1 = []
Rhos0 = []
q = sim.Qubit2(T1=50000,T2=50000)

class lzpulse(object):
    def __init__(self,tau0=0,tau1=37.5,taup=25):
        self.tau0 = tau0
        self.tau1 = tau1
        self.taup = taup
        self.tau2 = 3*tau1-2*tau0
        self.tauc = 2*tau1-2*tau0+taup
        self.taue = 4*tau1-3*tau0
        self.t0 = (4*tau1-2*tau0)/2 # 'the center time for pipulse'

        if self.tau0 >= self.tau1-self.taup/2:
            raise ValueError('tau0 must be less than tau1-taup/2')

    def lzx(self,k0=None):
        def timeFunc(t):
            if self.tau0<=t<=self.tau1-self.taup/2:
                nx=k0*1e-3*(t-self.tau0)
            elif self.tau1-self.taup/2<t<=self.tau1+self.taup/2:
                nx=(self.tau1-self.taup/2-self.tau0)*k0*1e-3
            elif self.tau1+self.taup/2<t<=self.tau2-self.taup/2:
                nx=-k0*1e-3*(t-(self.tau1+self.tau2)/2)
            elif self.tau2-self.taup/2<t<=self.tau2+self.taup/2:
                nx=-(self.tau1-self.taup/2-self.tau0)*k0*1e-3
            elif self.tau2+self.taup/2<t<=self.taue:
                nx=k0*1e-3*(t-self.taue)
            else:
                nx=0
            return nx
        return Envelope(timeFunc,start=self.tau0,end=self.taue)

    def lzy(self,k1=None):
        def timeFunc(t):
            if self.tau0<=t<=self.tau1-self.taup/2:
                ny=k1*1e-3*(t-self.tau0)
            elif self.tau1-self.taup/2<t<=self.tau1+self.taup/2:
                ny=(self.tau1-self.taup/2-self.tau0)*k1*1e-3
            elif self.tau1+self.taup/2<t<=self.tau2-self.taup/2:
                ny=-k1*1e-3*(t-(self.tau1+self.tau2)/2)
            elif self.tau2-self.taup/2<t<=self.tau2+self.taup/2:
                ny=-(self.tau1-self.taup/2-self.tau0)*k1*1e-3
            elif self.tau2+self.taup/2<t<=self.taue:
                ny=k1*1e-3*(t-self.taue)
            else:
                ny=0
            return ny
        return Envelope(timeFunc,start=self.tau0,end=self.taue)

    def lzz(self,k2=None):
        def timeFunc(t):
            if self.tau1-self.taup/2<t<=self.tau1+self.taup/2:
                m=k2*1e-3*(t-(self.tau1-self.taup/2))
            elif self.tau1+self.taup/2<t<self.tau2-self.taup/2:
                m=self.taup*1e-3*k2
            elif self.tau2-self.taup/2<=t<self.tau2+self.taup/2:
                m=-k2*1e-3*(t-(self.tau2+self.taup/2))
            else:
                m=0
            return m
        return Envelope(timeFunc,start=self.tau0,end=self.taue)

    def pipulse(self,A=np.pi/2,tau=2,phase=0):
        def timeFunc(t):
            value = (0.5*A*(1+np.cos(2*np.pi*(t-self.t0)/tau)) * (((t-self.t0)+tau/2.)>0) \
                    * ((-(t-self.t0)+tau/2.)>0) * np.exp(1j*phase))/np.pi
            val=np.real(value)
            return val
        return Envelope(timeFunc,start=self.t0-tau/2,end=self.t0+tau/2)


def func(theta,s):
    Plz = s
    return 1-4*Plz*(1-Plz)*(np.sin(theta))**2

def residuals(s,y,theta):
    return y-func(theta,s)

lp = lzpulse(tau0=0,tau1=37.5,taup=25)

for th in theta:
    q.uw=lp.lzx(k0=2)+1j*lp.lzy(k1=2*np.tan(th))+lp.pipulse(tau=2)
    q.df=lp.lzz(k2=4)
    qsys=sim.QuantumSystem([q])
    rhos0=qsys.simulate(psi0,T,method='other')
    Rhos0.append(rhos0)

Rhos = np.array(Rhos0)

for j in range(len(theta)):
    rhos = Rhos[j]
    p1.append(rhos[:,0][:,0][-1])

P1 = np.real(np.array(np.array(p1)))

def fitting(s0 = [0.5]):
    val = optimize.leastsq(residuals,s0,args=(P1,theta))
    return val

plsq = fitting()
print u"Landau-Zener Transition Probability P-LZ:",plsq[0]

if __name__ == '__main__':
    plt.xlabel('Geometric phase')
    plt.ylabel('P1')
    plt.scatter(theta,P1,label='real data')
    plt.plot(theta,func(theta,plsq[0]),'k',label='fitting data')
    plt.legend(loc=1)
    plt.show()



# if __name__ == '__main__':
#     x,y,z,p = lp.lzx(k0=2),lp.lzy(k1=2*np.tan(1.2)),lp.lzz(k2=4),lp.pipulse(tau=2)
#     figure(figsize=(8,8))
#     plt.xlabel('Time[ns]')
#     plt.ylabel('Pulse')
#     plt.plot(T,np.array([x.timeFunc(t) for t in T]),'-',label='lzx')
#     plt.plot(T,np.array([y.timeFunc(t) for t in T]),'-.',label='lzy')
#     plt.plot(T,np.array([z.timeFunc(t) for t in T]),'--',label='lzz')
#     plt.plot(T,np.array([p.timeFunc(t) for t in T]),'-',label='pipulse')
#     plt.legend(loc=1)
