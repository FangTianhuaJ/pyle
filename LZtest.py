import numpy as np
import matplotlib.pyplot as plt
import pylesim.quantsim as sim
import pylesim.plotting as pyplt
from scipy import optimize
from pylesim.envelopes import Envelope
from matplotlib.pyplot import figure

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

class simu(object):
    def __init__(self,start=0,end=150,step=0.1):
        self.start = start
        self.end = end
        self.step = step

    def H(self,t):
        lp = lzpulse(tau0=0,tau1=37.5,taup=25)
        sigmax=np.array([[0,1],[1,0]],dtype=np.complex)
        sigmay=np.array([[0,-1j],[1j,0]],dtype=np.complex)
        sigmaz=np.array([[1,0],[0,-1]],dtype=np.complex)
        return 0.5*((lp.lzx(k0=0).timeFunc(t)+lp.pipulse(tau=2).timeFunc(t))*sigmax+\
            lp.lzy(k1=2).timeFunc(t)*sigmay+lp.lzz(k2=3).timeFunc(t)*sigmaz)

    def evolution_with_time(self):
        T = np.arange(self.start,self.end,self.step)
        dt = T[1]-T[0]
        psi0 = np.array([0,1+0j])
        psit = psi0
        psiT = []

        for t in T:
            psit=psit-1j*np.dot(simu().H(t),psit)*dt
            psiT.append(psit)

        psiT = np.array(psiT)
        Pa = psiT[:,0]*psiT[:,0].conj()
        Pb = psiT[:,1]*psiT[:,1].conj()
        return [T,Pa,Pb]

    def evolution_with_theta(self):
        T = np.arange(self.start,self.end,self.step)
        theta = np.arange(0,2*np.pi,0.2)
        psi0 = np.array([0,1+0j])
        p1 = []
        Rhos0 = []
        q = sim.Qubit2(T1=10000,T2=10000)
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
        return [Rhos,P1]

    def func(self,theta,s):
        Plz = s
        return 1-4*Plz*(1-Plz)*(np.sin(theta))**2

    def residuals(self,s,y,theta):
        return y-simu().func(theta,s)

    def fitting(self,s0 = [0.5]):
        ev = simu().evolution_with_theta()
        theta = np.arange(0,2*np.pi,0.2)
        value = optimize.leastsq(simu().residuals,s0,args=(ev[1],theta))
        return value

    def print_PLZ(self):
        plsq = simu().fitting()
        print 'Landau-Zener Transition Probability P-LZ is',plsq[0][0]


if __name__ == '__main__':
    T = simu().evolution_with_time()[0]
    Pa = simu().evolution_with_time()[1]
    Pb = simu().evolution_with_time()[2]
    theta = np.arange(0,2*np.pi,0.2)
    plsq = simu().fitting()
    lp = lzpulse(tau0=0,tau1=37.5,taup=25)
    Rhos = simu().evolution_with_theta()[0]
    P1 = simu().evolution_with_theta()[1]
    x,y,z,p = lp.lzx(k0=2),lp.lzy(k1=2*np.tan(1.2)),lp.lzz(k2=4),lp.pipulse(tau=2)
    plt.figure()
    plt.xlabel('Time[ns]')
    plt.ylabel('Pulse')
    plt.plot(T,np.array([x.timeFunc(t) for t in T]),'-',label='lzx')
    plt.plot(T,np.array([y.timeFunc(t) for t in T]),'-.',label='lzy')
    plt.plot(T,np.array([z.timeFunc(t) for t in T]),'--',label='lzz')
    plt.plot(T,np.array([p.timeFunc(t) for t in T]),'-',label='pipulse')
    plt.legend(loc=1)
    plt.figure()
    plt.xlabel('time[ns]')
    plt.ylabel('energy')
    plt.plot(T,Pa,'--',label='p0')
    plt.plot(T,Pb,'-.',label='p1')
    plt.legend(loc=1)
    plt.grid()
    plt.ylim(0,1)
    plt.figure()
    plt.xlabel('Geometric phase')
    plt.ylabel('P1')
    plt.scatter(theta,P1,label='real data')
    plt.plot(theta,simu().func(theta,plsq[0]),'k',label='fitting data')
    plt.legend(loc=1)
    pyplt.plotTrajectory(Rhos[0][:,0:2,0:2],state=1,labels=True)
    plt.show()