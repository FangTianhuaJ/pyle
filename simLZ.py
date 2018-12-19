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

    def plot_lz(self):
        T = np.arange(0,150,0.1)
        lp = lzpulse()
        x,y,z,p = lp.lzx(k0=2),lp.lzy(k1=2*np.tan(1.2)),lp.lzz(k2=4),lp.pipulse(tau=2)
        plt.figure()
        plt.xlabel('Time[ns]')
        plt.ylabel('Pulse')
        plt.plot(T,np.array([x.timeFunc(t) for t in T]),'-',label='lzx')
        plt.plot(T,np.array([y.timeFunc(t) for t in T]),'-.',label='lzy')
        plt.plot(T,np.array([z.timeFunc(t) for t in T]),'--',label='lzz')
        plt.plot(T,np.array([p.timeFunc(t) for t in T]),'-',label='pipulse')
        plt.legend(loc=1)
        plt.show()

class method(object):
    def __init__(self,start=0,end=150,step=0.1):
        self.start = start
        self.end = end
        self.step = step

    def H(self,t,kx=10,ky=10,kz=5,tau=2,n=37.5):
        lp = lzpulse(tau0=0,tau1=n,taup=25)
        sigmax=np.array([[0,1],[1,0]],dtype=np.complex)
        sigmay=np.array([[0,-1j],[1j,0]],dtype=np.complex)
        sigmaz=np.array([[1,0],[0,-1]],dtype=np.complex)
        return 0.5*((lp.lzx(k0=kx).timeFunc(t)+lp.pipulse(tau=tau).timeFunc(t))*sigmax+\
            lp.lzy(k1=ky).timeFunc(t)*sigmay+lp.lzz(k2=kz).timeFunc(t)*sigmaz)

    def lzfunc(self,theta,s):
        Plz = s
        return 1-4*Plz*(1-Plz)*(np.sin(theta))**2

    def residuals(self,s,y,theta):
        return y-method().lzfunc(theta,s)

    def evolution_with_time(self,tau1=37.5,display=False,output=False,plot=False):
        T = np.arange(self.start,self.end,self.step)
        dt = T[1]-T[0]
        psi0 = np.array([0,1+0j])
        psit = psi0
        psiT = []
        for t in T:
            psit=psit-1j*np.dot(method().H(t,n=tau1),psit)*dt
            psiT.append(psit)
        psiT = np.array(psiT)
        P0 = psiT[:,0]*psiT[:,0].conj()
        P1 = psiT[:,1]*psiT[:,1].conj()
        data = [P0,P1]
        if plot:
            plt.figure()
            plt.xlabel('time[ns]')
            plt.ylabel('energy')
            plt.plot(T,P0,'--',label='p0')
            plt.plot(T,P1,'-.',label='p1')
            plt.legend(loc=1)
            plt.ylim(0,1)
            plt.show()
        if display:
            print 'P0 =',data[0]
            print 'P1 =',data[1]
        if output:
            return data

    def evolution_with_theta(self,display=False,output=False,m=25,s0 = [0.5],plot=False,fitting=False):
        T = np.arange(self.start,self.end,self.step)
        Theta = np.arange(0,2*np.pi,0.2)
        psi0 = np.array([0,1+0j])
        p1 = []
        Rhos0 = []
        q = sim.Qubit2(T1=10000,T2=10000)
        lp = lzpulse(tau0=0,tau1=37.5,taup=m)
        for th in Theta:
            q.uw=lp.lzx(k0=2)+1j*lp.lzy(k1=2*np.tan(th))+lp.pipulse(tau=2)
            q.df=lp.lzz(k2=4)
            qsys=sim.QuantumSystem([q])
            rhos0=qsys.simulate(psi0,T,method='other')
            Rhos0.append(rhos0)
        Rhos = np.array(Rhos0)
        for j in range(len(Theta)):
            rhos = Rhos[j]
            p1.append(rhos[:,0][:,0][-1])
        P1 = np.real(np.array(np.array(p1)))
        value = optimize.leastsq(method().residuals,s0,args=(P1,Theta))
        data = [Theta,Rhos,P1,value]
        if plot:
            plt.figure()
            plt.xlabel('Geometric phase')
            plt.ylabel('P1')
            plt.scatter(Theta,P1,label='real data')
            plt.grid()
            plt.legend(loc=1)
            plt.show()
        if fitting:
            plt.xlabel('Geometric phase')
            plt.ylabel('P1')
            plt.scatter(Theta,P1,label='real data')
            plt.plot(Theta,method().lzfunc(Theta,value[0]),'k',label='fitting data')
            plt.legend(loc=1)
            plt.show()
        if display:
            print 'Rhos =',data[0]
            print 'P1 =',data[1]
        if output:
            return data

    def evolution_with_taup(self,tp0=25,tp1=35,num=10,output=False,imshow=False):
        Taup = np.linspace(tp0,tp1,num)
        Theta = method().evolution_with_theta(output=True)[0]
        p1 = []
        for i in Taup:
            eth = method().evolution_with_theta(m=i,output=True)
            p1.append(eth[2])
        P1 = np.array(p1)
        data = [Taup,Theta,P1]
        if imshow:
            lta,lth = len(Taup),len(Theta)
            Theta_m,Taup_m = np.meshgrid(Theta,Taup)
            P1_m = P1.reshape(lta,lth)
            plt.pcolormesh(Taup_m,Theta_m,P1_m)
            plt.colorbar(label='P1')
            plt.xlabel('LZ time taup')
            plt.ylabel('Geometric phase')
            plt.show()
        if output:
            return data

    def evolution_with_tauc(self,tc0=100,tc1=120,num=200,taup=25,output=False,plot=True):
        Tauc = np.linspace(tc0,tc1,num)
        Tau1 = (Tauc-taup)/2
        p1 = []
        for i in Tau1:
            et = method().evolution_with_time(tau1=i,output=True)[1]
            p1.append(et[-1])
        P1 = np.array(p1)
        if plot:
            plt.xlabel('Time tauc')
            plt.ylabel('P1')
            plt.plot(Tauc,P1)
            plt.show()

    def plot_bloch(self,test_theta=np.pi/2):
        Theta = np.arange(0,2*np.pi,0.2)
        rho =method().evolution_with_theta(output=True)[1]
        idth = int(len(Theta)/(2*np.pi/test_theta))
        idth1 = int(round(idth))
        try:
            pyplt.plotTrajectory(rho[idth][:,0:2,0:2],state=1,labels=True)
        except IndexError:
            pyplt.plotTrajectory(rho[idth1][:,0:2,0:2],state=1,labels=True)
        plt.show()

    def print_PLZ(self):
        plsq = method().evolution_with_theta(output=True)
        print 'Landau-Zener Transition Probability P-LZ is',plsq[2][0][0]

if __name__ == '__main__':
    method().evolution_with_theta(fitting=True)
    Theta = np.arange(0,2*np.pi,0.2)
    test_theta = np.pi/2
    idth = int(len(Theta)/(2*np.pi/test_theta))
    rho = method().evolution_with_theta(output=True)[1]
    pi = rho[idth][:,0:2,0:2][0][1][1]*rho[idth][:,0:2,0:2][0][1][1].conj()
    pf = rho[idth][:,0:2,0:2][-1][1][1]*rho[idth][:,0:2,0:2][-1][1][1].conj()
    print pi,pf




