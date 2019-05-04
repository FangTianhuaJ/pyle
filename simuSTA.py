import numpy as np
import matplotlib.pyplot as plt
import pylesim.quantsim as sim
import pylesim.plotting as pyplt
from scipy import optimize
from pylesim.envelopes import Envelope
from matplotlib.pyplot import figure

class STAmethod(object):

    def __init__(self, start=0, end=30, step=2, thetaf=np.pi/6, T=30):
        self.start = start
        self.end = end
        self.step = step
        self.thetaf = thetaf
        self.T = T

    def STAx(self, m=10, omega0=1):
        def timeFunc(t):
            thetat = self.thetaf*np.sin(np.pi*t/(2*self.T))
            costh, sinth = np.cos(thetat), np.sin(thetat)
            omegat = omega0*np.cos(np.pi*t/(m*self.T))
            omegacos, omegasin = omegat*costh, omegat*sinth
            return 0.5*omegasin
        return Envelope(timeFunc, start=self.start, end=self.end)

    def STAz(self, m=10, omega0=1):
        def timeFunc(t):
            thetat = self.thetaf*np.sin(np.pi*t/(2*self.T))
            costh, sinth = np.cos(thetat), np.sin(thetat)
            omegat = omega0*np.cos(np.pi*t/(m*self.T))
            omegacos, omegasin = omegat*costh, omegat*sinth
            return 0.5*omegacos
        return Envelope(timeFunc, start=self.start, end=self.end)

    def STAy(self):
        def timeFunc(t):
            thetat = self.thetaf*np.sin(np.pi*t/(2*self.T))
            dth = self.thetaf*np.pi/(2*self.T)*np.cos((np.pi/2)*t/self.T)
            return 0.5*dth
        return Envelope(timeFunc, start=self.start, end=self.end)

    def H0(self, t, m=10, omega0=1):
        thetat = self.thetaf*np.sin(np.pi*t/(2*self.T))
        costh, sinth = np.cos(thetat), np.sin(thetat)
        omegat = omega0*np.cos(np.pi*t/(m*self.T))
        omegacos, omegasin = omegat*costh, omegat*sinth
        sigmax=np.array([[0,1],[1,0]],dtype=np.complex)
        sigmaz=np.array([[1,0],[0,-1]],dtype=np.complex)
        return 0.5*(omegasin*sigmax+omegacos*sigmaz)

    def Hcd(self, t):
        thetat = self.thetaf*np.sin(np.pi*t/(2*self.T))
        dth = self.thetaf*np.pi/(2*self.T)*np.cos((np.pi/2)*t/self.T)
        sigmay=np.array([[0,-1j],[1j,0]],dtype=np.complex)
        return 0.5*dth*sigmay

    def eigvect(self, t):
        thetat = self.thetaf*np.sin(np.pi*t/(2*self.T))
        costh2 = np.cos(thetat/2)
        sinth2 = np.sin(thetat / 2)
        return costh2, sinth2

    def evolution_with_time(self, display=False, output=False, plot=True):
        T0 = np.arange(self.start,self.end,self.step)
        dt = T0[1]-T0[0]
        psi0 = 0.5*np.array([1+0j,1])
        psit = psi0
        psiT = []
        for t in T0:
            psit=psit-1j*np.dot(STAmethod().H0(t)+STAmethod().Hcd(t),psit)*dt
            psiT.append(psit)
        psiT = np.array(psiT)
        P0 = psiT[:,0]*psiT[:,0].conj()
        P1 = psiT[:,1]*psiT[:,1].conj()
        data = [P0,P1]
        if plot:
            plt.figure()
            plt.xlabel('time[ns]')
            plt.ylabel('population')
            #plt.plot(T0,P0,'--',label='p0')
            plt.plot(T0,P1,'-.',label='p1')
            plt.legend(loc=1)
            plt.ylim(0,1.0)
            plt.show()
        if display:
            print 'P0 =',data[0]
            print 'P1 =',data[1]
        if output:
            return data

    def eigen_evolution(self, output=False, plot=True):
        T0 = np.arange(self.start,self.end,self.step)
        dt = T0[1]-T0[0]
        psi0 = np.array([1+0j,0])
        psit = psi0
        psiT = []
        coth2 = STAmethod().eigvect(t=T0)[0]
        sith2 = STAmethod().eigvect(t=T0)[1]
        for t in T0:
            psit=psit-1j*np.dot(STAmethod().H0(t)+STAmethod().Hcd(t),psit)*dt
            psiT.append(psit)
        psiT = np.array(psiT)
        Pup = (psiT[:, 0] * coth2 + psiT[:,1]*sith2)*(psiT[:, 0] * coth2 + psiT[:,1]*sith2).conj()
        Pdown = (psiT[:, 0] * -sith2 + psiT[:, 1] * coth2) * (psiT[:, 0] * -sith2 + psiT[:, 1] * coth2).conj()
        data = [Pup,Pdown]
        if plot:
            plt.figure()
            plt.xlabel('time[ns]')
            plt.ylabel('population')
            plt.plot(T0,Pup,'--',label='Pup')
            #plt.plot(T0,Pdown,'-.',label='Pdown')
            plt.legend(loc=1)
            plt.ylim(0, 1.2)
            plt.show()
        if output:
            return data

    def simuSTA(self, bloch=False, plot=False, output=False):
        T0 = np.arange(self.start, self.end, self.step)
        psi0 = np.array([1+0j,0])
        q = sim.Qubit2(T1=10000, T2=10000)
        q.uw = STAmethod().STAx()+1j*STAmethod().STAy()
        q.df = STAmethod().STAz()
        qsys = sim.QuantumSystem([q])
        rhos0 = qsys.simulate(psi0, T0, method='other')
        if bloch:
             pyplt.plotTrajectory(rhos0, state=1, labels=True)
             plt.show()
        data = rhos0
        P0 = rhos0[:, 0][:, 0]
        P1 = rhos0[:, 1][:, 1]
        if plot:
            plt.xlabel('time[ns]')
            plt.ylabel('population')
            #plt.plot(T0, P0, '--', label='p0')
            plt.plot(T0, P1, '-.', label='p1')
            plt.legend(loc=1)
            plt.ylim(0, 1)
            plt.show()
        if output:
            print 'P0=',P0,'rho=',data
        return data

if __name__ == '__main__':
    STAmethod(start=0, end=20, step=0.001, thetaf=np.pi/6, T=10).eigen_evolution()
    #STAmethod(start=0, end=60, step=0.2, thetaf=np.pi/6, T=30).simuSTA(output=True)
    #STAmethod(start=0, end=20, step=0.001, thetaf=np.pi/6, T=10).evolution_with_time()