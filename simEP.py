import numpy as np
from numpy import pi
from matplotlib import pyplot as plt
import pylesim.plotting as pyplt
from scipy import optimize
from pylesim import envelopes as env
from pylesim.envelopes import Envelope
from matplotlib.pyplot import figure
import pylesim.quantsim as sim

q0 = sim.Qubit3(T1=50000, T2=40000, nonlin=-0.24)
q1 = sim.Qubit3(T1=50000, T2=40000, nonlin=-0.2)
rc = sim.Resonator(T1=3000, T2=10e3, n=2, df=0.3)

S = 0.01

c0 = sim.Coupler(q0,rc,s=S)
c1 = sim.Coupler(q1,rc,s=S)
c12 = sim.Coupler(q0,q1,s=S)

system0 = sim.QuantumSystem([q0,rc], [c0])
system1 = sim.QuantumSystem([q1,rc], [c1])
system = sim.QuantumSystem([q0,q1],[c12])


class TwoqubitHeatEngine(object):
    def __init__(self, start=0, end=36, step=0.01):
        self.start = start
        self.end = end
        self.step = step    

    def f10A(self, t0=-1000, t1=1000, z=5.48315):
        def timeFunc(t):
            za = z*(t>t0)*(t<t1)
            return za
        return Envelope(timeFunc, start=t0, end=t1)

    def f10B(self, t0=-1000, t1=1000, z=5.24315):
        def timeFunc(t):
            za = z*(t>t0)*(t<t1)
            return za
        return Envelope(timeFunc, start=t0, end=t1)

    def zpulse(self, t0=0, uplen=10, keeplen=10, zinit=-0.2, zfina=-0.01):
        def timeFunc(t):
            za = ((zfina-zinit)/(uplen) * (t-t0) + zinit)*(t>t0)*(t<=t0+uplen)
            zb = zfina*(t>t0+uplen)*(t<=t0+uplen+keeplen)
            return za+zb
        return Envelope(timeFunc, start=t0, end=t0+uplen+keeplen)

    def cpulse(self, t0=0, uplen=10, z=-0.8):
        def timeFunc(t):
            za = z*(t>t0)*(t<=t0+uplen)
            return za
        return Envelope(timeFunc, start=t0, end=t0+uplen)

    def swap11and20(self, tuneA=np.arange(-0.2,0.2,0.001),delay=np.arange(0,200,0.01)):
        psi0 = np.array([0,0,0,0,1+0j,0,0,0,0])
        P_11,P_20 = [],[]
        q1.df = TwoqubitHeatEngine().f10B()
        for i in range(len(tuneA)):
            tuneAi = tuneA[i]
            q0.df = TwoqubitHeatEngine().f10A(z=5.48315+tuneAi)
            rhos0 = system.simulate(psi0, delay, method='other')
            P11 = rhos0[:, 4][:, 4]
            P20 = rhos0[:, 6][:, 6]
            P_11.append(P11)
            P_20.append(P20)
        P_11 = np.real(np.array(np.array(P_11)))
        P_20 = np.array(P_20)
        lx,ly = len(tuneA),len(delay)
        delay_m,tuneA_m = np.meshgrid(delay,tuneA)
        P11_m = P_11.reshape(lx,ly)
        plt.pcolormesh(tuneA_m,delay_m,P11_m,cmap='RdYlBu_r')
        plt.colorbar(label='P11')
        plt.xlabel('tuneA[GHz]')
        plt.ylabel('delay[ns]')
        plt.show()
        return len(P_11)

    def simisothermy(self, plot=False, output=False, bloch=True, trace=True, t0A=0, t0B=0, uplen=[5,5,5,5], keeplen=[5,5,3,3],\
        zamp=[-0.2,-0.05,-0.05,-0.02,-0.02,-0.01,-0.01, 0.0], detune=[-1.46,-1.46,-1.4,-1.4]):
        T0 = np.arange(self.start, self.end, self.step)
        psi0 = np.array([0,0,0,0,1+0j,0,0,0,0])
        # q0.uw = env.cosine(t0=10, w=20, amp=1.0/20, phase=0.0)
        # q1.uw = env.cosine(t0=10, w=20, amp=1.0/20, phase=0.0)
        q0.df = TwoqubitHeatEngine().f10A()
        q1.df = TwoqubitHeatEngine().f10B()

        for i in range(len(uplen)):
            upleni = uplen[i]
            keepleni = keeplen[i]
            ziniti = zamp[2*i]
            zfinai = zamp[2*i+1]
            q0.df += TwoqubitHeatEngine().zpulse(t0=t0A, uplen=upleni, keeplen=keepleni, zinit=ziniti, zfina=zfinai)
            t0A += upleni+keepleni

        for i in range(len(uplen)):
            upleni = uplen[i]
            keepleni = keeplen[i]
            zi = detune[i]
            q1.df += TwoqubitHeatEngine().cpulse(t0=t0B, uplen=upleni, z=zi)
            t0B += upleni+keepleni

        rhos0 = system.simulate(psi0, T0, method='other')

        P11 = rhos0[:, 4][:, 4]
        P20 = rhos0[:, 6][:, 6]
        tr_reduced_rho = []
        tr_reduced_rho2 = []
        reduced_rho = []

        for i in range(len(T0)):
            rhoi = np.array(rhos0[i])
            reduced_rhoi = [[rhoi[0,0]+rhoi[1,1]+rhoi[2,2],rhoi[0,3]+rhoi[1,4]+rhoi[2,5],rhoi[0,6]+rhoi[1,7]+rhoi[2,8]],\
                            [rhoi[3,0]+rhoi[4,1]+rhoi[5,2],rhoi[3,3]+rhoi[4,4]+rhoi[5,5],rhoi[3,6]+rhoi[4,7]+rhoi[5,8]],\
                            [rhoi[6,0]+rhoi[7,1]+rhoi[8,2],rhoi[6,3]+rhoi[7,4]+rhoi[8,5],rhoi[6,6]+rhoi[7,7]+rhoi[8,8]]]

            reduced_rho2i = np.dot(reduced_rhoi,reduced_rhoi)
            tr_reduced_rhoi = np.trace(reduced_rhoi)
            tr_reduced_rho2i = np.trace(reduced_rho2i)
            tr_reduced_rho.append(tr_reduced_rhoi)
            tr_reduced_rho2.append(tr_reduced_rho2i)
            reduced_rho.append(reduced_rhoi)
        tr_reduced_rho = np.array(tr_reduced_rho)
        tr_reduced_rho2 = np.array(tr_reduced_rho2)
        reduced_rho = np.array(reduced_rho)

        print 'reduced_rho_fina =', tr_reduced_rho[-1]
        print 'reduced_rhoi_fina =', tr_reduced_rho2[-1]
        data = [rhos0,q0.df,q1.df]

        if plot:
            plt.xlabel('time[ns]')
            plt.ylabel('population')
            plt.plot(T0, P11, '--', label='p11')
            plt.plot(T0, P20, '-.', label='p20')
            plt.legend(loc=1)
            plt.grid()
            plt.show()
        if trace:
            plt.xlabel('time[ns]')
            plt.ylabel('trace rho')
            plt.plot(T0, tr_reduced_rho, '--', label='tr_reduced_rho')
            plt.plot(T0, tr_reduced_rho2, '-.', label='tr_reduced_rho2')
            plt.legend(loc=1)
            plt.grid()
            plt.show()
        if bloch:
             pyplt.plotTrajectory(reduced_rho[:,0:2,0:2], state=1, labels=True)
             plt.show()
        if output:
            print 'P11=',P11,'rho=',data
        return data


if __name__ == '__main__':
    TwoqubitHeatEngine().simisothermy(plot=True, output=False, bloch=False, trace=True)
    TwoqubitHeatEngine().swap11and20()
    pulsez = TwoqubitHeatEngine().simisothermy(plot=False, output=True, bloch=False, trace=False)[1]
    pulsec = TwoqubitHeatEngine().simisothermy(plot=False, output=True, bloch=False, trace=False)[2]
    T = np.linspace(-20, 50, 1001)
    plt.figure()
    plt.xlabel('time[ns]')
    plt.ylabel('f10[GHz]')
    plt.plot(T, pulsez(T),label='f10A')
    plt.plot(T, pulsec(T),label='f10B')
    plt.legend(loc=1)
    plt.show()