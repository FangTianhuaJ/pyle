import numpy as np
import pyle
from pyle.envelopes import NumericalPulse, Envelope
from pyle import envelopes as env
from pyle.dataking import envelopehelpers as eh
from pyle.dataking import adjust
from pyle.util import sweeptools as st
from pyle.dataking import utilMultilevels as ml
from pyle.plotting import dstools
from pyle.fitting import fitting
from pyle.dataking import qubitpulsecal as qpc
from pyle.dataking import sweeps
from pyle.dataking import util
from pyle.dataking.fpgaseqTransmonV7 import runQubits
from pyle import gateCompiler as gc
from pyle import gates
from pyle.plotting import tomography
from pyle.interpol import interp1d_cubic
from pyle.analysis import readout
from pyle.dataking.benchmarking import randomizedBechmarking as rb
from pyle.dataking import zfuncs
import matplotlib.pyplot as plt
from scipy import optimize
from matplotlib.pyplot import figure
from pyle.gates import Gate
from labrad import units as U
from labrad.units import Unit
from pyle.util import convertUnits
from pyle.pipeline import returnValue, FutureList

V, mV, us, ns, GHz, MHz, dBm, rad, au = [Unit(s) for s in
                                         ('V', 'mV', 'us', 'ns', 'GHz', 'MHz', 'dBm', 'rad', 'au')]

class lzxPulse(NumericalPulse):
    @convertUnits(tau0='ns', tau1='ns', taup='ns')
    def __init__(self, q, tau0=0, tau1=37.5, taup=25, k0=1.0, amp=1.0):
        self.q = q
        self.tau0 = tau0  # 'the start time for lzpulse'
        self.tau1 = tau1
        self.taup = taup
        self.k0 = k0
        self.amp = amp
        self.tau2 = 3 * tau1 - 2 * tau0
        self.tauc = 2 * tau1 - 2 * tau0 + taup
        self.taue = 4 * tau1 - 3 * tau0  # 'the end time for lzpulse'
        self.t0 = (4 * tau1 - 2 * tau0) / 2  # 'the center time for pipulse'
        Envelope.__init__(self,start=tau0, end=self.taue)
        if self.tau0 >= self.tau1-self.taup/2:
            raise ValueError('tau0 must be less than tau1-taup/2')

    def timeFunc(self, t):
        nx1 = self.k0*1e-3*(t-self.tau0)*(t>=self.tau0)*(t<=self.tau1-self.taup/2)
        nx2 = (self.tau1-self.taup/2-self.tau0)*self.k0*1e-3*(t>self.tau1-self.taup/2)*(t<=self.tau1+self.taup/2)
        nx3 = -self.k0*1e-3*(t-(self.tau1+self.tau2)/2)*(t>self.tau1+self.taup/2)*(t<=self.tau2-self.taup/2)
        nx4 = -(self.tau1-self.taup/2-self.tau0)*self.k0*1e-3*(t>self.tau2-self.taup/2)*(t<=self.tau2+self.taup/2)
        nx5 = self.k0*1e-3*(t-self.taue)*(t>self.tau2+self.taup/2)*(t<=self.taue)
        nx6 = 0*(t<self.tau0)*(t>self.taue)
        return self.amp*(nx1+nx2+nx3+nx4+nx5+nx6)

class lzyPulse(NumericalPulse):
    @convertUnits(tau0='ns', tau1='ns', taup='ns')
    def __init__(self, q, tau0=0, tau1=37.5, taup=25, k1=2.0, amp=1.0):
        self.q = q
        self.tau0 = tau0  # 'the start time for lzpulse'
        self.tau1 = tau1
        self.taup = taup
        self.k1 = k1
        self.amp = amp
        self.tau2 = 3 * tau1 - 2 * tau0
        self.tauc = 2 * tau1 - 2 * tau0 + taup
        self.taue = 4 * tau1 - 3 * tau0  # 'the end time for lzpulse'
        self.t0 = (4 * tau1 - 2 * tau0) / 2  # 'the center time for pipulse'
        Envelope.__init__(self,start=tau0, end=self.taue)
        if self.tau0 >= self.tau1-self.taup/2:
            raise ValueError('tau0 must be less than tau1-taup/2')

    def timeFunc(self, t):
        ny1 = self.k1*1e-3*(t-self.tau0)*(t>=self.tau0)*(t<=self.tau1-self.taup/2)
        ny2 = (self.tau1-self.taup/2-self.tau0)*self.k1*1e-3*(t>self.tau1-self.taup/2)*(t<=self.tau1+self.taup/2)
        ny3 = -self.k1*1e-3*(t-(self.tau1+self.tau2)/2)*(t>self.tau1+self.taup/2)*(t<=self.tau2-self.taup/2)
        ny4 = -(self.tau1-self.taup/2-self.tau0)*self.k1*1e-3*(t>self.tau2-self.taup/2)*(t<=self.tau2+self.taup/2)
        ny5 = self.k1*1e-3*(t-self.taue)*(t>self.tau2+self.taup/2)*(t<=self.taue)
        ny6 = 0*(t<self.tau0)*(t>self.taue)
        return self.amp*(ny1+ny2+ny3+ny4+ny5+ny6)

class lzzPulse(NumericalPulse):
    @convertUnits(tau0='ns', tau1='ns', taup='ns')
    def __init__(self, q, tau0=0, tau1=37.5, taup=25, k2=1, amp=1.0):
        self.q = q
        self.tau0 = tau0  # 'the start time for lzpulse'
        self.tau1 = tau1
        self.taup = taup
        self.k2 = k2
        self.amp = amp
        self.tau2 = 3 * tau1 - 2 * tau0
        self.tauc = 2 * tau1 - 2 * tau0 + taup
        self.taue = 4 * tau1 - 3 * tau0  # 'the end time for lzpulse'
        self.t0 = (4 * tau1 - 2 * tau0) / 2 # 'the center time for pipulse'
        Envelope.__init__(self,start=tau0, end=self.taue)
        if self.tau0 >= self.tau1-self.taup/2:
            raise ValueError('tau0 must be less than tau1-taup/2')

    def timeFunc(self, t):
        nz1 = self.k2*1e-3*(t-(self.tau1-self.taup/2))*(t>=self.tau1-self.taup/2)*(t<= self.tau1+self.taup/2)
        nz2 = self.taup*1e-3*self.k2*(t>self.tau1+self.taup/2)*(t<self.tau2-self.taup/2)
        nz3 = -self.k2*1e-3*(t-(self.tau2+self.taup/2))*(t>self.tau2-self.taup/2)*(t<=self.tau2+self.taup/2)
        nz4  =0*(t<self.tau0)*(t>self.taue)
        return self.amp*(nz1+nz2+nz3+nz4)

class LZ_gate(Gate):
    def __init__(self, agents, tau0=0*ns, tau1=None, taup=None, k0=None, k1=None, k2=None,
                amp=None, freq=None, state=1):
        self.tau0 = tau0  # 'the start time for lzpulse'
        self.tau1 = tau1
        self.taup = taup
        self.k0 = k0
        self.k1 = k1
        self.k2 = k2
        self.amp = amp
        self.freq = freq
        self.state = state
        if self.tau1 == None:
            self.tau1 = agents[0]['lztau1']
        if self.taup == None:
            self.taup = agents[0]['lztaup']
        if self.amp == None:
            self.amp = agents[0]['lzamp']
        if self.k0 == None:
            self.k0 = agents[0]['k0']
        if self.k1 == None:
            self.k1 = agents[0]['k1']
        if self.k2 == None:
            self.k2 = agents[0]['k2']
        self.tau2 = 3 * self.tau1 - 2 * self.tau0
        self.tauc = 2 * self.tau1 - 2 * self.tau0 + self.taup
        self.taue = 4 * self.tau1 - 3 * self.tau0  # 'the end time for lzpulse'
        self.t0 = (4 * self.tau1 - 2 * self.tau0) / 2  # 'the center time for pipulse'
        Gate.__init__(self, agents)

    def updateAgents(self):
        ag = self.agents[0]
        t = ag['_t']
        l = self.taue
        ag['xy'] += eh.mix(ag,eh.piPulse(ag, t+(l/2),state=self.state)+lzxPulse(q=ag, tau0=t, tau1=t+self.tau1, taup=self.taup, k0=self.k0, amp=self.amp)+1j*lzyPulse(q=ag, tau0=t, tau1=t+self.tau1, taup=self.taup, k1=self.k1, amp=self.amp))
        ag['z'] += lzzPulse(q=ag, tau0=t, tau1=t+self.tau1, taup=self.taup, k2=self.k2, amp=self.amp)
        ag['_t'] += l

    def _name(self):
        return 'laudau_zenner transition'

def lztest_amp(Sample, measure=0, amp=st.r[0:1:0.02], stats=1200, name='lz amp test',
                  dataFormat='Amp', save=True, noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure=measure)
    axes = [(amp, "swiphtAmp")]
    if dataFormat == 'Amp':
        deps = [("Mag", "", ""), ("Phase", "", "")]
    else:
        deps = [("I", "", ""), ("Q", "", "")]
    kw = {'stats': stats}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure, kw=kw)

    def func(server, curramp):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        alg[LZ_gate([q0],amp=curramp)]
        alg[gates.Wait([q0],0*ns)]
        alg[gates.PiPulse([q0])]
        alg[gates.Readout([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        if dataFormat == 'Amp':
            mag, ang = readout.iqToPolar(readout.parseDataFormat(data,'iq'))
        else:
            data = readout.parseDataFormat(data,'iq')
            mag, ang = data[0], data[1]
        returnValue([mag, ang])

    data = sweeps.grid(func, axes=axes, save=save, dataset=dataset, noisy=noisy)

    return data

def lztest_taup(Sample, measure=0, delay=st.r[25:30:0.01,ns], state=1, stats=1200, name='lz taup test',
                  save=True, noisy=True, prob=True):
    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, True)
    axes = [(delay, 'taup')]
    if prob:
        deps = readout.genProbDeps(qubits, measure)
    else:
        deps = [("Mag", "|%s>" %state, ""), ("Phase", "|%s>" %state, "rad")]
    kw = {'stats': stats, 'prob': prob}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure, kw=kw)

    def func(server, currdelay):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        alg[gates.MoveToState([q0], 0, state-1)]
        alg[LZ_gate([q0],amp=2,taup=currdelay)]
        alg[gates.MoveToState([q0], state-1, 0)]
        alg[gates.Measure([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        if prob:
            probs = readout.iqToProbs(data, alg.qubits)
            probs = np.squeeze(probs)
            returnValue(probs)
        else:
            data = readout.parseDataFormat(data, 'iq')
            mag, phase = readout.iqToPolar(data)
            returnValue([mag, phase])

    data = sweeps.grid(func, axes=axes, save=save, dataset=dataset, noisy=noisy)

    return data

def lztest_theta(Sample, measure=0, theta=st.r[0:2*np.pi:0.1], state=1, stats=1200, name='lz theta test',
                  save=True, noisy=True, prob=True):
    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, True)
    axes = [(theta, 'Geometric phase')]
    if prob:
        deps = readout.genProbDeps(qubits, measure)
    else:
        deps = [("Mag", "|%s>" %state, ""), ("Phase", "|%s>" %state, "rad")]
    kw = {'stats': stats, 'prob': prob}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure, kw=kw)

    def func(server, currphase):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        k0 = q0['k0']
        alg[gates.MoveToState([q0], 0, state)]
        alg[LZ_gate([q0])]
        alg[LZ_gate([q0], k1=k0*np.tan(currphase))]
        alg[gates.MoveToState([q0], state-1, 0)]
        alg[gates.Measure([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        if prob:
            probs = readout.iqToProbs(data, alg.qubits)
            probs = np.squeeze(probs)
            returnValue(probs)
        else:
            data = readout.parseDataFormat(data, 'iq')
            mag, phase = readout.iqToPolar(data)
            returnValue([mag, phase])

    data = sweeps.grid(func, axes=axes, save=save, dataset=dataset, noisy=noisy)

    return data

def lztest_theta_and_taup(Sample, measure=0, taup=st.r[25:30:0.1,ns], theta=st.r[0:2*np.pi:0.1],
                     state=1, tBuf=0*ns, stats=600, name='lz test theta and taup', save=True, noisy=True):

    sample, devs, qubits = gc.loadQubits(Sample, measure)
    axes = [(taup, 'taup'), (theta, 'Geometric phase')]
    deps = readout.genProbDeps(qubits, measure, range(1+state))
    kw = {"stats": stats, 'tBuf': tBuf, "state": state}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure, kw=kw)

    def func(server, currdelay, currphase):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        k0 = q0['k0']
        alg[gates.MoveToState([q0], 0, state-1)]
        alg[gates.Wait([q0], waitTime=tBuf)]
        alg[LZ_gate([q0], taup=currdelay, k1=k0*np.tan(currphase))]
        alg[gates.Wait([q0], waitTime=tBuf)]
        alg[gates.Measure([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        probs = readout.iqToProbs(data, alg.qubits, states=range(1+state))
        returnValue(np.squeeze(probs))

    data = sweeps.grid(func, axes=axes, save=save, dataset=dataset, noisy=noisy)

    return data

if __name__ == '__main__':
    pulsex = lzxPulse(q=None, tau0=0, tau1=37.5, taup=25, k0=1, amp=1.0)
    pulsey = lzyPulse(q=None, tau0=0, tau1=37.5, taup=25, k1=2, amp=1.0)
    pulsez = lzzPulse(q=None, tau0=0, tau1=37.5, taup=25, k2=4, amp=1.0)
    T = np.linspace(-100, 200, 1001)
    env.test_env(pulsex)
    env.test_env(pulsez)
    plt.plot(T, pulsex(T),'-',label='lzx')
    plt.plot(T, pulsey(T),'-.',label='lzy')
    plt.plot(T, pulsez(T),'--',label='lzz')
    plt.legend(loc=1)
    plt.show()
