import numpy as np
import matplotlib.pyplot as plt

import time
import itertools
import labrad
import labrad.units as U

import pyle
import pyle.envelopes as env
from pyle import gates
from pyle.dataking import util
from pyle.envelopes import Envelope,test_env,NumericalPulse
from pyle.gates import Gate
from pyle.dataking import envelopehelpers as eh
from pyle.dataking import utilMultilevels as ml
from pyle.dataking import sweeps
from pyle import gateCompiler as gc
from pyle.dataking.fpgaseqTransmonV7 import runQubits
from pyle.util import convertUnits
from pyle.util import sweeptools as st
from pyle.analysis import readout
from pyle.pipeline import returnValue, FutureList


# COLORS
BLUE   = "#348ABD"
RED    = "#E24A33"
PURPLE = "#988ED5"
YELLOW = "#FBC15E"
GREEN  = "#8EBA42"
PINK   = "#FFB5B8"
GRAY   = "#777777"
COLORS = [BLUE, RED, GREEN, YELLOW, PURPLE, PINK, GRAY]

V, mV, us, ns, GHz, MHz, dBm, rad = [U.Unit(s) for s in ('V', 'mV', 'us', 'ns', 'GHz', 'MHz', 'dBm', 'rad')]


class func(object):
    def __init__(self, A=138.9, Delta=0.2, dt=1e-4):
        self.dt = dt
        self.A=A
        self.Delta=Delta
        self.tau=5.87/self.Delta
    def x(self,t):
        return (self.A*(t/self.tau)**4)*((1-t/self.tau)**4)+np.pi/4
    def x1(self,t):
        return (func().x(t+self.dt) - func().x(t-self.dt))/(2*self.dt)
    def x2(self,t):
        return (func().x1(t+self.dt) - func().x1(t-self.dt))/(2*self.dt)
    def V(self,t):
        return (func().x2(t)/(2*np.sqrt(self.Delta**2/4.0-func().x1(t)**2)))-(np.sqrt(self.Delta**2/4.0-func().x1(t)**2))/np.tan(2*func().x(t))


class SwiphtPulse(NumericalPulse):
    @convertUnits(t0='ns', w='ns', phase=None, df='GHz')
    def __init__(self, t0=5, w=29.35, phase=0.0, df=0.0, amp=1.0):
        self.t0 = t0
        self.w = w
        self.df = df
        self.phase = phase
        self.amp = amp
        Envelope.__init__(self, t0-w/2.0, t0+w/2.0)

    def timeFunc(self,t):
        return self.amp*func().V(t-self.t0)* ((t-self.t0)>0) * ((-(t-self.t0)+self.w)>0)

if __name__ == '__main__':
    T = np.linspace(-50,50,301)
    seq = SwiphtPulse(t0=5, w=29.35, phase=0.0, df=0.0)
    test_env(seq)
    plt.figure()
    plt.plot(T,seq(T))
    plt.show()


class SwiphtGate(Gate):
    def __init__(self, agents, amp=1.0, alpha=None, phase=0.0, freq=None, state=1, dualBlock=False):
        """
        PiPulse to state
        @param agents: agents, the gate is applied to the first element, agents[0]
        @param alpha: coefficient for DRAG,
                if None,
                  if state=1, use the agents[0]['alpha']
                  else, alpha=0
        @param phase: the phase of the microwave, 0 for +X, pi/2 for +Y ...
        @param freq: the frequency for mix, if None, use the frequency compatible with state
        @param state: the desired state after pipulse
        @param dualBlock: specific the block
        """
        self.phase = phase
        self.freq = freq
        self.state = state
        self.alpha = alpha
        self.dualBlock = dualBlock
        self.amp = amp
        self.agents = agents
        Gate.__init__(self, agents)

    def updateAgents(self):
        ag = self.agents[0]
        t = ag['_t']
        l = ag['piLen']
        phase, freq, state = self.phase, self.freq, self.state
        phase += ml.getMultiLevels(ag, 'xy_phase', state)
        pulse = eh.mix(ag, SwiphtPulse(t0=t, amp=self.amp), freq, state=state)
        if self.dualBlock:
            ag['xy_s'] += pulse
        else:
            ag['xy'] += pulse
        ag['_t'] += l #+ ag['piBuffer']

    def _name(self):
        return "SwiphtPulse"


def testSwiphtAmp(Sample, measure=0, amp=st.r[0:2:0.05], stats=1200, name='swipht amp test',
                 alpha=None, dataFormat='Amp', save=True, noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure=measure)
    axes = [(amp, "Amp")]
    if dataFormat == 'Amp':
        deps = [("Mag", "", ""), ("Phase", "", "")]
    else:
        deps = [("I", "", ""), ("Q", "", "")]
    kw = {'stats': stats}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure, kw=kw)

    def func(server, currAmp):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        #alg[SwiphtGate([q0], amp=currAmp)]
        #alg[gates.Wait([q0],50*ns)]
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
