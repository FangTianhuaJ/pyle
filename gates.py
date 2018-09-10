import numpy as np
from pyle import envelopes as env
from pyle.dataking import envelopehelpers as eh
from labrad import units as U
from pyle.dataking.benchmarking import randomizedBechmarking as rb
from pyle.dataking import utilMultilevels as ml

ns, GHz, us = (U.Unit(un) for un in ['ns', 'GHz', 'us'])

class Gate(object):

    def __init__(self, agents):
        self.agents = agents
        self.setSubgates(agents)
        for agent in agents:
            agent['gates'].append(self)
        self.name = self._name()

    def setSubgates(self, agents):
        """
        Define subgates for this gate
        This method may be overloaded
        """
        self.subgates = []

    def compile(self, globalAgents):
        """
        Recursively called method to build sequence
        This method must not be overloaded
        """
        self.globalAgents = globalAgents
        self.updateAgents()
        for gate in self.subgates:
            gate.compile(globalAgents)

    def updateAgents(self):
        """
        Gate specific agent mutations

        This method must be overloaded
        """
        raise Exception

    def _name(self):
        """This method should be overloaded"""
        return self.__class__.__name__

    def __str__(self):
        return '{} on agents'.format(self.name) \
               + ' [' + ' '.join([str(agent) for agent in self.agents]) + ']'

class PiPulse(Gate):
    def __init__(self, agents, alpha=None, phase=0.0, freq=None, state=1, dualBlock=False):
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
        Gate.__init__(self, agents)

    def updateAgents(self):
        ag  = self.agents[0]
        t = ag['_t']
        l = ag['piLen']
        phase, freq, state = self.phase, self.freq, self.state
        phase += ml.getMultiLevels(ag, 'xy_phase', state)
        pulse = eh.mix(ag, eh.piPulse(ag, t+(l/2), phase=phase, alpha=self.alpha,
                                          state=state), freq, state=state)
        if self.dualBlock:
            ag['xy_s'] += pulse
        else:
            ag['xy'] += pulse
        ag['_t'] += l + ag['piBuffer']

    def _name(self):
        return "PiPulse"

class PiHalfPulse(Gate):
    def __init__(self, agents, alpha=None, phase=0.0, freq=None, state=1, dualBlock=False):
        self.phase = phase
        self.freq = freq
        self.state = state
        self.alpha = alpha
        self.dualBlock = dualBlock
        Gate.__init__(self, agents)

    def updateAgents(self):
        ag  = self.agents[0]
        t = ag['_t']
        l = ag['piHalfLen']
        phase, freq, state = self.phase, self.freq, self.state
        phase += ml.getMultiLevels(ag, 'xy_phase', state)
        pulse = eh.mix(ag, eh.piHalfPulse(ag, t + (l / 2), phase=phase, alpha=self.alpha,
                                          state=state), freq, state=state)
        if self.dualBlock:
            ag['xy_s'] += pulse
        else:
            ag['xy'] += pulse
        ag['_t'] += l + ag['piBuffer']

    def _name(self):
        return "PiHalfPulse"

class RotPulse(Gate):
    def __init__(self, agents, angle, phase=0.0, freq=None, alpha=None, state=1):
        """
        similar with PiPulse,
        @param angle: the angle of rotation around axis
        """
        self.angle = angle
        self.alpha = alpha
        self.phase = phase
        self.freq = freq
        self.state = state
        Gate.__init__(self, agents)

    def updateAgents(self):
        ag = self.agents[0]
        t = ag['_t']
        l = ag['piLen']
        phase = self.phase + ml.getMultiLevels(ag, 'xy_phase', self.state)
        ag['xy'] += eh.mix(ag, eh.rotPulse(ag, t+l/2, angle=self.angle, phase=phase,
                                           alpha=self.alpha, state=self.state),
                           freq=self.freq, state=self.state)
        ag['_t'] += l + ag['piBuffer']

    def _name(self):
        f = self.angle/np.pi
        return "RotPulse({:3f})".format(f)

class RabiDrive(Gate):
    def __init__(self, agents, amp, tlen, w=None, state=1):
        """
        Rabi pulse
        @param agents: rabi pulse is applied to the first element, agents[0]
        @param amp: rabi amp
        @param tlen: length of rabi pulse
        @param w: rising width of the pulse, default is None, and it will take agents[0]['piFWHM']
        @param state: default is 1
        """
        self.amp = amp
        self.tlen = tlen
        self.w = w
        self.state = state
        Gate.__init__(self, agents)

    def updateAgents(self):
        ag = self.agents[0]
        t = ag['_t']
        tlen = self.tlen
        ag['xy'] += eh.mix(ag, eh.rabiPulse(ag, t, len=tlen, w=self.w, amp=self.amp, state=self.state),
                           state=self.state)
        ag['_t'] += tlen

    def _name(self):
        return "RabiDrive"

class NPiPulses(Gate):
    def __init__(self, agents, N, gap=0*ns, alpha=None, state=1, phase=0):
        """
        repeat PiPulse for N times,
        @param gap: gap between PiPulses, default is 0*ns
        @param N: the number of PiPulse
        @param alpha: the coefficient of DRAG
        @param state: the desired state
        @param phase: the phase of pipulses
        """
        self.N = N
        self.gap = gap
        self.alpha = alpha
        self.state = state
        self.phase = phase
        Gate.__init__(self, agents)

    def updateAgents(self):
        pass

    def setSubgates(self, agents):
        if self.gap <= 0:
            self.subgates = self.N*[PiPulse(agents, alpha=self.alpha,
                                            phase=self.phase, state=self.state)]
        else:
            self.subgates = (self.N-1) * [PiPulse(agents, alpha=self.alpha,
                                                  phase=self.phase, state=self.state),
                                          Wait(agents, self.gap)]
            self.subgates.append(PiPulse(agents, alpha=self.alpha, phase=self.phase,
                                         state=self.state))

    def _name(self):
        return "{} PiPulse".format(self.N)

class NPiHalfPulses(Gate):
    def __init__(self, agents, N, gap=0*ns, alpha=None, state=1, phase=0, amp=1.0):
        """
        repeat PiHalfPulse for N times,
        @param gap: gap between PiHalfPulses, default is 0*ns
        @param N: the number of PiPulse
        @param alpha: the coefficient of DRAG
        @param state: the desired state
        @param phase: the phase of piHalfpulses
        """
        self.N = N
        self.gap = gap
        self.alpha = alpha
        self.state = state
        self.phase = phase
        self.amp = amp
        Gate.__init__(self, agents)

    def updateAgents(self):
        pass

    def setSubgates(self, agents):
        if self.gap <= 0:
            self.subgates = self.N*[PiHalfPulse(agents, alpha=self.alpha,
                                                phase=self.phase,
                                                state=self.state)]
        else:
            self.subgates = (self.N-1) * [PiHalfPulse(agents, alpha=self.alpha,
                                                      phase=self.phase,
                                                      state=self.state),
                                          Wait(agents, self.gap)]
            self.subgates.append(PiHalfPulse(agents, alpha=self.alpha,
                                             phase=self.phase,
                                             state=self.state))

    def _name(self):
        return "{} PiHalfPulse".format(self.N)

class PingPong(Gate):
    def __init__(self, agents, N, alpha=None, phase=0.0, mode='piHalfPulse'):
        self.N = N
        self.alpha = alpha
        self.phase = phase
        self.mode = mode
        if mode == 'piHalfPulse':
            self.pigate = PiHalfPulse
        elif mode == 'piPulse':
            self.pigate = PiPulse
        Gate.__init__(self, agents)

    def updateAgents(self):
        pass

    def setSubgates(self, agents):
        N = self.N
        phase = self.phase
        alpha = self.alpha
        self.subgates = N * [self.pigate(agents, phase=phase, alpha=alpha),
                             self.pigate(agents, phase=phase+np.pi, alpha=alpha)]

    def _name(self):
        return "{} PingPong {}".format(self.N, self.mode)


class PiPulseZ(Gate):
    def __init__(self, agents):
        Gate.__init__(self, agents)

    def updateAgents(self):
        ag = self.agents[0]
        t = ag['_t']
        l = ag['piLenZ']
        ag['z'] += eh.piPulseZ(ag, t+l/2.0)
        ag['_t'] += l

    def _name(self):
        return "PiPulseZ"

class PiHalfPulseZ(Gate):
    def __init__(self, agents):
        Gate.__init__(self, agents)

    def updateAgents(self):
        ag = self.agents[0]
        t = ag['_t']
        l = ag['piLenZ']
        ag['z'] += eh.piHalfPulseZ(ag, t+l/2.0)
        ag['_t'] += l

    def _name(self):
        return "PiHalfPulseZ"

class NPiHalfPulsesZ(Gate):
    def __init__(self, agents, N):
        self.N = N
        Gate.__init__(self, agents)

    def updateAgents(self):
        pass

    def setSubgates(self, agents):
        self.subgates = self.N * [PiHalfPulseZ(agents)]

    def _name(self):
        return "{} PiHalfPulseZ".format(self.N)

class NPiPulsesZ(Gate):
    def __init__(self, agents, N):
        self.N = N
        Gate.__init__(self, agents)

    def updateAgents(self):
        pass

    def setSubgates(self, agents):
        self.subgates = self.N * [PiPulseZ(agents)]

    def _name(self):
        return "{} PiPulseZ".format(self.N)

class MoveToState(Gate):
    def __init__(self, agents, initState, endState, alpha=None, dualBlock=False):
        self.initState = initState
        self.endState = endState
        self.alpha = alpha
        self.dualBlock =  dualBlock
        Gate.__init__(self, agents)

    def updateAgents(self):
        pass

    def setSubgates(self, agents):
        initState = self.initState
        endState = self.endState
        if initState < endState:
            pi_pulses = range(initState+1, endState+1, 1)
        else:
            pi_pulses = range(initState, endState, -1)
        self.subgates = [PiPulse(agents, alpha=self.alpha, state=state, dualBlock=self.dualBlock) for state in pi_pulses]

    def _name(self):
        v = "MoveToState({0}->{1})".format(self.initState, self.endState)
        return v

class RFPiPulseZ(Gate):
    '''
    Z-axis pi pulse made with an X and Y pulse.
    '''
    def __init__(self, agents):
        Gate .__init__(self, agents)
    def updateAgents(self):
        pass
    def setSubGates(self, agents):
        self.subgates = [ PiPulse(agents, phase=0),
                          PiPulse(agents, phase=np.pi/2)]
    def _name(self):
        return "RFPiPulseZ"

class RFZPulse(Gate):
    '''
    Arbitrary Z pulse made with two pi-pulses
    '''
    def __init__(self, agents, angle):
        self.angle = angle
        Gate.__init__(self,agents)
    def updateAgents(self):
        pass
    def setSubGates(self, agents):
        self.subgates = [ PiPulse(agents, phase=0),
                          PiPulse(agents, phase=np.pi-self.angle/2) ]
    def _name(self):
        v = self.angle/np.pi
        return "RFZPulse({:.3f})".format(v)

class FastRFHadamard(Gate):
    """
    RF Hadamard with only two RF pulses
    """
    def __init__(self, agents):
        Gate.__init__(self, agents)

    def updateAgents(self):
        pass

    def setSubgates(self, agents):
        self.subgates = [PiHalfPulse(agents, phase=np.pi/2), PiPulse(agents)]

    def _name(self):
        return "FastRFHadamard"

class Spectroscopy(Gate):
    def __init__(self, agents, df=0.0, z=None, zlonger=0.0*ns):
        self.df = df
        self.z=z
        self.zlonger=zlonger
        # this allows for starting the z pulse earlier,
        # and ending it later than the spectroscopy pulse
        Gate.__init__(self, agents)

    def updateAgents(self):
        ag = self.agents[0]
        df=self.df
        z=self.z
        zlonger=self.zlonger
        ag['_t'] += zlonger
        t = ag['_t']
        tlen=ag['spectroscopyLen']
        ag['xy'] += eh.spectroscopyPulse(ag, t, df)
        if z is not None:
            # ag['z']  += env.rect(t-len-zlonger,len+2*zlonger,z) #set a z amplitude
            if zlonger>0.0:
                ag['z'] += env.flattop(t-zlonger, tlen+2*zlonger, w=zlonger/4, amp=z)
                #this way it's a little smooth around the start and end
            else:
                ag['z']  += env.rect(t-zlonger,tlen+2*zlonger,z) #set a z amplitude
        ag['_t'] += tlen
        ag['_t'] += zlonger

    def _name(self):
        return "Spectroscopy"

class DetuneFlattop(Gate):
    def __init__(self, agents, tlen=0.0, amp=0.0, w=1.0, overshoot=0.0):
        self.tlen=tlen
        self.amp=amp
        self.w = w
        self.overshoot = overshoot
        Gate.__init__(self, agents)

    def updateAgents(self):
        ag = self.agents[0]
        tlen = self.tlen
        amp = self.amp
        t = ag['_t']
        ag['z'] += env.flattop(t+self.w, tlen, w=self.w, amp=amp, overshoot=self.overshoot)
        # ag['z'] += env.rect(t-tlen, tlen, amp)
        ag['_t'] += tlen + 2*self.w
        phase = ag.get('detunePhase', 0.0)
        ag['xy_phase'] += phase

    def _name(self):
        return "DetuneFlattop"

class Detune(Gate):
    def __init__(self, agents, tlen=0.0, amp=0.0, overshoot=0.0, overshoot_w=1.0):
        self.tlen = tlen
        self.amp = amp
        self.overshoot = overshoot
        self.overshoot_w = overshoot_w
        Gate.__init__(self, agents)

    def updateAgents(self):
        ag = self.agents[0]
        tlen = self.tlen
        amp = self.amp
        t = ag['_t']
        ag['z'] += env.rect(t, tlen, amp, overshoot=self.overshoot,
                            overshoot_w=self.overshoot_w)
        ag['_t'] += tlen
        phase = ag.get('detunePhase', 0.0)
        ag['xy_phase'] += phase

    def _name(self):
        return "Detune"

class Wait(Gate):
    def __init__(self, agents, waitTime):
        self.waitTime = waitTime
        Gate.__init__(self, agents)

    def updateAgents(self):
        ag = self.agents[0]
        waitTime=self.waitTime
        t = ag['_t']
        ag['xy'] += env.wait(t, waitTime) # to get the gate windowing
        ag['_t'] += self.waitTime

    def _name(self):
        return "Wait"

class DualBlockWaitDetune(Gate):
    def __init__(self, agents, detuneAgent=None, tlen=0.0, w=0.0, amp=0.0):
        """
        Wait or Detune for tlen using Dual Block,
        if w>0, flattop will be used instead of rect
        tlen should be larger than 400*ns
        agents should be all the agents in the algorithm
        """
        assert tlen >= 400*ns
        self.tlen = tlen
        self.amp = amp
        self.w = w
        self.detuneAgent = detuneAgent or agents[0]
        Gate.__init__(self, agents)

    def updateAgents(self):
        tlen = self.tlen
        amp = self.amp
        w = self.w

        # Sync all agent
        t = max([ag['_t'] for ag in self.agents])
        for ag in self.agents:
            ag['_t'] = t
        # here is the idea:
        # the fist 200 ns is in block0, the last 200ns is in block1
        # and the remaining is in the dualblock delay
        # GHzFPGA server handles that when dualblock delay is not a multiply of 1024ns
        # the additional delay will be belong to the second block, block1
        if w>0*ns:
            env0 = env.flattop(t+w, 200*ns, w, amp)
            env1 = env.flattop(t+tlen+w-200*ns, 200*ns, w, amp)
        else:
            env0 = env.rect(t, 200*ns, amp)
            env1 = env.rect(t+tlen-200*ns, 200*ns, amp)
        self.detuneAgent['z'] += env0
        self.detuneAgent['z_s'] += env1

        # rebuild envelopes for all agents
        tNew = t + self.tlen + 2*w
        for agent in self.agents:
            agent['dualBlock'] = True
            agent['_t'] = tNew
            agent['xy_s'] += env.EnvZero(start=tNew, end=tNew)
            agent['z_s'] += env.EnvZero(start=tNew, end=tNew)

class Sync(Gate):
    def __init__(self, agents):
        Gate.__init__(self, agents)

    def updateAgents(self):
        t = max([ag['_t'] for ag in self.agents])
        for ag in self.agents:
            ag['_t'] = t

    def _name(self):
        return "Sync"

class EchoWait(Gate):
    def __init__(self, agents, waitTime, fringeFreq = None):
        self.waitTime = waitTime
        self.fringeFreq = fringeFreq
        Gate.__init__(self, agents)

    def updateAgents(self):
        pass

    def setSubgates(self, agents):
        ag = agents[0]
        piLen = ag['piLen']
        waitTime=self.waitTime
        dt = (waitTime - 2*piLen)/4.
        if self.fringeFreq:
            phase1 = np.pi*(self.fringeFreq['GHz']*self.waitTime['ns'])/2.0
            phase2 = 3*np.pi*(self.fringeFreq['GHz']*self.waitTime['ns'])/2.0
        else:
            phase1 = 0.0
            phase2 = 0.0
        if dt<0*ns:
            raise Exception("Echo dt < 0, overlapping pi pulses")
        self.subgates = [ Wait([ag], dt),
                          PiPulse([ag], phase=phase1),
                          Wait([ag], 2*dt),
                          PiPulse([ag], phase=phase2),
                          Wait([ag], dt) ]

    def _name(self):
        return "EchoWait"

class Echo(Gate):
    def __init__(self, agents, waitTime):
        self.waitTime = waitTime
        Gate.__init__(self, agents)

    def updateAgents(self):
        pass

    def setSubgates(self, agents):
        ag = agents[0]
        piLen = ag['piLen']
        waitTime=self.waitTime
        dt = (waitTime - piLen)/2.
        if dt<0*ns:
            raise Exception("Echo dt < 0")
        self.subgates = [Wait([ag], dt), PiPulse([ag]), Wait([ag], dt)]

    def _name(self):
        return "Echo"

class EmptyWait(Gate):
    def __init__(self, agents, waitTime):
        self.waitTime = waitTime
        Gate.__init__(self, agents)
    def updateAgents(self):
        ag = self.agents[0]
        ag['_t'] += self.waitTime
    def _name(self):
        return "EmptyWait"

class SetTime(Gate):
    def __init__(self, agents, t):
        self.t = t
        Gate.__init__(self, agents)
    def updateAgents(self):
        ag = self.agents[0]
        ag['_t'] = self.t
    def _name(self):
        try:
            t = self.t['ns']
        except:
            t = float(self.t)
        return "SetTime(t={:.3f} ns)".format(t)

class ChangePhase(Gate):
    def __init__(self, agents, xyPhase=0.0, state=1):
        self.xyPhase = xyPhase
        self.state = state
        Gate.__init__(self, agents)

    def updateAgents(self):
        ag = self.agents[0]
        key = ml.multiLevelKeyName('xy_phase', self.state)
        ag[key] += self.xyPhase

class Tomography(Gate):
    def __init__(self, agents, tomo, alpha = None):
        """
        tomo is a list with one operation for each qubit
        e.g. for one qubit: ['I'] OR ['X'] OR ['X/2']...
        for two qubits: ['I','I'] OR ['I','X'] OR ['X','I'] OR ['X','X']...
        self.tomoOps = {'I':(0,0),'X':(np.pi, 0),
                      'X/2':(np.pi/2.0,0), 'Y/2':(np.pi/2.0, np.pi/2.0),
                      '-X/2':(-np.pi/2.0,0), '-Y/2':(-np.pi/2.0, np.pi/2.0)}
        """
        self.tomoOps = {'I':lambda q: Wait([q], q['piLen']),
                        'X':lambda q: PiPulse([q], alpha=alpha),
                        'Y':lambda q: PiPulse([q], alpha=alpha, phase=np.pi/2.),
                        'X/2':lambda q: PiHalfPulse([q], alpha=alpha),
                        'Y/2':lambda q: PiHalfPulse([q], alpha=alpha, phase=np.pi/2.),
                        '-X':lambda q: PiPulse([q], alpha=alpha, phase = np.pi),
                        '-Y':lambda q: PiPulse([q], alpha=alpha, phase=3*np.pi/2.),
                        '-X/2':lambda q: PiHalfPulse([q], alpha=alpha, phase = np.pi),
                        '-Y/2':lambda q: PiHalfPulse([q], alpha=alpha, phase=3*np.pi/2.)}
        self.tomo = tomo
        self.alpha = alpha
        Gate.__init__(self, agents)

    def updateAgents(self):
        pass

    def setSubgates(self, agents):
        self.subgates = [Sync(agents)]
        for agent,op in zip(agents, self.tomo):
            self.subgates += [self.tomoOps[op](agent)]

class Tomography3(Gate):
    def __init__(self, agents, tomo, alpha=None):
        """
        Tomography for 3-level
        """
        self.tomoOps = {("I", "I"): lambda q: [Wait([q], q['piLen']),
                                               Wait([q], q['piLen'])],
                        ("X/2", "I"): lambda q: [PiHalfPulse([q], alpha=alpha),
                                                 Wait([q], q['piLen'])],
                        ("Y/2", "I"): lambda q: [PiHalfPulse([q], alpha=alpha, phase=np.pi/2),
                                                 Wait([q], q['piLen'])],
                        ("-X/2", "I"): lambda q: [PiHalfPulse([q], alpha=alpha, phase=np.pi),
                                                  Wait([q], q['piLen'])],
                        ("-Y/2", "I"): lambda q: [PiHalfPulse([q], alpha=alpha, phase=3*np.pi/2),
                                                  Wait([q], q['piLen'])],
                        ("X/2", "X"): lambda q: [PiHalfPulse([q], alpha=alpha),
                                                 PiPulse([q], state=2)],
                        ("Y/2", "X"): lambda q: [PiHalfPulse([q], alpha=alpha, phase=np.pi/2),
                                                 PiPulse([q], state=2)],
                        ("-X/2", "X"): lambda q: [PiHalfPulse([q], alpha=alpha, phase=np.pi),
                                                  PiPulse([q], state=2)],
                        ("-Y/2", "X"): lambda q: [PiHalfPulse([q], alpha=alpha, phase=3*np.pi/2),
                                                  PiPulse([q], state=2)],
                        ("X", "I"): lambda q: [PiPulse([q], alpha=alpha),
                                               Wait([q], q['piLen'])],
                        ("I", "X/2"): lambda q: [Wait([q], q['piLen']),
                                                 PiHalfPulse([q], state=2)],
                        ("I", "Y/2"): lambda q: [Wait([q], q['piLen']),
                                                 PiHalfPulse([q], phase=np.pi/2, state=2)],
                        ("I", "-X/2"): lambda q: [Wait([q], q['piLen']),
                                                  PiHalfPulse([q], phase=np.pi, state=2)],
                        ("I", "-Y/2"): lambda q: [Wait([q], q['piLen']),
                                                  PiHalfPulse([q], phase=3*np.pi/2, state=2)],
                        ("I", "X"): lambda q: [Wait([q], q['piLen']),
                                               PiPulse([q], state=2)],
                        ("X", "X/2"): lambda q: [PiPulse([q], alpha=alpha),
                                                 PiHalfPulse([q], state=2)],
                        ("X", "Y/2"): lambda q: [PiPulse([q], alpha=alpha),
                                                 PiHalfPulse([q], phase=np.pi/2.0, state=2)],
                        ("X", "-X/2"): lambda q: [PiPulse([q], alpha=alpha),
                                                  PiHalfPulse([q], phase=np.pi, state=2)],
                        ("X", "-Y/2"): lambda q: [PiPulse([q], alpha=alpha),
                                                  PiHalfPulse([q], phase=3*np.pi/2, state=2)],
                        ("X", "X"): lambda q: [PiPulse([q], alpha=alpha),
                                               PiPulse([q], state=2)]}
        self.tomo = tomo
        self.alpha = alpha
        super(Tomography3, self).__init__(agents)

    def updateAgents(self):
        pass

    def setSubgates(self, agents):
        self.subgates = [Sync(agents)]
        for agent, op in zip(agents, self.tomo):
            opfunc = self.tomoOps[op]
            g01, g02 = opfunc(agent)
            self.subgates += [g01, g02]

class MeasurePQ(Gate):
    def __init__(self, agents, state=1, sync=True):
        self.state = state
        self.sync = sync
        Gate.__init__(self, agents)

    def updateAgents(self):
        if self.sync:
            t_ = max(ag['_t'] for ag in self.agents)
            for ag in self.agents:
                ag['_t'] = t_
        for ag in self.agents:
            t = ag['_t']
            top = ag['measureLenTop']
            fall = ag['measureLenFall']
            ag['z'] += eh.measurePulse(ag, t, self.state)
            ag['_t'] += top + fall

    def _name(self):
        return "PhaseQubitMeasure |{}>".format(self.state)

class Readout(Gate):
    """
    We use env.shift here to ensure the start phase for readout pulse is zero, no matter
    how many readout pulse are used.
    This is because that the pulse adc get is extractly what readout dac outputs
    ignoring a hardware related phase.
    We do not need to care about when the readout dac outputs the pulse accoording to the xy-board.
    And also, this make the shift of the rr more easier.
    """
    def __init__(self, agents, name="DefaultReadout", ringup=True, ringdown=False):
        self.readoutName = name
        self.ringup = ringup
        self.ringdown = ringdown
        Gate.__init__(self, agents)

    def updateAgents(self):
        ag = self.agents[0]
        t = ag['_t']
        length = eh.fourNsCeil(ag['readoutLen'])
        readoutWidth = ag['readoutWidth']
        totalLength = length + 2*readoutWidth
        ag['readoutDemodLen'] = totalLength
        start = eh.fourNsCeil(t+readoutWidth) # to make sure the adc_window_start at 4ns s
        if self.ringup:
            ag['rr'] += env.shift(eh.readoutPulseRingup(ag, 0), start)
            # zero phase to compensate iq of ADC v7
        else:
            ag['rr'] += env.shift(eh.readoutPulse(ag, 0), start)

        adc_window_start = start
        adc_window_stop = adc_window_start + totalLength
        if self.readoutName in ag['adcReadoutWindows']:
            raise Exception("Trying to apply multiple readout windows for one qubit on a single ADC")
        ag['adcReadoutWindows'][self.readoutName] = (adc_window_start, adc_window_stop)
        ag['_t'] = start + length + readoutWidth # "start" include ag['_t'] and readoutWidth
        if self.ringdown:
            ag['rr'] += env.shift(eh.readoutRingdown(ag, totalLength), start)
        ag['_t'] += ag['readoutWidth'] + ag['readoutRingdownLen']

class Measure(Gate):
    def __init__(self, agents, name='DefaultReadout', align='start', ringdown=False):
        """
        align should be "start" or "end",
        align == "start" means that the readout pulses are aligned at start
        align == "end" means that the readout pulses are aligned at end
        When align at start, this gate does not sync at the end of the gate,
        so you should manually add Sync gate if it is needed.

        align == 'end' should be used more carefully, as I do not pay much attention
        to the code for this. check it before using align='end'.    -ZZX
        """
        self.name = name
        align = align.lower()
        assert align in ["start", "end"]
        self.align = align
        self.ringdown = ringdown
        super(Measure, self).__init__(agents)

    def updateAgents(self):
        pass

    def setSubgates(self, agents):
        subgates = []
        # sync at start
        # we just add readout gate for each qubit after sync qubits
        if self.align == 'start':
            subgates = [Sync(agents)] + [Readout([ag], name=self.name, ringdown=self.ringdown)
                                         for ag in agents]
        # readout gate is aligned at the end of the gate
        # this needs more carefully analysis
        # TODO: Fix it when ringdown is True.
        elif self.align == 'end':
            _ts = [ag["_t"] for ag in agents]
            demodLens = [ag["readoutLen"] + 2 * ag['readoutWith'] for ag in agents]
            max_end = max([_t + demod_len for _t, demod_len in zip(_ts, demodLens)])
            wait_time = [max_end - _t for _t in _ts]
            for wt, ag in zip(wait_time, agents):
                subgates.append(EmptyWait([ag], wt))
                subgates.append(Readout([ag], name=self.name))
            # this sync gate may be unnecessary, but to make sure.
            subgates.append(Sync(agents))
        self.subgates = subgates

class Herald(Gate):
    def __init__(self, agents, name='Herald', ringdown=True, align='start', sync=True):
        self.name = name
        align = align.lower()
        assert align in ["start", "end"]
        self.align = align
        self.sync = sync
        self.ringdown = ringdown
        super(Herald, self).__init__(agents)

    def updateAgents(self):
        pass

    def setSubgates(self, agents):
        subgates = []
        # align at start
        if self.align == 'start':
            subgates = [Measure(self.agents, name=self.name, ringdown=self.ringdown, align='start')]
            for ag in self.agents:
                delay = ag['heraldDelay'] - self.ringdown*(ag['readoutWidth']+ag['readoutRingdownLen'])
                subgates.append(Wait([ag], delay))
            if self.sync:
                subgates.append(Sync(self.agents))

        self.subgates = subgates


class ReadoutSpectroscopy(Gate):
    def __init__(self, agents, delay=0.0*ns, freq=None, ringdown=False):
        Gate.__init__(self, agents)
        self.delay = delay
        self.freq = freq
        self.ringdown = ringdown

    def updateAgents(self):
        ag = self.agents[0]
        t = ag['_t']
        readoutWidth = ag['readoutWidth']
        piLen = ag['piLen']
        totalLen = ag['readoutWidth']*2 + ag['readoutLen']
        ag['xy'] += eh.mix(ag, eh.piPulse(ag, t+piLen/2.0+self.delay), self.freq)

        # zero phase at t=0, shift w/o compensation adc v7
        start = np.ceil((t+readoutWidth)['ns']/4)*4*ns
        ag['rr'] += env.shift(eh.readoutPulseRingup(ag, 0), start)
        ag['_t'] = start + readoutWidth + ag['readoutLen']

        if self.ringdown:
            ag['rr'] += env.shift(eh.readoutRingdown(ag, totalLen), start)
            ag['_t'] += ag['readoutRingdownLen'] + readoutWidth

class ReadoutRingdown(Gate):
    def __init__(self, agents, tlen=None, amp=None, phase=None):
        self.tlen = tlen
        self.amp = amp
        self.phase = phase
        super(ReadoutRingdown, self).__init__(agents)

    def updateAgents(self):
        ag = self.agents[0]
        t = ag['_t']
        w = ag['readoutWidth']
        ag['rr'] += eh.readoutRingdown(ag, ag['_t']+w, tlen=self.tlen, amp=self.amp, phase=self.phase)
        ag['_t'] += self.tlen + 2*w

class Swap(Gate):
    def __init__(self, agents, paramName='', tlen=None, amp=None, w=None, overshoot=None):
        self.tlen = tlen
        self.w = w
        self.amp = amp
        self.overshoot = overshoot
        self.paramName = paramName
        self.swapTimeName = paramName + 'swapTime'
        self.swapAmpName = paramName + 'swapAmp'
        Gate.__init__(self, agents)

    def updateAgents(self):
        ag = self.agents[0]
        tlen = ag[self.swapTimeName] if self.tlen is None else self.tlen
        amp = ag[self.swapAmpName] if self.amp is None else self.amp
        w = ag.get(self.paramName + 'swapW', None) if self.w is None else self.w
        overshoot = ag.get(self.paramName + 'swapOS', 0) if self.overshoot is None else self.overshoot
        t = ag['_t']
        if w is None:
            ag['z'] += env.rect(t, tlen, amp, overshoot=overshoot)
            ag['_t'] += tlen
        else:
            ag['z'] += env.flattop(t+w, tlen, amp=amp, overshoot=overshoot)
            ag['_t'] += tlen+2*w
        ag['xy_phase'] += ag.get(self.paramName + 'swapPhase', 0.0)

    def _name(self):
        return "Swap({})".format(self.paramName)

class SqrtSwap(Gate):
    def __init__(self, agents, paramName='', tlen=None, amp=None, w=None, overshoot=None):
        self.tlen = tlen
        self.w = w
        self.amp = amp
        self.overshoot =  overshoot
        self.paramName = paramName
        self.sqrtSwapTimeName = paramName + 'sqrtSwapTime'
        self.sqrtSwapAmpName = paramName + 'sqrtSwapAmp'
        Gate.__init__(self, agents)

    def updateAgents(self):
        ag = self.agents[0]
        tlen = ag[self.sqrtSwapTimeName] if self.tlen is None else self.tlen
        amp = ag[self.sqrtSwapAmpName] if self.amp is None else self.amp
        w = ag.get(self.paramName + 'sqrtSwapW', None) if self.w is None else self.w
        overshoot = ag.get(self.paramName + 'sqrtSwapOS', 0) if self.overshoot is None else self.overshoot
        t = ag['_t']
        if w is None:
            ag['z'] += env.rect(t, tlen, amp, overshoot=overshoot)
            ag['_t'] += tlen
        else:
            ag['z'] += env.flattop(t+w, tlen, amp=amp, overshoot=overshoot)
            ag['_t'] += tlen+2*w
        ag['xy_phase'] += ag.get(self.paramName + 'sqrtSwapPhase', 0.0)

    def _name(self):
        return "SqrtSwap({})".format(self.paramName)

class QubitSwap(Gate):
    def __init__(self, agents, tlen=None, w=None, amp=None, overshoot=None):
        """
        swap from agents[0] to agents[1]
        z-bias on agents[0]
        """
        self.tlen = tlen
        self.w = w
        self.amp = amp
        self.overshoot = overshoot
        self.paramName = "{q0}-{q1}".format(q0=agents[0].__name__, q1=agents[1].__name__)
        Gate.__init__(self, agents)

    def updateAgents(self):
        ag0 = self.agents[0]
        ag1 = self.agents[1]
        paramName = self.paramName
        swapTimeName = paramName + '-swapTime'
        swapAmpName = paramName + '-swapAmp'
        swapWName = paramName + "-swapW"
        swapOSName = paramName + "-swapOS"
        tlen = ag0[swapTimeName] if self.tlen is None else self.tlen
        amp = ag0[swapAmpName] if self.amp is None else self.amp
        w = ag0.get(swapWName, None) if self.w is None else self.w
        overshoot = ag0.get(swapOSName, 0) if self.overshoot is None else self.overshoot
        t = ag0['_t']
        if w is None:
            ag0['z'] += env.rect(t, tlen, amp, overshoot=overshoot)
            ag0['_t'] += tlen
        else:
            ag0['z'] += env.flattop(t+w, tlen, amp=amp, overshoot=overshoot)
            ag0['_t'] += tlen+2*w
        ag0['xy_phase'] += ag0.get(paramName + '-swapPhase', 0.0)

    def _name(self):
        return "QubitSwap({})".format(self.paramName)

class QubitSqrtSwap(Gate):
    def __init__(self, agents, tlen=None, w=None, amp=None, overshoot=None):
        """
        sqrtSwap from agents[0] to agents[1]
        z bias on agents[0]
        """
        ag0 = agents[0]
        ag1 = agents[1]
        self.tlen = tlen
        self.w = w
        self.amp = amp
        self.overshoot =  overshoot
        self.paramName = "{q0}-{q1}".format(q0=ag0.__name__, q1=ag1.__name__)
        Gate.__init__(self, agents)

    def updateAgents(self):
        ag0 = self.agents[0]
        ag1 = self.agents[1]
        sqrtSwapTimeName = self.paramName + '-sqrtSwapTime'
        sqrtSwapAmpName = self.paramName + '-sqrtSwapAmp'
        sqrtSwapWName = self.paramName + '-sqrtSwapW'
        sqrtSwapOSName = self.paramName + "-sqrtSwapOS"
        tlen = ag0[sqrtSwapTimeName] if self.tlen is None else self.tlen
        amp = ag0[sqrtSwapAmpName] if self.amp is None else self.amp
        w = ag0.get(sqrtSwapWName, None) if self.w is None else self.w
        overshoot = ag0.get(sqrtSwapOSName, 0) if self.overshoot is None else self.overshoot
        t = ag0['_t']
        if w is None:
            ag0['z'] += env.rect(t, tlen, amp, overshoot=overshoot)
            ag0['_t'] += tlen
        else:
            ag0['z'] += env.flattop(t+w, tlen, amp=amp, overshoot=overshoot)
            ag0['_t'] += tlen+2*w
        ag0['xy_phase'] += ag0.get(self.paramName + '-sqrtSwapPhase', 0.0)

    def _name(self):
        return "QubitSqrtSwap({})".format(self.paramName)

class FullSwap(Gate):
    def __init__(self, agents, delay=None, sync=False):
        self.delay = delay
        self.sync = sync
        Gate.__init__(self, agents)

    def updateAgents(self):
        pass

    def setSubgates(self, agents):
        ag0 = agents[0]
        ag1 = agents[1]
        if self.delay is None:
            delay = ag0['piFWHM']
        else:
            delay = self.delay
        self.subgates = [QubitSwap([ag0, ag1]),
                         Wait([ag0], delay),
                         QubitSqrtSwap([ag0, ag1])]
        if self.sync:
            self.subgates.append(Sync([ag0, ag1]))

    def _name(self):
        return "FullSwap({})".format(self.paramName)

class iSwapBus(Gate):
    def __init__(self, agents, delay=None):
        """
        iSwap through Bus, qubit0 -> Bus -> qubit1
        @param delay: delay between qubit0-> Bus and Bus->qubit1
        @param paramName: parameter name of registry key
        """
        self.delay = delay
        Gate.__init__(self, agents)

    def updateAgents(self):
        pass

    def setSubgates(self, agents):
        ag0 = agents[0]
        ag1 = agents[1]
        self.subgates = [
            QubitSwap([ag0, ag1]),
            Wait([ag0], self.delay),
            Sync([ag0, ag1]),
            QubitSwap([ag1, ag0]),
            Sync([ag0, ag1])
         ]

    def _name(self):
        return "iSwapBus({0}-{1})".format(self.agents[0].__name__, self.agents[1].__name__)

class Pulseshape(Gate):
    def __init__(self, agents, offset=None, probeTime=None,
                 stepHeight=None, stepLen=None):
        self.offset = offset
        self.stepHeight = stepHeight
        self.stepLen = stepLen
        self.probeTime = probeTime
        Gate.__init__(self, agents)

    def updateAgents(self):
        # Rename variables
        ag = self.agents[0]
        offset = self.offset
        stepLen = self.stepLen
        stepHeight = self.stepHeight
        piLen = ag['piLen']
        offsetLen = self.probeTime + piLen
        # Square z pulses
        totalLen = offsetLen + stepLen
        tStart = ag['_t']
        ag['z'] += env.rect(tStart, stepLen, stepHeight + offset)
        ag['z'] += env.rect(tStart + stepLen, offsetLen, offset)
        # probe pulse
        probeTime = tStart + stepLen + self.probeTime
        ag['xy'] += eh.mix(ag, eh.rotPulse(ag, probeTime))
        # Update agent timer
        ag['_t'] += totalLen

    def _name(self):
        return "PulseShape"

class ACStark(Gate):
    def __init__(self, agents, ringUp=1000*ns, buffer=100*ns, amp=None):
        Gate.__init__(self, agents)
        self.ringUp = ringUp
        self.buffer = buffer
        self.amp = amp

    def updateAgents(self):
        ag = self.agents[0]
        t = ag['_t']
        piLen = ag['piLen']
        readoutWidth = ag['readoutWidth']
        totalLength = self.ringUp + piLen + self.buffer
        carrierFrequency = ag['readoutDevice']['carrierFrequency']
        df = ag['readoutFrequency'] - carrierFrequency
        ag['rr'] += env.mix(env.flattop(t, totalLength, w=readoutWidth, amp=self.amp), df)
        ag['xy'] += eh.mix(ag, eh.piPulse(ag, t + self.ringUp + piLen / 2))
        ag['_t'] += totalLength

class TestDelayZ(Gate):
    def __init__(self, agents, tshift=0.0, zpa=0.0, zpl=0.0):
        self.tshift=tshift
        self.zpa=zpa
        self.zpl=zpl
        Gate.__init__(self, agents)

    def updateAgents(self):
        """Modify this qubit's time and add the pulse data"""
        ag = self.agents[0]
        tshift = self.tshift
        zpa = self.zpa
        zpl = self.zpl
        t = ag['_t']
        l = ag['piLen']
        ag['xy'] += eh.mix(ag, eh.piPulse(ag, t+(l/2)))
        ag['z'] += env.rect( t+l/2.0-zpl/2.0+tshift, zpl, zpa)
        # since the pi pulse is centered at t+l/2,
        # we need to center this rectangular window around t+l/2 too
        ag['_t'] += l

    def _name(self):
        return "TestDelayZ"

class TestDelayBusSwap(Gate):
    def __init__(self, agents, tshift=0*ns, paramName='', delay=0*ns):
        self.tshift = tshift
        self.paramName = paramName
        self.delay = delay
        Gate.__init__(self, agents)

    def updateAgents(self):
        agD = self.agents[0] # drive qubit
        agM = self.agents[1] # measure qubit
        t = max(agD['_t'], agM['_t'])
        l = agD['piLen']
        pname = self.paramName
        # first drive the drive qubit to |1>
        agD['xy'] += eh.piPulse(agD, t+l/2.0)
        # swap into resonator
        agD['z'] += env.rect(t+l, agD['swapTime'+pname], agD['swapAmp'])
        # measure qubit swap with the resonator
        t += l + agD['swapTime'+pname] + self.delay + self.tshift
        agM['z'] += env.rect(t, agM['swapTime'+pname], agD['swapAmp'+pname])
        t += agM['swapTime'+pname]
        agM['_t'] = t
        agD['_t'] = t

    def _name(self):
        return "TestDelayBusSwap"

class cZControlSwap(Gate):
    def __init__(self, agents, tlen=None, amp=None, w=None, overshoot=None):
        self.tlen = tlen
        self.w = w
        self.amp = amp
        self.overshoot = overshoot
        Gate.__init__(self, agents)

    def updateAgents(self):
        ag = self.agents[0]
        tlen = ag['cZControlLen'] if self.tlen is None else self.tlen
        amp = ag['cZControlAmp'] if self.amp is None else self.amp
        w = ag.get('cZControlW', None) if self.w is None else self.w
        overshoot = ag.get('cZControlOS', 0) if self.overshoot is None else self.overshoot
        t = ag['_t']
        if w is None:
            ag['z'] += env.rect(t, tlen, amp, overshoot=overshoot)
            ag['_t'] += tlen
        else:
            ag['z'] += env.flattop(t+w, tlen, amp=amp, overshoot=overshoot)
            ag['_t'] += tlen+2*w
        phase = ag.get('cZControlPhaseCorr', 0.0)
        ag['xy_phase'] += phase

    def _name(self):
        return "cZControlSwap"

class cZTargetSwap(Gate):
    def __init__(self, agents, tlen=None, amp=None, w=None, overshoot=None):
        self.tlen = tlen
        self.w = w
        self.amp = amp
        self.overshoot = overshoot
        Gate.__init__(self, agents)

    def updateAgents(self):
        ag = self.agents[0]
        tlen = ag['cZTargetLen'] if self.tlen is None else self.tlen
        amp = ag['cZTargetAmp'] if self.amp is None else self.amp
        w = ag.get('cZTargetW', None) if self.w is None else self.w
        overshoot = ag.get('cZTargetOS', 0) if self.overshoot is None else self.overshoot
        t = ag['_t']
        if w is None:
            ag['z'] += env.rect(t, tlen, amp, overshoot=overshoot)
            ag['_t'] += tlen
        else:
            ag['z'] += env.flattop(t+w, tlen, amp=amp, overshoot=overshoot)
            ag['_t'] += tlen+2*w
        phase = ag.get('cZTargetPhaseCorr', 0.0)
        ag['xy_phase'] += phase

    def _name(self):
        return "cZTargetSwap"

class CZ(Gate):
    def __init__(self, agents):
        """
        cZ Gate in the system,  qubit <-> Bus <-> qubit
        realized by iswap
        @param agents: list, agents[0] is the control qubit
                             agents[1] is the target qubit
        """
        Gate.__init__(self, agents)

    def updateAgents(self):
        qC = self.agents[0]
        qT = self.agents[1]
        t = max(qC['_t'], qT['_t'])
        # iSwap the control qubit to the resonator
        qC['z'] += env.rect(t, qC['cZControlLen'], qC['cZControlAmp'])
        t += qC['cZControlLen']
        # Do (iSwap)**2 on target
        qT['z'] += env.rect(t, qT['cZTargetLen'], qT['cZTargetAmp'])
        t += qT['cZTargetLen']
        # compensate target phase
        qT['z'] += env.rect(t, qT['cZTargetPhaseCorrLen'], qT['cZTargetPhaseCorrAmp'])
        # Retrieve photon from resonator into control
        qC['z'] += env.rect(t, qC['cZControlLen'], qC['cZControlAmp'])
        # Compensate control phase
        qC['z'] += env.rect(t+qC['cZControlLen'], qC['cZControlPhaseCorrLen'], qC['cZControlPhaseCorrAmp'])
        t += max((qC['cZControlLen'] + qC['cZControlPhaseCorrLen'],
                  qT['cZTargetPhaseCorrLen']))
        qC['_t'] = t
        qT['_t'] = t

    def _name(self):
        return "cZGate"

class CZ2(Gate):
    def __init__(self, agents):
        """
        cZ Gate v2 in the system,  qubit <-> Bus <-> qubit
        realized by iswap
        @param agents: list, agents[0] is the control qubit
                             agents[1] is the target qubit
        """
        Gate.__init__(self, agents)

    def updateAgents(self):
        qC = self.agents[0]
        qT = self.agents[1]
        t = max(qC['_t'], qT['_t'])
        # iSwap the control qubit to the resonator
        qC['z'] += env.rect(t, qC['cZControlLen'], qC['cZControlAmp'])
        t += qC['cZControlLen']
        qC['xy_phase'] += qC['cZControlPhaseCorr']
        # Do (iSwap)**2 on target
        qT['z'] += env.rect(t, qT['cZTargetLen'], qT['cZTargetAmp'])
        t += qT['cZTargetLen']
        qT['xy_phase'] += qT['cZTargetPhaseCorr']
        # Retrieve photon from resonator into control
        qC['z'] += env.rect(t, qC['cZControlLen'], qC['cZControlAmp'])
        t += qC['cZControlLen']
        # # Compensate target phase
        # qT['z'] += env.rect(t, qT['cZTargetPhaseCorrLen'], qT['cZTargetPhaseCorrAmp'])
        # # Compensate control phase
        # qC['z'] += env.rect(t+qC['cZControlLen'], qC['cZControlPhaseCorrLen'], qC['cZControlPhaseCorrAmp'])
        # t += max((qC['cZControlLen'] + qC['cZControlPhaseCorrLen'],
        #           qT['cZTargetPhaseCorrLen']))
        qC['_t'] = t
        qT['_t'] = t

    def _name(self):
        return "cZGate v2"

class CNOT(Gate):
    """
    CNOT = Ry(-Pi/2) on target -> CZ -> Ry(Pi/2) on target
    """
    def __init__(self, agents, sync=True):
        """
        CNOT Gate
        @param agents: list of agents,
                       agents[0] is the control qubit,
                       agents[1] is the target qubit
        @param sync: bool,
                     if True, it will sync qubit before and after CNOT
        """
        self.sync = sync
        Gate.__init__(self, agents)

    def updateAgents(self):
        pass

    def setSubgates(self, agents):
        control, target = agents[0], agents[1]
        self.subgates = [
            PiHalfPulse([target], phase=-np.pi/2),
            Sync([target, control]),
            CZ([control, target]),
            Sync([target, control]),
            PiHalfPulse([target], phase=np.pi/2)
        ]
        if self.sync:
            self.subgates = [Sync([target, control])] + self.subgates + [Sync([target, control])]

    def _name(self):
        return "CNOT"

class CNOT2(Gate):
    """
    CNOT = Ry(-Pi/2) on target -> CZ -> Ry(Pi/2) on target
    """
    def __init__(self, agents, sync=True):
        """
        CNOT Gate v2
        @param agents: list of agents,
                       agents[0] is the control qubit,
                       agents[1] is the target qubit
        @param sync: bool,
                     if True, it will sync qubit before and after CNOT
        """
        self.sync = sync
        Gate.__init__(self, agents)

    def updateAgents(self):
        pass

    def setSubgates(self, agents):
        control, target = agents[0], agents[1]
        self.subgates = [
            PiHalfPulse([target], phase=-np.pi/2),
            Sync([target, control]),
            CZ2([control, target]),
            Sync([target, control]),
            PiHalfPulse([target], phase=np.pi/2)
        ]
        if self.sync:
            self.subgates = [Sync([target, control])] + self.subgates + [Sync([target, control])]

    def _name(self):
        return "CNOT v2"

class RBCliffordSingleQubit(Gate):
    def __init__(self, agents, gateList, alpha=None):
        """
        build RB gate sequence for single qubit, convert a list of gate string to the gate
        @param gateList: a list of string of gate
        @param alpha: alpha of DRAG
        """
        self.gateList = gateList
        self.gateOps = {
            'I': lambda q: Wait([q], q.get('identityLen', q['piLen'])),
            'IW': lambda q: Wait([q], q['identityWaitLen']),
            'IWSE': lambda q: EchoWait([q], q['identityWaitLen']),
            'SE': lambda q: Echo([q], q['identityWaitLen']),
            'IGN': lambda q: Wait([q], 0 * ns),
            'X': lambda q: PiPulse([q], alpha=alpha),
            'Y': lambda q: PiPulse([q], alpha=alpha, phase=np.pi/2.),
            'X/2': lambda q: PiHalfPulse([q], alpha=alpha),
            'Y/2': lambda q: PiHalfPulse([q], alpha=alpha, phase=np.pi/2.),
            '-X': lambda q: PiPulse([q], alpha=alpha, phase=np.pi),
            '-Y': lambda q: PiPulse([q], alpha=alpha, phase=3*np.pi/2.),
            '-X/2': lambda q: PiHalfPulse([q], alpha=alpha, phase=np.pi),
            '-Y/2': lambda q: PiHalfPulse([q], alpha=alpha, phase=3*np.pi/2.),
            'H': lambda q: FastRFHadamard([q]),
            'Z': lambda q: Detune([q]),
            'Zpi': lambda q: PiPulseZ([q]),
            'Zpi/2': lambda q: PiHalfPulseZ([q]),
        }
        super(RBCliffordSingleQubit, self).__init__(agents)

    def setSubgates(self, agents):
        ag = agents[0]
        subgates = []
        for gate in self.gateList:
            for sq_op in gate[0]:
                subgates.append(self.gateOps[sq_op](ag))
        self.subgates = subgates

    def updateAgents(self):
        pass

class RBCliffordMultiQubit(Gate):
    def __init__(self, agents, gateList, alphaList, sync=True):
        """
        build gate from gateList for multiQubit.
        @param agents: qubits
        @param gateList: a list of gate string, should be the same format of rbClass.randGen
        @param alphaList: a list of alpha, [alpha for q0, alpha for q1, ... ]
        @param sync: sync for each clifford gate, default is True
        """
        self.gateList = gateList
        self.sync = sync
        self.alphaDict = {}
        for idx, ag in self.agents:
            self.alphaDict[ag.__name__] = alphaList[idx]
        self.gateOps = {
            'I': lambda q: Wait([q], q.get('identityLen', np.min([q['piLen'], q['piHalfLen']]) * ns)),
            'IW': lambda q: Wait([q], q['identityWaitLen']),
            'IWSE': lambda q: EchoWait([q], q['identityWaitLen']),
            'SE': lambda q: Echo([q], q['identityWaitLen']),
            'IGN': lambda q: Wait([q], 0 * ns),
            'X': lambda q: PiPulse([q], alpha=self.alphaDict[q.__name__]),
            'Y': lambda q: PiPulse([q], alpha=self.alphaDict[q.__name__], phase=np.pi/2.),
            'X/2': lambda q: PiHalfPulse([q], alpha=self.alphaDict[q.__name__]),
            'Y/2': lambda q: PiHalfPulse([q], alpha=self.alphaDict[q.__name__], phase=np.pi/2.),
            '-X': lambda q: PiPulse([q], alpha=self.alphaDict[q.__name__], phase=np.pi),
            '-Y': lambda q: PiPulse([q], alpha=self.alphaDict[q.__name__], phase=3 * np.pi/2.),
            '-X/2': lambda q: PiHalfPulse([q], alpha=self.alphaDict[q.__name__], phase=np.pi),
            '-Y/2': lambda q: PiHalfPulse([q], alpha=self.alphaDict[q.__name__], phase=3*np.pi/2.),
            'H': lambda q: FastRFHadamard([q]),
            'Z': lambda q: Detune([q]),
            'Zpi': lambda q: PiPulseZ([q]),
            'Zpi/2': lambda q: PiHalfPulseZ([q]),
            "CZ": lambda q1, q2: CZ([q1, q2]),
            "CNOT": lambda q1, q2: CNOT([q1, q2])
        }
        super(RBCliffordMultiQubit, self).__init__(agents)

    def updateAgents(self):
        pass

    def setSubgates(self, agents):
        subgates = []
        twoQubitGates = ["CZ", "CNOT"]
        for cliffordGate in self.gateList:
            ops = rb.gate2OpList(len(agents), cliffordGate)
            for op in ops:
                twoQubitGatePresent = any([twoQubitGateElem in op for twoQubitGateElem in twoQubitGates])
                if twoQubitGatePresent:
                    subgates.append(self.gateOps[op](agents))
                else:
                    for sq_op, ag in zip(op, agents):
                        subgates.append(self.gateOps[sq_op](ag))
            if self.sync:
                subgates.append(Sync(agents))
        self.subgates = subgates
