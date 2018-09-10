# Copyright (C) 2012  Daniel Sank / Julian Kelly
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


from pyle import envelopes as env
from pyle.dataking import envelopehelpers as eh
from pyle.registry import AttrDict

from labrad.units import Unit, Value
import copy
import numpy as np

ns, GHz, us = (Unit(un) for un in ['ns', 'GHz', 'us'])

###############
### DEVICES ###
###############

def AttrDict2Qubit(q, qubitType):
    """Convert from a pure registry.AttrDict to a Qubit"""
    qubit = qubitType()
    #Copy all values from q into qubit
    for key,value in q.items():
        qubit[key] = value
    object.__setattr__(qubit, '__name__', q.__name__)
    return qubit

class Agent(AttrDict):
    def __init__(self):
        AttrDict.__init__(self)
        self['_t'] = 0.0*ns
        self['gates'] = []

    def __str__(self):
        return self.__name__

class Qubit(Agent):
    """
    Each Qubit object has a current time, self._t which is the current
    start time of this qubit's sequence.
    """
    def __init__(self):
        Agent.__init__(self)

    def excite(self):
        t = self['_t']
        l = self['piLen']
        self['xy'] += eh.mix(self, eh.piPulse(self, t+(l/2)))
        self['_t'] += l

    def measure(self):
        raise Exception('Not implemented')

class PhaseQubit(Qubit):
    pass

class Transmon(Qubit):
    def __init__(self):
        Qubit.__init__(self)
        self['adcReadoutWindows'] = {}

class Resonator(Qubit):
    def __init__(self):
        Agent.__init__(self)

    def measure(self):
        pass

class Readout(Agent):
    pass

class SimpleReadout(Readout):
    def __init__(self):
        Agent.__init__(self)
        self['qubits'] =  list()

    def measure(self, *args, **kwargs):
        pass

CONSTRUCTORS = {
    'phaseQubit': PhaseQubit,
    'transmon': Transmon,
    'resonator': Resonator,
    'simpleReadout': SimpleReadout,
}

def attrDict2Agent(deviceAsAttrDict):
    devType = deviceAsAttrDict['_type']
    dev = CONSTRUCTORS[devType]()
    for key, value in deviceAsAttrDict.items():
        # dev[key] = copy.deepcopy(value)
        dev[key] = value
    object.__setattr__(dev, '__name__', deviceAsAttrDict.__name__)
    object.__setattr__(dev, '_dir', deviceAsAttrDict._dir)
    return dev

def agent4gateComplier(sample, qubits, measure):
    config_names = sample['config']
    # qubit_names = [q.__name__ for q in qubits]
    measure_name = [ config_names[i] for i in measure ]
    qubits = copy.deepcopy(qubits) # make a deep copy of qubits
    devices = dict([(q.__name__, attrDict2Agent(q)) for q in qubits])
    readoutDevices = dict([(name, dev) for name, dev in devices.items() if isinstance(dev, Readout)])
    for dev in readoutDevices.values():
        dev['qubits'] = list()

    # add readoutDevice in qubits and add qubits to readoutDevice
    for name, dev in devices.items():
        readoutConfig = dev.get('readoutConfig', None)
        if readoutConfig:
            dev['readoutDevice'] = devices[readoutConfig]
            devices[readoutConfig]['qubits'].append(name)

    # add readout channels to the readoutDevice
    for name in measure_name:
        qubit = devices[name]
        # keep compatible with phase qubit measurement.
        rc = qubit.get('readoutConfig', None)
        if rc :
            readoutDevice = qubit['readoutDevice']
            readoutDevice['channels'].append(('readout-%s' %name, ("ADC", [readoutDevice['readout ADC']])))

    devs_list = [ devices[key] for key in config_names ]
    return devs_list

def loadQubits(Sample, measure, write_access=False):
    """
    New version of loadQubits, add 'readout'=True into the device
    @param Sample: RegistryWrapper (in pyle.util.registry_wrapper2)
    @param measure: can be integer or list of integer
    @param write_access: bool
    @return:
        if write_access: (sample, devices, qubits, Qubits)
        else: (sample, devices, qubits)
    """
    Qubits = [ Sample[q] for q in Sample['config'] ] # RegistryWrappers (writeable)
    sample = Sample.copy() # AttrDict, copy of registry
    qubits = [ sample[q] for q in sample['config'] ] # AttrDicts
    if isinstance(measure, (int, long)):
        measure = [measure]
    config =  sample['config']
    measureNames = [config[m] for m in measure]
    sample['measureNames'] = measureNames
    # the inputOrder shows in the sample
    # the inputOrder is given in measure
    for idx, m in enumerate(measure):
        qubits[m]['inputOrder'] = idx
        qubits[m]['readout'] = True
    # the readoutOrder is given in config
    for idx, m in enumerate(sorted(measure)):
        qubits[m]['readoutOrder'] = idx
    devices = agent4gateComplier(sample, qubits, measure)
    if write_access:
        return sample, devices, qubits, Qubits
    else:
        return sample, devices, qubits

def loadDeviceType(Sample, deviceType, measure, write_access=False):
    """
    New version of loadDeviceType
    this function need to be checked, typically, using loadQubits is a better choice.
    """
    raise RuntimeWarning("This function should be checked and "
                         "may not be compatiable with gateCompiler")
    Devices=[]
    devices=[]
    deviceNames = Sample['config']
    # First get writeable Devices
    # RegistryWrappers
    for deviceName in deviceNames:
        if Sample[deviceName]['_type'] == deviceType:
            Devices.append(Sample[deviceName])
    # Now make the unwritable devices
    sample = Sample.copy()
    if isinstance(measure, (int, long)):
        measure = [measure]
    for deviceName in deviceNames:
        if sample[deviceName]['_type'] == deviceType:
            devices.append(sample[deviceName])
    for idx, m in enumerate(measure):
        devices[m]['readoutOrder'] = idx
        devices[m]['readout'] = True
    devicesAgents = agent4gateComplier(sample, devices, measure)
    if write_access:
        return sample, devicesAgents, Devices
    else:
        return sample, devices

##################
### ALGORITHMS ###
##################

class Algorithm(object):

    def __init__(self, agents):
        """
        agents are specified in the constructor
        we make a local copy of them to prevent overwrite error
        we also set a flag to prevent adding an unknown agent.
        """
        self.gates = []
        self.preset_agent = True
        self.agents = copy.deepcopy(agents)
        self.agents_dict = {ag.__name__: ag for ag in self.agents}
        self._update_attr()
        self.compiled = False

    def _update_attr(self):
        """
        this function allow us to get the agent by alg.q0, alg.r0, ...
        if the agent is a resonator, use alg.r0, alg.r1, ...
        if the agent is a qubit, use alg.q0, alg.q1, ...
        the order of q0, q1, (or r0, r1) is specified in the measure,
        for example, measure = [0, 1]
        then q0 is qubit[measure[0]], q1 is qubit[measure[1]]
        measure = [1,0]

        resonator uses different index with qubit
        """
        N = len([ag for ag in self.agents if ag.get('readout', False)])
        self.qubits = [[] for i in range(N)]
        self.resonators = list()
        self.readout_devs = list()
        for ag in self.agents:
            if isinstance(ag, (PhaseQubit, Transmon)):
                q_idx = ag.get('inputOrder', None)
                if q_idx is not None:
                    setattr(self, "q{}".format(q_idx), ag)
                    self.qubits[q_idx] = ag
            elif isinstance(ag, Resonator):
                r_idx = ag.get('inputOrder', None)
                if r_idx is not None:
                    setattr(self, "r{}".format(r_idx), ag)
                    self.resonators.append(ag)
            elif isinstance(ag, Readout):
                self.readout_devs.append(ag)
            else:
                raise Exception("Unrecognized Device")


    def __getitem__(self, gate):
        if self.compiled:
            raise RuntimeError("Add a gate to a compiled algorithm")
        for agent in gate.agents:
            if self.preset_agent:
                if agent not in self.agents_dict.values():
                    raise RuntimeError("attemped to add gate with an unknown agent")
            else:
                self.agents_dict[agent.__name__] = agent
                self.agents.append(agent)
        #Add this gate to our list of gates.
        self.gates.append(gate)
        return self

    def compile(self, correctXtalkZ=False, config=None):
        """
        Compile the algorithm,
        updating all participating agents with relevant sequence data
        @param config, default is None. if not config should given in a way in sample['config']
                        for example, config = ['q1', 'r1', 'q3']...
                        resonator is ignore here.
        @param correctXtalkZ, default is False, if True, z correction is applied.
        """
        if self.compiled:
            raise RuntimeError("Algorithm compiled twice")
        self.compiled = True

        #Initialize all participating agents
        for agent in self.agents:
            agent['_t'] = 0.0*ns
            #set all envelopes to zero
            agent['xy'] = env.NOTHING
            agent['z'] = env.NOTHING
            agent['rr'] = env.NOTHING
            agent['xy_phase'] = 0.0 # phase for |1>
            agent['xy_phase2'] = 0.0 # phase for |2>
            agent['xy_s'] = env.NOTHING
            agent['z_s'] = env.NOTHING
            # xy_phase is for compensating for z-phase shift, e.g. from a swap


        # the code below is for gate compiled backwards,
        # and since it is not easy for me to understand why we should do in that way
        # so I change this behavior, and all the gate should be changed. 2016.7.20, ZZX
        # for gate in self.gates[::-1]:
        #     gate.compile(self.__agent_list)
        #
        # # Find the longest sequence and shift all sequences to the right by that amount.
        # tShift = -1*min([ag.get('_t', 0.0*ns)['ns'] for ag in self.agents.values()])*ns
        #
        # for agent in self.agents.values():
        #     agent['xy'] = env.shift(agent['xy'], dt=tShift)
        #     agent['z']  = env.shift(agent['z'],  dt=tShift)
        #     agent['rr'] = env.shift(agent['rr'], dt=agent.get('readoutLen', 0*ns))
        #     # Store the time of the end of coherence manipulations, ie. the time
        #     # at which the redaout pulse starts. Note that this is shifted to
        #     # the right by tShift.
        #     agent['_t'] = tShift*ns

        for gate in self.gates:
            gate.compile(self.agents)

        for agent in self.agents:
            if agent['xy_s'] is env.NOTHING:
                agent['xy_s'] = [agent['xy']]
            else:
                agent['xy_s'] = [agent['xy'], agent['xy_s']]
            if agent['z_s'] is env.NOTHING:
                agent['z_s'] = [agent['z']]
            else:
                agent['z_s'] = [agent['z'], agent['z_s']]

        xyenvs = [ag['xy'] for ag in self.agents]
        zenvs = [ag['z'] for ag in self.agents]
        renvs = [ag['rr'] for ag in self.agents]
        all_envs = sum([xyenvs, zenvs, renvs], [])
        tShift_env = -min([p.start for p in all_envs if p.start is not None])
        tShift_env = eh.fourNsCeil(tShift_env*ns)

        for agent in self.agents:
            if isinstance(agent, Transmon):
                agent['xy'] = env.shift(agent['xy'], dt=tShift_env)
                agent['z']  = env.shift(agent['z'],  dt=tShift_env)
                agent['rr'] = env.shift(agent['rr'], dt=tShift_env)
                agent['xy_s'] = [env.shift(x, dt=tShift_env) for x in agent['xy_s']]
                agent['z_s']  = [env.shift(x,  dt=tShift_env) for x in agent['z_s']]
                shiftedWindows = {}
                if 'adcReadoutWindows' in agent:
                    for key, window in agent['adcReadoutWindows'].items():
                        shiftedWindows[key] = (window[0]+tShift_env, window[1]+tShift_env)
                    agent['adcReadoutWindows'] = shiftedWindows

        # sync all the agents
        tShift = max([ag.get('_t', 0.0*ns)['ns'] for ag in self.agents])
        tShift =  tShift*ns
        for ag in self.agents:
            ag['_t'] = tShift + tShift_env

        if correctXtalkZ:
            if config is None:
                qubits = self.agents
            else:
                # config is the same in the sample
                # for example, config = ['q1', 'q2', 'q3', 'r1']
                qubits = []
                for q_name in config:
                    ag = self.agents_dict[q_name]
                    if isinstance(ag, (Transmon, PhaseQubit)):
                        qubits.append(ag)
            eh.correctCrosstalkZ(qubits)

    def printGate(self, seperate=False):
        """
        print gates, if seperate, then the gates are printed for every agent
        else, the gates are printed for all agents
        """
        if seperate:
            for q in self.qubits:
                print("Gates for {}".format(q.__name__))
                for g in q['gates']:
                    print("\t {}".format(g))
            for r in self.resonators:
                print("Gates for {}".format(r.__name__))
                for g in r['gates']:
                    print("\t {}".format(g))
        else:
            for g in self.gates:
                print("{}".format(g))


def plotAlg(alg):
    import matplotlib.pyplot as plt
    assert isinstance(alg, Algorithm)
    if not alg.compiled:
        raise Exception("Algorithm must be compiled")
    N = len(alg.qubits)
    start = [q['xy'].start for q in alg.qubits]
    start += [q['z'].start for q in alg.qubits]
    start += [q['rr'].start for q in alg.qubits]
    end = [q['xy'].end for q in alg.qubits]
    end += [q['z'].end for q in alg.qubits]
    end += [q['rr'].end for q in alg.qubits]
    start = min([x for x in start if x is not None])
    end = max([x for x in end if x is not None])
    T = np.arange(start-5, end+5, 0.1)
    plt.figure(figsize=(12, 4*N))
    for i, q in enumerate(alg.qubits):
        plt.subplot(N*2, 1, 2*i+1)
        xy_val = q['xy'](T)
        z_val = q['z'](T)
        rr_val = q['rr'](T)
        plt.plot(T, xy_val.real, label='%s-X' % q.__name__)
        plt.plot(T, xy_val.imag, label='%s-Y' % q.__name__)
        plt.plot(T, z_val.real, label='%s-Z' % q.__name__)
        plt.legend(loc=0)
        plt.grid()
        plt.subplot(N*2, 1, 2*i+2)
        plt.plot(T, rr_val.real, label='%s-RRx' % q.__name__)
        plt.plot(T, rr_val.imag, label='%s-RRy' % q.__name__)
        plt.legend(loc=0)
        plt.grid()
    plt.tight_layout()

##################
#### Example #####
##################
"""
from pyle import gateCompiler as gc
from pyle import gates
from pyle.dataking.fpgaseq import runQubits

mesure = [0, 1]
sample, devs, qubits = gc.loadQubits(sample, measure)
alg = gc.Algorithm(devs)
qC = alg.q0
qT = alg.q1
alg[gates.PiPulse([qC])]
alg[gates.Sync([qC, qT])]
alg[gates.CZ([qC, qT])]
alg[gates.Sync([qC, qT])]
alg[gates.Measure([qC, qT])]
alg.compile(correctXtalkZ=True)
# runQubit(server, alg.agents, stats)

gc.plotAlg(alg)

t = np.linspace(-100, 400, 5001)
sC = qC['xy'](t)
sT = qT['xy'](t)
plt.figure()
plt.plot(t, sC.real)
plt.plot(t, sT.real)
plt.xlabel("Time [ns]")
plt.ylabel("Sequence")
plt.show()
"""
