#Notes
#
from labrad.units import Unit,Value
ns,us,MHz,GHz = (Unit(st) for st in ['ns','us','MHz','GHz'])

from pyle.registry import AttrDict
from pyle import envelopes as env
from pyle.dataking import envelopehelpers as eh

class Sequence(object):
    def __init__(self):
        pass
        
    

class Qubit(AttrDict):
    def __init__(self,d=None):
        super(Qubit,self).__init__(d=None)
        self._t = 0.0*ns
        self.piDelay = self['piLen']/2.0
        self['xy'] = env.NOTHING
        self['z'] = env.NOTHING
        
    def piPulse(self,state):
        self['xy'] += eh.boostState(self, (self._t+self.piDelay, state))
        self._t += self[piLen]
        
    def zPulse(self, len, amp):
        self['z'] += env.rect(self._t,len,amp)
        self._t += self['piLen']    
        
def t1(qubits, t):
    for q in qubits:
        q['readout']=True
        #Sequence
        q.initialize()
        q.piPulse(0)
        q.wait(t)
        q.measure()
