import numpy as np

from labrad.units import Unit,Value
V, mV, us, ns, GHz, MHz, dBm, rad = [Unit(s) for s in ('V', 'mV', 'us', 'ns', 'GHz', 'MHz', 'dBm', 'rad')]

import pyle.envelopes as env
from pyle.pipeline import returnValue, FutureList
from pyle.dataking import envelopehelpers as eh
from pyle.dataking.fpgaseq import runQubits as runQubits
from pyle.util import sweeptools as st

import sweeps
import util
import labrad

def swap11Spectroscopy(sample,
                     swapLen=st.r[0:500:5,ns], swapAmp=np.arange(-0.05,0.05,0.002),
                     exciteSwapName='', excite=0, measure=0, stats=600L,
                     name='Swap Spectroscopy',
                     save=True, collect=False, noisy=True, state=1, piPulse=True,
                     username=None):
    """Measures T1 vs z-pulse amplitude"""
    sample, qubits = util.loadDeviceType(sample,'phaseQubit')
    qe = qubits[excite]
    qm = qubits[measure]

    if swapAmp is None:
        swapAmp = qm.swapAmp
    elif np.size(swapAmp) is 1:
        swapAmp = float(swapAmp)
    else:
        swapAmp = swapAmp[np.argsort(np.abs(swapAmp))]
    
    axes = [(swapAmp, 'swap pulse amplitude'), (swapLen, 'swap pulse length')]
    kw = {'stats': stats, 'state': state}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)
    
    def func(server, currAmp, currLen):
        t=0.0*ns
        # Excite excite into |1> and swap into resonator
        qe.xy = eh.boostState(qe, qe['piLen']/2, state=1)
        t+=qe['piLen']
        exciteSwapTime = qe[exciteSwapName+'SwapTime']
        qe.z = env.rect(t,qe[exciteSwapName+'SwapAmp'],exciteSwapTime)
        t+=exciteSwapTime
        #It can be useful to do swap spectroscopy without exciting the qubit to see
        #if the microwave carrier is leaking into your system.
        if piPulse:
            qm.xy = eh.boostState(qm, qm['piLen']/2, state=state)
            t+=qm['piLen']*state
        qm.z= env.rect(t,currLen,currAmp)
        t+=currLen#+10*ns
        qm.z+=eh.measurePulse(qm, t, state=state)
        qm['readout'] = True
        return runQubits(server, qubits, stats=stats, probs=[1])
    
    data = sweeps.grid(func, axes, save=save, dataset=dataset, collect=collect, noisy=noisy)
    if username is not None:
        with labrad.connect() as cxn:
            try:
                cxn.telecomm_server.send_sms('Scan complete','Swap spectroscopy is complete',username)
            except:
                print 'Failed to send text message'
    if collect:
        return data