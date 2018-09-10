#import numpy as np

from labrad.units import Unit
mK, V, mV, us, ns, GHz, MHz, dBm, rad = [Unit(s) for s in ('mK', 'V', 'mV', 'us', 'ns', 'GHz', 'MHz', 'dBm', 'rad')]

import pyle.envelopes as env
from pyle.pipeline import returnValue, FutureList
from pyle.dataking import envelopehelpers as eh
#from pyle.dataking import squid
from pyle.dataking.fpgaseq_adc import runQubits
from pyle.util import sweeptools as st
from pyle.dataking import utilMultilevels as ml
from pyle.dataking import sweeps
from pyle.dataking import util
from pyle.plotting import dstools
from pyle.dataking import multiqubit_adc as mq
import labrad
    
def getMixTemp(cxn, sample, measure=0):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    lr = cxn.lakeshore_ruox
    lr.select_device('Vince GPIB Bus - GPIB0::12')
    temp = lr.temperatures()[0][0][mK]
    Q['temperature'] = temp
    return temp

def phaseCutoff(sample, resets=st.r[-2.5:2.5:5,V], measure=0, stats=150,
               save=True, name='Phase Arc Cutoff', collect=False, noisy=False, update=True, average=False):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    axes = [(resets, 'Reset Bias')]
    deps = [('S11 Phase', '', rad)]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)
    
    def func(server, reset):
        reqs = []
        for q in qubits:
            q['adc demod frequency'] = q['readout frequency']-q['readout fc']
            q.rr = eh.readoutPulse(q, 0) 
            q['biasReset'] = [reset]
            q['readout'] = True
        data = yield FutureList([runQubits(server, qubits, stats, raw=True)])
        for q in qubits:
            Is = data[0][0][0][q['adc channel']::11]
            Qs = data[0][0][1][q['adc channel']::11]
            if average:
                Is = [np.mean(Is)]
                Qs = [np.mean(Qs)]
        reqs.append([atan2(Qs[i], Is[i]) for i in range(len(Is))])    
        returnValue(np.vstack(reqs).T)
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=False)
    
    if update:
        resetSign = resets[-1]-resets[0]
        flux=data[:,0]
        times=data[:,1]
        meantime = np.mean(times)
        fluxvals=np.unique(flux)
        lowtimes=times[flux==fluxvals[0]]
        lm=np.mean(lowtimes)
        ls=np.std(lowtimes)
        hightimes=times[flux==fluxvals[1]]
        hm=np.mean(hightimes)
        hs=np.std(hightimes)
        if abs(ls-hs)<0.05: cutoff=meantime
        else:
            cutoffvals0=np.array([-1,1])
            cutoffvals=(-hs**2*lm+ls**2*hm+cutoffvals0*hs*ls*np.sqrt((lm-hm)**2+2*(ls**2-hs**2)*(np.log(ls)-np.log(hs))))/(ls**2-hs**2)
            diffs=cutoffvals-meantime
            cutoff=meantime+float(diffs[abs(diffs)==np.min(abs(diffs))])
        Q['critical phase']=cutoff
    if collect:
        return data


def measure2(sample, stats=3000L, measstate=2, zeroing=st.r[0:1:1], pi=st.r[0:1:1], measure=0, save=True, collect=True, noisy=True):
    """A single pi-pulse on one qubit, with other qubits also operated."""
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    ml.setMultiKeys(q,max([2,measstate]))
    
    axes = [(pi, '|1>-to-|2> Pi Pulse'), (zeroing, 'Swap to Resonator')]
    kw = {'stats': stats}
    name='Measure |1) with |'+str(measstate)+')'
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)
    
    def func(server, pi, zeroing):
        startPi = q['cZControlLen']+q['piFWHM']
        q.z = env.NOTHING
        if zeroing==0:
            startPi = q['piFWHM']
        elif zeroing==1:
            startPi = q['cZControlLen']+q['piFWHM']
            q.z +=  env.rect(0, q['cZControlLen'], q['cZControlAmp'])
        elif zeroing==2:
            startPi = q['cZControlLen']+q['cZTargetLen']+q['piFWHM']
            q.z +=  env.rect(0, q['cZControlLen'], q['cZControlAmp'])+env.rect(0, q['cZTargetLen'], q['cZTargetAmp'])
        if pi:
            q.xy = eh.mix(q, eh.piPulse(q, startPi, state=2), state=2)
        q.z += eh.measurePulse(q, startPi+2*q['piFWHM'], state=measstate)
        q['readout'] = True
        q['adc demod frequency'] = q['readout frequency']-q['readout fc']
        q.rr = eh.readoutPulse(q,0) 
        data = yield FutureList([runQubits(server, qubits, stats, raw=True)])
        returnValue([mq.tunneling(q,data)])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    
def plotSwap(cxn,datasetNumber,session=None,dir=None):
    dv=cxn.data_vault
    if session is None and dir is None:
        raise Exception('Dummy, you must specify either the session or the directory with the swap spectroscopy.')
    if session is not None and dir is not None:
        raise Exception('You can only specify one of session or dir.')
    if session is not None:
        dir = session._dir
    dstools.plotSwapSpectro(dstools.getSwapSpectroscopy(dv,datasetNumber,dir))