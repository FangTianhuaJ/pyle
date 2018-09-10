import numpy as np
from numpy import *
from scipy.optimize import leastsq
from scipy.special import erf, erfc
import pylab as plt
import pylab
import numpy
import labrad
from labrad.units import ns, GHz, MHz, mV, V, dBm, us,rad
from labrad.units import Unit,Value
import random

from math import atan2,asin
from scipy import optimize, interpolate


import pyle
from pyle.pipeline import returnValue, FutureList
from pyle.dataking import util
from pyle import envelopes as env
from pyle.dataking import envelopehelpers as eh
from pyle.dataking import measurement
from pyle.dataking import multiqubitPQ as mq
from pyle.dataking import sweeps
from pyle.dataking.DevJim.QND.fpgaseq_QND import runQubits
from pyle.util import sweeptools as st
from pyle.dataking import utilMultilevels as ml
from pyle.plotting import dstools
from pyle.fitting import fitting
from pyle.dataking.multiqubitPQ import freqtuner, find_mpa, rabihigh
import pdb
from fourierplot import fitswap
import time
from pyle import tomo
from pyle.dataking.multiqubitPQ import testdelay, pulseshape
from pyle.dataking import adjust
from pyle.analysis import FluxNoiseAnalysis as fna
from pyle.dataking.DevJim.QND import fpgaseq_QND
from pyle.fitting import fitting

from newarbitrarystate import sequence

fpgaseq_QND.PREPAD = 250

extraDelay = 0*ns


def resonatorSpectroscopy(sample, freqScan=None, swapTime=300*ns, stats=600L, measure=0,SZ=0.0,SZ0=0.0,paraName='0',amplitude=None,
       name='Resonator spectroscopy', save=True, collect=True, noisy=True,tBuf=20*ns):
    """A single pi-pulse on one qubit, with other qubits also operated.

    INPUT PARAMETERS
    sample: Object defining qubits to measure, loaded from registry.
    freqScan - iterable: Frequencies at which to drive resonator.
    swapTime - time: Time to swap between resonator and qubit.
    stats - scalar: Number of times a point will be measured.
    measure - scalar: Number of qubit to measure. Only one qubit allowed.
    SZ - float: Coupler zpa during resonator drive
    SZ0 - float: Coupler zpa during qubit-resonator swap.
    paraName - string: Name of swap registry keys (swapAmpName and swapTimeName).
    amplitude - float or None: Amplitude of resonator drive
    name - string: Name of dataset.
    save - bool: Whether to save data to the datavault.
    collect - bool: Whether or not to return data.
    noisy - bool: Whether to print out probabilities while the scan runs
    tBuf - time: Buffer time
    """
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    if freqScan is None:
        freqScan = st.r[q['readout frequency']['GHz']-0.002:q['readout frequency']['GHz']+0.002:0.00002,GHz]
    axes = [(freqScan, 'Resonator frequency')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, curr):
        sb_freq = curr-q['readout fc']
        #q['readout fc']=curr-sb_freq
        q['readout frequency']=curr
        if amplitude is not None:
            q['readout amplitude'] = amplitude
        q.rr = eh.ResSpectroscopyPulse(q, 0, sb_freq)
        q.cz =env.rect(0,q['readout length']+0.0*ns,SZ)
        #else:
        #    q.rr = eh.ResSpectroscopyPulse2(q, 0, sb_freq)
        q.z = env.rect(q['readout length']+tBuf, swapTime, q['swapAmp'+paraName])
        q.cz += env.rect(q['readout length']+tBuf, swapTime, SZ0)
        q.z +=eh.measurePulse(q, q['readout length']+tBuf+swapTime+tBuf)
        #else:
        #    q.z=  env.rect(q.piLen/2+20*ns, swapTime, q['swapAmp'+paraName])
        #    q.z=  env.rect(q.piLen/2+20*ns, swapTime, SZ0)
        #    q.z +=eh.measurePulse(q, q.piLen/2+30*ns+swapTime)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if collect:
        return data


def resSpectroscopySZ(sample, freqScan=st.r[6.52:6.586:0.0005,GHz], swapTime=300*ns, SZ=st.r[-0.9:0.9:0.0025],
                      stats=600L, measure=0, paraName='0', amplitude=0.04, tBuf=10*ns,
                      name='Resonator Spectroscopy SZ', save=True, collect=True, noisy=True):
    """2-D spectroscopy of resonator vs coupler SZ and drive frequency.

    INPUT PARAMETERS
    sample: Object defining qubits to measure, loaded from registry.
    freqScan - iterable: Frequencies at which to drive resonator.
    swapTime - time: Time to swap between resonator and qubit.
    SZ - iterable: Coupler zpa during resonator drive
    stats - scalar: Number of times a point will be measured.
    measure - scalar: Number of qubit to measure. Only one qubit allowed.
    paraName - string: Name of swap registry keys (swapAmpName and swapTimeName).
    amplitude - float or None: Amplitude of resonator drive
    tBuf - time: Buffer time
    name - string: Name of dataset.
    save - bool: Whether to save data to the datavault.
    collect - bool: Whether or not to return data.
    noisy - bool: Whether to print out probabilities while the scan runs
    """
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    if freqScan is None:
        freqScan = st.r[q['readout frequency']['GHz']-0.002:q['readout frequency']['GHz']+0.002:0.00002,GHz]
    axes = [(SZ,'squid z pulse amp'),(freqScan, 'Resonator frequency')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, currSZ,curr):
        sb_freq = curr-q['readout fc']
        q['readout amplitude'] = amplitude
        q['readout length'] = 2000*ns
        q.rr = eh.ResSpectroscopyPulse(q, 0, sb_freq)
        q.cz =env.rect(0,q['readout length'],currSZ)
        #r.z =env.rect(0,r.spectroscopyLen,RZ)
        q.z = env.rect(q['readout length']+tBuf, swapTime, q['swapAmp'+paraName])
        q.z += eh.measurePulse(q, q['readout length']+tBuf+swapTime+tBuf)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if collect:
        return data


def resSwapSpectroscopySZ(sample, zpaScan=None, SZ=st.r[-0.9:0.9:0.0025],
                      stats=600L, measure=0, paraName='0', tBuf=10*ns,
                      name='Resonator Swap Spectroscopy SZ', save=True, collect=True, noisy=True):
    """2-D spectroscopy of resonator vs coupler SZ and qubit zpa. Measures
    resonator frequency by exciting qubit and then swapping to resonator with
    varying qubit zpa (akin to Swap Spectroscopy).

    INPUT PARAMETERS
    sample: Object defining qubits to measure, loaded from registry.
    zpaScan - iterable: Qubit zpas over which to swap from qubit to resonator.
    SZ - iterable: Coupler zpa during resonator drive
    stats - scalar: Number of times a point will be measured.
    measure - scalar: Number of qubit to measure. Only one qubit allowed.
    paraName - string: Name of swap registry keys (swapAmpName and swapTimeName).
    tBuf - time: Buffer time
    name - string: Name of dataset.
    save - bool: Whether to save data to the datavault.
    collect - bool: Whether or not to return data.
    noisy - bool: Whether to print out probabilities while the scan runs"""
    Sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    if zpaScan is None:
        minZpa = mq.freq2zpa(sample,measure,q['readout frequency']-2*MHz)
        maxZpa = mq.freq2zpa(sample,measure,q['readout frequency']+2*MHz)
        zpaStep = mq.freq2zpa(sample,measure,q['readout frequency'])-mq.freq2zpa(sample,measure,q['readout frequency']+0.02*MHz)
        zpaScan = st.r[minZpa:maxZpa:zpaStep]
    axes = [(SZ,'squid z pulse amp'),(zpaScan, 'Qubit Swap ZPA')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(Sample, name, axes, measure=measure, kw=kw)
    piLen = q['piLen']
    swapTime = q['swapTime'+paraName]

    def func(server, currSZ, currZPA):
        t = 0
        q.xy = eh.mix(q, eh.piPulseHD(q, t+piLen/2))
        t += (piLen + tBuf)
        # Start cz pulse (located at end of pulse sequence)
        t += tBuf
        q.z = env.rect(t, swapTime, currZPA)
        t += swapTime + tBuf
        q.cz = env.rect(piLen+tBuf, t-(piLen+tBuf), currSZ)
        t+= tBuf
        q.z += eh.measurePulse(q, t)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if collect:
        return data


def fitResSpectroscopySZ(dir, datasetId, minBias=None, maxBias=None, minFreq=None, maxFreq=None,
       name='Resonator Freq Fit vs Dynamic Tuning', savePath=None, save=True, collect=True):
    """Fit resonator Freq with dynamic z pulse tuning on the adjustable coupler. Requires
    data from resSpectroscopySZ.

    INPUT PARAMETERS
    dir: Location of data from resT1DynamicCoupler (eg, s._dir)
    datasetId - int: Dataset number for data from resT1DynamicCoupler
    minBias - float: Minimum bias for which to fit T1
    maxBias - float: Maximum bias for which to fit T1
    minFreq - float: Minimum frequency to use in fitting T1
    maxFreq - float: Maximum frequency to use in fitting T1
    name - string: Name of dataset.
    savePath: Location in datavault to which to save data
    save - bool: Whether to save data to the datavault.
    collect - bool: Whether or not to return data.
    """
    with labrad.connect() as cxn:
        #Get the data from the data vault
        dv=cxn.data_vault
        dataset = dstools.getDeviceDataset(dv, datasetId=datasetId, session=dir)
    data = dataset.data
    freqUnit = Unit(dataset.variables[0][1][1])
    params = dataset.parameters

    #Determine the squid biases & separation between biases
    biases = np.unique(data[:,0])
    #Underestimate of separation between biases (in case multiple copies of same bias kept)
    dbias = np.abs(biases[-1]-biases[0])/len(biases)
    # If min,max not defined, select min and max over data.
    if minBias==None:
        minBias = np.min(biases)
    if maxBias==None:
        maxBias = np.max(biases)
    if minFreq==None:
        minFreq = np.min(data[:,1])
    if maxFreq==None:
        maxFreq = np.max(data[:,1])
    # Determine biases in [minBias,maxBias]
    biasesCut = [bias for bias in biases if (bias>=minBias and bias<=maxBias)]

    # For each bias in [minBias,maxBias], fit T1.
    freqs = []
    for bias in biasesCut:
        # Select only data with the desired bias.
        dataBias = data[np.abs(data[:,0]-bias)<dbias/2.0,1:]
        dataBiasCut = dataBias[np.logical_and(dataBias[:,0]>=minFreq, dataBias[:,0]<=maxFreq)]
        freqMaxProb = dataBiasCut[np.argmax(dataBiasCut[:,1]),0]
        vGuess = [0.5, freqMaxProb, 0.0001, 0.05]
        fits, cov, fitFunc = fitting.fitCurve('gaussian', dataBiasCut[:,0], dataBiasCut[:,1], vGuess)
        freqs.append(fits[1])
    newdata = np.array([biasesCut,freqs]).T

    # Now save T1s to the data vault
    if save:
        if savePath==None:
            savePath = dir
        with labrad.connect() as cxn:
            dv = cxn.data_vault
            dv.cd(savePath)
            dv.new(name,[dataset.variables[0][0]],[('Resonator Frequency','','GHz')])
            dv.add(newdata)
            for key in params.keys():
                if key in params.config:
                    for qubitkey in params[key].keys():
                        dv.add_parameter(key+'.'+qubitkey,params[key][qubitkey])
                else:
                    dv.add_parameter(key,params[key])
            dv.add_parameter('datasetId',datasetId)
            dv.add_parameter('dataPath',dir)
            dv.add_parameter('minBias',minBias)
            dv.add_parameter('maxBias',maxBias)
            dv.add_parameter('minFreq',minFreq)
            dv.add_parameter('maxFreq',maxFreq)
    if collect:
        return newdata


def resonatorT1(sample, delay=st.arangePQ(-0.01,1,0.01,'us')+st.arangePQ(1,7.5,0.05,'us'),paraName='0',stats=1200L,zpa=0.0,SZ=0.0,
                measure=0,name='resonator T1 MQ', excited = False, adi = False, save=True, collect=True, noisy=True):
    sample, qubits = util.loadQubits(sample)
    """Measures T1 of resonator using qubit to excite/readout.

    INPUT PARAMETERS
    sample: Object defining qubits to measure, loaded from registry.
    delay - iterable (us): Resonator delay time (between excite/readout)
    paraName - string: Name of swap registry keys (swapAmpName and swapTimeName).
    stats - scalar: Number of times a point will be measured.
    zpa - iterable or float: Qubit zpa during resonator delay time
    SZ - float: Coupler zpa during resonator delay time
    measure - scalar: Number of qubit to measure. Only one qubit allowed.
    name - string: Name of dataset.
    excited - bool: Whether to excite resonator into |2>
    adi - bool: Whether to use trapezoidal qubit z-pulses instead of rectangular
    save - bool: Whether to save data to the datavault.
    collect - bool: Whether or not to return data.
    noisy - bool: Whether to print out probabilities while the scan runs
    """
    q = qubits[measure]

    if excited: name = name + ' (with qubit excitation)'
    axes = [(zpa, 'z pulse amp'),(delay, 'Measure pulse delay')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, paraName+' '+name, axes, measure=measure, kw=kw)

    sl = q['swapTime'+paraName]
    sa = q['swapAmp'+paraName]

    def func(server, currZpa, delay):
        q['z'] = env.NOTHING
        q.xy = eh.mix(q, eh.piPulseHD(q, 0))
        q.z += env.rect(q.piLen/2, sl, sa)
        if excited:
            q.xy += eh.mix(q, eh.piPulseHD(q,1.0*q.piLen+sl))
            q.cz = env.rect(q.piLen*1.5+sl, delay, SZ)
            if adi:
                q.z += env.trapezoid(q.piLen*1.5+sl, q['adiRampLen'], delay, q['adiRampLen'], currZpa)
                q.z += env.rect(q.piLen*1.5+sl+delay+2*q['adiRampLen'], q['swapLen1'], q['swapAmp1'])
                q.z += env.rect(q.piLen*1.5+sl+delay+q['swapLen1']+2*q['adiRampLen'], sl, sa)
                q.z += eh.measurePulse(q, q.piLen*1.5+sl+delay+q['swapLen1']+sl+2*q['adiRampLen'])
            else:
                q.z += env.rect(q.piLen*1.5+sl, delay, currZpa)
                q.z += env.rect(q.piLen*1.5+sl+delay, q['swapLen1'], q['swapAmp1'])
                q.z += env.rect(q.piLen*1.5+sl+delay+q['swapLen1'], sl, sa)
                q.z += eh.measurePulse(q, q.piLen*1.5+sl+delay+q['swapLen1']+sl)
        else:
            q.cz = env.rect(q.piLen/2+sl, delay, SZ)
            if adi:
                q.z += env.trapezoid(q.piLen/2+sl, q['adiRampLen'], delay, q['adiRampLen'], currZpa)
                q.z += env.rect(q.piLen/2+sl+delay+2*q['adiRampLen'], sl, sa)
                q.z += eh.measurePulse(q, q.piLen/2+sl+delay+sl+2*q['adiRampLen'])
            else:
                q.z += env.rect(q.piLen/2+sl, delay, currZpa)
                q.z += env.rect(q.piLen/2+sl+delay, sl, sa)
                q.z += eh.measurePulse(q, q.piLen/2+sl+delay+sl)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if collect:
        return data


def resT1StaticCoupler(sample, delay=st.arangePQ(0,1,0.01,'us')+st.arangePQ(1,7.5,0.05,'us'),
                       couplings=st.r[0:0.2:0.02,mV],paraName='0',stats=600L, measure=0,
                       name='Resonator T1 Static Coupling', save=True, collect=True, noisy=True):
    """Measure resonator T1 with static z pulse tuning on the adjustable coupler

    INPUT PARAMETERS
    sample: Object defining qubits to measure, loaded from registry.
    delay - iterable (us): Resonator delay time (between excite/readout)
    couplings - iterable: Static coupler bias during resonator delay time
    paraName - string: Name of swap registry keys (swapAmpName and swapTimeName).
    stats - scalar: Number of times a point will be measured.
    measure - scalar: Number of qubit to measure. Only one qubit allowed.
    name - string: Name of dataset.
    save - bool: Whether to save data to the datavault.
    collect - bool: Whether or not to return data.
    noisy - bool: Whether to print out probabilities while the scan runs
    """
    sample, qubits, Qubits = util.loadQubits(sample,write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]
    #c= qubits[measureC]
    #r = qubits[measureR]

    axes = [(couplings,'squid Z bias'),(delay, 'Measure pulse delay')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    sl = q['swapTime'+paraName]
    sa = q['swapAmp'+paraName]

    def func(server, coupling, delay):
        fluxCouplerXtalk = q['calFluxCouplerStatic']
        dFlux = (np.polyval(fluxCouplerXtalk,coupling[mV]) - np.polyval(fluxCouplerXtalk,q['couplerfluxBias'][mV]))*mV
        q['couplerfluxBias'] = coupling
        q['biasOperate'] += dFlux
        q['biasReadout'] += dFlux
        #q['swapAmp'] = np.polyval(q['calSwapCouplerStatic'],coupling[V])
        q['swapAmp'] = getResSwapStatic(q, coupling)
        print 'Vals: ',coupling,q['biasOperate'],q['swapAmp']
        q.xy = eh.mix(q, eh.piPulseHD(q, 0))
        q.z = env.rect(q.piLen/2, sl, q['swapAmp'])
        q.z += env.rect(q.piLen/2+sl+delay, sl, q['swapAmp'])
        q.z += eh.measurePulse(q, q.piLen/2+sl+delay+sl)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if collect:
        return data


def resT1DynamicCoupler(sample, delay=st.arangePQ(0,1,0.01,'us')+st.arangePQ(1,7.5,0.05,'us'),SZ=st.r[0:0.2:0.02,None],SZ0=0.0,paraName='0',
       stats=600L, measure=0,name='resonator T1 with dynamic tuning', save=True, collect=True, noisy=True):
    """Measure resonator T1 with dynamic z pulse tuning on the adjustable coupler

    INPUT PARAMETERS
    sample: Object defining qubits to measure, loaded from registry.
    delay - iterable (us): Resonator delay time (between excite/readout)
    SZ - iterable: Dynamic (Fast) coupler zpa during resonator delay time
    SZ0 - float: Coupler zpa during swaps between qubit/resonator
    paraName - string: Name of swap registry keys (swapAmpName and swapTimeName).
    stats - scalar: Number of times a point will be measured.
    measure - scalar: Number of qubit to measure. Only one qubit allowed.
    name - string: Name of dataset.
    save - bool: Whether to save data to the datavault.
    collect - bool: Whether or not to return data.
    noisy - bool: Whether to print out probabilities while the scan runs
    """
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    #c= qubits[measureC]
    #r = qubits[measureR]

    axes = [(SZ,'squid Z bias'),(delay, 'Measure pulse delay')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    sl = q['swapTime'+paraName]
    sa = q['swapAmp'+paraName]

    def func(server, squidz,delay):
        q.xy = eh.mix(q, eh.piPulseHD(q, 0))
        q.z = env.rect(q.piLen/2, sl, sa)
        q.cz = env.rect(q.piLen/2, sl, SZ0)
        q.cz += env.rect(q.piLen/2+sl,delay,squidz)
        #r.z =env.rect(q.piLen/2+sl,delay,RZ)
        q.z += env.rect(q.piLen/2+sl+delay, sl, sa)
        q.cz += env.rect(q.piLen/2+sl+delay, sl, SZ0)
        q.z += eh.measurePulse(q, q.piLen/2+sl+delay+sl)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if collect:
        return data


def fitResT1DynamicCoupler(dir, datasetId, minBias=None, maxBias=None, minTime=None, maxTime=None,
       name='Resonator T1 Fit vs Dynamic Tuning', savePath=None, save=True, collect=True):
    """Fit resonator T1 with dynamic z pulse tuning on the adjustable coupler. Requires
    data from resT1DynamicCoupler.

    INPUT PARAMETERS
    dir: Location of data from resT1DynamicCoupler (eg, s._dir)
    datasetId - int: Dataset number for data from resT1DynamicCoupler
    minBias - float: Minimum bias for which to fit T1
    maxBias - float: Maximum bias for which to fit T1
    minTime - float: Minimum time to use in fitting T1
    maxTime - float: Maximum time to use in fitting T1
    name - string: Name of dataset.
    savePath: Location in datavault to which to save data
    save - bool: Whether to save data to the datavault.
    collect - bool: Whether or not to return data.
    """
    with labrad.connect() as cxn:
        #Get the data from the data vault
        dv=cxn.data_vault
        dataset = dstools.getDeviceDataset(dv, datasetId=datasetId, session=dir)
    data = dataset.data
    timeUnit = Unit(dataset.variables[0][1][1])

    #Determine the squid biases & separation between biases
    biases = np.unique(data[:,0])
    #Underestimate of separation between biases (in case multiple copies of same bias kept)
    dbias = np.abs(biases[-1]-biases[0])/len(biases)
    # If min,max not defined, select min and max over data.
    if minBias==None:
        minBias = np.min(biases)
    if maxBias==None:
        maxBias = np.max(biases)
    if minTime==None:
        minTime = np.min(data[:,1])
    if maxTime==None:
        maxTime = np.max(data[:,1])
    # Determine biases in [minBias,maxBias]
    biasesCut = [bias for bias in biases if (bias>=minBias and bias<=maxBias)]

    # For each bias in [minBias,maxBias], fit T1.
    t1s = []
    for bias in biasesCut:
        # Select only data with the desired bias.
        dataBias = data[np.abs(data[:,0]-bias)<dbias/2.0,1:]
        dataBiasCut = dataBias[np.logical_and(dataBias[:,0]>minTime, dataBias[:,0]<maxTime)]
        vGuess = [0.8, 1.0, 0.05]
        fits, cov, fitFunc = fitting.fitCurve('exponential', dataBiasCut[:,0], dataBiasCut[:,1], vGuess)
        t1s.append(fits[1])
    newdata = np.array([biasesCut,t1s]).T

    # Now save T1s to the data vault
    if save:
        if savePath==None:
            savePath = dir
        params = dataset.parameters
        with labrad.connect() as cxn:
            dv = cxn.data_vault
            dv.cd(savePath)
            dv.new(name,[dataset.variables[0][0]],[('Resonator T1','','us')])
            dv.add(newdata)
            for key in params.keys():
                if key in params.config:
                    for qubitkey in params[key].keys():
                        dv.add_parameter(key+'.'+qubitkey,params[key][qubitkey])
                else:
                    dv.add_parameter(key,params[key])
            dv.add_parameter('datasetId',datasetId)
            dv.add_parameter('dataPath',dir)
            dv.add_parameter('minBias',minBias)
            dv.add_parameter('maxBias',maxBias)
            dv.add_parameter('minTime',minTime)
            dv.add_parameter('maxTime',maxTime)
    if collect:
        return newdata


def testResDelay(sample, delay=st.r[-80:80:0.5,ns], pulseLength=8*ns, pulseSep=20*ns, amp=0.2,
                 stats=600L, measure=0, paraName='0', SZ=0.0, name='Resonator-qubit test delay',
                 tBuf=100*ns, save=True, collect=False, noisy=True, plot=True, update=True):
    """Test delay between resonator uwave drive and qubit z pulse line

    INPUT PARAMETERS
    sample: Object defining qubits to measure, loaded from registry.
    delay - iterable (us): Delay time from qubit z-pulse center to center between
        two resonator pulses
    pulseLength - time: FWHM of Gaussian pulses on resonator
    pulseSep - time: Time between resonator pulses
    amp - float: Amplitude of resonator pulses
    stats - scalar: Number of times a point will be measured.
    measure - scalar: Number of qubit to measure. Only one qubit allowed.
    paraName - string: Name of swap registry keys (swapAmpName and swapTimeName).
    SZ - iterable: Dynamic (Fast) coupler zpa during resonator pulses
    name - string: Name of dataset.
    tBuf - time: Time after qubit z-pulse to wait before measure pulse
    save - bool: Whether to save data to the datavault.
    collect - bool: Whether or not to return data.
    noisy - bool: Whether to print out probabilities while the scan runs
    plot - bool: Whether to plot data and fit curve
    update - bool: Whether to update timingLabRRUwave registry key
    """
    sample, qubits, Qubits = util.loadQubits(sample,write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]

    axes = [(delay, 'Delay of Res wrt Qubit')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, start):
        #q.rr = eh.mix(q, env.gaussian(100*ns+start-20, pulseLength, amp = amp), freq = 'readout frequency')+eh.mix(q, env.gaussian(100*ns+start+20, pulseLength, amp = -amp), freq = 'readout frequency')
        q['readout length'] = pulseLength
        q['readout amplitude'] = amp
        sb_freq = q['readout frequency'] - q['readout fc']
        checkTime = q['swapTime'+paraName]/2 + start - pulseSep - pulseLength
        if checkTime>0:
            t = q['swapTime'+paraName]/2 + start
        else:
            t = pulseSep + pulseLength
        # t is time in middle of two rr/cz pulses
        q.rr = env.mix(env.gaussian(t-pulseSep, pulseLength, amp=amp), sb_freq)
        q.cz = env.rect(t-pulseSep-pulseLength,2*pulseLength, SZ)
        q.rr += env.mix(env.gaussian(t+pulseSep, pulseLength, amp=-amp), sb_freq)
        q.cz += env.rect(t+pulseSep-pulseLength,2*pulseLength, SZ)
        q.z = env.rect(t-start-q['swapTime'+paraName]/2, q['swapTime'+paraName], q['swapAmp'+paraName])
        q.z += eh.measurePulse(q, t-start+q['swapTime'+paraName]/2+tBuf)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    topLen = q['swapTime'+paraName]['ns']*2*0.8
    translength = pulseLength[ns]
    def fitfunc(x, p):
        return (p[1] -
                p[2] * 0.5*erfc((x - (p[0] - topLen/2.0)) / translength) -
                p[3] * 0.5*erf((x - (p[0] + topLen/2.0)) / translength))
    x, y = result.T
    fit, _ok = leastsq(lambda p: fitfunc(x, p) - y, [0.0, 0.05, 0.85, 0.85])
    if plot:
        plt.figure()
        plt.plot(x, y, '.')
        plt.plot(x, fitfunc(x, fit))
    print 'uwave lag:', -fit[0]
    if update:
        print 'uwave lag corrected by %g ns' % fit[0]
        Q['timingLagRRUwave'] += fit[0]*ns
    if collect:
        return result


def testCouplerZDelayWrtQubit(sample, t0=st.r[-30:30:0.25,ns],measure=0,measureC=1, measureR=2,
                              tBuf=5*ns, paraName='0', SZ=st.r[0.8:-0.8:-0.05], stats=1200, update=True,
                              save=True, name='Coupler Z-Qubit Test Delay', plot=True, noisy=True, collect=False):
    """ This is to calibrate the test delay between qubit z pulse and adjustable
    coupler dynamic z pulse.

    Note: If SZ is not a float or int, will take 2D scan. Once scan is complete or
    cancelled, will prompt for which SZ to use to calculate testdelay time.

    INPUT PARAMETERS
    sample: Object defining qubits to measure, loaded from registry.
    t0 - iterable (us): Delay time between qubit and coupler z-pulses.
    measure - scalar: Number of qubit to measure. Only one qubit allowed.
    measureC - scalar: ?????
    measureR - scalar: ??????
    paraName - string: Name of swap registry keys (swapAmpName and swapTimeName).
    SZ - iterable: Amplitude of coupler z-pulse.
    stats - scalar: Number of times a point will be measured.
    update - bool: Whether to update timingLabRRUwave registry key
    save - bool: Whether to save data to the datavault.
    name - string: Name of dataset.
    plot - bool: Whether to plot data and fit curve
    noisy - bool: Whether to print out probabilities while the scan runs
    collect - bool: Whether or not to return data.
    """
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    #c = qubits[measureC]
    #C = Qubits[measureC]
    #r = qubits[measureR]

    axes = [(SZ,'Squid Z Bias'),(t0, 'Detuning pulse center')]
    kw = {'stats': stats,}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    swapTime = q['swapTime'+paraName]

    def func(server, currSZ,t0):
        if t0>0:
            t=0
        else:
            t = -t0
        q.xy = eh.mix(q, eh.piPulse(q, t-q['piLen']/2))
        q.cz = env.rect(t+t0,swapTime,currSZ)
        q.z = env.rect(t, swapTime, q['swapAmp'+paraName])
        measBuf = (t0>0)*t0 + tBuf
        q.z += eh.measurePulse(q, t+swapTime+measBuf)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    result = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

    if not (isinstance(SZ,int) or isinstance(SZ,float)):
        selectBias = float(raw_input('Select Squid Z Bias: '))
        biasSep = SZ[1]-SZ[0]
        result = result[(np.abs(result[:,0]-selectBias))<biasSep/2.0]
        result = result[:,1:]

    zpl = swapTime[ns]*0.8
    translength = 0.8*q['piFWHM'][ns]
    def fitfunc(x, p):
        return (p[1] -
                p[2] * 0.5*erfc((x - (p[0] - zpl/2.0)) / translength) -
                p[3] * 0.5*erf((x - (p[0] + zpl/2.0)) / translength))
    x, y = result.T
    fit, _ok = leastsq(lambda p: fitfunc(x, p) - y, [0.0, 0.05, 0.85, 0.85])
    if plot:
        plt.figure()
        plt.plot(x, y, '.')
        plt.plot(x, fitfunc(x, fit))
    print 'Coupler Dynamic Z lag:', fit[0]
    if update:
        print 'Coupler Dynamic Z delay corrected by %g ns' % fit[0]
        Q['timingLagCZ'] += fit[0]*ns
    if collect:
        return result


def fluxCouplerCrosstalk(sample, couplerScan=st.r[-500:500:10,mV], measure=0, spectroscopyAmp = None, startFluxShift = 0*mV,
                  fluxBelow=2*mV, fluxAbove=2*mV, fluxStep=0.1*mV, sb_freq=0*GHz, stats=300L, plot=True,
                  name='Flux - Static Coupler Crosstalk', save=True, collect=False, noisy=True, update=True):
    """Find the static coupler bias vs. flux bias voltage. A fit to the data is
    stored in the registry for the measured qubit as calFluxCouplerStatic.

    INPUT PARAMETERS
    sample: Object defining qubits to measure, loaded from registry.
    couplerScan - iterable (V): Voltage to apply to static coupler bias.
    measure - scalar: Number of qubit to measure. Only one qubit allowed.
    spectroscopyAmp - float: Spectroscopy pulse amplitude
    fluxBelow - float: How far below center flux to scan
    fluxAbove - float: How far above center flux to scan
    fluxStep - float: Step between scanned flux points
    sb_freq - float (Hz): Sideband frequency for spectroscopy pulse
    stats - scalar: Number of times a point will be measured.
    plot - bool: Whether to plot data and fit curve
    name - string: Name of dataset.
    save - bool: Whether to save data to the datavault.
    collect - bool: Whether or not to return data.
    noisy - bool: Whether to print out probabilities while the scan runs
    update - bool: Whether to update timingLabRRUwave registry key
    """
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]

    operateBias = q['biasOperate']
    readoutBias = q['biasReadout']
    measAmplFunc = mq.get_mpa_func(q)
    if spectroscopyAmp is not None:
        q['spectroscopyAmp'] = spectroscopyAmp
    couplerScan = np.array([bias[mV] for bias in couplerScan])
    couplerScan = couplerScan[np.argsort(np.abs(couplerScan))]
    fluxScan = np.arange(-fluxBelow[mV], fluxAbove[mV], fluxStep[mV])
    fluxScan = fluxScan[np.argsort(abs(fluxScan))]
    fluxPoints = len(fluxScan)
    fluxStep = fluxStep[mV]

    sweepData = {
        'fluxFunc': np.array([startFluxShift[mV]]),
        'fluxIndex': 0,
        'couplerIndex': 0,
        'flux': 0*fluxScan,
        'prob': 0*fluxScan,
        'maxima': 0*couplerScan,
    }

    axes = [('Flux Bias Shift', 'mV'), ('Coupler Bias', 'mV')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def sweep():
        for couplerBias in couplerScan:
            center = np.polyval(sweepData['fluxFunc'], couplerBias)
            center = st.nearest(center, fluxStep)
            for dFlux in center + fluxScan:
                yield dFlux*mV, couplerBias*mV

    def func(server, args):
        dFlux, couplerBias = args
        q['couplerfluxBias'] = couplerBias
        q['biasOperate'] = operateBias + dFlux
        q['biasReadout'] = readoutBias + dFlux
        q['measureAmp'] = measAmplFunc(operateBias + dFlux)
        q.xy = eh.spectroscopyPulse(q, 0, sb_freq)
        q.z = eh.measurePulse(q, q['spectroscopyLen'] + q['piLen'])
        q['readout'] = True
        prob = yield runQubits(server, qubits, stats, probs=[1])

        flux_idx = sweepData['fluxIndex']
        sweepData['flux'][flux_idx] = dFlux[mV]
        sweepData['prob'][flux_idx] = prob[0]
        if flux_idx + 1 == fluxPoints:
            # one row is done.  find the maximum and update the spectroscopy fit
            coupler_idx = sweepData['couplerIndex']
            paramsGuess = [0.5, sweepData['flux'][np.argmax(sweepData['prob'])], 0.15, 0.02]
            fits, cov, fitFunc = fitting.fitCurve('gaussian', sweepData['flux'], sweepData['prob'], paramsGuess)
            sweepData['maxima'][coupler_idx] = fits[1]
            #sweepData['maxima'][coupler_idx] = sweepData['flux'][np.argmax(sweepData['prob'])]
            sweepData['fluxFunc'] = np.polyfit(couplerScan[:coupler_idx+1],
                                               sweepData['maxima'][:coupler_idx+1],
                                               1)
            sweepData['fluxIndex'] = 0
            sweepData['couplerIndex'] += 1
        else:
            # just go to the next point
            sweepData['fluxIndex'] = flux_idx + 1
        returnValue([dFlux, couplerBias, prob[0]])
    sweeps.run(func, sweep(), dataset=save and dataset, collect=collect, noisy=noisy)

    # create a flux function and return it
    poly = sweepData['fluxFunc']
    coupler_idx = sweepData['couplerIndex']
    if plot:
        indices = np.argsort(sweepData['maxima'][:coupler_idx])
        data = np.vstack((sweepData['maxima'][:coupler_idx][indices],couplerScan[0:coupler_idx][indices])).T
        fig=dstools.plotDataset1D(data,[('Flux Bias Shift','mV')],[('Coupler Bias','','mV')],style='.',legendLocation=None,show=False,
                                  markersize=15,title='')
        fig.get_axes()[0].plot(np.polyval(poly,data[:,1]),data[:,1],'r',linewidth=3)
        fig.show()
    if update:
        Q['calFluxCouplerStatic'] = poly
    return poly


def operateQubitCoupler(qubit, couplerBias, paramName='Bus'):
    """Update qubit parameters to operate at the specified static coupler bias."""
    fluxCouplerXtalk = qubit['calFluxCouplerStatic']
    dFlux = (np.polyval(fluxCouplerXtalk,couplerBias[mV]) - np.polyval(fluxCouplerXtalk,qubit['couplerfluxBias'][mV]))*mV
    print 'couplerBias: %s, dFlux: %s' % (couplerBias, dFlux)
    qubit['couplerfluxBias'] = couplerBias
    qubit['biasOperate'] += dFlux
    qubit['biasReadout'] += dFlux
    qubit['biasStepEdge'] += dFlux
    qubit['swapAmp'+paramName] = getResSwapStatic(qubit, couplerBias)


def operateCoupler(sample, couplerBias, paramName='Bus', measure=0):
    """Update qubit parameters to operate at the specified static coupler bias."""
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    Q = Qubits[measure]
    operateQubitCoupler(Q,couplerBias,paramName)


def resSwapSpectroscopyStatic(sample, zpaScan=None, couplings=st.r[-0.9:0.9:0.0025],
                      stats=600L, measure=0, paraName='0', tBuf=10*ns,
                      name='Resonator Swap Spectroscopy Static', save=True, collect=True, noisy=True):
    """2-D spectroscopy of resonator vs static coupler bias and qubit zpa. Measures
    resonator frequency by exciting qubit and then swapping to resonator with
    varying qubit zpa (akin to Swap Spectroscopy).

    INPUT PARAMETERS
    sample: Object defining qubits to measure, loaded from registry.
    zpaScan - iterable: Qubit zpas over which to swap from qubit to resonator.
    couplings - iterable: Static coupler bias
    stats - scalar: Number of times a point will be measured.
    measure - scalar: Number of qubit to measure. Only one qubit allowed.
    paraName - string: Name of swap registry keys (swapAmpName and swapTimeName).
    tBuf - time: Buffer time
    name - string: Name of dataset.
    save - bool: Whether to save data to the datavault.
    collect - bool: Whether or not to return data.
    noisy - bool: Whether to print out probabilities while the scan runs"""
    Sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    couplerBiasOld = q['couplerfluxBias']

    if zpaScan is None:
        minZpa = mq.freq2zpa(sample,measure,q['readout frequency']-10*MHz)
        maxZpa = mq.freq2zpa(sample,measure,q['readout frequency']+10*MHz)
        zpaStep = mq.freq2zpa(sample,measure,q['readout frequency'])-mq.freq2zpa(sample,measure,q['readout frequency']+0.02*MHz)
        zpaScan = st.r[minZpa:maxZpa:zpaStep]
    axes = [(couplings,'Squid Static Coupling'),(zpaScan, 'Qubit Swap ZPA')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(Sample, name, axes, measure=measure, kw=kw)
    piLen = q['piLen']
    swapTime = q['swapTime'+paraName]

    def func(server, coupling, zpa):
        fluxCouplerXtalk = q['calFluxCouplerStatic']
        dFlux = (np.polyval(fluxCouplerXtalk,coupling[mV]) - np.polyval(fluxCouplerXtalk,q['couplerfluxBias'][mV]))*mV
        q['couplerfluxBias'] = coupling
        q['biasOperate'] += dFlux
        q['biasReadout'] += dFlux
        t = 0
        q.xy = eh.mix(q, eh.piPulseHD(q, t+piLen/2))
        t += (piLen + tBuf)
        q.z = env.rect(t, swapTime, zpa)
        t += swapTime + tBuf
        q.z += eh.measurePulse(q, t)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if collect:
        return data


def fitResSpectroscopyStatic(dir, datasetId, sample=None, minBias=None, maxBias=None, minZPA=None, maxZPA=None,
                             minFitBias=None, maxFitBias=None, name='Resonator Freq Fit vs Static Tuning',
                             savePath=None, save=True, collect=True, update=True, plot=True):
    """Fit resonator Freq with static z pulse tuning on the adjustable coupler. Requires
    data from resSwapSpectroscopyStatic. A fit to the data is stored in the registry for the
    measured qubit as calSwapCouplerStatic.

    INPUT PARAMETERS
    dir: Location of data from resT1DynamicCoupler (eg, s._dir)
    datasetId - int: Dataset number for data from resSwapSpectroscopyStatic
    sample: Object defining qubits to measure, loaded from registry. Only needed if update is True.
    minBias - float: Minimum bias for which to determine swap Amp
    maxBias - float: Maximum bias for which to determine swap Amp
    minZPA - float: Minimum frequency to use to determine swap Amp
    maxZPA - float: Maximum frequency to use to determine swap Amp
    minBias - float: Minimum bias to use in fitting
    maxBias - float: Maximum bias to use in fitting
    name - string: Name of dataset.
    savePath: Location in datavault to which to save data
    save - bool: Whether to save data to the datavault.
    collect - bool: Whether or not to return data.
    update - bool: Whether to update calSwapCouplerStatic registry key
    plot - bool: Whether to plot data corrected for FastBias resolution and fit.
    """
    if sample==None and update:
        raise Exception('Must give sample to update registry')
    with labrad.connect() as cxn:
        #Get the data from the data vault
        dv=cxn.data_vault
        dataset = dstools.getDeviceDataset(dv, datasetId=datasetId, session=dir)
    data = dataset.data
    couplingUnit = Unit(dataset.variables[0][0][1])
    params = dataset.parameters

    #Determine the squid biases & separation between biases
    biases = np.unique(data[:,0])
    #Underestimate of separation between biases (in case multiple copies of same bias kept)
    dbias = np.abs(biases[-1]-biases[0])/len(biases)
    # If min,max not defined, select min and max over data.
    if minBias==None:
        minBias = np.min(biases)
    if maxBias==None:
        maxBias = np.max(biases)
    if minZPA==None:
        minZPA = np.min(data[:,1])
    if maxZPA==None:
        maxZPA = np.max(data[:,1])
    # Determine biases in [minBias,maxBias]
    biasesCut = [bias for bias in biases if (bias>=minBias and bias<=maxBias)]

    # For each bias in [minBias,maxBias], fit T1.
    zpas = []
    for bias in biasesCut:
        # Select only data with the desired bias.
        dataBias = data[np.abs(data[:,0]-bias)<dbias/2.0,1:]
        dataBiasCut = dataBias[np.logical_and(dataBias[:,0]>=minZPA, dataBias[:,0]<=maxZPA)]
        zpaMinProb = dataBiasCut[np.argmin(dataBiasCut[:,1]),0]
        vGuess = [-0.5, zpaMinProb, 0.01, 0.5]
        fits, cov, fitFunc = fitting.fitCurve('gaussian', dataBiasCut[:,0], dataBiasCut[:,1], vGuess)
        zpas.append(fits[1])
    newdata = np.array([biasesCut,zpas]).T

    # Now save T1s to the data vault
    if save:
        if savePath==None:
            savePath = dir
        with labrad.connect() as cxn:
            dv = cxn.data_vault
            dv.cd(savePath)
            dv.new(name,[dataset.variables[0][0]],[('Qubit Swap ZPA','','')])
            dv.add(newdata)
            for key in sorted(params):
                if key in params.config:
                    for qubitkey in sorted(params[key]):
                        dv.add_parameter(key+'.'+qubitkey,params[key][qubitkey])
                else:
                    dv.add_parameter(key,params[key])
            dv.add_parameter('datasetId',datasetId)
            dv.add_parameter('dataPath',dir)
            dv.add_parameter('minBias',minBias)
            dv.add_parameter('maxBias',maxBias)
            dv.add_parameter('minZPA',minZPA)
            dv.add_parameter('maxZPA',maxZPA)
    if update:
        if minFitBias==None:
            minFitBias = minBias
        if maxFitBias==None:
            maxFitBias = maxBias

        # Correcting experimental data for resolution of FastBiases
        measure = params['measure'][0]
        q = params[params['config'][0]]
        fluxStep = 2.5*V/2**14 # FastBias resolution
        couplingFluxStep = (fluxStep/q['calFluxCouplerStatic'][0]) # Convert resolution into coupling units
        dataCut = newdata[np.logical_and(newdata[:,0]>minFitBias,newdata[:,0]<maxFitBias)]
        dFluxStep = dataCut[:,0]/couplingFluxStep[couplingUnit]-np.round(dataCut[:,0]/couplingFluxStep[couplingUnit]) #Determine fraction of FastBias step for given point
        dzpas = np.array([mq.freq2zpa(sample,0,mq.bias2freq(sample,0,q['biasOperate']-d*fluxStep)*GHz) for d in dFluxStep]) #Offset in swapAmps due to FastBias resolution
        couplings = dataCut[:,0]
        swapAmps = dataCut[:,1]+dzpas

        width = maxFitBias-minFitBias
        mid = (maxFitBias+minFitBias)/2.0
        minVal = swapAmps[np.argmin(np.abs(couplings-mid))]
        amp = (swapAmps[np.argmin(np.abs(couplings-(mid+maxFitBias)/2.0))]-minVal)/(1-np.sqrt(2))
        fits,cov,fitFunc = fitting.fitCurve('secant',couplings,swapAmps,[amp,width,mid,minVal])
        sample,qubits,Qubits = util.loadQubits(sample,write_access=True)
        Q = Qubits[measure]
        Q['calSwapCouplerStatic'] = fits

        if plot:
            fig = dstools.plotDataset1D(np.array([couplings,swapAmps]).T, [dataset.variables[0][0]],
                                        [(dataset.variables[0][1][0],'',dataset.variables[0][1][1])],
                                        marker='.', markersize=15)
            ax = fig.get_axes()[0]
            ax.plot(couplings,fitFunc(couplings,*fits),'k')
            ax.grid()
            ax.set_title(dataset['path'])
            fig.show()
    if collect:
        return newdata


def getResSwapStatic(qubit, couplerBias, correctFastBias=True):
    """Return resonator swap amplitude for a given static coupler bias"""
    amp,width,mid,minVal = qubit['calSwapCouplerStatic']
    print amp, width,mid,minVal
    trueSwap = fitting.secant(couplerBias,amp,width,mid,minVal) #Swap without correcting for FastBias resolution
    if not correctFastBias:
        return trueSwap

    fluxStep = 2.5*V/2**14 # FastBias resolution
    couplingFluxStep = (fluxStep/qubit['calFluxCouplerStatic'][0]) # Convert resolution into coupling units
    dFluxStep = couplerBias['V']/couplingFluxStep['V']-np.round(couplerBias['V']/couplingFluxStep['V']) #Determine fraction of FastBias step for given point

    #mq.bias2freq
    pF = qubit['calFluxFunc']
    bias = qubit['biasOperate']-dFluxStep*fluxStep
    freq = (((bias-qubit['biasStepEdge'])[V]-pF[1])/pF[0])**(0.25)*GHz

    #mq.freq2zpa
    pZ = qubit['calZpaFunc']
    dzpa = pZ[1] + pZ[0]*(freq['GHz']**4)
    # dzpa is offset in swapAmps due to FastBias resolution

    correctedSwap = trueSwap-dzpa
    return correctedSwap


def pulse_test(Sample, cxn, amp=0.05,freq = 6.56508*GHz,SZcapture=0.0,readLen=100*ns,SZread=0.0,SZoff=0.0,release=True,
               fullTime=7*us,delay = 200*ns,release2=False,delay2=0*ns,releaseLen2=5000*ns,
               tail=2000*ns,sb_freq = 50*MHz, save=False, stats=150, measure=0, releaseMode=0,releaseLen=2000*ns,
               session=['','Yi','QND','wfr20110330','r4r3']):
    """ Apply a pulse on resonator with coupler open, close coupler, then open up to
    release photon.

    INPUT PARAMETERS
    sample: Object defining qubits to measure, loaded from registry.
    cxn: cxn=labrad.connect()
    amp - float: Amplitude of drive pulse
    freq - frequency: Resonator drive frequency.
    SZcapture - float: Squid dynamic bias during capture
    readLen - time: Length of capture pulse
    SZread - float: Squid dynamic bias for reading out (releasing) photon
    SZoff - float: Squid dynamic bias for turning off coupling
    release - bool: Whether to release photon
    fullTime - time: Length of total sequence.
    delay - time: Time during which coupling turned off
    release2 - bool: ?????
    delay2 - time: ?????
    releaseLen2 - time: ?????
    tail - time: ?????
    sb_freq - frequency: Sideband frequency of resonator drive source
    save - bool: Whether to save data to the datavault.
    stats - scalar: Number of times a point will be measured. Total signal is sum
        of each signal (so average is trace/states).
    measure - scalar: Number of qubit to measure. Only one qubit allowed.
    releaseMode - int: Shape of readout pulses
    releaseLen - time: Duration of release pulse.
    session - dir: Directory to which to save data.
    """
    sample, qubits = util.loadQubits(Sample)
    q = qubits[measure]
    q['readout'] = True
    q['readoutType'] = 'resonator'
    q['biasReadout'] = q['biasOperate']
    q['adc_start_delay'] = 0*ns
    q['readout DAC start delay'] = 0*ns
    q['adc mode'] = 'average'
    q['readout amplitude']=amp
    q['readout frequency'] = freq
    q['adc demod frequency'] = sb_freq
    q['readout fc'] = q['readout frequency'] - sb_freq
    q['readout length'] = readLen
    q.rr = eh.ResSpectroscopyPulse(q, 0*ns, sb_freq)
    if release==True:
        if releaseMode==0:
            q.cz = env.rect(0,q['readout length'],SZcapture)
            q.cz += env.rect(q['readout length'],delay,SZoff)
            q.cz += env.rect(q['readout length']+delay,releaseLen,SZread)
            q.cz += env.rect(q['readout length']+delay+releaseLen,fullTime-(q['readout length']+delay+releaseLen),SZoff)
        elif releaseMode==1:
            q.cz = env.rect(0,q['readout length'],SZcapture)
            q.cz += env.gaussian(q['readout length']+delay, releaseLen/2.0, amp=SZread, phase=0.0, df=0.0)
            if release2:
                q.cz += env.rect(q['readout length']+delay+delay2,releaseLen2,SZread)
        elif releaseMode==2:
            q.cz = env.rect(0,q['readout length'],SZcapture)
            q.cz += env.gaussian(q['readout length']+delay, releaseLen/2.0, amp=SZread, phase=0.0, df=0.0)
            q.cz += env.gaussian(q['readout length']+delay+delay, releaseLen/2.0, amp=SZread, phase=0.0, df=0.0)
            q.cz += env.gaussian(q['readout length']+delay+delay*2, releaseLen/2.0, amp=SZread, phase=0.0, df=0.0)
    else:
        q.cz =env.rect(0,q['readout length']+tail,SZcapture)

    qseq = cxn.qubit_sequencer
    ans = runQubits(qseq, qubits, stats, raw=True).wait()
    Is = np.asarray(ans[0][0])
    Qs = np.asarray(ans[0][1])


    data_output = np.empty([len(Is),3])
    for i in range(len(Is)):
        data_output[i,0]=i*2
    data_output[:,1] = Is
    data_output[:,2] = Qs

    if save:
        dv = cxn.data_vault
        dv.cd(session)
        dv.new('pulse qstate', ['time [ns]'],['I', 'Q'])
        all = [[i*2,Is[i], Qs[i]] for i in range(len(Is))]
        dv.add(all)
        for elem in sorted(sample):
            if isinstance(sample[elem],dict):
                qubit = sorted(sample[elem])
                for key in qubit:
                    if key not in ['cz','rr']:
                        dv.add_parameter(elem+'.'+key, sample[elem][key])
            else:
                dv.add_parameter(elem,sample[elem])

    return data_output


def circlePlot(session,dataNum=1,calData=False,xGuess=0,yGuess=0,pGuess=[0.0,0.0],ampRescale=False,calUpdate=False):
    """ Plot circle for s_scanning. Apply calibration if desired.

    INPUT PARAMETERS
    session: Directory where data saved.
    dataNum - int: Number of dataset.
    calData - bool: Whether to apply calibration to data.
    xGuess - float: Guess for x-axis center of circle.
    yGuess - float: Guess for y-axis center of circle.
    pGuess - [float,float]: Guess for line describing phase of transmission
        line in format [intercept~0,slope~300]
    ampRescale - bool: Whether to rescale data such that fit circle radius is 1.
    calUpdate - bool: Whether to find fit parameters for circle.
    """
    with labrad.connect() as cxn:
        dv = cxn.data_vault
        dataset = dstools.getDeviceDataset(dv,datasetId=dataNum,session=session)
    data = dataset.data
    params = dataset.parameters
    q = params[params['config'][params['measure'][0]]]
    freqs = data[:,0]
    phases = data[:,1]
    amps = data[:,2]

    freqCutMin = float(raw_input('Resonance Peak Min (in GHz): '))
    freqCutMax = float(raw_input('Resonance Peak Max (in GHz): '))

    if calUpdate:
        params = circleCal(data,freqCutMin,freqCutMax,xGuess=xGuess,yGuess=yGuess,pGuess=pGuess)
    else:
        params = [q.get('calCircleX',xGuess), q.get('calCircleY',yGuess), q.get('calCircleR',np.mean(amps)),[q.get('calCircleP0',pGuess[0]),q.get('calCircleP1',pGuess[1])]]

    if calData:
        x = amps*np.cos(phases)
        y = amps*np.sin(phases)
        xCentered = x-params[0]
        yCentered = y-params[1]
        sCentered = xCentered + 1j * yCentered
        amps = np.abs(sCentered)
        phases = np.angle(sCentered)
    if ampRescale:
        amps /= params[2]

    x = amps*np.cos(phases)
    y = amps*np.sin(phases)
    plt.figure(2)
    plt.plot(x,y,'bo-')

    if calData:
        phases = np.mod(phases-freqs*params[3][1]-params[3][0],2*pi)
        x = amps*np.cos(phases)
        y = amps*np.sin(phases)
        plt.figure(2)
        plt.plot(x,y,'go-')
    return freqs,x,y,params


def circleCal(data,freqCutMin,freqCutMax,xGuess=0,yGuess=0,pGuess=[0.0,300.0],ampRescale=False):
    """ Determine calibration for s_scanning circle.

    INPUT PARAMETERS
    data: Circle data.
    xGuess - float: Guess for x-axis center of circle.
    yGuess - float: Guess for y-axis center of circle.
    pGuess - [float,float]: Guess for line describing phase of transmission
        line in format [intercept,slope]
    ampRescale - bool: Whether to rescale data such that fit circle radius is 1.
    """
    #freqs = data[:,0]
    if freqCutMax<freqCutMin:
        freqCutMin,freqCutMax = freqCutMax,FreqCutMin
    dFreq = data[1,0]-data[0,0]
    whereResonator = np.logical_or(data[:,0]<=freqCutMin,data[:,0]>=freqCutMax)
    dataCut = data[whereResonator,:]

    x=dataCut[:,2]*np.cos(dataCut[:,1])
    y=dataCut[:,2]*np.sin(dataCut[:,1])

    def calc_R(xc, yc):
        """ Calculate the distance of each 2D points from the center (xc, yc) """
        return sqrt((x-xc)**2 + (y-yc)**2)

    def f_2(c):
        """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_2, ier = leastsq(f_2, [xGuess, yGuess])

    xc_2, yc_2 = center_2
    Ri_2 = calc_R(xc_2, yc_2)
    R_2 = Ri_2.mean()

    x_new = x-xc_2
    y_new = y-yc_2
    phase_new = np.unwrap(np.angle(x_new+1j*y_new))
    plt.figure(3)
    plt.plot(dataCut[:,0],phase_new,'-')
    num2Pis = int(raw_input('Number of 2*pi to add for freq>resonator: '))
    plt.close(3)
    phaseCutMin = np.min(np.where(whereResonator==False)[0])
    addPhaseCut = num2Pis*2*np.pi*(np.arange(len(phase_new))>=phaseCutMin)
    phase_new = phase_new+addPhaseCut
    plt.figure(3)
    plt.plot(dataCut[:,0],phase_new,'-')

    def fitfunc(freq,p):
        return p[0]+p[1]*freq
    def errfunc(p):
        return phase_new-fitfunc(dataCut[:,0],p)
    p,ok = leastsq(errfunc, pGuess)
    print p
    plt.figure(3)
    plt.plot(dataCut[:,0],np.unwrap(p[0]+p[1]*dataCut[:,0])+addPhaseCut)

    return xc_2, yc_2, R_2, p



def s_scanning(Sample, freq=None, amplitude=.01, zpa=0.0, SZ=None, pulseLen=None,
               sb_freq = -50*MHz, measure=0, stats=150, tail=2800*ns, update = False,
               save=True, name='S parameter scanning', collect=True, noisy=True):
    """ This is the original s_scaning with sz"""
    sample, qubits, Qubits = util.loadQubits(Sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    if freq is None:
        f = st.nearest(q['readout frequency'][GHz], 1e-5)
        freq = st.r[f-0.0015:f+0.0015:1e-5, GHz]
    if pulseLen is None: pulseLen = q['readout length']
    #if amp is None: amp = q['readout amplitude']
    if SZ is None:
        SZ = 0.0
    if update:
        name = 'Cali ' + name
        if q.get('calCircleX',None)!=None:
            q.pop('calCircleX')
            q.pop('calCircleY')
            q.pop('calCircleP0')
            q.pop('calCircleP1')
            q.pop('calCircleR')

    axes = [(amplitude,'amplitude'), (freq, 'Frequency')]
    deps = [('Phase', 'S11 for %s'%q.__name__, rad) for q in qubits]+ [('Amplitude','S11 for %s'%q.__name__,'') for q in qubits]
    kw = {'stats': stats, 'zpa': zpa, 'SZ': SZ}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, ampCurr, freqCurr):
        amp =[]
        phase =[]
        q['biasReadout'] = q['biasOperate']
        q['readout amplitude'] = ampCurr
        q['adc mode'] = 'demodulate'
        q['readout frequency']=freqCurr
        q['readout length']=pulseLen
        q['readout fc'] = q['readout frequency'] - sb_freq
        q['adc demod frequency'] = q['readout frequency']-q['readout fc']
        q['readout'] = True
        q['readoutType'] = 'resonator'
        q.rr = eh.ResSpectroscopyPulse(q, 0*ns, sb_freq)

        q.z = env.rect(0, q['readout length']+tail, zpa)
        q.cz =env.rect(0,q['readout length']+tail,SZ)

        #q.xy = eh.mix(q, eh.piPulseHD(q, -q['piLen']/2))
        if noisy: print freqCurr, ampCurr

        data = yield FutureList([runQubits(server, qubits, stats, raw=True)])
        I_mean = np.mean(data[0][0][0])
        Q_mean = np.mean(data[0][0][1])
        ref = I_mean + 1j*Q_mean
        amp.append(abs(I_mean+1j*Q_mean))
        phase.append(atan2(Q_mean,I_mean))
        returnValue(phase+amp)

    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=False)

    if update:
        freq,x,y,params = circlePlot(dataNum=-1,session=sample._dir,calData=True,xGuess=0,yGuess=0,
                                     pGuess=[0.0,350.0],ampRescale=True,calUpdate=True)
        x0,y0,r0,p = params
        Q['calCircleX']=x0
        Q['calCircleY']=y0
        Q['calCircleR']=r0
        Q['calCircleP0']=p[0]
        Q['calCircleP1']=p[1]
    if collect:
        return data


def sScanningDynamicCoupler(sample, SZs, dirT1, numT1, dirF, numF):
    with labrad.connect() as cxn:
        dv=cxn.data_vault
        datasetT1 = dstools.getDeviceDataset(dv,numT1,dirT1)
        datasetF = dstools.getDeviceDataset(dv,numF,dirF)
    freqBiases = datasetF.data[:,0]
    freqs = datasetF.data[:,1]
    t1Biases = datasetT1.data[:,0]
    t1s = datasetT1.data[:,1]
    if not all(freqBiases==t1Biases):
        raise Exception('Biases must be equal in freq, T1')
    dfreqs = 1e-3/(2*np.pi*t1s)
    biasSelects = [np.where(np.abs(t1Biases-SZ)<1e-8)[0][0] for SZ in SZs]
    szSelects = t1Biases[biasSelects]
    freqSelects = freqs[biasSelects]
    dfreqSelects = dfreqs[biasSelects]
    for freq,dfreq,SZ in zip(freqSelects,dfreqSelects,szSelects):
        s_scanning(sample, SZ=SZ, name='S Scanning '+str(SZ), freq=np.linspace((freq-20*dfreq),(freq+20*dfreq),4000)*GHz)
