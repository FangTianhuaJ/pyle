#TODO
#1. Write function to fit high power spectroscopy data to extracted anharmonicity vs freq. etc. DTS
#2. Update GRAPE code to take all qubit variables, eg. relevant parameters in the registry.
#3. Write a LabRAD server that can invoke GRAPE.
#   i. Set up python and pylabrad on DE's computer.
#  ii. Get DE's computer access to the users share
# iii. Write the server.
#4. How should experiment code handle the GRAPE pulse?
#   i. GRAPE spits out DAC samples. GRAPE has complete description of electronics/qubit response
#  ii. GRAPE spits out an attainable time series, and lab code decides what the DAC samples are (ie. does deconvolution)
#5. Do gate with control in either |0> or |1> and target in |0>+|1>. Check for pi phase shift.
#6. Make Bell states and do state tomography.
#7. Process tomography.
#8. Compare with Strauch.
#9. Beer!
#...
#10. Graduate.

from pyle.dataking import util, sweeps
from pyle.dataking.fpgaseq import runQubits
from pyle.dataking import envelopehelpers as eh

from pyle import envelopes as env
from pyle.envelopes import Envelope
from pyle.envelopes import convertUnits
from pyle.dataking import measurement

from pyle.pipeline import returnValue, FutureList

from scipy import interpolate
from pylab import find
import numpy as np

import matplotlib.pyplot as plt
from pyle.fitting import fitting

from labrad.units import Unit,Value

ns,GHz = (Unit(unStr) for unStr in ['ns','GHz'])

# Convert GRAPE values to z-pulse amplitude
# Add to GRAPE's "qubit-bus detunning" the bus frequency to get the qubit's frequency
# convert this to a Z-pulse amplitude
def calibrateGrape(busfreq, samples, p):
    samples = samples + busfreq['GHz']
    ZPAsamples = (p[0].value)*samples**4 + (p[1].value)
    # Set first and last to zero to avoid fridge heating, compensate for ZPA calibration errors
    ZPAsamples[-20:-1] = 0.0
    ZPAsamples[-1] = 0.0
    ZPAsamples[0:20] = 0.0
    return ZPAsamples

@convertUnits(t0='ns', w='ns', amp=None, phase=None, df='GHz')
def offsetgauss(t0, w, amp=1.0, phase=0.0, df=0.0,offset=0.1):
    """A gaussian pulse with specified center and full-width at half max."""
    sigma = w / np.sqrt(8*np.log(2)) # convert fwhm to std. deviation
    def timeFunc(t):
        return amp * np.exp(-(t-t0)**2/(2*sigma**2) - 2j*np.pi*df*(t-t0) + 1j*phase) + offset
    
    sigmaf = 1 / (2*np.pi*sigma) # width in frequency space
    ampf = amp * np.sqrt(2*np.pi*sigma**2) # amp in frequency space
    def freqFunc(f):
        N = len(f)
        z = np.zeros(N)
        o = np.ones(N)*ampf*1000
        return (ampf * np.exp(-(f+df)**2/(2*sigmaf**2) - 2j*np.pi*f*t0 + 1j*phase)) + \
               np.where(f!=0, z, o)
    
    return Envelope(timeFunc, freqFunc, start=t0-w, end=t0+w)


def grapePulse(t0, times, samples, padTime):
    """
    A numerically defined GRAPE sequence
    
    INPUTS
    t0: desired start time of the numerical sequence
    times: array of time samples. This is assumed to begin with 0,
    which corresponds to the 
    """    
    pulseLength = times[-1] - times[0]
    times = times - times[0] + t0
    ### TIME DOMAIN INTERPOLATING FUNCTION ###
    interpFunc = interpolate.interp1d(times, samples, kind='cubic')
    ### FREQUENCY DOMAIN INTERPOLATING FUNCTION ###
    # Python DFT convention:
    #       N - 1
    #       ----
    #       \            -2*pi*i*n*k/N
    # a_k =  >    a_n * e
    #       /
    #       ----
    #      n = 0
    # But the actual FT relates to the DFT through a(v_k) = DFT(k)*T/N
    # Need to have the Fourier transform of [times,samples]
    # Best way to do this would probably be to pad the GRAPE pulse with starting values
    # (maybe remove the DC componant) and take the DFT
    dt = times[1]-times[0]
    numPadPoints = int(float(padTime)/float(dt))
    padding = samples[0]*np.array(np.ones(numPadPoints))
    paddedSamples = np.hstack((padding, samples, padding))
    sampleSpectrum = np.fft.fft(paddedSamples)
    # Give the time vector the appropriate length
    paddedTime = dt*len(paddedSamples)
    #Compute frequency axis
    freqs = np.fft.fftfreq(len(paddedSamples),dt)
    #Shift DFT values and frequency axis
    freqs = np.fft.fftshift(freqs)
    sampleSpectrum = paddedTime*np.fft.fftshift(sampleSpectrum)/len(paddedSamples)
    #Correct for time offset
    sampleSpectrum = sampleSpectrum * np.exp(-1.0j * 2.0 * np.pi * t0 * freqs) * np.exp(1.0j * 2 * np.pi * padTime * freqs)
    # Construct interpolating function in frequency space
    # Apparently does not like to do 'cubic', memory fails
    interpSpecFuncR = interpolate.interp1d(freqs, np.real(sampleSpectrum), kind='cubic')
    interpSpecFuncI = interpolate.interp1d(freqs, np.imag(sampleSpectrum), kind='cubic')
    
    #Define callable time and frequency functions
    #Time domain
    def funcT(t):
        if t < times[0]:
            return samples[0]
        elif t > times[-1]:
            return samples[-1]
        else:
            result = interpFunc(t)
            result.shape = (1,)
            return result[0]
    
    def timeFunc(t):
        return np.array(map(funcT, t))
    #Frequency domain
    def funcF(f):
        if abs(f) > freqs[np.argmax(freqs)]:   # DFT should be symmetric
            return 0
        else:
            return (interpSpecFuncR(f) + 1.0j*interpSpecFuncI(f))
    
    def freqFunc(f):
        return np.array(map(funcF, f))
    
    return env.Envelope(timeFunc, freqFunc, start=t0, end=t0+pulseLength)
    
    
#Measure anharmonicity and frequency vs bias.
#Extract anharmonicity and frequency from the data - make a table.
def getAnharmonicityData(s, qubit0, qubit1):
    dataset0 = ds.getOneDeviceDataset(dv, datasetNumber, path)
    

def pulseTest(sample, measure, N, grapeData, busFreq,
              stats=300, name='grape CZ', noisy=True):
    """Test pulse sequences"""

    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    q['readout'] = True
    iterations = np.linspace(0,N,N-1)
    plt.figure()
    
    controlSequence = calibrateGrape(busFreq, grapeData[:,1], q['calZpaFunc'])
    pulseData = grapePulse(0, grapeData[:,0], controlSequence, 100)
    plt.plot(grapeData[:,0], controlSequence)
    
    #Set up axes for scanning over a parameter. There can be any number of axes.
    axes = [(iterations, 'Iteration')]
    
    def func(server, iteration):
        q['xy'] = env.NOTHING
        q['z'] = env.NOTHING
        q['z'] += pulseData
        return runQubits(server, qubits, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=False, noisy=noisy)

    
def findCrossing(sample, measure, freqScan, zpaScan, spectroscopyAmplitude = None,
                 sb_freq=0*GHz, stats=300L,
                 name='Resonator Crossing', save=True, collect=False, noisy=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    qubit = qubits[measure]
    qubit['readout'] = True
    if spectroscopyAmplitude is not None:
        qubit['spectroscopyAmp'] = spectroscopyAmplitude
    #Set up flux scan
    zpaScan = zpaScan[np.argsort(abs(zpaScan))]
    
    axes = [(zpaScan, 'Z Pulse Amplitude'), (freqScan, 'Frequency')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)
    
    def func(server, zpa, freq):
        for q in qubits:
            q['fc'] = freq - sb_freq # set all frequencies since they share a common microwave source
        dt = qubit['spectroscopyLen']
        qubit.xy = eh.spectroscopyPulse(qubit, 0, sb_freq)
        qubit.z = env.rect(0, dt, zpa) + eh.measurePulse(qubit, dt)
        return runQubits(server, qubits, stats=stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)
    if collect:
        return data
    
    
def grapeCZ(sample, control, target, busFreqControl, busFreqTarget, grapeData, phases = np.linspace(-0.5,0.5,50),
            dBusFreqControl = 0.0*GHz, dBusFreqTarget = 0.0*GHz, tBuf = 5.0*ns,
            stats=300, name='grape CZ', save=True, noisy=True, collect=False):
    """A single pi-pulse on one qubit, with other qubits also operated."""

    sample, qubits = util.loadQubits(sample)
    qC = qubits[control]
    qT = qubits[target]
    qT['readout'] = True

    controlSequence = calibrateGrape(busFreqControl + dBusFreqControl, grapeData[:,2], qC['calZpaFunc'])
    targetSequence  = calibrateGrape(busFreqTarget + dBusFreqTarget, grapeData[:,1], qT['calZpaFunc'])
    
    envStartTime = max(qC['piLen']/2, qT['piLen']/2) + tBuf
    controlEnvelope = grapePulse(envStartTime['ns'], grapeData[:,0], controlSequence,100)
    targetEnvelope =  grapePulse(envStartTime['ns'], grapeData[:,0], targetSequence,100)
    
    #Set up axes for scanning over a parameter. There can be any number of axes.
    axes = [(phases, 'Target phase')]
    deps = [('Probability', 'Control = |%d>'%state, '') for state in [0,1]]
    kw = {'stats': stats, 'dBusFreqControl': dBusFreqControl*GHz, 'dBusFreqTarget': dBusFreqTarget*GHz,
          'busFreqControl': busFreqControl, 'busFreqTarget':busFreqTarget, 'piPulse':  control}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=target, kw=kw)
    
    def func(server, currentPhase):
        reqs = []
        for piPulse in [False, True]:
            for q in qubits:
                q['xy'] = env.NOTHING
                q['z']  = env.NOTHING
            t = 0.0*ns
            #Pi pulse or not on 'control' qubit
            if piPulse:
                qC['xy'] += eh.piPulse(qC, 0)
            qT['xy'] += eh.piHalfPulse(qT, t, phase = 0)
            t += max(qC['piLen']/2, qT['piLen']/2) + tBuf
            #Do CZ gate - This will probably execute quickly, but fpgaseq.py may take a long time to actually evaluate the interpolating functions
            qC['z'] += controlEnvelope
            qT['z'] += targetEnvelope
            t += grapeData[-1,0]*ns + tBuf
            #Pi/2 on target
            t += qT['piLen']/2
            qT['xy'] += eh.piHalfPulse(qT, t, phase = currentPhase * 2*np.pi)
            t += qT['piLen']/2
            #Measure the target qubit
            qT['z'] += eh.measurePulse(qT, t)
            #Apply sideband mixing
            qC['xy'] = eh.mix(qC, qC['xy'])
            qT['xy'] = eh.mix(qT, qT['xy'])
            eh.correctCrosstalkZ(qubits)
            reqs.append(runQubits(server, qubits, stats, probs=[1]))
        data = yield FutureList(reqs)
        probs = [p[0] for p in data]
        returnValue(probs)
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if collect:
        return data

        
def scanDBusFreq(sample, control, target, dBusFreqControl, dBusFreqTarget,
                 grapeData, busFreq, phases = np.linspace(-0.5,0.5,50),
                 tBuf = 5.0*ns, saveAllData=False,
                 stats=300, name='grape CZ', save=True, noisy=True):
    """"""

    sample, qubits = util.loadQubits(sample)
    qC = qubits[control]
    qT = qubits[target]
    qT['readout'] = True
    
    axes = [(dBusFreqControl, 'd Bus Frequency Control'), (dBusFreqTarget, 'd Bus Frequency Target')]
    deps = [('Phase Shift', '', 'Cycles')]
    kw = {'stats': stats, 'dBusFreqControl': dBusFreqControl*GHz, 'dBusFreqTarget': dBusFreqTarget*GHz, 'busFreq': busFreq, 'piPulse':  control}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=target, kw=kw)
    def func(server, dBusFreqControl, dBusFreqTarget):
        data = grapeCZ(sample, control, target, grapeData, busFreq, phases = np.linspace(-0.5,0.5,50),
                       dBusFreqControl = dBusFreqControl, dBusFreqTarget = dBusFreqTarget, tBuf = 5.0*ns,
                       stats=300, name='grape CZ', save=saveAllData, noisy=True, collect=True)
        fitsNoPi, _, fitFuncNoPi  = fitting.fitCurve('sine', data[:,0], data[:,1], [0.2, 1, 0, 0.5])
        fitsPi,   _, fitFuncPi    = fitting.fitCurve('sine', data[:,0], data[:,2], [0.2, 1, 0, 0.5])
        phaseNoPi = fitsNoPi[2]
        phasePi = fitsPi[2]
        phaseShift = phaseNoPi - phasePi
        print 'Found phase difference %f' %phaseShift
        returnValue([dBusFreqTarget, phaseShift])
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=False)     
        
        
def czQPT(s, control, target, busFreqControl, busFreqTarget, grapeData, dBusFreqControl = 0.0*GHz, dBusFreqTarget = 0.0*GHz,
          tBuf=5.0*ns, tBufMeasure = 5.0*ns, tomoType = 'octomo',
          name = 'czQPT', stats=600,
          collect=False, save=True, noisy=True):
    
    sample, qubits = util.loadQubits(s)
    qC = qubits[control]
    qT = qubits[target]
    measure = [control, target]
    qT['readout'] = True
    qC['readout'] = True
    N = len(s['config'])
    #Convert from bus detuning to zpa
    controlSequence = calibrateGrape(busFreqControl + dBusFreqControl, grapeData[:,2], qC['calZpaFunc'])
    targetSequence  = calibrateGrape(busFreqTarget + dBusFreqTarget, grapeData[:,1], qT['calZpaFunc'])
    #Create GRAPE envelopes from numerical data
    envStartTime = 0.5*max(qC['piLen'],qT['piLen']) + tBuf
    controlEnvelope = grapePulse(envStartTime['ns'], grapeData[:,0], controlSequence, 100)
    targetEnvelope =  grapePulse(envStartTime['ns'], grapeData[:,0], targetSequence, 100)

    if tomoType == 'tomo':
        measureFunc = measurement.Tomo(N, measure, tBuf=tBufMeasure)
    elif tomoType == 'octomo':
        measureFunc = measurement.Octomo(N, measure, tBuf=tBufMeasure)
    
    prepOps = [
        ('I', 0, 0),
        ('Xpi', 1, 0),
        ('Ypi/2', 0.5, 0.5),
        ('Xpi/2', 0.5, 1.0)]

    opNumbers = range(len(prepOps)**2)
    opNames = []
    opAmps = []
    opPhases = []
    for opC in prepOps:
        for opT in prepOps:
            opNames.append(opC[0]+opT[0])       #name
            opAmps.append([opC[1],opT[1]])      #amplitude
            opPhases.append([opC[2],opT[2]])    #phase
    
    axes = [(opNumbers,'Prep Op Number')]
    kw = {'stats': stats, 'dBusFreq1': dBusFreqControl, 'dBusFreqTarget': dBusFreqTarget, 'target': target, 'control': control,
          'prepOps': prepOps, 'prepNames': opNames, 'busFreqControl': busFreqControl, 'busFreTarget': busFreqTarget}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measureFunc, kw=kw)
    
    def func(server, opNumber):
        reqs=[]
        #Initialize everything
        t = 0.0*ns
        for q in qubits:
            q['z'] = env.NOTHING
            q['xy'] = env.NOTHING
        #Single qubit preparations
        qC['xy'] += eh.rotPulse(qC, t, angle = opAmps[opNumber][0]*np.pi, phase = opPhases[opNumber][0]*np.pi)
        qT['xy'] += eh.rotPulse(qT, t, angle = opAmps[opNumber][1]*np.pi, phase = opPhases[opNumber][1]*np.pi)
        t += 0.5*max(qC['piLen'],qT['piLen']) + tBuf
        ### BEGIN CONTROLLED Z GATE ###
        qC['z'] += controlEnvelope
        qT['z'] += targetEnvelope
        t += grapeData[-1,0]*ns + tBuf
        ### END CONTROLLED Z GATE ###
        #Apply sideband mixing
        qC['xy'] = eh.mix(qC, qC['xy'])
        qT['xy'] = eh.mix(qT, qT['xy'])
        eh.correctCrosstalkZ(qubits)
        return measureFunc(server, qubits, t, stats=stats)
    data = sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy, pipesize=1)
    if collect:
        return data
        
# Daniel start working on the Bell states        
def grapeCZBellStateTomoRect(s, control, target, busFreqControl, busFreqTarget, grapeData,
                             dBusFreqControl = 0.0*GHz, dBusFreqTarget = 0.0*GHz,
                        tBuf=5.0*ns, tBufMeasure = 5.0*ns, tomoType = 'tomo',                        
                        name = 'czBellTomo', stats=600,
                        collect=False, save=True, noisy=True):
                   
    sample, qubits = util.loadQubits(s)
    qC = qubits[control]
    qT = qubits[target]
    measure = [control, target]
    qT['readout'] = True
    qC['readout'] = True
    N = len(s['config'])

    #Convert from bus detuning to zpa
    controlSequence = calibrateGrape(busFreqControl + dBusFreqControl, grapeData[:,2], qC['calZpaFunc'])
    targetSequence  = calibrateGrape(busFreqTarget + dBusFreqTarget, grapeData[:,1], qT['calZpaFunc'])
    #Create GRAPE envelopes from numerical data
    envStartTime = 0.5*max(qC['piLen'],qT['piLen']) + tBuf
    controlEnvelope = grapePulse(envStartTime['ns'], grapeData[:,0], controlSequence, 100)
    targetEnvelope =  grapePulse(envStartTime['ns'], grapeData[:,0], targetSequence, 100)
    
    if tomoType == 'tomo':
        measureFunc = measurement.Tomo(N, measure, tBuf=tBufMeasure)
    elif tomoType == 'octomo':
        measureFunc = measurement.Octomo(N, measure, tBuf=tBufMeasure)
    
    axes = [([0], 'crap')]
    kw = {'stats': stats, 'measureType': tomoType, 'dBusFreqControl': dBusFreqControl, 'dBusFreq2': dBusFreqTarget, 'busFreqControl': busFreqControl, 'busFreqTarget': busFreqTarget}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measureFunc, kw=kw)
    
    def func(server, _):
        reqs=[]
        #Initialize everything
        t = 0.0*ns
        for q in qubits:
            q['z'] = env.NOTHING
            q['xy'] = env.NOTHING
        #Single qubit preparations
        qC['xy'] += eh.piHalfPulse(qC, t, phase = np.pi/2)
        qT['xy'] += eh.piHalfPulse(qT, t, phase = np.pi/2)
        t += 0.5*max(qC['piLen'],qT['piLen']) + tBuf
        ### BEGIN CONTROLLED Z GATE ###
        qC['z'] += controlEnvelope
        qT['z'] += targetEnvelope
        t += grapeData[-1,0]*ns + tBuf
        ### END CONTROLLED Z GATE ###
        t += qT['piLen']/2
        qT['xy'] += eh.piHalfPulse(qT, t, phase=np.pi/2)
        t += qT['piLen']/2 + tBuf
        #Apply sideband mixing
        qC['xy'] = eh.mix(qC, qC['xy'])
        qT['xy'] = eh.mix(qT, qT['xy'])
        eh.correctCrosstalkZ(qubits)
        return measureFunc(server, qubits, t, stats=stats)
    data = sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy, pipesize=1)
    if collect:
        return data
        
def findBusFreq(dataset):
    data = dataset.data
    timeTraces = np.array([])
    zpas = np.unique(dataset.data[:,0])
    N = len(zpas)
    for zpa in zpas:
        idxs = find(data[:,0]==zpa)
        timeTraces = np.hstack((timeTraces, data[:,2][idxs]))
    timeTraces = np.reshape(timeTraces, (N,-1))
    #Subtract the mean from each row
    for i,trace in enumerate(timeTraces):
        timeTraces[i,:] = trace - np.mean(trace)
    #Add some padding
    padding = np.zeros((N,10000))
    timeTraces = np.hstack((timeTraces, padding))
    plt.figure()
    plt.plot(timeTraces[20])
    pss = np.array([])
    for timeTrace in timeTraces:
        ps = np.abs(np.fft.fft(timeTrace))
        l = len(ps)
        ps = ps[50:l/2]
        pss = np.hstack((pss,ps))
    pss = np.reshape(pss, (N,-1))
    maxs = np.array([])
    for ps in pss:
        maxs = np.hstack((maxs,np.argmax(ps)))
    return zpas, maxs