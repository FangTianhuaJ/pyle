import labrad
from labrad.units import Unit
from labrad.types import Value
V, GHz, MHz, Hz, dBm, dB, K, ns = [Unit(s) for s in ('V','GHz','MHz', 'Hz', 'dBm', 'dB', 'K', 'ns')]
import numpy as np
import time
import setbias as sb
import powerScan as ps
import pyle.dataking.util as util
import frequencyScan as fs
import powerScan as ps

def varyFreq(sample):

    cxn = sample._cxn
    samp, dev = util.loadDeviceType(sample, 'PNA')
    ampMeas = dev[0]
    params = sample['PNA']
    anritsu = cxn.anritsu_server
    anritsu.select_device(ampMeas['anritsu'])
    anritsu.output(False)
    anritsu.output(True)
    anritsu.output(False)
    pumpFreq = ampMeas['pumpFreq']*GHz
    pumpPower = ampMeas['pumpPower']*dBm
    freqSteps = ampMeas['steps']
    startFreq = ampMeas['startFreq']
    stopFreq = ampMeas['stopFreq']
    steps = ampMeas['steps']
    anritsu.frequency(float(pumpFreq.inUnitsOf('MHz')))
    anritsu.amplitude(float(pumpPower))
    freqStep = (stopFreq - startFreq)/steps
    freqs = np.arange(startFreq, stopFreq + freqStep, freqStep)
    
    anritsu.output(True)
    
    for freq in freqs:
        params['CWfrequency'] = freq
        ps.powerScan(sample)
        print freq + ' GHz'
        
    anritsu.output(False)

    
def varyPower(sample, detuning):

    cxn = sample._cxn
    samp, dev = util.loadDeviceType(sample, 'PNA')
    ampMeas = dev[0]
    params = sample['PNA']
    anritsu = cxn.anritsu_server
    anritsu.select_device(ampMeas['anritsu'])
    anritsu.output(False)
    anritsu.output(True)
    anritsu.output(False)
    pumpFreq = ampMeas['pumpFreq']*GHz
    pumpPower = ampMeas['pumpPower']
    anritsu.frequency(float(pumpFreq.inUnitsOf('MHz')))
    powers = np.arange(pumpPower - 0.20, pumpPower + 0.22, 0.02)
    
    anritsu.output(True)
    
    for power in powers:
        anritsu.amplitude(float(power))
        params['pumpPower'] = power
        params['CWfrequency'] = float(pumpFreq) + detuning
        ps.powerScan(sample)
        print power + ' dBm'
        
    anritsu.output(False)