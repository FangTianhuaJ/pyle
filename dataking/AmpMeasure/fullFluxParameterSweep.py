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
import noiseScan as ns

def measure(sample):

    cxn = sample._cxn
    samp, dev = util.loadDeviceType(sample, 'fluxScan')
    fluxScan = dev[0]
    points = fluxScan['points']
    anritsu = cxn.anritsu_server
    anritsu.select_device(fluxScan['anritsu'])
    anritsu.output(False)
    anritsu.output(True)
    anritsu.output(False)
    dac = fluxScan['dacBoard']
    chan = fluxScan['fastbiasDacChannel']
    PNA = sample['PNA']
    SA = sample['noiseMeas']
    

    for point in points:
        bias = float(point[0])/1000
        freq = float(point[1])
        pumpPow = float(point[2])
        
        sb.set_fb(dac, cxn, bias*V, chan)
        
        SA['signalFreq'] = freq*1000
        SA['signalPower'] = pumpPow
        SA['freq'] = freq*1000
        ns.doubleNoiseScan(sample)
        
        PNA['startFreq'] = freq-0.05
        PNA['stopFreq'] = freq+0.05
        PNA['pumpFreq'] = freq
        PNA['pumpPower'] = 'off'
        PNA['power'] = -60
        fs.freqScan(sample)
        
        anritsu.output(True)
        PNA['pumpPower'] = pumpPow
        fs.freqScan(sample)
        
        PNA['startPow'] = -70
        PNA['stopPow'] = -40
        PNA['CWfrequency' ] = freq+0.003
        ps.powerScan(sample)
        
        anritsu.output(False)
        
        print 'done with point'
        
        
        
        
        
        
        
        
        
        
        
                