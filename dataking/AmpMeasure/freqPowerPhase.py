import labrad
from labrad.units import Unit
from labrad.types import Value
V, GHz, MHz, Hz, dBm, dB, K, ns = [Unit(s) for s in ('V','GHz','MHz', 'Hz', 'dBm', 'dB', 'K', 'ns')]
import numpy as np
import time
import pyle.dataking.util as util

def measure(sample):

    cxn = sample._cxn
    samp, dev = util.loadDeviceType(sample, 'PNA')
    ampMeas = dev[0]
    start = ampMeas['startFreq']*GHz
    stop = ampMeas['stopFreq']*GHz
    num_points = ampMeas['number of points']
    averages = ampMeas['averages']
    startPower = ampMeas['startPow']*dBm
    stopPower = ampMeas['stopPow']*dBm
    bandwidth = ampMeas['bandwidth']*Hz
    edelay = ampMeas['electricalDelay'].value
    
    # num_points is the number of points per pna sweep
        
    if ampMeas['port'] == 'Port1':
        independents = ['Frequency [GHz]', 'Power [dBm]']
        dependents = ['Phase (S21) [deg]']
    elif ampMeas['port'] == 'Port2':
        independents = ['Frequency [GHz]', 'Power [dBm]']
        dependents = ['Phase (S12) [deg]']
    
    # get the parameters from the data vault
    
    def get_pna_phase(start_freq, stop_freq):
        pna.frequency_range(start_freq, stop_freq)
        raw_data = pna.freq_sweep_phase()
        return raw_data
    
    pna = cxn.pna_x
    pna.select_device(ampMeas['pna_gpib'])
    pna.bandwidth(bandwidth)
    pna.num_points(int(num_points))
    #pna.electrical_delay(edelay)
    
    if ampMeas['port'] == 'Port1':
        pna.s_parameters(['S21'])
    elif ampMeas['port'] == 'Port2':
        pna.s_parameters(['S12'])
        
    # set up data vault parameters
    dv = cxn.data_vault
    dv.cd('')
    dv.cd(sample._dir, True)
    nameString = 'Phase vs Frequency and Power (frequency sweep) ' + ampMeas['name']
        
    dv.new(nameString, independents, dependents)
    p = dv.packet()
    for key in ampMeas.keys():
        p.add_parameter(key, ampMeas.__getitem__(key))
    p.send()
    
    powerstep = float(stopPower-startPower)/100.0
    powerValues = np.arange(startPower, stopPower+powerstep, powerstep)
    for power in powerValues:
        pna.power(power)
        raw_data = get_pna_phase(start, stop)
        freqs = np.asarray(raw_data[0])*1e-9 # (downconvert to GHz)
        powerPoints = np.asarray([power]*len(freqs))
        freqs.shape = (len(freqs),1)
        powerPoints.shape = (len(powerPoints),1)
        phase21 = np.asarray(raw_data[1][0])
        
        data = np.array(np.hstack((freqs, powerPoints, phase21)))
        
        dv.add(data)

def measurePow(sample):

    cxn = sample._cxn
    samp, dev = util.loadDeviceType(sample, 'PNA')
    ampMeas = dev[0]
    start = ampMeas['startFreq']*GHz
    stop = ampMeas['stopFreq']*GHz
    num_points = ampMeas['number of points']
    averages = ampMeas['averages']
    startPower = ampMeas['startPow']*dBm
    stopPower = ampMeas['stopPow']*dBm
    bandwidth = ampMeas['bandwidth']*Hz
    edelay = ampMeas['electricalDelay'].value
    steps = float(ampMeas['steps'])
    
    # num_points is the number of points per pna sweep
        
    if ampMeas['port'] == 'Port1':
        independents = ['Frequency [GHz]', 'Power [dBm]']
        dependents = ['Phase (S21) [dB]']
    elif ampMeas['port'] == 'Port2':
        independents = ['Frequency [GHz]', 'Power [dBm]']
        dependents = ['Phase (S12) [dB]']
    
    # get the parameters from the data vault
    
    def get_power_phase(start_pow, stop_pow):
        pna.power_range(start_pow, stop_pow)
        raw_data = pna.power_sweep_phase()
        return raw_data
    
    pna = cxn.pna_x
    pna.select_device(ampMeas['pna_gpib'])
    pna.bandwidth(bandwidth)
    pna.num_points(int(num_points))
    pna.electrical_delay(edelay)
    
    if ampMeas['port'] == 'Port1':
        pna.s_parameters(['S21'])
    elif ampMeas['port'] == 'Port2':
        pna.s_parameters(['S12'])
        
    # set up data vault parameters
    dv = cxn.data_vault
    dv.cd('')
    dv.cd(sample._dir, True)
    nameString = 'Phase vs Frequency and Power (power sweep) ' + ampMeas['name']
        
    dv.new(nameString, independents, dependents)
    p = dv.packet()
    for key in ampMeas.keys():
        p.add_parameter(key, ampMeas.__getitem__(key))
    p.send()
    
    freqstep = float(stop-start)/steps
    freqValues = np.arange(start, stop+freqstep, freqstep)
    for freq in freqValues:
        pna.frequency(freq*GHz)
        raw_data = get_power_phase(startPower, stopPower)
        powers = np.asarray(raw_data[0]) # (downconvert to GHz)
        freqPoints = np.asarray([freq]*len(powers))
        powers.shape = (len(powers),1)
        freqPoints.shape = (len(powers),1)
        phase21 = np.asarray(raw_data[1][0])
        
        data = np.array(np.hstack((freqPoints, powers, phase21)))
        
        dv.add(data)

