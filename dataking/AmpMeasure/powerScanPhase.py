import labrad
from labrad.units import Unit
from labrad.types import Value
V, GHz, MHz, Hz, dBm, dB, K = [Unit(s) for s in ('V','GHz','MHz', 'Hz', 'dBm', 'dB', 'K')]
import numpy as np
import time
import pyle.dataking.util as util

def phaseSweep(sample):

    cxn = sample._cxn
    samp, dev = util.loadDeviceType(sample, 'PNA')
    ampMeas = dev[0]
    start = ampMeas['startPow']*dBm
    stop = ampMeas['stopPow']*dBm
    num_points = ampMeas['number of points']
    averages = ampMeas['averages']
    freq = ampMeas['CWfrequency']*GHz
    bandwidth = ampMeas['bandwidth']*Hz
    phasOffs = ampMeas['phaseOffset']
    pumpPhase = ampMeas['pumpPhase']
    
    # num_points is the number of points per pna sweep
        
    if ampMeas['port'] == 'Port1':
        independents = ['Power [dBm]']
        dependents = ['Phase (S21) [deg]']
    elif ampMeas['port'] == 'Port2':
        independents = ['Power [dBm]']
        dependents = ['Phase (S21) [deg]']
    
    # get the parameters from the data vault
    
    def get_pna_data(start, stop):
        pna.power_range(start, stop)
        raw_data = pna.power_sweep_phase()
        return raw_data
    
    def getMagPhase(c): #returns magnitude and phase of pna data
        return 2 * 10 * np.log10(abs(c)), np.angle(c)
    
    pna = cxn.pna_x
    pna.select_device(ampMeas['pna_gpib'])
    pna.frequency(freq)
    pna.bandwidth(bandwidth)
    pna.phase_offset(phasOffs)
    pna.source_phase_offset(pumpPhase)
    pna.num_points(int(num_points))
    
    if ampMeas['port'] == 'Port1':
        pna.s_parameters(['S11', 'S21'])
    elif ampMeas['port'] == 'Port2':
        pna.s_parameters(['S12', 'S22'])
        
    # set up data vault parameters
    dv = cxn.data_vault
    dv.cd('')
    dv.cd(sample._dir, True)
    nameString = 'Power Sweep phase only ' + ampMeas['name']
        
    dv.new(nameString, independents, dependents)
    p = dv.packet()
    for key in ampMeas.keys():
        p.add_parameter(key, ampMeas.__getitem__(key))
    p.send()
    
    raw_data = get_pna_data(start, stop)
    powers = np.asarray(raw_data[0]).reshape(num_points,1) 
    points = len(powers)
    phase21 = np.asarray(raw_data[1][1])
    
    data = np.array(np.hstack((powers, phase21)))
    
    dv.add(data)

    
def comparePhaseSweep(sample):

    cxn = sample._cxn
    samp, dev = util.loadDeviceType(sample, 'PNA')
    ampMeas = dev[0]
    start = ampMeas['startPow']*dBm
    stop = ampMeas['stopPow']*dBm
    num_points = ampMeas['number of points']
    averages = ampMeas['averages']
    freq = ampMeas['CWfrequency']*GHz
    bandwidth = ampMeas['bandwidth']*Hz
    phasOffs = ampMeas['phaseOffset']
    phas1 = ampMeas['pumpPhase']
    
    # num_points is the number of points per pna sweep
        
    if ampMeas['port'] == 'Port1':
        independents = ['Power [dBm]']
        dependents = ['Phase + Sig [deg]','Phase - Sig[deg]']
    elif ampMeas['port'] == 'Port2':
        independents = ['Power [dBm]']
        dependents = ['Phase + Sig [deg]','Phase - Sig[deg]']
    
    # get the parameters from the data vault
    
    def get_pna_data(start, stop):
        pna.power_range(start, stop)
        raw_data = pna.power_sweep_phase()
        return raw_data
    
    def getMagPhase(c): #returns magnitude and phase of pna data
        return 2 * 10 * np.log10(abs(c)), np.angle(c)
    
    pna = cxn.pna_x
    pna.select_device(ampMeas['pna_gpib'])
    pna.frequency(freq)
    pna.bandwidth(bandwidth)
    pna.phase_offset(phasOffs)
    pna.num_points(int(num_points))
    
    if ampMeas['port'] == 'Port1':
        pna.s_parameters(['S11', 'S21'])
    elif ampMeas['port'] == 'Port2':
        pna.s_parameters(['S12', 'S22'])
        
    # set up data vault parameters
    dv = cxn.data_vault
    dv.cd('')
    dv.cd(sample._dir, True)
    nameString = 'Power Sweep phase compare ' + ampMeas['name']
        
    dv.new(nameString, independents, dependents)
    p = dv.packet()
    for key in ampMeas.keys():
        p.add_parameter(key, ampMeas.__getitem__(key))
    p.send()
    
    pna.source_phase_offset(phas1)
    raw_data = get_pna_data(start, stop)
    powers = np.asarray(raw_data[0]).reshape(num_points,1) 
    points = len(powers)
    inphase = np.asarray(raw_data[1][1])
    
    pna.source_phase_offset(phas1+180)
    raw_data = get_pna_data(start, stop)
    powers = np.asarray(raw_data[0]).reshape(num_points,1) 
    points = len(powers)
    outphase = np.asarray(raw_data[1][1])
    
    data = np.array(np.hstack((powers, inphase, outphase)))
    
    dv.add(data)