import labrad
from labrad.units import Unit
from labrad.types import Value
V, GHz, MHz, Hz, dBm, dB, K, ns = [Unit(s) for s in ('V','GHz','MHz', 'Hz', 'dBm', 'dB', 'K', 'ns')]
import numpy as np
import time
import pyle.dataking.util as util
import setbias as sb
import fpgatools2 as fp

def measure(sample):

    cxn = sample._cxn
    samp, dev = util.loadDeviceType(sample, 'PNA')
    ampMeas = dev[0]
    start = ampMeas['startFreq']*GHz
    stop = ampMeas['stopFreq']*GHz
    num_points = ampMeas['number of points']
    averages = ampMeas['averages']
    bandwidth = ampMeas['bandwidth']*Hz
    edelay = ampMeas['electricalDelay']*ns
    dac = ampMeas['dacBoard']
    channel = ampMeas['fastbiasDacChannel']
    power = ampMeas['power']*dBm
    res = float(ampMeas['biasResistance'])
    steps = float(ampMeas['steps'])
    if dac == 'ADR lab FPGA 5':
        seq = cxn.sequencer
        seq.select_device(dac)
    
    # num_points is the number of points per pna sweep
        
    if ampMeas['port'] == 'Port1':
        independents = ['Iflux [mA]','Frequency [GHz]']
        dependents = ['Phase (S21) [dB]']
    elif ampMeas['port'] == 'Port2':
        independents = ['Iflux [mA]','Frequency [GHz]']
        dependents = ['Phase (S12) [dB]']
    
    # get the parameters from the data vault
    
    def get_pna_phase(start_freq, stop_freq):
        pna.frequency_range(start_freq, stop_freq)
        raw_data = pna.freq_sweep_phase()
        return raw_data
    
    pna = cxn.pna_x
    pna.select_device(ampMeas['pna_gpib'])
    pna.power(power)
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
    nameString = 'Phase vs Flux and Frequency ' + ampMeas['name']
        
    dv.new(nameString, independents, dependents)
    p = dv.packet()
    for key in ampMeas.keys():
        p.add_parameter(key, ampMeas.__getitem__(key))
    p.send()
    
    fluxStep = float(5.0)/steps
    fluxValues = np.arange(-2.5, 2.5+fluxStep, fluxStep)
    for flux in fluxValues:
        if dac == 'ADR lab FPGA 5':
            chan = fp.constMem(flux, fluxDAC=0)
            seq.packet().memory(chan).run_sequence(30).send()
        else:
            sb.set_fb(dac, cxn, flux*V, channel)
        raw_data = get_pna_phase(start, stop)
        freqs = np.asarray(raw_data[0])*1e-9 # (downconvert to GHz)
        fluxPoints = np.asarray([(flux/res)*1000]*len(freqs))
        freqs.shape = (len(freqs),1)
        fluxPoints.shape = (len(freqs),1)
        phase21 = np.asarray(raw_data[1][0])
        
        data = np.array(np.hstack((fluxPoints, freqs, phase21)))
        
        dv.add(data)
    