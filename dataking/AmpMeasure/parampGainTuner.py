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
    startPower = ampMeas['startPow']*dBm
    stopPower = ampMeas['stopPow']*dBm
    num_points = ampMeas['number of points']
    averages = ampMeas['averages']
    bandwidth = ampMeas['bandwidth']*Hz
    edelay = ampMeas['electricalDelay']*ns
    dac = ampMeas['dacBoard']
    channel = ampMeas['fastbiasDacChannel']
    pnaPower = ampMeas['power']*dBm
    res = float(ampMeas['biasResistance'])
    steps = float(ampMeas['steps'])
    anritsuAddr = ampMeas['anritsu']
    if dac == 'ADR lab FPGA 5':
        seq = cxn.sequencer
        seq.select_device(dac)

    # num_points is the number of points per pna sweep

    if ampMeas['port'] == 'Port1':
        independents = ['Frequency [GHz]','Power [dBm]']
        dependents = ['Magnitude (S21) [dB]']
    elif ampMeas['port'] == 'Port2':
        independents = ['Frequency [GHz]','Power [dBm]']
        dependents = ['Magnitude (S12) [dB]']

    # get the parameters from the data vault

    def get_pna_data(start_freq, stop_freq):
        pna.frequency_range(start_freq, stop_freq)
        raw_data = pna.freq_sweep()
        f = np.asarray(raw_data[0])
        s = np.asarray(raw_data[1]).T
        f.shape = (len(f), 1)
        pna_data = np.hstack((f, s))
        return pna_data

    def getMagPhase(c): #returns magnitude and phase of pna data
        return 2 * 10 * np.log10(abs(c)), np.angle(c)

    def smooth(data, kick):
        end = len(data)-1
        counter = 0
        while (counter<end):
            if(data[counter]-data[counter+1]>kick):
                data = np.r_[data[:counter],data[counter+2:]]
                end = len(data)-1
                counter = -1
            counter+=1
        return data

    pna = cxn.pna_x
    pna.select_device(ampMeas['pna_gpib'])
    pna.power(pnaPower)
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
    nameString = 'Gain vs Power and Frequency ' + ampMeas['name']

    dv.new(nameString, independents, dependents)
    p = dv.packet()
    for key in ampMeas.keys():
        p.add_parameter(key, ampMeas.__getitem__(key))
    p.send()

    freqStep = (float(stop-start))/steps
    freqs = np.arange(start, stop+freqStep, freqStep)
    powStep = float(stopPower-startPower)/steps
    powers = np.arange(startPower, stopPower+powStep, powStep)
    anritsu = cxn.anritsu_server
    anritsu.select_device(anritsuAddr)
    anritsu.output(True)

    for freq in freqs:
        anritsu.frequency(float(freq*1000))
        centGain = 0
        centPower=0
        for power in powers:
            anritsu.amplitude(power)
            raw_data = get_pna_data((freq-0.05)*1e9, (freq+0.05)*1e9)
            Sdata = getMagPhase(raw_data[:,1])
            gain = np.asarray(Sdata[0])
            gain = smooth(gain,5)
            maxGain = gain.max()
            if maxGain>centGain:
                centGain = maxGain
                centPower = power
            dv.add(freq, power, maxGain)

        powers = np.arange(centPower-0.2,centPower+0.21,0.01)




