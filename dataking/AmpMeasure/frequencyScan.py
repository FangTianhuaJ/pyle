import labrad
from labrad.units import Unit
from labrad.types import Value
V, GHz, MHz, Hz, dBm, dB, K = [Unit(s) for s in ('V','GHz','MHz', 'Hz', 'dBm', 'dB', 'K')]
import numpy as np
import time
import pyle.dataking.util as util

def freqScan(sample):

    cxn = sample._cxn
    samp, dev = util.loadDeviceType(sample, 'PNA')
    ampMeas = dev[0]
    start = ampMeas['startFreq']*GHz
    stop = ampMeas['stopFreq']*GHz
    num_points = ampMeas['number of points']
    averages = ampMeas['averages']
    power = ampMeas['power']*dBm
    bandwidth = ampMeas['bandwidth']*Hz

    # num_points is the number of points per pna sweep

    if ampMeas['port'] == 'Port1':
        independents = ['frequency [GHz]']
        dependents = ['Magnitude (S21) [dB]', 'Phase (S21) [rad]',
                                                   'Magnitude (S11) [dB]', 'Phase (S11) [rad]']
    elif ampMeas['port'] == 'Port2':
        independents = ['frequency [GHz]']
        dependents = ['Magnitude (S22) [dB]', 'Phase (S22) [rad]',
                                                   'Magnitude (S12) [dB]', 'Phase (S12) [rad]']

    # get the parameters from the data vault

    def get_pna_data(start, stop):
        pna.power_range(start, stop)
        raw_data = pna.power_sweep()
        f = np.asarray(raw_data[0])
        s = np.asarray(raw_data[1]).T
        f.shape = (len(f), 1)
        pna_data = np.hstack((f, s))
        return pna_data

    def getMagPhase(c): #returns magnitude and phase of pna data
        return 2 * 10 * np.log10(abs(c)), np.angle(c)

    pna = cxn.pna_x
    pna.select_device(ampMeas['pna_gpib'])
    pna.power(power)
    pna.bandwidth(bandwidth)
    pna.num_points(int(num_points))

    if ampMeas['port'] == 'Port1':
        pna.s_parameters(['S11', 'S21'])
    elif ampMeas['port'] == 'Port2':
        pna.s_parameters(['S12', 'S22'])

    # set up data vault parameters
    dv = cxn.data_vault
    dv.cd('')
    dv.cd(sample._dir, True)
    nameString = 'Frequency Sweep ' + ampMeas['name']

    dv.new(nameString, independents, dependents)
    p = dv.packet()
    for key in ampMeas.keys():
        p.add_parameter(key, ampMeas.__getitem__(key))
    p.send()

    raw_data = get_pna_data(start, stop)

    # add live view later
    freqs = raw_data[:,0]*1e-9 # (downconvert to GHz)
    S_through = getMagPhase(raw_data[:,2]) # S12 or S21, depending on PORT
    S_reflected = getMagPhase(raw_data[:,1]) # S11 or S22 depending on PORT

    data = np.array(np.vstack((freqs.T, S_through[0], S_through[1], S_reflected[0], S_reflected[1]))).T.astype(float)

    dv.add(data)
