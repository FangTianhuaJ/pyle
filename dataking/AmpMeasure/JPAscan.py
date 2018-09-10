# -*- coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

import time
import itertools

import labrad
import labrad.units as U

import pyle
from pyle.dataking import sweeps
from pyle.dataking.util import loadQubits
from pyle.util import sweeptools as st
import pyle.gateCompiler as gc
import pyle.gates as gates
from pyle.dataking.fpgaseqTransmonV7 import runQubits
from pyle.analysis import readout
from pyle.pipeline import returnValue, FutureList


# COLORS
BLUE   = "#348ABD"
RED    = "#E24A33"
PURPLE = "#988ED5"
YELLOW = "#FBC15E"
GREEN  = "#8EBA42"
PINK   = "#FFB5B8"
GRAY   = "#777777"

FPGANAME = "GHz FPGAs"

V, mV, us, ns, GHz, MHz, dBm, rad = [U.Unit(s) for s in ('V', 'mV', 'us', 'ns', 'GHz', 'MHz', 'dBm', 'rad')]

# helper functions
def makeUwaveSourcePacket(cxn, dev):
    pump_dev = dev['pumpChannels']
    server_name, device_name = pump_dev
    p = cxn[server_name].packet()
    p.select_device(device_name)
    p.frequency(dev['pumpFrequency'])
    p.amplitude(dev['pumpPower'])
    p.output(dev['pumpOutput'])
    return p

def makeFPGAPacket(cxn, dev, zero=False):
    flux_dev = dev['fluxChannels']
    flux_type, (boardName, fpgaChannel) = flux_dev
    bias = 0*mV if zero else dev['biasOperate']
    dacName = dev['biasOperateDAC']
    p = cxn[FPGANAME].packet()
    p.select_device(boardName)
    mem = makeMemValue(bias, channel=fpgaChannel, dacName=dacName)
    p.memory(mem)
    p.sram(np.array([0]*30, dtype='uint32'))
    p.daisy_chain([boardName])
    p.run_sequence(30, False)
    return p

def makeMemValue(volt, channel='out0', dacName='FAST'):
    if dacName.upper() == 'FAST':
        dac = 1
        slow = 0
    elif dacName.upper() == 'SLOW':
        dac = 1
        slow = 1
    elif dacName.upper() == 'FINE':
        dac = 0
        slow = 0
    else:
        dac = 1
        slow = 0
    if dacName.upper() in ["FINE"]:
        data = long((volt['mV']/2500.0*0xFFFF))
    else:
        data = long((volt['mV']+2500)/5000.0*0xFFFF)

    data = data & 0xFFFF
    data = data << 3
    dac = (dac<<19) #DAC1
    slow = (slow<<2) #FAST
    channel = channel.lower()
    if channel == 'out0':
        mem = [0x000000,
               0x300250, #delay 250cycle
               (1<<20) | ((dac+data+slow) & 0xFFFFF),
               0xF00000]
    elif channel == 'out1':
        mem = [0x000000,
               0x300250, #delay 250cycle
               (2<<20) | ((dac+data+slow) & 0xFFFFF),
               0xF00000]
    else:
        mem = [0x000000, 0, 0, 0xF00000]
    return mem

def setFastbiasLevel(fpga, board, channel, volt):
    fpga.select_device(board)
    mem = makeMemValue(volt, channel=channel)
    fpga.memory(mem)
    fpga.sram(np.array([0]*30,dtype='uint32'))
    fpga.daisy_chain([board])
    fpga.run_sequence(30, False)

def VNA(vna, center, span, power, numpoints, bandwidth, average, waittime, cable_delay=0*ns):
    vna.select_device()
    start = center - span/2.0
    stop = center + span/2.0
    # VNA setup
    if cable_delay != 0*ns:
        vna.electrical_delay(cable_delay)
        vna.port_extensions(True)
    else:
        vna.port_extensions(False)
    vna.s_parameters(["S21"])
    vna.frequency_range([start, stop])
    vna.power(power)
    vna.num_points(numpoints)
    vna.bandwidth(bandwidth)
    vna.averages(1)
    time.sleep(1)
    vna.autoscale()
    vna.averages(average)

    # sleep to ensure the
    time.sleep(waittime/2.0)
    vna.autoscale()
    time.sleep(waittime/2.0)
    vna.autoscale()
    freq, spara = vna.freq_sweep()
    freq = np.array(freq['MHz'])
    spara = np.asarray(spara[0])
    mag = 20*np.log10(np.abs(spara))
    phase = np.angle(spara)

    return freq, spara, mag, phase

def runVNA(dev, zero=True):
    with labrad.connect() as cxn:
        p_uwave = makeUwaveSourcePacket(cxn, dev)
        p_fpga = makeFPGAPacket(cxn, dev, zero=False)
        p_uwave.send()
        p_fpga.send()
        vna = cxn[dev['VNA Server']]
        center = dev['VNA Center Frequency']
        span = dev['VNA Span']
        power = dev['VNA power']
        numpoints = dev["VNA number of points"]
        bandwidth = dev['VNA bandwidth']
        average = dev['average']
        waittime = dev['wait time']
        cable_delay = dev.get('VNA cable delay time', 0*ns)
        freq, Spara, mag, phase = VNA(vna, center, span, power, numpoints, bandwidth, average, waittime, cable_delay)
        p_fpga_zero = makeFPGAPacket(cxn, dev, zero=zero)
        p_fpga_zero.send()

    return freq, Spara, mag, phase

def gridSweep(axes):
    if not len(axes):
        yield (), ()
    else:
        (param, _label), rest = axes[0], axes[1:]
        if np.iterable(param):
            for val in param:
                for all, swept in gridSweep(rest):
                    yield (val,) + all, (val,) + swept
        else:
            for all, swept in gridSweep(rest):
                yield (param,) + all, swept

# measurement functions by using VNA and Spectrum Analyzer
def JPAfluxSpectroscoy(Sample, measure=0, fluxbias=st.r[-1.0:1.0:0.02, V], center=6.0*GHz, span=4*GHz, average=50,
                    name='JPA flux spectroscopy', save=True, noisy=True, waittime=5):
    """
    flux bias the JPA, and measure the S parameter without pumping
    """
    sample, devs = loadQubits(Sample)
    dev = devs[measure]

    dev['VNA Center Frequency'] = center
    dev['VNA Span'] = span
    dev['average'] = average
    dev['wait time'] = waittime
    dev['pumpOutput'] = False

    axes = [(fluxbias, 'flux bias'), ("signal freq", "MHz")]
    deps = [('Mag', "S21", ""), ("Phase", "S21", "")]
    kw = {'pumpOutput': False}

    dataset = sweeps.prepDataset(sample, name, axes=axes, dependents=deps, measure=measure, kw=kw)

    with dataset:
        data = []
        for fb in fluxbias:
            dev['biasOperate'] = fb
            if noisy:
                print("%s" %fb)
            freq, Spara, mag, phase = runVNA(dev)
            fb_ = np.ones_like(freq)*(fb[fb.unit])
            dat = np.vstack([fb_, freq, mag, phase]).T
            data.append(dat)
            if save:
                dataset.add(dat)

    data = np.array(data)

    return data


def JPASpectroscopy(Sample, measure=0, pumpFrequency=st.r[12.0:14.0:0.05, GHz], pumpPower=st.r[-10:10:0.5, dBm],
                    fluxbias=st.r[-1.0:1.0:0.04, V], center=6.0*GHz, span=4*GHz, average=50, pump=True,
                    name='JPA Spectroscopy', save=True, waittime=5):
    """
    sweep pumpFrequency, pumpPower, fluxbias, measure the S parameter,
    find an operation point.
    """
    sample, devs = loadQubits(Sample)
    dev = devs[measure]

    dev['VNA Center Frequency'] = center
    dev['VNA Span'] = span
    dev['average'] = average
    dev['wait time'] = waittime
    dev['pumpOutput'] = pump

    axes = [(fluxbias, 'flux bias'), (pumpFrequency, "pump frequency"),
            (pumpPower, "pump power"), ("signal freq", "MHz")]
    deps = [("Mag", "S21", ""), ("Phase", "S21", "")]

    name += " Sweep"
    if np.iterable(fluxbias):
        name += " flux"
    if np.iterable(pumpFrequency):
        name += " pumpFreq"
    if np.iterable(pumpPower):
        name += " pumpPower"

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure)

    def func(fb, pumpf, power):
        print("%s, %s, %s" %(fb, pumpf, power))
        dev['biasOperate'] = fb
        dev['pumpFrequency'] = pumpf
        dev['pumpPower'] = power
        freq, Spara, mag, phase = runVNA(dev, zero=False)
        data = np.vstack([freq, mag, phase]).T
        return data

    data = list()
    with dataset:
        for all_, swept in gridSweep(axes[:-1]):
            ans = func(*all_)
            pre = np.array(sweeps.getValue(swept))
            pre = np.tile(pre, (ans.shape[0], 1))
            dat = np.hstack((pre, ans))
            data.append(dat)
            if save:
                dataset.add(dat)

    data = np.array(data)

    return data

def JPANoiseTemperatureScan(Sample, measure, center, span, average=100, fluxbias=None, pumpPower=None,
                            pumpFrequency=None, name="JPA Noise Temperature Scan", save=True, waittime=30):
    sample, devs = loadQubits(Sample)
    dev = devs[measure]
    cxn = Sample._cxn
    spec = cxn[dev['SPEC Server']]
    fluxbias = fluxbias if fluxbias is not None else dev['biasOperate']
    pumpPower = pumpPower if pumpPower is not None else dev['pumpPower']
    pumpFrequency = pumpFrequency if pumpFrequency is not None else dev['pumpFrequency']

    dev['VNA Center Frequency'] = center
    dev['VNA Span'] = span
    dev['average'] = average
    dev['wait time'] = waittime

    def run_spec(dev):
        spec.select_device(dev["SPEC ID"])
        spec.set_center_frequency(dev["VNA Center Frequency"])
        spec.set_span(dev['VNA Span'])
        spec.number_of_averages(max(dev['average'], dev.get('SPEC Min average', 100)))
        spec.number_of_points(dev['VNA number of points'])
        f_start, f_step, vals = spec.get_averaged_trace()
        return vals

    axes = [(fluxbias, 'flux bias'), (pumpFrequency, "pump frequency"),
            (pumpPower, "pump power"), ("signal freq", "MHz")]
    deps = [("Mag", "S21 (pump On)", "dB"), ("Noise Level", "pump On", "dBm"),
            ("Mag", "S21 (pump Off)", "dB"), ("Noise Level", "pump Off", "dBm")]


    name += " Sweep"
    if np.iterable(fluxbias):
        name += " flux"
    if np.iterable(pumpFrequency):
        name += " pumpFreq"
    if np.iterable(pumpPower):
        name += " pumpPower"

    def func(fb, pumpF, pumpP):
        print("%s, %s, %s" %(fb, pumpF, pumpP))
        dev['biasOperate'] = fb
        dev['pumpFrequency'] = pumpF
        dev['pumpPower'] = pumpP

        # pump off
        print("Pump Off ... ")
        dev['pumpOutput'] = False
        freq, Spara_off, mag_off, phase_off = runVNA(dev, zero=False)
        noise_level_off = run_spec(dev)

        time.sleep(1)
        # pump on
        print("Pump On ...")
        dev['pumpOutput'] = True
        freq, Spara_on, mag_on, phase_on = runVNA(dev, zero=False)
        noise_level_on = run_spec(dev)

        data = np.vstack([freq, mag_on, noise_level_on, mag_off, noise_level_off]).T
        return data

    dataset = sweeps.prepDataset(sample, name, axes, deps)
    data = list()
    with dataset:
        for all_, swept in gridSweep(axes[:-1]):
            ans = func(*all_)
            pre = np.array(sweeps.getValue(swept))
            pre = np.tile(pre, (ans.shape[0], 1))
            dat = np.hstack((pre, ans))
            data.append(dat)
            if save:
                dataset.add(dat)

    data = np.array(data)
    return data

def JPANoiseTemperatureFixedFreq(Sample, measure, freq, average=20, fluxbias=None, pumpPower=None,
                                 pumpFrequency=None, name="JPA Noise Temperature Fixed Freq", waittime=5, save=True):
    sample, devs = loadQubits(Sample)
    dev = devs[measure]
    cxn = Sample._cxn
    spec = cxn[dev['SPEC Server']]
    fluxbias = fluxbias if fluxbias is not None else dev['biasOperate']
    pumpPower = pumpPower if pumpPower is not None else dev['pumpPower']
    pumpFrequency = pumpFrequency if pumpFrequency is not None else dev['pumpFrequency']

    dev['VNA Center Frequency'] = freq
    dev['VNA Span'] = 2*dev['VNA bandwidth']
    dev['average'] = average
    dev['wait time'] = waittime

    def run_spec(dev):
        spec.select_device(dev["SPEC ID"])
        spec.set_center_frequency(dev["VNA Center Frequency"])
        spec.set_span(0*MHz)
        spec.number_of_averages(max(dev['average'], dev.get('SPEC Min average', 100)))
        spec.number_of_points(dev['VNA number of points'])
        f_start, f_step, vals = spec.get_averaged_trace()
        return np.mean(vals)

    axes = [(fluxbias, 'flux bias'), (pumpFrequency, "pump frequency"), (pumpPower, "pump power")]
    deps = [("Mag", "S21 (pump On)", "dB"), ("Noise Level", "pump On", "dBm"),
            ("Mag", "S21 (pump Off)", "dB"), ("Noise Level", "pump Off", "dBm")]
    kw = {'signal freq': freq}

    name += " %s" % freq
    name += " Sweep"
    if np.iterable(fluxbias):
        name += " flux"
    if np.iterable(pumpFrequency):
        name += " pumpFreq"
    if np.iterable(pumpPower):
        name += " pumpPower"

    def func(fb, pumpF, pumpP):
        print("%s, %s, %s" %(fb, pumpF, pumpP))
        dev['biasOperate'] = fb
        dev['pumpFrequency'] = pumpF
        dev['pumpPower'] = pumpP

        # pump off
        print("Pump Off ... ")
        dev['pumpOutput'] = False
        freq, Spara_off, mag_off, phase_off = runVNA(dev, zero=False)
        mag_off = np.mean(mag_off)
        noise_level_off = run_spec(dev)

        time.sleep(0.5)
        # pump on
        print("Pump On ...")
        dev['pumpOutput'] = True
        freq, Spara_on, mag_on, phase_on = runVNA(dev, zero=False)
        mag_on = np.mean(mag_on)
        noise_level_on = run_spec(dev)

        data = np.vstack([mag_on, noise_level_on, mag_off, noise_level_off]).T
        print "%.3f %.3f %.3f %.3f" %(mag_on, noise_level_on, mag_off, noise_level_off)
        return data

    dataset = sweeps.prepDataset(sample, name, axes, deps, kw=kw)
    data = list()
    with dataset:
        for all_, swept in gridSweep(axes):
            ans = func(*all_)
            pre = np.array(sweeps.getValue(swept))
            pre = np.tile(pre, (ans.shape[0], 1))
            dat = np.hstack((pre, ans))
            data.append(dat)
            if save:
                dataset.add(dat)

    data = np.array(data)
    return data


def JPANoiseTemperature(Sample, measure, center, span, average=100, fluxbias=None, pumpPower=None,
                        pumpFrequency=None, name='JPA Noise Temperature', save=True, plot=True, waittime=60):
    sample, devs = loadQubits(Sample)
    dev = devs[measure]
    cxn = Sample._cxn
    spec = cxn[dev["SPEC Server"]]

    axes = [('freq', 'MHz')]
    deps = [("Mag", "S21 (pump On)", "dB"), ("Noise Level", "pump On", "dBm"),
            ("Mag", "S21 (pump Off)", "dB"), ("Noise Level", "pump Off", "dBm")]

    fb = fluxbias if fluxbias is not None else dev['biasOperate']
    dev['biasOperate'] = fb
    pumpPower = pumpPower if pumpPower is not None else dev['pumpPower']
    dev['pumpPower'] = pumpPower
    pumpFrequency = pumpFrequency if pumpFrequency is not None else dev["pumpFrequency"]
    dev['pumpFrequency'] = pumpFrequency
    kw = {"biasOperate": fb, 'pumpFrequency': pumpFrequency, "pumpPower": pumpPower}

    def run_spec(dev):
        spec.select_device(dev["SPEC ID"])
        spec.set_center_frequency(dev["VNA Center Frequency"])
        spec.set_span(dev['VNA Span'])
        spec.number_of_averages(max(dev['average'], dev.get('SPEC Min average', 100)))
        spec.number_of_points(dev['VNA number of points'])
        spec.set_resolution_bandwidth_mhz(dev.get('SPEC res bandwidth', 1*MHz))
        f_start, f_step, vals = spec.get_averaged_trace()
        return vals

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    dev['VNA Center Frequency'] = center
    dev['VNA Span'] = span
    dev['average'] = average
    dev['wait time'] = waittime

    # pump on
    dev['pumpOutput'] = True
    print("Measuring when Pump On...")
    freq, Spara_on, mag_on, phase_on = runVNA(dev, zero=False)
    noise_level_on = run_spec(dev)

    time.sleep(1)

    # pump off
    dev['pumpOutput'] = False
    print("Measuring when Pump Off...")
    freq, Spara_off, mag_off, phase_off = runVNA(dev, zero=False)
    noise_level_off = run_spec(dev)


    data = np.vstack([freq, mag_on, noise_level_on, mag_off, noise_level_off]).T
    if save:
        with dataset:
            dataset.add(data)

    Y = 10**((noise_level_on-noise_level_off)/10.0)
    Gi = 10**(dev['insertion loss']['dB']/10.0)
    Ga = 10**(dev['cable loss']['dB']/10.0)
    gain = 10**((mag_on - mag_off)/10.0)
    temp = (Y-1)*(dev['T_HEMT']['K'])/(gain*Gi**2*Ga)

    if plot:
        plt.figure(figsize=(9,4))
        plt.subplot(121)
        plt.plot(freq, mag_on - mag_off, '.-')
        plt.xlabel("Freq [MHz]")
        plt.ylabel("Gain [dB]")
        plt.grid()
        plt.subplot(122)
        plt.plot(freq, temp, '.', label='T(Amp)')
        plt.plot(freq, 6.63e-34/1.38e-23*freq*1e6, label='quantum limit')
        plt.xlabel("Freq [MHz]")
        plt.ylabel("Noise Temperature")
        plt.grid()
        plt.legend()
        plt.tight_layout()

    return data, temp

def JPASaturation(Sample, measure, signalPower, center, span, average=100, name="JPA Saturation Power", save=True,
                  pump=True, waittime=30):
    """
    change the signal power, and find the saturation power of JPA
    """
    sample, devs = loadQubits(Sample)
    dev = devs[measure]

    dev['VNA Center Frequency'] = center
    dev['VNA Span'] = span
    dev['average'] = average
    dev['wait time'] = waittime
    dev['pumpOutput'] = pump

    axes = [(signalPower, 'signal power'), ("signal freq", "MHz")]
    deps = [("Mag", "S21", ""), ("Phase", "S21", "")]


    kw = {"pumpOutput": pump}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(signalP):
        print("signal power %s" %signalP)
        dev['VNA power'] = signalP
        freq, Spara, mag, phase = runVNA(dev, zero=False)
        data = np.vstack([freq, mag, phase]).T
        return data

    data = list()
    with dataset:
        for all_, swept in gridSweep(axes[:-1]):
            ans = func(*all_)
            pre = np.array(sweeps.getValue(swept))
            pre = np.tile(pre, (ans.shape[0], 1))
            dat = np.hstack((pre, ans))
            data.append(dat)
            if save:
                dataset.add(dat)

    data = np.array(data)

    return data

def plot_noise_temp(dataset, window_len=1):
    """
    plot gain and noise temperature
    @param dataset: localDataset Object
    @param window_len: if window_len>1, smooth the gain data
    @return: freq, gain, noisetemp
    """
    from pyle.signalProcess import smooth
    data = np.array(dataset)
    params = dataset.parameters
    dev_name = params['config'][params['measure'][0]]
    dev = params[dev_name]
    freq, mag_on, noise_level_on, mag_off, noise_level_off = data.T
    if window_len>1:
        mag_on = smooth(mag_on,   window_len=window_len)
        mag_off = smooth(mag_off, window_len=window_len)
        noise_level_off = smooth(noise_level_off, window_len=window_len)
        noise_level_on = smooth(noise_level_on, window_len=window_len)
    Y = 10**((noise_level_on-noise_level_off)/10.0)
    Gi = 10**(dev['insertion loss']['dB']/10.0)
    Ga = 10**(dev['cable loss']['dB']/10.0)
    gain = 10**((mag_on - mag_off)/10.0)
    temp = (Y-1)*(dev['T_HEMT']['K'])/(gain*Gi**2*Ga)
    plt.figure(figsize=(9,4))
    plt.subplot(121)
    plt.plot(freq, mag_on - mag_off, '.-')
    plt.xlabel("Freq [MHz]")
    plt.ylabel("Gain [dB]")
    plt.grid()
    plt.subplot(122)
    plt.plot(freq, temp, '.', label='T(Amp)')
    plt.plot(freq, 6.63e-34/1.38e-23*freq*1e6, label='quantum limit')
    plt.xlabel("Freq [MHz]")
    plt.ylabel("Noise Temperature")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    return freq, mag_on-mag_off, temp

############ JPA measurement using ADC
def spectroscopy(Sample, pumpFrequency=st.r[11.0:12.0:0.005, GHz], pumpPower=st.r[-10:10:0.5, dBm],
                 fluxbias=st.r[-1.0:1.0:0.05, V], freqScan=st.r[5:6:0.01, GHz], readoutLen=2*us,
                 readoutPower=-20*dBm, demodFreq=10*MHz, pump=True, stats=600, log_scale=False,
                 name="spectroscopy", save=True, noisy=True):
    sample, devs, _ = gc.loadQubits(Sample, measure=0)

    axes = [(pumpFrequency, "pump freq"), (pumpPower, "pump power"),
            (fluxbias, "flux bias"), (freqScan, 'signal freq')]
    deps = [("Mag", "S21", "dB" if log_scale else ""), ("Phase", "S21", "rad")]
    kw = {'stats': stats, "demodFreq": demodFreq, 'log_scale': log_scale,
          "readoutLen": readoutLen, "readoutPower": readoutPower}

    name += " Sweep"
    if np.iterable(pumpFrequency):
        name += " pumpFreq"
    if np.iterable(pumpPower):
        name += " pumpPower"
    if np.iterable(fluxbias):
        name += " flux"
    if np.iterable(freqScan):
        name += " signal"

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=0, kw=kw)

    def func(server, pump_f, pump_p, fb, signal_f):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        rc = alg.readout_devs[0]
        rc['biasOperate'] = fb
        rc['carrierFrequency'] = signal_f - demodFreq
        rc['pumpFrequency'] = pump_f
        rc['pumpPower'] = pump_p
        rc['pumpOutput'] = pump
        q0['readoutFrequency'] = signal_f
        q0['readoutPower'] = readoutPower
        q0['readoutLen'] = readoutLen
        alg[gates.Readout([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats=stats, dataFormat='iqRaw')
        mag, ang = readout.iqToPolar(readout.parseDataFormat(data))
        if log_scale:
            mag = 20*np.log10(mag)
        returnValue([mag, ang])

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data

def noiseSpectroscopy(Sample, pumpFrequency=st.r[11.0:12.0:0.005, GHz], pumpPower=st.r[-10:10:0.5, dBm],
                      fluxbias=st.r[-1.0:1.0:0.05, V], freqScan=st.r[5:6:0.01, GHz], readoutLen=2*us,
                      demodFreq=10*MHz, pump=True, stats=600, name="Noise Spectroscopy",  save=True, noisy=True):
    sample, devs, _ = gc.loadQubits(Sample, measure=0)

    axes = [(pumpFrequency, "pump freq"), (pumpPower, "pump power"),
            (fluxbias, "flux bias"), (freqScan, 'signal freq')]
    deps = [("Mag", "S21", ""), ("Phase", "S21", "rad")]
    kw = {'stats': stats, "demodFreq": demodFreq,
          "readoutLen": readoutLen}

    name += " Sweep"
    if np.iterable(pumpFrequency):
        name += " pumpFreq"
    if np.iterable(pumpPower):
        name += " pumpPower"
    if np.iterable(fluxbias):
        name += " flux"
    if np.iterable(freqScan):
        name += " signal"

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=0, kw=kw)

    def func(server, pump_f, pump_p, fb, signal_f):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        rc = alg.readout_devs[0]
        rc['biasOperate'] = fb
        rc['carrierFrequency'] = signal_f - demodFreq
        rc['pumpFrequency'] = pump_f
        rc['pumpPower'] = pump_p
        rc['pumpOutput'] = pump
        q0['readoutFrequency'] = signal_f
        q0['readoutPower'] = -100*dBm
        q0['readoutLen'] = readoutLen
        alg[gates.Readout([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats=stats, dataFormat='iqRaw')
        # mag, ang = readout.iqToPolar(readout.parseDataFormat(data))
        mags, angs = readout.iqToPolar(np.squeeze(data))
        mag = np.mean(mags)
        ang = np.mean(angs)
        returnValue([mag, ang])

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data

def noiseTemperature(Sample, pumpFrequency=st.r[11.0:12.0:0.005, GHz], pumpPower=st.r[-10:10:0.5, dBm],
                    fluxbias=st.r[-1.0:1.0:0.05, V], freqScan=st.r[5:6:0.01, GHz], readoutLen=2*us,
                    readoutPower=-20*dBm, demodFreq=10*MHz, stats=600, name="noise temperature",
                    plot=True, save=True, noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure=0)

    axes = [(pumpFrequency, "pump freq"), (pumpPower, "pump power"),
            (fluxbias, "flux bias"), (freqScan, 'signal freq')]
    deps_on = [("Mag", "signal level", ""), ("Mag", "noise level", "")]
    deps_off = [("Mag", "signal level", ""), ("Mag", "noise level", "")]

    kw = {'stats': stats, "demodFreq": demodFreq,
          "readoutLen": readoutLen, "readoutPower": readoutPower}

    name += " Sweep"
    if np.iterable(pumpFrequency):
        name += " pumpFreq"
    if np.iterable(pumpPower):
        name += " pumpPower"
    if np.iterable(fluxbias):
        name += " flux"
    if np.iterable(freqScan):
        name += " signal"

    dataset_on = sweeps.prepDataset(sample, name+" pumpOn", axes, deps_on, measure=0, kw=kw)
    dataset_off = sweeps.prepDataset(sample,name+" pumpOff", axes, deps_off, measure=0, kw=kw)

    def func(server, pump_f, pump_p, fb, signal_f, pump):
        reqs = list()
        for rp in [readoutPower, -100*dBm]: # -100*dBm means very low power
            alg = gc.Algorithm(devs)
            q0 = alg.q0
            rc = alg.readout_devs[0]
            rc['biasOperate'] = fb
            rc['carrierFrequency'] = signal_f - demodFreq
            rc['pumpFrequency'] = pump_f
            rc['pumpPower'] = pump_p
            rc['pumpOutput'] = pump
            q0['readoutFrequency'] = signal_f
            q0['readoutPower'] = rp
            q0['readoutLen'] = readoutLen
            alg[gates.Readout([q0])]
            alg.compile()
            reqs.append(runQubits(server, alg.agents, stats=stats, dataFormat='iqRaw'))
        data = yield FutureList(reqs)
        mag_sig, ang_sig = readout.iqToPolar(readout.parseDataFormat(data[0]))
        mag_noise = np.mean(readout.iqToPolar(data[1])[0])
        # mag_noise, ang_noise = readout.iqToPolar(readout.parseDataFormat(data[1]))
        returnValue([mag_sig, mag_noise])

    data_on = sweeps.grid(func, axes+[(True, "pumpOutput")], dataset=dataset_on, save=save, noisy=noisy)
    data_off = sweeps.grid(func, axes+[(False, "pumpOutput")], dataset=dataset_off, save=save, noisy=noisy)

    if plot:
        rc = devs[0]['readoutDevice']
        gain = np.array(data_on[:,1]/data_off[:,1])**2
        Gi = 10**(rc['insertion loss']['dB']/10.0)
        Ga = 10**(rc['cable loss']["dB"]/10.0)
        y = (data_on[:,2]/data_off[:,2])**2
        temp = (y-1)*rc['T_HEMT']['K']/(gain*Gi**2*Ga)
        plt.figure(figsize=(9,4))
        plt.subplot(121)
        plt.plot(data_on[:,0], 10*np.log10(gain))
        plt.subplot(122)
        plt.plot(data_on[:,0], temp, '.')
        plt.xlabel("Freq")
        plt.ylabel("noise temperature")
    return data_on, data_off

def saturation(Sample, freqScan=st.r[6:7:0.001, GHz], signalPower=st.r[-30:0:0.2, dBm], fluxbias=None,
               pumpFrequency=None, pumpPower=None, readoutLen=2*us, demodFreq=10*MHz, pump=True, stats=600,
               name='saturation power', save=True, noisy=True):
    sample, devs, _ = gc.loadQubits(Sample, measure=0)

    pa_dev = devs[0]['readoutDevice']
    fluxbias = fluxbias if fluxbias is not None else pa_dev['biasOperate']
    pumpFrequency = pumpFrequency if pumpFrequency is not None else pa_dev['pumpFrequency']
    pumpPower = pumpPower if pumpPower is not None else pa_dev['pumpPower']

    axes = [(freqScan, 'signal frequency'), (signalPower, 'signal power')]
    deps = [("Mag", "", "")]
    kw = {"stats": stats, "demodFreq": demodFreq, 'readoutLen': readoutLen, 'pumpOutput': pump,
          'pumpFrequency': pumpFrequency, 'pumpPower': pumpPower, 'fluxbias': fluxbias}

    name += "pumpOutput=%s" %(pump)
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=0, kw=kw)

    def func(server, signalF, signalP):
        alg = gc.Algorithm(devs)
        q = alg.q0
        jpa = q['readoutDevice']
        jpa['pumpFrequency'] = pumpFrequency
        jpa['pumpPower'] = pumpPower
        jpa['biasOperate'] = fluxbias
        jpa['carrierFrequency'] = signalF - demodFreq
        jpa['pumpOutput'] = pump
        q['readoutLen'] = readoutLen
        q['readoutPower'] = signalP
        q['readoutFrequency'] = signalF
        alg[gates.Readout([q])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        mag, ang = readout.iqToPolar(readout.parseDataFormat(data, 'iq'))
        returnValue([mag])

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data
