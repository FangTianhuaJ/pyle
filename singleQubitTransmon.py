import numpy as np
from scipy.optimize import leastsq, curve_fit
from scipy.special import erf, erfc
import matplotlib.pyplot as plt
from matplotlib import mlab
from scipy.integrate import quad

import time
import itertools

import labrad
import labrad.units as U

import pyle
from pyle import tomo, fidelity
import pyle.envelopes as env
from pyle.pipeline import returnValue, FutureList
from pyle.dataking import envelopehelpers as eh
from pyle.dataking import adjust
from pyle.util import sweeptools as st
from pyle.dataking import utilMultilevels as ml
from pyle.plotting import dstools
from pyle.fitting import fitting
from pyle.dataking import qubitpulsecal as qpc
from pyle.dataking import sweeps
from pyle.dataking import util
from pyle.dataking.fpgaseqTransmonV7 import runQubits
from pyle import gateCompiler as gc
from pyle import gates
from pyle.plotting import tomography
from pyle.interpol import interp1d_cubic
from pyle.analysis import readout
from pyle.dataking.benchmarking import randomizedBechmarking as rb
from pyle import optimize as popt
from pyle.dataking import zfuncs

# COLORS
BLUE   = "#348ABD"
RED    = "#E24A33"
PURPLE = "#988ED5"
YELLOW = "#FBC15E"
GREEN  = "#8EBA42"
PINK   = "#FFB5B8"
GRAY   = "#777777"
COLORS = [BLUE, RED, GREEN, YELLOW, PURPLE, PINK, GRAY]

V, mV, us, ns, GHz, MHz, dBm, rad = [U.Unit(s) for s in ('V', 'mV', 'us', 'ns', 'GHz', 'MHz', 'dBm', 'rad')]

# temp util function

def fluxVoltage(Sample, measure, voltage):
    cxn = Sample._cxn
    sample, qubits = util.loadQubits(Sample)
    q = qubits[measure]
    p = cxn.qubit_sequencer.packet()
    fluxChannel = dict(q['channels'])['flux']
    p.initialize([(q.__name__, [('flux', fluxChannel)])])
    p.mem_delay(10*us)
    p.mem_bias([((q.__name__, 'flux'), 'dac1', voltage)])
    p.build_sequence()
    p.run(30)
    p.send()

def fluxZero(Sample):
    cxn = Sample._cxn
    sample, qubits = util.loadQubits(Sample)
    p = cxn.qubit_sequencer.packet()
    channels = list()
    for q in qubits:
        tmpChannel = dict(q['channels'])
        if 'flux' in tmpChannel:
            channels.append((q.__name__, [('flux', tmpChannel['flux'])]))
    p.initialize(channels)
    p.mem_delay(10*us)
    # p.mem_start_timer()
    p.mem_bias([((q.__name__, 'flux'), 'dac1', 0*V) for q in qubits if 'flux' in dict(q['channels'])])
    # p.mem_stop_timer()
    p.build_sequence()
    p.run(30)
    p.send()

# VNA Measurement
def VNAScan(Sample, name, center, span, power, bandwidth, numpoints=1001,
            average=100, waittime=5, save=True, plot=False):
    cxn = Sample._cxn
    vna = cxn['VNA']
    vna.select_device(Sample["VNA ID"])
    start = center - span/2.0
    stop = center + span/2.0
    # VNA setup
    vna.s_parameters(["S21"])
    vna.frequency_range([start, stop])
    vna.power(power)
    vna.num_points(numpoints)
    vna.bandwidth(bandwidth)
    vna.averages(1) # average off
    time.sleep(0.5)
    vna.averages(average)
    # sleep to ensure the average is done
    time.sleep(waittime/2.0)
    vna.autoscale()
    time.sleep(waittime/2.0)
    vna.autoscale()
    freq, spara = vna.freq_sweep()
    freq = freq['MHz']
    spara = spara[0]
    mag = 20*np.log10(np.abs(spara))
    phase = np.angle(spara)
    savename = "VNAScan {name} {center} {span} {power}".format(name=name, center=center,
                                                               span=span, power=power)
    if save:
        dv = cxn.data_vault
        dv.cd(Sample._dir)
        dv.new(savename, [('freq', 'MHz')], [('Mag', 'S21', 'dB'), ("Phase", "S21", "rad")])
        dv.add_parameter(("VNA Frequency Range", (start, stop)))
        dv.add_parameter(("VNA Frequency Center", center))
        dv.add_parameter(("VNA Frequency Span", span))
        dv.add_parameter(("VNA Bandwidth", bandwidth))
        dv.add_parameter(("VNA Power", power))
        dv.add_parameter(("VNA Points Number", numpoints))
        dv.add_parameter(("VNA Measure", "S21"))
        dv.add_parameter(("VNA Average", average))
        data = np.vstack([freq, mag, phase]).T
        dv.add(data)
        print("data saved")
    if plot:
        plt.figure(figsize=(9, 4))
        plt.subplot(121)
        plt.plot(freq, mag)
        plt.xlabel("Freq [MHz]")
        plt.title("Magnetitude")
        plt.subplot(122)
        plt.plot(freq, phase)
        plt.xlabel("Freq [MHz]")
        plt.title("Phase")

    return freq*MHz, spara, mag, phase


def VNAPowerScan(Sample, name, power, center, span, bandwidth, numpoints=1001, average=100, waittime=10,
                 save=True, plot=False):
    savename = 'VNA Power Scan {name} {center} {span}'.format(name=name, center=center, span=span)
    cxn = Sample._cxn

    if save:
        dv = cxn.data_vault
        dv.cd(Sample._dir)
        dv.new(savename, [('VNA Power', 'dBm'), ('freq', 'MHz')],
               [('Mag', 'S21', 'dB'), ("Phase", "S21", "rad")])
        dv.add_parameter(("VNA Frequency Center", center))
        dv.add_parameter(("VNA Frequency Span", span))
        dv.add_parameter(("VNA Bandwidth", bandwidth))
        dv.add_parameter(("VNA Points Number", numpoints))
        dv.add_parameter(("VNA Measure", "S21"))
        dv.add_parameter(("VNA Average", average))

    all_data = list()
    for p in power:
        print("VNA Power: %s" %p)
        f, spara, mag, phase = VNAScan(Sample, name, center, span, p, bandwidth, numpoints,
                                       average, waittime, save=False, plot=False)
        p = float(p['dBm'])
        f = f['MHz']
        data = np.vstack([np.ones_like(f)*p, f, mag, phase]).T
        if save:
            dv.add(data)
        all_data.append(data)

    all_data = np.vstack(all_data)

    if plot:
        powers = np.unique(np.asarray(all_data[:,0]))
        freq = np.unique(all_data[:,1])
        mag = all_data[:,2].reshape(len(powers), -1)
        phase = all_data[:,3].reshape(len(powers), -1)
        plt.figure(figsize=(9,4))
        plt.subplot(121)
        plt.pcolormesh(freq, powers, mag)
        plt.xlabel("Freq [MHz]")
        plt.ylabel("VNA power [dBm]")
        plt.subplot(122)
        plt.pcolormesh(freq, powers, phase)
        plt.xlabel("Freq [MHz]")
        plt.ylabel("VNA power [dBm]")
        plt.tight_layout()

    return all_data

def VNAScanWithFlux(Sample, measure, name, bias, center, span, power, bandwidth, numpoints=1001,
                    average=100, waittime=5, save=True, plot=False):
    savename = 'VNA Scan with Flux {name} {center} {span}'.format(name=name, center=center, span=span)
    cxn = Sample._cxn
    if save:
        dv = cxn.data_vault
        dv.cd(Sample._dir)
        dv.new(savename, [('flux bias', str(bias.unit)), ('freq', 'MHz')],
               [("Mag", "S21", "dB"), ("Phase", "S21", "rad")])
        dv.add_parameter(("measure", measure))
        dv.add_parameter(("VNA Frequency Center", center))
        dv.add_parameter(("VNA Frequency Span", span))
        dv.add_parameter(("VNA Bandwidth", bandwidth))
        dv.add_parameter(("VNA Points Number", numpoints))
        dv.add_parameter(("VNA Measure", "S21"))
        dv.add_parameter(("VNA Average", average))

    all_data = list()
    try:
        for fb in bias:
            print("FluxBias: %s" %fb)
            util.setFluxVoltage(Sample, measure, fb)
            f, spara, mag, phase = VNAScan(Sample, name, center, span, power, bandwidth, numpoints,
                                           average, waittime, save=False, plot=False)
            util.setFluxVoltage(Sample, measure, 0 * V)
            f = f['MHz']
            fb = fb[fb.unit]
            data = np.vstack([np.ones_like(f)*fb, f, mag, phase]).T
            if save:
                dv.add(data)
            all_data.append(data)
    except:
        # fluxZero(Sample)
        util.setFluxVoltage(Sample, measure, 0*V)

    all_data = np.vstack(all_data)

    return all_data

def VNAScanFixedFlux(Sample, measure, name, bias, center, span, power, bandwidth, numpoints=1001,
                     average=100, waittime=5, save=True, plot=False):
    savename = "VNA Scan Fixed Flux {name} {center} {span}".format(name=name, center=center, span=span)
    cxn = Sample._cxn
    if save:
        dv = cxn.data_vault
        dv.cd(Sample._dir)
        dv.new(savename, [('freq', 'MHz')], [("Mag", "S21", "dB"), ("Phase", "S21", "rad")])
        dv.add_parameter(("measure", measure))
        dv.add_parameter(("biasOperate", bias))
        dv.add_parameter(("VNA Frequency Center", center))
        dv.add_parameter(("VNA Frequency Span", span))
        dv.add_parameter(("VNA Bandwidth", bandwidth))
        dv.add_parameter(("VNA Points Number", numpoints))
        dv.add_parameter(("VNA Measure", "S21"))
        dv.add_parameter(("VNA Average", average))
    try:
        print("FluxBias: %s" %bias)
        util.setFluxVoltage(Sample, measure, bias)
        f, spara, mag, phase = VNAScan(Sample, name, center, span, power, bandwidth, numpoints,
                                       average, waittime, save=False, plot=False)
        util.setFluxVoltage(Sample, measure, 0 * V)
        f = f['MHz']
        data = np.vstack([f, mag, phase]).T
        if save:
            dv.add(data)
    except:
        # fluxZero(Sample)
        util.setFluxVoltage(Sample, measure, 0*V)

# ADC measurement
def adc_bringup(plot=True):

    with labrad.connect() as cxn:
        fpga = cxn['GHz FPGAs']
        filter_bytes = (np.zeros(4024, dtype='<u1')+128).tostring()
        for adc_board in fpga.list_adcs():
            fpga.select_device(adc_board)
            retry = True
            retryTime = 0
            while retry and retryTime < 10:
                fpga.adc_recalibrate()
                fpga.adc_bringup()
                fpga.adc_filter_func(filter_bytes)
                I, Q = fpga.adc_run_average()
                if np.max(np.abs(I)) > 16000 or np.max(np.abs(Q)) > 16000:
                    retry = True
                    retryTime += 1
                    print("Retry bringup ADC")
                else:
                    retry = False
            if plot:
                plt.figure()
                plt.plot(I)
                plt.plot(Q)
                plt.title(adc_board)


def readoutTimeTrace(Sample, measure=0, sb_freq=10*MHz, readoutPower=None, readoutLen=None, name='Readout Time Trace',
                     piPulse=False, stats=600, average=100, plot=True, save=True, noisy=True):
    """
    @param sb_freq: readoutFrequency - carrierFrequency
    @param average: the average number. This is because the ADC V2 does not average,
                    and we need to do the average ourselves.
    @return:
    """
    import pyle.dataking.qubitsequencer
    sample, devs, qubits = gc.loadQubits(Sample, measure=measure)

    if readoutPower is None:
        readoutPower = devs[measure]['readoutPower']
    if readoutLen is None:
        readoutLen = devs[measure]['readoutLen']

    axes = [("time", 'ns')]
    deps = [("Amp", "I", ""), ("Amp", "Q", "")]
    kw = {'stats': stats, 'readoutPower': readoutPower, 'readoutLen': readoutLen, 'piPulse': piPulse}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    data = list()
    for i in range(average):
        if noisy:
            print("stats: %d/%d" %(i, average))
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        q0['adc mode'] = 'average'
        q0['readoutDevice']['carrierFrequency'] = q0['readoutFrequency'] - sb_freq
        q0['readoutDevice']['adcTimingLag'] = 0*ns
        q0['readoutPower'] = readoutPower
        q0['readoutLen'] = readoutLen
        if piPulse:
            alg[gates.PiPulse([q0])]
        alg[gates.Readout([q0])]
        alg.compile()
        with pyle.QubitSequencer() as sequencer:
            dat = runQubits(sequencer, alg.agents, stats=stats, dataFormat='iqRaw')
            dat = dat.wait()
            data.append(dat)
    data = np.squeeze(data)
    data = np.average(data, axis=0)
    # print('data.shape', data.shape)
    # print("timePoints.shape", timePoints.shape)

    timePoints = np.arange(data.shape[0])*2
    savedata = np.hstack([timePoints.reshape((-1, 1)), data])
    if save:
        with dataset:
            dataset.add(savedata)
    if plot:
        I, Q = data[:, 0], data[:, 1]
        plt.figure()
        plt.plot(timePoints, I, '.-', label='I')
        plt.plot(timePoints, Q, '.-', label='Q')
        plt.xlabel("Time [ns]")
        plt.ylabel("Trace")
        plt.legend()
        plt.tight_layout()
    return savedata

def readoutTimeTracev7(Sample, measure=0, sb_freq=10*MHz, readoutPower=None, readoutLen=None, name='Readout Time Trace',
                       piPulse=False, stats=600, average=5, plot=True, save=True, noisy=True):
    """
    @param sb_freq: readoutFrequency - carrierFrequency
    @param average: the average number.
    @return:
    """
    import pyle.dataking.qubitsequencer
    sample, devs, qubits = gc.loadQubits(Sample, measure=measure)

    if readoutPower is None:
        readoutPower = devs[measure]['readoutPower']
    if readoutLen is None:
        readoutLen = devs[measure]['readoutLen']

    axes = [("time", 'ns')]
    deps = [("Amp", "I", ""), ("Amp", "Q", "")]
    kw = {'stats': stats, 'readoutPower': readoutPower, 'readoutLen': readoutLen, 'piPulse': piPulse}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    data = list()
    for i in range(average):
        if noisy:
            print("average: %d/%d" %(i, average))
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        q0['adc mode'] = 'average'
        q0['readoutDevice']['carrierFrequency'] = q0['readoutFrequency'] - sb_freq
        q0['readoutDevice']['adcTimingLag'] = 0*ns
        q0['readoutPower'] = readoutPower
        q0['readoutLen'] = readoutLen
        if piPulse:
            alg[gates.PiPulse([q0])]
        alg[gates.Readout([q0])]
        alg.compile()
        with pyle.QubitSequencer() as sequencer:
            dat = runQubits(sequencer, alg.agents, stats=stats, dataFormat='iqRaw')
            dat = dat.wait()
            data.append(dat)
    data = np.squeeze(data)
    data = np.average(data, axis=0).T
    timePoints = np.arange(data.shape[0])*2
    print('data.shape', data.shape)
    print("timePoints.shape", timePoints.shape)

    savedata = np.hstack([timePoints.reshape((-1, 1)), data])
    if save:
        with dataset:
            dataset.add(savedata)
    if plot:
        I, Q = data[:, 0], data[:, 1]
        plt.figure()
        plt.plot(timePoints, I, '.-', label='I')
        plt.plot(timePoints, Q, '.-', label='Q')
        plt.xlabel("Time [ns]")
        plt.ylabel("Trace")
        plt.legend()
        plt.tight_layout()
    return savedata

def testDiffAmp(Sample, measure, adc_version, freqScan=st.r[6:7:0.01, GHz], readoutPower=-10*dBm, readoutLen=1*us,
                carrierFreq=6.5*GHz, name='test GHz Diff AMP', stats=600, save=True, noisy=True):
    """
    test the bandwidth of the GHz Diff Amp
    """
    sample, devs, qubits = gc.loadQubits(Sample, measure=measure)

    axes = [(freqScan, 'frequency')]
    deps = [('Mag', 'S21', ''), ('Phase', 'S21', 'rad')]

    kw = {'readoutPower': readoutPower, 'readoutLen': readoutLen,
          'carrierFrequency': carrierFreq, 'stats': stats, 'ADC Version': adc_version}
    name += ' V%d' %adc_version
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, currFreq):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        q0['readoutDevice']['carrierFrequency'] = carrierFreq
        q0['readoutFrequency'] = currFreq
        q0['readoutPower'] = readoutPower
        q0['readoutLen'] = readoutLen
        alg[gates.Readout([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats=stats, dataFormat='iqRaw')
        data = readout.parseDataFormat(data, 'iq')
        mag, ang = readout.iqToPolar(data)
        returnValue([mag, ang])

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data


def Sparameter(Sample, measure=0, freqScan=None, readoutPower=None, readoutLen=1*us,
               demodFreq=10*MHz, name='S Paramter', stats=600, rescale=True,
               plot=False, save=True, noisy=True):
    """
    @param freqScan: the readout frequency, can be single value or a list of frequencies
    @param readoutPower: the readout power, can be single value or a list of powers
    @param readoutLen: the length of readout pulse, the maximum value is 1us for ADC V7 and 16us for ADCV1
                       while 16us for ADC V1 is not a safe value, 15us is a safe value.
    @param demodFreq: the demodulate frequency of ADC (the difference between of readoutFrequency and carrierFrequency)
    @param rescale: the magnitude is rescaled with readoutPower or not
    @return:
    """
    sample, devs, qubits = gc.loadQubits(Sample, measure=measure)

    q = devs[measure]
    if freqScan is None:
        freqScan = st.r[q['readoutFrequency']["MHz"]-5.0: q['readoutFrequency']["MHz"]+5.0: 0.2, MHz]
    if readoutPower is None:
        readoutPower = q['readoutPower']

    axes = [(freqScan, 'frequency'), (readoutPower, 'readoutPower')]
    deps = [('Mag', 'S21', ''), ('Phase', 'S21', 'rad')]
    kw = {'stats': stats, 'readoutLen': readoutLen, 'demodFreq': demodFreq, 'rescale': rescale}

    if np.iterable(freqScan):
        name += " ScanFreq"
    if np.iterable(readoutPower):
        name += " ScanPower"
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, f, currPower):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        q0['readoutDevice']['carrierFrequency'] = f-demodFreq
        q0['readoutFrequency'] = f
        q0['readoutPower'] = currPower
        q0['readoutLen'] = readoutLen
        alg[gates.Readout([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats=stats, dataFormat='iqRaw')
        data = readout.parseDataFormat(data, 'iq')
        mag, ang = readout.iqToPolar(data)
        if rescale:
            amp = eh.power2amp(currPower)
            mag = mag/amp
        returnValue([mag, ang])

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    if plot:
        if np.iterable(readoutPower) and np.iterable(freqScan):
            plt.figure(figsize=(9.0, 4.3))
            freq = data[:, 0]
            power = data[:, 1]
            mag = data[:, 2].reshape(len(power), -1)
            phase = data[:, 3].reshape(len(power), -1)
            plt.subplot(121)
            plt.pcolormesh(freq, power, mag)
            plt.xlabel("Freq")
            plt.ylabel("Power")
            plt.subplot(122)
            plt.pcolormesh(freq, power, phase)
            plt.xlabel("Freq")
            plt.ylabel("Power")
            plt.tight_layout()
        else:
            # only scan freq or scan power
            freq = data[:, 0]
            mag = data[:, 1]
            phase = data[:, 2]
            plt.figure(figsize=(9.0, 4.3))
            plt.subplot(121)
            plt.plot(freq, mag, '.-')
            plt.ylabel("Mag")
            plt.subplot(122)
            plt.plot(freq, phase, '.-')
            plt.ylabel("Phase")
            plt.tight_layout()

    return data


def spectroscopy(Sample, measure=0, freqScan=st.r[4.5:7.0:0.01, GHz], zbias=0.0, uwaveAmp=None, sb_freq=0*MHz,
                 readoutFrequency=None, readoutPower=None, demodFreq=10*MHz, stats=600, zlonger=100*ns,
                 name='Spectroscopy', save=True, dataFormat="Amp", update=False, noisy=True):
    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, True)

    if uwaveAmp is None:
        uwaveAmp = devs[measure]['spectroscopyAmp']
    if not np.iterable(uwaveAmp):
        uwaveAmp = [uwaveAmp]
    if readoutFrequency is None:
        readoutFrequency = devs[measure]['readoutFrequency']
    if readoutPower is None:
        readoutPower = devs[measure]['readoutPower']
    if np.iterable(zbias):
        name += ' scanZ'
        # start from zero zbias
        zbias = np.array(zbias)
        idx = np.argsort(np.abs(zbias))
        zbias = zbias[idx]

    axes = [(zbias, 'zbias'), (freqScan, 'drive frequency'),
            (readoutFrequency, 'readoutFrequency'), (readoutPower, 'readoutPower')]
    if dataFormat == "Amp":
        deps = [[("Mag", "uwaveAmp=%g" %amp, ''), ("Phase", "uwaveAmp=%g" %amp, "rad")] for amp in uwaveAmp]
    else:
        deps = [[("I", "uwaveAmp=%g" %amp, ""), ("Q", "uwaveAmp=%g" %amp, "")] for amp in uwaveAmp]
    deps = sum(deps, [])

    kw = {'stats': stats, 'demodFreq': demodFreq, 'zlonger': zlonger, 'uwaveAmp': uwaveAmp,
          'sb_freq': sb_freq}
    if not np.iterable(readoutFrequency):
        kw['readoutFrequency'] = readoutFrequency
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)
    qubitNameCarrier = util.otherQubitNamesOnCarrier(devs[measure], devs)

    def func(server, z, freq, readoutF, readoutP):
        reqs = []
        for amp in uwaveAmp:
            alg = gc.Algorithm(devs)
            q0 = alg.q0
            q0['readoutFrequency'] = readoutF
            q0['readoutPower'] = readoutP
            q0['readoutDevice']['carrierFrequency'] = readoutF - demodFreq
            q0['spectroscopyAmp'] = amp
            q0['fc'] = freq - sb_freq
            for name in qubitNameCarrier:
                alg.agents_dict[name]['fc'] = freq - sb_freq
            alg[gates.Spectroscopy([q0], df=sb_freq, z=z, zlonger=zlonger)]
            # alg[gates.EmptyWait([q0], -overlap)]
            alg[gates.Readout([q0])]
            alg.compile()
            reqs.append(runQubits(server, alg.agents, stats, dataFormat='iqRaw'))

        data = yield FutureList(reqs)
        values = []
        for dat in data:
            dat = readout.parseDataFormat(dat, 'iq')
            if dataFormat == "Amp":
                mag, phase = readout.iqToPolar(dat)
                values.extend([mag, phase])
            else:
                values.extend(([dat[0], dat[1]]))
        returnValue(values)

    data = sweeps.grid(func, axes, save=save, dataset=dataset, noisy=noisy)

    if update:
        Q = Qubits[measure]
        adj = adjust.Adjust()
        f = data[:,0]
        adj.plot(data[:,0], data[:, 1], '.-')
        adj.x_param('f10', Q['f10'][GHz], np.min(f), np.max(f), 'r')
        result = adj.run()
        if result:
            f10 = result['f10']
            Q['f10'] = f10*GHz
    return data

def fluxSpectroscopy(Sample, measure=0, freqScan=st.r[4.5:7.0:0.01, GHz], fluxBias=st.r[-1.0:1.0:0.05, V],
                     uwaveAmp=None, readoutFrequency=None, readoutPower=None, demodFreq=10*MHz,
                     stats=600, name='flux Spectroscopy', dataFormat='Amp', save=True, noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure)

    if uwaveAmp is None:
        uwaveAmp = devs[measure]['spectroscopyAmp']
    if not np.iterable(uwaveAmp):
        uwaveAmp = [uwaveAmp]
    if readoutFrequency is None:
        readoutFrequency = devs[measure]['readoutFrequency']
    if readoutPower is None:
        readoutPower = devs[measure]['readoutPower']

    axes = [(fluxBias, 'flux bias'), (freqScan, 'drive frequency'),
            (readoutFrequency, 'readoutFrequency'), (readoutPower, 'readoutPower')]
    if dataFormat == 'Amp':
        deps = [[("Mag", "uwaveAmp=%g" %amp, ''), ("Phase", "S21", "rad")] for amp in uwaveAmp]
    else:
        deps = [[("I", "uwaveAmp=%g" % amp, ""), ("Q", "uwaveAmp=%g" % amp, "")] for amp in uwaveAmp]
    deps = sum(deps, [])

    kw = {'stats': stats, 'demodFreq': demodFreq, }
    if not np.iterable(readoutFrequency):
        kw['readoutFrequency'] = readoutFrequency
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)
    qubitNameCarrier = util.otherQubitNamesOnCarrier(devs[measure], devs)

    def func(server, fb, freq, readoutF, readoutP):
        reqs = []
        for amp in uwaveAmp:
            alg = gc.Algorithm(devs)
            q0 = alg.q0
            q0['biasOperate'] = fb
            q0['readoutFrequency'] = readoutF
            q0['readoutPower'] = readoutP
            q0['readoutDevice']['carrierFrequency'] = readoutF - demodFreq
            q0['spectroscopyAmp'] = amp
            q0['fc'] = freq
            for qname in qubitNameCarrier:
                alg.agents_dict[qname]['fc'] = freq
            alg[gates.Spectroscopy([q0], df=0, zlonger=100*ns)]
            alg[gates.Readout([q0])]
            alg.compile()
            reqs.append(runQubits(server, alg.agents, stats, dataFormat='iqRaw'))

        data = yield FutureList(reqs)
        values = []
        for dat in data:
            dat = readout.parseDataFormat(dat, 'iq')
            if dataFormat == 'Amp':
                mag, phase = readout.iqToPolar(dat)
                values.extend([mag, phase])
            else:
                values.extend([dat[..., 0], dat[..., 1]])
        returnValue(values)

    data = sweeps.grid(func, axes, save=save, dataset=dataset, noisy=noisy)

    return data

def SparameterZ(Sample, measure=0, freqScan=None, readoutPower=None, zbias=0.0, readoutLen=None, zlonger=20*ns,
                demodFreq=10*MHz, name='S parameter with Z', stats=600, rescale=False, save=True, noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure=measure)

    if readoutLen is None:
        readoutLen = devs[measure]['readoutLen']
    if freqScan is None:
        f = devs[measure]['readoutFrequency']["MHz"]
        freqScan = st.r[f-5.0:f+5.0:0.1, MHz]
    if readoutPower is None:
        readoutPower = devs[measure]['readoutPower']

    axes = [(freqScan, "frequency"), (readoutPower, "readoutPower"), (zbias, "z bias")]
    deps = [("Mag", "S21", ""), ("Phase", "S21", "rad")]
    kw = {"stats": stats, "readoutLen": readoutLen, "demodFreq": demodFreq, "rescale": rescale, "zlonger": zlonger}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    class TestReadout(gates.Gate):
        def __init__(self, agents, amp, zlonger=20*ns):
            self.amp = amp
            self.zlonger = zlonger
            super(TestReadout, self).__init__(agents)

        def updateAgents(self):
            ag = self.agents[0]
            amp = self.amp
            zlonger = self.zlonger
            t = ag['_t']
            ag['rr'] += eh.readoutPulse(ag, t+zlonger)
            readoutDemodLen = ag['readoutLen']+2*ag['readoutWidth']
            ag['z'] += env.rect(t, readoutDemodLen+2*zlonger, amp)
            ag['readoutDemodLen'] = readoutDemodLen
            ag['adcReadoutWindows']['DefaultReadout'] = (t+zlonger, t+readoutDemodLen+zlonger)
            ag['_t'] += readoutDemodLen + 2*zlonger

    def func(server, f, power, z):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        q0['readoutDevice']['carrierFrequency'] = f-demodFreq
        q0['readoutFrequency'] = f
        q0['readoutPower'] = power
        q0['readoutLen'] = readoutLen
        alg[TestReadout([q0], amp=z, zlonger=zlonger)]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats=stats, dataFormat='iqRaw')
        mag, ang = readout.iqToPolar(readout.parseDataFormat(data, 'iq'))
        if rescale:
            amp = eh.power2amp(power)
            mag = mag/amp
        returnValue([mag, ang])

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data


def SparameterFlux(Sample, measure=0, freqScan=None, readoutPower=None, bias=0.0, readoutLen=None,
                demodFreq=10*MHz, name='S parameter with flux', stats=600, rescale=False, save=True, noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure=measure)

    if readoutLen is None:
        readoutLen = devs[measure]['readoutLen']
    if freqScan is None:
        f = devs[measure]['readoutFrequency']["MHz"]
        freqScan = st.r[f-5.0:f+5.0:0.1, MHz]
    if readoutPower is None:
        readoutPower = devs[measure]['readoutPower']

    axes = [(bias, "flux bias"), (freqScan, "frequency"), (readoutPower, "readoutPower")]
    deps = [("Mag", "S21", ""), ("Phase", "S21", "rad")]
    kw = {"stats": stats, "readoutLen": readoutLen, "demodFreq": demodFreq, "rescale": rescale}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, flux, f, power):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        q0['biasOperate'] = flux
        q0['readoutDevice']['carrierFrequency'] = f-demodFreq
        q0['readoutFrequency'] = f
        q0['readoutPower'] = power
        q0['readoutLen'] = readoutLen
        alg[gates.Readout([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats=stats, dataFormat='iqRaw')
        mag, ang = readout.iqToPolar(readout.parseDataFormat(data))
        if rescale:
            amp = eh.power2amp(power)
            mag = mag/amp
        # mag = 20*np.log10(mag)
        returnValue([mag, ang])

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data


def SparameterState(Sample, measure=0, state=0, freqScan=None, readoutPower=None, readoutLen=None,
                    demodFreq=10*MHz, name='S parameter of State', stats=600, tBuf=5.0*ns,
                    rescale=False, save=True, update=True, noisy=True):
    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure=measure, write_access=True)

    if not np.iterable(state):
        state = [state]

    if readoutPower is None:
        readoutPower = devs[measure]['readoutPower']
    if readoutLen is None:
        readoutLen  = devs[measure]['readoutLen']
    if freqScan is None:
        fs = []
        for s in state:
            f = devs[measure]['resonatorFrequency%d' %s]["MHz"]
            fs.append(f)
        freqScan = st.r[min(fs)-5.0:max(fs)+5.0:0.1, MHz]

    axes = [(freqScan, "frequency"), (readoutPower, "readoutPower"),]
    deps = [[("Mag", "|%s>" %s, ""), ("Phase", "|%s>" %s, "rad")] for s in state]
    deps = sum(deps, [])
    kw = {"stats": stats, "readoutLen": readoutLen, "demodFreq": demodFreq, "rescale": rescale, 'tBuf': tBuf}

    name += "".join([" |%s> " %s for s in state])

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, f, power):
        reqs = list()
        for s in state:
            alg = gc.Algorithm(devs)
            q0 = alg.q0
            q0['readoutDevice']['carrierFrequency'] = f-demodFreq
            q0['readoutFrequency'] = f
            q0['readoutPower'] = power
            q0['readoutLen'] = readoutLen
            alg[gates.MoveToState([q0], 0, s)]
            alg[gates.Wait([q0], tBuf)]
            alg[gates.Readout([q0])]
            alg.compile()
            reqs.append(runQubits(server, alg.agents, stats, dataFormat='iqRaw'))
        data = yield FutureList(reqs)
        ans = list()
        for dat in data:
            mag, ang = readout.iqToPolar(readout.parseDataFormat(dat,'iq'))
            if rescale:
                amp = eh.power2amp(power)
                mag = mag/amp
            ans.extend([mag, ang])
        returnValue(ans)

    data = sweeps.grid(func, axes=axes, save=save, dataset=dataset, noisy=noisy)

    if update:
        Q = Qubits[measure]
        adj = adjust.Adjust()
        f = data[:, 0]
        for idx, currState in enumerate(state):
            adj.plot(f, data[:, 2*idx+1], '.-')
            key = 'resonatorFrequency%d' %currState
            if key in Q:
                init_val = Q[key][freqScan[0].unit]
            else:
                init_val = np.min(f)
            adj.x_param(key, init_val, np.min(f), np.max(f), )
        result = adj.run()
        if result:
            for currState in state:
                key = 'resonatorFrequency%d' %currState
                Q[key] = round(result[key], 5)*GHz
    return data


# functions with high power readout. Just averaging the mag or phase, do not detect probability

def testRabi(Sample, measure=0, rabiLen=st.r[0:400:5, ns], rabiAmp=0.5, delay=10*ns, stats=1200,
             name="Rabi test", save=True, dataFormat="Amp", noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure=measure)
    axes = [(rabiLen, 'Rabi Drive Length'), (rabiAmp, 'Rabi Drive Amp')]
    if dataFormat == "Amp":
        deps = [("Mag", "", ""), ("Phase", "", "rad")]
    else:
        deps = [("I", "", ""), ("Q", "", "")]
    # deps = [("Prob", "%s" %(devs[measure].__name__), "")]
    kw = {'stats': stats, 'delay': delay, 'dataFormat': dataFormat}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure, kw=kw)

    def func(server, currLen, currAmp):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        alg[gates.RabiDrive([q0], currAmp, currLen)]
        alg[gates.Wait([q0], delay)]
        alg[gates.Readout([q0])]
        alg.compile()
        data = yield  runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        if dataFormat == 'Amp':
            mag, ang = readout.iqToPolar(readout.parseDataFormat(data,'iq'))
        else:
            data = readout.parseDataFormat(data,'iq')
            mag = data[0]
            ang = data[1]
        returnValue([mag, ang])

    data = sweeps.grid(func, axes=axes, save=save, dataset=dataset, noisy=noisy)

    return data

def testRabiHigh(Sample, measure=0, amp=st.r[0:1.75:0.05], stats=1200, name='Rabi High Test',
                 alpha=None, dataFormat='Amp', save=True, noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure=measure)
    axes = [(amp, "Amp")]
    if dataFormat == 'Amp':
        deps = [("Mag", "", ""), ("Phase", "", "")]
    else:
        deps = [("I", "", ""), ("Q", "", "")]
    kw = {'stats': stats}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure, kw=kw)

    def func(server, currAmp):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        q0['piAmp'] = currAmp
        alg[gates.PiPulse([q0], alpha)]
        alg[gates.Readout([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        if dataFormat == 'Amp':
            mag, ang = readout.iqToPolar(readout.parseDataFormat(data,'iq'))
        else:
            data = readout.parseDataFormat(data,'iq')
            mag, ang = data[0], data[1]
        returnValue([mag, ang])

    data = sweeps.grid(func, axes=axes, save=save, dataset=dataset, noisy=noisy)

    return data

def testScurve(Sample, measure=0, readoutPower=st.r[-20:-10:0.1,dBm], readoutLen=None, readoutFrequency=None,
               stats=3000, name='test Scurve', save=True, noisy=True, plot=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure)
    if readoutFrequency is None:
        readoutFrequency = devs[measure]['readoutFrequency']
    if readoutLen is None:
        readoutLen = devs[measure]['readoutLen']
    axes = [(readoutPower, 'readoutPower'), (readoutLen, 'readoutLen')]
    deps = [("Mag", "noPi", ""), ("Phase", "noPi", ""), ("Mag", "Pi", ""), ("Phase", "Pi", "")]
    kw = {'stats': stats, "readoutFrequency": readoutFrequency}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure, kw=kw)

    def func(server, currPower, currLen):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        q0['readoutFrequency'] = readoutFrequency
        q0['readoutPower'] = currPower
        q0['readoutLen'] = currLen
        alg[gates.Wait([q0], waitTime=q0['piLen'])]
        alg[gates.Readout([q0])]
        alg.compile()
        data1 = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')

        alg = gc.Algorithm(devs)
        q0 = alg.q0
        q0['readoutFrequency'] = readoutFrequency
        q0['readoutPower'] = currPower
        q0['readoutLen'] = currLen
        alg[gates.PiPulse([q0])]
        alg[gates.Readout([q0])]
        alg.compile()
        data2 = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')

        mag1, ang1 = readout.iqToPolar(readout.parseDataFormat(data1,'iq'))
        mag2, ang2 = readout.iqToPolar(readout.parseDataFormat(data2,'iq'))
        returnValue([mag1, ang1, mag2, ang2])

    data = sweeps.grid(func, axes, save=save, dataset=dataset, noisy=noisy)

    if plot:
        plt.figure()
        plt.plot(data[:,0], data[:,1], label='noPi')
        plt.plot(data[:,0], data[:,3], label='Pi')
        plt.xlabel("readout Power")
        plt.ylabel("Mag")
        plt.tight_layout()
        plt.legend()

def testMeasurementError(Sample, measure=0, readoutPower=st.r[-20:-10:0.1, dBm], readoutLen=None, stats=1200,
                         delay=10*ns, name='test Measurement Error', save=True, noisy=True, plot=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure)

    if readoutLen is None:
        readoutLen = devs[measure]['readoutLen']

    axes = [(readoutPower, "readoutPower"), (readoutLen, "readoutLen")]
    deps = [("Measurement Error", "Mag", ""), ("Measurement Error", "Phase", "")]
    kw = {'stats': stats, "delay": delay}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure, kw=kw)

    def extractError(data0, data1):
        bins, bin_edges = np.histogram(data0, bins='sqrt', density=True)
        x0 = bin_edges[1:] - (bin_edges[1]-bin_edges[0])/2.0
        func0 = interp1d_cubic(x0, bins, bounds_error=False, fill_value=0.0)
        bins, bin_edges = np.histogram(data1, bins='sqrt', density=True)
        x1 = bin_edges[1:] - (bin_edges[1]-bin_edges[0])/2.0
        func1 = interp1d_cubic(x1, bins, bounds_error=False, fill_value=0.0)
        mfunc = lambda x: np.max([func1(x), func0(x)], axis=0)
        left = min([x0[0], x1[0]])
        right = max([x0[-1], x1[-1]])
        area = quad(mfunc, left, right)[0]
        error = 1.0 - area/2.0
        return error

    def func(server, currPower, currLen):
        reqs = []
        for piPluse in [False, True]:
            alg = gc.Algorithm(devs)
            q0 = alg.q0
            q0['readoutPower'] = currPower
            q0['readoutLen'] = currLen
            if piPluse:
                alg[gates.PiPulse([q0])]
            else:
                alg[gates.Wait([q0], waitTime=q0['piLen'])]
            alg[gates.Wait([q0], waitTime=delay)]
            alg[gates.Readout([q0])]
            alg.compile()
            reqs.append(runQubits(server, alg.agents, stats, dataFormat='iqRaw'))

        data = yield FutureList(reqs)
        data = np.squeeze(data)

        mags, angs = readout.iqToPolar(data)
        # mags = np.array([np.sqrt(dat[:,0]**2+dat[:,1]**2) for dat in data])
        # angs = np.array([np.arctan2(dat[:,1], dat[:,0]) for dat in data])

        error = [extractError(mags[0], mags[1]), extractError(angs[0], angs[1])]
        returnValue(error)

    data = sweeps.grid(func, axes, save=save, dataset=dataset, noisy=noisy)

    if plot:
        plt.figure()
        plt.plot(data[:,0], data[:,1], label="Mag")
        plt.plot(data[:,0], data[:,2], label='Phase')
        plt.xlabel("Readout Power [dBm]")
        plt.ylabel("Measurement Error")
        plt.legend()
        plt.tight_layout()


def testT1(Sample, measure=0, delay=st.r[0:10:0.1, us], stats=1200, name='T1 test',
           save=True, noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure=measure)
    axes = [(delay, "delay")]
    deps = [("Mag", "", ""), ("Phase", "", "")]
    kw = {'stats': stats}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure, kw=kw)

    def func(server, currLen):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        alg[gates.PiPulse([q0])]
        alg[gates.EmptyWait([q0], currLen)]
        alg[gates.Readout([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        mag, ang = readout.iqToPolar(readout.parseDataFormat(data,'iq'))
        returnValue([mag, ang])

    data = sweeps.grid(func, axes=axes, save=save, dataset=dataset, noisy=noisy)

    return data

def testRamsey(Sample, measure=0, delay=st.r[0:2000:4, ns], stats=1200, name='Ramsey Test', fringeFreq=50*MHz,
               tomo=True, save=True, noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure)
    axes = [(delay, "delay")]

    if tomo:
        tomoKeys = ["+X", "+Y", "-X", "-Y"]
        tomoPhases = {"+X": 0.0, "+Y": 0.25, "-X": -0.5, "-Y": -0.25}
    else:
        tomoKeys = ['+X']
        tomoPhases = {"+X": 0.0}

    deps = []
    for key in tomoKeys:
        deps.extend([("Mag", key, ""), ("Phase", key, "")])
    kw = {"stats": stats, 'fringeFrequency': fringeFreq, 'tomo': tomo}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure, kw=kw)

    def func(server, currDelay):
        reqs = []
        for key in tomoKeys:
            phase0 = 2*np.pi*tomoPhases[key]
            alg = gc.Algorithm(devs)
            q0 = alg.q0
            alg[gates.PiHalfPulse([q0])]
            alg[gates.Wait([q0], currDelay)]
            phase = float(fringeFreq*currDelay)*2*np.pi
            alg[gates.PiHalfPulse([q0], phase=phase0+phase)]
            alg[gates.Readout([q0])]
            alg.compile()
            reqs.append(runQubits(server, alg.agents, stats, dataFormat='iqRaw'))
        data = yield FutureList(reqs)
        ans = []
        for dat in data:
            mag, ang = readout.iqToPolar(readout.parseDataFormat(dat,'iq'))
            ans.extend([mag, ang])
        returnValue(ans)

    data = sweeps.grid(func, axes=axes, save=save, dataset=dataset, noisy=noisy)

    return data


def testSpinEcho(Sample, measure=0, delay=st.r[0:400:1, ns], stats=6000, name='SpinEcho Test', fringeFreq=50*MHz,
                 tomo=True, save=True, noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure)

    axes = [(delay, "delay")]

    if tomo:
        tomoKeys = ["+X", "+Y", "-X", "-Y"]
        tomoPhases = {"+X": 0.0, "+Y": 0.25, "-X": -0.5, "-Y": -0.25}
    else:
        tomoKeys = ['+X']
        tomoPhases = {"+X": 0.0}

    deps = []
    for key in tomoKeys:
        deps.extend([("Mag", key, ""), ("Phase", key, "")])
    kw = {"stats": stats, 'fringeFreq': fringeFreq, 'tomo': tomo}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure, kw=kw)

    def func(server, currDelay):
        reqs = []
        for key in tomoKeys:
            phase0 = 2*np.pi*tomoPhases[key]
            alg = gc.Algorithm(devs)
            q0 = alg.q0
            alg[gates.PiHalfPulse([q0])]
            alg[gates.Wait([q0], currDelay/2.0)]
            phase = float(fringeFreq*currDelay)*np.pi + float(q0['piLen']*fringeFreq)*np.pi
            alg[gates.PiPulse([q0], phase=phase)]
            alg[gates.Wait([q0], currDelay/2.0)]
            alg[gates.PiHalfPulse([q0], phase=phase0)]
            alg[gates.Readout([q0])]
            alg.compile()
            reqs.append(runQubits(server, alg.agents, stats, dataFormat='iqRaw'))
        data = yield FutureList(reqs)
        ans = []
        for dat in data:
            mag, ang = readout.iqToPolar(readout.parseDataFormat(dat,'iq'))
            ans.extend([mag, ang])
        returnValue(ans)

    data = sweeps.grid(func, axes=axes, save=save, dataset=dataset, noisy=noisy)

    return data

def testSwapSpectroscopy(Sample, measure=0, swapLen=st.r[0:500:1, ns], swapAmp=st.r[-0.5:0.5:0.05], stats=600,
                         name='Swap Spectroscopy Test', tBuf=5*ns, save=True, noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure)
    axes = [(swapAmp, 'swap Amp'), (swapLen, 'swapLen')]
    deps = [("Mag", "", ""), ("Phase", "", "")]
    kw = {"stats": stats, 'tBuf': tBuf}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure, kw=kw)

    def func(server, currAmp, currLen):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        alg[gates.PiPulse([q0])]
        alg[gates.Wait([q0], waitTime=tBuf)]
        alg[gates.Detune([q0], currLen, currAmp)]
        alg[gates.Wait([q0], waitTime=tBuf)]
        alg[gates.Readout([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        mag, ang = readout.iqToPolar(readout.parseDataFormat(data, 'iq'))
        returnValue([mag, ang])

    data = sweeps.grid(func, axes=axes, save=save, dataset=dataset, noisy=noisy)

    return data

def testDetuningT1(Sample, measure=0, delay=st.r[0:5000:50, ns], amp=st.r[-0.5:0.5:0.05], stats=600,
                   name='test Detuning T1', tBuf=5*ns, save=True, noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure)
    axes = [(amp, 'z amplitude'), (delay, 'delay')]
    deps = [("Mag", "", ""), ("Phase", "", "")]
    kw = {"stats": stats, 'tBuf': tBuf}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure, kw=kw)

    def func(server, currAmp, currLen):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        alg[gates.PiPulse([q0])]
        alg[gates.Wait([q0], waitTime=tBuf)]
        alg[gates.Detune([q0], currLen, currAmp)]
        alg[gates.Wait([q0], waitTime=tBuf)]
        alg[gates.Readout([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        mag, ang = readout.iqToPolar(readout.parseDataFormat(data, 'iq'))
        returnValue([mag, ang])

    data = sweeps.grid(func, axes=axes, save=save, dataset=dataset, noisy=noisy)

    return data


def testDetuneDephasing(Sample, measure=0, detuneAmp=st.r[-0.5:0.5:0.05], delay=st.r[0:1000:5, ns], echo=False,
                        riseTime=5*ns, fringeFreq=5*MHz, stats=600, name='test Detune Dephasing', save=True, noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure)

    axes = [(detuneAmp, "Z Pulse Amplitude"), (delay, "Delay")]
    deps = [("Mag", "+X", ""), ("Mag", "+Y", ""), ("Mag", "-X", ""),
            ("Mag", "-Y", ""), ("Envelope", "Mag", "")]
    tomoPhaseNames = ["+X", "+Y", "-X", "-Y"]
    tomoPhases = {"+X": 0.0, "+Y": 0.25, "-X": -0.5, "-Y": -0.25}
    kw = {"echo": echo, "stats": stats, "riseTime": riseTime, "fringeFreq": fringeFreq}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure, kw=kw)

    def func(server, currAmp, currDelay):

        reqs = []

        for tomoKey in tomoPhaseNames:
            alg = gc.Algorithm(devs)
            q0 = alg.q0
            alg[gates.PiHalfPulse([q0])]
            if echo:
                alg[gates.DetuneFlattop([q0], tlen=currDelay/2.0, amp=currAmp, w=riseTime)]
                alg[gates.PiPulse([q0])]
                alg[gates.DetuneFlattop([q0], tlen=currDelay/2.0, amp=currAmp, w=riseTime)]
            else:
                alg[gates.DetuneFlattop([q0], tlen=currDelay/2.0, amp=currAmp, w=riseTime)]
            phase = 2*np.pi*(fringeFreq['GHz']*currDelay['ns'] + tomoPhases[tomoKey])
            alg[gates.PiHalfPulse([q0], phase = phase)]
            alg[gates.Measure([q0])]
            alg.compile()

            reqs.append(runQubits(server, alg.agents, stats, dataFormat='iqRaw'))

        data = yield FutureList(reqs)
        results = []
        for dat in data:
            dat = readout.parseDataFormat(dat, 'iq')
            mag, ang = readout.iqToPolar(dat)
            results.append(mag)
        envelope = np.sqrt((results[0]-results[2])**2 + (results[1]-results[3])**2)

        results.append(envelope)
        returnValue(results)

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data


def oldTestDelayZ(Sample, measure=0, delay=st.r[-30:30:0.5,ns], zpa=0.1, zpl=20*ns,
                  stats=3000, name='Old Test Delay Z', update=True, save=True, plot=False, noisy=True):
    sample, devs, qubits, Qubit = gc.loadQubits(Sample, measure, True)

    axes = [(delay, 'Time')]
    deps = [("Mag", "", ""), ("Phase", "", "rad")]
    kw = {'stats': stats, "zpa": zpa, "zpl": zpl}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)
    maxShift = max(delay)

    def func(server, tshift):
        alg = gc.Algorithm(agents=devs)
        q0 = alg.q0
        alg[gates.TestDelayZ([q0], tshift, zpa, zpl)]
        alg[gates.Wait([q0], zpl/2.0+maxShift)] # fixed the start time of readout
        alg[gates.Readout([q0])]
        alg.compile()

        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        mag, ang = readout.iqToPolar((readout.parseDataFormat(data, 'iq')))
        returnValue([mag, ang])

    result = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return result
    zpl = zpl[ns]
    translength = 0.4*devs[measure]['piLen'][ns]
    def fitfunc(x, p):
        return (p[1] +
                p[2] * 0.5*erfc((x - (p[0] - zpl/2.0)) / translength) +
                p[3] * 0.5*erf((x - (p[0] + zpl/2.0)) / translength))
    x=result[:,0]
    y=result[:,2] #P1
    xfound = getMaxGauss(x,y,fitToMax=False)
    if noisy:
        print xfound
    guess = [xfound, min(y), max(y), max(y)]
    fit, _ok = leastsq(lambda p: fitfunc(x, p) - y, guess)
    if noisy:
        print guess,fit
    if plot:
        plt.figure()
        plt.plot(x, y, '.')
        plt.plot(x, fitfunc(x, fit))
    if noisy:
        print 'uwave lag:', -fit[0]
    if update:
        print 'uwave lag corrected by %g ns' % -fit[0]
        Qubit['timingLagUwave'] -= fit[0]*ns

def oldTestDelayRR(Sample, measure=0, delay=st.r[-30:30:0.5,ns], stats=900, name='Old Test Delay RR',
                   save=True, update=True, plot=False, noisy=True):
    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure=measure, write_access=True)

    axes = [(delay, 'Time')]
    deps = [('Mag', '', ''), ('Phase', '', 'rad')]
    kw = {'stats': stats}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure, kw=kw)

    def func(server, currDelayRR):
        alg = gc.Algorithm(agents=devs)
        alg.q0['readoutDevice']['timinglagRRUwave'] = currDelayRR
        alg[gates.PiPulse([alg.q0])]
        alg[gates.Measure([alg.q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        mag, ang = readout.iqToPolar((readout.parseDataFormat(data, 'iq')))
        returnValue([mag, ang])

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data

def testRecoverFlux(Sample, measure=0, fluxbias=None, stats=900, name='test Recover Flux',
                    tBuf=5*ns, save=True, noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure)

    qubit = devs[measure]
    if fluxbias is None:
        fb = qubit['biasOperate']
        fluxbias = st.r[fb['V']-0.1:fb['V']+0.1:0.002, V]

    axes = [(fluxbias, 'Bias Voltage')]
    deps = [("Mag", "", ""), ("Phase", "", "rad")]
    kw = {"stats": stats, 'tBuf': tBuf}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, fb):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        q0['biasOperate'] = fb
        alg[gates.PiPulse([q0])]
        alg[gates.Wait([q0], tBuf)]
        alg[gates.Measure([q0])]
        alg.compile()

        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        mag, ang = readout.iqToPolar((readout.parseDataFormat(data, 'iq')))
        returnValue([mag, ang])

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data

def testACStarkShift(Sample, measure=0, freqShift=None, ampSquare=None, excitationLen=1*us, tBuf=20*ns,
                     name='test AC Stark Shift', stats=300, save=True, noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure=measure)

    q = devs[measure]
    if freqShift is None:
        freqShift = st.r[-0.15:0.06:0.005, GHz]
    if ampSquare is None:
        ampSquare = np.arange(0, 0.03, 0.001)
    eta = q['f21'] - q['f10']

    axes = [(ampSquare, 'dac amplitude square'), (freqShift, 'Frequecy Shift to f10')]
    deps = [("Mag", "", ""), ("Phase", "", "rad")]

    refPower = q['readoutPower']
    df = q['readoutFrequency'] - q['readoutDevice']['carrierFrequency']
    f10 = q['f10']
    kw = {'stats': stats, 'refPower': refPower, 'f10': f10, 'tBuf': tBuf}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, currAmpSquared, fShift):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        q0['f10'] = f10 + fShift
        q0['f21'] = f10 + eta + fShift
        alg[gates.ACStark([q0], ringUp=excitationLen, buffer=tBuf, amp=np.sqrt(currAmpSquared))]
        alg[gates.Measure([q0])]
        alg.compile()

        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        mag, ang = readout.iqToPolar((readout.parseDataFormat(data, 'iq')))
        returnValue([mag, ang])

    data = sweeps.grid(func, axes, dataset=dataset, save=save)

    return data


def lollipop(Sample, measure=0, state=0, readoutPower=None, readoutLen=None, readoutFrequency=None,
             stats=6000, delay=20*ns, name='lollipop', save=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure)
    axes = [(range(stats), 'stats')]
    stateStr = "|%s>" %(state)
    deps = [("I", stateStr, ""), ("Q", stateStr, ""), ("Mag", stateStr, ""), ("Phase", stateStr, "")]
    if readoutPower is None:
        readoutPower = devs[measure]['readoutPower']
    if readoutLen is None:
        readoutLen = devs[measure]['readoutLen']
    if readoutFrequency is None:
        readoutFrequency = devs[measure]['readoutFrequency']
    kw = {'stats': stats, 'readoutPower': readoutPower, 'delay': delay,
          'readoutLen': readoutLen, 'readoutFrequncy': readoutFrequency, "state": state}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure, kw=kw)

    with pyle.QubitSequencer() as server:
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        q0['readoutPower'] = readoutPower
        q0['readoutLen'] = readoutLen
        q0['readoutFrequency'] = readoutFrequency
        # alg[gates.Wait([q0], waitTime=q0['piLen'])]
        alg[gates.MoveToState([q0], initState=0, endState=state)]
        alg[gates.Wait([q0], waitTime=delay)]
        alg[gates.Readout([q0])]
        alg.compile()
        data = runQubits(server, alg.agents, stats, dataFormat='iqRaw').wait()
        data = np.squeeze(data)
        mag, ang = readout.iqToPolar(data)

    data = np.vstack([range(stats), data.T, mag, ang]).T
    if save:
        with dataset:
            dataset.add(data)

    return data

def readoutBlobs(Sample, measure, states=(0,1), readoutPower=None, readoutLen=None, readoutFrequency=None,
                 stats=6000, delay=20*ns, name='Readout Blobs', save=True, plot=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure)

    if readoutPower is None:
        readoutPower = devs[measure]['readoutPower']
    if readoutLen is None:
        readoutLen = devs[measure]['readoutLen']
    if readoutFrequency is None:
        readoutFrequency = devs[measure]['readoutFrequency']
    kw = {'stats': stats, 'readoutPower': readoutPower, 'delay': delay,
          'readoutLen': readoutLen, 'readoutFrequency': readoutFrequency, "states": states}

    name += "".join([" |%s>" %s for s in states])

    axes = [(range(stats), "Clicks")]
    deps = [[("I", "|%s>" % s, ""), ("Q", "|%s>" % s, ""),
             ("Mag", "|%s>" % s, ""), ("Phase", "|%s>" % s, "rad")]
            for s in states]
    deps = sum(deps, [])
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    all_data = [np.array(range(stats)).reshape(stats, 1)]
    for state in states:
        print("Measuring state |%s> ..." %state)
        data = lollipop(sample, measure=measure, state=state, readoutPower=readoutPower,
                        readoutLen=readoutLen, readoutFrequency=readoutFrequency,
                        stats=stats, delay=delay, save=False)
        all_data.append(data[:, 1:])
    all_data = np.hstack(all_data)

    if save:
        with dataset:
            dataset.add(all_data)
    if plot:
        colors = [BLUE, RED, GREEN, YELLOW, PURPLE, PINK, GRAY]
        # plot IQ scatter
        plt.figure(figsize=(13, 4))
        ax = plt.subplot(131)
        for idx, state in enumerate(states):
            ax.plot(all_data[:,1+idx*4], all_data[:,2+idx*4], '.', markersize=2,
                    label='|%s>' %state, color=colors[idx], alpha=0.5)
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        plt.legend()
        ax.set_title("readout power: %s" %readoutPower)
        ax.set_aspect('equal')
        # plot Mag and Phase
        ax = plt.subplot(132)
        for idx, state in enumerate(states):
            ax.hist(all_data[:,4*idx+3], bins='sqrt', alpha=0.75, label='|%s>' %state,
                    normed=False, color=colors[idx])
        ax.set_xlabel("Mag")
        ax.set_ylabel("Counts")
        plt.legend()
        ax = plt.subplot(133)
        for idx, state in enumerate(states):
            ax.hist(all_data[:,4*idx+4], bins='sqrt', alpha=0.75, label='|%s>' %state,
                    normed=False, color=colors[idx])
        ax.set_xlabel("Phase")
        ax.set_ylabel("Counts")
        plt.legend()
        plt.tight_layout()
    return all_data

def plotReadoutBlobs(dataset):
    parameters = dataset.parameters
    states = parameters['states']
    colors = [BLUE, RED, GREEN, YELLOW, PURPLE, PINK, GRAY]
    # plot IQ scatter
    plt.figure()
    ax = plt.subplot(111)
    for idx, state in enumerate(states):
        IQ = dataset[:, 1+4*idx:3+4*idx]
        center, _, std = readout.iqToReadoutCenter(IQ)
        ax.plot(IQ[:,0], IQ[:,1], '.', markersize=2, label="|%s>" %state,
                color=colors[idx], alpha=0.5)
        ax.plot([center.real], [center.imag], '*', color='k', zorder=15)
        cir = plt.Circle((center.real, center.imag), radius=std, zorder=10, fill=False,
                         fc='k', lw=1, ls='-')
        ax.add_patch(cir)

    ax.set_xlabel("I [a.u.]")
    ax.set_ylabel("Q [a.u.]")
    ax.set_title("readoutPower: %s, readoutLen: %s, readoutFrequency: %s"
                 %(parameters['readoutPower'], parameters['readoutLen'], parameters['readoutFrequency']))
    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()

def calculateReadoutCenters(Sample, measure=0, states=[0, 1], readoutFrequency=None, readoutPower=None,
                            readoutLen=None, stats=6000, delay=0*ns,
                            update=True, plot=True, save=True, noisy=True):
    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, True)
    q = qubits[measure]

    if readoutFrequency is None:
        readoutFrequency = q['readoutFrequency']
    if readoutPower is None:
        readoutPower = q['readoutPower']
    if readoutLen is None:
        readoutLen = q['readoutLen']

    axes = [(range(stats), "Clicks")]
    deps = [[("I", "|%s>" %s, ""), ("Q", "|%s>" %s, "")] for s in states]
    deps = sum(deps, [])
    IQLists = []

    kw = {"stats": stats, 'states': states, 'readoutFrequency': readoutFrequency,
          'readoutPower': readoutPower, 'readoutLen': readoutLen}
    dataset = sweeps.prepDataset(sample, 'calculate Readout Centers', axes, deps, measure=measure, kw=kw)
    with pyle.QubitSequencer() as server:
        for state in states:
            print("Measuring state |%s> " %state)
            alg = gc.Algorithm(devs)
            q0 = alg.q0
            q0['readoutFrequency'] = readoutFrequency
            q0['readoutPower'] = readoutPower
            q0['readoutLen'] = readoutLen
            alg[gates.MoveToState([q0], 0, state)]
            alg[gates.Wait([q0], delay)]
            alg[gates.Measure([q0])]
            alg.compile()
            data = runQubits(server, alg.agents, stats, dataFormat='iqRaw').wait()
            IQLists.append(np.squeeze(data))
    fids, probs, centers, stds = readout.iqToReadoutFidelity(IQLists, states, k_means=True)

    all_data = [np.array(range(stats)).reshape(-1, 1)]
    [all_data.append(np.squeeze(data)) for data in IQLists]
    all_data = np.hstack(all_data)
    all_data = np.array(all_data, dtype='float')

    if save:
        with dataset:
            dataset.add(all_data)

    if noisy:
        for state, fid in zip(states, fids):
            print("|%d> Fidelity: %s" %(state, fid))
        print("Average Fidelity: %s " %np.mean(fids))

    if update:
        Q = Qubits[measure]
        Q["readoutCenterStates"] = centers.keys()
        Q["readoutCenters"] = [round(v.real, 6)+1j*round(v.imag, 6) for v in centers.values()]

    if plot:
        fig = plt.figure(figsize=(6, 4.8))
        ax = fig.add_subplot(1,1,1, aspect='equal')
        for idx, state, color in zip(range(len(states)), states, COLORS):
            IQs = np.squeeze(IQLists[idx])
            center = centers[state]
            ax.plot(IQs[:,0], IQs[:,1], '.', markersize=2, color=color, alpha=0.5, label='|%s>' %state)
            ax.plot([center.real], [center.imag], '*', color='k', zorder=15)
            cir1 = plt.Circle((center.real, center.imag), radius=stds[state], zorder=10,
                             fill=False, fc='k', lw=2, ls='-')
            ax.add_patch(cir1)
            # cir = plt.Circle((center.real, center.imag), radius=stds[state]*2, zorder=5,
            #                   fill=False, fc='k', lw=2, ls='--')
            # ax.add_patch(cir3)
        plt.legend()
        plt.xlabel("I [a.u.]")
        plt.ylabel("Q [a.u.]")

    return fids, centers, stds

def rabihigh(Sample, measure, amps=None, state=1, numPulses=None, mode='piPulse', prob=True,
             name='Rabi High', stats=600, save=True, noisy=True, update=True):
    """
    Rabi High
    @param state: target state, default is 1
    @param numPulses: number of pulse (piPulse or piHalfPulse) used in the tuning up
    @param mode: default is 'piPulse', tune up piPulse,
                'piHalfPulse' is also supported, tune up piHalfPulse
    @param prob: whether the return data is probability or amp and phase
    """
    assert mode in ("piPulse", "piHalfPulse")
    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, True)

    name = name + " " + mode + " |%s>" %state

    if amps is None:
        amps = np.linspace(0, 1.5, 151)

    if prob:
        deps = readout.genProbDeps(qubits, measure)
    else:
        deps = [("Mag", "|%s>" %state, ""), ("Phase", "|%s>" %state, "rad")]

    if numPulses is None and mode=='piPulse':
        numPulses = 1
    elif numPulses is None and mode=='piHalfPulse':
        numPulses = 2
    elif numPulses == 1 and mode == 'piHalfPulse':
        raise Exception("Mode %s and numPulse %s is not compatiable")

    axes = [(amps, "%s Amplitude" %mode)]
    kw = {"mode": mode, 'stats': stats, 'numPulses': numPulses, 'prob': prob}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, amp):
        alg = gc.Algorithm(devs)
        q = alg.q0
        alg[gates.MoveToState([q], 0, state-1)]
        if mode == 'piPulse':
            ml.setMultiLevels(q, 'piAmp', amp, state)
            alg[gates.NPiPulses([q], numPulses, state=state)]
        elif mode == 'piHalfPulse':
            ml.setMultiLevels(q, 'piHalfAmp', amp, state)
            alg[gates.NPiHalfPulses([q], numPulses, state=state)]
        alg[gates.MoveToState([q], state-1, 0)] # move back to |0>
        alg[gates.Measure([q])]
        alg.compile()

        data = yield runQubits(server, alg.agents, stats=stats, dataFormat='iqRaw')
        if prob:
            probs = readout.iqToProbs(data, alg.qubits)
            probs = np.squeeze(probs)
            returnValue(probs)
        else:
            data = readout.parseDataFormat(data, 'iq')
            mag, phase = readout.iqToPolar(data)
            returnValue([mag, phase])

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    if update:
        Q = Qubits[measure]
        if mode == 'piPulse':
            keyName = 'piAmp'
        elif mode == 'piHalfPulse':
            keyName = 'piHalfAmp'
        keyName += str(state) if state > 1 else ""
        oldVal = devs[measure][keyName]
        if prob:
            adj = adjust.Adjust()
            adj.plot(amps, data[:,state], 'b.--')
            adj.x_param('Pulse Amplitude', oldVal, np.min(amps)-0.1, np.max(amps)+0.1)
            result = adj.run()
            if result:
                newVal = result['Pulse Amplitude']
                print("Old %s : %.6f" %(keyName, oldVal))
                print("New %s : %.6f" %(keyName, newVal))
                Q[keyName] = round(newVal, 6)
        else:
            adj1 = adjust.Adjust()
            adj1.plot(amps, data[:, 1], 'b.--')
            adj1.x_param("Pulse Amplitude", oldVal, np.min(amps)-0.1, np.max(amps)+0.1)
            result1 = adj1.run()
            adj2 = adjust.Adjust()
            adj2.plot(amps, data[:, 2], 'r.--')
            adj2.x_param("Pulse Amplitude", oldVal, np.min(amps)-0.1, np.max(amps)+0.1)
            result2 = adj2.run()
            if result1:
                newVal = result1['Pulse Amplitude']
                print("Old %s : %.6f" %(keyName, oldVal))
                print("New %s : %.6f" %(keyName, newVal))
                Q[keyName] = round(newVal, 6)
            if result2:
                newVal = result2['Pulse Amplitude']
                print("Old %s : %.6f" %(keyName, oldVal))
                print("New %s : %.6f" %(keyName, newVal))
                Q[keyName] = round(newVal, 6)

    return data

def rabihighZ(Sample, measure, amps=None, numPulses=None, mode='piPulseZ',
              name='Rabi High Z', stats=600, save=True, noisy=True, update=True):
    assert mode in ["piPulseZ", "piHalfPulseZ"]
    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, write_access=True)

    if amps is None:
        amps = np.linspace(0, 0.4, 201)

    if numPulses is None:
        if mode == 'piPulseZ':
            numPulses = 1
        elif mode == 'piHalfPulseZ':
            numPulses = 2

    name += " {}".format(mode)

    axes = [(amps, "%s Z Amplitude" %mode)]
    deps = readout.genProbDeps(qubits, measure, states=[0,1])
    kw = {'stats': stats, 'mode': mode, 'numPulses': numPulses}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, currAmp):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        alg[gates.PiHalfPulse([q0])]
        if mode == 'piPulseZ':
            q0['piAmpZ'] = currAmp
            alg[gates.NPiPulsesZ([q0], numPulses)]
        elif mode == 'piHalfPulseZ':
            q0['piHalfAmpZ'] = currAmp
            alg[gates.NPiHalfPulsesZ([q0], numPulses)]
        alg[gates.PiHalfPulse([q0], phase=np.pi)]
        alg[gates.Measure([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        probs = readout.iqToProbs(data, alg.qubits, states=[0,1])
        returnValue(np.squeeze(probs))

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    if update:
        Q = Qubits[measure]
        if mode == 'piPulseZ':
            keyName = 'piAmpZ'
        elif mode == 'piHalfPulseZ':
            keyName = 'piHalfAmpZ'
        oldVal = devs[measure][keyName]
        adj = adjust.Adjust()
        adj.plot(amps, data[:,1], 'b.--')
        adj.x_param('Pulse Amplitude', oldVal, np.min(amps)-0.1, np.max(amps)+0.1)
        result = adj.run()
        if result:
            newVal = result['Pulse Amplitude']
            print("Old %s : %.6f" %(keyName, oldVal))
            print("New %s : %.6f" %(keyName, newVal))
            Q[keyName] = round(newVal, 6)

def rabi(Sample, measure, rabiLen=st.r[10:200:2, ns], rabiAmp=st.r[0.05:1.0:0.05], prob=True,
         state=1, name='Rabi Drive', stats=600, save=True, noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure=measure)

    measureStates = range(state+1)
    axes = [(rabiLen, 'Rabi Pulse Length'), (rabiAmp, "Rabi Amplitude")]
    if prob:
        deps = readout.genProbDeps(devs, measure, states=measureStates)
    else:
        deps = [("Mag", "", ""), ("Phase", "", "")]
    kw = {'stats': stats, 'prob': prob, 'state': state}
    if state > 1:
        name += " for |%s>" %state

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, currLen, currAmp):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        alg[gates.MoveToState([q0], 0, state-1)]
        alg[gates.RabiDrive([q0], currAmp, currLen, state=state)]
        alg[gates.MoveToState([q0], state-1, 0)]
        alg[gates.Measure([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        if prob:
            probs = readout.iqToProbs(data, alg.qubits, states=measureStates)
            returnValue(np.squeeze(probs))
        else:
            mag, phase = readout.iqToPolar(readout.parseDataFormat(data, 'iq'))
            returnValue([mag, phase])

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data

def rabilong(Sample, measure, rabiLen=st.r[0:200:1, ns], rabiAmp=st.r[0.05:1.0:0.05],
             name='Rabi Long', stats=600, save=True, noisy=True):
    sample, devs, qubits =  gc.loadQubits(Sample, measure=measure)
    axes = [(rabiLen, 'Rabi Pulse Length'), (rabiAmp, 'Rabi Amplitude')]
    deps = readout.genProbDeps(devs, measure)
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure, kw=kw)

    def func(server, currLen, currAmp):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        alg[gates.RabiDrive([q0], currAmp, currLen)]
        alg[gates.Measure([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        probs = readout.iqToProbs(data, alg.qubits)
        returnValue(np.squeeze(probs))

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data

def T1(Sample, measure=0, delay=st.r[0:5000:50, ns], stats=1200, state=1,
       name='T1', save=True, update=True, noisy=True, plot=True):
    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure=measure, write_access=True)

    if state > 1:
        name += " for |%s>" %state

    axes = [(delay, "Time")]
    deps = readout.genProbDeps(qubits, measure, states=range(state+1))
    deps += [("Mag", "|%s>" %state, ""), ("phase", "|%s>" %state, "")]
    kw = {'stats': stats, "state": state}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure, kw=kw)

    def func(server, currLen):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        alg[gates.MoveToState([q0], 0, state)]
        alg[gates.EmptyWait([q0], currLen)]
        alg[gates.Measure([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        probs = readout.iqToProbs(data, alg.qubits, states=range(state+1))
        prob = np.squeeze(probs)
        mag, ang = readout.iqToPolar(readout.parseDataFormat(data,'iq'))
        returnValue(np.r_[prob, mag, ang])

    data = sweeps.grid(func, axes=axes, save=save, dataset=dataset, noisy=noisy)

    if plot or update:
        fitResult = fitting.t1(data[:, (0, state+1)], timeRange=(10*ns, delay[-1]))
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(data[:,0], data[:,state+1], '.')
        ax.plot(data[:,0], fitResult['fitFunc'](data[:,0],*fitResult['fitParams']), 'k')
        ax.grid()
        ax.set_title(data.name + " - T1: %s" % (fitResult['T1']))
        fig.show()
    if update:
        Q = Qubits[measure]
        if state > 1:
            key = "calT1 |%s>" %state
        else:
            key = "calT1"
        Q[key] = fitResult['T1']

    return data


def ramsey(Sample, measure=0, delay=st.r[0:1000:2, ns], state=1, stats=3000,
           name='Ramsey', fringeFreq=50*MHz, tomo=True, save=True, noisy=True,
           plot=True, update=True):
    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, True)
    if state > 1:
        name += " for |%s>" %state

    axes = [(delay, "Time")]

    if tomo:
        tomoKeys = ["+X", "+Y", "-X", "-Y"]
        tomoPhases = {"+X": 0.0, "+Y": 0.25, "-X": -0.5, "-Y": -0.25}
    else:
        tomoKeys = ['+X']
        tomoPhases = {"+X": 0.0}

    deps = [("Probability |%s>" %state, key, "") for key in tomoKeys]
    if tomo:
        deps.append(("Envelope", "|%s>" %state, ""))

    kw = {"stats": stats, 'fringeFrequency': fringeFreq, 'tomo': tomo, 'state': state}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure, kw=kw)

    def func(server, currDelay):
        reqs = []
        for key in tomoKeys:
            phase0 = 2*np.pi*tomoPhases[key]
            alg = gc.Algorithm(devs)
            q0 = alg.q0
            alg[gates.MoveToState([q0], 0, state-1)]
            alg[gates.PiHalfPulse([q0], state=state)]
            dualBlock = False
            if currDelay > 6000*ns:
                alg[gates.DualBlockWaitDetune(alg.qubits, alg.q0, tlen=currDelay)]
                dualBlock = True
            else:
                alg[gates.Wait([q0], currDelay)]
            phase = float(fringeFreq*currDelay)*2*np.pi
            alg[gates.PiHalfPulse([q0], phase=phase0+phase, state=state, dualBlock=dualBlock)]
            alg[gates.MoveToState([q0], state-1, 0, dualBlock=dualBlock)] # move back to |0>
            alg[gates.Measure([q0])]
            alg.compile()
            reqs.append(runQubits(server, alg.agents, stats, dataFormat='iqRaw'))
        data = yield FutureList(reqs)
        ans = []
        for dat in data:
            prob = readout.iqToProbs(dat, alg.qubits, states=range(state+1))
            prob = np.squeeze(prob)[-1]
            ans.append(prob)
        if tomo:
            envelope = np.sqrt((ans[0]-ans[2])**2  + (ans[1] - ans[3])**2)
            ans.append(envelope)
        returnValue(ans)

    data = sweeps.grid(func, axes=axes, save=save, dataset=dataset, noisy=noisy)

    if update or plot:
        if tomo:
            fitResult = fitting.ramseyTomo_noLog(data)
            T2 = fitResult['T2']
            if plot:
                t = data[:, 0]
                envScaled = fitResult['envScaled']
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(t, envScaled, '.')
                ax.plot(t, fitResult['fitFunc'](t, *fitResult['fitParams']), 'k')
                ax.grid()
                ax.set_title("Ramsey - %s - T2: %s" %(data.name, T2))
                fig.show()
        else:
            fitResult = fitting.ramseyExponential(data)
            T2 = fitResult['T2']
            T2 = 1./(1./T2 - 0.5/devs[measure]['calT1'])
            if plot:
                t = data[:, 0]
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(t, data[:, 1], '.')
                ax.plot(t, fitResult['fitFunc'](t, *fitResult['fitParams']), 'k')
                ax.grid()
                ax.set_title("Ramsey - %s - T2: %s" %(data.name, T2))
                fig.show()
        if update:
            Q = Qubits[measure]
            key = "calT2Ramsey"
            if state > 1:
                key += " |%s>" %state
            Q[key] = T2

    return data


def spinEcho(Sample, measure=0, delay=st.r[0:2000:5, ns], state=1, stats=3000,
             name='SpinEcho', fringeFreq=50*MHz, tomo=True, save=True,
             noisy=True, update=True, plot=True):
    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, True)

    if state > 1:
        name += " for |%s>" %state

    axes = [(delay, "delay")]

    if tomo:
        tomoKeys = ["+X", "+Y", "-X", "-Y"]
        tomoPhases = {"+X": 0.0, "+Y": 0.25, "-X": -0.5, "-Y": -0.25}
    else:
        tomoKeys = ['+X']
        tomoPhases = {"+X": 0.0}

    deps = [("Probability |%s>" %state, key, "") for key in tomoKeys]
    if tomo:
        deps.append(("Envelope", "|%s>" %state, ""))

    kw = {"stats": stats, 'fringeFreq': fringeFreq, 'tomo': tomo, 'state': state}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure, kw=kw)

    def func(server, currDelay):
        reqs = []
        for key in tomoKeys:
            phase0 = 2*np.pi*tomoPhases[key]
            alg = gc.Algorithm(devs)
            q0 = alg.q0
            alg[gates.MoveToState([q0], 0, state-1)]
            alg[gates.PiHalfPulse([q0], state=state)]
            alg[gates.Wait([q0], currDelay/2.0)]
            phase = float(fringeFreq*currDelay)*np.pi + float(q0['piLen']*fringeFreq)*np.pi
            alg[gates.PiPulse([q0], phase=phase, state=state)]
            dualBlock = False
            if currDelay/2.0 > 6000*ns:
                alg[gates.DualBlockWaitDetune([q0], q0, tlen=currDelay/2.0)]
                dualBlock = True
            else:
                alg[gates.Wait([q0], currDelay/2.0)]
            alg[gates.PiHalfPulse([q0], phase=phase0, state=state, dualBlock=dualBlock)]
            alg[gates.MoveToState([q0], state-1, 0, dualBlock=dualBlock)]
            alg[gates.Measure([q0])]
            alg.compile()
            reqs.append(runQubits(server, alg.agents, stats, dataFormat='iqRaw'))
        data = yield FutureList(reqs)
        ans = []
        for dat in data:
            probs = readout.iqToProbs(dat, alg.qubits, states=range(state+1))
            prob = np.squeeze(probs)[-1]
            ans.append(prob)
        if tomo:
            envelope = np.sqrt((ans[0]-ans[2])**2  + (ans[1] - ans[3])**2)
            ans.append(envelope)
        returnValue(ans)

    data = sweeps.grid(func, axes=axes, save=save, dataset=dataset, noisy=noisy)

    if update or plot:
        if tomo:
            fitResult = fitting.ramseyTomo_noLog(data)
            T2 = fitResult['T2']
            if plot:
                t = data[:, 0]
                envScaled = fitResult['envScaled']
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(t, envScaled, '.')
                ax.plot(t, fitResult['fitFunc'](t, *fitResult['fitParams']), 'k')
                ax.grid()
                ax.set_title("Ramsey - %s - T2: %s" %(data.name, T2))
                fig.show()
        else:
            fitResult = fitting.ramseyExponential(data)
            T2 = fitResult['T2']
            T2 = 1./(1./T2 - 0.5/devs[measure]['calT1'])
            if plot:
                t = data[:, 0]
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(t, data[:, 1], '.')
                ax.plot(t, fitResult['fitFunc'](t, *fitResult['fitParams']), 'k')
                ax.grid()
                ax.set_title("Ramsey - %s - T2: %s" %(data.name, T2))
                fig.show()
        if update:
            Q = Qubits[measure]
            key = 'calT2Echo'
            if state > 1:
                key += " |%s>" %state
            Q[key] = T2

    return data

def nDemod(Sample, measure=0, delay=20*ns, ndemod=2, piPulse=False, stats=3000,
           name='n Demod', save=True, plot=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure)

    axes = [(range(stats), 'clicks')]
    deps = [[("I", "%d" %(i), ""), ("Q", "%d"%(i), "")] for i in range(1, ndemod+1)]
    deps = sum(deps, [])
    kw = {'stats': stats, 'nDemod': ndemod, 'delay': delay, 'piPulse': piPulse}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    with pyle.QubitSequencer() as server:
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        for n in range(ndemod):
            if piPulse:
                alg[gates.PiPulse([q0])]
            alg[gates.Measure([q0], name=n)]
            alg[gates.Wait([q0], waitTime=delay)]
        alg.compile()
        if plot:
            gc.plotAlg(alg)
        data = runQubits(server, alg.agents, stats, dataFormat='iqRaw').wait()

    # data.shape ( nChannel, stats, nDemod, I/Q )
    data = data[0] # only 1 channel
    if save:
        save_data = np.hstack([np.array(range(stats))[:, np.newaxis], data.reshape(stats, -1)])
        save_data = np.asarray(save_data, dtype=float)
        with dataset:
            dataset.add(save_data)
    if plot:
        plt.figure()
        ax = plt.subplot(111, aspect='equal')
        for i in range(ndemod):
            ax.plot(data[:, i, 0], data[:, i, 1], '.', label='No.%d' %(i+1))
        ax.legend()
    return data


def readoutHerald(Sample, measure=0, delay=400*ns, stats=6000, ringdown=False,
                  name='readout Herald', save=False, plot=False, ):

    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, write_access=True)

    ### ZERO STATE

    alg = gc.Algorithm(agents=devs)
    alg[gates.Measure([alg.q0], name='Herald', ringdown=ringdown)]
    alg[gates.Wait([alg.q0], delay)]
    alg[gates.Measure([alg.q0], name='Measure')]
    alg.compile()
    with pyle.QubitSequencer() as server:
        rv = runQubits(server, alg.agents, stats=stats, dataFormat='iqRaw')
        r = rv.wait()
        r0 = np.array(r)
    # probs0 = readout.iqToProbs(r0, alg.qubits, herald=True)
    centersList = readout.genCentersList(qubits, states=[0, 1])
    zs = readout.iqToZ(r0)
    states = readout.zsToStates(zs, centersList)
    heraldedStates, allChannelIndices = readout.heraldStates(states, heraldState=0)
    probs0 = readout.statesToProbs(heraldedStates, centersList)
    zsFirst = zs[0, :, 0]
    zsSecond = zs[0, :, 1]
    zsHeralded0 = zs[0, allChannelIndices, 1]

    if plot:
        plt.figure()
        ax = plt.subplot(111, aspect='equal')
        ax.plot(zsFirst.real, zsFirst.imag, '.g', markersize=2,
                label='First Measurement (Prep 0)', alpha=0.5)
        ax.plot(zsSecond.real, zsSecond.imag, '.m', markersize=2,
                label='Second Measurement (Re Prep 0)', alpha=0.5)
        plt.legend()

    print "Preparing state |0> state"
    print "Avg pops for first measurement: %s" % np.mean(states[0,:,0])
    print "Avg pops for second measurement: %s" % np.mean(states[0,:,1])
    print "Heralding Success rate: %s" % (heraldedStates.shape[1]/float(stats))
    print "Heralded probs for second measurement: %s" % np.squeeze(probs0)
    print ""

    ### ONE STATE

    alg = gc.Algorithm(agents=devs)
    alg[gates.Readout([alg.q0], name='Herald', ringdown=ringdown)]
    alg[gates.Wait([alg.q0], delay)]
    alg[gates.PiPulse([alg.q0])]
    alg[gates.Readout([alg.q0], name='Measure')]
    alg.compile()
    with pyle.QubitSequencer() as server:
        rv = runQubits(server, alg.agents, stats=stats, dataFormat='iqRaw')
        r = rv.wait()
        r1 = np.array(r)
    # probs1 = readout.iqToProbs(r1, alg.qubits, herald=True)
    zs = readout.iqToZ(r1)
    states = readout.zsToStates(zs, centersList)
    heraldedStates, allChannelIndices = readout.heraldStates(states, heraldState=0)
    probs1 = readout.statesToProbs(heraldedStates, centersList)
    zsFirst = zs[0, :, 0]
    zsSecond = zs[0, :, 1]
    zsHeralded0Prepare1 = zs[0, allChannelIndices, 1]

    if plot:

        plt.figure()
        ax = plt.subplot(111, aspect='equal')
        ax.plot(zsFirst.real, zsFirst.imag, '.g', markersize=2,
                label='First Measurement (Prep 0)', alpha=0.5)
        ax.plot(zsSecond.real, zsSecond.imag, '.m', markersize=2,
                label='Second Measurement (Prep 1)', alpha=0.5)
        plt.legend()

        plt.figure()
        ax = plt.subplot(111, aspect='equal')
        ax.plot(zsHeralded0.real, zsHeralded0.imag, '.', markersize=2, color=BLUE,
                label='Herald 0', alpha=0.5)
        ax.plot(zsHeralded0Prepare1.real, zsHeralded0Prepare1.imag, '.', markersize=2, color=RED,
                label='Herald 0, Prepare 1', alpha=0.5)
        plt.legend()

    print "Preparing state |1> state"
    print "Avg pops for first measurement: %s" % np.mean(states[0, :, 0])
    print "Avg pops for second measurement: %s" % np.mean(states[0, :, 1])
    print "Heralding Success rate: %s" % (heraldedStates.shape[1] / float(stats))
    print "Heralded probs for second measurement: %s" % np.squeeze(probs1)
    print ""
    print "Readout Fidelity: %4f" % (np.mean((np.squeeze(probs0)[0], np.squeeze(probs1)[1])))

    if save:
        axes = [(range(stats), "Clicks")]
        deps = [("I Prep0", "Herald", ""), ("Q Prep0", "Herald", ""),
                ("I Prep0", "Measure", ""), ("Q Prep0", "Measure", ""),
                ("I Prep1", "Herald", ""), ("Q Prep1", "Herald", ""),
                ("I Prep1", "Measure", ""), ("Q Prep1", "Measure", ""),]
        kw = {'stats': stats, "delay": delay}
        dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)
        r0 = r0.reshape((-1, 4))
        r1 = r1.reshape((-1, 4))
        data = np.hstack((np.array(range(stats))[:, None], r0, r1))
        data = np.asarray(data, dtype=float)
        with dataset:
            dataset.add(data)

def readoutHerald2(Sample, measure=0, delay=400*ns, stats=6000, ringdown=False,
                   name='readout Herald 2', save=False, plot=False, ):

    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, write_access=True)

    ### ZERO STATE

    alg = gc.Algorithm(agents=devs)
    alg[gates.Measure([alg.q0], name='Herald', ringdown=ringdown)]
    alg[gates.Wait([alg.q0], delay)]
    alg[gates.Measure([alg.q0], name='Measure')]
    alg.compile()
    with pyle.QubitSequencer() as server:
        rv = runQubits(server, alg.agents, stats=stats, dataFormat='iqRaw')
        r = rv.wait()
        r0 = np.array(r)
    centersList = readout.genCentersList(qubits, states=[0, 1, 2])

    zs = readout.iqToZ(r0)
    states = readout.zsToStates(zs, centersList)
    heraldedStates, allChannelIndices = readout.heraldStates(states, heraldState=0)
    probs0 = readout.statesToProbs(heraldedStates, centersList)
    zsFirst = zs[0, :, 0]
    zsSecond = zs[0, :, 1]
    zsHeralded0 = zs[0, allChannelIndices, 1]

    if plot:
        plt.figure()
        ax = plt.subplot(111, aspect='equal')
        ax.plot(zsFirst.real, zsFirst.imag, '.g', markersize=2,
                label='First Measurement (Prep 0)', alpha=0.5)
        ax.plot(zsSecond.real, zsSecond.imag, '.m', markersize=2,
                label='Second Measurement (Re Prep 0)', alpha=0.5)
        plt.legend()

    print "Preparing state |0> state"
    print "Avg pops for first measurement: %s" % np.mean(states[0,:,0])
    print "Avg pops for second measurement: %s" % np.mean(states[0,:,1])
    print "Heralding Success rate: %s" % (heraldedStates.shape[1]/float(stats))
    print "Heralded probs for second measurement: %s" % np.squeeze(probs0)
    print ""

    ### ONE STATE

    alg = gc.Algorithm(agents=devs)
    alg[gates.Readout([alg.q0], name='Herald', ringdown=ringdown)]
    alg[gates.Wait([alg.q0], delay)]
    alg[gates.PiPulse([alg.q0])]
    alg[gates.Readout([alg.q0], name='Measure')]
    alg.compile()
    with pyle.QubitSequencer() as server:
        rv = runQubits(server, alg.agents, stats=stats, dataFormat='iqRaw')
        r = rv.wait()
        r1 = np.array(r)
    zs = readout.iqToZ(r1)
    states = readout.zsToStates(zs, centersList)
    heraldedStates, allChannelIndices = readout.heraldStates(states, heraldState=0)
    probs1 = readout.statesToProbs(heraldedStates, centersList)
    zsFirst = zs[0, :, 0]
    zsSecond = zs[0, :, 1]
    zsHeralded0Prepare1 = zs[0, allChannelIndices, 1]

    if plot:
        plt.figure()
        ax = plt.subplot(111, aspect='equal')
        ax.plot(zsFirst.real, zsFirst.imag, '.g', markersize=2,
                label='First Measurement (Prep 0)', alpha=0.5)
        ax.plot(zsSecond.real, zsSecond.imag, '.m', markersize=2,
                label='Second Measurement (Prep 1)', alpha=0.5)
        plt.legend()

    print "Preparing state |1> state"
    print "Avg pops for first measurement: %s" % np.mean(states[0,:,0])
    print "Avg pops for second measurement: %s" % np.mean(states[0,:,1])
    print "Heralding Success rate: %s" % (heraldedStates.shape[1]/float(stats))
    print "Heralded probs for second measurement: %s" % np.squeeze(probs1)
    print ""

    ### TWO STATE
    alg = gc.Algorithm(agents=devs)
    alg[gates.Readout([alg.q0], name='Herald', ringdown=ringdown)]
    alg[gates.Wait([alg.q0], delay)]
    alg[gates.PiPulse([alg.q0])]
    alg[gates.PiPulse([alg.q0], state=2)]
    alg[gates.Readout([alg.q0], name='Measure')]
    alg.compile()
    with pyle.QubitSequencer() as server:
        rv = runQubits(server, alg.agents, stats=stats, dataFormat='iqRaw')
        r = rv.wait()
        r2 = np.array(r)
    zs = readout.iqToZ(r2)
    states = readout.zsToStates(zs, centersList)
    heraldedStates, allChannelIndices = readout.heraldStates(states, heraldState=0)
    probs2 = readout.statesToProbs(heraldedStates, centersList)
    zsFirst = zs[0, :, 0]
    zsSecond = zs[0, :, 1]
    zsHeralded0Prepare2 = zs[0, allChannelIndices, 1]

    if plot:

        plt.figure()
        ax = plt.subplot(111, aspect='equal')
        ax.plot(zsFirst.real, zsFirst.imag, '.g', markersize=2,
                label='First Measurement (Prep 0)', alpha=0.5)
        ax.plot(zsSecond.real, zsSecond.imag, '.m', markersize=2,
                label='Second Measurement (Prep 2)', alpha=0.5)
        plt.legend()

        plt.figure()
        ax = plt.subplot(111, aspect='equal')
        ax.plot(zsHeralded0.real, zsHeralded0.imag, '.', markersize=2, color=BLUE,
                label='Herald 0', alpha=0.5)
        ax.plot(zsHeralded0Prepare1.real, zsHeralded0Prepare1.imag, '.', markersize=2, color=RED,
                label='Herald 0, Prepare 1', alpha=0.5)
        ax.plot(zsHeralded0Prepare2.real, zsHeralded0Prepare2.imag, '.', markersize=2, color=GREEN,
                label='Herald 0, Prepare 2', alpha=0.5)
        plt.legend()

    print "Preparing state |2> state"
    print "Avg pops for first measurement: %s" % np.mean(states[0, :, 0])
    print "Avg pops for second measurement: %s" % np.mean(states[0, :, 1])
    print "Heralding Success rate: %s" % (heraldedStates.shape[1] / float(stats))
    print "Heralded probs for second measurement: %s" % np.squeeze(probs2)
    print ""

    print "Readout Fidelity: %4f" % (np.mean((np.squeeze(probs0)[0], np.squeeze(probs1)[1],
                                              np.squeeze(probs2)[2])))

    if save:
        axes = [(range(stats), "Clicks")]
        deps = [("I Prep0", "Herald", ""), ("Q Prep0", "Herald", ""),
                ("I Prep0", "Measure", ""), ("Q Prep0", "Measure", ""),
                ("I Prep1", "Herald", ""), ("Q Prep1", "Herald", ""),
                ("I Prep1", "Measure", ""), ("Q Prep1", "Measure", ""),
                ("I Prep2", "Herald", ""), ("Q Prep2", "Herald", ""),
                ("I Prep2", "Measure", ""), ("Q Prep2", "Measure", ""),]
        kw = {'stats': stats, "delay": delay}
        dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)
        r0 = r0.reshape((-1, 4))
        r1 = r1.reshape((-1, 4))
        r2 = r2.reshape((-1, 4))
        data = np.hstack((np.array(range(stats))[:, None], r0, r1, r2))
        data = np.asarray(data, dtype=float)
        with dataset:
            dataset.add(data)


def heatingRate(Sample, measure=0, delay=st.r[20:2000:20, ns],
                stats=3000, name='Heating Rate', save=False):
    sample, devs, qubits = gc.loadQubits(Sample, measure)

    axes = [(delay, 'Delay')]
    deps = readout.genProbDeps(qubits, measure)
    kw = {'stats': stats}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, currDelay):
        alg = gc.Algorithm(devs)
        q = alg.q0
        alg[gates.Measure([q], name='herald')]
        alg[gates.Wait([q], waitTime=currDelay)]
        alg[gates.Measure([q])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        probs = readout.iqToProbs(data, alg.qubits, herald=True)
        returnValue(np.squeeze(probs))

    data = sweeps.grid(func, axes, dataset=dataset, save=save)

    return data

def readoutStability(Sample, measure=0, repeat=100, sleepTime=10*U.s, piPulse=False,
                     stats=3000, name='Readout Stability', save=True, delay=0.5*us):
    sample, devs, qubits = gc.loadQubits(Sample, measure)

    axes = [("Time", "s")]
    deps = readout.genProbDeps(qubits, measure) + [("I Center", "", ""), ("Q Center", "", ""), ("STD", "", "")]
    kw = {'sleepTime': sleepTime, "stats": stats, 'piPulse': piPulse, 'delay':delay}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def sweep():
        for i in range(repeat):
            yield i

    tic = time.time() # the datetime the dataking begins
    def func(server, t):
        alg = gc.Algorithm(devs)
        if piPulse:
            alg[gates.PiPulse([alg.q0])]
            alg[gates.Wait([alg.q0],delay)]
        alg[gates.Measure([alg.q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        toc = time.time() - tic
        time.sleep(sleepTime['s'])
        probs = np.squeeze(readout.iqToProbs(data, alg.qubits))
        center, _, std = readout.iqToReadoutCenter(np.squeeze(data))
        returnValue(np.hstack([toc, probs, center.real, center.imag, std]))

    data = sweeps.run(func, sweep(), save=save, dataset=dataset)

    return data


def readoutHeraldDelay(Sample, measure=0, delay=None, ringdown=False, stats=6000,
                       name='Readout Herald Delay', save=True, update=True, noisy=True):
    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, write_access=True)

    axes = [(delay, 'Readoud Herald Delay')]
    deps = readout.genProbDeps(qubits, measure)
    kw = {'stats': stats, 'ringdown': ringdown}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, currDelay):
        alg = gc.Algorithm(devs)
        q0 = alg.qubits[0]
        q0['heraldDelay'] = currDelay
        alg[gates.Herald([q0], ringdown=ringdown)]
        alg[gates.MoveToState([q0], 0, 1)]
        alg[gates.Measure([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        probs = readout.iqToProbs(data, alg.qubits, herald=True)
        returnValue(np.squeeze(probs))

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    if update:
        Q = Qubits[measure]
        q = qubits[measure]
        init_val = q.get('heraldDelay', 0*ns)
        adj = adjust.Adjust()
        adj.plot(data[:, 0], data[:, 1], '.-')
        adj.x_param('heraldDelay', init_val[delay[0].unit], np.min(data[:,0]), np.max(data[:,0]))
        result = adj.run()
        if result:
            val = result['heraldDelay']
            val = U.Value(val, delay[0].unit)['ns']
            val = round(val/4)*4*ns
            Q['heraldDelay'] = val
    return data


def pituner(Sample, measure=0, diff=0.4, iterations=1, numPulses=3, numpoints=101,
            state=1, stats=1200, mode='piPulse', save=False, update=True, noisy=True):
    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, True)
    q, Q = devs[measure], Qubits[measure]

    if mode is 'piPulse':
        keyName = 'piAmp'
    elif mode is 'piHalfPulse':
        keyName = 'piHalfAmp'
    else:
        raise Exception("unknown mode")
    amp = ml.getMultiLevels(q, keyName, state)
    ampstart = amp
    damp = diff / numPulses

    if mode is 'piHalfPulse':
        numPulses *= 2
        # we need to tune up piHalfPulses by using 2 at a time to equal a pi pulse

    for _ in range(iterations):
        # optimize amplitude
        data = rabihigh(sample, measure, amps=np.linspace((1 - damp) * amp, (1 + damp) * amp, numpoints),
                        numPulses=numPulses, stats=stats, prob=True, update=False,
                        noisy=noisy, save=save, state=state, mode=mode)
        amp, val = fitting.getMaxPoly(data[:,0], data[:,1], fit=True)
        amp = round(amp, 6)
        ml.setMultiLevels(qubits[measure], keyName, amp, state)
        if noisy: print('Amplitude: %g' % amp)

    # save updated values
    if noisy: print('Old Amplitude: %g' % ampstart)
    if update:
        if mode is 'piPulse':
            ml.setMultiLevels(Q, 'piAmp', amp, state)
        elif mode is 'piHalfPulse':
            ml.setMultiLevels(Q, 'piHalfAmp', amp, state)
    return amp

def freqtuner(Sample, measure=0, iterations=1, state=1, fringeFreq=20.0*MHz, delay=st.r[0:1000:5,ns],
              nfftpoints=4096, update=True, save=False, noisy=True, stats=1200):
    """Tune up qubit frequency via Ramsey fringes."""
    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, write_access=True)
    q, Q = devs[measure], Qubits[measure]
    fkey = ml.multiLevelKeyName('frequency', state)
    oldVal = q[fkey]

    for _ in range(iterations):
        data = ramsey(sample, measure, delay=delay, state=state, name='freq tuner',
                      fringeFreq=fringeFreq, save=save, noisy=noisy,
                      tomo=False, stats=stats, plot=False, update=False)
        fringe = fitting.maxFreq(data[:,:2], nfftpoints, plot=False)
        delta_freq = fringeFreq - fringe
        print 'Desired Fringe Frequency: %s' % fringeFreq
        print 'Actual Fringe Frequency: %s' % fringe
        print 'Qubit frequency adjusted by %s' % delta_freq
        qubits[measure][fkey] = oldVal - delta_freq

    if update:
        newFreq = q[fkey]-st.nearest(delta_freq['GHz'], 0.000001)*GHz
        ml.setMultiLevels(Q, 'frequency', newFreq, state)

def testDelayZ(Sample, measure=0, delay=st.r[-30:30:0.5,ns], zpa=0.1, zpl=20*ns, stats=3000,
               name='Test Delay Z', update=True, save=True, plot=True, noisy=True):
    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, True)

    axes = [(delay, 'Time')]
    deps = readout.genProbDeps(devs, measure, states=[0,1])
    kw = {'stats': stats, "zpa": zpa, "zpl": zpl}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)
    maxShift = max(delay)

    def func(server, tshift):
        alg = gc.Algorithm(agents=devs)
        q0 = alg.q0
        alg[gates.TestDelayZ([q0], tshift, zpa, zpl)]
        alg[gates.Wait([q0], zpl/2.0+maxShift)] # fixed the start time of readout
        alg[gates.Measure([q0])]
        alg.compile()

        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        probs = readout.iqToProbs(data, alg.qubits, states=[0,1])
        probs = np.squeeze(probs)
        returnValue(probs)

    result = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    zpl = zpl[ns]
    translength = 0.4*devs[measure]['piLen'][ns]

    def fitfunc(x, p):
        return (p[1] +
                p[2] * 0.5*erfc((x - (p[0] - zpl/2.0)) / translength) +
                p[3] * 0.5*erf((x - (p[0] + zpl/2.0)) / translength))

    x = result[:, 0]
    y = result[:, 2]  # P1
    xfound, _ = fitting.getMaxPoly(x, y, fit=False)
    if noisy:
        print xfound
    guess = [xfound, min(y), max(y), max(y)]
    fit, _ok = leastsq(lambda p: fitfunc(x, p) - y, guess)
    if noisy:
        print guess, fit
    if plot:
        plt.figure()
        plt.plot(x, y, '.')
        plt.plot(x, fitfunc(x, fit))
    if noisy:
        print 'uwave lag:', -fit[0]
    if update:
        print 'uwave lag corrected by %g ns' % -fit[0]
        Qubits[measure]['timingLagUwave'] -= fit[0]*ns

def testDelayRR(Sample, measure=0, delay=st.r[-30:30:0.5,ns], stats=900, name='Test Delay RR',
                   save=True, update=True, plot=True, noisy=True):
    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure=measure, write_access=True)

    axes = [(delay, 'Time')]
    deps = readout.genProbDeps(devs, measure, states=[0,1])
    kw = {'stats': stats}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure, kw=kw)

    def func(server, currDelayRR):
        alg = gc.Algorithm(agents=devs)
        alg.q0['readoutDevice']['timinglagRRUwave'] = currDelayRR
        alg[gates.PiPulse([alg.q0])]
        alg[gates.Measure([alg.q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        probs = readout.iqToProbs(data, alg.qubits, states=[0,1])
        probs = np.squeeze(probs)
        returnValue(probs)

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    if plot:
        plt.figure()
        plt.plot(data[:, 0], data[:, 2], '.-')
        plt.xlabel("Time [ns]")
        plt.ylabel("Prob |1>")
    if update:
        time = data[:, 0]
        p1 = data[:, 2]
        x_val, p_val = fitting.getMaxPoly(time, p1, True)
        name = devs[measure]['readoutDevice'].__name__
        idx = sample['config'].index(name)
        Qubits[idx]['timingLagRRUwave'] = x_val*ns

    return data

def xypaFunc(Sample, measure=0, amps=st.r[0.02:0.5:0.02], length=st.r[0:1000:2, ns], stats=1200,
             save=True, noisy=True, nfftpoints=4000, update=True, plot=True, forceZero=True):
    """
    get rabi amp vs rabi frequency
    freq = a * amp^2 + b*amp + c, and registry key calXYpaFunc=[a, b, c]
    and registry key calXYpaFunc=[a,b,c]

    get rabi frequency vs rabi amp
    amp = a * freq^2 + b * freq + c
    and registry key calRabiFreqFunc=[a,b,c]

    fitting these functions by 2-order is due to the experimental results.
    if forceZero is True, then the fitting function will be a*x^2 + b*x
    """

    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, write_access=True)
    data = rabi(Sample, measure, rabiLen=length, rabiAmp=amps, prob=True,
                name='XYPA Func', save=save, noisy=noisy, stats=stats)

    def quadratic(x, a, b):
        return a*x**2 + b*x

    data = np.array(data)
    length, amps, probs0 = dstools.format2D(data[:,(0,1,2)])
    rabi_freqs = np.array([fitting.maxFreq(np.vstack((length, prob)).T, nfftpoints,
                           plot=False)['GHz'] for prob in probs0])
    if forceZero:
        func, _ = curve_fit(quadratic, amps, rabi_freqs)
        func_I, _ = curve_fit(quadratic, rabi_freqs, amps)
        func = np.hstack([func, 0.0])
        func_I = np.hstack([func_I, 0.0])
    else:
        func = np.polyfit(amps, rabi_freqs, 2)
        func_I = np.polyfit(rabi_freqs, amps, 2)
    if update:
        Q = Qubits[measure]
        Q['calXYpaFunc'] = list(func)
        Q['calRabiFreqFunc'] = list(func_I)
    if plot or update:
        plt.figure()
        plt.plot(amps, rabi_freqs,'ro')
        plt.plot(amps, np.poly1d(func)(amps),'b-')
        plt.xlabel('rabi amp')
        plt.ylabel('rabi freq')
        plt.figure()
        plt.plot(rabi_freqs, amps, 'ro')
        plt.plot(rabi_freqs, np.poly1d(func_I)(rabi_freqs), 'b-')
        plt.xlabel('rabi freq')
        plt.ylabel('rabi amp')

    return data, func, func_I

def readoutOpt(Sample, measure=0, average=5, states=[0,1], readoutPower=None, readoutLen=None,
               readoutFrequency=None, stats=3000, name='Readout Opt', save=True, noisy=True):
    """
    scan a range of readout parameters to find the optimal point of readout
    """
    sample, devs, qubits = gc.loadQubits(Sample, measure)
    qM = qubits[measure]

    if readoutPower is None:
        readoutPower = qM['readoutPower']
    if readoutLen is None:
        readoutLen = qM['readoutLen']
    if readoutFrequency is None:
        readoutFrequency = qM['reaodutFrequency']

    axes = [(readoutPower, 'readoutPower'), (readoutLen, 'readoutLen'),
            (readoutFrequency, 'readoutFrequency')]
    deps = [("Average Fidelity", "", "")]
    kw = {'stats': stats, 'average': average, "states": states}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, currPower, currLen , currFreq):
        average_data = []
        for i in range(average):
            reqs = []
            for state in states:
                alg = gc.Algorithm(devs)
                q0 = alg.q0
                q0['readoutPower'] = currPower
                q0['readoutLen'] = currLen
                q0['readoutFrequency'] = currFreq
                alg[gates.MoveToState([q0], 0, state)]
                alg[gates.Measure([q0])]
                alg.compile()
                reqs.append(runQubits(server, alg.agents, stats=stats, dataFormat='iqRaw'))
            data = yield FutureList(reqs)
            data = [np.squeeze(dat) for dat in data]
            fids = readout.iqToReadoutFidelity(data, states=states)[0]
            average_data.append(np.mean(fids))
        returnValue([np.mean(average_data)])

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data

def readoutSNR(Sample, measure=0, readoutFrequency=None, readoutPower=None, readoutLen=None,
               stats=3000, name='Readout SNR', save=True, noisy=True, update=True):
    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, write_access=True)
    qM = qubits[measure]

    if readoutPower is None:
        readoutPower = st.centerscanPQ(qM['readoutPower']['dBm'], 5, 0.2, dBm)
    if readoutLen is None:
        readoutLen = st.centerscanPQ(qM['readoutLen']['us'], 1, 0.05, us)
    if readoutFrequency is None:
        readoutFrequency = st.centerscanPQ(qM['readoutFrequency']['MHz'], 2, 0.1, MHz)

    axes = [(readoutPower, 'readoutPower'), (readoutLen, 'readoutLen'), (readoutFrequency, 'readoutFrequency')]
    deps = [("SNR", "", "")]
    kw = {'stats': stats}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, currPower, currLen , currFreq):
        reqs = []
        for state in [0, 1]:
            alg = gc.Algorithm(devs)
            q0 = alg.q0
            q0['readoutPower'] = currPower
            q0['readoutLen'] = currLen
            q0['readoutFrequency'] = currFreq
            alg[gates.MoveToState([q0], 0, state)]
            alg[gates.Measure([q0])]
            alg.compile()
            reqs.append(runQubits(server, alg.agents, stats=stats, dataFormat='iqRaw'))

        data = yield FutureList(reqs)
        data = np.squeeze(data)
        fids, probs, centers, stds = readout.iqToReadoutFidelity(data, states=[0,1])
        dist = np.abs(centers[0] - centers[1])
        std = np.mean([stds[0], stds[1]])
        snr = (dist/std)**2
        returnValue([snr])

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    if update and (data.shape[1] == 2): # scan over readoutFrequency
        Q = Qubits[measure]
        adj = adjust.Adjust()
        f = data[:, 0]
        snr = data[:, 1]
        adj.plot(f, snr, '.-')
        adj.x_param('readoutFrequency', Q['readoutFrequency']['GHz'], np.min(f), np.max(f), 'r')
        result = adj.run()
        if result:
            freq = result['readoutFrequency']
            Q['readoutFrequency'] = freq*GHz
    return data

def readoutNM(Sample, measure=0, average=10,
              paramNames=['readoutPower', 'readoutLen', 'readoutFrequency',
                          'readoutRingupFactor', 'readoutRingupLen'],
              states=[0,1], name='Readout Nelder-Mead',
              nonzdelt=0.2, zdelt=0.2, xtol=0.001, ftol=0.001,
              maxiter=None, maxfun=None, stats=1500, save=True, noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure)
    params = popt.Parameters(devs, measure, paramNames=paramNames)

    axes, deps, inputs = params.makeInputsAxesDeps(nelderMead=True)
    kw = {'stats': stats, 'paramNames': paramNames, 'average': average, 'states': states,
          'nonzdelta': nonzdelt, 'zdelt': zdelt, 'xtol': xtol, 'ftol': ftol,
          'maxiter': maxiter, 'maxfun': maxfun}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure, kw=kw)

    def func(server, args):
        currParam = params.args2Params(args)

        alg = gc.Algorithm(devs)
        params.updateQubits(alg.agents, currParam, noisy=False)
        alg[gates.Measure([alg.q0])]
        alg.compile()
        data0 = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')

        alg = gc.Algorithm(devs)
        params.updateQubits(alg.agents, currParam, noisy=False)
        alg[gates.PiPulse([alg.q0])]
        alg[gates.Measure([alg.q0])]
        alg.compile()
        data1 = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')

        fids = readout.iqToReadoutFidelity([np.squeeze(data0), np.squeeze(data1)], states=states)[0]
        error = 1 - np.mean(fids)
        if noisy:
            print currParam, error
        returnValue([error])

    funcWrapper = popt.makeFunctionWrapper(average, func, axes, deps, measure, sample, noisy=noisy)
    output = sweeps.fmin(funcWrapper, inputs, dataset, xtol=xtol, ftol=ftol,
                         nonzdelt=nonzdelt, zdelt=zdelt, maxiter=maxiter, maxfun=maxfun)

    return output


def swapSpectroscopy(Sample, measure=0, swapAmp=st.r[-1:1:0.05], swapLen=st.r[0:200:2, ns], overshoot=0.0,
                     state=1, tBuf=5*ns, stats=600, name='swap spectroscopy', save=True, noisy=True):

    sample, devs, qubits = gc.loadQubits(Sample, measure)
    axes = [(swapAmp, 'swap Amp'), (swapLen, 'swapLen'), (overshoot, 'overshoot')]
    deps = readout.genProbDeps(qubits, measure, range(1+state))
    kw = {"stats": stats, 'tBuf': tBuf, "state": state}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure, kw=kw)

    def func(server, currAmp, currLen, currOs):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        alg[gates.MoveToState([q0], 0, state)]
        alg[gates.Wait([q0], waitTime=tBuf)]
        alg[gates.Detune([q0], currLen, currAmp, currOs)]
        alg[gates.Wait([q0], waitTime=tBuf)]
        alg[gates.Measure([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        probs = readout.iqToProbs(data, alg.qubits, states=range(1+state))
        returnValue(np.squeeze(probs))

    data = sweeps.grid(func, axes=axes, save=save, dataset=dataset, noisy=noisy)

    return data

def swapTuner(Sample, measure=0, swapAmp=0.0, swapTime=0.0*ns, iterations=3,
              ampBound=0.025, timeBound=10.0*ns, state=1,
              save=False, stats=1200, noisy=True):
    """Finds the best qubit z-pulse amplitude and pulse length to do a swap"""
    name = 'Swap Tuner |%d>' %state
    sample, devs, qubits = gc.loadQubits(Sample, measure)
    qubit=qubits[measure]

    if noisy:
        print 'Original swap amplitude: %f' %swapAmp
        print 'Original swap length: %f ns' %swapTime['ns']

    for iteration in range(iterations):
        dataStats=stats*(2**iteration)
        dSwapAmp = ampBound/(2.0**iteration)
        dSwapTime = (timeBound/(2.0**iteration))['ns']
        data = swapSpectroscopy(sample, measure=measure,
                                swapLen=swapTime,
                                swapAmp=np.linspace(swapAmp-dSwapAmp,swapAmp+dSwapAmp,21),
                                state=state, save=save, stats=dataStats, noisy=noisy, name=name)
        swapAmp, probability = fitting.findMinimum(np.array(data[:,(0,2)]), fit=True)
        if noisy: print 'Best swap amplitude: %f' %swapAmp
        data = swapSpectroscopy(sample,
                                swapLen=np.linspace(swapTime['ns']-dSwapTime,swapTime['ns']+dSwapTime,21)*ns,
                                swapAmp=swapAmp, measure=measure, state=state, save=save,
                                stats=dataStats, noisy=noisy, name=name)
        swapTime, probability = fitting.findMinimum(np.array(data[:, (0,2)]),fit=True)
        swapTime = swapTime*ns
        if noisy: print 'Best swap time: %f ns' %swapTime[ns]

    return swapTime,swapAmp

def dragScanAlpha(Sample, measure=0, alphas=st.r[-1:2.5:0.01], numPulses=10,
                  stats=6000, name='DRAG scan Alpha', save=True, update=True, noisy=True):
    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, write_access=True)
    axes = [(alphas, 'alphaDRAG')]
    deps = [("Probability |1>", "X", ""), ("Probability |1>", "Y", "")]
    kw = {"stats": stats, "numPulses": numPulses}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, currAlpha):
        reqs = []
        # PingPong at X axis
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        q0['alphaDRAG'] = currAlpha
        alg[gates.PiHalfPulse([q0])]
        alg[gates.PingPong([q0], numPulses, phase=0)]
        alg[gates.PiHalfPulse([q0])]
        alg[gates.Measure([q0])]
        alg.compile()
        reqs.append(runQubits(server, alg.agents, stats, dataFormat='iqRaw'))

        # PingPong at Y axis
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        q0['alphaDRAG'] = currAlpha
        alg[gates.PiHalfPulse([q0])]
        alg[gates.PingPong([q0], numPulses, phase=np.pi/2.0)]
        alg[gates.PiHalfPulse([q0])]
        alg[gates.Measure([q0])]
        alg.compile()
        reqs.append(runQubits(server, alg.agents, stats, dataFormat='iqRaw'))

        data = yield FutureList(reqs)

        probX = readout.iqToProbs(data[0], alg.qubits, states=[0,1])
        probY = readout.iqToProbs(data[1], alg.qubits, states=[0,1])
        probX = np.squeeze(probX)[1]
        probY = np.squeeze(probY)[1]
        returnValue([probX, probY])

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    def fit_cos(x, f, x0, amp, offset):
        return np.cos(2*np.pi*f*(x-x0)) * np.abs(amp) + offset

    if update:
        x = data[:, 0]
        xp = np.linspace(np.min(x), np.max(x), 10 * len(x) + 1)
        p1X, p1Y = data[:, 1], data[:, 2]
        pfitX, pcov = curve_fit(fit_cos, x, p1X,
                                p0=[0.75, 0.0, 0.5 * (np.max(p1X) - np.min(p1X)),
                                    0.5 * (np.max(p1X) + np.min(p1X))])
        pfitY, pcov = curve_fit(fit_cos, x, p1Y,
                                p0=[0.75, 0.0, 0.5 * (np.max(p1X) - np.min(p1X)),
                                    0.5 * (np.max(p1X) + np.min(p1X))])
        alphaX = 1. / pfitX[0] + pfitX[1]
        alphaY = 1. / pfitY[0] + pfitY[1]
        alpha = 0.5 * (alphaX + alphaY)
        plt.figure()
        plt.plot(x, p1X, 'o', color=BLUE, label='X')
        plt.plot(x, p1Y, 'o', color=RED, label='Y')
        plt.plot(xp, fit_cos(xp, *pfitX), color=BLUE)
        plt.plot(xp, fit_cos(xp, *pfitY), color=RED)
        plt.legend()
        plt.xlabel("alpha DRAG")
        plt.ylabel("P|1>")
        Qubits[measure]['alphaDRAG'] = round(alpha, 8)
    return data

def equatorTuners(Sample, measure=0, iterations=st.r[0:10:1, None], phase=0.0,
                  name='Equator Tuners', stats=1500, save=True, noisy=True):

    sample, devs, qubit = gc.loadQubits(Sample, measure, write_access=False)

    axes = [(iterations, 'Pulse iterations')]
    deps = [('Probability |1>', '%s' % i, '') for i in ['x/2','x','y/2','y']]
    kw = {'stats': stats, 'initial phase': phase}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, iteration):

        reqs = []

        #x/2
        alg = gc.Algorithm(agents=devs)
        alg[gates.PiHalfPulse([alg.q0], phase=phase)]
        alg[gates.NPiHalfPulses([alg.q0], iteration*2, phase = 0)]
        alg[gates.Measure([alg.q0])]
        alg.compile()
        reqs.append(runQubits(server, alg.agents, stats, dataFormat='iqRaw'))

        #x
        alg = gc.Algorithm(agents=devs)
        alg[gates.PiHalfPulse([alg.q0], phase=phase)]
        alg[gates.NPiPulses([alg.q0], iteration, phase = 0)]
        alg[gates.Measure([alg.q0])]
        alg.compile()
        reqs.append(runQubits(server, alg.agents, stats, dataFormat='iqRaw'))

        #y/2
        alg = gc.Algorithm(agents=devs)
        alg[gates.PiHalfPulse([alg.q0], phase=phase)]
        alg[gates.NPiHalfPulses([alg.q0], iteration*2, phase = np.pi/2.)]
        alg[gates.Measure([alg.q0])]
        alg.compile()
        reqs.append(runQubits(server, alg.agents, stats, dataFormat='iqRaw'))

        #y
        alg = gc.Algorithm(agents=devs)
        alg[gates.PiHalfPulse([alg.q0], phase=phase)]
        alg[gates.NPiPulses([alg.q0], iteration, phase = np.pi/2.)]
        alg[gates.Measure([alg.q0])]
        alg.compile()
        reqs.append(runQubits(server, alg.agents, stats, dataFormat='iqRaw'))

        results = yield FutureList(reqs)
        probs = []
        for r in results:
            prob = readout.iqToProbs(r, alg.qubits)
            p1 = np.squeeze(prob)[1]
            probs.append(p1)
        probs = np.hstack(probs)
        returnValue(probs)

    """
    if we consider the amplitude of piPulse gets an error `err`
    We repeat piPulse N times, then we have 
    |psiT> = exp(-i*pi/2*sigmax*(1+err)*N)|psi0>
    |psi0> = 1/sqrt(2) (|0> - i|1>) # X/2 pulse on |0>
    ====>>> P1 = 0.5*( 1+sin(N*pi*(1+err)) )
    
    We repeat piHalfPulse 2N times, then we have
    |psiT> = exp(-i*pi/4*sigmax*(1+err)*2*N)|psi0>
    ====>>> P1 = 0.5*( 1+sin(N*pi*(1+err) )
    """
    def fitFunc(n, err, a):
        return a*(1+np.sin(n*np.pi*(1+err)))

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data

def amplifyPhaseError(Sample, measure=0, phase=np.arange(0, 2, 0.01)*np.pi, numAmp=(0, 3, 6, 9),
                      alpha=0.5, stats=1200, name='Amplify Phase Error',
                      save=True, noisy=True, plot=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure=measure)

    axes = [(phase, "Phase of Pi/2 Pulse")]
    deps = [("Probability |1>", "numPulse=%d" %n, "") for n in numAmp]
    kw = {'stats': stats, "alphaDRAG": alpha, }
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, currPhase):
        reqs = []
        for num in numAmp:
            alg = gc.Algorithm(devs)
            q0 = alg.qubits[0]
            alg[gates.PiHalfPulse([q0], alpha=alpha)]
            alg[gates.PingPong([q0], num, alpha=alpha)]
            alg[gates.PiHalfPulse([q0], alpha=alpha, phase=currPhase)]
            alg[gates.Measure([q0])]
            alg.compile()
            reqs.append(runQubits(server, alg.agents, stats, dataFormat='iqRaw'))
        data = yield FutureList(reqs)
        probs = []
        for dat in data:
            prob = readout.iqToProbs(dat, alg.qubits)
            probs.append(np.squeeze(prob)[1])
        returnValue(probs)

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    def fit_cos(x, w, x0, amp, offset):
        return np.cos(w*(x-x0)) * amp + offset

    if plot:
        colors = COLORS
        x = data[:,0]
        xp = np.linspace(np.min(x), np.max(x), 10*len(x))
        new_data = data[:,1:].T
        ps = []
        plt.figure()
        for y, c in zip(new_data, colors):
            y_max, y_min = np.max(y), np.min(y)
            p0 = [1.0, 0.0, 0.5*(y_max-y_min), 0.5*(y_max+y_min)]
            p, cov = curve_fit(fit_cos, x, y, p0=p0)
            plt.plot(x/np.pi, y, 'o', color=c)
            plt.plot(xp/np.pi, fit_cos(xp, *p), c)
            ps.append(p)
        plt.xlabel("phase [pi]")
        plt.ylabel("P1")
        plt.grid()
        return data, ps

    return data

def ramseyXY(Sample, measure=0, iterations=st.r[0:20:1], name='Ramsey XY', stats=1200, save=True,
             noisy=True):
    """
    To see if X Y axis is really 90 degree
    """
    sample, devs, qubits = gc.loadQubits(Sample, measure=measure)

    axes = [(iterations, 'pulse iteration')]
    deps = readout.genProbDeps(qubits, measure)
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, currN):
        alg = gc.Algorithm(devs)
        q0 = alg.qubits[0]
        alg[gates.PiHalfPulse([q0], phase=0)]
        for i in range(currN):
            alg[gates.PiPulse([q0], phase=0)]
            alg[gates.PiPulse([q0], phase=np.pi/2)]
        alg[gates.PiHalfPulse([q0], phase=-np.pi/2)]
        alg[gates.Measure([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        probs = readout.iqToProbs(data, alg.qubits)
        returnValue(np.squeeze(probs))

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data

def nonlinearTuner(Sample, measure=0, freqScan=None, name='nonlinear tuner', stats=1500,
                   save=True, noisy=True, update=False):
    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure=0, write_access=True)

    if freqScan is None:
        f10_GHz = qubits[measure]['f10']['GHz']
        freqScan = st.r[f10_GHz-0.35:f10_GHz-0.15:0.001, GHz]

    axes = [(freqScan, 'f21 frequency')]
    deps = readout.genProbDeps(qubits, measure)
    kw = {'stats': stats, }
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, currFreq):
        alg = gc.Algorithm(devs)
        q0 = alg.qubits[0]
        q0['f21'] = currFreq
        alg[gates.MoveToState([q0], 0, 1)]
        alg[gates.PiPulse([q0], state=2)]
        alg[gates.MoveToState([q0], 1, 0)]
        alg[gates.Measure([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        probs = readout.iqToProbs(data, alg.qubits)
        returnValue(np.squeeze(probs))

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data


def piDetuneScan(Sample, measure=0, df=st.r[-50:50:1, MHz], alpha=None, numPulses=2, stats=1200,
                 mode='piPulse', name='pi detune scanning', save=True, noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure)

    if alpha is None:
        alpha = qubits[measure]['alphaDRAG']
    if mode == 'piHalfPulse':
        numPulses *= 2
        modeKey = 'piHalfDetune'
    elif mode == 'piPulse':
        modeKey = 'piDetune'
    else:
        raise Exception("Unsupport mode %s" %mode)

    axes = [(df, "%s detune" %mode)]
    deps = readout.genProbDeps(qubits, measure)
    kw = {'stats': stats, "mode": mode, "alpha": alpha, "numPulses": numPulses}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, currdf):
        alg = gc.Algorithm(devs)
        q0 = alg.qubits[0]
        q0[modeKey] = currdf
        alg[gates.PingPong([q0], N=numPulses, alpha=alpha, mode=mode)]
        alg[gates.Measure([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats)
        probs = readout.iqToProbs(data, alg.qubits)
        returnValue(np.squeeze(probs))

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data

def readoutFidelity(Sample, measure=0, reps=100, state=1, herald=False, stats=3000,
                    name='readout fidelity', save=True, update=True, noisy=True):
    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, write_access=True)

    axes = [(range(reps), "repetition")]
    deps = readout.genProbDeps(qubits, measure, states=range(state+1))
    kw = {"states": range(state+1), 'stats': stats, 'herald': herald}

    name += " " + " ".join(["|%d>" %l for l in range(state+1)])

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, curr):
        reqs = []
        for currState in range(state+1):
            alg = gc.Algorithm(devs)
            q0 = alg.q0
            if herald:
                alg[gates.Measure([q0], name='herald')]
                alg[gates.Wait([q0], q0['readoutRingdownLen'])]
            alg[gates.MoveToState([q0], 0, currState)]
            alg[gates.Measure([q0])]
            alg.compile()
            reqs.append(runQubits(server, alg.agents, stats=stats, dataFormat='iqRaw'))
        data = yield FutureList(reqs)
        probs = []
        for idx, dat in enumerate(data):
            prob = readout.iqToProbs(dat, alg.agents, herald=herald, states=range(1+state))
            prob = np.squeeze(prob)[idx] # for |0>, readout P0, for |1> readout P1, ...
            probs.append(prob)
        returnValue(probs)

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    if update:
        probs = data[:, 1:]
        mean_fid = np.mean(probs, axis=0)
        Q = Qubits[measure]
        if herald:
            Q['calHeraldReadoutFids'] = [round(f, 6) for f in mean_fid]
        else:
            Q['calReadoutFids'] = [round(f, 6) for f in mean_fid]

    return data

def readoutFidMat(Sample, measure=0, reps=100, state=1, herald=False, stats=3000,
                  name='readout fidelity matrix', save=True, update=True, noisy=True):
    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, write_access=True)

    axes = [(range(reps), "repetition")]
    deps = []
    for prep_s, meas_s in itertools.product(range(state+1), repeat=2):
        deps.append( ("Probability |%s>" %meas_s, "@|%s>" %prep_s, ""))

    kw = {"states": range(state+1), 'stats': stats, 'herald': herald}

    name += " " + " ".join(["|%d>" %l for l in range(state+1)])

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, curr):
        reqs = []
        for currState in range(state+1):
            alg = gc.Algorithm(devs)
            q0 = alg.q0
            if herald:
                alg[gates.Herald([q0], ringdown=False)]
            alg[gates.MoveToState([q0], 0, currState)]
            alg[gates.Measure([q0])]
            alg.compile()
            reqs.append(runQubits(server, alg.agents, stats=stats, dataFormat='iqRaw'))
        data = yield FutureList(reqs)
        probs = []
        for idx, dat in enumerate(data):
            prob = readout.iqToProbs(dat, alg.agents, herald=herald, states=range(1+state))
            prob = np.squeeze(prob) # record all the probs
            probs.append(prob)
        returnValue(np.hstack(probs))

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    if update:
        probs = data[:, 1:]
        mean_fid = np.mean(probs, axis=0)
        Q = Qubits[measure]
        if herald:
            fmat = np.array([round(f, 6) for f in mean_fid]).reshape(state+1, state+1)
            Q['calHeraldReadoutFMat'] = fmat
            Q['calHeraldReadoutFids'] = np.diag(fmat)
        else:
            fmat = np.array([round(f, 6) for f in mean_fid]).reshape(state+1, state+1)
            Q['calReadoutFMat'] = fmat
            Q['calReadoutFids'] = np.diag(fmat)

    return data


def spectroscopyZ2D(Sample, measure=0, freqScan=st.r[4.5:6.0:0.01, GHz], zbias=st.r[-1:1:0.02], uwaveAmp=None,
                  zlonger=100*ns, sb_freq=0*MHz, stats=600, name='Spectroscopy Z 2D', save=True, noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure)

    if uwaveAmp is None:
        uwaveAmp = qubits[measure]['spectroscopyAmp']
    axes = [(zbias, 'Z pulse amplitude'), (freqScan, 'drive frequency')]
    deps = [("Probability |1>", "", "")]
    kw = {'stats': stats, 'zlonger': zlonger, 'sb_freq': sb_freq, 'uwaveAmp': uwaveAmp}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)
    qubitNameCarrier = util.otherQubitNamesOnCarrier(qubits[measure], qubits)

    def func(server, currZ, currF):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        q0['spectroscopyAmp'] = uwaveAmp
        q0['fc'] = currF - sb_freq
        for name in qubitNameCarrier:
            alg.agents_dict[name]['fc'] = currF - sb_freq
        alg[gates.Spectroscopy([q0], df=sb_freq, z=currZ, zlonger=zlonger)]
        alg[gates.Measure([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        probs = readout.iqToProbs(data, alg.qubits, states=[0,1])
        returnValue([np.squeeze(probs)[1]])

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data

def spectroscopyTwoState(Sample, measure=0, freqScan=None, uwaveAmps=None, sb_freq=0*MHz, stats=600,
                           name='spectroscopy |2>', save=True, noisy=True, update=True):
    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, True)

    if freqScan is None:
        f10 = qubits[measure]['f10']['GHz']
        freqScan = st.r[f10-0.2 : f10+0.08 : 0.002, GHz]
    if uwaveAmps is None:
        amp = qubits[measure]['spectroscopyAmp']
        uwaveAmps = [amp, 5*amp, 10*amp]
    axes = [(freqScan, "qubit drive frequency")]
    deps = [[("Mag", "amp=%s" %a, ""), ("Phase", "amp=%s" %a, "") ] for a in uwaveAmps]
    deps = sum(deps, [])
    qubitNameCarrier = util.otherQubitNamesOnCarrier(qubits[measure], qubits)

    kw = {'stats': stats, "uwaveAmps": uwaveAmps, "sb_freq": sb_freq}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, freq):
        reqs = []
        for amp in uwaveAmps:
            alg = gc.Algorithm(devs)
            q0 = alg.q0
            q0['spectroscopyAmp'] = amp
            q0['fc'] = freq- sb_freq
            for name in qubitNameCarrier:
                alg.agents_dict[name]['fc'] = freq - sb_freq
            alg[gates.Spectroscopy([q0], df=sb_freq, )]
            alg[gates.Measure([q0])]
            alg.compile()
            reqs.append(runQubits(server, alg.agents, stats, dataFormat='iqRaw'))

        ans = yield FutureList(reqs)
        data = []
        for dat in ans:
            mag, phase = readout.iqToPolar(readout.parseDataFormat(dat, 'iq'))
            data.extend([mag, phase])
        returnValue(data)

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    if update:
        idx = [0] + [1+2*n for n in range(len(uwaveAmps))]
        idx = tuple(idx)
        q, Q = qubits[measure], Qubits[measure]
        adjust.adjust_frequency_02(q, data[:, idx])
        Q['f10'] = round(q['f10']['GHz'], 6)*GHz
        Q['f21'] = round(q['f21']['GHz'], 6)*GHz
    return data

def spectroscopyZfunc(Sample, measure=0, zrange='coarse', fspan=40.0*MHz,
                      fspansteps=41, coarseSteps=25, coarseAmp=0.5,
                      stats=450, sb_freq=-100.0*MHz, name='Spectroscopy Z func',
                      save=True, noisy=True, update=True, correctXtalkZ=False):

    if zrange == 'coarse':
        halfRange = np.logspace(np.log10(0.001), np.log10(coarseAmp), coarseSteps)
        zrange = np.hstack((halfRange, -halfRange))

    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, write_access=True)
    qubit = qubits[measure]

    otherQubitCarrierNames = util.otherQubitNamesOnCarrier(qubits[measure], qubits)

    zrange = zrange[np.argsort(np.abs(zrange))] # reorder zrange to start at 0, and then move up and down

    fsweep = np.linspace(-1.0, 1.0, fspansteps) * fspan['MHz']
    fsweep = fsweep[np.argsort(abs(fsweep))] * MHz
    fStep = 1.0 * MHz / fspansteps
    freqPoints = len(fsweep)

    axes = [('Frequency', 'GHz'), ('Z', '')]
    deps = [('Probability |1>', '', '')]
    kw = {'stats': stats, 'correctXtalkZ': correctXtalkZ}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def qubitFreqModel(v, A, B, C):
        """Approximate qubit frequency as a function of bias

        This ignores the offset from Ec so do not use this to fit data
        """
        return A*np.cos(B*v - C)**0.5

    sweepData={
        'freqIdx': 0,
        'fluxIdx': 0,
        'freq': np.zeros_like(fsweep, dtype=float),
        'prob': np.zeros_like(fsweep, dtype=float),
        'maxima': np.zeros_like(zrange, dtype=float),
        'freqFunc': np.array([0]),
        'parameters': (qubit['f10']['GHz'], 1.0, 0.0),
        'linearFit': None
        }

    def sweep():
        for z in zrange:
            if sweepData['fluxIdx'] in [0,1,2] or sweepData['parameters'] is None:
                freqCenter_GHz = qubit['f10']['GHz']
            else:
                freqCenter_GHz = qubitFreqModel(z, *sweepData['parameters'])
            freqCenter_GHz = st.nearest(freqCenter_GHz, fStep['GHz'])
            for df in fsweep:
                yield z, (df["GHz"] + freqCenter_GHz)*GHz
                # yield z, (df+(freqCenter_GHz*GHz))['GHz']*GHz #Don't ask

    def func(server, args):
        print('fluxIdx: %d' %sweepData['fluxIdx'])
        zAmplitude, frequency = args
        print frequency, zAmplitude
        # Set carrier and do spectroscopy
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        q0['fc'] = frequency - sb_freq
        alg[gates.Spectroscopy([q0], df=sb_freq, z=zAmplitude, zlonger=100*ns)]
        if otherQubitCarrierNames:
            for otherQubitName in otherQubitCarrierNames: # null out other lines w same carrier
                alg.agents_dict[otherQubitName]['fc'] = frequency - sb_freq
        alg[gates.Measure([q0])]
        alg.compile(correctXtalkZ=correctXtalkZ)

        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        prob = np.squeeze(readout.iqToProbs(data, alg.qubits, states=[0,1]))[1]

        idx = sweepData['freqIdx']
        sweepData['freq'][idx] = frequency["GHz"]
        sweepData['prob'][idx] = prob

        if idx+1 != freqPoints: #Not done with this flux
            sweepData['freqIdx'] += 1
        else: # Update fit
            # Store max frequency
            fluxIdx = sweepData['fluxIdx']
            maxFreqIdx = np.argmax(sweepData['prob'])
            maxFreq = sweepData['freq'][maxFreqIdx]
            sweepData['maxima'][fluxIdx] = maxFreq
            # update running fit
            fluxToFit = zrange[:sweepData['fluxIdx']+1]
            freqToFit_GHz = sweepData['maxima'][:sweepData['fluxIdx']+1]
            if fluxIdx < 4:
                pass
            else:
                try:
                    guess = list(sweepData['parameters'])
                    print "fluxToFit: %s" % fluxToFit
                    print "freqToFit_GHz: %s" % freqToFit_GHz
                    print "guess: ", guess
                    pOpt, pCov = curve_fit(qubitFreqModel, fluxToFit, freqToFit_GHz, guess, maxfev=10000)
                    sweepData['parameters'] = pOpt
                except:
                    print("WARNING: fitting did not converge on this pass")
            sweepData['fluxIdx'] += 1
            sweepData['freqIdx'] = 0
            # if fluxIdx > 8:
            #     print("Switching to 40MHz freq span")
            #     fsweep=np.linspace(-1.0,1.0,fspansteps)*40
        returnValue([frequency["GHz"], zAmplitude, prob])
    data = sweeps.run(func, sweep(), dataset=dataset, save=save, noisy=False)

    if update:
        calculateZpaFunc(data, Sample=Sample, update=True)

def calculateZpaFunc(dataset, Sample=None, update=False, skipIdx=[], p0=None):
    """
    Recalculates the ZpaFunc, using the transmon sqrt(|cos(pi*f)|) analytic formula.

    note that, for dataset: freq, zAmp, prob

    The Ec in the fitting function can not fitting exactly, and usually be negative,
    so we keep the Ec fixed when we fitting this function.
    Ec is set as f10 - f21
    """

    measure = dataset.parameters['measure'][0]
    config = dataset.parameters['config']
    qDict = dataset.parameters[config[measure]]
    Ec = qDict['f10']['GHz'] - qDict['f21']['GHz']

    # data format (rows): f (GHz), zAmp, prob
    dataLen = len(dataset[:, 0])
    num_freqSteps = len(mlab.find(dataset[:, 1] == dataset[0, 1])) # get the number of frequency for each zAmp
    num_zAmp = dataLen/num_freqSteps  # the number of zAmp

    print dataLen, num_freqSteps, num_zAmp

    counter = 0

    freqs = []
    bias = []

    for tel in range(num_zAmp):

        fs   = np.array(dataset[counter:counter + num_freqSteps, 0])
        zs   = np.array(dataset[counter:counter + num_freqSteps, 1]) # now zs should be the same value
        z = np.unique(zs)
        if len(z) > 1:
            print("Error!")
        probs = np.array(dataset[counter:counter + num_freqSteps, 2])

        if tel in skipIdx:
            print("skip %d,  z=%s" %(tel, z[0]))
            counter += num_freqSteps
            continue

        center_freq, prob = fitting.getMaxGauss(fs, probs, fit=False)
        freqs.append(center_freq)
        bias.append(z[0])

        counter += num_freqSteps

    freqs = np.array(freqs)
    bias = np.array(bias)

    # fitFunc = zfuncs.fitFunc # (x, M, fmax, offset, fc)
    # we keep the fc as the nonlinear (f10 - f21)
    fitFunc = lambda x, M, fmax, offset: zfuncs.fitFunc(x, M, fmax, offset, Ec)
    # fitFunc = lambda v, A, B, C : B*np.cos(A*(v - C))**0.5

    idx = np.argmax(freqs) # get max

    if p0 is None:
        p0 = [0.5, np.max(freqs), bias[idx]]

    p1, _ = curve_fit(fitFunc, bias, freqs, p0, maxfev=int(1e5))
    p1, _ = curve_fit(fitFunc, bias, freqs, p1, maxfev=int(1e5))
    print np.hstack((p1, Ec))
    M = p1[0]  # this is an effective mutual
    offset = p1[2]


    contb = np.linspace(-2.0, 2.0, 1001)

    plt.figure(figsize=(9, 3.5))
    plt.subplot(1,2,1)
    plt.plot(bias, freqs, 'o')
    plt.plot(contb, fitFunc(contb, *p1), )
    plt.xlim([-1.1, 1.1])
    plt.xlabel("Z Amplitude [a.u.]")
    plt.ylabel("Frequency [GHz]")
    plt.subplot(1,2,2)
    plt.plot(bias, freqs - fitFunc(bias, *p1), 'o')
    plt.xlim([-1.1, 1.1])
    plt.xlabel("Z Amplitude [a.u.]")
    plt.ylabel("Error [GHz]")
    plt.tight_layout()


    p1 = np.hstack([p1, Ec])
    if update:
        sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, write_access=True)
        Qubits[measure]['calZpaFunc'] = [round(x, 8) for x in p1]
        print 'updated analytic function parameters'

    return bias, freqs, p1

def currentZpaFunc(Sample, measure, zrange=np.linspace(-2,2,1e3)):

    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, write_access=True)

    func = zfuncs.AmpToFrequency(qubits[measure])
    f10s = func(zrange)
    plt.figure()
    plt.plot(zrange, f10s)
    plt.xlabel('z amplitude')
    plt.ylabel('frequency [GHz]')
    plt.xlim((-1.0,1.0))
    plt.ylim((-0.1+np.min(f10s),np.max(f10s)+0.1))
    plt.plot(0, qubits[measure]['f10']['GHz'], 'or')

    return zrange, func

def acStarkShift(Sample, measure=0, freqShift=None, ampSquare=None, excitationLen=1*us, buffer=100*ns,
                 delay=500*ns, name='AC Stark Shift', stats=300, save=True, noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure=measure)

    q = devs[measure]
    if freqShift is None:
        freqShift = st.r[-0.15:0.06:0.005, GHz]
    if ampSquare is None:
        ampSquare = np.arange(0, 0.03, 0.001)
    eta = q['f21'] - q['f10']

    axes = [(ampSquare, 'dac amplitude square'), (freqShift, 'Frequecy Shift to f10')]
    deps = readout.genProbDeps(qubits, measure, states=[0,1])

    refPower = q['readoutPower']
    df = q['readoutFrequency'] - q['readoutDevice']['carrierFrequency']
    f10 = q['f10']
    kw = {'stats': stats, 'refPower': refPower, 'f10': f10, 'buffer': buffer, 'delay': delay}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, currAmpSquared, fShift):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        q0['f10'] = f10 + fShift
        q0['f21'] = f10 + eta + fShift
        alg[gates.ACStark([q0], ringUp=excitationLen, buffer=buffer, amp=np.sqrt(currAmpSquared))]
        alg[gates.Wait([q0], waitTime=delay)]
        alg[gates.Measure([q0])]
        alg.compile()

        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        probs = readout.iqToProbs(data, alg.qubits, states=[0, 1])
        returnValue(np.squeeze(probs))

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data

def readoutSpectroscopy(Sample, measure=0, delay=None, freqShift=None, stats=600, padding=500*ns,
                        ringdown=False, name='Readout Spectroscopy', save=True, noisy=True):
    """
    @param padding, the time between the two readout pulse
    """
    sample, devs, qubits = gc.loadQubits(Sample, measure)

    q = qubits[measure]
    if delay is None:
        readoutLength = q['readoutWidth']*2 + q['readoutLen']
        ringdownLength = q.get('readoutRingdownLen', 300*ns)
        start = -2*q['readoutWidth']['ns']-100
        end = readoutLength['ns'] + q['readoutWidth']['ns']*2 + ringdownLength['ns']
        delay = st.r[start:end:10, 'ns']
    if freqShift is None:
        freqShift = st.r[-0.20:0.06:0.005, GHz]

    eta = q['f21'] - q['f10']
    f10_fc = q['f10'] - q['fc']

    axes = [(delay, 'Delay'), (freqShift, "Frequency Shift to f10")]
    deps = [("Probability |1>", "", "")]
    kw = {"stats": stats, "padding": padding, "ringdown": ringdown}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)
    qubitNameCarrier = util.otherQubitNamesOnCarrier(q, qubits)

    def func(server, currDelay, currFreqShift):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        f10 = q0['f10']
        # change fc to keep the detuning between fc and f10, so that the piAmp should not change too much
        fc = q0['fc'] + currFreqShift
        q0['fc'] = fc
        for name in qubitNameCarrier:
            alg.agents_dict[name]['fc'] = fc
        alg[gates.ReadoutSpectroscopy([q0], delay=currDelay, freq=currFreqShift+f10, ringdown=ringdown)]
        alg[gates.Wait([q0], padding)]
        alg[gates.Measure([q0])]
        alg.compile()

        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        probs = readout.iqToProbs(data, alg.qubits)
        returnValue([np.squeeze(probs)[1]]) # readout P1

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data

def readoutSpectroscopyRingdown(Sample, measure=0, ringdownAmp=None, ringdownPhase=None, ringdownLen=None,
                                delay=400*ns, freqShift=None, padding=500*ns, stats=600,
                                name='Readout Spectroscopy Ringdown', save=True, noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure)

    q = qubits[measure]
    if ringdownLen is None:
        ringdownLen = q['readoutRingdownLen']
    if ringdownPhase is None:
        ringdownPhase = q['readoutRingdownPhase']
    if ringdownAmp is None:
        ringdownAmp = q['readoutRingdownAmp']
    if delay is None:
        delay = q['readoutLen'] + 2*q['readoutWidth'] + q.get('heraldDelay', 500*ns)/5.0
    if freqShift is None:
        freqShift = st.r[-0.20:0.06:0.005, GHz]

    axes = [(ringdownAmp, 'readoutRingdown Amplitude'), (ringdownLen, 'readoutRingdown Length'),
            (ringdownPhase, 'readoutRingdown Phase'), (freqShift, 'Frequency shift to f10')]

    deps = [("Probability |1>", "", "")]
    kw = {'stats': stats, 'padding': padding, 'delay': delay}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)
    qubitNameCarrier = util.otherQubitNamesOnCarrier(q, qubits)

    def func(server, currAmp, currLen, currPhase, currFreqShift):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        f10 = q0['f10']
        # change fc to keep the detuning between fc and f10, so that the piAmp should not change too much
        fc = q0['fc'] + currFreqShift
        q0['fc'] = fc
        for name in qubitNameCarrier:
            alg.agents_dict[name]['fc'] = fc
        q0['readoutRingdownAmp'] = currAmp
        q0['readoutRingdownLen'] = currLen
        q0['readoutRingdownPhase'] = currPhase
        alg[gates.ReadoutSpectroscopy([q0], delay=delay, freq=currFreqShift+f10, ringdown=True)]
        alg[gates.Wait([q0], padding)]
        alg[gates.Measure([q0])]
        alg.compile()

        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        probs = readout.iqToProbs(data, alg.qubits)
        returnValue([np.squeeze(probs)[1]]) # readout P1

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data

def detuneT1(Sample, measure=0, detuneAmp=None, delay=st.r[0:6000:50, ns], state=1,
             stats=600, name='Detune T1', save=True, noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure)
    axes = [(detuneAmp, 'Z Pulse Amplitude'), (delay, 'Delay')]
    deps = readout.genProbDeps(qubits, measure, range(1+state))
    kw = {"stats": stats, "state": state}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure, kw=kw)

    def func(server, currAmp, currLen):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        alg[gates.MoveToState([q0], 0, state)]
        if currLen > 14*us:
            alg[gates.DualBlockWaitDetune([q0], q0, currLen, amp=currAmp)]
        else:
            alg[gates.Detune([q0], currLen, currAmp)]
        alg[gates.Measure([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        probs = readout.iqToProbs(data, alg.qubits, states=range(1+state))
        returnValue(np.squeeze(probs))

    data = sweeps.grid(func, axes=axes, save=save, dataset=dataset, noisy=noisy)

    return data

def detuneDephasing(Sample, measure=0, detuneAmp=st.r[-0.5:0.5:0.05], delay=st.r[0:1000:5, ns], echo=False,
                    riseTime=5*ns, fringeFreq=5*MHz, stats=600, name='Detune Dephasing', save=True, noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure)

    axes = [(detuneAmp, "Z Pulse Amplitude"), (delay, "Delay")]
    deps = [("Probability", "+X", ""), ("Probability", "+Y", ""), ("Probability", "-X", ""),
            ("Probability", "-Y", ""), ("Envelope", "", "")]
    tomoPhaseNames = ["+X", "+Y", "-X", "-Y"]
    tomoPhases = {"+X": 0.0, "+Y": 0.25, "-X": -0.5, "-Y": -0.25}
    kw = {"echo": echo, "stats": stats, "riseTime": riseTime, "fringeFreq": fringeFreq}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure, kw=kw)

    def func(server, currAmp, currDelay):

        reqs = []

        for tomoKey in tomoPhaseNames:
            alg = gc.Algorithm(devs)
            q0 = alg.q0
            alg[gates.PiHalfPulse([q0])]
            if echo:
                alg[gates.DetuneFlattop([q0], tlen=currDelay/2.0, amp=currAmp, w=riseTime)]
                alg[gates.PiPulse([q0])]
                alg[gates.DetuneFlattop([q0], tlen=currDelay/2.0, amp=currAmp, w=riseTime)]
            else:
                alg[gates.DetuneFlattop([q0], tlen=currDelay/2.0, amp=currAmp, w=riseTime)]
            phase = 2*np.pi*(fringeFreq['GHz']*currDelay['ns'] + tomoPhases[tomoKey])
            alg[gates.PiHalfPulse([q0], phase = phase)]
            alg[gates.Measure([q0])]
            alg.compile()

            reqs.append(runQubits(server, alg.agents, stats, dataFormat='iqRaw'))

        data = yield FutureList(reqs)
        results = []
        for dat in data:
            prob = readout.iqToProbs(dat, alg.qubits)
            prob = np.squeeze(prob)[1] # P1
            results.append(prob)
        envelope = np.sqrt((results[0]-results[2])**2 + (results[1]-results[3])**2)

        results.append(envelope)
        returnValue(results)

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data

def recoverFlux(Sample, measure=0, fluxbias=None, stats=900, name='Recover Flux',
                tBuf=5*ns, save=True, noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure)

    qubit = devs[measure]
    if fluxbias is None:
        fb = qubit['biasOperate']
        fluxbias = st.r[fb['V']-0.1:fb['V']+0.1:0.002, V]

    axes = [(fluxbias, 'Bias Voltage')]
    deps = readout.genProbDeps(qubits, measure)
    kw = {"stats": stats, 'tBuf': tBuf}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, fb):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        q0['biasOperate'] = fb
        alg[gates.PiPulse([q0])]
        alg[gates.Wait([q0], tBuf)]
        alg[gates.Measure([q0])]
        alg.compile()

        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        probs = readout.iqToProbs(data, alg.qubits)
        returnValue(np.squeeze(probs))

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data

def pulseshape(Sample, measure=0, height=-1.0, pulseLen=2000*ns,
               time=None, mpaStep=0.005, minPeak=0.4, stopAt=0.3, minCount=2,
               stats=300, save=True, name='Pulse shape measurement',
               plot=True, noisy=True, update=True):
    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    if time is None:
        time = range(3, 25) + [round(10**(0.02*i)) for i in range(70, 110, 2)]
        time = np.array(time)*ns

    axes = [("Time after step", time[0].units), ("Z Offset", '')]
    deps = [("Probability |1>", "", "")]
    kw = {'step height': height, 'stats': stats, 'pulseLen': pulseLen,
          "mpaStep": mpaStep, "minPeak": minPeak, "stopAt": stopAt, "minCount": minCount}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    scanInfo = {'peakFound': False, 'lowCount': 0, 'highCount': 0}
    def sweep():
        center = 0
        for t in time:
            scanInfo['time'] = t
            scanInfo['center'] = center
            yield t, center
            low = center
            high = center
            scanInfo['lowCount'] = 0
            scanInfo['highCount'] = 0
            scanInfo['peakFound'] = False
            while ((scanInfo['lowCount'] < minCount) or
                   (scanInfo['highCount'] < minCount) or
                   (not scanInfo['peakFound'])):
                if (scanInfo['lowCount'] < minCount) or (not scanInfo['peakFound']):
                    low -= mpaStep
                    yield t, low
                if (scanInfo['highCount'] < minCount) or (not scanInfo['peakFound']):
                    high += mpaStep
                    yield t, high
            center = round(0.5*(low+high)/mpaStep)*mpaStep

    def func(server, args):
        t, ofs = args
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        q0['settlingAmplitudes'] = []
        q0['settlingRates'] = []
        alg[gates.Pulseshape([q0], offset=ofs, probeTime=t, stepHeight=height, stepLen=pulseLen)]
        alg[gates.Measure([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        probs = readout.iqToProbs(data, alg.qubits, states=[0, 1])
        prob = np.squeeze(probs)[1]
        if t == scanInfo['time']:
            if prob >= minPeak:
                scanInfo['peakFound'] = True
            side = 'highCount' if ofs > scanInfo['center'] else 'lowCount'
            if prob < stopAt:
                scanInfo[side] += 1
            else:
                scanInfo[side] = 0
        returnValue([t['ns'], ofs, prob])

    pulsedata = sweeps.run(func, sweep(), dataset=dataset, save=save, noisy=noisy)

    p, func = qpc._getstepfunc(pulsedata, height, plot=plot, ind=dataset.independents, dep=dataset.dependents)

    if update:
        Q['settlingRates'] = p[2::2]
        Q['settlingAmplitudes'] = p[1::2]/float(height)

def pulseshape2D(Sample, measure=0, height=-1.0, pulseLen=2000*ns, zpa=None,
                 time=None, stats=300, save=True, name='Pulse shape measurement',
                 plot=True, noisy=True, update=True):
    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    if time is None:
        time = range(3, 25) + [round(10**(0.02*i)) for i in range(70, 100, 2)]
        time = np.array(time)*ns

    if zpa is None:
        zpa = np.linspace(-1, 1, 51)
    axes = [("Time after step", str(time[0].units)), ("Z Offset", "")]
    deps = [("Probability |1>", "", "")]
    kw = {'step height': height, 'stats': stats, 'pulseLen': pulseLen,}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def sweep():
        for t in time:
            for z in zpa:
                yield t, z

    def func(server, arg):
        t, ofs = arg
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        q0['settlingAmplitudes'] = []
        q0['settlingRates'] = []
        alg[gates.Pulseshape([q0], offset=ofs, probeTime=t, stepHeight=height, stepLen=pulseLen)]
        alg[gates.Measure([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        probs = readout.iqToProbs(data, alg.qubits, states=[0, 1])
        prob = np.squeeze(probs)[1]
        returnValue([t['ns'], ofs, prob])

    pulsedata = sweeps.run(func, sweep(), dataset=dataset, save=save, noisy=noisy)

    p, func = qpc._getstepfunc(pulsedata, height, plot=plot, ind=dataset.independents, dep=dataset.dependents)

    if update:
        Q['settlingRates'] = p[2::2]
        Q['settlingAmplitudes'] = p[1::2]/float(height)

def zPulseTailing(Sample, measure=0, delay=st.r[0:500:5, ns], toffset=50*ns, height=-1.0, zpulseLen=2*us,
                  meas_delay=1*us, stats=1200, correction=True, name='z pulse tailing', save=True, noisy=True, plot=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure=measure)

    axes = [(delay, 'delay')]
    deps = readout.genProbDeps(qubits, measure)
    if correction:
        name += ' zpulse corrected'
    else:
        name += ' zpulse uncorrected'

    kw = {'stats': stats, 'step height': height, 'step length': zpulseLen, 'measure delay': meas_delay,
          'time offset': toffset, "z pulse correction": correction}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, currDelay):
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        if not correction:
            q0['settlingAmplitudes'] = []
            q0['settlingRates'] = []
        alg[gates.Detune([q0], zpulseLen, amp=height)]
        alg[gates.Wait([q0], currDelay)]
        alg[gates.PiHalfPulse([q0])] # X/2
        alg[gates.Wait([q0], toffset)]
        alg[gates.PiHalfPulse([q0], phase=np.pi/2)] # Y/2
        alg[gates.Wait([q0], meas_delay)]
        alg[gates.Measure([q0])]
        alg.compile()
        dat = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        probs = readout.iqToProbs(dat, alg.qubits)
        returnValue(np.squeeze(probs))

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data

def pulseTrajectory(Sample, measure=0, fraction=st.r[0.0:1.0:0.01], phase=0.0, alpha=None,
                    stats=1500L, name='Pulse Trajectory', save=True, noisy=True, plot=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure)

    axes = [(fraction, 'fraction of Pi-pulse')]
    deps = readout.genProbDeps(qubits, measure)
    ops = tomo.gen_qst_tomo_ops(tomo.octomo_names, 1)
    opList = [op for op in ops]
    kw = {'stats': stats, 'phase':phase, 'tomoOps': opList}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, fraction):
        reqs = []
        for op in opList:
            alg = gc.Algorithm(agents=devs)
            q0 = alg.q0
            alg[gates.RotPulse([q0], angle=fraction*np.pi,
                               phase=phase, alpha=alpha)]
            alg[gates.Tomography([q0], op, alpha=alpha)]
            alg[gates.Measure([q0])]
            alg.compile()
            reqs.append(runQubits(server, alg.agents, stats, dataFormat='iqRaw'))
        reqs = yield FutureList(reqs)
        data = []
        for dat in reqs:
            probs = readout.iqToProbs(dat, alg.qubits)
            data.append(np.squeeze(probs))
        data = np.vstack(data)
        returnValue(data)

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    if plot:
        probs = np.array(data)
        probs = probs[:, (1, 2)].reshape((-1, 6, 2))
        x = np.unique(data[:,0])
        plt.figure(figsize=(8, 3))
        plt.subplot(121)
        for idx, op in enumerate(opList):
            plt.plot(x, probs[:, idx, 0], label=op)
        plt.xlabel("fraction of Pi Pulse")
        plt.ylabel("P0")
        plt.subplot(122)
        for idx, op in enumerate(opList):
            plt.plot(x, probs[:, idx, 1], label=op)
        plt.xlabel("fraction of Pi Pulse")
        plt.ylabel("P0")

        rhos = [tomo.qst(prob, 'octomo') for prob in probs]
        tomography.plotTrajectory(rhos)
    return data


def ramseyFilter(Sample, measure=0, delay=st.r[0:100:1, ns], stats=2400,
                 name='Ramsey Error Filter', save=True, noisy=True, plot=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure)

    axes = [(delay, "Delay")]
    deps = readout.genProbDeps(qubits, measure)
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, currDelay):
        alg = gc.Algorithm(devs)
        q0 = alg.qubits[0]
        alg[gates.PiPulse([q0])]
        alg[gates.Wait([q0], currDelay)]
        alg[gates.PiPulse([q0])]
        alg[gates.Measure([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        probs = readout.iqToProbs(data, alg.qubits)
        returnValue(np.squeeze(probs))

    data = sweeps.grid(func, axes, dataset=dataset, noisy=noisy, save=save)

    if plot:
        plt.plot(data[:, 0], data[:, 3], '.-')
        plt.xlabel("Delay [ns]")
        plt.ylabel("P2")
    return data

def rotPopulation(Sample, measure=0, fraction=None, stats=3000, herald=False,
                  name='Rotpulse Population', save=True, noisy=True, plot=True):
    """
    measure thermal population, make sure |2> is tuned up.
    Reference: Geerlings PRL 110, 120501 (2013)
    """
    sample, devs, qubits = gc.loadQubits(Sample, measure)

    if fraction is None:
        fraction = np.linspace(-1.25, 1.25, 126)

    axes = [(fraction, "fraction of piPulse")]
    deps = [("Probability |%s>" %s, "Ref", "") for s in range(3)]
    deps += [("Probability |%s>" %s, "Sig", "") for s in range(3)]
    kw = {"stats": stats, "Herald": herald}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure, kw=kw)

    def func(server, currFrac):
        reqs = []
        for flag in [True, False]:
            alg = gc.Algorithm(devs)
            q0 = alg.qubits[0]
            if herald:
                alg[gates.Herald([q0])]
            if flag:
                # reference, g <-> e pi-pulse
                alg[gates.PiPulse([q0], state=1)]
            # when no pipulse g <-> e, signal
            # f <-> e, theta-pulse
            alg[gates.RotPulse([q0], angle=currFrac*np.pi, state=2)]
            # g <-> e, pi-pulse
            alg[gates.PiPulse([q0], state=1)]
            # measure
            alg[gates.Measure([q0])]
            alg.compile()
            reqs.append(runQubits(server, alg.agents, stats))

        ans = yield FutureList(reqs)
        probRef = np.squeeze(readout.iqToProbs(ans[0], alg.qubits, herald=herald))
        probSig = np.squeeze(readout.iqToProbs(ans[1], alg.qubits, herald=herald))
        probs = np.hstack((probRef, probSig))
        returnValue(probs)

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    if plot:
        x = data[:, 0]
        xp = np.linspace(x[0], x[-1], 10*len(x)+1)
        p0ref = data[:, 1]
        p0sig = data[:, 4]
        p_ref = [0.5*(np.max(p0ref)-np.min(p0ref)), 0.5, 0.0, 0.5*(np.max(p0ref)+np.min(p0ref))]
        p_sig = [0.5*(np.max(p0sig)-np.min(p0sig)), 0.5, 0.0, 0.5*(np.max(p0sig)+np.min(p0sig))]
        fitRef, _, func = fitting.fitCurve('cosine', x, p0ref, p_ref)
        fitSig, _, func = fitting.fitCurve('cosine', x, p0sig, p_sig)
        plt.figure()
        plt.plot(x, p0ref, '.', label='Ref', color=BLUE)
        plt.plot(x, p0sig, '.', label='Sig', color=RED)
        plt.plot(xp, func(xp, *fitRef), color=BLUE)
        plt.plot(xp, func(xp, *fitSig), color=RED)
        plt.legend()
        amp_sig = fitSig[0]
        amp_ref = fitRef[0]
        pe = amp_sig/(amp_ref+amp_sig)
        f10 = qubits[measure]['f10']['GHz']
        Teff = -0.048*f10/np.log(pe)
        print("A_Sig: %s, A_Ref: %s" %(amp_sig, amp_ref))
        print("Thermal Population %s, Effective Temp %s" %(pe, Teff))

    return data

def rabiPopulation(Sample, measure=0, rabiLen=st.r[0:100:1, ns], rabiAmp=0.2, stats=3000,
                   herald=False, name='Rabi Population', save=True, noisy=True, plot=True):
    """
    measure thermal population, make sure |2> is tuned up.
    Refence: XYJin PRL 114, 240511 (2015)
    """
    sample, devs, qubits = gc.loadQubits(Sample, measure)

    axes = [(rabiLen, "Rabi Drive Len")]
    deps = [("Probability |%s>" %s, "Ref", "") for s in range(3)]
    deps += [("Probability |%s>" %s, "Sig", "") for s in range(3)]
    kw = {"stats": stats, "RabiAmp": rabiAmp, 'Herald': herald}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, currLen):
        reqs = []
        for flag in [True, False]:
            # reference
            alg = gc.Algorithm(devs)
            q0 = alg.qubits[0]
            if herald:
                alg[gates.Herald([q0], ringdown=False)]
            if flag:
                # g <-> e pipulse, reference
                alg[gates.PiPulse([q0], state=1)]
                # when no g<->e pipulse, signal
            # e <-> f rabipulse
            alg[gates.RabiDrive([q0], amp=rabiAmp, tlen=currLen, state=2)]
            # g <-> e pipulse
            # the paper shows we should measure Pe. And we apply a pipulse to measure Pg
            alg[gates.PiPulse([q0], state=1)]
            alg[gates.Measure([q0])]
            alg.compile()
            reqs.append(runQubits(server, alg.agents, stats))

        ans = yield FutureList(reqs)
        probRef = np.squeeze(readout.iqToProbs(ans[0], alg.qubits, herald=herald))
        probSig = np.squeeze(readout.iqToProbs(ans[1], alg.qubits, herald=herald))
        returnValue(np.hstack([probRef, probSig]))

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    if plot:
        x = data[:, 0]
        xp = np.linspace(x[0], x[-1], 10*len(x)+1)
        p0ref = data[:, 1]
        p0sig = data[:, 4]
        f = fitting.maxFreq(np.vstack([x, p0ref]).T, 4000, False)["GHz"]
        p_ref = [0.5*(np.max(p0ref)-np.min(p0ref)), f, 0.0, 0.5*(np.max(p0ref)+np.min(p0ref))]
        p_sig = [0.5*(np.max(p0sig)-np.min(p0sig)), f, 0.0, 0.5*(np.max(p0sig)+np.min(p0sig))]
        fitRef, _, func = fitting.fitCurve('cosine', x, p0ref, p_ref)
        fitSig, _, func = fitting.fitCurve('cosine', x, p0sig, p_sig)
        plt.figure()
        plt.plot(x, p0ref, '.', label='Ref', color=BLUE)
        plt.plot(x, p0sig, '.', label='Sig', color=RED)
        plt.plot(xp, func(xp, *fitRef), color=BLUE)
        plt.plot(xp, func(xp, *fitSig), color=RED)
        plt.legend()
        amp_sig = fitSig[0]
        amp_ref = fitRef[0]
        pe = amp_sig/(amp_ref+amp_sig)
        f10 = qubits[measure]['f10']['GHz']
        Teff = -0.048*f10/np.log(pe)
        print("A_Sig: %s, A_Ref: %s" %(amp_sig, amp_ref))
        print("Thermal Population %s, Effective Temp %s" %(pe, Teff))

    return data


def randomizedBenchmarking(Sample, measure=0, ms=None, k=30, interleaved=False, maxtime=14*us,
                           name='SQ RB Clifford', stats=900, plot=True, save=True, noisy=False):
    """
    single Qubit RB Clifford,
    ms is a sequence for the number of gates,
    k is the repetition of each number of gates
    interleaved is the gate name, in the format ["X"], or ["X", "Y/2"]
    available gate names are
        ["I", "X", "Y", "X/2", "Y/2", "-X/2", "-Y/2", "-X", "-Y"]

    """

    sample, devs, qubits = gc.loadQubits(Sample, measure)
    rbClass = rb.RBClifford(1, False)

    if ms is None:
        rbClass.setGateLength([devs[measure]])
        m_max = rb.getlength(rbClass, maxtime, interleaved=interleaved, )
        ms = np.unique([int(m) for m in np.logspace(0, np.log10(m_max), 30, endpoint=True)])

    def getSequence(m):
        sequence = rbClass.randGen(m, interleaved=interleaved, finish=True)
        return sequence

    axesname = 'm - number of Cliffords'
    if interleaved:
        name += ' interleaved: ' + str(interleaved)
        axesname = "m - number of set of Clifford+interleaved"

    axes = [(ms, axesname), (range(k), 'sequence')]
    deps = [("Sequence Fidelity", "", "")]

    kw = {"stats": stats, "interleaved": interleaved, 'k': k, 'axismode': 'm', "maxtime": maxtime}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, currM, currK):
        print("m = {m}, k = {k}".format(m=currM, k=currK))
        gate_list = getSequence(currM)
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        alg[gates.RBCliffordSingleQubit([q0], gate_list)]
        alg[gates.Measure([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        probs = readout.iqToProbs(data, alg.qubits)
        returnValue([np.squeeze(probs)[0]])

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    if plot:
        plotRBClifford(data)

    return data

def randomizedBenchmarkingState2(Sample, measure=0, ms=None, k=30, interleaved=False, maxtime=14*us,
                           name='SQ RB Clifford State |2>', stats=900, plot=True, save=True, noisy=False):
    """
    single Qubit RB Clifford, we record the probability of 0, 1, 2 state
    ms is a sequence for the number of gates,
    k is the repetition of each number of gates
    interleaved is the gate name, in the format ["X"], or ["X", "Y/2"]
    available gate names are
        ["I", "X", "Y", "X/2", "Y/2", "-X/2", "-Y/2", "-X", "-Y"]

    """

    sample, devs, qubits = gc.loadQubits(Sample, measure)
    rbClass = rb.RBClifford(1, False)

    if ms is None:
        rbClass.setGateLength([devs[measure]])
        m_max = rb.getlength(rbClass, maxtime, interleaved=interleaved, )
        ms = np.unique([int(m) for m in np.logspace(0, np.log10(m_max), 30, endpoint=True)])

    def getSequence(m):
        sequence = rbClass.randGen(m, interleaved=interleaved, finish=True)
        return sequence

    axesname = 'm - number of Cliffords'
    if interleaved:
        name += ' interleaved: ' + str(interleaved)
        axesname = "m - number of set of Clifford+interleaved"

    axes = [(ms, axesname), (range(k), 'sequence')]
    deps = [("Sequence Fidelity", "", ""), ("Probability |1>", "", ""), ("Probability |2>", "", "")]

    kw = {"stats": stats, "interleaved": interleaved, 'k': k, 'axismode': 'm', "maxtime": maxtime}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, currM, currK):
        print("m = {m}, k = {k}".format(m=currM, k=currK))
        gate_list = getSequence(currM)
        alg = gc.Algorithm(devs)
        q0 = alg.q0
        alg[gates.RBCliffordSingleQubit([q0], gate_list)]
        alg[gates.Measure([q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        probs = readout.iqToProbs(data, alg.qubits)
        returnValue([np.squeeze(probs)])

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    if plot:
        plotRBClifford(data)

    return data


def plotRBClifford(dataset, A=None, B=None):
    ms, ks, probs = dstools.format2D(dataset)
    prob_mean = np.mean(probs, axis=0)
    prob_std = np.std(probs, axis=0)
    if A and B:
        p0 = [0.99]
    elif A:
        p0 = [0.99, np.min(prob_mean)]
    elif B:
        p0 = [0.99, np.max(prob_mean) - np.min(prob_mean)]
    else:
        p0 = [0.99, np.max(prob_mean)-np.min(prob_mean), np.min(prob_mean)]
    ans = rb.fitData(ms, prob_mean, A=A, B=B, p0=p0)
    ms_plot = np.linspace(1, np.max(ms)+1, 5*len(ms)+1)
    func = rb.fitFunc(ans['A'], ans['B'])[0]
    plt.figure()
    plt.errorbar(ms, prob_mean, yerr=prob_std, fmt='o')
    plt.plot(ms_plot, func(ms_plot, ans['p']))
    plt.xlabel("m - number of Cliffords")
    plt.ylabel("Sequence Fidelity")
    plt.title(r" $F = {A:.3f} \times {p:.5f}^m + {B:.3f} $".format(A=ans['A'], B=ans['B'], p=ans['p']))
    plt.tight_layout()
    return ans

def orbitXY(Sample, measure=0, piAmp=None, piHalfAmp=None, alpha=None, f10=None, interleaved=None,
            name='RB XY Optimization', m=30, k=10, stats=900, save=True, noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure)
    q = qubits[measure]

    if piAmp is None:
        piAmp = q['piAmp']
    if piHalfAmp is None:
        piHalfAmp = q['piHalfAmp']
    if alpha is None:
        alpha = q['alphaDRAG']
    if f10 is None:
        f10 = q['f10']

    axes = [(piAmp, "PiPulse Amplitude"), (piHalfAmp, "PiHalfPulse Amplitude"),
            (alpha, "alpha DRAG"), (f10, 'f10 frequency')]
    deps = [("Sequence Fidelity", "k=%d" %i, "") for i in range(k)]
    deps.append(("Sequence Fidelity", "average", ""))
    kw = {"m": m, 'stats': stats, 'interleaved': interleaved, 'axismode': 'm', 'k': k}

    if interleaved:
        name += " interleaved %s" %interleaved
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    rbClass = rb.RBClifford(1)
    def getSequence(m):
        sequence = rbClass.randGen(m, interleaved=interleaved, finish=True)
        return sequence

    def func(server, currPiAmp, currPiHalfAmp, currAlpha, currf10):
        reqs = []
        for i in range(k):
            print("m = {m}, k = {k} ".format(m=m, k=i))
            alg = gc.Algorithm(devs)
            q0 = alg.q0
            q0['piAmp'] = currPiAmp
            q0['piHalfAmp'] = currPiHalfAmp
            q0['alphaDRAG'] = currAlpha
            q0['f10'] = currf10
            gate_list = getSequence(m)
            alg[gates.RBCliffordSingleQubit([q0], gate_list)]
            alg[gates.Measure([q0])]
            alg.compile()
            reqs.append(runQubits(server, alg.agents, stats, dataFormat='iqRaw'))

        data = yield FutureList(reqs)

        probs = []
        for dat in data:
            prob = readout.iqToProbs(dat, alg.qubits, states=[0,1])
            probs.append(np.squeeze(prob)[0])

        probs.append(np.mean(probs))
        returnValue(probs)

    data = sweeps.grid(func, axes, dataset=dataset, save=save, noisy=noisy)

    return data


def orbitXYNM(Sample, measure=0, paramNames=['piAmp', 'piHalfAmp', 'alphaDRAG'], m=30, k=10,
              nonzdelt=0.2, zdelt=0.2, xtol=0.001, ftol=0.001, maxiter=None, maxfun=None,
              stats=1500, interleaved=None, name='RB Nelder Mead XY', noisy=True):
    sample, devs, qubits = gc.loadQubits(Sample, measure)

    params = popt.Parameters(devs, measure, paramNames=paramNames)
    if interleaved:
        name += " interleaved %s" %interleaved

    axes, deps, inputs = params.makeInputsAxesDeps(nelderMead=True)

    kw = {'stats': stats, "paramNames": paramNames, 'k': k, 'm': m, 'axismode': "m",
          "nonzdelt": nonzdelt, "zdelt": zdelt, "xtol": xtol, "ftol": ftol, 'maxiter': maxiter,
          'maxfun': maxfun, "interleaved": interleaved}

    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)
    rbClass = rb.RBClifford(1)

    def func(server, args):
        currParam = params.args2Params(args)
        alg = gc.Algorithm(devs)
        params.updateQubits(alg.agents, currParam, noisy=noisy)
        gate_list = rbClass.randGen(m, interleaved=interleaved, finish=True)
        alg[gates.RBCliffordSingleQubit([alg.q0], gate_list)]
        alg[gates.Measure([alg.q0])]
        alg.compile()
        data = yield runQubits(server, alg.agents, stats, dataFormat='iqRaw')
        probs = readout.iqToProbs(data, alg.qubits, states=[0,1])
        probs = np.squeeze(probs)
        err = 1 - probs[0]
        returnValue([err])

    funcWrapper = popt.makeFunctionWrapper(k, func, axes, deps, measure, sample, noisy=noisy)
    output = sweeps.fmin(funcWrapper, inputs, dataset, xtol=xtol, ftol=ftol, nonzdelt=nonzdelt,
                         zdelt=zdelt, maxiter=maxiter, maxfun=maxfun)

    return output


def bitReadout(Sample, measure=0, states=[0, 1], readoutFrequency=None, readoutPower=None,
               readoutLen=None, stats=6000, delay=0 * ns,
               update=True, plot=True, save=True, noisy=True):
    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, True)
    q = qubits[measure]

    if readoutFrequency is None:
        readoutFrequency = q['readoutFrequency']
    if readoutPower is None:
        readoutPower = q['readoutPower']
    if readoutLen is None:
        readoutLen = q['readoutLen']

    axes = [(range(stats), "Clicks")]
    deps = [[("I", "|%s>" % s, ""), ("Q", "|%s>" % s, "")] for s in states]
    deps = sum(deps, [])
    IQLists = []

    kw = {"stats": stats, 'states': states, 'readoutFrequency': readoutFrequency,
          'readoutPower': readoutPower, 'readoutLen': readoutLen}
    dataset = sweeps.prepDataset(sample, 'calculate Readout Centers', axes, deps, measure=measure,
                                 kw=kw)
    with pyle.QubitSequencer() as server:
        for state in states:
            print("Measuring state |%s> " % state)
            alg = gc.Algorithm(devs)
            q0 = alg.q0
            q0['readoutFrequency'] = readoutFrequency
            q0['readoutPower'] = readoutPower
            q0['readoutLen'] = readoutLen
            alg[gates.MoveToState([q0], 0, state)]
            alg[gates.Wait([q0], delay)]
            alg[gates.Measure([q0])]
            alg.compile()
            data = runQubits(server, alg.agents, stats, dataFormat='iqRaw').wait()
            IQLists.append(np.squeeze(data))

    return IQLists

    all_data = [np.array(range(stats)).reshape(-1, 1)]
    [all_data.append(np.squeeze(data)) for data in IQLists]
    all_data = np.hstack(all_data)
    all_data = np.array(all_data, dtype='float')

    if plot:
        fig = plt.figure(figsize=(6, 4.8))
        ax = fig.add_subplot(1, 1, 1, aspect='equal')
        for idx, state, color in zip(range(len(states)), states, COLORS):
            IQs = np.squeeze(IQLists[idx])
            center = centers[state]
            ax.plot(IQs[:, 0], IQs[:, 1], '.', markersize=2, color=color, alpha=0.5,
                    label='|%s>' % state)
            ax.plot([center.real], [center.imag], '*', color='k', zorder=15)
            cir1 = plt.Circle((center.real, center.imag), radius=stds[state], zorder=10,
                              fill=False, fc='k', lw=2, ls='-')
            ax.add_patch(cir1)
            # cir = plt.Circle((center.real, center.imag), radius=stds[state]*2, zorder=5,
            #                   fill=False, fc='k', lw=2, ls='--')
            # ax.add_patch(cir3)
        plt.legend()
        plt.xlabel("I [a.u.]")
        plt.ylabel("Q [a.u.]")
