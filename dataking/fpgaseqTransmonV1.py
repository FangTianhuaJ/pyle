import os
from labrad.units import Unit
from pyle import gateCompiler as gc
import pyle.envelopes as env
from pyle.dataking import envelopehelpers as eh
import numpy as np
import copy

ns, us, V, mV, GHz = [Unit(s) for s in ('ns', 'us', 'V', 'mV', 'GHz')]

# bias command names
DACNames = {'FAST': 'dac1',
            'SLOW': 'dac1slow',
            'FINE': 'dac0'}

# default channel identifiers
FLUX = lambda q: (q.__name__, 'flux')
SQUID = lambda q: (q.__name__, 'squid')
DAC = lambda q: DACNames[(q.get('biasOperateDAC', 'FAST')).upper()]

# default sram padding
# this two value should be a multiple of 4
# the start delay's have to be multiples of 4ns because the clock on the FPGA boards is 250MHz.
# Because of the way the start delay's are computed, if PREPAD isn't a multiple of 4, different xyz sequences
# could lead to different relative start times of the readout DAC and ADC, which would give you a phase shift
# in the detected signal. If this happens, you're gunna have a bad time.
PREPAD = 200
POSTPAD = 200


# debug dumps
DEBUG_PATH = os.path.join(os.path.expanduser('~'), '.packet-dump')

def adcFilter(filterLen, filterStart=0, filterFunc=None, adcLen=4024, filterAmp=128):
    if filterFunc is None:
        filterFunc = lambda N: np.kaiser(N, 2.4)
    filter_pts = int(filterLen/4)
    start_pt = int(filterStart/4)
    w = filterFunc(filter_pts)
    w /= np.max(w)
    f = np.zeros(adcLen)
    f[start_pt:start_pt+filter_pts] = w*filterAmp
    return f

def filterBytes(filterWindow):
    return filterWindow.astype("<u1").tostring()


def dataShapeTransform(dataIn):
    """
    this function is used to transform the returned data for ADC V1
    to the shape compatiable with the data for ADC V7
    when used in the demodulate mode

    In the demodulate mode:
        the shape of data for ADC V1:
        (channel, I/Q, stats), e.g. (4, 2, 300)
        the shape of data for ADC V7:
        (channel, stats, retrigger, I/Q), e.g. (12, 300, 3, 2)

    In the average mode:
        the shape of data for ADC V1 is the same as that for ADC V7
        (board in timing order, I/Q, waveform),
        e.g. (1, 2, 8192) for ADC V1
             (1, 2, 4096) for ADC V7

    the data in the shape (channel, I/Q, stats) will be transformed into
    (channel, stats, 1, I/Q), e.g. from (4, 2, 3000) -> (4, 3000, 1, 2)

    @param dataIn: the demodulate data from GHz FPGA Server
    @return: the data with the shape compatiable with ADC V7
    """
    dataIn = np.array(dataIn)
    dataIn = np.swapaxes(dataIn, -1, -2) # shape from (channel, I/Q, stats) to (channel, stats, I/Q)
    shape =  [s for s in dataIn.shape[:-1]] + [1, dataIn.shape[-1]]
    dataIn = np.reshape(dataIn, shape)
    return dataIn

def runQubits(server, devices, stats, dataFormat='iqRaw',
              debug=False, forceDualBlock=False, localDemod=False):
    """
    Wrapper for run qubits.  Allows you to set the localDemod flag to invoke
    the software demodulator.
    while the localdemod is not implemented now.
    """

    if not localDemod:
        return runQubitsNormal(server, devices, stats, dataFormat, debug, forceDualBlock)
    else:
        for q in devices:
            if q.get('adc mode', None) == 'average':
                raise RuntimeError('Cannot do local demod in average mode!')
        return runQubitsLocalDemod(server, devices, stats, dataFormat, debug, forceDualBlock)

def runQubitsLocalDemod(server, devices, stats, dataFormat=None,
        debug=False, forceDualBlock=False):
    """
    This works exactly like runQubits, but sets the boards to
    'average' (time-trace) mode, then applies a software demodulator.
    The advantage here is that you can do things the ADC boards do not
    support like demodulating multiple times in a single acquisition.
    It is, however, extremely slow as it reruns the sequence 'stats'
    times.
    """
    raise NotImplementedError("LocalDemod is not implemented yet")
    fl = []
    devices = copy.deepcopy(devices)
    for q in devices:
        if isinstance(q, gc.Transmon):
            q['adc mode'] = 'average'
    #Run the experiment as many time as we have stats
    def callback(list_result):
        iq = None # this line should be removed once the four function is implemented
        qubits = [q for q in devices if isinstance(q, gc.Transmon) and q.get('readout', False)]
        all_data = [iq.timeDomainDataToIqData(x, qubits, PREPAD) for x in list_result]
        x = np.array(all_data)
        #print "all data shape: %s " % (x.shape,)
        if dataFormat == 'iqRaw':
            result = iq.concatenate_stats(all_data)
        elif dataFormat == 'iqRawLocalDemod':
            result = iq.format_something_useful(all_data)
        else:
            result = iq.average_stats(all_data)
        x = np.array(result)
        #print "result shape: %s" % (x.shape,)
        return result

    p = runQubitsGetPacket(server, devices, 1, probs, 'iqRaw', forceDualBlock)
    #qubits = [q for q in devices if isinstance(q, gc.Transmon) and q.get('readout', False)]
    fl = [ sendPacket(p, debug) for _ in range(stats) ]
    fl = pipeline.FutureList(fl)
    fl.addCallback(callback)
    return fl

def runQubitsNormal(server, devices, stats, dataFormat='iqRaw',
              debug=False, forceDualBlock=False):
    """
    build a packet for the qubit sequencer and send it once.
    """
    p = runQubitsGetPacket(server, devices, stats, dataFormat, forceDualBlock)

    # This is stupid that this has to be here.  sendPacket needs to know the shape
    # to return fake data in "dry_run" mode, and I couldn't find a better way to get
    # it that information.  Sorry.  --ERJ
    # nqubits = np.sum([d.get('readout', False) for d in devices])
    # if any([d.get('adc mode', 0) == 'average' for d in devices]):
    #     shape = nqubits, 2, 8192
    # elif dataFormat == 'iqRaw':
    #     shape = (nqubits, 2, stats)
    # else:
    #     shape = (nqubits, 2)
    data = sendPacket(p, debug)
    if dataFormat == 'iqRaw':
        data.addCallback(dataShapeTransform)
        return data
    else:
        return data
    # return sendPacket(p, debug)

def runQubitsGetPacket(server, devices, stats, dataFormat='iqRaw', forceDualBlock=False):
    # Make a packet for the target server (qubit sequencer), add sequence data, and then send the packet.
    p = server.packet()
    nblocks = [len(dev.get('xy_s', [])) for dev in devices]

    if forceDualBlock or max(nblocks)>1:
        # print "making dual block sequence %d blocks / forceDualBlock: %s" % (max(nblocks), forceDualBlock)
        makeSequenceDualBlock(p, devices)
    else:
        makeSequence(p, devices)

    p.build_sequence()
    p.run(long(stats))
    # get the data in the desired format
    if dataFormat == 'iqRaw':               # No shift, not averaged
        p.get_data_raw(key='data')
    elif dataFormat == 'iqRawShifted':      # Shifted, not averaged
        p.get_data_raw_shifted(key='data')
    elif dataFormat == 'iq':                # No shift, averaged
        p.get_data_iq(key='data')
    elif dataFormat == 'iqShifted':         # Shifted, averaged
        p.get_data_iq_shifted(key='data')
    elif dataFormat == 'phasesRaw':         # Phases (computed with shift), not averaged
        p.get_data_raw_phases(key='data')
    elif dataFormat == 'phases':
        p.get_data_phases(key='data')
    elif dataFormat == 'raw_switches':
        p.get_data_raw_switches(key='data')
    else:
        raise Exception('dataFormat %s not recognized' %str(dataFormat))

    # Send the packet with wait=False, therefore returning a Future
    # return sendPacket(p, debug)
    return p

def adjustCableDelays(transmons):
    """
    The envelopes come in set to ideal 'qubit chip' time. Adjust all of the envelopes to
    'DAC output time' which includes the cable delays.  We have to phase shift the RR
    envelope to make the ADC happy
    """
    envelopesXYZ = []
    envelopesRR = []
    for dev in transmons:
        xy_delay = dev['timingLagWrtMaster'] + dev['timingLagUwave']
        z_delay = dev['timingLagWrtMaster']
        rr_delay =  dev['readoutDevice']['timingLagRRUwave']
        dev['xy'] = env.shift(dev.get('xy', env.NOTHING), xy_delay) # timinglagWRTMaster + timinglagUUWave
        dev['z'] = env.shift(dev.get('z', env.NOTHING), z_delay)    # timinglagWRTMaster
        for idx in range(len(dev['xy_s'])):
            dev['xy_s'][idx] = env.shift(dev['xy_s'][idx], xy_delay)
            dev['z_s'][idx] = env.shift(dev['z_s'][idx], z_delay)

        # carrierFrequency = dev['readoutDevice']['carrierFrequency']
        # df = dev['readoutFrequency'] - carrierFrequency
        # rr delay moves the readout DAC signal an arbitrary amount,
        # but can't move the ADC without changing the phase.
        # We try to change the phase to keep the carrier constant relative to the ADC.  Which is weird.
        # We comment the phase because the way we build the rr pulse is base on the start time
        # of the pulse itself, not the global zero as like building the xy-pulse
        dev['rr'] = env.shift(dev.get('rr', env.NOTHING), rr_delay) # * np.exp(-1.0j*2*np.pi*df*rr_delay)
        # We do not shift adcReadoutWindows here, for 2 reasons:
        # 1. rr_delay is small, maybe less than 4ns. If this is the case, shifting
        #    the adcReadoutWindows does nothing, since the resolution of adcReadoutWindows is 4ns
        #    When the rr_delay is large, in that case, I think this should be solved properly.
        # 2. We do not find a proper way to calibrate the rr_delay
        #                                                            -ZZX
        # if 'adcReadoutWindows' in dev:
        #     newWindows = {}
        #     for name, win in dev['adcReadoutWindows'].items():
        #         shiftedWin = (win[0] + rr_delay, win[1] + rr_delay)
        #         newWindows[name] = shiftedWin
        #     dev["adcReadoutWindows"] = newWindows

def checkTimingSingleBlock(transmons):
    envelopesRR = []
    envelopesXYZ = []
    for dev in transmons:
        envelopesXYZ.extend([dev['xy'], dev['z']])
        envelopesRR.append(dev['rr'])
    tXYZ = checkTiming(envelopesXYZ)
    tRR = checkTiming(envelopesRR)
    return tXYZ, tRR

def checkTimingDualBlock(transmons):
    envelopesRR = []
    envelopesXYZ = []
    for dev in transmons:
        envelopesXYZ.extend(dev['xy_s'])
        envelopesXYZ.extend(dev['z_s'])
        envelopesRR.append(dev['rr'])
    tXYZ = checkTiming(envelopesXYZ)
    tRR = checkTiming(envelopesRR)
    return tXYZ, tRR

def checkTimingDualBlockIdx(transmons, idx):
    envelopesXYZ = []
    for dev in transmons:
        envelopesXYZ.append(dev['xy_s'][idx])
        envelopesXYZ.append(dev['z_s'][idx])
    tXYZ = checkTiming(envelopesXYZ)
    return tXYZ

def adjustBoardStartDelays(transmons, readoutConfigs, readoutStartDelayIn):
    # print "FT: adjustBoardStartDelays"
    # print "FT: readoutStartDelayIn: %s" % readoutStartDelayIn
    for dev in transmons:
        # print "FT: device: %s" % (dev.__name__,)
        if readoutStartDelayIn < 0.0 * ns:
            # The readout is actually scheduled to start before the XYZ dacs.
            # Move their signals later so that the readout dac doesn't have to start negative.
            dev['xy'] = env.shift(dev['xy'], readoutStartDelayIn)
            dev['z'] = env.shift(dev['z'], readoutStartDelayIn)
            XYZstartDelay = -readoutStartDelayIn
            readoutStartDelayOut = 0*ns
            # print "FT: XYZstartDelay: %s" %XYZstartDelay
            # print "FT: readoutStartDelayOut: %s" %readoutStartDelayOut

        else:
            # The readout dac is supposed to start after the XYZ dacs ('normal),  Shift it earlier and
            # compensate with a start delay
            # the shift here is used for compensate the start delay,
            # in other words, we take advantage of the start delay.
            # carrierFrequency = dev['readoutDevice']['carrierFrequency']
            # df = dev['readoutFrequency'] - carrierFrequency
            # dev['rr'] = env.shift(dev['rr'], -readoutStartDelayIn) * np.exp(+1.0j*2*np.pi*df*readoutStartDelayIn)
            dev['rr'] = env.shift(dev['rr'], -readoutStartDelayIn)
            # the adcReadoutWindow should be automatically added in ReadoutGate
            # according determined by some registry keys:
            # for example, readoutWidth, readoutLen, ringdownLen...
            # that is why we should shift the window here. -ZZX
            # readoutStartDelayIn must be a multiple of 4ns when we input the parameter
            if "adcReadoutWindows" in dev:
                newWindows = {}
                for key, win in dev["adcReadoutWindows"].items():
                    shiftWin = (win[0]-readoutStartDelayIn, win[1]-readoutStartDelayIn)
                    newWindows[key] = shiftWin
                dev['adcReadoutWindows'] = newWindows
            XYZstartDelay = 0*ns
            readoutStartDelayOut = readoutStartDelayIn
            # print "FT: XYZstartDelay: %s" %XYZstartDelay
            # print "FT: readoutStartDelayOut: %s" %readoutStartDelayOut
        # dev['rr'] = dev['rr'] * np.exp(-2j*np.pi*df*dev['rr'].start)
        # print "fpgaseqTransmon: phase shifting by -%s " % (dev['rr'].end,)
        # print "fpgaseqTransmon: start phase shift would be: -%s " % (dev['rr'].start,)
        # print "fpgaseqTransmon: sequence length: %s " % (dev['rr'].end - dev['rr'].start)
        # print "setting XYZ start delay for %s to %s" % (dev.__name__, XYZstartDelay)
        dev['XYZstartDelay'] = XYZstartDelay['ns']
    adcStartDelay = readoutStartDelayOut
    for cfg in readoutConfigs:
        # print "setting DAC start delay to %s" % readoutStartDelayOut
        # print "setting ADC start delay to %s" % adcStartDelay
        cfg['readoutDACStartDelay'] = readoutStartDelayOut['ns']
        # adcTimingLag describe the time the signal propagating in the wires.
        # here I change this timing to the ADC start delay and modifies the 'adcTimingLag'
        # also, The PREPAD of DAC output is also moved to the start delay
        timingLag = cfg['adcTimingLag']['ns']
        timingLag4ns = np.floor(timingLag/4)*4
        cfg['adcTimingLag'] = (timingLag - timingLag4ns) * ns
        cfg['readoutADCStartDelay'] = adcStartDelay['ns'] + timingLag4ns + PREPAD

def makeSequence(p, devices):
    """
    Make a memory/sram sequence to be passed to the Qubit Sequencer server.

    Sequences are made in several steps
    1. Find the total time spanned by all SRAM sequences and save this information for later
    2. Initialize the qubit sequencer, telling it all of the channels we need to use.
    3. Configure external resources. For example, we must declare the microwave source frequency to
    the qubit sequencer so that it can check that shared sources have the same frequency, etc.
    4. Add memory commands to the DAC boards
    5. Add SRAM to the dac boards
    """

    # sortedDevices = sortDevices(devices)
    # transmons = sortedDevices['transmon']
    transmons = [d for d in devices if isinstance(d, gc.Transmon) ]
    readoutConfigs = [d for d in devices if isinstance(d, gc.Readout)]
    blockName = 'block0'

    # 1. Figure out timing parameters.
    # Mutate the qubit dictionary to shift by the cable delays
    # and return the time window (start,end) for the XYZ and RR
    adjustCableDelays(transmons)
    tXYZ,tRR = checkTimingSingleBlock(transmons)
    # Now we need the readout DAC to start whenever the first readout envelope shows up.
    # This is needed for e.g., T1 scans.  We do this by adding a start delay, and shifting the
    # envelope time.  However, if that delay is negative we instead need to shift the XYZ signals.

    readoutStartDelay = np.floor((tRR[0]-tXYZ[0]) / 4.0)*4*ns
    # print "readoutStartDelay (pre): ", readoutStartDelay
    # print "tXYZ: ",tXYZ
    # print "tRR: ",tRR

    # the shift of rr pulse according to the startdelay is done in adjustBoardStartDelays.
    adjustBoardStartDelays(transmons, readoutConfigs, readoutStartDelay)
    tXYZ, tRR = checkTimingSingleBlock(transmons)

    # print "tXYZ post shift: ", tXYZ
    # print "tRR post shift: ", tRR

    # 2. Construct the packet for the Qubit Sequencer
    p.initialize([(d.__name__, d['channels']) for d in devices])
    addConfig(p, devices)
    # 3. SRAM
    addSram(p, devices, tXYZ, tRR, blockName)
    # 4. Memory sequence
    p.new_mem()
    memdevices = [d for d in devices if 'flux' in dict(d['channels'])]
    addMem(p, memdevices, [blockName])
    for readoutConfig in readoutConfigs:
        # print "readout device: ", readoutConfig.__name__
        # print "ADC/ DAC start delays", readoutConfig['readoutADCStartDelay'], readoutConfig['readoutDACStartDelay']
        qubits = [t for t in transmons if t['readoutConfig']==readoutConfig.__name__
                    and t.get('readout', False)]
        # print "FT: qubits to be readout",str([q.__name__ for q in qubits])
        # print "ADC start delay: %s" % (extra_delay+adcStartDelay)
        addADC(p, qubits, readoutConfig['readoutADCStartDelay'])
        # print 'adcStartDelay', adcStartDelay

def makeSequenceDualBlock(p, devices):
    #print "makeSequenceDualBlock"
    transmons = [d for d in devices if isinstance(d, gc.Transmon) ]
    readoutConfigs = [d for d in devices if isinstance(d,gc.Readout)]
    nblocks = len(devices[0].get('xy_s'))
    blockNames = [ "block%d" % n for n in range(nblocks) ]
    adjustCableDelays(transmons)
    tXYZ,tRR = checkTimingDualBlock(transmons)
    readoutStartDelay = np.floor((tRR[0]-tXYZ[0]) / 4.0)*4*ns
    adjustBoardStartDelays(transmons, readoutConfigs, readoutStartDelay)
    tXYZ,tRR = checkTimingDualBlock(transmons)
    p.initialize([(d.__name__, d['channels']) for d in devices])
    addConfig(p, devices)
    #3. SRAM
    addSramDualBlock(p, devices, tXYZ, tRR, blockNames)
    #4. Memory sequence
    p.new_mem()
    memdevices = [d for d in devices if 'flux' in dict(d['channels'])]
    addMem(p, memdevices, blockNames)

    for readoutConfig in readoutConfigs:
        #print "readout device: ", readoutConfig.__name__
        #print "ADC/ DAC start delays", readoutConfig['readoutADCStartDelay'], readoutConfig['readoutDACStartDelay']

        qubits = [t for t in transmons if t['readoutConfig']==readoutConfig.__name__
                    and t.get('readout', False)]
        #print "ADC start delay: %s" % (extra_delay+adcStartDelay)
        addADC(p, qubits, readoutConfig['readoutADCStartDelay'])
        #print 'adcStartDelay', adcStartDelay


def sendPacket(p, debug):
    """Finalize a packet for the Qubit Sequencer, send it, and add callback to get data.

    Also sets up debugging if desired to log the packet being sent to the
    Qubit Sequencer, as well as the packet sent from the Qubit Sequencer to the GHz DACs.


    """
    if debug:
        fname = os.path.join(DEBUG_PATH, 'qubitServerPacket.txt')
        with open(fname, 'w') as f:
            print >>f, p
        p.dump_sequence_packet(key='pkt')

    # send the request
    req = p.send(wait=False)

    if debug:
        def dump(result):
            from pyle.dataking.qubitsequencer import prettyDump
            fname = os.path.join(DEBUG_PATH, 'ghzDacPacket.txt')
            with open(fname, 'w') as f:
                print >>f, prettyDump(result['pkt'])
            return result
        req.addCallback(dump)

    # add a callback to unpack the data when it comes in
    req.addCallback(lambda result: result['data'])
    return req


def checkTiming(envelopes):
    """Calculate the timing interval to encompass a set of envelopes.

    RETURNS:
    tuple of (tStart, tEnd)
    """
    start, end = env.timeRange(envelopes)
    if start is not None and end is None:
        raise Exception('sequence has start but no end')
    elif start is None and end is not None:
        raise Exception('sequence has end but no start')
    elif start is None and end is None:
        t = 0, 40 # default time range
    else:
        t = start, end
    return t


def addConfig(p, devices, autotrigger='S3'):
    """Add config information to a Qubit Sequencer packet.

    Config information includes:
        - which qubits to read out (timing order)
        - microwave source settings (freq, power)
        - preamp settings (offset, polarity, etc.)
        - settling rates for analog channels
        - autotrigger
    """
    p.new_config()
    # Set up timing order. This is the list of devices to be read out.
    transmons = [d for d in devices if isinstance(d, gc.Transmon)]
    readoutConfigs = [d for d in devices if isinstance(d,gc.Readout)]
    timing_order = []
    for rc in readoutConfigs:
        qubits = [t for t in transmons if t['readoutConfig']==rc.__name__ and t.get('readout',False)]
        for idx, q in enumerate(qubits):
            if q['adc mode'] == 'demodulate':
                # print "adding %s to timing order" % ((rc.__name__, 'readout-%s::%d' % (q.__name__, idx)),)
                timing_order.append((rc.__name__, 'readout-%s::%d' % (q.__name__, idx)))
            elif q['adc mode'] == 'average':
                # print "adding %s to timing order" % ((rc.__name__, 'readout-%s' % q.__name__),)
                timing_order.append((rc.__name__, 'readout-%s' % q.__name__))
            else:
                raise Exception('Demodulator and average mode are the only readout modes')
    p.config_timing_order(timing_order)
    # Set up each device's drive microwave source, readout microwave
    # source, and z pulse line settling rates
    for d in devices:
        channels = dict(d['channels'])
        if 'uwave' in channels:
            p.config_microwaves((d.__name__, 'uwave'), d['fc'], d['uwavePower'])
        if 'signal' in channels:
            p.config_microwaves((d.__name__, 'signal'), d['carrierFrequency'], d['carrierPower'])
        if 'pump' in channels:
            p.config_microwaves((d.__name__, 'pump'), d['pumpFrequency'], d['pumpPower'])
        if 'settlingRates' in d and 'meas' in channels:
            p.config_settling((d.__name__, 'meas'), d['settlingRates']*GHz,  d['settlingAmplitudes'])

    # build paramp packets for readout configs
    addSetupPumpPackets(p, readoutConfigs)
    # Add trigger pulse to SRAM sequence on the channel specified by autotrigger
    if autotrigger is not None:
        p.config_autotrigger(autotrigger)

def addSetupPumpPackets(p, devices):
    """
    build setup packets for pump channel. The pump channel is for paramp (JPA, IMPA, JTWPA...).
    The microwave source for the pump channel does not connect to any GHzDACs, thus we should
    control the microwave source individually.

    It is totally a hack. But I can not find a better way to do this.

    the registry key used here is "pumpChannels",
    describing the server and the address of microwave source.
    e.g. pumpChannels = ("Hittite T2100 Server", "DR GPIB Bus - GPIB0::10")
    """
    _ctx = p._kw.get("context", (0L, 2L))
    _ctx = (long(p._server.ID), long(_ctx[1]))
    packets = list()
    states = list()
    flag =  False
    for d in devices:
        if 'pumpChannels' in d:
            flag = True
            server_name, device_name = d['pumpChannels']
            records = (("Select Device", device_name), ("Amplitude", d['pumpPower']),
                       ("Frequency", d['pumpFrequency']), ("output", d.get("pumpOutput", True)))
            state = "{name}: f={freq}, p={power}, {output}".format(
                name=device_name, freq=d['pumpFrequency'],
                power=d['pumpPower'], output=d.get("pumpOutput", True)
            )
            packets.append((_ctx, server_name, records))
            states.append(state)
    setup_packets = (states, tuple(packets))
    if flag:
        # setup_packets should not be an empty packets
        p.config_setup_packets(setup_packets)

def addSram(p, devices, tXYZ, tRR, blockName):
    """Add SRAM data to a Qubit Sequencer packet.

    INPUTS

    p: packet for the (qubit sequencer) server to which we add the
    commands to add SRAM data.

    devices: list of device (qubit) objects that have sequences

    tXYZ: time range of form (tStart, tEnd) in nanoseconds.

    idx: For interlaced SRAM sequences. (removed, ERJ)

    We input the data for a set of sram blocks.  Currently, at most two
    blocks are supported, which will make use of the split sram feature.
    The first block will be prepadded and the last block postpadded to
    prevent aliasing, or if only one block is given, then that block will
    be padded on both ends.

    Note that the sequences are also shifted relative to the time intervals
    to compensate for various delays, as controlled by the 'timingLagUwave'
    and 'timingLagWrtMaster' parameters.  Because of this shift, when using
    dual-block SRAM, you should make sure that both time intervals are
    given with at least enough padding already included to handle any shifts
    due to timing delays.
    """

    # Get a dictionary in which ADC names key lists of qubits using that
    # ADC
    # sortedDevices = sortDevices(devices)
    transmons = [d for d in devices if isinstance(d, gc.Transmon) ]
    # transmons = sortedDevices['transmon']
    readoutConfigs = [d for d in devices if isinstance(d,gc.Readout)]
    # Essentially four things happen in the rest of this function:
    # 1. We compute timing needed for various SRAM channels and create the
    # frequency samples that will be needed to evaluate the envelope
    # data in the frequency domain.
    # 2. We compute the start delay for resonator readout and write
    # the data to the qubit sequencer packet.
    # 3. We compute the xy pulses, shifting in time to account for
    # timing lags, and then write it to the qubit sequencer packet.
    # 4. We add SRAM bits to provide a 4ns trigger pulse at the start
    # of the SRAM on all channels.

    # 1. Determine total time span of sequence, including pre and post
    # padding, then compute frequency samples.
    tStart = min(tXYZ[0],tRR[0])
    tEnd = max(tXYZ[1],tRR[1])
    time = (PREPAD + tEnd - tStart + POSTPAD)
    # xy gets complex freqs, z gets real freqs
    fxy, fz = env.fftFreqs(time)
    frr = fxy
    # notice that len(np.fft.fftfreq(n)) == n
    # len(fxy) should equal time if time is the power of 2.
    # p.new_sram_block(blockName, len(fxy)) #XXX Why do we use len(fxy) and not len(fz)?
    # 2. XY and Z control data
    # print "block name: %s" % (blockName,)
    addSramXYZ(p, transmons, fxy, fz, blockName)
    addSramReadout(p, transmons, readoutConfigs, frr, blockName)

def addSramXYZ(p, transmons, fxy, fz, blockName):
    # generate sram according to GHzDAC,
    # as different qubits may share a single GHzDAC
    xy_seqs = {}
    z_seqs = {}
    for d in transmons:
        channels = dict(d['channels'])
        if 'uwave' in channels:
            xy_board =  tuple(channels['uwave'][1]) # iq board has only one output
            xy = xy_seqs.get(xy_board, env.NOTHING)
            xy_seqs[xy_board] = xy + d.get('xy', env.NOTHING)
        if 'meas' in channels:
            z_board = tuple(channels['meas'][1]) # analog board has two output
            z = z_seqs.get(z_board, env.ZERO)
            z_seqs[z_board] = z + d.get('z', env.ZERO)

    for d in transmons:
        channels = dict(d['channels'])
        if 'uwave' in channels:
            t0 = -PREPAD
            # xy = d.get('xy', env.NOTHING)
            xy_board = tuple(channels['uwave'][1])
            xy = xy_seqs.get(xy_board, env.NOTHING)
            # Add data to the currently selected block
            dacStartDelayClockCycles = int(d['XYZstartDelay']/4)
            p.new_sram_block(blockName, len(fxy), (d.__name__, 'uwave'))
            p.sram_iq_data_fourier((d.__name__, 'uwave'), xy(fxy, fourier=True), t0*ns)
            p.set_start_delay((d.__name__,'uwave'), dacStartDelayClockCycles)
        if 'meas' in channels:
            t0 = -PREPAD
            # z = d.get('z', env.ZERO)
            z_board = tuple(channels['meas'][1])
            z = z_seqs.get(z_board, env.ZERO)
            # if there is an xy drive, but nothing on the z a small dc offset is observed.
            # That's why it's changed to env.zero. RB.

            # add constant offset in z, if specified
            # This is incompatible with repeated measurements on a single qubit.
            # If you need the extra dynamic range figure out how to fix it.  -- ERJ

            # if 'zOffsetDC' in d:
            #    #rectangular window is a bad idea, considering deconv
            #    z += env.flattop(t0 + PREPAD/2, tXYZ[1]+extra_delay/2. -(t0 + PREPAD/2),
            #                     10 , amp=d['zOffsetDC']) # 4 ns should be decent
            #    if abs(d['zOffsetDC'])>0. and abs(extra_delay)==0.0:
            #        print ('You have a DC offset, but no delay between last XYZ pulse and readout.'
            #               ' Set readoutStartDelay to a nonzero value')

            dacStartDelayClockCycles = int(d['XYZstartDelay']/4)
            p.new_sram_block(blockName, len(fxy), (d.__name__, 'meas'))
            p.sram_analog_data_fourier((d.__name__,'meas'), z(fz, fourier=True), t0*ns)
            p.set_start_delay((d.__name__,'meas'), dacStartDelayClockCycles)

def addSramReadout(p, transmons, readoutConfigs, frr, blockName):
    # Readout data
    for rc in readoutConfigs:
        qubits = [d for d in transmons if d['readoutConfig']==rc.__name__]
        channels = dict(rc['channels'])
        rr = env.NOTHING
        # extra delay must be a multiple of 4 ns, or Bad Things (tm)
        # might happen.  I think it actually doesn't matter, but I
        # am being cautious -- ERJ

        # Start delay must be an integer number of cycles because qubit sequence doesn't know how to shift envelopes
        dacStartDelayClockCycles = int(rc['readoutDACStartDelay']/4)
        for q in qubits:
            rr = rr + q.get('rr', env.NOTHING)
            # print "qubit {} readout start/end: ({}, {})".format(q.__name__,
            #         q.get('rr', env.NOTHING).start, q.get('rr', env.NOTHING).end)

        # The following line is of dubious value. It puts a single time shift on the entire readout envelope.
        # In principle, it could/should be per-qubit, and implemented in gate compiler.
        t0  = -PREPAD
        if 'signal' in channels:
            p.new_sram_block(blockName, len(frr), (rc.__name__, 'signal'))
            p.sram_iq_data_fourier((rc.__name__, 'signal'), rr(frr, fourier = True), t0*ns)
            # print("setting readout start delay to {} clock cycles or {} ns".format(
            #     dacStartDelayClockCycles, dacStartDelayClockCycles*4))
            p.set_start_delay((rc.__name__, 'signal'), dacStartDelayClockCycles)


def deconvolveAndChop(cxn, data, board, minpad=2000, dac=None, freq=None, blockNum=None):
    if dac is not None and freq is not None:
        raise Exception("Can't have frequency and dac")
    if not (dac or freq):
        raise Exception("Must have either freq or dac")
    leadingVal = data[0]
    trailingVal = data[-1]
    total_size = data.shape[0] + 2*minpad
    total_size = int(2**np.ceil(np.log2(total_size)))
    left_pad = (total_size - data.shape[0])//2
    right_pad = (total_size - left_pad - data.shape[0])
    data = np.hstack((leadingVal*np.ones(left_pad), data, trailingVal*np.ones(right_pad)))
    calibration = cxn.dac_calibration
    p = calibration.packet()
    p.board(board)
    p.loop(True)
    if dac is not None:
        #This is analog mode
        p.dac(dac)
        if blockNum is not None:
            #The deconv server enforces zero values at the start and end of the signal. This is for ensuring that the dac end output is reproducible.
            #However, for dualblock we need to change that:
            if blockNum==0:
                #first block: first element is zero, last is nonzero
                #pass
                p.borderValues([0.0,data[-1]]) #This is needed to enforce first and last value on the dacs
            else:
                #last block: first element is non zero, last is zero
                #pass
                p.borderValues([data[0],0.0]) #This is needed to enforce first and last value on the dacs
        #result = {}
        #result['corrected'] = p.correct(data)
        p.correct(data, key='corrected')
        result = p.send()
        correctedData = result['corrected'].asarray*1.0/(2**13)
        correctedDataOld=correctedData
        correctedData = correctedData[left_pad:-right_pad]
        correctedData[0:4] = correctedDataOld[0:4];correctedData[-4:] = correctedDataOld[-4:] #the first and last 4 values are set by the deconvolver, this remove oscillations when idling. We need to copy these values after clipping.
    else:
        #This is IQ mode
        p.frequency(freq)
        p.correct(data, key='corrected')
        result = p.send()
        correctedI,correctedQ = result['corrected']
        correctedData = (correctedI.asarray + 1.0j*correctedQ.asarray)*1.0/(2**13)
        correctedDataOld=correctedData
        correctedData = correctedData[left_pad:-right_pad]
        correctedData[0:4] = correctedDataOld[0:4];correctedData[-4:] = correctedDataOld[-4:] #the first and last 4 values are set by the deconvolver, this remove oscillations when idling. We need to copy these values after clipping.
    # plt.figure()
    # plt.plot(correctedData*1.0,label='corrected')
    # plt.plot(data,label='padded data')
    # plt.plot(correctedData[padpts:-padpts]*1.0,label='truncated')
    # plt.legend(loc='upper left')
    # plt.grid()
    return(correctedData)

def addSramXYZ_TD(p, transmons, tXYZ, blockNum, blockName):
    # Don't pad the gap between block 0 and block 1
    prepad = PREPAD if blockNum == 0 else 0*ns
    postpad = POSTPAD if blockNum == 1 else 0*ns
    t0 = np.floor(tXYZ[0])*ns-prepad # start time must be exact nanosecond
    npts = np.ceil((tXYZ[1]*ns+postpad - t0)['ns']) # round up to nanosecond
    t = np.arange(npts) + t0['ns']
    #print "sram XYZ block %s time range: (%s, %s) / npts %s" % (blockName, t0, t[-1]+1, npts)

    for d in transmons:
        channels = dict(d['channels'])
        if 'uwave' in channels:
            t0 = -PREPAD
            # XXX should throw error if t[0] < t0
            xy = d['xy_s'][blockNum]
            #Add data to the currently selected block
            dacStartDelayClockCycles = int(d['XYZstartDelay']/4)
            p.new_sram_block(blockName, len(t), (d.__name__, 'uwave'))
            board = channels['uwave'][1][0]
            data = deconvolveAndChop(p._server._cxn, xy(t, fourier=False), board, freq=d['fc'])
            p.sram_iq_data((d.__name__, 'uwave'), data, False)
            p.set_start_delay((d.__name__,'uwave'), dacStartDelayClockCycles)
        if 'meas' in channels:
            z = d['z_s'][blockNum]
            dacStartDelayClockCycles = int(d['XYZstartDelay']/4)
            p.new_sram_block(blockName, len(t), (d.__name__, 'meas'))
            _,(board,dac) = channels['meas']
            data = deconvolveAndChop(p._server._cxn, z(t, fourier=False), board, dac=dac,blockNum=blockNum)
            p.sram_analog_data((d.__name__,'meas'), data, False)
            p.set_start_delay((d.__name__,'meas'), dacStartDelayClockCycles)
    return t[0], t[-1]+1

def addSramReadout_TD(p, transmons, readoutConfigs, tRR, blockNum, blockName):
    prepad = PREPAD
    postpad = POSTPAD
    t0 = np.floor(tRR[0]-prepad)
    npts = np.ceil(tRR[1]+postpad - t0)
    t = np.arange(npts) + t0
    #print "adding SRAM for readout block %d(%s).  t0=%s, npts=%s" % (blockNum, blockName, t0, npts)
    #print "sram readout time range: (%s, %s) / npts %s" % (t0, t[-1]+1, npts)
    for readoutConfig in readoutConfigs:
        qubits = [d for d in transmons if d['readoutConfig']==readoutConfig.__name__]
        channels = dict(readoutConfig['channels'])
        rr = env.NOTHING
        # extra delay must be a multiple of 4 ns, or Bad Things (tm)
        # might happen.  I think it actually doesn't matter, but I
        # am being cautious -- ERJ
        #print "readout device: ", readoutConfig.__name__
        # Start delay must be an integer number of cycles because qubit sequence doesn't know how to shift envelopes
        dacStartDelayClockCycles = int(readoutConfig['readoutDACStartDelay']/4)
        for q in qubits:
            rr = rr + q.get('rr', env.NOTHING)
            #print "qubit %s readout start/end: (%s, %s)" % (q.__name__, q.get('rr', env.NOTHING).start, q.get('rr', env.NOTHING).end)
        # The following line is of dubious value.  It puts a single time shift on the entire readout envelope.
        #  In principle, it could/should be per-qubit, and implemented in gate compiler.
        if 'signal' in channels:
            p.new_sram_block(blockName, len(t), (readoutConfig.__name__, 'signal'))
            readoutBoard = channels['signal'][1][0]
            raw_wfm =  rr(t, fourier=False)
            readout_wfm = deconvolveAndChop(p._server._cxn, raw_wfm, readoutBoard, freq=readoutConfig['carrierFrequency'])
            p.sram_iq_data((readoutConfig.__name__, 'signal'), readout_wfm, False)
            p.set_start_delay((readoutConfig.__name__, 'signal'), dacStartDelayClockCycles)

def addSramDualBlock(p, devices, tXYZ, tRR, blockNames):
    transmons = [d for d in devices if isinstance(d, gc.Transmon) ]
    readoutConfigs = [d for d in devices if isinstance(d,gc.Readout)]
    tXYZ_block0 = checkTimingDualBlockIdx(transmons, 0)
    tXYZ_block1 = checkTimingDualBlockIdx(transmons, 1)

    start0, end0 = addSramXYZ_TD(p, transmons, tXYZ_block0, 0, blockNames[0])
    start1, end1 = addSramXYZ_TD(p, transmons, tXYZ_block1, 1, blockNames[1])
    dual_block_delay = start1 - end0

    dt = (tRR[1] - tRR[0])*ns
    time = (PREPAD + tRR[1]*ns-tRR[0]*ns + POSTPAD)
    frr, _ = env.fftFreqs(time['ns'])
    addSramReadout(p, transmons, readoutConfigs, frr, blockNames[0])
    #addSramReadout_TD(p, transmons, readoutConfigs, tRR, 0, blockNames[0])
    tRR_block1_fake = (np.round(tRR[1])+dual_block_delay, np.round(tRR[1])+dual_block_delay+100)
    addSramReadout_TD(p, transmons, readoutConfigs, tRR_block1_fake, 1, blockNames[1])
    dual_block_delay = start1 - end0
    p.sram_dual_block_delay(dual_block_delay*ns)


def addMem(p, devices, blocks):
    """Add memory commands to a Qubit Sequencer packet.

    The sequence consists of resetting all qubits then setting
    them to their operating bias.  Next, the SRAM is called,
    then the qubits are read out.  Finally, all DC lines
    are set to zero.
    The sequence is:

    1. Set bias lines to zero
    2. Go to operation bias and wait until the qubit with the longest
       settling time has settled.
    3. Call the SRAM.
    4. Go to zero for some time
    """
    # Add memory delay to all channels - why?
    p.mem_delay(4.3*us)
    # 1. Set bias to zero and use default (4.3us) delay
    # p.mem_bias([(FLUX(q), FAST,  0*V) for q in qubits])
    # 2. Go to operating point
    if devices:
        operate_settle = max(q['biasOperateSettling'][us] for q in devices) * us
        p.mem_bias([(FLUX(q), DAC(q), q['biasOperate'][V]*V) for q in devices], operate_settle)
    # 3. Call SRAM
    p.mem_sync_delay()
    p.mem_call_sram(*blocks)
    p.mem_sync_delay()
    # 3a. Give each board a specific post-SRAM delay
    # p.mem_delay_single(FLUX(qubits[0]), 4.3*us)
    # 4. memory delay
    p.mem_delay(4.3*us)
    # 5. Finally, set everything to zero
    # if devices:
    #     p.mem_bias([(FLUX(q), DAC(q), 0*V) for q in devices])

filters = { 'square': np.ones,
            'kaiser24': lambda N: np.kaiser(N, 2.4),
            "hamming": lambda N: np.hamming(N)
}


def windowBytes(qubits):
    """
    check the window parameters, and make sure that only one window
    configuration is used for qubits.
    @param qubits: the qubits should share one readoutDevice
    @return: the bytes of the window function
    """
    ADC_WINDOW_TIMESTEP = 4.0 # 4.0 ns
    ADC_FILTER_SIZE = 4024 # For 16096 ns
    starts = list()
    ends = list()
    types = [q.get("readoutWindowFunc", "square") for q in qubits]
    if len(set(types)) > 1:
        raise Exception("The window function should be the same for one readoutDevice")
    for q in qubits:
        if len(q['adcReadoutWindows']) > 1:
            raise Exception("Only One ReadoutWindow is allowed for ADC V1")
        win = q['adcReadoutWindows'].values()[0]
        start, end = win
        tlen = end['ns'] - start['ns']
        # start = q['readoutDevice']['adcTimingLag']['ns'] + start['ns']
        start = q['readoutDevice']['adcTimingLag']['ns'] + start['ns']
        end = start + tlen
        starts.append(start)
        ends.append(end)
    filterStart = min(starts)
    filterEnd = max(ends)
    filterType = types[0]
    if filterType not in filters:
        raise RuntimeError('ADC filter type {} is unknown'.format(filterType))

    start_idx = int(filterStart/ADC_WINDOW_TIMESTEP)
    filterLen = filterEnd - int(start_idx*ADC_WINDOW_TIMESTEP)

    window = np.zeros(ADC_FILTER_SIZE)
    filter_pts = int(round(filterLen / ADC_WINDOW_TIMESTEP))
    filter_pts = min(ADC_FILTER_SIZE, filter_pts)
    filter_vals = filters[filterType](filter_pts)
    window[start_idx:start_idx+filter_pts] = filter_vals
    window = window * 128.0 / np.max(window)
    return filterBytes(window)

def addADC(p, qubits, startDelay):
    """
    Add ADC configration
    the readoutDevice in qubits should be the same
    """

    # This needs to be fixed for multi-qubit readout.
    # probably in filter window, since filterBytes should be the same for one ADC
    # but different filterBytes maybe used here. - ZZX
    # Now it should be fixed. -2017.01.23, ZZX
    assert(startDelay >= 0)

    window_str = windowBytes(qubits)
    for idx, q in enumerate(qubits):
        channel_id = (q['readoutDevice'].__name__, "readout-%s"%q.__name__)
        if q['adc mode'] == 'demodulate':
            p.adc_set_mode(channel_id, (q['adc mode'], idx))
            # print "setting ADC mode for %s::%d to %s" % (channel_id, idx, q['adc mode'])
            adc_phase = q['adc demod phase']
            p.adc_demod_phase(channel_id, (q['readoutFrequency'] - q['readoutDevice']['carrierFrequency'], adc_phase))
            p.adc_set_trig_magnitude(channel_id, q['adc sinAmp'], q['adc cosAmp'])
            p.adc_set_filter_function(channel_id, window_str, q['adcFilterStretchLen'], q['adcFilterStretchAt'])
        else:
            p.adc_set_mode(channel_id, q['adc mode'])
            # print "setting ADC mode for channel %s to %s" % (channel_id, q['adc mode'])
        start_delay_clock_cycles = int(np.floor(startDelay/4.0))    # this 4 corresponds to 4ns per clock cycle
        p.set_start_delay(channel_id, start_delay_clock_cycles)
        # print "setting start delay for channel %s to %s" % (channel_id, start_delay_clock_cycles)
