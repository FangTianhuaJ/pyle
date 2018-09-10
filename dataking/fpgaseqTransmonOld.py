import os

from labrad.units import Unit
ns, us, V, mV, GHz = [Unit(s) for s in ('ns', 'us', 'V', 'mV', 'GHz')]

import pyle.envelopes as env
from pyle.dataking import envelopehelpers as eh
import numpy as np

# default channel identifiers
FLUX = lambda q: (q.__name__, 'flux')
SQUID = lambda q: (q.__name__, 'squid')

# bias command names
FAST = 'dac1'
SLOW = 'dac1slow'

# default sram padding
PREPAD = 1000
POSTPAD = 50


# debug dumps 
DEBUG_PATH = os.path.join(os.path.expanduser('~'), '.packet-dump')

def filterBytes(filterLen = 16384*ns):
    filter_len = round(filterLen['ns']/4)
    filt = np.zeros(filter_len,dtype='<u1')
    filt = filt+128
    return filt.tostring()
    
def sortDevices(devices):
    phaseQubits=[]
    transmons=[]
    resonators=[]
    for device in devices:
        deviceType = device['_type']
        if deviceType == 'phaseQubit':
            phaseQubits.append(device)
        elif deviceType == 'resonator':
            resonators.append(device)
        elif deviceType == 'transmon':
            transmons.append(device)
        else:
            raise Exception('Device type %s not recognized' %deviceType)
    return {'phaseQubits':phaseQubits, 'resonators':resonators, 'transmons':transmons}
    
def sortqubits_ADC(devices):
    """Get a dictionary where ADC names key a list of devices that use that ADC"""
    groups = {}
    for d in devices:
        channels = dict(d['channels'])
        ADCname = channels['readout ADC'][1][0]
        if ADCname in groups.keys():
            groups[ADCname].append(d)
        else:
            groups[ADCname] = [d]
    return groups
    
def sortqubits_readoutDAC(qubits):
    groups = {}
    for q in qubits:
        channels = dict(q['channels'])
        readoutDACname = channels['readout uwave'][1][0]
        if readoutDACname in groups.keys():
            groups[readoutDACname].append(q)
        else:
            groups[readoutDACname] = [q]
    return groups
    
    
def runQubits(server, devices, stats, probs=None, dataFormat=None,
              debug=False):
    """A generic sequence for running multiple devices with the Qubit Sequencer.
    qubits
    These parameters control the sequence and how it should be run:
    
    server - labrad server to which we send the sequence (Qubit Sequencer).
    devices - list of dict-like objects (typically loaded from the registry)
        containing the configuration parameters of all devices involved in
        the sequence.
    stats - the number of repetitions of the experiment to be run.
    
    
    These parameters control how the data should be processed before it is
    returned to the caller (note that this processing is done by the Qubit
    Sequencer server for speed; see the server documentation for more info): 
    
    dataFormat - string description of how data should be returned
    
    DEPRICATED DATA FORMAT FLAGS
    probs - a list of which probabilities should be returned.  For example,
        when only measuring a single qubit this should be [1] since we only
        care about the |1>-state probability.
    raw - a flag indicating whether to return raw switching data (converted
        to microseconds) or processed probabilities (the default).
    separate - a flag indicating whether to return separate probabilities
        for each qubit (N possibilities), or combined probabilities of the
        multi-qubit states (2**N posibilities; the default).
    """
    if dataFormat is None:
        dataFormat = 'probs'
    if dataFormat is not 'probs' and probs is not None:
        raise Exception('Data format has been specified as %s, but probs was also specified. This is not consistent.' %dataFormat)
    sortedDevices = sortDevices(devices)
    #Make a packet for the target server (qubit sequencer), add sequence data, and then send the packet.
    p = server.packet()
    makeSequence(p, devices)
    p.build_sequence()
    p.run(long(stats))
    
    # get the data in the desired format
    if dataFormat == 'iq':
        p.get_data_raw(key='data')
    elif dataFormat == 'phases':
        p.get_data_raw_phases(key='data')
    elif dataFormat == 'raw_switches':
        p.get_data_raw_switches(key='data')
    elif dataFormat == 'probs_separate':
        p.get_data_probs_separate(probs, key='data')
    elif dataFormat == 'probs':
        p.get_data_probs(probs, key='data')
    else:
        raise Exception('dataFormat %s not recognized' %str(dataFormat))

    #Send the packet with wait=False, therefore returning a Future
    return sendPacket(p, debug)


def runInterlaced(server, qubits, stats, probs=None, raw=False, separate=False, debug=False):
    # TODO: handle generic interlacing (not just defferent SRAM blocks)
    raise Exception('not implemented yet')


def runInterlacedSRAM(server, qubits, stats, probs=None, raw=False, separate=False, debug=False):
    """A generic sequence for running multiple qubits with the Qubit Sequencer.
    
    These parameters control the sequence and how it should be run:
    
    server - labrad server to which we send the sequence (Qubit Sequencer).
    qubits - list of dict-like objects (typically loaded from the registry)
        containing the configuration parameters of all qubits involved in
        the sequence.
    stats - the number of repetitions of the experiment to be run.
    
    
    These parameters control how the data should be processed before it is
    returned to the caller (note that this processing is done by the Qubit
    Sequencer server for speed; see the server documentation for more info): 
    
    probs - a list of which probabilities should be returned.  For example,
        when only measuring a single qubit this should be [1] since we only
        care about the |1>-state probability.
    raw - a flag indicating whether to return raw switching data (converted
        to microseconds) or processed probabilities (the default).
    separate - a flag indicating whether to return separate probabilities
        for each qubit (N possibilities), or combined probabilities of the
        multi-qubit states (2**N posibilities; the default).
    """
    numSRAMs = set([len(q['xy']) for q in qubits if 'xy' in q] + [len(q['z']) for q in qubits if 'z' in q])
    if len(numSRAMs) > 1:
        raise Exception('all qubits must have same number of xy a z sequences')
    numSRAM = numSRAMs.pop()
    
    p = server.packet()
    makeSequenceInterlaced(p, qubits, numSRAM)
    p.build_sequence()

    p.run(long(stats))
    
    # get the data in the desired format with deinterlacing enabled
    if raw: p.get_data_raw_microseconds(numSRAM, key='data')
    elif separate: p.get_data_probs_separate(probs, numSRAM, key='data')
    else: p.get_data_probs(probs, numSRAM, key='data')
    
    return sendPacket(p, debug)


def runDualblock(server, qubits, stats, probs=None, raw=False, separate=False, debug=False):
    # TODO: handle dual-block SRAM (need another argument)
    raise Exception('not implemented yet')


def makeSequence(p, devices): 
    """Make a memory/sram sequence to be passed to the Qubit Sequencer server.
    
    Sequences are made in several steps
    1. Find the total time spanned by all SRAM sequences and save this information for later
    2. Initialize the qubit sequencer, telling it all of the channels we need to use.
    3. Configure external resources. For example, we must declare the microwave source frequency to
    the qubit sequencer so that it can check that shared sources have the same frequency, etc.
    4. Add memory commands to the DAC boards
    5. Add SRAM to the dac boards
    """
    #Note on 1:
    #Check total xyz (SRAM) envelope length, taking all devices into account. Devices with no 'xy', 'z',
    #or 'rr' defined have their envelope default to env.NOTHING
    #We need to get the lengths of two different sets of envelopes:
    # The coherent operations, X, Y, Z
    # The readout signals, RR
    #The reason we treat these seperately is that transmon sequences are
    #long so if we were two put the XYZ and RR sequences together we
    #woudl exceed the size of the SRAM. Instead, we reference the
    #readout signals to their own time coordinate system and use a start
    #delay in the readout DAC and ADC to make sure the readout occurs
    #after the coherent manipulations.
    
    sortedDevices = sortDevices(devices)
    transmons = sortedDevices['transmons']
    groups_ADC = sortqubits_ADC(transmons)
    #1. Figure out timing parameters.
    envelopesXYZ = []
    envelopesRR = []
    for dev in devices:
        envelopesXYZ.extend([dev.get('xy',env.NOTHING),dev.get('z',env.NOTHING)])
        envelopesRR.extend([dev.get('rr', env.NOTHING)])
    tXYZ = checkTiming(envelopesXYZ)
    tRR =  checkTiming(envelopesRR)
    #ADC start delay is set to make the ADC start demodulation when the readout pulses begin.
    #See Dan's notebook. #5 pg 41
    measureStartDelay = max([dev.get('measureStartDelay', None) for dev in devices])
    if measureStartDelay is not None:
        adcStartDelay = measureStartDelay
    else:
        adcStartDelay = (tXYZ[1] - tRR[0])*ns
    #2. Construct the packet for the Qubit Sequencer
    p.initialize([(d.__name__, d['channels']) for d in devices])
    addConfig(p, devices)
    #3. SRAM
    addSram(p, devices, [tXYZ], [tRR], ['block0'])   
    #4. Memory sequence
    p.new_mem()
    memqubits = [d for d in devices if 'flux' in dict(d['channels'])]
    addMem(p, memqubits, ['block0'])
    #TODO: wtf is this?
    for key in groups_ADC.keys():
        addADC(p, groups_ADC[key], adcStartDelay)


def makeSequenceInterlaced(p, qubits, numSRAM):
    """Make a memory/sram sequence using the Qubit Sequencer server.
    
    This version allows for interlacing with multiple SRAM blocks.
    """
    
    p.initialize([(q.__name__, q['channels']) for q in qubits])
    addConfig(p, qubits)
    p.new_mem() # call this once, even if we use multiple blocks
    memqubits = [q for q in qubits if 'flux' in dict(q['channels'])]
    
    for i in range(numSRAM):
        # calculate time range from given control envelopes
        xys = [q['xy'][i] for q in qubits if 'xy' in q]
        zs = [q['z'][i] for q in qubits if 'z' in q]
        t = checkTiming(xys + zs)
        
        block = 'block%d' % i
        addMem(p, memqubits, [block])
        addSram(p, qubits, [t], [block], idx=i)


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
    #Set up timing order. This is the list of devices to be read out.
    timing_order = []
    for d in devices:
        if (d.get('adc mode', None) == 'demodulate') and (d.get('readout', False)):
            timing_order.append(d.__name__+ '::' + str(d['adc channel']))
        elif (d.get('adc mode', None) == 'average') and (d.get('readout', False)):
            timing_order.append(d.__name__)
        elif d.get('readout', False):
            raise Exception('Demodulator mode is the only supported readout mode at this time.')
    p.config_timing_order(timing_order)
    #Set up each device's drive microwave source, readout microwave
    #source, and z pulse line settling rates
    for d in devices:
        channels = dict(d['channels'])
        if 'uwave' in channels:
            p.config_microwaves((d.__name__, 'uwave'), d['fc'], d['uwavePower'])
        if 'readout uwave' in channels:
            p.config_microwaves((d.__name__, 'readout uwave'), d['readout fc'], d['readout uwPower'])
        if 'settlingRates' in d:
            p.config_settling(d.__name__, d['settlingRates'], d['settlingAmplitudes'])
    #Add trigger pulse to SRAM sequence on the channel specified by autotrigger
    if autotrigger is not None:
        p.config_autotrigger(autotrigger)


def addSram(p, devices, tsXYZ, tsRR, blocks, idx=None):
    """Add SRAM data to a Qubit Sequencer packet.
    
    INPUTS
    
    p: packet for the (qubit sequencer) server to which we add the
    commands to add SRAM data.
    
    devices: list of device (qubit) objects that have sequences
    
    ts: list of time ranges. Each range is in the form (tStart, tEnd)
    in nanoseconds.
    
    blocks: SRAM blocks to use for this SRAM
    
    idx: I have no idea.
    
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
    #Get a dictionary in which ADC names key lists of qubits using that
    #ADC
    sortedDevices = sortDevices(devices)
    transmons = sortedDevices['transmons']
    groups_ADC = sortqubits_ADC(transmons)
    #Essentially four things happen in the rest of this function:
    #1. We compute timing needed for various SRAM channels and create the
    #frequency samples that will be needed to evaluate the envelope
    #data in the frequency domain.
    #2. We compute the start delay for resonator readout and write
    #the data to the qubit sequencer packet.
    #3. We compute the xy pulses, shifting in time to account for
    #timing lags, and then write it to the qubit sequencer packet.
    #4. We add SRAM bits to provide a 4ns trigger pulse at the start
    #of the SRAM on all channels.
    #Loop over SRAM blocks
    for i, (tXYZ, tRR, block) in enumerate(zip(tsXYZ, tsRR, blocks)):    #In single block operation this only executes once with i=0
        #1. Determine total time span of sequence, including pre and post
        #padding, then compute frequency samples.
        prepad = PREPAD if (i == 0) else 0 # only prepad the first block
        tStart = min(tXYZ[0], tRR[0])
        tEnd = max(tXYZ[1], tRR[1])
        postpad = POSTPAD if (i == len(tsXYZ)-1) else 0 # only postpad the last block
        time = (prepad + tEnd-tStart + postpad)*ns
        #xy gets complex freqs, z gets real freqs
        fxy, fz = env.fftFreqs(time['ns'])
        frr = fxy
        p.new_sram_block(block, len(frr))
        #2. Resonator readout data
        for adcGroup, qubits in groups_ADC.items():
            rr = env.NOTHING
            #Assert that all qubits in this ADC group have the same readout DAC
            if len(sortqubits_readoutDAC(qubits)) is not 1:
                raise Exception('DAC setup error: All qubits connected to the same ADC should be connected to the same DAC.')
            measureStartDelay = max([q.get('measureStartDelay', None) for q in qubits])
            if measureStartDelay is not None:
                dacStartDelay = measureStartDelay
            else:
                dacStartDelay = (tXYZ[1] - tRR[0])*ns
            #XXX This is a bullshit hack due to an error in the qubit sequencer.
            dacStartDelayClockCycles = round(dacStartDelay['ns']/4)*us
            #XXX There's a problem here. t0 is determined by the last qubit over which we loop.
            #Really, the readout dealy lag should be determined in only one place.
            #The right way to fix this is to make the readout resonator it's own device.
            for q in qubits:
                ofs = q['timingLagRRUwave']
                t0  = -ofs - prepad
                rr = rr + q.get('rr', env.NOTHING)
                p.sram_iq_data_fourier((q.__name__, 'readout uwave'), rr(frr, fourier = True), float(t0))
                p.set_start_delay((q.__name__, 'readout uwave'), dacStartDelayClockCycles)
                #2a. Turn on the paramp pump during readout
                #Same start delay as the readout signal rr.
                if q.get('readoutParampEnable', False):
                    pa = q.get('pa', env.NOTHING)
                    p.sram_iq_data_fourier((q.__name__, 'readoutParampPump'), pa(frr, fourier = True), float(t0))
        #3. XY and Z control data
        for d in transmons:
            channels = dict(d['channels'])
            if 'uwave' in channels:
                ofs = d['timingLagWrtMaster']+d['timingLagUwave']
                t0 = -ofs - prepad
                # XXX should throw error if t[0] < t0
                xy = d.get('xy', env.NOTHING)
                if idx is not None and xy is not env.NOTHING:
                    xy = xy[idx]
                p.sram_iq_data_fourier((d.__name__, 'uwave'), xy(fxy, fourier=True), float(t0))
            if 'meas' in channels:
                ofs = d['timingLagWrtMaster']
                t0 = -ofs - prepad
                # XXX should throw error if t[0] < t0
                z = d.get('z', env.NOTHING)
                if idx is not None and z is not env.NOTHING:
                    z = z[i]
                # add constant offset in z, if specified
                # FIXME: this might break if PRE- or POSTPAD is 0.  Need to check how such sequences are deconvolved
                if 'zOffsetDC' in d:
                    z += env.rect(t0 + prepad/2, time - prepad/2 - postpad/2, d['zOffsetDC'])
                p.sram_analog_data_fourier(d.__name__, z(fz, fourier=True), float(t0))
            #4. Add trigger data (the qubit sequencer actually does this as of July 2012)
            if 'trigger' in channels:
                # TODO: support trigger channels
                #I think this is currently done in the qubit sequencer (DTS 4 Aug 2012)
                pass


def addMem(p, qubits, blocks):
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
    operate_settle = max(q['biasOperateSettling'][us] for q in qubits) * us
    #Add memory delay to all channels - why?
    p.mem_delay(4.3*us)
    #1. Set bias to zero and use default (4.3us) delay
    p.mem_bias([(FLUX(q), FAST,  0*V) for q in qubits])
    #2. Go to operating point
    p.mem_bias([(FLUX(q), FAST, q['biasOperate'][V]*V) for q in qubits], operate_settle)
    #3. Call SRAM
    p.mem_call_sram(*blocks)
    #3a. Give each board a specific post-SRAM delay
    #p.mem_delay_single(FLUX(qubits[0]), 4.3*us)
    #4. Finally, set everything to zero
    p.mem_bias([(FLUX(q), FAST, 0*V) for q in qubits])


def addADC(p, qubits, startDelay):
    """ Add ADC configration"""
    
    for q in qubits:
        if q['adc mode'] == 'demodulate':
            p.adc_set_mode(q.__name__, (q['adc mode'], q['adc channel']))
        else:
            p.adc_set_mode(q.__name__, q['adc mode'])
        if q['adc mode'] == 'demodulate':
            adc_phase = q['adc demod phase']+q['adc adjusted phase']
            p.adc_demod_phase(q.__name__, (q['adc demod frequency'], adc_phase))
            p.adc_set_trig_magnitude(q.__name__, q['adc sinAmp'], q['adc cosAmp'])
            p.config_critical_phase(q.__name__, q['readoutCriticalPhase'])
            p.reverse_critical_phase_comparison(q.__name__, q['criticalPhaseGreater'])
            p.set_iq_offsets(q.__name__, q['readoutIqOffset'])
            p.adc_set_filter_function(q.__name__, filterBytes(q['adc filterLen']), q['adc filterStretchLen'], q['adc filterStretchAt'])
    start_delay_clock_cycles = round(startDelay['ns']/4)*us    #This is a bullshit hack to deal with an error in the qubit sequencer
    p.set_start_delay((q.__name__, 'readout ADC'), start_delay_clock_cycles)
