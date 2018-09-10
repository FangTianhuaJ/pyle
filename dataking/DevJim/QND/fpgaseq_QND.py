import os

from labrad.units import Unit
ns, us, V, mV, GHz = [Unit(s) for s in ('ns', 'us', 'V', 'mV', 'GHz')]

import pyle.envelopes as env
#from pyle.dataking import envelopehelpers as eh
import numpy as np


# default channel identifiers
FLUX = lambda q: (q.__name__, 'flux')
SQUID = lambda q: (q.__name__, 'squid')
COUPLERFLUX = lambda q: (q.__name__, 'couplerflux')

# bias command names
FAST = 'dac1'
SLOW = 'dac1slow'

# default sram padding
PREPAD = 250
POSTPAD = 50


# debug dumps 
DEBUG_PATH = os.path.join(os.path.expanduser('~'), '.packet-dump')


def filterBytes(filterLen = 16384*ns, filterWindowStart= 0*ns, filterWindowEnd=16384*ns):
    if filterWindowStart['ns'] < 0 or filterWindowStart['ns'] > filterLen['ns']:
        filterWindowStart = 0*ns
    if filterWindowEnd['ns'] < 0 or filterWindowEnd['ns'] > filterLen['ns']:
        filterWindowEnd = filterLen
    filter_len = round(filterLen['ns']/4)
    filt = np.zeros(filter_len,dtype='<u1')
    indexStart = filterWindowStart['ns']/4
    indexEnd = filterWindowEnd['ns']/4 +1
    filt[indexStart:indexEnd] = 128
    return filt.tostring()
    

def checkChannel(qubit, name):
    """Finds out if a ADC is used by qubit"""
    channelNames = dict(qubit['channels']).keys()
    if name in channelNames:
        return True
    else:
        return False
    

def readoutType(qubit):
    """Finds out what type of readout circuit is used by qubit"""
    channelNames = dict(qubit['channels']).keys()
    readouts=[]
    if 'squid' in channelNames:
        readouts.append('squid')
    elif 'readout ADC' in channelNames:
        readouts.append('resonator')
    else:
        raise Exception('No readout type recognized')
    if len(readouts)>1:
        raise Exception('More than one readout type detected on qubit %s' %qubit.__name__)
    return readouts[0]

def readoutTypeExclusive(qubits):
    """Assumes only one type of readout per experiment
    
       This function really needs to be phased out. The system should be able to
       handle cases with multiple qubits, all with different readout circuit types
       """
    for q in qubits:
        if q['readoutType'] == 'squid':
            return 'squid'
        elif q['readoutType'] == 'resonator':
            return 'resonator'
        else:
            raise Exception('Try an existing readout method ')

def sortDevices(devices):
    phaseQubits=[]
    resonators=[]
    for device in devices:
        deviceType = device['_type']
        if deviceType == 'phaseQubit':
            phaseQubits.append(device)
        elif deviceType == 'resonator':
            resonators.append(device)
        else:
            raise Exception('Device type %s not recognized' %deviceType)
    return {'phaseQubits':phaseQubits, 'resonators':resonators}

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
    
def runQubits(server, devices, stats, probs=None, raw=False, separate=False, switches=False, dataFormat=None, debug=False):
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
    sortedDevices = sortDevices(devices)
    phaseQubits = sortedDevices['phaseQubits']
    
    p = server.packet()
    makeSequence(p, devices)
    p.build_sequence()
    p.run(long(stats))
    
    # get the data in the desired format
    readtype = readoutTypeExclusive(phaseQubits)
    if readtype is 'squid':
        if dataFormat == 'raw':
            p.get_data_raw(key='data')
        elif (dataFormat == 'raw_microseconds' or raw):
            p.get_data_raw_microseconds(key='data')
        elif (dataFormat == 'raw_switches' or switches):
            p.get_data_raw_switches(key='data')
        elif (dataFormat == 'probs_separate' or separate):
            p.get_data_probs_separate(probs, key='data')
        elif dataFormat == 'probs':
            p.get_data_probs(probs, key='data')
        else:
            raise Exception('dataFormat %s not recognized' %str(dataFormat))
    elif readtype is 'resonator': 
        if (dataFormat == 'iq' or raw):
            p.get_data_raw(key='data')
        elif dataFormat == 'phases':
            p.get_data_raw_phases(key='data')
        elif dataFormat == 'raw_switches':
            p.get_data_raw_switches(key='data')
        elif (dataFormat == 'probs_separate' or separate):
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
    1. First we add anything to the SRAM sequences that we need but wasn't added by the user.
    For example, the phase qubits with resonator readout need to have a readout pulse in the SRAM
    of the DAC controlling the readout resonator. We add that pulse here.
    2. Find the total time spanned by all SRAM sequences and save this information for later
    3. Initialize the qubit sequencer, telling it all of the channels we need to use.
    4. Configure external resources. For example, we must declare the microwave source frequency to
    the qubit sequencer so that it can check that shared sources have the same frequency, etc.
    5. Add memory commands to the DAC boards
    6. Add SRAM to the dac boards
    """
    sortedDevices = sortDevices(devices)
    phaseQubits = sortedDevices['phaseQubits']
    groups_ADC = sortqubits_ADC([q for q in phaseQubits if checkChannel(q, 'readout ADC')] )

    envelopes = []
    for dev in devices:
        envelopes.extend([dev.get('xy',env.NOTHING),dev.get('z',env.NOTHING),dev.get('cz',env.NOTHING),dev.get('rr',env.NOTHING)])
    t = checkTiming(envelopes)
    # xys = [q['xy'] for q in phaseQubits if 'xy' in q]
    # zs = [q['z'] for q in phaseQubits if 'z' in q]
    # czs = [q['cz'] for q in phaseQubits if 'cz' in q]
    # rs = [q['r'] for q in phaseQubits if 'r' in q]
    # start, end = env.timeRange(xys + zs+czs+rs)
    # if start is not None and end is None:
        # raise Exception('sequence has start but no end')
    # elif start is None and end is not None:
        # raise Exception('sequence has end but no start')
    # elif start is None and end is None:
        # t = 0, 40 # default time range
    # else:
        # t = start, end

    
    
    # construct the packet for the Qubit Sequencer
    p.initialize([(d.__name__, d['channels']) for d in devices])
    addConfig(p, devices)
    p.new_mem()
    #checked to here
    memqubits = [q for q in phaseQubits if 'flux' in dict(q['channels'])]
    addMem(p, memqubits, ['block0'])
    addSram(p, devices, [t], ['block0'])   
    for key in groups_ADC.keys():
        addADC(p, groups_ADC[key])


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
    """Calculate the timing interval to encompass a set of envelopes."""
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
    timing_order = []
    for d in devices:
        # need to make sure it's a ADC readout even if the 'adc mode' key is there
        if (d.get('readoutType', None) == 'resonator'):  
            if (d.get('adc mode', None)=='demodulate'):
                timing_order.append((d.__name__,  'readout ADC::' + str(d['adc channel'])))
            else:
                timing_order.append((d.__name__, 'readout ADC'))
        elif (d.get('readoutType', None) == 'squid'):           
            timing_order.append((d.__name__, 'timing'))
            #timing_order.append((d.__name__, 'timing::'+str(d['adc channel'])))
        else:
            raise Exception('Readout type is not identified')
    p.config_timing_order(timing_order)
    for d in devices:
        channels = dict(d['channels'])
        if 'uwave' in channels:
            p.config_microwaves((d.__name__, 'uwave'), d['fc'], d['uwavePower'])
        if 'readout uwave' in channels:
            p.config_microwaves((d.__name__, 'readout uwave'), d['readout fc'], d['readout uwPower'])
        if 'squid' in channels:
            if 'squidPreampConfig' in d:
                p.config_preamp(d.__name__, *d['squidPreampConfig'])
            if 'squidSwitchIntervals' in d:
                p.config_switch_intervals(d.__name__, d['squidSwitchIntervals'])
        if 'settlingRates' in d:
            p.config_settling((d.__name__,'meas'), d['settlingRates'], d['settlingAmplitudes'])
            if 'couplerfluxZ' in channels:
                p.config_settling((d.__name__,'couplerfluxZ'), d['settlingRates'], d['settlingAmplitudes'])
    if autotrigger is not None:
        p.config_autotrigger(autotrigger)


def addSram(p, devices, ts, blocks, idx=None):
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
    sortedDevices = sortDevices(devices)
    phaseQubits = sortedDevices['phaseQubits']
    resonators = sortedDevices['resonators']
    groups_ADC = sortqubits_ADC([q for q in phaseQubits if checkChannel(q, 'readout ADC')])
    
    readtype = readoutTypeExclusive(phaseQubits)
    for i, (t, block) in enumerate(zip(ts, blocks)):
        #Determine total time span of sequence, including pre and post padding,
        #then compute frequency samples.
        prepad = PREPAD if (i == 0) else 0 # only prepad the first block
        postpad = POSTPAD if (i == len(ts)-1) else 0 # only postpad the last block
        time = prepad + t[1]-t[0] + postpad
        fxy, fz = env.fftFreqs(time)
        frr = fxy
        fcz = fz
        if readtype is 'squid':
            p.new_sram_block(block, len(fxy))            
        if readtype is 'resonator':
            #frr = env.fftFreqs(time)
            p.new_sram_block(block, len(frr))
        for key in groups_ADC.keys():
            qubits = groups_ADC[key]
            q0 = qubits[0]            
            rr = env.NOTHING            
            if len(sortqubits_readoutDAC(qubits)) is not 1:
                raise Exception('DAC setup error: All qubits connected to the same ADC should be connected to the same DAC.')
            for q in groups_ADC[key]:
                channels = dict(q['channels'])                                
                if 'readout uwave' in channels:
                    ofs = q['timingLagWrtMaster']+q['timingLagRRUwave']
                    t0  = -ofs - prepad
                    rr = q.get('rr', env.NOTHING)        
            p.sram_iq_data_fourier((q0.__name__, 'readout uwave'), rr(frr, fourier = True), float(t0))
            start_delay = round(q0['readout DAC start delay']['ns']/4)*us     # this 4 corresponds to 4ns per clock cycle
            p.set_start_delay((q0.__name__, 'readout uwave'), start_delay)    
        for d in phaseQubits+resonators:
            channels = dict(d['channels'])
            if 'uwave' in channels:
                ofs = d['timingLagWrtMaster']+d['timingLagUwave']
                t0 = -ofs - prepad
                # XXX should throw error if t[0] < t0
                xy = d.get('xy', env.NOTHING)
                if idx is not None and xy is not env.NOTHING:
                    xy = xy[idx]
                p.sram_iq_data_fourier((d.__name__, 'uwave'), xy(fxy, fourier=True), float(t0))
            if 'readout uwave' in channels and 'squid' in channels:
                ofs = d['timingLagWrtMaster']+d['timingLagRRUwave']
                t0  = -ofs - prepad
                rr = d.get('rr', env.NOTHING)
                p.sram_iq_data_fourier((d.__name__, 'readout uwave'), rr(frr, fourier = True), float(t0))
				#now readout DAC is used as a drive and we don't need to set the readout DAC start delay
            if 'meas' in channels:
                ofs = d['timingLagWrtMaster']
                t0 = -ofs - prepad
                # XXX should throw error if t[0] < t0
                z = d.get('z', env.NOTHING)
                #if idx is not None and z is not env.NOTHING:
                    #z = z[i]
                # add constant offset in z, if specified
                # FIXME: this might break if PRE- or POSTPAD is 0.  Need to check how such sequences are deconvolved
                #if 'zOffsetDC' in d:
                    #z += env.rect(t0 + prepad/2, time - prepad/2 - postpad/2, d['zOffsetDC'])
                p.sram_analog_data_fourier((d.__name__,'meas'), z(fz, fourier=True), float(t0))
            if 'couplerfluxZ' in channels:
                ofs = d['timingLagWrtMaster'] + d['timingLagCZ']
                t0 = -ofs -prepad
                cz = d.get('cz',env.NOTHING)
                p.sram_analog_data_fourier((d.__name__,'couplerfluxZ'),cz(fcz,fourier=True),float(t0))				
            if 'trigger' in channels:
                # TODO: support trigger channels
                pass


def addMem(p, qubits, blocks):
    """Add memory commands to a Qubit Sequencer packet.
    
    The sequence consists of resetting all qubits then setting
    them to their operating bias.  Next, the SRAM is called,
    then the qubits are read out.  Finally, all DC lines
    are set to zero.
    """
    operate_settle = max(q['biasOperateSettling'][us] for q in qubits) * us
    
    p.mem_delay(4.3*us)
    #Run through qubit reset procedure
    resetQubits(p, qubits)
    #Go to qubit (and SQUID) operating point
    p.mem_bias([(FLUX(q), FAST, q['biasOperate'][V]*V) for q in qubits] +
                   [(SQUID(q), FAST, q['squidBias'][V]*V) for q in qubits if checkChannel(q, 'squid')] +[(COUPLERFLUX(q), FAST, q['couplerfluxBias'][V]*V) for q in qubits], operate_settle)
		   				   
    #Call SRAM
    p.mem_call_sram(*blocks)
    #Readout
    readoutQubits(p, qubits)
    #Finally, set everything to zero
    p.mem_bias([(FLUX(q), FAST, 0*V) for q in qubits if checkChannel(q, 'squid')] +
				[(SQUID(q), FAST, 0*V) for q in qubits] +
				[(COUPLERFLUX(q), FAST, 0*V) for q in qubits])


def addMemSquidHeat(p, qubits, blocks):
    """Add memory commands to a Qubit Sequencer packet.
    
    The sequence consists of resetting all qubits then setting
    them to their operating bias.  Next, the SRAM is called,
    then the qubits are read out.  Finally, all DC lines
    are set to zero.
    """
    operate_settle = max(q['biasOperateSettling'][us] for q in qubits) * us
    heat_duration = max(q['squidheatDuration'][us] for q in qubits) * us
    heat_settle = max(q['squidheatSettling'][us] for q in qubits) * us
    
    p.mem_delay(4.3*us)
    resetQubits(p, qubits)
    p.mem_bias([(SQUID(q), FAST, q['squidheatBias'][mV]*mV) for q in qubits], heat_duration)
    p.mem_bias([(SQUID(q), FAST, 0*V) for q in qubits], heat_settle)
    p.mem_bias([(FLUX(q), FAST, q['biasOperate'][V]*V) for q in qubits] +
               [(SQUID(q), FAST, q['squidBias'][V]*V) for q in qubits], operate_settle)
    p.mem_call_sram(*blocks)
    readoutQubits(p, qubits)
    p.mem_bias([(FLUX(q), FAST, 0*V) for q in qubits] +
               [(SQUID(q), FAST, 0*V) for q in qubits])

               
def addADC(p, qubits):
    """ Add ADC configration
    
    """
    for q in qubits:
        if q['adc mode'] == 'demodulate':p.adc_set_mode(q.__name__, (q['adc mode'], q['adc channel']))
        else: p.adc_set_mode(q.__name__, q['adc mode'])
        if q['adc mode'] == 'demodulate':
            adc_phase = q['adc demod phase']+q['adc adjusted phase']
            p.adc_demod_phase(q.__name__, (q['adc demod frequency'],adc_phase))
            p.adc_set_trig_magnitude(q.__name__, q['adc sinAmp'], q['adc cosAmp'])
            p.config_critical_phase(q.__name__, q['adc critical phase'])
            #p.adc_set_filter_function(q.__name__, filterBytes(q['adc filterLen']), q['adc filterStretchLen'], q['adc filterStretchAt'])   
            p.adc_set_filter_function(q.__name__, filterBytes(q['adc filterLen'],q['adc filterWindowStart'],q['adc filterWindowEnd']), q['adc filterStretchLen'], q['adc filterStretchAt'])               
        start_delay = round(q['adc_start_delay']['ns']/4)*us    # this 4 corresponds to 4ns per clock cycle, and there is a conversion error between ns and us
        p.set_start_delay((q.__name__, 'readout ADC'), start_delay)

# def addADC(p, qubits):
    # """ Add ADC configration
    
    # """
    # for q in qubits:
        # if q['adc mode'] != qubits[0]['adc mode']:
            # raise Exception('adc mode need to be set to the same for all qubtits connected to the same adc')
        # p.adc_set_mode(q.__name__, q['adc mode'])
        # if q['adc_start_delay'] != qubits[0]['adc_start_delay'] or q['adc filterLen'] != qubits[0]['adc filterLen'] or q['adc filterStretchLen'] != qubits[0]['adc filterStretchLen'] or q['adc filterStretchAt'] != qubits[0]['adc filterStretchAt']:
            # raise Exception('adc start delay, filterLen, filterStretchLen or filterSretchAt need to set to equal numbers for all the qubits using the same ADC')                  
        # if q['adc mode'] == 'demodulate':
            # adc_phase = q['adc demod phase']+q['adc adjusted phase']
            # p.adc_demod_phase(q.__name__, q['adc channel'],(q['adc demod frequency'],adc_phase))
            # p.adc_set_trig_magnitude(q.__name__, q['adc channel'], q['adc sinAmp'], q['adc cosAmp'])
            # p.config_critical_phases(q.__name__, (q['adc channel'], q['adc critical phase']))
    # if qubits[0]['adc mode'] == 'demodulate':
            # p.adc_set_filter_function(q.__name__, filterBytes(qubits[0]['adc filterLen'],qubits[0]['adc filterWindowStart'],qubits[0]['adc filterWindowEnd']), qubits[0]['adc filterStretchLen'], qubits[0]['adc filterStretchAt'])                   
    # start_delay = round(qubits[0]['adc_start_delay']['ns']/4)*us          # this 4 corresponds to 4ns per clock cycle
    # p.set_start_delay((q.__name__, 'readout ADC'), start_delay)

def resetQubits(p, qubits):
    """Add commands to reset a list of qubits.
    
    The resets are performed simultaneously on all qubits, with
    the reset settling time chosen to be the maximum reset settling
    time for any of the qubits.
    """
    reset_settle = max(q['biasResetSettling'][us] for q in qubits) * us
    for q in qubits:
        if not isinstance(q['biasReset'], list):
            q['biasReset'] = [q['biasReset']]
    reset_steps = max(len(q['biasReset']) for q in qubits)
    for q in qubits:
        shortfall = reset_steps - len(q['biasReset']) #shortfall>=0
        if shortfall:
            q['biasReset'].extend([q['biasOperate']] * shortfall)
    for i in xrange(reset_steps):
        p.mem_bias([(FLUX(q), FAST, q['biasReset'][i][V]*V) for q in qubits], reset_settle)


def readoutQubits(p, qubits):
    """Add commands to readout a list of qubits with SQUID ramps.
    
    The SQUID ramps are performed in groups, based on the value of
    squid_readout_delay for each qubit.
    """
    measure_settle = max(q['biasReadoutSettling'][us] for q in qubits) * us
 
    # in QND readout, biasReadout should be the same as the bias Operate
    p.mem_bias([(FLUX(q), FAST, q['biasReadout'][V]*V) for q in qubits] + 
               [(SQUID(q), FAST, 0*V) for q in qubits], measure_settle)
    
    # build groups based on readout_delay
    readout = [q for q in qubits if q.get('readout', False)]
    no_readout = [q for q in qubits if not q.get('readout', False)]
    readout_groups = {}
    for q in readout:
        delay = q['squidReadoutDelay'][us]
        if delay not in readout_groups:
            readout_groups[delay] = []
        readout_groups[delay].append(q)
        
    # readout each qubit group in sequence
    # along with the first group, we run the timers for all qubits
    # that are not actually being read out, just to make sure the timers run
    time = 0
    first = True
    for delay in sorted(readout_groups.keys()):
        group = readout_groups[delay]
        if delay > time:
            p.mem_delay((delay - time)*us)
        ramp_len = max(q['squidRampLength'][us] for q in group if checkChannel(q, 'squid'))
        time += ramp_len + 10
        if first:
            timers = group + no_readout
            first = False
        else:
            timers = group
        p.mem_start_timer([q.__name__ for q in timers if checkChannel(q, 'squid')])
        p.mem_bias([(SQUID(q), FAST, q['squidRampBegin'][V]*V) for q in group if checkChannel(q, 'squid')])
        p.mem_bias([(SQUID(q), SLOW, q['squidRampEnd'][V]*V) for q in group if checkChannel(q, 'squid')], ramp_len*us)
        p.mem_stop_timer([q.__name__ for q in timers if checkChannel(q, 'squid')])
        p.mem_bias([(SQUID(q), FAST, q['squidReset'][V]*V) for q in group if checkChannel(q, 'squid')])
        p.mem_bias([(SQUID(q), FAST, 0*V) for q in group if checkChannel(q, 'squid')])    
    p.mem_bias([(FLUX(q), FAST, q['biasReadout'][V]*V) for q in qubits if checkChannel(q, 'resonator')], measure_settle)
"""
def readoutQubits(p, qubits):

    measure_settle = max(q['biasReadoutSettling'][us] for q in qubits) * us
 
    # in QND readout, biasReadout should be the same as the bias Operate
    for q in qubits:
        if checkChannel(q, 'squid'):
            p.mem_bias([(FLUX(q), FAST, q['biasReadout'][V]*V) for q in qubits] + 
                       [(SQUID(q), FAST, 0*V) for q in qubits], measure_settle)
            
            # build groups based on readout_delay
            readout = [q for q in qubits if q.get('readout', False)]
            no_readout = [q for q in qubits if not q.get('readout', False)]
            readout_groups = {}
            for q in readout:
                delay = q['squidReadoutDelay'][us]
                if delay not in readout_groups:
                    readout_groups[delay] = []
                readout_groups[delay].append(q)
                
            # readout each qubit group in sequence
            # along with the first group, we run the timers for all qubits
            # that are not actually being read out, just to make sure the timers run
            time = 0
            first = True
            for delay in sorted(readout_groups.keys()):
                group = readout_groups[delay]
                if delay > time:
                    p.mem_delay((delay - time)*us)
                ramp_len = max(q['squidRampLength'][us] for q in group)
                time += ramp_len + 10
                if first:
                    timers = group + no_readout
                    first = False
                else:
                    timers = group
                p.mem_start_timer([q.__name__ for q in timers])
                p.mem_bias([(SQUID(q), FAST, q['squidRampBegin'][V]*V) for q in group])
                p.mem_bias([(SQUID(q), SLOW, q['squidRampEnd'][V]*V) for q in group], ramp_len*us)
                p.mem_stop_timer([q.__name__ for q in timers])
                p.mem_bias([(SQUID(q), FAST, q['squidReset'][V]*V) for q in group])
                p.mem_bias([(SQUID(q), FAST, 0*V) for q in group])
        elif checkChannel(q,'resonator'):     
            p.mem_bias([(FLUX(q), FAST, q['biasReadout'][V]*V) for q in qubits], measure_settle)
            
        else:
            raise Exception('No readout channel detected')

"""