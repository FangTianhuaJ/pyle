import labrad
from labrad.units import Unit,Value
import numpy as np
mK = Unit('mK')

def loadDeviceType(sample ,deviceType, write_access=False):
    Devices=[]
    devices=[]
    deviceNames = sample['config']
    #First get writeable Devices
    for deviceName in deviceNames:
        if sample[deviceName]['_type'] == deviceType:
            Devices.append(sample[deviceName])
    #Now make the unwritable devices
    sample = sample.copy()
    for deviceName in deviceNames:
        if sample[deviceName]['_type'] == deviceType:
            devices.append(sample[deviceName])
    if write_access:
        return sample, devices, Devices
    else:
        return sample, devices

def loadQubits(sample, write_access=False):
    """Get local copies of the sample configuration stored in the registry.

    Returns the local sample config, and also extracts the individual
    qubit configurations, as specified by the sample['config'] list.  If
    write_access is True, also returns the qubit registry wrappers themselves,
    so that updates can be saved back into the registry.
    """
    Qubits = [sample[q] for q in sample['config']]  #RegistryWrappers
    sample = sample.copy()                          #AttrDict
    qubits = [sample[q] for q in sample['config']]  #AttrDicts

    # only return original qubit objects if requested
    if write_access:
        return sample, qubits, Qubits
    else:
        return sample, qubits

def loadDevices(sample, write_access=False):
    """Get local copies of the sample configuration stored in the registry.

    Returns the local sample config, and also extracts the individual
    device configurations, as specified by the sample['config'] list.  If
    write_access is True, also returns the qubit registry wrappers themselves,
    so that updates can be saved back into the registry.
    """
    devices={}
    Devices={}
    #The order of these lines is important, as we want devices to be assigned
    #after we make a copy of sample, whereas Devices is not a copy.
    for d in sample['config']:
        Devices[d]=sample[d]
    sample = sample.copy()
    for d in sample['config']:
        devices[d]=sample[d]

    if write_access:
        return sample, devices, Devices
    else:
        return sample, devices

def otherQubitNamesOnCarrier(qubit, devs):
    qubitNames = []
    for dev in devs:
        if dev['_type'] == 'transmon' and dev.__name__ != qubit.__name__:
            qubitNames.append(dev.__name__)
    return qubitNames

def updateQubitCarrier(qubit, devs):
    fc = qubit['fc']
    for dev in devs:
        if dev['_type'] == 'transmon' and dev.__name__ != qubit.__name__:
            dev['fc'] = fc

def dcZero():
    for id in [2, 3, 5, 6]:
        board = 'DR Lab FastBias %d' % id
        for chan in ['A', 'B', 'C', 'D']:
            dcVoltage(board, chan, 0)


def dcVoltage(board, chan, voltage):
    with labrad.connect() as cxn:
        channels = [('b', ('FastBias', [board, chan]))]
        p = cxn.qubit_sequencer.packet()
        p.initialize([('dev', channels)])
        p.mem_start_timer()
        p.mem_bias([('b', 'dac1', voltage)])
        p.mem_stop_timer()
        p.build_sequence()
        p.run(30)
        p.send()


def fastbiasZero():
    data = 0x7FFF << 3
    dac = (1<<19) # DAC1
    slow = (0<<2) # FAST
    mem = [0x000000,
           0x300250,  # delay 250cycle
           (1 << 20) | ((dac + data + slow) & 0xFFFFF),
           (2 << 20) | ((dac + data + slow) & 0xFFFFF),
           0xF00000]
    with labrad.connect() as cxn:
        fpga = cxn.ghz_fpgas
        for board in fpga.list_dacs():
            p = fpga.packet()
            p.select_device(board)
            p.memory(mem)
            p.sram([0]*30)
            p.daisy_chain([board])
            p.run_sequence(30, False)
            p.send()


FLUXWIRING = {}
def updateWiring():
    with labrad.connect() as cxn:
        reg = cxn.registry
        reg.cd(["", "Servers", "Qubit Server", "Wiring"])
        wiring = reg.get("wiring")
        fb_conn = wiring[1]
        fb_conn = {v:k for k, v in fb_conn}
    global FLUXWIRING
    FLUXWIRING.update(fb_conn)
# updateWiring()

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
        data = long((volt['mV']+2500.0)/5000.0*0xFFFF)

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

def setFastbiasLevel(fpga, board, channel, volt, dacName='FAST'):
    fpga.select_device(board)
    mem = makeMemValue(volt, channel=channel, dacName=dacName)
    fpga.memory(mem)
    fpga.sram(np.array([0]*30,dtype='uint32'))
    fpga.daisy_chain([board])
    fpga.run_sequence(30, False)

def setFluxVoltage(sample, measure, voltage, dacName='FAST'):
    cxn = sample._cxn
    fpga = cxn.ghz_fpgas
    sample, qubits = loadQubits(sample)
    q = qubits[measure]
    fluxChannel = dict(q['channels'])['flux']
    key = tuple(fluxChannel[1])
    board, channel = FLUXWIRING[key]
    setFastbiasLevel(fpga, board, channel, voltage, dacName=dacName)
