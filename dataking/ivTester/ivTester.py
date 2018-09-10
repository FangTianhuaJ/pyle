import labrad
import numpy as np
from pylab import plot, show

from labrad.units import Unit
from labrad.types import Value
V, mV, us, ns, GHz, MHz, uA = [Unit(s) for s in ('V','mV','us','ns','GHz','MHz', 'uA')]

DATASET_NAME = 'SQUID IV'

def squid_iv(sample, qubit, currentChannel = 1, voltageChannel = 2, correctCurrent = True):
    """Records an IV curve from the Tek2014B and stores to data vault.

    Current correction not implemented yet. Complain to Dan about this.
    """

    print 'Check preamp gain and channel connections!'

    sample = sample.copy()
    preampGain = sample['preampGain']
    currentCal = sample['currentCal'] #Should be ('DIODE_BOX',<diodeBoxName>)
    scopeAddress = sample['scopeAddress']
    correctedCurrent = None

    with labrad.connect() as cxn:
        scope = cxn.tektronix_2014b_oscilloscope
        scope.select_device(scopeAddress)

        #The Tek2014B server returns traces as (time[s],voltage[v])
        #Get voltage trace
        voltageTrace = scope.get_trace(voltageChannel)
        voltageTrace = (voltageTrace[0].asarray,voltageTrace[1].asarray)
        voltageCorrected = voltageTrace[1]*(1.0/preampGain)
        #Get current trace and normalize
        currentTrace = scope.get_trace(currentChannel)
        currentTrace = (currentTrace[0].asarray,currentTrace[1].asarray)
        if correctCurrent is True:
            correctedCurrent = correctCurrentFn(cxn, currentTrace[1],currentCal)

        currentSourceVoltage = currentTrace[1]
        time = voltageTrace[0]
        #Uncalibrated data
        independents = [('time','s')]
        dependents = [('Current Source Voltage', '', 'V'), ('Voltage', '', 'V')]
        parameters = {'preampGain':preampGain,'currentCal':currentCal}
        data = np.vstack((time, currentSourceVoltage, voltageCorrected))
        if correctCurrent is True:
            dependents.append(('Calibrated Current', '', 'uA'))
            data = np.vstack((data, correctedCurrent))
        data = np.transpose(data)

        name = 'q'+str(qubit)+' '+DATASET_NAME
        makeDataset(sample, cxn, independents, dependents, parameters, name, data)
        
        return data


def makeDataset(sample, cxn, independents, dependents, parameters, nameString, data):
    dv = cxn.data_vault
    dv.cd('')
    dv.cd(sample._dir, True)
    dv.new(nameString, independents, dependents)
    p = dv.packet()
    for parameter in sample.items():
        p.add_parameter(parameter)
    p.send()
    dv.add(data)
    
def correctCurrentFn(cxn, data,currentCal):
    if currentCal[0].upper() == 'DIODE_BOX':
        dv = cxn.data_vault
        path = ['','Devices',currentCal[1]]
        dv.cd(path)
        files = dv.dir()[1]
        dv.open(files[-1])
        raw = dv.get().asarray
        variables = dv.variables()
        voltageUnit = variables[0][0][1]
        currentUnit = variables[1][0][2]
        voltage = raw[:,0]*(Value(1.0,voltageUnit)['V'])
        current = raw[:,1]*(Value(1.0,currentUnit)['uA'])

        voltage = np.hstack((-voltage[::-1],voltage[1:]))
        current = np.hstack((-current[::-1],current[1:]))

        currentCalibrated = np.interp(data, voltage, current)
        return currentCalibrated
        
    elif currentCal[0] == 'None':
        return data
    else:
        raise Exception('currentCal must be "DIODE_BOX" or "None"')
