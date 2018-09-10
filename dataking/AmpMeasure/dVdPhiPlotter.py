import labrad
from labrad.types import Value
from labrad.units import V
import numpy as np
import time
import setbias as sb


folder = ['','Ted','SLUG', 'DR', '111213']

def VPhiPlotter(cxn):
    cxn = labrad.connect()
    dv = cxn.data_vault
    dv.cd(folder, True)
    dv.new('V Phi Curve', ['Flux Bias [uA]','Ib [uA]'], ['Gain [dB]'])
    ag = cxn.agilent_8720es
    ag.select_device('Vince GPIB Bus - GPIB1::15')
    sqVoltages = np.arange(0,2.55,0.05)
    flVoltages = np.arange(-2.5,2.52,0.02)
    for a in range(len(sqVoltages)):
        sb.set_fb('Vince DAC 11', cxn, sqVoltages[a]*V, 0)
        for b in range(len(flVoltages)):
            sb.set_fb('Vince DAC 11', cxn, flVoltages[b]*V, 1)
            result = ag.get_maximum()
            gain = result[0]+50
            dv.add(voltToCurrFlux(flVoltages[b]),voltToCurrSQUID(sqVoltages[a]),gain)
            print '(' + str(sqVoltages[a]) + ', ' + str(flVoltages[b]) + ')'
        
def voltToCurrSQUID(voltage):
    current = (float(voltage)/101500.0)*1000000
    return current

def voltToCurrFlux(voltage):
    current = (float(voltage)/19700.0)*1000000
    return current