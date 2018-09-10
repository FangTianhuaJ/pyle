# Author: Daniel Sank
# Created: 2011

# CHANGELOG
#
#


import sys
import time
import numpy as np
from msvcrt import getch, kbhit
import matplotlib.pyplot as plt

import labrad
from labrad.units import Unit, Value
s,Hz,kHz,MHz = (Unit(u) for u in ['s','Hz','kHz','MHz'])


import pyle.registry
import pyle.dataking.util as dataUtil
import pyle.util.sweeptools as st

#This provides a useful scan over the range accessible by the SR770
           #span        start       ave,  overlap
RANGES = [#(99*kHz,      0.0*Hz,     1000,   0.0),
          #(12.5*kHz,    1*kHz,      1000,   0.0),
          3(1.0*kHz,     100*Hz,     100,    0.0),
          #(100*Hz,      10*Hz,      50,     50.0),
          #(10*Hz,        1*Hz,      100,    90.0),
          (1*Hz,        0.1*Hz,     20,     80.0),
          (0.1*Hz,      0.01*Hz,    20,     90.0)]#,
#          (0.1*Hz,      0.0*Hz,     20,     90.0)]

def doDatasets(s, cxn):
    recipient = s._dir[1]
    sample,devices = dataUtil.loadDevices(s)
    dataDir = s._dir
    for frequencySpan,frequencyStart,averageNum,averageOverlap in RANGES:
        devices['analyzer']['frequencySpan']=frequencySpan
        devices['analyzer']['frequencyStart']=frequencyStart
        devices['analyzer']['averageNum']=averageNum
        devices['analyzer']['averageOverlap']=averageOverlap
        getSpectralAmplitude(cxn, sample, devices, dataDir)
    try:
        cxn.telecomm_server.send_sms('noise test', 'measurement complete', recipient)
    except:
        print 'Text message failed'

def getSpectralAmplitude(cxn, sample, devices, dataDir, save=True):
    sr770 = cxn.signal_analyzer_sr770
    #Set up the device and wait for settling
    setupSR770(devices['analyzer'], sr770)
    #Acquire spectral density, will wait for averaging to complete
    data = sr770.power_spectral_amplitude(0)
    data = np.asarray(data)
    #Save to datavault
    indeps = [('Frequency','Hz')]
    deps = [('Spectral amplitude','','V/Hz^1/2')]
    if save is True:
        makeDataset(cxn, sample, devices, indeps, deps, data, dataDir)
    return data


def setupSR770(device, server):
    p = server.packet()
    p.select_device(device['_addr'])
    trace = device['trace']
    #Input setup
    p.input_range(device['inputRange'],         key='inputRange')
    p.coupling(device['inputCoupling'],         key='inputCoupling')
    p.grounding(device['inputGrounding'],       key='inputGrounding')
    #Set measure,display,units,window
    p.display(trace,device['measureDisplay'],   key='measureDisplay')
    p.measure(trace,device['measureType'],      key='measureType')
    p.units(trace,device['measureUnits'],       key='measureUnits')
    p.window(trace,device['measureWindow'],     key='measureWindow')
    #Go to desired span and start frequency, and wait for settling
    p.freq_and_settle(device['frequencySpan'],device['frequencyStart'], key='spanAndStart')
    #Averaging setup
    p.average(device['average'],                key='average')
    p.num_averages(device['averageNum'],        key='averageNum')
    p.overlap(device['averageOverlap'],         key='averageOverlap')
    result = p.send()
    device['inputRange'] = result['inputRange']
    device['inputCoupling']=result['inputCoupling']
    device['inputGrounding']=result['inputGrounding']
    device['measureDisplay']=result['measureDisplay'][1]
    device['measureType'] = result['measureType'][1]
    device['measureUnits']=result['measureUnits'][1]
    device['measureWindow']=result['measureWindow']
    device['frequencySpan']=result['spanAndStart'][0]
    device['frequencyStart']=result['spanAndStart'][1]
    device['average']=result['average']
    device['averageNum']=result['averageNum']
    device['averageOverlap']=result['averageOverlap']

def makeDataset(cxn, sample, devices, independents, dependents, data, dataDir):
    name = sample['dut']['name']
    parameters = parseSampleParameters(sample)
    for device,parms in devices.items():
        for key,value in devices[device].items():
            parameters.append((device+'.'+key,value))
    dv = cxn.data_vault
    dv.cd('')
    dv.cd(dataDir, True)
    dv.new(name, independents, dependents)
    p = dv.packet()
    for parameter in parameters:
        p.add_parameter(parameter)
    p.send()
    dv.add(data)

def parseSampleParameters(sample): #this is a total hack
    parameters=[]
    for key,value in sample.items():
        if isinstance(value,dict):
            continue
        else:
            parameters.append((key,value))
    return parameters
