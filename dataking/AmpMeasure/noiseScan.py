import labrad
from labrad.units import Unit
from labrad.types import Value
from pyle.util import sweeptools as st
from pyle.plotting import dstools as ds
V, GHz, MHz, Hz, dBm, dB, K, ms = [Unit(s) for s in ('V','GHz','MHz', 'Hz', 'dBm', 'dB', 'K', 'ms')]
import numpy as np
import time
import pyle.dataking.util as util
import sys
k = 1.381e-23

def takeNoiseScan(sample):
    cxn = sample._cxn
    samp, dev = util.loadDeviceType(sample, 'noiseMeas')
    specAnal = dev[0]
    sa = cxn.spectrum_analyzer_server

    print 'Setting Spectrum Analyzer to Correct mode...'
    address =  specAnal['sa_address']
    sa.select_device(address)

    #Desired presets when using source
    fileName = specAnal['name']
    date = specAnal._dir[-1]                                          #specifies the date of measurement
    det = specAnal['det']                                 #Sets spec analyz mode
    freq = specAnal['freq']                               #Sets freq in MHz
    numberOfAverages = specAnal['numberOfAverages']       #Sets number of Averages
    numberOfPoints = specAnal['numberOfPoints']           #Sets number of data points taken in a trace
    refLev = specAnal['refLevel']                      	    #Sets visible range of spec analyz
    resBand = specAnal['resBand']                         #Sets resolution bandwidth in MHz
    span = specAnal['span']                               #Sets span in MHz
    vidBand = specAnal['videoBand']                         #Sets video bandwidth in KHz
    yScale = specAnal['yScale']                      	    #Sets spec Analyz to linear scale

    # The following lines of code set the spectrum analyzer to the correct mode using the
    # given parameters

    sa.set_center_frequency(freq)
    sa.set_span(span)
    sa.set_resolution_bandwidth_mhz(resBand)
    sa.set_video_bandwidth_khz(vidBand)
    sa.y_scale(yScale)
    sa.reference_level_dbm(refLev)
    sa.average_on_off('OFF')
    sa.number_of_points(numberOfPoints)
    sa.number_of_averages(numberOfAverages)
    sa.detector_type(det)
    sa.trigger_source('IMM')


    print 'Collecting Data...'


    finalData = []    #this is to set up an array for the final data in advance
    averaging = True
    sa.gpib_write('*CLS')
    sa.gpib_write('*ESE 1')
    sa.average_on_off('ON')
    sa.gpib_write(':INIT:IMM')
    sa.gpib_write('*OPC')
    while averaging:
        result = sa.gpib_query('*STB?')
        if int(result)&(1<<5):
            averaging = False
            print 'done'
        time.sleep(1)
    try:
        data = sa.get_trace()
    except:
        time.sleep(1)
        try:
            data = sa.get_trace()
        except:
            print 'data retrieval failure'
            sa.average_on_off('OFF')
            sys.exit(0)
    sa.average_on_off('OFF')

    powerValues = np.asarray(data[2])           # This takes the data which is the second element of what the spectrum analyzer returns and structures it
                                            # as an array
    for i in range(numberOfPoints):
        linPower = powerValues[i]          #  This Loop converts each power value from dBm into Kelvin
        finalData.append(linPower)                                                      #  note this numbers should be quite large as they are still

    #denote how many points for plotting

    points = len(finalData)
    freqStep = float(span)/(numberOfPoints)
    start = freq-(float(span)/2.0)
    freqs = np.arange(start, start+span+freqStep, freqStep)


    dv = cxn.data_vault
    dv.cd(sample._dir, True)
    dv.new(fileName,['Frequency [MHz]'],['Noise[dBm]'])
    p = dv.packet()
    for key in specAnal.keys():
        p.add_parameter(key, specAnal.__getitem__(key))
    p.send()

    for num in range(points):
        dv.add(freqs[num], finalData[num])



def doubleNoiseScan(sample):
    cxn = sample._cxn
    samp, dev = util.loadDeviceType(sample, 'noiseMeas')
    specAnal = dev[0]
    sa = cxn.spectrum_analyzer_server
    an = cxn.anritsu_server

    print 'Setting Spectrum Analyzer to Correct mode...'
    address =  specAnal['sa_address']
    sa.select_device(address)
    an.select_device(specAnal['anritsu'])
    an.frequency(specAnal['signalFreq'])
    an.amplitude(specAnal['signalPower'])
    an.output(False)


    #Desired presets when using source
    fileName = 'Double Noise Scan ' + specAnal['name']
    date = specAnal._dir[-1]                                          #specifies the date of measurement
    det = specAnal['det']                                 #Sets spec analyz mode
    freq = specAnal['freq']                               #Sets freq in MHz
    numberOfAverages = specAnal['numberOfAverages']       #Sets number of Averages
    numberOfPoints = specAnal['numberOfPoints']           #Sets number of data points taken in a trace
    refLev = specAnal['refLevel']                      	    #Sets visible range of spec analyz
    resBand = specAnal['resBand']                         #Sets resolution bandwidth in MHz
    span = specAnal['span']                               #Sets span in MHz
    vidBand = specAnal['videoBand']                         #Sets video bandwidth in KHz
    yScale = specAnal['yScale']                      	    #Sets spec Analyz to linear scale

    # The following lines of code set the spectrum analyzer to the correct mode using the
    # given parameters

    sa.set_center_frequency(freq)
    sa.set_span(span)
    sa.set_resolution_bandwidth_mhz(resBand)
    sa.set_video_bandwidth_khz(vidBand)
    sa.y_scale(yScale)
    sa.reference_level_dbm(refLev)
    sa.average_on_off('OFF')
    sa.number_of_points(numberOfPoints)
    sa.number_of_averages(numberOfAverages)
    sa.detector_type(det)
    sa.trigger_source('IMM')


    print 'Collecting Data...'


    finalData1 = []    #this is to set up an array for the final data in advance
    finalData2 = []
    averaging = True
    sa.gpib_write('*CLS')
    sa.gpib_write('*ESE 1')
    sa.average_on_off('ON')
    sa.gpib_write(':INIT:IMM')
    sa.gpib_write('*OPC')
    while averaging:
        result = sa.gpib_query('*STB?')
        if int(result)&(1<<5):
            averaging = False
            print 'done'
        time.sleep(1)
    try:
        data = sa.get_trace()
    except:
        time.sleep(1)
        try:
            data = sa.get_trace()
        except:
            print 'data retrieval failure'
            sa.average_on_off('OFF')
            sys.exit(0)
    sa.average_on_off('OFF')

    powerValues = np.asarray(data[2])           # This takes the data which is the second element of what the spectrum analyzer returns and structures it
                                            # as an array
    for i in range(numberOfPoints):
        linPower = powerValues[i]          #  This Loop converts each power value from dBm into Kelvin
        finalData1.append(linPower)                                                      #  note this numbers should be quite large as they are still

    averaging = True
    an.output(True)
    sa.gpib_write('*CLS')
    sa.gpib_write('*ESE 1')
    sa.average_on_off('ON')
    sa.gpib_write(':INIT:IMM')
    sa.gpib_write('*OPC')
    while averaging:
        result = sa.gpib_query('*STB?')
        if int(result)&(1<<5):
            averaging = False
            print 'done'
        time.sleep(1)
    try:
        data = sa.get_trace()
    except:
        time.sleep(1)
        try:
            data = sa.get_trace()
        except:
            print 'data retrieval failure'
            sa.average_on_off('OFF')
            sys.exit(0)
    sa.average_on_off('OFF')

    powerValues = np.asarray(data[2])           # This takes the data which is the second element of what the spectrum analyzer returns and structures it
                                            # as an array
    for i in range(numberOfPoints):
        linPower = powerValues[i]          #  This Loop converts each power value from dBm into Kelvin
        finalData2.append(linPower)
    an.output(False)
    #denote how many points for plotting

    points = len(finalData1)
    freqStep = float(span)/(numberOfPoints)
    start = freq-(float(span)/2.0)
    freqs = np.arange(start, start+span+freqStep, freqStep)


    dv = cxn.data_vault
    dv.cd(sample._dir, True)
    dv.new(fileName,['Frequency [MHz]'],['Noise(Pump Off)[dBm]', 'Noise(Pump On)[dBm]'])
    p = dv.packet()
    for key in specAnal.keys():
        p.add_parameter(key, specAnal.__getitem__(key))
    p.send()

    for num in range(points):
        dv.add(freqs[num], finalData1[num], finalData2[num])






