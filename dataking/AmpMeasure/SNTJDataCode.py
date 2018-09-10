
print 'Importing Libraries...'

#import needed modules
import sys
import labrad
from pylab import*
import time
import math
import numpy
import matplotlib.pyplot as plt
import os
import pyle.dataking.util as util
from labrad.units import Value, kHz,ms,Hz,s

k = 1.381e-23    #boltzmann's constan


# opens a connection to labrad using the with command which will
# ensure the connection is closed if the sequence is terminated early

def takeSNTJData(sample):
    cxn = sample._cxn
    samp, dev = util.loadDeviceType(sample, 'SNTJ')
    sntj = dev[0]
    specAnal = cxn.spectrum_analyzer_server
    funcGen = cxn.agilent_33120a_generator

    print 'Setting Spectrum Analyzer to Correct mode...'
    #select spectrum analyzer.  make sure to check which device to connect to.
	#Select the correct DC source, in this case the agilent 33120 function generator
    try:
        specAnal.select_device(sntj['specAnal'])
        funcGen.select_device(sntj['funcGen'])
    except:
        print 'Device connection error'
        print 'Please verify devices are connected as desired'


    #Desired presets when using source
    fileName = 'Shot Noise Curve'
    date = sample._dir[-1]                                          #specifies the date of measurement
    det = sntj['det']                                     #Sets spec analyz mode
    freq = sntj['freq']                                   #Sets freq in MHz
    guessGain = sntj['guessGain']                         #Records expected gain
    guessNoise = sntj['guessNoise']                       #Records expected noise
    guessTemp = sntj['guessTemp']                         #Records expected tain
    inputAttenuation = sntj['inputAttenuation']           #Records input attenuation
    numberOfAverages = sntj['numberOfAverages']           #Sets number of Averages
    numberOfPoints = sntj['numberOfPoints']               #Sets number of data points taken in a trace
    refLev = sntj['refLevel']                             #Sets visible range of spec analyz
    resBand = sntj['resBand']                             #Sets resolution bandwidth in MHz
    span = sntj['span']                                   #Sets span in MHz
    sweepTime = sntj['sweepTime']*ms                      #Sets sweep time in ms
    truncate = sntj['truncate']                           #Sets number of points left off of the end of the data set
    vidBand = sntj['videoBand']                           #Sets video bandwidth in KHz
    voltageAmplitude = sntj['voltageAmplitude']           #Output Voltage from sig gen
    yScale = sntj['yScale']                               #Sets spec Analyz to linear scale
    delay = (sweepTime/2)                                   #Added so spec Analyz triggers on correct part of signal
    sourceFreq = 1.0/(sweepTime*2.0)                        #Used to set sig gen freq
    voltagesPtP = 2*float(voltageAmplitude)                 #Used to set sig gen voltage
    VRamp = '%f'%voltagesPtP                                #Converts Vptp into a string
    voltStep = float(voltagesPtP / numberOfPoints)          #Creates voltage division size based on number of points
    voltages = numpy.arange(-(voltageAmplitude),
                            (voltageAmplitude) + voltStep,
                            voltStep)                               #aranges the peak to peak voltage as a series of discrete points

    # The following lines of code set the spectrum analyzer to the correct mode using the
    # given parameters

    specAnal.set_center_frequency(freq)
    specAnal.set_span(span)
    specAnal.set_resolution_bandwidth_mhz(resBand)
    specAnal.set_video_bandwidth_khz(vidBand)
    specAnal.y_scale(yScale)
    specAnal.reference_level_dbm(refLev)
    specAnal.sweep_time_msec(sweepTime)
    specAnal.average_on_off('OFF')
    specAnal.number_of_points(numberOfPoints)
    specAnal.number_of_averages(numberOfAverages)
    specAnal.detector_type(det)
    specAnal.trigger_source('EXT')
    specAnal.gpib_write('TRIG:EXT:SLOP NEG')
    specAnal.gpib_write('TRIG:DEL:STAT ON')
    specAnal.gpib_write('TRIG:DEL %f'%delay['s'])
    funcGen.set_impedance('INF')





    print 'Collecting Data...'

    finalData = []    #this is to set up an array for the final data in advance
    funcGen.gpib_write('APPL:TRI %f HZ, '%sourceFreq['Hz'] + VRamp + ' VPP, 0 V')  # This code tells the function generator to begin biasing the junction
    averaging = True
    specAnal.gpib_write('*CLS')
    specAnal.gpib_write('*ESE 1')
    specAnal.average_on_off('ON')     #This code sets up the spec analyz to average a given number of traces
    specAnal.gpib_write(':INIT:IMM')  #and then ask for the data once the averaging is done
    specAnal.gpib_write('*OPC')
    while averaging:
        result = specAnal.gpib_query('*STB?')
        if int(result)&(1<<5):
            averaging = False
            print 'done'
        time.sleep(1)
    try:
        data = specAnal.get_trace()
    except:
        time.sleep(1)
        try:
            data = specAnal.get_trace()
        except:
            print 'data retrieval failure'
            specAnal.average_on_off('OFF')
            funcGen.set_dc(0)
            sys.exit(0)
    specAnal.average_on_off('OFF')
    funcGen.set_dc(0)

    powerValues = np.asarray(data[2])           # This takes the data which is the second element of what the spectrum analyzer returns and structures it
                                            # as an array
    for i in range(numberOfPoints):
        linPower = (((10.**(powerValues[i]/10.))/1000.)/(k*resBand*1000000))            #  This Loop converts each power value from dBm into Kelvin
        finalData.append(linPower)                                                      #  note this numbers should be quite large as they are still
        voltages[i]= inputAttenuation* voltages[i]                                      #  multiplied by the gain of the amplifier chain.

    #denote how many points for plotting

    points = len(finalData)

    dv = cxn.data_vault
    dv.cd(sample._dir, True)
    dv.new(fileName, ['Bias [V]'], ['Noise[K]'])
    p = dv.packet()
    for key in sntj.keys():
        p.add_parameter(key, sntj[key])
    p.send()

    # This code truncates the data set to avoid possibly erronius data at the edges
    # of the sweep and then saves the result to the data vault
    for num in range(points-2*truncate):
        dv.add(voltages[num+(truncate)],finalData[num+(truncate)])







