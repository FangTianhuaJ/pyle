import numpy as np
import matplotlib.pyplot as plt
from pyle.plotting import dstools as ds
import time


def uwaveTraces(sample, channels = [1,2,3], name = '', muxServ = None, scopeServ = None, 
                scopeMuxChan = 1, plotData = True, spacing = 100, holdFig=False): # spacing in units of [mv], just for the plot
    
    # This script requires that you pass it the servers for the rf mux and scope that you want to use,
    # this means that you must also prepare the servers with 'select_device'
    
    path = sample._dir
    cxn = sample._cxn
    
    labelDict = eval(sample['wiringLabels'])
    
    voltages = []
    for k in channels:
        muxServ.set_channel(k)
        time.sleep(1) # there was some miscommunication between mux and scope, this is a lame fix
        scopeTime, trace = scopeServ.get_trace(scopeMuxChan)
        voltages = np.hstack((voltages, trace))
    data = np.hstack((scopeTime, voltages))
    data = data.reshape((-1, len(channels)+1), order = 'F') # time & voltages used to be long vectors, its now an array in the correct format for the data vault
    
    dv = cxn.data_vault
    dv.cd(path, True)
    
    independents = ['time [ns]']
    dependents = ['%s [mV]' % labelDict[k] for k in channels]
    
    dv.new(name + ' ScopeTrace', independents, dependents)
    
    dv.add(data)
    
    dv.add_parameter('Channels',channels)
    
    if plotData:
        if not holdFig:
            plt.figure()
        [plt.plot(data[:,0], data[:,k+1] - k * spacing) for k in range(len(channels))]
        plt.legend([labelDict[k] for k in channels], loc=1)
        plt.xlabel('Time [ns]')
        plt.ylabel('Traces [mV]')
        plt.grid('on')
        
def rePlotTrace(sample, dataSetNum, spacing = 100):
    
    labelDict = eval(sample['wiringLabels'])
    
    dv = sample._cxn.data_vault
    
    data = ds.getDataset(dv, dataSetNum)
    
    chanLen = len(data.T) - 1
    
    dv.open(dataSetNum)
    channels = dv.get_parameters()[0][1]
    
    [plt.plot(data[:,0], data[:,k+1] - k * spacing) for k in range(chanLen)]
    #plt.legend([labelDict[k] for k in channels])
    plt.xlabel('Time [ns]')
    plt.ylabel('Traces [mV]')
    plt.grid('on')