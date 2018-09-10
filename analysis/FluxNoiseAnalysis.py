import numpy as np
import scipy
from numpy import floor, log, exp, pi, cos
import matplotlib.pyplot as plt
import ramseyIntegrals


from labrad.units import Value,Unit
sec, ns, GHz, Hz, MHz = (Unit(s) for s in ['s','ns','GHz','Hz','MHz'])

import signalProcessing as sp
import mathHelpers as mh

from scipy.optimize import leastsq
from pyle.util.structures import AttrDict
from pyle.plotting import dstools


###################################
##RAMSEY OSCILLOSCOPE
###################################

def fixRamseyOscilloscopeData(dataset, timeStep, useArcTan=False):
    x = dataset.data[:,1]-dataset.data[:,3]
    y = dataset.data[:,2]-dataset.data[:,4]
    if useArcTan:
        phase = np.arctan2(y,x)/(2.0*np.pi)
    else:
        phase = mh.xy2cycles(x,y)
    patches = sp.findPatches(dataset.data[:,5],timeStep)
    if patches != []:
        phase = sp.insertZeros(phase,patches)
    return phase

def ramseyOscilloscope(dataset, weighted=False, chunks=1, smooth=None, fitRange=None, useArcTan=False, plotPhase=False, plotSpectrum=False,
                       magOnly=False, slope=None):
    #Get parameters needed for computation of power spectrum
    holdTime = float(dataset.parameters.holdTime['ns'])
    totalTime = float(dataset.data[-1,5])*sec
    timeStep = float(dataset.parameters.timeStep['s'])
    phase  = fixRamseyOscilloscopeData(dataset, timeStep, useArcTan=useArcTan)
    phase = phase - np.mean(phase)
    #Compute power spectra
    #Sphase = sp.DFT_interleaved(phase,totalTime, smooth=('1f',200.0))
    Sphase = sp.dftInterleavedBootstrapped(phase, totalTime, chunks=chunks, smooth=smooth)
    frequencies = Sphase['frequencies']
    #Convert to frequency and flux units
    Sf = Sphase['S']*((1.0/holdTime)**2) #GHz^2/Hz
    if dataset.parameters.dfdPhi0.isCompatible('GHz'):
        Sphi = Sf*((1/dataset.parameters.dfdPhi0['GHz'])**2)
    elif dataset.parameters.dfdPhi0.isCompatible('GHz/PhiZero'):
        Sphi = Sf*((1/dataset.parameters.dfdPhi0['GHz/PhiZero'])**2)
    #Plot spectrum
    plt.figure()
    plt.loglog(frequencies,Sphi,'.')
    #Fit
    if fitRange is None:
        freqMin = float(raw_input('Minimum frequency to fit [Hz]: '))
        freqMax = float(raw_input('Maximum frequency to fit [Hz]: '))
    else:
        freqMin=fitRange[0]
        freqMax=fitRange[1]
    indexList = (frequencies>freqMin) * (frequencies<freqMax)
    freqsToFit = frequencies[indexList]
    SphiToFit = Sphi[indexList]
    plt.loglog(frequencies[indexList],Sphi[indexList],'r.')
    if weighted and not magOnly:
        weights = 1/np.linspace(1,len(freqsToFit),len(freqsToFit))
        p = mh.weightedLinearFit(np.log(frequencies[indexList]),np.log(Sphi[indexList]),weights)
        p = [p[1][0],p[0][0]]
        slope = p[0]
        A = exp(p[1])
    elif weighted and magOnly:
        weights = 1/np.linspace(1,len(freqsToFit),len(freqsToFit))
        mag = mh.weightedLinearFitFixedSlope(np.log(frequencies[indexList]),np.log(Sphi[indexList]),slope,weights)
        A = exp(mag)
    else:
        p = np.polyfit(log(frequencies[indexList]),np.log(Sphi[indexList]),1)
        slope = p[0]
        A = exp(p[1])
        
    #Show fit line
    plt.loglog(frequencies[indexList],A*frequencies[indexList]**slope,'k.')
    #Package up data to return
    S = AttrDict({'frequencies':frequencies,'Sf':Sf,'Sphi':Sphi,
                  'A':A,'slope':slope,'phase':phase})
    return S

def plotRamseyOscilloscope(dataset):
    amp = str(np.sqrt(dataset.S.A)*1.0E-6)[0:3] #uPhi0/rtHz
    slope = str(dataset.S.slope)[0:4]
    frequencies = dataset.S.frequencies
    description = raw_input('Description of device: ')
    plt.figure()
    plt.loglog(frequencies,dataset.S.Sphi,'.')
    plt.text(0.01,1.0E-7,'Amplitude: '+ amp +'uPhi0/rtHz')
    plt.text(0.01,3.0E-8,'slope: '+slope)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Sphi [Phi0^2/rtHz]')
    plt.title('Low Frequency Flux Noise - '+dataset.path[3]+
              dataset.path[4]+ dataset.parameters.config[0] +
              '_'+dataset.path[5]+'_'+str(dataset.path[6])+
              '_'+description)
    plt.loglog(frequencies, dataset.S.A*(frequencies**dataset.S.slope),'r.')
    plt.grid()

#################################
## RAMSEY  ######################
#################################

def plotRamsey(data,T1):
    T1 = float(T1)
    t = data[:,0]
    p = data[:,1]
    pCentered = p - np.mean(p)
    pScaled = pCentered/exp(-t/(2.0*T1))
    plotWithAxes(t,pScaled,'Time [ns]','Probability (Centered)')

def plotScaledRamsey(data,T1):
    T1 = float(T1)
    t = data[:,0]
    p = data[:,1]
    pCentered = p-np.mean(p)
    pScaled = pCentered/exp(-t/(2.0*T1))
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t, pCentered,'.')
    plt.xlabel('Time [ns]')
    plt.ylabel('Probability (centered)')
    plt.subplot(3,1,2)
    plt.plot(t, pScaled,'r.')
    plt.xlabel('Time [ns]')
    plt.ylabel('Probability Scaled by T1')
    plt.subplot(3,1,3)
    plt.plot(t,pCentered,'.')
    plt.plot(t,pScaled,'r.')
    plt.xlabel('Time [ns]')
    plt.ylabel('Probability')    

#################################
##RAMSEY TOMOGRAPHY
#################################

def makeEnvelope(dataset):
    xp = dataset.data[:,1]
    xm = dataset.data[:,3]
    yp = dataset.data[:,2]
    ym = dataset.data[:,4]
    env = np.sqrt((xp-xm)**2+(yp-ym)**2)
    return env

def scaledEnvelope(dataset,T1):
    T1 = float(T1)
    t = dataset.data[:,0]
    env = makeEnvelope(dataset)
    envScaled = env/exp(-t/(2.0*T1))
    return envScaled

def ramseyTomo_noLog(dataset, T1=None, timeRange=None, plot=False):
    """
    """
    #CHANGELOG
    # 17 Aug 2011-Daniel Sank
    #  Now uses the plotting functions in pyle.plotting.dstools
    #  Checked that the plots seems to work properly
    time = dataset.data[:,0]
    probabilities = dataset.data[:,1:]
    freq = dataset.parameters['fringeFrequency']['GHz']
    
    if T1 is None:
        T1 = float(raw_input('T1 [ns]: '))
    T1=float(T1)
    colors = ['.','r.','g.','k.']
    xp = probabilities[:,0]
    yp = probabilities[:,1]
    xm = probabilities[:,2]
    ym = probabilities[:,3]
    env = np.sqrt( (xp-xm)**2 + (yp-ym)**2)
    envScaled = env/exp(-time/(2*T1))
    #Plot raw data, envelope, and evelope with T1 scaled out
    if plot:
        figRaw=dstools.plotDataset1D(dataset.data,dataset.variables[0],dataset.variables[1],
                                     title='Raw Data')
        figRaw.show()
        figEnv=dstools.plotDataset1D(np.vstack((time,env)).T,[('Time','ns')],
                                    [('Envelope Probability','','')],title='Envelope')
        figEnv.show()
        figEnvScaled=dstools.plotDataset1D(np.vstack((time,envScaled)).T,[('Time','ns')],
                                          [('Scaled Envelope Probability','','')],
                                          title='Scaled Envelope')
        figEnvScaled.show()
    #If time range for fitting has not been specified, ask user
    if timeRange is None:
        minTime = float(raw_input('Minimum time to keep [ns]: '))
        maxTime = float(raw_input('Maximum time to keep [ns]: '))
        timeRange = (minTime,maxTime)
    #Generate array mask for elements to be fitted
    indexList = (time>timeRange[0]) * (time<timeRange[1])
    t = time[indexList]
    e = env[indexList]
    #Create fit function. We fit the unscaled data
    fitFunc = lambda v,x: v[0]*exp(-((x/v[1])**2))*\
              exp(-x/(2*T1))*exp(-x/v[2])+v[3]
    errorFunc = lambda v,x,y: fitFunc(v,x)-y
    vGuess = [0.85, 200.0, 700.0, 0.02]
    result = leastsq(errorFunc, vGuess, args=(t,e), maxfev=10000, full_output=True)
    v = result[0]
    cov=result[1]
    success = result[4]
    #Jim's formula for error bars. Got this from the Resonator fit server.
    rsd = (result[2]['fvec']**2).sum()/(len(t)-len(v))
    bars = np.sqrt(rsd*np.diag(result[1]))
    #Plot the fitted curve
    if plot:
        ax = figEnv.get_axes()[0]
        ax.plot(time, fitFunc(v,time),'g',linewidth=4)
        figEnv.show()
    T2 = v[1]*ns
    dT2 = bars[1]*ns
    Tphi = v[2]*ns
    dTphi = bars[2]*ns
    return AttrDict({'T2':T2, 'dT2':dT2, 'Tphi':Tphi, 'dTphi':dTphi, 'success':success,
                     'fitFunc':fitFunc,'fitVals':v,'envScaled':envScaled})


def ramseyTomoWithLogFactor(dataset, T1):
    time = dataset.data[:,0]
    probabilities = dataset.data[:,1:]
    freq = dataset.parameters['fringeFrequency']['GHz']
    
    TIME = 3600.0*(10**9) #One hour in nanoseconds
    T1=float(T1)
    colors = ['.','r.','g.','k.']
    xp = probabilities[:,0]
    yp = probabilities[:,1]
    xm = probabilities[:,2]
    ym = probabilities[:,3]
    env = np.sqrt( (xp-xm)**2 + (yp-ym)**2)
    plt.figure()
    for i in range(4):
        plt.plot(time,probabilities[:,i],colors[i])
    plt.figure()
    plt.plot(time,env)
    minTime = float(raw_input('Minimum time to keep [ns]: '))
    maxTime = float(raw_input('Maximum time to keep [ns]: '))
    #Generate array mask for elements you want to keep
    indexList = (time>minTime) * (time<maxTime)
    t = time[indexList]
    e = env[indexList]

    fitFunc = lambda v,x: v[0]*exp(-(x**2)*np.log(0.4*TIME/x)/(v[1]**2))*\
              exp(-x/(2*T1))+v[2]
    errorFunc = lambda v,x,y: fitFunc(v,x)-y
    
    vGuess = [0.85, 130.0, 0.02]
    result = leastsq(errorFunc, vGuess, args=(t,e), maxfev=10000, full_output=True)
    v = result[0]
    cov=result[1]
    success = result[4]
    #Jim's formula for error bars. Got this from the Resonator fit server.
    rsd = (result[2]['fvec']**2).sum()/(len(t)-len(v))
    bars = np.sqrt(rsd*np.diag(result[1]))
    #Plot the fitted curve
    plt.plot(time, fitFunc(v,time),'g')
    T2 = v[1]*ns
    dT2 = bars[1]*ns
    S = 2.0*(1.0/(2*pi*T2['ns'])**2)*(dataset.parameters.dfdPhi0['GHz'])**-2
    
    return AttrDict({'T2':v[1]*ns, 'dT2':dT2, 'success':success, 'rtS':np.sqrt(S)})

def ramseyTomoAlpha(dataset, T1, alphas, fitRange=None, makePlots=False):
    if not np.iterable(alphas):
        alphas = np.array([alphas])
    time = dataset.data[:,0]
    probabilities = dataset.data[:,1:]
    #freq = dataset.parameters['fringeFrequency']['GHz']
    
    TIME = 3600.0*(10**9) #One hour in nanoseconds
    T1=float(T1)
    colors = ['.','r.','g.','k.']
    xp = probabilities[:,0]
    yp = probabilities[:,1]
    xm = probabilities[:,2]
    ym = probabilities[:,3]
    env = np.sqrt( (xp-xm)**2 + (yp-ym)**2)
    if makePlots:
        plt.figure()
        for i in range(4):
            plt.plot(time,probabilities[:,i],colors[i])
        plt.figure()
        plt.plot(time,env)
    if fitRange is None:
        minTime = float(raw_input('Minimum time to keep [ns]: '))
        maxTime = float(raw_input('Maximum time to keep [ns]: '))
    else:
        minTime = float(fitRange[0])
        maxTime = float(fitRange[1])
    #Generate array mask for elements you want to keep
    indexList = (time>minTime) * (time<maxTime)
    t = time[indexList]
    e = env[indexList]
    noiseAmplitudes = np.array([])
    for alpha in alphas:
        integralFunc = ramseyIntegrals.interpForAlpha(alpha,t,1.0/TIME)
        fitFunc = lambda x,amp,b,offset: amp*exp(-x/(2*T1)) * exp(-(x**(1.0+alpha))*b*integralFunc(x))+offset
        result,cov = scipy.optimize.curve_fit(fitFunc,t,e,[1.0,1.0/1000000,0.02])
        a,b,c = result
        S = 2.0*(1.0/(2*pi)**2)*b*(dataset.parameters.dfdPhi0['GHz'])**-2
        noiseAmplitudes = np.hstack((noiseAmplitudes,np.sqrt(S)))
    return AttrDict({'noiseAmplitudes':noiseAmplitudes, 'alphas':alphas})
    

def spinEchoTomo(dataset,T1,timeRange,makePlots=False):
    time = dataset.data[:,0]
    probabilities = dataset.data[:,1:]
    T1=float(T1)
    colors = ['.','r.','g.','k.']
    xp = probabilities[:,0]
    yp = probabilities[:,1]
    xm = probabilities[:,2]
    ym = probabilities[:,3]
    env = np.sqrt( (xp-xm)**2 + (yp-ym)**2)
    if makePlots:
        plt.figure()
        for i in range(4):
            plt.plot(time,probabilities[:,i],colors[i])
        plt.figure()
        plt.plot(time,env)
    if timeRange is None:
        minTime = float(raw_input('Minimum time to keep [ns]: '))
        maxTime = float(raw_input('Maximum time to keep [ns]: '))
    else:
        minTime = float(timeRange[0])
        maxTime = float(timeRange[1])
    #Generate array mask for elements you want to keep
    indexList = (time>minTime) * (time<maxTime)
    t = time[indexList]
    e = env[indexList]
    fitFunc = lambda x,amp,rate,offset: (amp*exp(-x/(2*T1)) * exp(-(x/rate)**2))+offset
    result,cov = scipy.optimize.curve_fit(fitFunc,t,e,[1.0,0.0025,0.02])
    a,tEcho,c = result
    try:
        dfdPhi0 = dataset.parameters.dfdPhi0['GHz/PhiZero']
    except:
        print 'Frequency sensitivity was not in GHz/Phi0, trying GHz'
        dfdPhi0 = dataset.parameters.dfdPhi0['GHz']
    #Compute noise amplitude, in units of Phi0. There is no frequency unit because
    #we're assuming the spectral density is S
    S = (4.0)/(((2*np.pi)**2)*(tEcho**2)*(dfdPhi0**2)*1.35)
    noiseAmplitude = np.sqrt(S)
    return AttrDict({'noiseAmplitude':noiseAmplitude,'fitFunc':fitFunc})


def spinEchoTomoAlpha(dataset,inputT1,alphas,timeRange=None,makePlots=False,totalTime=3600*sec,upperFreq=10*MHz):
    results = []
    colors = ['.','r.','g.','k.']
    if not np.iterable(alphas):
        alphas = np.array([alphas])
    if dataset.parameters.dfdPhi0.isCompatible('GHz/PhiZero'):
        sUnit = 'GHz/PhiZero'
    elif dataset.parameters.dfdPhi0.isCompatible('GHz'):
        sUnit = 'GHz'
    if not np.iterable(alphas):
        alphas = np.array([alphas])
    time = dataset.data[:,0]
    probabilities = dataset.data[:,1:]
    lowerFreq = (1.0/(totalTime['ns']))
    T1 = inputT1['ns']
    xp = probabilities[:,0]
    yp = probabilities[:,1]
    xm = probabilities[:,2]
    ym = probabilities[:,3]
    env = np.sqrt( (xp-xm)**2 + (yp-ym)**2 )
    #Plot raw data and envelope
    if makePlots:
        plt.figure()
        for i in range(4):
            plt.plot(time,probabilities[:,i],colors[i])
        plt.grid()
        plt.figure()
        plt.plot(time,env/(exp(-time/(2.0*T1))))
        plt.grid()
    if timeRange is None:
        minTime = float(raw_input('Minimum time to keep [ns]: '))
        maxTime = float(raw_input('Maximum time to keep [ns]: '))
    else:
        minTime = float(fitRange[0])
        maxtime = float(fitRange[1])
    indexList = (time>minTime) * (time<maxTime)
    t = time[indexList]
    e = env[indexList]
    noiseAmplitudes = np.array([])
    for alpha in alphas:
        integralFunc = ramseyIntegrals.echoInterpForAlpha(alpha,t,lowerFreq,upperFreq)
        fitFunc = lambda x,amp,b,offset: amp*exp(-t/(2*T1)) * exp(-(x**(1.0+alpha))*b*integralFunc(x))+offset
        result,cov = scipy.optimize.curve_fit(fitFunc,t,e,[1.0,1.7E-8,0.02])
        a,b,c = result
        S = 2.0*(1.0/(2*pi)**2)*b*(dataset.parameters.dfdPhi0[sUnit])**-2
        noiseAmplitudes = np.hstack((noiseAmplitudes,np.sqrt(S)))
        results.append(AttrDict({'noiseAmplitude':noiseAmplitude,'alpha':alpha,
                                 'fitFunc':fitFunc}))
    return results
    
def rabi(dataset, freq, T1, timeRange):
    T1 = float(T1['ns'])
    time = dataset.data[:,0]
    probability = dataset.data[:,1]
    probability = probability-np.mean(probability)
    plt.figure()
    plt.plot(time,probability,'.')
    plt.xlabel('Time [ns]')
    plt.ylabel('Probability')
    #Select range to fit
    if timeRange is None:
        minTime = float(raw_input('Minimum time to keep [ns]: '))
        maxTime = float(raw_input('Maximum time to keep [ns]: '))
    else:
        minTime=timeRange[0]
        maxTime=timeRange[1]
    #Generate array mask for elements you want to keep
    indexList = (time>minTime) * (time<maxTime)
    t = time[indexList]
    p = probability[indexList]
    plt.plot(t,p,'r.')
    #Fit oscillations with decaying exponential
    fitFunc = lambda v,x: v[0]*exp(-(1.0/2.0)*x/v[1])*exp(-(3.0/4.0)*x/T1)*\
              cos(2*pi*x*v[2]+v[3])+v[4]
    errorFunc = lambda v,x,y: fitFunc(v,x)-y
    vGuess = [0.5, 400, freq['GHz'], np.pi, 0.0]
    result = leastsq(errorFunc, vGuess, args=(t,p), maxfev=10000, full_output=True)
    v = result[0]
    cov=result[1]
    if cov is None:
        raise Exception('Singular Covariance matrix was found.')
    success = result[4]
    #Jim's formula for error bars. Got this from the Resonator fit server.
    rsd = (result[2]['fvec']**2).sum()/(len(t)-len(v))
    bars = np.sqrt(rsd*np.diag(result[1]))
    #Plot the fitted curve
    timePlot = np.arange(time[0],time[-1],1.0)
    plt.plot(timePlot, fitFunc(v,timePlot),'g')
    plt.grid()
    Tnu = v[1]*ns
    dTnu = bars[1]*ns
    freq=v[2]
    #Compute Sf and Sphi
    Sf = ((1.0/Tnu['ns'])*((1.0/np.pi)**2)*(1.0E-9))*((1.0*GHz)**2)/(1.0*Hz) #GHz^2/Hz
    Sphi = Sf*(dataset.parameters.dfdPhi0)**-2

    return AttrDict({'Tnu':Tnu, 'dTnu':dTnu, 'freq':freq*GHz, 'Sf':Sf, 'Sphi':Sphi,
                     'fitFunc':fitFunc, 'fitVals':result[0]})
    
def plotWithAxes(x,y,xlabel='',ylabel='',marker='.'):
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x,y,marker)
