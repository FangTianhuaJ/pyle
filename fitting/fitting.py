import numpy as np
from numpy import pi, exp
from matplotlib import pyplot as plt

from labrad.units import Unit,Value
V, mV, us, ns, GHz, MHz = [Unit(s) for s in ('V', 'mV', 'us', 'ns', 'GHz', 'MHz')]

from scipy.optimize import curve_fit, leastsq
from scipy.special import erf,erfc
from pyle.plotting import dstools
from pyle.util.structures import AttrDict

# CHANGELOG
#
# 2012 March 28 - Daniel Sank
# Cleaned up T1 fitter to use fitCurve
# Also fixed up multiqubit.t1, note the use of the * operator

#Main library of fitting functions for use in all pyle code





def fitCurve(curve, x, y, vGuess):
    curves = {
              'cosine':cosine,
              'exponential':exponential,
              'gaussian':gaussian,
              'gaussianExp':gaussianExp,
              'line':line,
              'sine':sine,
              'exponentialCosine': exponentialCosine,
              'gaussianExpCosine': gaussianExpCosine,
              }
    if isinstance(curve,str):
        fitFunc = curves[curve]
    else:
        fitFunc = curve
    v, cov = curve_fit(fitFunc, x, y, vGuess)
    variances = [cov[i,i] for i in range(len(v))]
    return v, cov, fitFunc

def line(x, slope, offset):
    return x*slope + offset

def exponential(x, a, timeConst, offset):
    """a*exp(-(x/timeConst))+offset

    a: amplitude
    timeConst: decay constant
    ofset: DC offset
    """
    return (a*exp(-(x/timeConst)))+offset

def gaussian(x, maxAmp, mean, sigma, offset):
    """maxAmp*exp(-0.5*[(x-mean)/sigma]^2)+offset

    maxAmp: amplitude at peak
    mean: value of x at peak
    sigma: standard deviation in x
    offset: DC offset
    """
    return (maxAmp*exp(-0.5*((x-mean)/sigma)**2))+offset

def gaussianExp(x, amp, r1, r2, offset):
    """amp*exp(-(x/r1) - (x/r2)**2) + offset"""
    return amp*np.exp(-(x/r1) - (x/r2)**2) + offset

def sine(x, amp, freq, phase, offset):
    """amp*sin(2*pi*(x*freq+phase))+offset

    amp: amplitude
    freq: frequency (cycles per unit x)
    phase: phase (cycles)
    offset: DC offset
    """
    return amp*np.sin(2*pi*(x*freq+phase))+offset

def cosine(x, amp, freq, phase, offset):
    """amp*cos(2*pi*(x*freq+phase))+offset

    amp: amplitude
    freq: frequency (cycles per unit x)
    phase: phase (cycles)
    offset: DC offset
    """
    return amp*np.cos(2*pi*(x*freq+phase))+offset

def exponentialCosine(x, amp, freq, phase, timeConst, offset):
    """
    amp*exp(-x/timeConst)*cos(2*pi*(x*freq+phase)) + offset

    amp: amplitude
    freq: frequency (cycles per unit x)
    phase: phase (cycles)
    timeConst: decay constant
    offset: DC offset
    """
    return amp*np.exp(-x/timeConst)*np.cos(2*np.pi*(x*freq+phase)) + offset

def gaussianExpCosine(x, amp, r1, r2, f, phi0, offset):
    """
    amp*exp(-(x/r1) - (x/r2)**2) * cos(2*pi*f*x+phi0) + offset
    """
    return amp*np.exp(-(x/r1) - (x/r2)**2) * np.cos(2*np.pi*f*x+phi0) + offset

#Linear transformation that goes from frontLevel,backLevel,depth
#to variables used in the square function definition.
MAT = np.array([[-0.25,0.25,0.5],[0.25,-0.25,0.5],[0.25,0.75,-0.5]])

def bucket(x, center, topLen, frontLevel, backLevel, depth):
    """ A square pulse with possibly different levels before and after.

    The length of the square and the width of the transition regions
    must be given by the user.

    center: x value for center of square pulse
    depth: depth of bucket relative to average DC level of front and end tails
    frontLevel: level of front tail
    backLevel: level of end tail
    """
    A,B,T = np.dot(MAT,np.array([frontLevel,backLevel,depth]))
    return A*erf(x-center-0.5*topLen) + B*erfc(x-center+0.5*topLen) + T

# Other utilities for finding properties of curves

def findMinimum(data,fit=True):
    if fit:
        p = np.polyfit(data[:,0],data[:,1],2)
        xMin = -1.0*p[1]/(2.0*p[0])
        yMin = np.polyval(p,xMin)
        return (xMin,yMin)
    else:
        index = np.argmin(data[:,1])
        return data[index,0],data[index,1]

def getMaxPoly(x, y, fit=True):
    if fit:
        p = np.polyfit(x, y, 2)
        x_val = -1.0*p[1]/(2.0*p[0])
        y_val = np.polyval(p, x_val)
        return x_val, y_val
    else:
        index = np.argmax(y)
        return x[index], y[index]

def getMaxGauss(x, y, fit=True, valley=False, returnAll=False):
    org_y = y[:]
    if valley:
        y = np.max(org_y) - org_y
    if fit:
        maxy, miny = np.max(y), np.min(y)
        offset = miny
        idx = np.argmax(y)
        x_mean = x[idx]
        std = (np.max(x)-np.min(x))/6.0
        halfmax_vals = np.where(y>(maxy+miny)/2.0)[0]
        if len(halfmax_vals)>2:
            std = (x[halfmax_vals[-1]] - x[halfmax_vals[0]])/2.0
            x_mean = x[halfmax_vals].mean()
        # std = np.sum((y-offset)*(x-x_mean)**2)/np.sum(y-offset)
        amp = np.max(y) - np.min(y)
        p0 = [amp, x_mean, std, offset]
        p, cov, func = fitCurve('gaussian', x, y, p0)
        x_val = p[1]
        y_val = func(x_val, *p)
    else:
        idx = np.argmax(y)
        x_val = x[idx]
        y_val = y[idx]
    if valley:
        y_val = np.max(org_y) - y_val
    if fit and returnAll:
        return x_val, y_val, p, cov
    return x_val, y_val

#Specific dataset fit functions. These should probably be moved

def t1(dataset, timeRange=None):
    """Fit T1 for a single qubit dataset
    """
    timeUnit = Unit(dataset.indep[0][1])
    time = np.array(dataset[:,0])
    probabilities = np.array(dataset[:,1])
    #If no time range was specified, use the min and max time
    if timeRange is None:
        minTime = np.min(time)
        maxTime = np.max(time)
        timeRange = (minTime*timeUnit, maxTime*timeUnit)
    #Generate array mask for points that will be kept
    indexList = (time>timeRange[0][timeUnit]) * (time<timeRange[1][timeUnit])
    t = time[indexList]
    p=probabilities[indexList]
    #Fit the data to an exponential
    vGuess = [0.85, np.max(t)/2.0, 0.02]
    fits, cov, fitFunc = fitCurve('exponential', t, p, vGuess)
    T1 = fits[1]*timeUnit
    dT1 = cov[1]*timeUnit
    return AttrDict({'T1':T1,'dT1':dT1, 'fitFunc':fitFunc, 'fitParams':fits,
                     'indexList':indexList})

def ramseyTomo_noLog(dataset, T1=None, timeRange=None):
    """
    """
    #CHANGELOG
    # 13 April 2012
    # Copied from FluxNoiseAnalysis
    time = np.array(dataset[:,0])
    timeUnit = Unit(dataset.indep[0][1])
    probabilities = np.array(dataset[:,1:])

    if T1 is None:
        measure = dataset.parameters['measure'][0]
        qName = dataset.parameters['config'][measure]
        T1 = dataset.parameters[qName]['calT1']
    T1=float(T1[timeUnit])
    colors = ['.','r.','g.','k.']
    xp = probabilities[:,0]
    yp = probabilities[:,1]
    xm = probabilities[:,2]
    ym = probabilities[:,3]
    env = np.sqrt( (xp-xm)**2 + (yp-ym)**2)
    envScaled = env/np.exp(-time/(2*T1))
    #If time range for fitting has not been specified, ask user
    if timeRange is None:
        minTime = np.min(time)
        maxTime = np.max(time)
        timeRange = (minTime*timeUnit, maxTime*timeUnit)
    #Generate array mask for elements to be fitted
    indexList = (time>timeRange[0][timeUnit]) * (time<timeRange[1][timeUnit])
    t = time[indexList]
    e = envScaled[indexList]
    #Create fit function. We fit the unscaled data
    # vGuess = [0.85, 1000.0, 200.0, 0.05]
    vGuess = [np.max(envScaled), np.max(t)/2.0, np.max(t)/2.0, np.min(envScaled)]
    fits, cov, fitFunc = fitCurve('gaussianExp', t, e, vGuess)
    T2 = fits[2]*timeUnit
    dT2 = cov[2]*timeUnit
    Tphi = fits[1]*timeUnit
    dTphi = cov[1]*timeUnit
    return AttrDict({'T2':T2, 'dT2':dT2, 'Tphi':Tphi, 'dTphi':dTphi,
                     'fitFunc':fitFunc, 'fitParams':fits,
                     'envScaled':envScaled, 'indexList':indexList})

def ramseyExponential(dataset, T1=None, timeRange=None):
    """
    fitFunc is exponential decay together with oscillation
    """
    time = np.array(dataset[:, 0])
    timeUnit = Unit(dataset.indep[0][1])
    prob = np.array(dataset[:, 1])
    if timeRange is None:
        timeRange = (np.min(time)*timeUnit, np.max(time)*timeUnit)
    indexList = (time > timeRange[0][timeUnit]) * (time < timeRange[1][timeUnit])
    t = time[indexList]
    p = prob[indexList]
    ampGuess = 0.5*(np.max(p) - np.min(p))
    freqGuess = maxFreq(dataset[:,(0,1)], 4000, False)[timeUnit**-1]
    phaseGuess = 0.0
    T2Guess = np.max(time)/2.0
    offsetGuess = p[-1]
    vGuess = [ampGuess, freqGuess, phaseGuess, T2Guess, offsetGuess]
    fits, cov, fitFunc = fitCurve('exponentialCosine', t, p, vGuess)
    T2 = fits[3]*timeUnit
    dT2 =cov[3]*timeUnit
    freq = fits[1] * (timeUnit**-1)
    dfreq = cov[1] * (timeUnit**-1)
    return AttrDict({'T2': T2, 'dT2': dT2, 'freq': freq, 'dfreq': dfreq,
                     'fitFunc': fitFunc, 'fitParams': fits, 'indexList': indexList})


def ramseyZPulse(dataset, ampRange=None):
    amp = dataset.data[:,0]
    probabilities = dataset.data[:,1]
    if ampRange is None:
        minAmp = np.min(amp)
        maxAmp = np.max(amp)
        ampRange = (minAmp,maxAmp)
    #Generate array mask for points that will be kept
    indexList = (amp>ampRange[0]) * (amp<ampRange[1])
    a = amp[indexList]
    p=probabilities[indexList]
    #Fit the data to an exponential
    vGuess = [0.85, 16.0, 0.0, 0.5]
    fits, cov, fitFunc = fitCurve('cosine', amp, probabilities, vGuess)
    bestAmp = (1-fits[2])/(2*fits[1])
    return AttrDict({'bestAmp':bestAmp, 'fitFunc':fitFunc, 'fitParams':fits,
                     'indexList':indexList})


def squarePulse(dataset, topLen, transLen, timeRange=None,
                guesses=None):
    if guesses is None:
        guesses=[0.0,0.0,1.0,1.0]
    x = dataset.data[:,0]
    y = dataset.data[:,1]
    indepUnit = Unit(dataset.variables[0][0][1])
    topLen = topLen[indepUnit]
    transLen = transLen[indepUnit]
    if timeRange is None:
        minTime = np.min(x)
        maxTime = np.max(x)
        timeRange = (minTime,maxTime)
    indexMask = (x>timeRange[0][indepUnit]) * (x<timeRange[1][indepUnit])
    xSection = x[indexMask]
    ySection = y[indexMask]
    print transLen
    print topLen
    def fitFunc(x,p):
        return (-1.0 + p[1]+
                p[2]*0.5*erfc((x-(p[0]-topLen/2.0))/transLen) +
                p[3]*0.5*erf((x-(p[0]+topLen/2.0))/transLen)
                )
    fit, _ok = leastsq(lambda p: fitFunc(xSection,p)-ySection, guesses)
    return {'horzOffset':fit[0]*indepUnit,'vertOffset':fit[1],
            'fitFunc':fitFunc,'fit':fit}


def maxFreq(data, nfftpoints, plot=False):
    ts, ps = data.T
    y = ps - np.polyval(np.polyfit(ts, ps, 1), ts) # detrend
    timestep = ts[1] - ts[0]
    freq = np.fft.fftfreq(nfftpoints, timestep)
    fourier = abs(np.fft.fft(y, nfftpoints))
    maxFreq = abs(freq[np.argmax(fourier)])*1e3*MHz
    if plot:
        plt.plot(1000*np.fft.fftshift(freq), np.fft.fftshift(fourier))
        plt.xlabel('FFT Frequency (MHz)')
        plt.ylabel('FFT Amplitude')
    return maxFreq

def acStark(dataset):
    data = np.array(dataset)
    parameter = dataset.parameters
    measure = parameter['measure'][0]
    qName = parameter['config'][measure]
    qubit = parameter[qName]
    # chi = qubit['reasonatorFrequency1'] - qubit['resonatorFrequency0']
    # data of acStark: ampSquared, frequency, P0, P1
    ampSquared, freq, p1s = dstools.format2D(data[:,(0,1,3)])
    center_freq = []
    prob = []
    for p1 in p1s.T:
        f, p = getMaxGauss(freq, p1, fit=True)
        center_freq.append(f)
        prob.append(p)
    center_freq = np.array(center_freq)
    probs = np.array(prob)
    return probs, ampSquared, center_freq

def readoutSpectroscopy(dataset):
    data = np.array(dataset)
    delay, freq, p1s = dstools.format2D(data)
    center_freq = []
    prob = []
    for p1 in p1s.T:
        f, p = getMaxGauss(freq, p1, fit=True)
        center_freq.append(f)
        prob.append(p)
    center_freq = np.array(center_freq)
    probs = np.array(prob)
    return probs, delay, center_freq

def detuneT1(dataset):
    data = np.array(dataset)
    # dataset: z, delay, p0, p1
    zAmps, delay, probs = dstools.arrange2Dscan(data[:,(0,1,3)])
    T1s = []
    for prob in probs.T:
        vGuess = [np.max(prob), np.max(delay)/2.0, np.min(prob)]
        p, cov, func = fitCurve("exponential", delay, prob, vGuess)
        T1s.append(p[1])
    T1s = np.array(T1s)
    return zAmps, T1s

def detuneDephasing(dataset, gauss=False):
    """
    fitting T2 for detune dephasing scan, mention that the effect T1 is not removed.
    You should use 1./T*2 = 1./(1./T2 - 0.5/T1) to remove the effect of T1

    @param gauss: if True, use gaussian and exponential decay,
                  else use exponential decay
    @return: if gauss is True, return (zAmps, T2, Tphi)
             else, return (zAmps, T2)
    """
    data = np.array(dataset)
    # dataset: z, delay, P(+X), P(+Y), P(-X), P(-Y), Envelope
    zAmps, delay, probs = dstools.arrange2Dscan(data[:,(0,1,6)])
    if gauss:
        T2s = []
        Tphis = []
        for prob in probs.T:
            vGuess = [np.max(prob), np.max(delay)/2.0, np.max(delay)/4.0, np.min(prob)]
            p_fit, cov, func = fitCurve('gaussianExp', delay, prob, vGuess)
            T2s.append(p_fit[2])
            Tphis.append(p_fit[1])
        T2s = np.array(T2s)
        Tphis = np.array(Tphis)
        return zAmps, T2s, Tphis
    else:
        T2s = []
        for prob in probs.T:
            vGuess = [np.max(prob), np.max(delay)/2.0, np.min(prob)]
            p_fit, cov, func = fitCurve('exponential', delay, prob, vGuess)
            T2s.append(p_fit[1])
        T2s = np.array(T2s)
        return zAmps, T2s

