#Created: A while ago
#Author: Daniel Sank


#CHANGELOG

# 24 May 2011
# Added lots of documentation to make this module usable for other people.
# Did this so that I can upload to pyle.



import numpy as np
from pyle.util import structures as structures
from numpy import pi, linspace, cos, exp
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt

from labrad.units import Unit

MHz, GHz, ns, sec = (Unit(st) for st in ['MHz','GHz','ns','s'])
I = 1.0j


#Signal transforms

def hilbert(x):
    """Compute the Hilbert transform of a zero mean signal
    PARAMETERS
    x - 1D array like: signal you want to transform
    OUTPUT:
    1D array like: Hilbert transform of x
    COMMENTS:
    This function will throw away the last point if the total number of points
    in x is even.
    """
    #Force x to have an odd number of points. Kind of a hack.
    if len(x)%2 != 1:
        x = x[:-1]
    #Subtract mean
    x = x-np.mean(x)
    #Go to frequency space
    ft = fft(x)
    N = len(x)
    L = (N-1)/2
    #Separate positive and negative frequencies and multiply them by 90 phase
    #shift. The Hilbert transform puts a positive 90 degree phase shift on the
    #positive frequencies, and a negative phase shift on negative frequencies.
    pos = ft[1:L+1]*I
    neg = ft[L+1:]*-I
    #Recombine frequency space data
    ft = np.hstack((ft[0],pos,neg))
    #Go back to real space
    transformed = ifft(ft)
    return transformed

def envelope(x):
    """Computes the envelope of a signal with mean subtracted
    PARAMETERS:
    x - 1D array like: the signal of which you want to find the evelope
    OUTPUTS:
    1D array: Envelope of input signal
    COMMENTS:
    Computes envelope by contructing a 90deg phase shifted version of x
    using a Hilbert transform. The shifted signal, y, is combined with x to get
    and envelope via envelope=sqrt(x^2+y^2). Since the Hilbert transform
    function throws away one point if the input has an even number of points,
    y may have one less point than x. In this case we drop one point from x
    when computing the envelope
    """
    x = x - np.mean(x)
    y = hilbert(x)
    if len(x) != len(y): #x had a point removed by hilbert
        x=x[:-1] #Remove a point from y. This is kind of a hack.
    env = np.sqrt(x**2 + y**2)
    return env

def DFT(x,T):
    """
    Computes the discrete Fourier transform of a real signal and returns the
    spectrum, power spectral density, integrated power spectral density, and
    the correct physical frequency axis. Works for even and odd length inputs.
    Always remember that discretely sampled data suffers aliasing and spectral
    leakage. See Dan's writeup on the DFT for more information.

    INPUTS:
    T - Value:      the total physical time of your signal
    x - 1D array:   signal. ie, measured data points
 
    OUTPUT:
    AttrDict with one member for each of the quantities mentioned above
    dft - complex array:    the full spectrum of the DFT of the input signal
    frequencies - array:    frequencies of the power spectrum in Hz
    S - array:              Normalized power spectral density for each of the
                            frequency bins. Units are x^2/Hz.
    integrated - AttrDict:  
        frequencies - array:        List of frequencies
        cumulativePower - array:    Total power contained in all frequencies up
                                    to this point, ie.
                cumulativePower[i]  = \int_fmin^frequencies[i] S(f) df
        
    COMMENT ON NORMALIZATION:
    The normalization used by this code returns a truly physical power in units
    of x^2/Hz where x is the unit of the input signal. The total power
    represented by a single point is the value of that point multiplied by the
    width of the frequency bins in Hertz, which is equal to 1/T.
    
    A way to see this is to put as the input signal a perfect cosine:
    t = np.linspace(0,1,N)
    x = np.cos(2*np.pi*10*t)
    Then compute the DFT assuming that the data represents one second,
    dft1 = DFT(x,1.0)
    If you plot the result you'll see a single point up at 1/2. The bin width
    is 1/1.0sec = 1Hz, so the power in that point is 1/2, which is correct for
    a cosine.
    Now try the same frequency signal, but acquired for 2.0 seconds,
    t = np.linspace(0,1,2*N)
    x = np.cos(2*pi*20*t)
    dft2 = DFT(x,2.0)
    Now you'll see a single point at 1. The bin width in this case is 1/2 Hz,
    so multiplying for the total power still gives 1/2, which again is correct
    for a cosine.
    
    This seems weird, that the power level changes depending on the total data
    acquisition time, but if you think about it, for a perfect sinusoid this is
    exactly what should happen because the true power spectral density is a
    delta function. As the acquisition time goes up, the bins narrow in
    frequency and the height of the point correspondingly goes up.
    """

    output = structures.AttrDict()
    N = float(len(x))
    T = T[sec]
    dft = fft(x)/N
    output.dft = dft
    if N%2==0:  #if N is even
        output.frequencies = np.linspace(0,(N/2),(N/2)+1)/T
        output.S = 2*T*(abs(dft[0:(N/2)+1])**2)
    else:       #if N is odd
        output.frequencies = np.linspace(0,(N-1)/2,(N+1)/2)/T
        output.S = 2*T*(abs(dft[0:(N+1)/2])**2)
    output.integrated = cumulativeSpectralDensity(output.S,output.frequencies)
    return output

def dftBootstrapped(data,T,chunks):
    """ A DFT that is averaged by splitting the time series into chunks,
    computing the DFT of each chunk, and averaging corresponding frequency bins
    together.
    """
    T=T[sec]
    x=data.copy()
    #Find the time step per point
    dt = float(T)/(len(data)-1)
    #Find how many points to drop in order to make the length of the sequence
    #a multiple of the number of chunks we want.
    ptsToChop = len(x)%chunks
    #Drop the points
    if not ptsToChop==0:
        x = x[0:-ptsToChop]
    #Compute the total time of the part of the signal we kept.
    T = T-(dt*ptsToChop)
    chunkLength = len(x)/chunks
    results=[]
    #Compute the DFT for each chunk
    for i in range(chunks):
        xChunk = data[i*chunkLength:(i*chunkLength)+chunkLength]
        result = DFT(xChunk,float(T)*sec/chunks)
        results.append(result)
    frequencies = results[0]['frequencies']
    S = np.zeros(len(results[0].S))
    #Find the average power
    for result in results:
        S = S+result['S']
    S = S/chunks
    integrated = cumulativeSpectralDensity(S,frequencies)
    return structures.AttrDict({'S':S, 'frequencies':frequencies, 'integrated':integrated})

def DFT_interleaved(dataIn, T, smooth=None):
    """Computes a power spectrum by splitting the time series into two
    interleaved series, computing the DFT of each one, and then combining them
    as dft1*conjugate(dft2) to get the spectral density.
    
    INPUTS:
    T - Value:      total signal acquisition time
    data - array:   the signal you want to transform
    smooth - tuple: (type of smoothing, windowsize)
        Allowed types of smoothing are '1f' and 'flat'
        
    OUTPUT
    AttrDict:
        S - array:              Spectral density in x^2/Hz
        frequencies - array:    frequency axis in Hz
        integrated - AttrDict:  Integrated power
            frequencies - array:    frequency axis in Hz
            
    """
    T=T[sec]
    N = len(dataIn)
    dt = float(T)/(N-1) #Time step
    print 'N before processing: %d' %N
    #Length of data has to be a multiple of 4
    ptsRemoved = N%4
    N = N - ptsRemoved
    data = dataIn[0:N]
    T = T-(dt*ptsRemoved)
    print 'N after modulo 4: %d' %N
    #Split the signal into two signals
    dataOdd = data[0::2]
    dataEven = data[1::2]
    n = N/2
    print 'n = %d' %n
    #Make frequency axis
    frequencies = np.linspace(0,(n/2),(n/2)+1)/T
    #-DEBUGGING-Make sure data lengths make sense.
    assert n == len(dataOdd)
    assert n == len(dataEven)
    #Window the two signals
    window = hann(n)
    dataOddWindowed = dataOdd*window
    dataEvenWindowed = dataEven*window
    #Compute DFT
    ftOdd = fft(dataOddWindowed)
    ftEven = fft(dataEvenWindowed)
    #Combine transforms into a pseudo power spectrum. SIGNAL^2/HERTZ!
    S = (2.0*T*ftOdd[0:(n/2)+1]*np.conjugate(ftEven[0:(n/2)+1]))/(n**2)
    #Average neighboring bins before taking the modulus, this should eliminate the effect of white noise
    windowFunc = lambda x: exp(-(x**2)/25)
    windowSizes = 10+np.zeros(n)
    S = smooth_v0(S,windowFunc,windowSizes)
    #Take modulus
    S = np.abs(S)
    #Normalize for the Hann window function
    S = S/0.375
    #Smooth the result
    if isinstance(smooth,tuple):
        if smooth[0] == '1f':
            windowSizes = (np.linspace(0,n,n)**2)/float(n)
            smoothFunc = lambda x: exp(-(x**2)/smooth[1])
            S = smooth_v0(S,smoothFunc,windowSizes)
        elif smooth[0] == 'flat':
            windowSizes = smooth[1]+np.zeros(n)
            smoothFunc = lambda x: exp(-(x**2)/smooth[2])
            S = smooth_v0(S,smoothFunc,windowSizes)
    elif smooth is not None:
        raise Exception('Smoothing type not recognized')        
    integrated = cumulativeSpectralDensity(S,frequencies)
    output = structures.AttrDict({'frequencies':frequencies,'S':S,'integrated':integrated})
    return output

def dftInterleavedBootstrapped(dataIn, T, chunks=1, smooth=None):
    T=T[sec]
    dt = float(T)/(len(dataIn)-1)
    ptsToChop = len(dataIn)%chunks
    if not ptsToChop==0:
        data = dataIn[0:-ptsToChop]
    else:
        data = dataIn
    T = T-(dt*ptsToChop)
    chunkLength = len(data)/chunks
    results=[]
    for i in range(chunks):
        dataChunk = data[i*chunkLength:(i*chunkLength)+chunkLength]
        result = DFT_interleaved(dataChunk, (float(T)*sec)/chunks, smooth=smooth)
        results.append(result)
    frequencies = results[0]['frequencies']
    S = np.zeros(len(results[0].S))
    for result in results:
        S = S+result['S']
    S = S/chunks
    integrated = cumulativeSpectralDensity(S,frequencies)
    output = structures.AttrDict({'S':S, 'frequencies':frequencies, 'integrated':integrated})
    return output

def paddedDFT(x,numPadPoints,timeStep):
    """paddedDFT(x,numPadPoints,timeStep)
    Pads the signal x with numPadPoints zeros to do spectral interpolation.
    Note that because the length of the sigal is changed by the padding
    the vertical scale of the output spectrum is meaningless."""
    raise Exception('Depricated')
    x = x-np.mean(x)
    n = np.size(x)
    t = timeStep*n
    T = t+(numPadPoints*timeStep)
    dataWindowed = (x-np.mean(x))*hann(n)
    dataExtended = np.append(dataWindowed,np.zeros(numPadPoints))
    plt.plot(dataExtended)
    S = DFT(T,dataExtended)
    return S

def cumulative(a):
    N=np.size(a)
    integrated = np.zeros(N)
    integrated[0]=a[0]
    for i in map(lambda x: x+1,range(N-1)):
        integrated[i] = integrated[i-1]+a[i]
    return integrated

def hann(N):
    """hann(N)
    Returns the N point Hann window

    INPUTS
    N - number of points

    OUTPUT
    length N array: containing Hann window
    """
    indices = linspace(0,N-1,N)
    return 0.5*(1-cos((2*pi*indices)/(N-1)))

def blackmanHarris(N):
    n = linspace(0,N-1,N)
    a0=0.35875
    a1=0.48829
    a2=0.14128
    a3=0.01168
    return a0-a1*cos(2*pi*n/(N-1))+a2*cos(4*pi*n/(N-1))-a3*cos(6*pi*n/(N-1))
    
    
## Helpers
    
def insertZeros(series, patches):
    """Insert a specified numbers of zeroes into a series at specified locations.

    PARAMTERS
    series - the data you're trying to fix
    patches - [(index, numberZeros),...]
    
    COMMENTS
    There is probably a slick way to do this without a 'for' loop. If you know how
    please tell me! :)
    """
    ptsAdded = 0
    for index, number in patches:
        index = index+ptsAdded
        series = np.hstack((series[0:index+1],np.zeros(number),series[index+1:]))
        ptsAdded+=number
    return series

def findPatches(series, step):
    """Finds list indices at which a series has missing elements and finds out
    how many elements are missing at each of these places.
    OUTPUTS
    list of (index,missing) tuples. Each tuple indicates that there are missing
    entries missing after element at index. For example, for
    series = [1,2,3,6,7,11]
    the result is [(2,2), (4,3)]
    indicating that after the second entry there are two entries missing, and
    after the fourth entry there are three entries missing.
    """
    series = np.array(series, dtype = np.uint32)
    shifted = np.hstack((series[1:],0))
    diffs = (shifted-series)[:-1]
    indices = np.nonzero(diffs>step)[0] #array of indices
    Indices = indices
    Missed=[]
    #Each element i in indices is such that series[i+1]-series[i]>step
    for counter in range(len(indices)):
        index = indices[counter]
        #Find out how many points were missed
        missed = ((series[index+1]-series[index])/step)-1
        Missed.append(missed)
    return zip(Indices,Missed)

def smooth_v0(data, windowFunc, windowSizes):
    data = np.array(data)
    L = len(data)
    smoothed = np.zeros(L)
    for i,point in enumerate(data):
        winSize = np.floor(windowSizes[i])
        if winSize == 0:                    #If the window size is zero
            smoothed[i]=point               #Don't average this point
            continue                        #and move on to the next point
        indicesBelow = winSize
        indicesAbove = winSize
        lowerIndex = i-winSize
        upperIndex = i+winSize
        #If the lower index is less than zero, we have to reduce indicesBelow
        #to make sure we don't try to index at less than zero.
        if lowerIndex <0:
            indicesBelow = winSize+lowerIndex
            lowerIndex=0
        #Similarly for indices above
        if upperIndex > L-1:
            indicesAbove = winSize-(upperIndex-(L-1))
            upperIndex = L-1
        #Now we know how many points will be involved in the average, so we can
        #make a list of the indices of the data we want to average
        indepLen = indicesAbove+indicesBelow+1
        #Create the window
        indepWindow = np.linspace(-indicesBelow,indicesAbove,indepLen)
        window = windowFunc(indepWindow)
        windowNorm = np.sum(window)

        dep = data[lowerIndex:upperIndex+1]
##        print 'indicesBelow: %d' %indicesBelow
##        print 'indicesAbove: %d' %indicesAbove
##        print 'Length of dep: %d' %len(dep)
##        print 'Length of window: %d' %len(window)
        
        smoothed[i] = np.sum(window*dep)/windowNorm
    return smoothed

def findExtrema(x, width=2):
    """Finds the indices extrema in a listlike collecion of data

    This function is really stupid. At each point in x it checks width points to the left and right,
    and if these points are all either larger or lesser than the current point in x, the current
    point is declared an extremum.
    """
    extrema=[]
    for i,elem in enumerate(x):
        if i-width<0:
            lowerIndex = 0
        else:
            lowerIndex = i-width
        if i+width > len(x)-1:
            upperIndex = len(x)-1
        else:
            upperIndex = i+width
        hasHigher=False
        hasLower=False
        for val in x[lowerIndex:upperIndex]:
            if val>elem:
                hasHigher=True
            if val<elem:
                hasLower=True
        if hasHigher and hasLower:
            pass
        else:
            extrema.append(i)
    return extrema
    
def getCrestIndices(x, width=2):
    extrema = findExtrema(x, width=width)
    indices=set()
    for extremum in extrema:
        for count in range(2*width+1):
            dIndex = count-width
            value = extremum+dIndex
            if value>=0 and value<len(x):
                indices.add(extremum+dIndex)
    return list(indices)
    
def indices2ArrayMask(indices, length):
    mask = np.array([False]*length,dtype='bool')
    for index in indices:
        mask[index]=True
    return mask

def logSpacedIndices(totalLength, numPoints):
    indices = np.int32(np.floor(np.logspace(0,np.log10(totalLength),numPoints)))
    indices[-1] = indices[-1]-1
    return indices
    
def arrayMask2Indices(mask):
    raise Exception('Not implemented')
    
def integrateSpectralDensity(S,frequencies):
    raise Exception('DEPRICATED. Use cumulativeSpectralDensity instead')
    #Make array of frequency bin widths
    binWidths = (np.hstack((frequencies,0))-np.hstack((0,frequencies)))[1:-1]
    SAveraged = ((np.hstack((0,S))+np.hstack((S,0)))[1:-1])/2
    return sum(SAveraged*binWidths)
    
def cumulativeSpectralDensity(inputS,inputFrequencies):
    S=inputS[1:-1]
    freqCenters = (np.hstack((inputFrequencies,0))+np.hstack((0,inputFrequencies)))[1:-1]/2
    freqBinWidths = (np.hstack((freqCenters,0))-np.hstack((0,freqCenters)))[1:-1]
    cumulative = np.cumsum(freqBinWidths*S)
    return structures.AttrDict({'frequencies':inputFrequencies[1:-1],'cumulativePower':cumulative})
    
    