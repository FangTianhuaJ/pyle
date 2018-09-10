import numpy as np
from scipy.optimize import leastsq
from matplotlib import cm
import matplotlib.pyplot as plt

from pyle.plotting import dstools
from pyle.fitting import fitting
import util


def _getstepfunc(d, height, start=5, plot=False, ind=None, dep=None, w=0.01, timeconstants=1):
    """Determine z-pulse time constant.
    
    INPUT PARAMETERS
    d - 2D array: data in format [time, z offset, prob.]
    height - float: height of z-pulse
    start - int or float: time at which to start fit
    plot - bool: Whether to plot data (data in grayscale + fit)
    ind: Independent variable (i.e., time) - use dv.variables()
    dep: Dependent variable (i.e., z offset) - use dv.variables() 
    w - float: Scaling factor for Gaussian data smoothing
    timeconstants - int: How many RC time constants to use.
    """
    
    # Plot raw data as grayscale
    if plot:
        dstools.plotDataset2Dexact(d, ind, dep, cmap=cm.get_cmap('gray_r'))
        
    # Determine z offset which gives maximum prob for a given time.
    # Determines z amplitude vs. time.
    # 1) Determines in which rows of d the time was changed.
    d = d[np.argsort(d[:,0]),:]    
    boundaries = np.argwhere(np.diff(d[:,0]) > 0)[:,0]
    boundaries = np.hstack((0, boundaries, len(d)))
    n = len(boundaries)-1
    result = np.zeros((n, 2))
    result[:,0] = d[boundaries[:-1],0]
    for i in np.arange(n):
        # 2) For each time, set p to be array of [z offset, prob].
        # Then perform Gaussian smoothing and find maximum.
        p = d[boundaries[i]:boundaries[i+1],:]
        p = p[np.argsort(p[:,1]),:]
        result[i,1] = maximum(p[:,1], p[:,2], w=w)
    # 3) Result is [time, z offset for max prob]. Only keep data for
    # z offset below 0.2 and times above 0.
    good = np.argwhere((abs(result[:,1]) < 0.2) * (result[:,0] > 0))[:,0]
    result = result[good,:]
    # 4) Plot z offsets for each time on grayscale plot.
    if plot:
        plt.plot(result[:,0], result[:,1], 'w.')
        
    # Determine time constants for z pulse.
    # 1) Only keep data for times greater than start.
    result = result[np.argwhere(result[:,0] >= start)[:,0],:] 
    # 2) Fit to sum of exponentials (# of exp given by timeconstants).
    t = np.linspace(result[0,0], result[-1,0], 1000)
    def fitfunc(t, p):
        return (t > 0) * (p[0] + np.sum(p[1::2,None]*np.exp(-p[2::2,None]*t[None,:]), axis=0))
    def errfunc(p):
        return fitfunc(result[:,0], p) - result[:,1]
    p = np.zeros(2*timeconstants+1)
    p[2::2] = np.linspace(0.001, 0.3, timeconstants)
    p, _ = leastsq(errfunc, p)
    # 3) Print amplitude and time constant for each exponential to screen.
    print 'Pulse relaxation:'
    for i in range(timeconstants):
        print '    amplitude: %g %%' % (100.0*p[1+2*i]/float(height))
        print '    time constant: %g ns' % (1.0/p[2+2*i])
    print ' RMS error: %g' % np.sqrt(np.average(errfunc(p)**2))
    # 4) Plot fit function to grayscale plot.
    if plot:
        plt.plot(t, fitfunc(t, p), 'w-')
    p[0] = float(height)
    return p, lambda t: fitfunc(t, p)


def getstepfunc(dataset, sample=None, save=False, start=5, plot=False, w=0.01, timeconstants=1):
    """Determine z-pulse time constant (from data vault file).
    Uses _getstepfunc.
    
    INPUT PARAMETERS
    dataset - the dataset, contains data and parameters
    sample: Object defining qubits to measure, loaded from registry.
    save - bool: Whether to save calibration key to registry
    start - int or float: time at which to start fit
    plot - bool: Whether to plot data (data in grayscale + fit)
    w - float: Scaling factor for Gaussian data smoothing
    timeconstants - int: How many RC time constants to use.
    """
    d = np.array(dataset)
    ind, dep = dataset.indep, dataset.dep
    parameters = dataset.parameters
    height = parameters['step height']
    p, func = _getstepfunc(d, height, start=start, plot=plot, ind=ind, dep=dep,
                           w=w, timeconstants=timeconstants)
    if save:
        if sample is None:
            raise Exception('Can not save if no sample provided.')
        else:
            sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
            Q = Qubits[parameters['measure'][0]]
            Q['settlingRates'] = p[2::2]
            Q['settlingAmplitudes'] = p[1::2]/float(height)
    return p, func


def maximum(x, y, w=0.01):
    """Gaussian smooth data and then find x to maximize y.
    
    INPUT PARAMETERS
    x: array of x values
    y: array of y values
    w - float: Scaling factor for Gaussian data smoothing
    """
    n = len(x)
    dx = (x[-1] - x[0]) / (n-1)
    
    # Gaussian smooth data
    f = np.exp(-np.linspace(-n*dx/w, n*dx/w, 2*n-1)**2)
    smooth = np.real(np.convolve(y, f, 'valid')) / np.real(np.convolve(1-np.isnan(y), f, 'valid'))
    #plt.plot(smooth)
    i = np.argmax(smooth)
    if i > 0 and i < n-1:
        yl, yc, yr = smooth[i-1:i+2]
        d = 2.0 * yc - yl - yr # Second derivative of y(x)
        if d <= 0:
            return 0
        d = 0.5 * (yr - yl) / d
        d = np.clip(d, -1, 1)
    else:
        d = 0
        print 'warning: no maximum found'
    return (1-abs(d))*x[i] + d*(d>0)*x[i+1] - d*(d<0)*x[i-1]
    
                         
