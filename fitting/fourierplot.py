#Originally from Max Hoffhienz
#First edited by Haohua Wang
#Second revision by Erik Lucero

from copy import copy

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq

from pyle.plotting.dstools import getMatrix, getDataset, columns2Matrix

#import pdb #uncomment if you want to use debug session in python

#session = '080108 - Max - 071209r7c6'
#cmap = cm.gray_r
cmap = None



def minPos(l, c, r):
    """Calculates minimum of a parabola to three equally spaced points.
    The return value is in units of the spacing relative to the center point.
    It is bounded by -1 and 1.
    """
    d = l + r - 2.0*c
    if d <= 0:
        return 0
    d = 0.5*(l-r)/d
    d = np.clip(d, -1, 1)
    return d


def plotFourier2D(data_server, dataset=None, Xindep=0, Yindep=1, vmin=0, vmax=None, cmap=None):
    m = getMatrix(data_server, dataset=dataset, independentColumns=[Xindep, Yindep], default=0.0)
    info = m[1]
    m = np.transpose(m[0])
    h, l = np.shape(m)
    n = 10000
    data = np.zeros((h,n),dtype=float)
    t = np.arange(l)
    plt.figure()
    for i in xrange(h):
        data[i,0:l] = (m[i] - np.polyval(np.polyfit(t, m[i], 1), t))

    data = abs(np.fft.rfft(data))

    Xstep = 1000.0/n
    Xmin = 0
    Xmax = 500 - info['Xstep']
    
    plt.imshow(data, aspect='auto', origin='lower',
               interpolation='nearest',
               extent=(Xmin - 0.5 * Xstep,
                       Xmax + 0.5 * Xstep,
                       info['Ymin'] - 0.5 * info['Ystep'],
                       info['Ymax'] + 0.5 * info['Ystep']),
               vmin=vmin, vmax=vmax, cmap=cmap)
    plt.xlabel('Rabi frequency (MHz)')
    plt.ylabel(info['Yname'])
    return data
  

def fitswap(data, axis=0, return_fit=False):
    """Fits swap data to find coupling frequency.
    
    Data is an array giving population versus time delay in ns
    on the first independent axis, and z-pulse amplitude on the
    second independent axis.
    
    Returns a tuple of fitted swap frequency in MHz, and z-pulse
    amplitude for resonant swapping.
    """
    
    plt.ioff()
    plt.clf()
    data, info = columns2Matrix(data)
    n = 10000
    avg = np.average(data, axis=axis)
    if axis == 1:
        infoAxis = axis
        axis = 0
        data = data.T
    fdata = abs(np.fft.rfft(data-avg[None,:], n=10000, axis=axis))
    finfo = info.copy()
    finfo['Isteps'] = copy(info['Isteps'])
    finfo['Imin'] = copy(info['Imin'])
    finfo['Imax'] = copy(info['Imax'])
    finfo['Isteps'][infoAxis] = 1000.0/n/info['Isteps'][infoAxis]
    finfo['Imin'][infoAxis] = 0.0
    finfo['Imax'][infoAxis] = 500.0/info['Isteps'][infoAxis]
    freqs = np.argmax(fdata, axis=axis) * finfo['Isteps'][infoAxis]
    good = np.argwhere(freqs > 10)[:,0]
    ampl = np.max(fdata, axis=axis)
    zpas = np.linspace(info['Imin'][1-infoAxis], info['Imax'][1-infoAxis], len(freqs))
    freqs = freqs[good]
    ampl = ampl[good]
    zpas = zpas[good]
    avg = avg[good]
    def fitfunc(zpa, p):
        return np.sqrt(p[0]**2 + (np.polyval(p[1:], zpa))**2) # fit a hyperbola
    def errfunc(p):
        return (freqs - fitfunc(zpas, p))*ampl

    p = [20.0, -30.0, zpas[np.argmin(freqs)]]
    p, ok = leastsq(errfunc, p)
    p[0] = abs(p[0])
    
    # plot raw swap data
    ax = plt.subplot(311)
    plt.imshow(data, aspect='auto', origin='lower', cmap=cmap,
               interpolation='nearest',
               extent=(info['Imin'][1-infoAxis] - 0.5 * info['Isteps'][1-infoAxis],
                       info['Imax'][1-infoAxis] + 0.5 * info['Isteps'][1-infoAxis],
                       info['Imin'][infoAxis] - 0.5 * info['Isteps'][infoAxis],
                       info['Imax'][infoAxis] + 0.5 * info['Isteps'][infoAxis]), vmin=0)
    
    # plot fourier transform with fit
    plt.subplot(312, sharex=ax)
    plt.imshow(fdata, aspect='auto', origin='lower', cmap=cmap,
               interpolation='nearest',
               extent=(finfo['Imin'][1-infoAxis] - 0.5 * finfo['Isteps'][1-infoAxis],
                       finfo['Imax'][1-infoAxis] + 0.5 * finfo['Isteps'][1-infoAxis],
                       finfo['Imin'][infoAxis] - 0.5 * finfo['Isteps'][infoAxis],
                       finfo['Imax'][infoAxis] + 0.5 * finfo['Isteps'][infoAxis]), vmin=0)
    plt.plot(zpas, freqs, 'w+', mew=2)
    plt.plot(zpas, fitfunc(zpas, p), 'w-', lw=2)
    
    # plot excitation probability and average
    plt.subplot(313, sharex=ax)
    plt.plot(zpas, avg, label='mean excitation')
    plt.plot(zpas, ampl/np.shape(data)[1-axis], label='max excitation')
    plt.legend()
    
    plt.show()
    plt.ion()

    if return_fit:
        return p[0], -p[2]/p[1], p[1]
    else:
        return p[0], -p[2]/p[1]


def plotFourier1D(data_server, dataset=None):
    n = 1000000
    points = getDataset(data_server, dataset=dataset)
    step = points[1,0] - points[0,0]
    plt.figure()
    plt.subplot(211)
    plt.plot(points[:,0], points[:,1])
    plt.xlabel('time (ns)')
    plt.ylabel('P1 (%)')
    l = np.shape(points)[0]
    fourier = np.zeros(n, dtype=float)
    fourier[0:l] = (points[:,1] - np.polyval(np.polyfit(points[:,0], points[:,1], 1), points[:,0]))*np.hanning(l)
    fourier = abs(np.fft.rfft(fourier))
    fourier /= np.max(fourier)
    plt.subplot(212)
    plt.plot(np.arange(len(fourier))*1000.0/step/n, fourier)
    plt.xlim((0,150))
    m = np.argmax(fourier)
    m = (float(m) + minPos(fourier[m-1], fourier[m], fourier[m+1])) / step / n
    
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Probability (a.u.)')
    print 'Maximum at %g MHz' % (m*1000)
    return m
 
