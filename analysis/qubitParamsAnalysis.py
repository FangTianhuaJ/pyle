import numpy as np
from numpy import pi
from scipy.optimize.zeros import bisect
from pylab import plot, xlabel, ylabel, ioff, ion

from labrad.types import Value

from dstools import getMatrix
from myplotutils import plot2d
import physcon as SI

r = 'r'
l = 'l'
#branches = [{r:-0.95},{l:-1.85,r:0.05},{l:-0.85,r:1.05},{l:0.2,r:2.05},{l:1.2}]
branches = [{r:-0.2},{l:-1.7,r:1.75},{l:0.2}]

def fitspectroscopy(ds, dataset=None, session=None, steps=None, order=2, FBrange=None):
    spec, info = getMatrix(ds, dataset=dataset, session=session, default=0.0, steps=steps)
    spec = np.transpose(spec)
    cf = float(Value(1, info['Yunits'])['Hz'])*2*pi/0.9
    cfb = float(Value(1, info['Xunits'])['V'])
    freq = (info['Ymin'] + info['Ystep'] * np.argmax(spec, axis=0)) * cf
    fb = np.linspace(info['Xmin']*cfb, info['Xmax']*cfb, len(freq))
    if range is not None:
        w = np.argwhere((fb >= FBrange[0]) & (fb <= FBrange[1]))[:,0]
        print np.shape(w)
        fb = fb[w]
        freq = freq[w]
    
    poly = np.polyfit(fb, freq**4, order)
    ioff()
    plot2d(spec, info, vmin=0, vmax=50)
    plot(fb/cfb, np.polyval(poly, fb)**0.25/cf, 'k')
    xlabel('%s (%s)' % (info['Xname'], info['Xunits']))
    ylabel('%s (%s)' % (info['Yname'], info['Yunits']))
    ion()
    return poly, info
     

def qubitparams(branches, ds=None, dataset=None, session=None, loopinductance=720e-12, FBrange=None, order=2, steps=None):
    l = len(branches)
    left = []
    right = []
    for i, b in enumerate(branches):
        if 'l' in b:
            left += [[i,b['l']]]
        if 'r' in b:
            right += [[i,b['r']]]
    left = np.array(left)
    right = np.array(right)

    phi0 = (len(left)*np.polyfit(left[:,0], left[:,1], 1)[0] +
            len(right)*np.polyfit(right[:,0], right[:,1], 1)[0])/\
            (len(left)+len(right))
    phiC = 0.5 * (np.average(right[:,1] - right[:,0] * phi0) -
                  np.average(left[:,1] - left[:,0] * phi0))
    delta = bisect(lambda d: 2*pi*phiC/phi0 - d + np.tan(d), 0.51*pi, 1.49*pi)
    Lj0 = -loopinductance * np.cos(delta)
    I0 = 0.5*SI.hbar/SI.e/Lj0
    print 'Flux quantum corresponds to %g V fluxbias' % phi0
    print 'Chritical flux:  %g phi0, corresponds to %g V' % (phiC/phi0,phiC)
    print 'critical phase   %g' %  delta
    print 'Loop inductance  %g pH' % (loopinductance * 1e12)
    print 'Lj0              %g pH'  % (Lj0 * 1e12)
    print 'critical current %g uA' %  (I0*1e6)
    
    if dataset is not None:
        poly, info = fitspectroscopy(ds, dataset, session, steps=steps, order=order, FBrange=FBrange)
        cfb = float(Value(1, info['Xunits'])['V'])
        #where the frequency becomes 0?
        stepedge = np.roots(poly)
        #pick the closest one to the scanned range
        stepedge = stepedge[np.argmin(abs(stepedge-0.5*cfb*(info['Xmin'] + info['Xmax'])))]
        C = 1.0/loopinductance*np.sqrt(4*pi*np.tan(delta)/poly[-2]/phi0)
        print poly
        print 'stepedge:        %g V' % stepedge
        print 'capacitance:     %g pF' % (C*1e12)
        print 'maximum frequency (center of branch): %g GHz' % (1e-9/2/pi/np.sqrt(Lj0*C))
        
        
    

    

    
    
