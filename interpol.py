import numpy as np

# signal processing routines which aren't in scipy/numpy, or are very slow
#
# LOG
#
# July 2013 - R. Barends
# added interp1d_cubic and moving_average

def interp1d_cubic(x,h,bounds_error=False,fill_value=None):
    """Fast cubic interpolator (slightly faster than linear version of scipy interp1d; much, much faster than cubic version of scipy interp1d). 
    Returns a function in the same fashion as interp1d works.
    Uses linear interpolation at the edges.
    If bounds_error is True, an error is raised if values are outsde of the range.   
    Otherwise, it returns the fill_value if set, or values at the edges outside of the range.
    x needs to be equidistant, and can be monotonically increasing or decreasing. RB.
    """    
    #check whether x is equidistant:
    diff = x[1:]-x[:-1]
    machineprecision=1e-11
    diffrel=abs(diff-diff[0])>machineprecision #if it is equidistant, it should be zero within the machine precision
    if diffrel.any():
        raise Exception('x is not equidistant')
    xstart=x[0]
    xlen=len(x)
    dx=1.*(x[1]-x[0])
    if type(h) is not np.ndarray:
        #we need a numpy array
        h=1.*np.array(h)
    def func(xdet):
        if type(xdet) is not list and type(xdet) is not np.ndarray:
            xdet=np.array([xdet])        
        yout=np.zeros(np.alen(xdet)).astype(h.dtype) #predefine in the same type as h (old:complex)
        x2 = (xdet-xstart)/dx #map xdet onto h index: x ->   (x-xstart)/dx = 0... length
        
        #indices outside of the range
        xdet_idx = x2<0 #maps which index in x2 it is
        if xdet_idx.any():
            if bounds_error:
                raise Exception('interpolation outside of range')
            x2_idx = x2[ xdet_idx ] #maps x2 to x index
            h_idx = np.array(x2_idx).astype(int) #maps which h,x to take
            if fill_value is None:
                yout[xdet_idx]=h[0]
            else:
                yout[xdet_idx]=fill_value
        xdet_idx = x2>(xlen-1) #maps which index in x2 it is
        if xdet_idx.any():
            if bounds_error:
                raise Exception('interpolation outside of range')        
            x2_idx = x2[ xdet_idx ] #maps x2 to x index
            h_idx = np.array(x2_idx).astype(int) #maps which h,x to take
            if fill_value is None:            
                yout[xdet_idx]=h[xlen-1]
            else:
                yout[xdet_idx]=fill_value                
            
        #indices on the rim: linear interpolation
        xdet_idx =  np.logical_and(x2>=0,x2<1) #maps which index in x2 it is
        if xdet_idx.any():
            x2_idx = x2[ xdet_idx ] #maps x2 to x index
            h_idx = np.array(x2_idx).astype(int) #maps which h,x to take        
            yout[xdet_idx]=(h[1]-h[0])*x2_idx  + h[0]
        xdet_idx =  np.logical_and(x2>=(xlen-2),x2<=(xlen-1)) #maps which index in x2 it is
        if xdet_idx.any():
            x2_idx = x2[ xdet_idx ] #maps x2 to x index
            h_idx = np.array(x2_idx).astype(int) #maps which h,x to take        
            yout[xdet_idx]=(h[xlen-1]-h[xlen-2])*(x2_idx-h_idx[0])  + h[xlen-2]
            
        #indices inside the range: cubic interpolation        
        xdet_idx = np.logical_and(x2>=1,x2<(xlen-2)) #maps which index in x2 it is
        if xdet_idx.any():        
            x2_idx = x2[ xdet_idx ] #maps x2 to x index
            h_idx = np.array(x2_idx).astype(int) #maps which h,x to take
            hp2=h[h_idx+2]
            hp1=h[h_idx+1]
            hp0=h[h_idx]
            hm1=h[h_idx-1]     
            d=hp0
            c=(hp1-hm1)/2.
            b=(-hp2+4*hp1-5*hp0+2*hm1)/2.
            a=(hp2-3*hp1+3*hp0-hm1)/2.
            xi=(x2_idx - h_idx)
            yout[xdet_idx]=((a * xi + b) * xi + c) * xi + d
            
        return np.array(yout)          
    return func

def moving_average(x,m):
    """Moving average on x, with length m. Expects a numpy array for x. Elements are given by
    y[i] = Sum_{k=0..m-1}   y[l] / m
    with l=i-fix(m/2)+k between 0 and length(x)-1. RB."""
    n=np.alen(x)
    before=-np.fix(int(m)/2.0)
    y=[]
    for i in np.arange(len(x)):
        a=0.0
        for tel in np.arange(int(m)):
            idx=i+before+tel
            if idx<0:
                idx=0
            elif idx>=n:
                idx=n-1
            a += x[idx]/np.float(m)
        y.append(a)
    return np.array(y)

    
"""
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp1d
def test(): 
    x=np.linspace(0,2,2001)
    y=np.sin(6.23*x)*x**2 +1j*np.cos(10*x) +0.1*np.random.rand(np.alen(x))
    x2=np.linspace(-3,3,30001)
    t=time.time()
    y2=interp1d_cubic(x,y)(x2)
    print time.time()-t

    t=time.time()
    yy=interp1d(x,y,'linear',bounds_error=False)(x2)
    print time.time()-t


    plt.figure()
    print len(x2),len(y2)
    plt.plot(x,np.real(y),'k.',x2,np.real(y2)) 
    plt.plot(x,np.imag(y),'k.',x2,np.imag(y2)) 
    plt.show()
"""