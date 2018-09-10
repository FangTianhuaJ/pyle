import numpy as np

from pyle.plotting import dstools
import myPlotUtils


def xyz2rtp(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2), np.arctan2(np.sqrt(x**2+y**2),z), np.arctan2(y,x)


def octomo2rtp(I, X, Y, Z, mX, mY):
    return xyz2rtp(X-mX, Y-mY, I-Z)


def rtp2xyz(r, theta, phi):
    return r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)


def hsl2rgb(h, s, l):
    q = (l * (1+s)) * (l < 0.5) + (l + s - l*s) * (l >= 0.5)
    p = 2 * l - q
    def channel(t):
        t = (t % 1) * 6
        return (p + (q-p) * t) * (t < 1) + \
               q * (t>=1) * (t<3) + \
               (p + (q-p) * (4-t)) * (3 <= t) * (t < 4) + \
               p * (t>= 4)

    ax = np.arange(1, len(np.shape(h))+2)
    ax[-1] = 0
    return np.transpose(np.array([channel(h+1./3.), channel(h), channel(h-1./3.)]), axes=ax)


def plotTomo2D(ds, dataset=None, session=None, Xindep=0, Yindep=1):
    data = dstools.getMatrix(ds, dataset, session, Xindep=Xindep, Yindep=Yindep,
                             dependent=np.array([1,2,0,3,4,5]))
    info = data[1]
    data = data[0]
    d = data[:,:,0:3] - data[:,:,3:6]
    s = data[:,:,0:3] + data[:,:,3:6]
    print max(data[:,:,5]),min(data[:,:,5])
    print max(data[:,:,2]),min(data[:,:,2])
    a = np.average(s)
    x = d[:,:,0]/100
    y = d[:,:,1]/100
    z = d[:,:,2]/100
    print np.min(z), np.max(z)
    r, t, p = xyz2rtp(x, y, z)
    myPlotUtils.plot2d(hsl2rgb(p, r, 0.5*(1+z)), info)
    
    
def getrtp(ds, dataset=None, session=None, p0=0.0, p1=100.0):
    data = dstools.getDataset(ds, dataset, session)
    data = (np.average(data,axis=0)-p0) / (p1-p0)
    d = (data[4:7]-data[1:4])[[1,2,0]]
    print d
    return xyz2rtp(*d)

