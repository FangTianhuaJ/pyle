import numpy as np
from numberanalysis import getVis, getMatrix, coherentanalysis
import myplotutils

def fresnelInt(ds, vis, coherent, wigner, session=None, minContrast=1.5,
               cmap=myplotutils.mycolormap('wrkbw'), vmin=-2.0/np.pi, vmax=2.0/np.pi):
    p0,p1 = getVis(ds, vis, session)
    if isinstance(wigner, tuple):
        data, info = wigner
    else:
        data, info = getMatrix(ds, dataset=wigner, session=session,default=0.0)
    
    rabifreq,maxima, amplscale, visibilities = \
        coherentanalysis(ds, dataset=coherent, session=session,
                         minContrast=minContrast, chopForVis=info['Imax'][0],
                         p0=p0, p1=p1, doPlot=False)
    t = np.arange(np.shape(data)[0])*info['Isteps'][0]*np.pi*rabifreq
    kernel = np.exp(1j*t**2/np.pi)
    kernel[0] *= 0.5
    kernel *= (t[1]-t[0])
    kernel *= (t < 2*np.pi)
    
    data = 8/np.pi**2/np.sqrt(1j)*np.sum(kernel[:,None,None]*(0.5-(data-p0)/(p1-p0)),axis=0)
    
    
    myplotutils.plot2d(np.transpose(data), extent=(amplscale*info['Imin'][1],
                         amplscale*info['Imax'][1],
                         amplscale*info['Imin'][2],
                         amplscale*info['Imax'][2]), aspect='equal',
           vmin=vmin, vmax=vmax, cmap=cmap)
