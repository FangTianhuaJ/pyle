from pyle.plotting.dstools import getMatrix
from pyle.analysis.tomoPlotAnalysis import octomo2rtp
import numpy as np
import pylab

def phasegate(ds, dataset=0, session=None):
    p, info = getMatrix(ds, dataset, session=session, dependent=[-4,-3,-2,-1])
    print info
    s = np.shape(p)
    p = np.reshape(p, (np.size(p)/24,6,4))
    p = p[:,:,2]/(p[:,:,0]+p[:,:,2])
    rtp = octomo2rtp(p[:,0],p[:,1],p[:,2],p[:,3],p[:,4],p[:,5])
    return [np.reshape(i,s[:-2]) for i in rtp]


def plotphasegate(r, theta, phase):
    s = np.shape(phase)
    a = np.average(phase,axis=1)
    pylab.clf();
    colors = ['b','g','r','c','y','k']
    for i in np.arange(s[1]):
        pylab.subplot(311)
        pylab.plot((phase[:,i]-phase[:,0]+np.pi)%(2*np.pi)-np.pi,
                   label='|%d>' % i,color=colors[i])
        pylab.subplot(312)
        pylab.plot(theta[:,i],label='|%d>' % i,color=colors[i])
        pylab.subplot(313)
        pylab.plot(r[:,i],label='|%d>' % i,color=colors[i])


