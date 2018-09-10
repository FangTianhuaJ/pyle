'''
Created back in the day ('09?)
author: Max Hofheinz and/or probably Matthew Neeley
Recovered by Erik Lucero May 2011

Needs some retouching to get aligned with current pyle release.
Looks useful for Resonator measurement analysis
'''
import numpy as np
from numpy import pi
from numpy.fft import rfft
from scipy.linalg import lstsq, eigh, eigvalsh
from scipy.misc import factorial
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

from pyle.plotting.dstools import getMatrix, getDataset
from myPlotUtils import mycolormap, saveImage, plot2d
from pyle.sim.JaynesCummings import Ujc
from tomoPlotAnalysis import xyz2rtp
from wignerAnalysis import QktoRho, displacement, pure

simMatrix = [[]]

PLOT = 1
PLOT_BAR = 2
PLOT_DIFF = 4
PLOT_COHERENT = 8

def coherentanalysis(ds, dataset=None, session=None, rabifreq=None,
                     doPlot=True, chopForVis=None, p0=0,p1=100, nfft=10000,
                     minContrast=2.0, fitDetuning=False, scale=None, nfit=None,
                     qT1=650.0, qT2=250.0, rT1=3500.0):
    """rabifreq, maxima, driveScale, amplitudeScale = \
       coherentanalysis(data_vault, dataset)

       The dataset has to contain a 2d coherent state scan with the
       x-axis a scan over qubit-resonator interaction time and y-axis
       the resonator drive amplitude.  detects the n-photon Rabi
       frequencies (maxima) and fits a vacuum Rabi frequency to it
       (rabifreq). It then fits the poisson distribution to the data
       to extract the drive amplitude to alpha scale factor
       (driveScale) and the photon number visibilites
       (amplitudeScale). The amplitude scale is now only used for
       consitany checks. All the visibility factors should be close to
       1.

       Keyword parameters:

       - rabifreq: Initial guess for the rabi
       frequency. Use that if coherentanalysis does not detect the
       vacuum Rabi frequency correctly.

       - doPlot: plot what is fitted

       - chopForVis: time cutoff in ns for fitting to the Poisson
         distribution

       - p0: Readout probability (in %) corresponding to the zero state
         (default 0)

       - p1: Same for 1 state (default 100)

       - nfft: Number of points in fft (default 10000)

       - minContrast: Minium contrast for Rabi frequency peak
         detection. Use lower value to find more Rabi frequencies
         (default 2)

       - fitDetuning=True to fit the Rabi frequencies not with
         sqrt(n)Omega but with sqrt(n Omega^2 + Delta^2). The return
         value rabifreq will contain these to values (default: False)

       - scale: initial value for driveScale. Use if the fit does not
         converge.

       - nfit->n in numberProb, qT1, qT2, rT1: see numberProb
       """

    data, info = getMatrix(ds, dataset=dataset, session=session,default=0.0)
    if doPlot:
        plt.ioff()
        plt.figure(1)
        plt.clf()
    rabifreq, maxima = findswapmax(data, info['Isteps'][0], doPlot=doPlot,
                                   rabifreq=rabifreq, minContrast=minContrast,
                                   fitDetuning=fitDetuning, nfft=nfft)
    maxima = np.hstack((0.0,maxima))
    if chopForVis is None:
        chopForVis = np.shape(data)[0]
    else:
        chopForVis = int(np.round(float(chopForVis)/info['Isteps'][0]))
    if nfit is None:
        nfit = np.alen(maxima)
    Pn = numberProb(data[:chopForVis,:], rabifreq, info['Isteps'][0],
                        p0, p1, n=nfit, rT1=rT1, qT1=qT1, qT2=qT2)
    if doPlot:
        plt.figure(2)
        plt.clf()
    amplitudescale = fitpoisson(np.linspace(info['Imin'][1],info['Imax'][1],
                                         np.shape(data)[1]),
                               Pn, doPlot=doPlot, scale=scale)
    if doPlot:
        plt.draw()
        plt.ion()
    return rabifreq, maxima, amplitudescale[-1], amplitudescale[0:-1]



def findswapmax(data, timestep,  rabifreq=None, doPlot=False, minContrast=2.0, fitDetuning=False, nfft=10000):
    """used by coherentanalysis"""
    l = np.shape(data)[0]
    data = 1.0*data
    t = np.arange(l)
    for i in np.arange(np.shape(data)[1]):
        data[:,i] -= np.polyval(np.polyfit(t, data[:,i], 1), t)
    data = rfft(data,axis=0,n=nfft)/np.size(data)
    data = np.abs(np.sum(data,axis=1))
    freqstep = 1.0/nfft/timestep

    if rabifreq is None:
        rabifreq = np.argmax(data)
    else:
        rabifreq = int(np.round(rabifreq/freqstep))
    boundaries = np.round(0.5*rabifreq*np.array([1,1+np.sqrt(2)])).astype(int)
    maxima = [0]
    minima = []
    i=0
    while boundaries[1] < np.alen(data):
        i+=1
        maxima += [boundaries[0]+np.argmax(data[boundaries[0]:boundaries[1]])]
        boundaries = np.round(maxima[-1]/np.sqrt(i)*0.5*(np.sqrt(np.array([i,i+1]))+np.sqrt(np.array([i+1,i+2])))).astype(int)
        minima += [maxima[-2]+np.argmin(data[maxima[-2]:maxima[-1]])]
        minval = minContrast*data[minima[-1]]
        if minval > data[maxima[-1]] or (i > 1 and minval > data[maxima[-2]]):
            break
    if i < 3:
        raise Exception('findswapmax: Not enough contrast')
    print 'found %d peaks' % (i-2)
    minima = np.array(minima)
    maxima = np.array(maxima[1:-2])
    errval = np.sqrt(0.5)
    lerr = np.zeros(len(maxima),dtype=int)
    rerr = np.zeros(len(maxima),dtype=int)
    for i,f in enumerate(maxima):
        ml = minima[i]
        mr = minima[i+1]
        lim = np.max((errval * data[f], data[ml]))
        lerr[i] = ml+np.argwhere((data[ml+1:f]-lim)*(data[ml:f-1]-lim) <= 0)[-1:,0]
        lim = np.max((errval * data[f], data[mr]))
        rerr[i] = f+1+np.argwhere((data[f+1:mr]-lim)*(data[f+2:mr+1]-lim) <= 0)[0,0]
    err = rerr-lerr
    n = 1+np.arange(len(maxima))
    def freqfunc(p,n):
        rabifreq = p[0]
        if np.alen(p) > 1:
            detuning = p[1]
        else:
            detuning = 0.0
        return np.sqrt(n*rabifreq**2+detuning**2)
    def errfunc(p):
        return (freqfunc(p,n)-maxima)/err
    rabifreq = float(rabifreq)
    if fitDetuning:
        rabifreq = [rabifreq, 0.1*rabifreq]
    else:
        rabifreq = [rabifreq]
    rabifreq, _ = leastsq(errfunc,rabifreq)
    if not fitDetuning:
        rabifreq = np.array([rabifreq])
    if doPlot:
        plt.plot(np.arange(len(data))*freqstep,data,'k-')
        plt.plot(maxima*freqstep, data[maxima],'b+')
        plt.plot((lerr+0.5)*freqstep, 0.5*(data[lerr]+data[lerr+1]), 'gx')
        plt.plot((rerr+0.5)*freqstep, 0.5*(data[rerr]+data[rerr+1]), 'gx')
        plt.plot(freqfunc(rabifreq,n)*freqstep,data[maxima],'r|')

    for i,d in enumerate(rabifreq):
        print ['Rabi frequency','Detuning'][i] + (': %g MHz' % (d*freqstep*1000))
    return rabifreq*freqstep, maxima*freqstep


def fitpoisson(drive, probabilities, doPlot=False, scale=None):
    """used by coherentanalysis"""
    def color(n):
        return ['k','b','g','r','m','c','y'][n%7]
    def marker(n):
        return ['x','+','o','s','D','h',][n/7]
    nmax = np.shape(probabilities)[0]
    def poisson(a, n):
        return np.exp(-a) * a**n / factorial(n)

    Mdrive,Mn = np.meshgrid(drive, np.arange(nmax))
    def fitfunc(p,drive,n):
        if scale is None:
            s = p[-1]
        else:
            s = np.abs(scale)
        result = poisson((s*drive)**2,n) * p[n] + (1-p[0])*(n==0)
        return result

    def errfunc(p):
        result = probabilities - fitfunc(p,Mdrive,Mn)
        return np.reshape(result,np.size(result))

    poissonfit = [0.1]*nmax
    if scale is None:
        poissonfit += [2.0]
    poissonfit = np.array(poissonfit)
    poissonfit, _ = leastsq(errfunc, poissonfit)
    if scale is None:
        print 'drive amplitude scale factor: %g'% poissonfit[-1]
    if doPlot:
        for i in np.arange(nmax):
            plt.plot(drive, fitfunc(poissonfit, drive, i), '-', color=color(i))
            plt.plot(drive, probabilities[i,:], marker(i), color=color(i))
        plt.xlabel('IQ mixer amplitude')
        plt.ylabel('Probability P_n')
        plt.xlim(drive[0], drive[-1])
        plt.axes([0.55, 0.50, 0.35, 0.4])
        for i in np.arange(nmax):
            plt.plot([i], [poissonfit[i]], marker(i), color=color(i), markeredgewidth=1.5)
    return poissonfit


def getVis(ds, dataset=None, session=None, quiet=False):
    """extract the probabilities corresponding to 0 and 1 state from a qubit tomography."""
    data = getDataset(ds, dataset=dataset, session=session)
    data = np.average(data, axis=0)
    vis = data[4] - data[1]
    if not quiet:
        names = [' X', ' Y', '-X', '-Y']
        indices = [2,3,5,6]
        print 'visibility: %.1f %%' % (vis)
        print 'pulse height errors: '
        for i in np.arange(4):
            print ' '+ names[i] + ': %4.1f %%' % (((data[indices[i]]-data[1])/vis*2-1)*100)
    return np.average(data[1]), np.average(data[4])

def getBloch(ds, dataset, vis=None,session=None):
    """extract the Block vector from a qubit tomography."""

    if vis is None:
        p0 = 0.0
        p1 = 100.0
    else:
        p0, p1 = getVis(ds, vis, session=session, quiet=True)
    data = getDataset(ds, dataset=dataset, session=session)
    data = np.average(data, axis=0)
    data = (data-p0)/(p1-p0)
    data = data[1:4]-data[4:7]
    r, theta, phi = xyz2rtp(data[1],data[2],-data[0])
    print 'r    : %.3f' % r
    print 'theta: %.3f (%5.1f deg)' % (theta, theta*180/pi)
    print 'phi  : %.3f (%5.1f deg)' % (phi, phi*180/pi)


def plotPn(ds, coherentVis, coherent, rabiVis, rabis=None, session=None, n=5,nCalc=None,minContrast=2.0,chop=None,qT1=650.0, qT2=250.0, rT1=3500.0):
    """Plot the photon number probability distribution"""
    p0, p1 = getVis(ds,coherentVis,session)
    if isinstance(rabis, np.ndarray):
        data = rabis
    else:
        data = getDataset(ds, dataset=rabis, session=session)

    timestep = data[1,0]-data[0,0]


    rabifreq, maxima, amplscale, visibilities = \
        coherentanalysis(ds, dataset=coherent, session=session,
                         chopForVis=data[-1,0], minContrast=minContrast,
                         doPlot=False, p0=p0, p1=p1)
    if n > len(maxima):
        print 'Not enough Rabi frequencies found, using fitted vacuum Rabi frequency'
        maxima = rabifreq[0]

    data = data[:,1]
    if not chop is None:
        data = data[:int(np.round(chop/timestep))]

    r0, r1 = getVis(ds,rabiVis,session)
    Pk = numberProb(data, rabifreq, timestep, p0, p1, r0,r1, n=n, nCalc = nCalc,
                        qT1=qT1, qT2=qT2, rT1=rT1, normalize=True)
    plt.bar(np.arange(n)-0.4, Pk)
    plt.xlim(-0.5,n-0.5)
    plt.ylim(-0.1,1)
    plt.xlabel('photon number')
    plt.ylabel('probability')




def numberProb(data, rabifreq, timestep, vis0, vis1, p0=None, p1=None,
               n=10, nCalc=None, qT1=650.0, qT2=250.0, rT1=3500.0,
               normalize=False):
    """Calculate the resonator photon number probability distribution
    corresponding to the scans over interaction time given by
    data. Data can have any dimensionality. The first dimension is
    supposed to be the scan over interaction time. timestep is in
    ns. vis0 and vis1 are the readout probabilities corresponding to
    qubit 0 and 1 state. p0 and p1 are the measured 0 and 1 state
    probabilites after the resonator state preparation. n is the
    number of probabilities to be returned, nCalc the number to be
    calculated (defaults to n+1, nCalc should be higher than n because
    the highest photon number probability calculated is not very
    accurate), qT1, qT2, rT1 are the qubit and resonator relaxation
    parameters qT2 should be about twice the measured qubit T2 (pure
    dephasing) because the qubit is less sensitive to flux noise when
    on resonance with the resonator. Normalize=True to normalize the
    probabilies to 1 (default: False).
    """
    if nCalc is None:
        #The highest photon number calculated is not reliable
        #because it absorbs all higher photon numbers,
        #so we must caluclate at least 1 photon number more than analyzed
        nCalc = n+1
    s = np.shape(data)
    if nCalc < n:
        raise Exception('numberProb: nCalc must be at least n and should be at least n+1')
    # scale out measurement visibility
    data = (np.reshape(data, (s[0], np.size(data)/s[0]))-vis0)/(vis1-vis0) - 0.5
    if p0 == None:
        p0 = 1.0
    else:
        print 'w   pi-pulse: %g' % ((p1-vis0)/(vis1-vis0))
        print 'w/o pi-pulse: %g' % ((p0-vis1)/(vis0-vis1))
        p0 = 0.5*((p1-vis0)/(vis1-vis0) + (p0-vis1)/(vis0-vis1))
        print 'average     : %g' % p0
    p1 = 1.0 - p0
    time = timestep*np.arange(s[0])
    rabifreq = rabifreq[0]
    matrix = np.resize(-0.5,(s[0],nCalc))
    for i in np.arange(1, nCalc):
        rho = np.zeros((2, (i+1), 2, (i+1)))
        rho[0,i,0,i] = 1.0
        rho = np.reshape(rho, (2*i+2, 2*i+2))
        simul = Ujc(time, fHO=6.0, fTLS=6.0, fDrive=6.0, coupling=rabifreq,
                    initial=rho, T1HO=rT1, T1TLS=qT1, T2TLS=qT2)
        a = np.arange(2*(i+1))
        matrix[:,i] += np.sum(np.reshape(simul[:,a,a], (s[0], 2, i+1))[:,1,:], axis=1)
    probas = lstsq(matrix, data)[0][:n]
    # correct for decohered qubit, suppose qubit density matrix is diagonal
    # and qubit and resonator are not entangled.
    #correction = 0.0
    probas /= p0
    #for i in np.arange(1,n):
    #    correction = (correction + probas[i-1,:]) * (p1/p0)
    #    probas[i,:] += correction

    norm = np.sum(probas, axis=0)[None,:]
    print 'Norm avg: %g' % np.average(norm)
    print 'Norm min: %g' % np.min(norm)
    print 'Norm max: %g' % np.max(norm)
    if normalize:
        probas /= np.sum(probas, axis=0)[None,:]
    probas = np.reshape(probas, [n]+list(s[1:]))
    simMatrix[0] = matrix
    return probas


def wignerimg(ds, filename, coherentVis, coherent, wignerVis, wigner,
              session=None, n=10, nCalc=None, qT1=650,qT2=300,rT1=3500,
              cmap=mycolormap('wrkbw'), minContrast=2.0,
              scale=None, initialAngle=0.0,
              vmin=-2.0/pi, vmax=2.0/pi, normalize=True, chop=None):
    """
    Save a bitmap of the Wigner function. Works like wignerplot but
    does not care about scaling in phase space.
    """

    p0,p1 = getVis(ds, coherentVis, session=session, quiet=True)
    if isinstance(wigner,tuple):
        data, info = wigner
    else:
        data, info = getMatrix(ds, dataset=wigner, session=session,default=0.0)

    if (chop is not None) and (chop < info['Imax'][0]):
        chop = int(chop/info['Isteps'][0])
        data = data[:chop,:,:]
        info['Imax'][0] = chop*info['Isteps'][0]


    rabifreq, maxima, amplscale, visibilities = \
        coherentanalysis(ds,dataset=coherent, session=session,
                         minContrast=minContrast, chopForVis=info['Imax'][0],
                         p0=p0, p1=p1, scale=scale, doPlot=False)

    r0,r1 = getVis(ds, wignerVis, session=session)
    data = numberProb(data, rabifreq, info['Isteps'][0], p0, p1,
                      r0, r1, n=n, nCalc=nCalc, qT1=qT1, qT2=qT2, rT1=rT1,
                      normalize=normalize)
    data = np.transpose(data,[1,2,0])
    data = np.sum(data*((-1)**np.arange(n))[None,None,:], axis=2)*2.0/pi
    saveImage(data, filename, cmap=cmap, vmin=vmin, vmax=vmax)



def wignerplot(ds, coherentVis, coherent, wignerVis, wigner, session=None,
               psi=[1.0], n=10, nCalc=None, qT1=650, qT2=300, rT1=3500,
               cmap=mycolormap('wrkbw'), minContrast=2.0,
               scale=None, initialAngle=0.0, nfit=None,
               vmin=-2.0/pi, vmax=2.0/pi,
               normalize=True, normalizeRho=False, chop=None, doPlot=PLOT):
    """ plot the Wigner function and return the density matrix.
    coherentVis: Qubit tomography to determine bare qubit visibility
    coherent: Coherent state scan to determine the Rabi frequency (see
    coherentstate)

    wignerVis: Qubit tomography to determine qubit
    state after state preparation

    wigner: Scan over qubit-resonator interaction time and
    displacement psi: state the wigner function represents (used to
    scale and rotate the wigner function correctly)

    minContrast: see coherent

    n, nCalc, qT1, qT2, rT1: see numberProb

    normalize: normalize photon number distributions for each displacement
    (bad for large displacement and low n, default: False, see numberProb)

    cmap: colormap to use for wigner function
    vmin, vmax: limits of the color scale (default -2/pi, 2/pi)
    showBar: plot Colorbar

    scale: fix drive amplitude to alpha scale
    initialAngle: initial value for the rotation of the Wigner function
    (default 0)

    nfit: number of states used to fit the density matrix, see QktoRho

    normalizeRho: normalize the density matrix to have trace 1, see
    QktoRho

    chop: Chop the time traces at chop ns before analyzing them
    (default: None)

    showdiffs: if true plot differences between calculated and
    extracted photon number probabilites

    doPlot: Sets what will be plotted. The following elements can be
    via bitwise or:
       PLOT: plot the Wigner function
       PLOT_BAR: plot a ColorBar
       PLOT_DIFF: plot the difference between measured and calculated
       photon number probabilities
       PLOT_COHERENT: plot information for the coherent state analysis

    """

    psi = np.asarray(psi)
    psi = 1.0*psi/np.sum(np.abs(psi)**2)
    p0,p1 = getVis(ds, coherentVis, session=session, quiet=True)
    if nfit == None:
        nfit = len(psi+2)
    if isinstance(wigner, tuple):
        data, info = wigner
    else:
        data, info = getMatrix(ds, dataset=wigner, session=session,default=0.0)

    if (chop is not None) and (chop < info['Imax'][0]):
        chop = int(chop/info['Isteps'][0])
        data = data[:chop,:,:]
        info['Imax'][0] = chop*info['Isteps'][0]


    rabifreq, maxima, amplscale, visibilities = \
        coherentanalysis(ds, dataset=coherent, session=session,
                         minContrast=minContrast, chopForVis=info['Imax'][0],
                         p0=p0, p1=p1, doPlot=doPlot & PLOT_COHERENT,
                         scale=scale)

    r0, r1 = getVis(ds, wignerVis, session=session)
    data = numberProb(data, rabifreq, info['Isteps'][0], p0, p1,
                      r0, r1, n=n, nCalc=nCalc, qT1=qT1, qT2=qT2, rT1=rT1,
                      normalize=normalize)
    s = np.shape(data)
    drive = np.linspace(info['Imin'][1],info['Imax'][1],s[1])[:,np.newaxis] \
        + 1.0j*np.linspace(info['Imin'][2],info['Imax'][2],s[2])[np.newaxis,:]
    data = np.transpose(data,[1,2,0])
    if scale is None:
        def fitfunc(p):
            scale = p[0] + 1.0j*p[1]
            calculated = np.dot(displacement(-drive*scale, N=(n, len(psi))), psi)
            return np.reshape(data - np.abs(calculated)**2, np.size(data))
        scale, _ = leastsq(fitfunc, [amplscale*np.cos(initialAngle),
                                     amplscale*np.cos(initialAngle)])
        scale = scale[0] + 1.0j*scale[1]

    elif not isinstance(scale,complex):
        def fitfunc(p):
            calculated = np.dot(displacement(-drive*scale*np.exp(1.0j*p),
                                          N=(n, len(psi))), psi)
            return np.reshape(data - np.abs(calculated)**2, np.size(data))
        phase, _ = leastsq(fitfunc, initialAngle)
        scale = scale*np.exp(1.0j*phase)


    print 'scale: ', scale, ' abs: ', np.abs(scale)
    if doPlot and PLOT_DIFF is not None:
        plt.figure(10)
        h = int(np.sqrt(n+1))
        w = (n+h)/h
        q = np.abs(np.dot(displacement(-drive*scale, N=(2*n, len(psi))), psi))**2
        for i in np.arange(n):
            plt.subplot(h, w, i+1)
            plot2d(data[:,:,i]-q[:,:,i],
                   extent=(info['Imin'][1],info['Imax'][1],
                           info['Imin'][2],info['Imax'][2]),
                   vmin=-1,vmax=1,cmap=cmap,aspect='equal')
        plt.subplot(h,w,n+1)
        i = np.arange(n,2*n)
        diff = np.sum(q[:,:,i]*(-1)**i, axis=2)
        plot2d(diff,
                   extent=(info['Imin'][1],info['Imax'][1],
                           info['Imin'][2],info['Imax'][2]),
                   vmin=-1,vmax=1,cmap=cmap,aspect='equal')
        plt.figure(1)

    # return data
    rho = QktoRho(drive*scale, data, N=nfit)
    if normalizeRho:
        # up to March 9 2009, I just had here rho /= trace(rho)
        # this blows up the matrix if there are negative entries on the
        # diagonal and the Fidelity gets too optimistic.
        # One could either normalize the absolute trace (otherwise the
        # fidelity of a density matrix with itself could be > 1
        # Or one could set negative eigenvalues of the density matrix to 0,
        # this is what validDensityMatrix does (it also normalizes)
        rho = validDensityMatrix(rho)
    data = np.sum(data*((-1)**np.arange(n))[None,None,:], axis=2)*2.0/pi
    if (doPlot & PLOT) == 0:
        return rho, drive*scale, data

    drive = (np.linspace(info['Imin'][1]-0.5*info['Isteps'][1],
                         info['Imax'][1]+0.5*info['Isteps'][1], s[1]+1)[:,np.newaxis] +
        1.0j*np.linspace(info['Imin'][2]-0.5*info['Isteps'][2],
                         info['Imax'][2]+0.5*info['Isteps'][2], s[2]+1)[np.newaxis,:])
    drive *= scale

    plt.pcolor(np.real(drive), np.imag(drive), data,
           vmin=vmin, vmax=vmax, cmap=cmap)
    corners = np.array([drive[0,0],drive[-1,0],
                     drive[-1,-1],drive[0,-1],drive[0,0]])
    plt.plot(np.real(corners), np.imag(corners), ':w')
    plt.xlabel(r'Re($\alpha$)')
    plt.ylabel(r'Im($\alpha$)')

    #rho = WtoRho(alpha*scale, 1.0*data, N=n)
    if doPlot & PLOT_BAR:
        ax = plt.colorbar()
        ax.set_label(r'W($\alpha$)')

    return rho


def tracedist(ideal,rho):
    return 0.5*np.sum(np.abs(eigvalsh(ideal-rho)))


def validDensityMatrix(rho, power=None):
    """make rho hermitian, positive semidefinite and normalize trace to 1"""
    # make matrix hermitian
    rho = 0.5 * (rho + np.transpose(np.conjugate(rho)))
    p, psi = eigh(rho)
    # make matrix positive semidefinite
    p = p * (p>0)
    # normalize trace
    p /= np.sum(p)
    if power == 0.5:
        p = np.sqrt(p)
    elif power is not None:
        p = p ** power
    return np.dot(psi, p[:,None]*np.transpose(np.conjugate(psi)))




def fidelity(rho1, rho2):
    """
    calculates the fidelity sqrt(sqrt(rho1).rho2.sqrt(rho1))/
    rho1 and rho2 can be density matrices or state vectors.
    rho1 and rho2 are normalized beforehand (see validDensityMatrix),
    otherwise the fidelity could be > 1.
    """
    if len(np.shape(rho1)) == 1:
        rho1 = pure(rho1)
    if len(np.shape(rho2)) == 1:
        rho2 = pure(rho2)
    rho1 = validDensityMatrix(rho1, power=0.5)
    rho2 = validDensityMatrix(rho2)
    w = eigvalsh(np.dot(np.dot(rho1, rho2), rho1))
    return np.sum(np.sqrt(w*(w>0)))



