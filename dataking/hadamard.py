import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import labrad
from labrad.units import Unit
V, mV, us, ns, GHz, MHz = [Unit(s) for s in ('V', 'mV', 'us', 'ns', 'GHz', 'MHz')]

import pyle
from pyle.pipeline import returnValue, FutureList
from pyle.dataking import util
from pyle import envelopes as env
from pyle.dataking import envelopehelpers as eh
from pyle.dataking import measurement
from pyle.dataking import sweeps
from pyle.dataking.fpgaseq import runQubits
from pyle.util import sweeptools as st
from pyle.plotting import dstools


def pulseTrajectoryHD(s, fraction=st.r[0.0:1.0:0.01], phase=0.0, alpha=None,
                      measure=0, stats=1500L, useHD=True, useTomo=True, tBuf=0*ns,
                      name='Pulse Trajectory', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(s)
    q = qubits[measure]
    N = len(s['config'])

    if useTomo:
        #print 'measuring qubit %d out %d with Octomography ' %(measure,N)
        measureFunc = measurement.Octomo(N, measure, tBuf=tBuf,)
    else:
        #print 'measuring qubit %d out %d with Simultaneous measurement' %(measure,N)
        measureFunc = measurement.Simult(N, measure, tBuf=tBuf,)

    name = '%s useHD=%d useTomo=%d' % (name, useHD, useTomo)

    axes = [(fraction, 'fraction of Pi-pulse')]
    kw = {'stats': stats, 'useHD': useHD, 'phase':phase, 'tBuf':tBuf}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measureFunc, kw=kw)

    def func(server, fraction):
        if useHD:
            pulse = eh.mix(q, eh.rotPulseHD(q, 0, angle=fraction*np.pi, phase=phase, alpha=alpha))
        else:
            pulse = eh.mix(q, env.gaussian(0, q['piFWHM'], amp=fraction*q['piAmp']))
        q.xy = pulse
        return measureFunc(server, qubits, tBuf+q['piLen']/2, stats=stats)
    return sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy, pipesize=1)


def hadamardTrajectory(sample, fraction=st.r[0.0:1.0:0.01], measure=0, stats=1500L, useHD=True, useTomo=True, tBuf=0*ns,
                       name='Hadamard Trajectory', save=True, collect=False, noisy=True):
    #TODO Add flag to plot trajectory on bloch sphere
    #TODO add line in qstSweep to grab the most recent dataset from datavault see Ramsey for example
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    N= len(sample['config'])

    if useTomo:
        print 'measuring qubit %d out %d with Octomography ' %(measure,N)
        measureFunc = measurement.Octomo(N, measure, tBuf=tBuf)
    else:
        print 'measuring qubit %d out %d with Simultaneous measurement' %(measure,N)
        measureFunc = measurement.Simult(N, measure, tBuf=tBuf)

    name = '%s useHD=%d useTomo=%d' % (name, useHD, useTomo)

    axes = [(fraction, 'fraction of Pi-pulse')]
    kw = {'stats': stats, 'useHD': useHD}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measureFunc, kw=kw)

    def func(server, fraction):
        if useHD:
            q.xy = eh.mix(q, eh.rotPulseHD(q, 0, angle=fraction*np.pi/np.sqrt(2)))
        else:
            q.xy = eh.mix(q, env.gaussian(0, q['piFWHM'], amp=fraction*q['piAmp']/np.sqrt(2)))
        q.z = eh.rotPulseZ(q, 0, angle=fraction*np.pi/np.sqrt(2))
        return measureFunc(server, qubits, q['piLen']/2, stats=stats)
    return sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy, pipesize=1)



def hadamardInverseTrajectory(sample, fraction=st.r[0.0:2.0:0.01], measure=0, stats=1500L, useHD=True, useTomo=True, tBuf=0*ns,
                              initPulse=None, name='Hadamard Trajectory MQ', save=True, collect=False, noisy=True):
    """Performs two hadamards in sequence for a complete Identity operation.
    Pulse sequence is set to NOT use sideband mix."""

    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    N = len(sample['config'])

    if useTomo:
        print 'measuring qubit %d out %d with Octomography ' %(measure,N)
        measureFunc = measurement.Octomo(N, measure, tBuf=tBuf)
    else:
        print 'measuring qubit %d out %d with Simultaneous measurement' %(measure,N)
        measureFunc = measurement.Simult(N, measure, tBuf=tBuf)

    name = '%s useHD=%d useTomo=%d initPulse=%s' % (name, useHD, useTomo, initPulse)

    axes = [(fraction, 'fraction of HadamardGate')]
    kw = {'stats': stats, 'useHD': useHD, 'initPulse': str(initPulse)}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measureFunc, kw=kw)

    if useHD:
        def pfunc(t0, angle):
            return eh.rotPulseHD(q, t0, angle=angle)
    else:
        def pfunc(t0, angle):
            return env.gaussian(t0, q['piFWHM'], amp=angle/np.pi*q['piAmp'])

    def func(server, fraction):
        q.xy = env.NOTHING
        q.z = env.NOTHING
        if initPulse:
            axis, angle = initPulse
            q.xy += eh.rotPulseHD(q, -q['piLen'], angle=angle, phase=axis)
        if fraction >= 0:
            f0 = np.clip(fraction, 0, 1)
            q.xy += pfunc(0, angle=f0*np.pi/np.sqrt(2))
            q.z += eh.rotPulseZ(q, 0, angle=f0*np.pi/np.sqrt(2))
            tm = q['piLen']/2.0
        if fraction > 1:
            f1 = np.clip(fraction-1, 0, 1)
            q.xy += pfunc(q['piLen'], angle=f1*np.pi/np.sqrt(2))
            q.z += eh.rotPulseZ(q, q['piLen'], angle=f1*np.pi/np.sqrt(2))
            tm = q['piLen']*1.5
        q.xy = eh.mix(q, q.xy)
        return measureFunc(server, qubits, tm, stats=stats)
    return sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy, pipesize=1)

def pingPong(s, theta=st.r[-0.25:1.75:0.01], alpha = 0.5, measure=0, numPingPongs=1, stats=900, useHD=True,
             name='PingPong MQ', save=True, collect=False, noisy=True, analyze=False):
    sample, qubits = util.loadQubits(s)
    q = qubits[measure]

    name = '%s useHD=%d nPulses=%d' % (name, useHD, numPingPongs)

    axes = [(theta, '2nd pulse phase (divided by pi)'),(alpha, 'alpha')]
    deps = [('Probability', '%d identities' % i, '') for i in range(numPingPongs+1)]
    kw = {'stats': stats, 'useHD': useHD, 'numPingPongs': numPingPongs}
    dataset = sweeps.prepDataset(sample, name, axes, dependents=deps, measure=measure, kw=kw)

    if useHD:
        def pfunc(t0, phase=0, alpha=0.5):
            return eh.piHalfPulseHD(q, t0, phase=phase, alpha=alpha)
    else:
        def pfunc(t0, phase=0, alpha=0.0): #The alpha parameter is just there for call signature compatibility. I know this is stupid. Sorry. DTS
            return env.gaussian(t0, q['piFWHM'], amp=q['piAmp']/2.0, phase=phase)

    def func(server, theta, alpha):
        dt = q['piLen']
        reqs = []
        for i in range(numPingPongs+1):
            q.xy = eh.mix(q,
                pfunc(-dt) +
                sum(pfunc(2*k*dt, alpha=alpha) - pfunc((2*k+1)*dt, alpha=alpha) for k in range(i)) +
                pfunc(2*i*dt, phase=theta*np.pi, alpha = alpha)
            )
            tm = 2*i*dt + dt/2.0
            q.z = eh.measurePulse(q, tm)
            q['readout'] = True
            reqs.append(runQubits(server, qubits, probs=[1], stats=stats))
        probs = yield FutureList(reqs)
        returnValue([p[0] for p in probs])
    return sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)


def pingPongFrac(s,
                 phase=st.r[-0.125:0.875:0.02], alphas=st.r[0.4:0.6:0.005],
                 angle=np.pi/2.0,
                 measure=0, numPulses=1, stats=1200, useHD=True,
                 name='PingPongVsAlpha', save=True, collect=False, noisy=True,
                 message=False):
    sample, qubits = util.loadQubits(s)
    q = qubits[measure]

    name = '%s useHD=%d nPulses=%d' % (name, useHD, numPulses)

    axes = [(phase, '2nd pulse phase'),(alphas, 'drag alpha')]
    deps = [('Probability', '%d pseudo-identities' %numPulses, '')]
    kw = {'stats': stats, 'useHD': useHD, 'nPulses': numPulses}
    dataset = sweeps.prepDataset(sample, name, axes, dependents=deps, measure=measure, kw=kw)

    if useHD:
        def pfunc(t0, angle, phase=0, alpha=0.5):
            return eh.rotPulseHD(q, t0, angle, phase=phase, alpha=alpha)
    else:
        def pfunc(t0, phase=0):
            return env.gaussian(t0, q['piFWHM'], amp=q['piAmp']/2.0, phase=phase)

    def func(server, currPhase, currAlpha):
        dt = q['piLen']
        n = numPulses
        q.xy = eh.mix(q,
            pfunc(-dt, angle, alpha=currAlpha) +
            sum(pfunc(2*k*dt, angle, alpha=currAlpha) - pfunc((2*k+1)*dt, angle, alpha=currAlpha) for k in range(n)) +
            pfunc(2*n*dt, angle, alpha=currAlpha, phase=2*np.pi*currPhase)
        )
        tm = 2*n*dt + dt/2.0
        q.z = eh.measurePulse(q, tm)
        q['readout'] = True
        return runQubits(server, qubits, probs=[1], stats=stats)

    data = sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)
    if message:
        s._cxn.telecomm_server.send_sms('scan done','Scan %s has finished.' %name, s._dir[1])
    if collect:
        return data


def qstSweep(path, dataset):
    with labrad.connect() as cxn:
        dv = cxn.data_vault
        dataset = dstools.getOneDeviceDataset(dv, datasetNumber=dataset, session=path)
        probs = dataset.data[:,1:]
        #data = dstools.getDataset(dv, dataset, path)
        #probs = data[:,1:]
        #correct for measurement infidelities
        #f1 = dataset.parameters.q.measureF1
        #e0 = dataset.parameters.q.measureE0
        #probs = (probs - e0)/f1
        #Reshape
        # s01,s11 = dataset.parameters.q['calScurve1']
        probs = probs.reshape((-1,6,2))
        # for mat in probs:
            # mat[:,0] = (mat[:,0]-s01)/(s11-s01)
            # mat[:,1] = 1-mat[:,0]
        rhos = [pyle.tomo.qst(diags, 'octomo') for diags in probs]
        return rhos

def rho2bloch(rho):
    sigmas = [pyle.tomo.sigmaX, pyle.tomo.sigmaY, pyle.tomo.sigmaZ]
    return np.array([np.trace(np.dot(rho, sigma)) for sigma in sigmas])


def plotTrajectory(path, dataset, state, labels=True):
    rhos = qstSweep(path, dataset)
    blochs = np.array([rho2bloch(rho) for rho in rhos])

    ax = drawBlochSphere(True, labels)

    if state is None:
        state = range(4)
    if isinstance(state,
    int):
        state = [state]

    for s, c in zip(state, 'rgby'):
        ax.plot(blochs[:,0], blochs[:,1], blochs[:,2], 'o', color=c, markersize=8)


def drawBlochSphere(bloch = True, labels = True): # takes a sequence and projects the trajectory of the qubit onto the bloch sphere, works for T1/T2!
    # if fig == 0:
        # ax = Axes3D(figure())
    # else:
        # ax = Axes3D(figure(fig))
        # hold
    ax = Axes3D(plt.figure())
    #draw a bloch sphere
    if bloch:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z,  rstride=5, cstride=5, color = '0.8', alpha = 0.9)#, cmap = cm.PuBu) #, alpha = 1.0)
        ax.set_aspect('equal')

    # clean this up

    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')

    #Axis label
    if labels:
        ax.text(0, 0, 1, '|0>')
        ax.text(0, 0, -1, '|1>')
        ax.text(1, 0, 0, '|0> + |1>')
        ax.text(-1, 0, 0, '|0> - |1>')
        ax.text(0, 1, 0, '|0> + i|1>')
        ax.text(0, -1, 0, '|0> - i|1>')


    #Axis
    ax.plot([-1.2,1.2],[0,0],[0,0], color = '0.1', linewidth=2)
    ax.plot([0,0],[-1.2,1.2],[0,0], color = '0.1', linewidth=2)
    ax.plot([0,0],[0,0],[-1.2,1.2], color = '0.1', linewidth=2)
    ax.set_aspect('equal')

    return ax

def plotDensityArrowPlot(path, dataset):
    """Takes a single density matrix 'rho' and plots the arrow density matrix representation.
    Uses qstSweep to take the probabilities from an Octomo-scan and convert them to rhos """

    rhos = qstSweep(path, dataset)
    rho = rhos[-1]
    ax = createArrowPlot(rho)


def createArrowPlot(rho,legend=None,scale=1.0,color=None, width=0.05, ax=None):
    s=np.shape(rho)
    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.set_aspect(1.0)
    pos = ax.get_position()
    arrlabelsize = ax.xaxis.get_ticklabels()[0].get_fontsize()
    r = np.real(rho)
    i = np.imag(rho)
    x = np.arange(s[1])[None,:] + 0*r
    y = np.arange(s[0])[:,None] + 0*i
    plt.quiver(x,y,r,i,units='x',scale=1.0/scale, width=width, color=color)
    plt.xlabel('$n$')
    plt.ylabel('$m$')
    if legend is not None:
        if np.shape(legend)== (2,2):
            pass
        elif np.shape(legend) == (2,):
            legend = [[legend[0],legend[0]], [legend[1],legend[1]+1]]
        else:
            legend = [[s[0]+0.5]*2,[-1,0]]
        x = plt.quiver(legend[0], legend[1], [1,0],[0,1],
                         units='x',scale=1.0/scale, color=color,
                         width=width)
        x.set_clip_on(False)
        plt.text(legend[0][0]+scale*0.5,
                   legend[1][0]+width*(-1.5+3.0*(legend[1][0]>0))*scale, r'$1$',
                   ha='center',va=['top','bottom'][int(legend[1][0]>0)],
                   fontsize=arrlabelsize)
        plt.text(legend[0][1]+width*(-1.5+3.0*(legend[0][1]>0))*scale,
                   legend[1][1]+0.5*scale,r'$i$',
                   ha=['right','left'][int(legend[0][1]>0)],
                   va='center',fontsize=arrlabelsize)
        plt.text(legend[0][0]+0.5*scale,legend[1][1]+scale+0.1,r'$\rho_{mn}$',va='bottom',ha='center')
    plt.xlim(-0.9999,s[1]-0.0001)
    plt.ylim(-0.9999,s[0]-0.0001)

    return ax

