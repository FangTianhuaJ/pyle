import numpy as np
from numpy import cos, sin, exp, pi

import re

from pyle import math
from pyle import envelopes as env
from pyle.dataking import envelopehelpers as eh
from pyle.dataking import util
from pyle.dataking import measurement
from pyle.dataking import sweeps
from pyle.dataking import multiqubitPQ as mq
from pyle.util import sweeptools as st
from pyle.fitting import fitting

from pyle.plotting import dstools as ds
from pyle import math as pylemath
import pyle.tomo as tomo
from pyle.analysis import stateTomography

from pyle.dataking import hadamard

from labrad.units import Unit,Value
ns, us, GHz, MHz = (Unit(un) for un in ['ns', 'us', 'GHz', 'MHz'])

#BRINGUP PROCEDURE
# The pulse bringup in tuneUp.py is essentially what you need to do to bring up the qubit.
# However, careful bringup of the single qubit rotations requires a bit more careful work.
# Here we outline the steps and their purpose/
#
# . Choose a pi length which is reasonable for your anharmonicity. Then tune up the pulses
# using tuneUp.py
# . Check for |2> leakage with ramseyFilter
# . Check frequency tuning by doing hadamard.pulseTrajectory with a short and then long delay
# between the initial and final pulses. If the phase delay accumulates, adjust the frequency
# appropriately.
# . Now we want to tune up the pi/2 pulses. This is done with hadamard.pingPong. Do pingPong
# and analyze the phase error per gate.
# .
#

I = 0.0+1.0j

SIGMAS = [tomo.sigmaX, tomo.sigmaY, tomo.sigmaZ]

def pingPong(s, measure, theta=st.r[-0.25:1.75:0.01], alpha = None, dfs = None, numPingPongs=3,
             stats=900,
             name='PingPong MQ', save=True, collect=False, noisy=True, analyze=False):
    """Ping pong traces for a range of detuning frequencies"""
    raise Exception('This does not work')
    sample, qubits = util.loadQubits(s)
    q = qubits[measure]

    axes = []

    if alpha is None:
        alpha = 0.5
    else:
        axes.append((alpha,'alpha'))
    if 1:
        pass

    name = '%s useHD=%d nPulses=%d' % (name, useHD, numPingPongs)

    axes = [(theta, '2nd pulse phase (divided by pi)'),(alpha, 'alpha')]
    deps = [('Probability', 'Frequency = %f GHz' %f['GHz'], 'GHz') for i in range(numPingPongs+1)]
    kw = {'stats': stats, 'useHD': useHD, 'numPingPongs': numPingPongs}
    dataset = sweeps.prepDataset(sample, name, axes, dependents=deps, measure=measure, kw=kw)

    def pfunc(t0, phase=0, alpha=0.5):
        return eh.piHalfPulseHD(q, t0, phase=phase, alpha=alpha)

    def func(server, theta, alpha):
        dt = q['piLen']
        reqs = []
        for freq in freqScan:
            q['f10'] = freq
            q.xy = eh.mix(q,
                pfunc(-dt) +
                sum(pfunc(2*k*dt, alpha=alpha) - pfunc((2*k+1)*dt, alpha=alpha) for k in range(numPingPongs)) +
                pfunc(2*i*dt, phase=theta*np.pi, alpha = alpha)
            )
            tm = 2*i*dt + dt/2.0
            q.z = eh.measurePulse(q, tm)
            q['readout'] = True
            reqs.append(runQubits(server, qubits, probs=[1], stats=stats))
        probs = yield FutureList(reqs)
        returnValue([p[0] for p in probs])
    return sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)

def tunePiPulse(s, measure, piAmp, detuning, stats=1500):
    pass
    #Scan over frequency detuning and pi pulse amplitude shifting
    #until you get really good pi/2 pulse angle fidelity.

def characterizePulses(s, measure):
    """Evaluate quality of single qubit control pulses"""
    #Get |2> occupation
    mq.ramseyFilter(s, delay=st.r[5:100:1,ns], theta=0, measure=measure, stats=3000L,
                    name='Ramsey Error Filter', save=True, collect=False, noisy=True)
    updateRegistryDict(s, measure, '_characterizationData', 'ramseyFilter', str(lastDatasetNum(s)))
    #Characterize pulse quality
    # Measure phase error
    hadamard.pingPong(s, theta=st.r[-0.25:1.75:0.01], measure=measure, numPingPongs=3, stats=3000, useHD=True,
             name='PingPong')
    updateRegistryDict(s, measure, '_characterizationData', 'pingPong', lastDatasetNum(s))
    #Measure amplitude error of sub-pi pulses
    fraction = np.linspace(0,1,17)
    pulseTrajectory(s, measure=measure, fraction=fraction, stats=6000)
    updateRegistryDict(s, measure, '_characterizationData', 'pulseTrajectoryX', lastDatasetNum(s))
    pulseTrajectory(s, measure=measure, fraction=fraction, stats=6000, phase=np.pi/2)
    updateRegistryDict(s, measure, '_characterizationData', 'pulsetrajectoryY', lastDatasetNum(s))
    zPulseTrajectory(s, measure)
    updateRegistryDict(s, measure, '_characterizationData', 'pulseTrajectoryZ', lastDatasetNum(s))
    #Z Pulses
    mq.ramseyZPulse(s, measure=MEAS, numPulses=11, stats = 12000)

def updateRegistryDict(s, configNumber, dictName, key, value):
    qubitKey = s['config'][configNumber]
    d = dict(s[qubitKey][dictName])
    d[key] = value
    s[qubitKey][dictName] = tuple(d.items())

def lastDatasetNum(s):
    PATH = s._dir
    dv = s._cxn.data_vault
    dv.cd(PATH)
    datasetName = dv.dir()[1][-1]
    m = re.match(r'\d*',datasetName)
    num = int(datasetName[m.start():m.end()])
    return num

def loadLastDataset(s):
    PATH = s._dir
    cxn = s._cxn
    dv = cxn.data_vault
    dataset = ds.getOneDeviceDataset(dv,-1,session=PATH, deviceName=None,
                                     averaged=False, correctVisibility=False)
    return dataset

def pulseTrajectory(s, measure, fraction, detuning, phase=0.0, alpha=0.5, tomoPhase=None,
                    stats=1500L, tBuf=20*ns, tBufMeasure = 5.0*ns,
                    name='Pulse Trajectory', save=True, collect=False, noisy=True):
    sample, qubits = util.loadQubits(s)
    q = qubits[measure]
    N = len(s['config'])
    startFreq = q['f10']
    if tomoPhase is not None:
        q['tomoPhase']=tomoPhase
    measureFunc = measurement.Octomo(N, measure, tBuf=tBufMeasure)

    axes = [(fraction, 'fraction of Pi-pulse'), (detuning, 'detuning'), (alpha, 'alpha')]
    kw = {'stats': stats, 'phase':phase, 'tBuf':tBuf}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measureFunc, kw=kw)

    def func(server, fraction, detuning, alpha):
        t = 0.0*ns
        q['f10'] = startFreq + detuning
        q['xy'] = eh.mix(q, eh.rotPulseHD(q, 0, angle=fraction*np.pi, phase=phase, alpha=alpha))
        t += q['piLen']/2.0
        return measureFunc(server, qubits, t + tBuf, stats=stats)
    return sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy, pipesize=1)

def zPulseTrajectory(s, measure, fraction=np.linspace(0, 1.75, 8),
                     tBuf = 0.0*ns, useTomo=True, stats=6000,
                     name='Z Pulse Trajectory', collect=True, save=True, noisy=True):
    sample, qubits = util.loadQubits(s)
    q = qubits[measure]
    q['readout'] = True
    N = len(s['config'])
    if useTomo:
        measureFunc = measurement.Octomo(N, measure, tBuf=tBuf,)
    else:
        measureFunc = measurement.Simult(N, measure, tBuf=tBuf,)

    axes = [(fraction, 'Z-Pulse Fraction of pi')]
    kw = {'stats': stats, 'tBuf':tBuf}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measureFunc, kw=kw)

    def func(server, fraction):
        t = 0.0*ns
        q['xy'] = env.NOTHING
        q['z'] = env.NOTHING
        q['xy'] += eh.piHalfPulse(q, t)
        t += q['piLen']
        q['z'] += eh.rotPulseZ(q, t, angle = fraction*np.pi)
        t += q['piLen']/2
        q['xy'] = eh.mix(q, q['xy'])
        return measureFunc(server, qubits, t+tBuf, stats=stats)
    return sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy, pipesize=1)

#####################################
## STATE REPRESENTATION CONVERSION ##
#####################################

def qubitAngles2Matrix(theta,phi):
    return np.array([[cos(theta/2.0)**2,exp(-I*phi)*cos(theta/2.0)*sin(theta/2.0)],
                     [exp(I*phi)*cos(theta/2.0)*sin(theta/2.0), sin(theta/2.0)**2]])

def qubitAngles2Vector(theta,phi):
    return np.array([sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)])

def qubitMatrix2Vector(rho):
    return np.array([np.trace(np.dot(rho, sigma)) for sigma in SIGMAS])

def qubitMatrix2Angles(rho):
    vec = np.real(qubitMatrix2Vector(rho))
    angles = qubitVector2Angles(vec)
    return angles

def qubitVector2Matrix(blochVector):
    return 0.5*(np.eye(2)+sum([v*sig for v,sig in zip(blochVector,SIGMAS)]))

def qubitVector2Angles(blochVector):
    """Convert a bloch vector into spherical angles theta and phi"""
    x,y,z = np.real(blochVector)
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y,x)
    return theta,phi

def angleBetweenVectors(v, u):
    """The angle between two vectors v and u"""
    return np.arccos(np.dot(u,v)/np.sqrt(np.dot(v,v)*np.dot(u,u)))


#######################
## DATASET ANALYZERS ##
#######################

def analyzePulseTrajectory(dataset, phase):
    fraction = dataset.data[:,0]
    probs = dataset.data[:,1:]
    F = stateTomography.fMatrix(dataset)
    #Get maximum likelihood fit density matrices
    Us, U = tomo._qst_transforms['octomo']
    pxmsArray = probs.reshape((-1,6,2))
    rhos = np.array([tomo.qst_mle(pxms, Us, F) for pxms in pxmsArray])
    def rotX(theta):
        return np.array([[1,  0,             0],
                         [0,  np.cos(theta), -np.sin(theta)],
                         [0,  np.sin(theta),  np.cos(theta)]])
    def rotY(theta):
        return np.array([[np.cos(theta), 0, np.sin(theta)],
                         [0,             1, 0],
                         [-np.sin(theta),0, np.cos(theta)]])
    def rotZ(phi):
        return np.array([[np.cos(phi), -np.sin(phi), 0],
                         [np.sin(phi), np.cos(phi), 0],
                         [0, 0, 1]])
    if phase == 'x':
        def rotationFunc(theta):
            return np.dot(rotZ(np.pi/2), rotX(-1*((np.pi*fraction) - (np.pi/2))))
    elif phase == 'y':
        def rotationFunc(theta):
            return rotY(-1*((np.pi*fraction) - (np.pi/2)))
    elif phase == 'z':
        def rotationFunc(phi):
            return rotZ(np.pi/2 - phi)

    errors = np.array([])
    #Nomatter what the rotation was in the experiment, we will move the
    #measured Bloch vector so that it _should_ be at theta=pi/2,
    #phi=0
    anglesTarget = np.array([np.pi/2, 0])
    for fraction, rho in zip(fraction, rhos):
        #Choose rotation matrix
        rotation = rotationFunc(fraction*np.pi)
        vector = np.dot(rotation, np.real(qubitMatrix2Vector(rho)))
        length = np.linalg.norm(vector)
        angles = qubitVector2Angles(vector)
        anglesError = (angles-anglesTarget)/(np.pi*2)
        errors = np.hstack((errors,anglesError,1-length))
    return np.reshape(errors, (-1,3))

def analyzePingPong(dataset):
    """
    Extracts pi/2 pulse phi error from a pingPong dataset

    INPUT
    A dataset object from a pingPong scan

    OUTPUT
    2D array. First column is number of pingPongs, second column is
    the phi error in cycles.

    NOTES

    Each angle error is in CYCLES, not radians.
    """
    numPingPongs = dataset['parameters']['numPingPongs']
    #The pingPong scan uses fractions of pi as the indep. axis
    #We want cycles so we divide by 2, ie. one 'pi' is half a cycle.
    phase = dataset.data[:,0]/2.0
    angleErrors = np.array([])
    fitResults = []
    amp, freq, angleError, offset = 0.4, 1.0, 0.0, 0.5
    #There are numPingPongs+1 traces because there is one trace with no pingPongs
    for col in range(numPingPongs+1):
        trace = dataset.data[:,col+1]
        fits,cov,fitFunc = fitting.fitCurve('cosine', phase, trace, (amp, freq, angleError, offset))
        fitResults.append(fits)
        #Update guesses for fits to subsequent trace
        amp,freq,angleError,offset = fits
        angleErrors = np.hstack((angleErrors,angleError))
    return fitResults, fitFunc

def analyzeRandomBenchmarking(dataset):
    """Extract error per gate from randomized benchmarking"""
    data = dataset.data[:,1:]
    averaged = np.mean(data, 1)
    fits, cov, fitFunc = fitting.fitCurve('exponential',
                                          dataset.data[:,0], averaged,
                                          [0.1, 20, 0.5])
    decayRate = fits[1]
    return averaged, decayRate, fits, fitFunc


######################################################

import matplotlib.pyplot as plt

THETA_MARKER = '^'
PHI_MARKER = 's'
THETA_MARKERSIZE = 20
PHI_MARKERSIZE = 20

RB_MARKER = '.'
RB_COLOR = 'b'
RB_MARKERSIZE = 30

FILTER_MARKER = '.'
FILTER_COLOR = 'b'
FILTER_MARKERSIZE = 30

X_COLOR = 'b'
Y_COLOR = 'r'
Z_COLOR = 'g'

ROOT_PATH = ['','Daniel','Benchmarking','w110427A','r4c9']

#                                    path                                      T1, T2, X,  Y,  Z, errorFilt, pingPong
DATASETS = {
            'qubit1':(['','Daniel','Benchmarking','w110427A','r4c9'], '120507', 31, 32, 67, 68, ),
            'qubit2':(['','Daniel','Benchmarking','w110427A','r4c9'], '120507', 56, 57, 91, 92, ),
            'qubit3':(['','Daniel','Benchmarking','w110427A','r4c9'], '120508', 16, 17, 24, 31, 35, 38,       39)
            }

QUBIT1 = {
          'xpulses': {},
          'ypulses': {},
          'zpulses': {},
          'pingPongs': {},
          'errorFilters': {},
          'randomBenches': {}
          }

QUBIT2 = {'name': 'qubit2',
          'T1': [('120507',56)],
          'T2': [('120507',57)],
          'errorFilters': [('120509',17)],
          'pingPongs': [('120509',19)],
          'xpulses': [('120507',91),('120509',11)],
          'ypulses': [('120507',92),('120509',12)],
          'zpulses': [('120509',15)],
          'randomBenches': [('120509',28), ('120509',21)]
          }

QUBIT3 = {'name': 'qubit3',
          'T1': [('120508', 16)],
          'T2': [('120508', 17)],
          'errorFilters': [('120508',38)],
          'pingPongs': [('120508',39)],
          'xpulses': [('120508',24),('120508',25),('120508',26)],
          'ypulses': [('120508',31)],
          'zpulses': [('120508',35),('120508',36),('120508',37)],
          'randomBenches': [('120509',56),('120509',57)]
          }


def addRamseyFilter(dv, ax, qubit, which):
    path, num = qubit['errorFilters'][which]
    path = ROOT_PATH + [path]
    dataset = ds.getOneDeviceDataset(dv, num, path)
    ax.plot(dataset.data[:,0], dataset.data[:,1], marker=FILTER_MARKER, color=FILTER_COLOR, markersize=FILTER_MARKERSIZE, linewidth=3)
    plt.rcParams['font.size']=25
    ax.set_title(str(dataset.path)+' '+qubit['name']+' '+'Ramsey Error Filter')
    return dataset


def addRandomBench(dv, ax, qubit, which):
    path, num = qubit['randomBenches'][which]
    path = ROOT_PATH + [path]
    dataset = ds.getOneDeviceDataset(dv, num, path)
    averaged, decayRate, fits, fitFunc = analyzeRandomBenchmarking(dataset)
    ax.plot(dataset.data[:,0], averaged, marker=RB_MARKER, color=RB_COLOR, markersize=RB_MARKERSIZE, linewidth=0)
    ax.plot(dataset.data[:,0], fitFunc(dataset.data[:,0], *fits), 'r', linewidth=4)
    plt.rcParams['font.size']=25
    ax.set_title(str(dataset.path)+' '+qubit['name']+' '+'Randomized Benchmarking')
    dataset.decayRate = decayRate
    return dataset


def addQubitXYPulses(dv, ax, qubit, whichX, whichY):
    pathX, numX = qubit['xpulses'][whichX]
    pathY, numY = qubit['ypulses'][whichY]
    pathX = ROOT_PATH + [pathX]
    pathY = ROOT_PATH + [pathY]
    datasetX = ds.getOneDeviceDataset(dv, numX, pathX)
    datasetX.resultX = analyzePulseTrajectory(datasetX, 'x')
    datasetY = ds.getOneDeviceDataset(dv, numY, pathY)
    datasetY.resultY = analyzePulseTrajectory(datasetY, 'y')
    #X rotation
    ax.plot(datasetX.data[:,0], datasetX.resultX[:,0]*360, THETA_MARKER,
            color=X_COLOR, markersize=THETA_MARKERSIZE,
            label=r'X - $\theta$')
    ax.plot(datasetX.data[:,0], datasetX.resultX[:,1]*360, PHI_MARKER,
            color=X_COLOR, markersize=PHI_MARKERSIZE,
            label=r'X - $\phi$')
    #Y rotation
    ax.plot(datasetY.data[:,0], datasetY.resultY[:,0]*360, THETA_MARKER,
            color=Y_COLOR, markersize=THETA_MARKERSIZE,
            label=r'Y - $\theta$')
    ax.plot(datasetY.data[:,0], datasetY.resultY[:,1]*360, PHI_MARKER,
            color=Y_COLOR, markersize=PHI_MARKERSIZE,
            label=r'Y - $\phi$')
    plt.rcParams['font.size']=25
    ax.set_title(str(pathX)+' '+qubit['name']+' '+'XY pulse trajectory')
    return datasetX, datasetY

def addQubitZPulses(dv, ax, qubit, which, showTheta=False):
    path, num = qubit['zpulses'][which]
    path = ROOT_PATH + [path]
    dataset = ds.getOneDeviceDataset(dv, num, path)
    result = analyzePulseTrajectory(dataset, 'z')
    dataset.result = result
    #Z rotation
    if showTheta:
        ax.plot(dataset.data[:,0]/2, result[:,0]*360, THETA_MARKER,
                color=Z_COLOR, markersize=THETA_MARKERSIZE,
                label=r'$\theta$')
    ax.plot(dataset.data[:,0]/2, result[:,1]*360, PHI_MARKER,
            color=Z_COLOR, markersize=PHI_MARKERSIZE,
            label=r'$\phi$')
    plt.rcParams['font.size']=25
    ax.set_title(str(path)+' '+qubit['name']+' '+'Z pulse trajectory')
    return dataset

def addPingPong(dv, ax, qubit, which, showText=False):
    colors = ['k','b','r','g']
    path, num = qubit['pingPongs'][which]
    path = ROOT_PATH + [path]
    dataset = ds.getOneDeviceDataset(dv, num, path)
    fits, fitFunc = analyzePingPong(dataset)
    for i, fit in enumerate(fits):
        if showText:
            ax.text(0.75, 0.8-0.1*i, 'Error: %f per gate' %(360*fit[2]/(2*(i+1))))
        ax.plot(dataset.data[:,0], dataset.data[:,i+1], '.', color=colors[i], markersize=20, label='%d identities'%i)
        ax.plot(dataset.data[:,0], fitFunc(dataset.data[:,0]/2,*fit), color=colors[i], linewidth=2)
    ax.set_title(str(dataset.path)+' '+qubit['name']+' '+'Ping Pong')
    return dataset
