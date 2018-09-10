'''
Created on Sep 3, 2011

@author: Daniel Sank
'''
import numpy as np
from pyle.dataking import util
from pyle.util import sweeptools as st
from pyle.dataking import envelopehelpers as eh
from pyle.dataking.fpgaseq import runQubits
from pyle.dataking import utilMultilevels
import pyle.envelopes as env
from pyle.plotting import dstools
from pyle.fitting import fitting
from pyle.util import structures
from pyle.plotting import dstools as ds
from pyle.pipeline import returnValue
from pyle.dataking import adjust
#from pyle.dataking import multiqubit as mq
import sweeps

import labrad
from labrad.units import Unit
us,ns,MHz,GHz = (Unit(s) for s in ['us','ns','MHz','GHz'])

from scipy.special import erf,erfc
from scipy.optimize import leastsq

def resonatorSpectroscopy(Sample, measure, measureR, paramName, freqScan=None, pulseTime=None, uwaveAmp=None,
                          swapTime=300*ns, stats=600L,
                          name='Resonator spectroscopy', save=True, collect=False, noisy=True, update=True):
    """Resonant drive of a resonator with detection by resonator->qubit swap"""
    sample, qubits = util.loadDeviceType(Sample, 'phaseQubit')
    sample, resonators, Resonators = util.loadDeviceType(Sample, 'resonator', write_access=True)
    q = qubits[measure]
    r = resonators[measureR]
    R = Resonators[measureR]
    q['readout']=True
    if freqScan is None:
        f = st.nearest(r['freq'][GHz], 0.001)
        freqScan = st.r[f-0.02:f+0.02:0.0005,GHz]
    if uwaveAmp is None:
        uwaveAmp = r['spectroscopyAmp']
    if pulseTime is None:
        pulseTime = r['spectroscopyLen']

    axes = [(uwaveAmp, 'Microwave Amplitude'), (freqScan, 'Frequency')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    swapAmp = q['swapAmp'+paramName]

    def func(server, amp, f):
        #Determine resonator parameters
        sidebandFreq = f-r['fc']
        r['spectroscopyAmp'] = amp
        #Excite resonator
        r.xy = eh.spectroscopyPulse(r, 0, sidebandFreq)
        #Use qubit to grab resonator excitation
        q.z = env.rect(r['spectroscopyLen']+20*ns, swapTime, swapAmp)+eh.measurePulse(q, r['spectroscopyLen']+swapTime+30*ns)
        #eh.correctCrosstalkZ(qubits)
        return runQubits(server, qubits+resonators, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if update:
        adjust.adjust_frequency(R, data, paramName='freq')
    if collect:
        return data

def resonatorT1(Sample,measure,measureR,paramName,delay=None,stats=1200L, name='resonator T1',
                save=True, collect=True, noisy=True, update=True, plot=False):
    sample, qubits = util.loadQubits(Sample)
    sample, resonators, Resonators = util.loadDeviceType(Sample,'resonator',write_access=True)
    qubit = qubits[measure]
    Resonator = Resonators[measureR]
    if delay is None:
        delay = structures.ValueArray(np.hstack((np.linspace(-0.01,0,5),np.logspace(-2,0.845,50))),'us')
        #delay = st.r[0:1:0.01,us]
    axes = [(delay, 'Idle time')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, paramName+' '+name, axes, measure=measure, kw=kw)

    swapTime = qubit['swapTime'+paramName]
    swapAmp = qubit['swapAmp'+paramName]

    def func(server, delay):
        qubit.xy = eh.mix(qubit, eh.piPulseHD(qubit, 0))
        qubit.z = env.rect(qubit.piLen/2, swapTime, swapAmp)
        qubit.z += env.rect(qubit.piLen/2+swapTime+delay, swapTime, swapAmp)
        qubit.z += eh.measurePulse(qubit, qubit.piLen/2+swapTime+delay+swapTime)
        qubit['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if update or plot:
        with labrad.connect() as cxn:
            dv = cxn.data_vault
            dataset = dstools.getOneDeviceDataset(dv,-1,session=sample._dir,
                                                  deviceName='res',averaged=False)
        result = fitting.t1(dataset,timeRange=(20.0*ns,delay[-1]))
    if plot:
        fig = dstools.plotDataset1D(dataset.data,
                                    dataset.variables[0],dataset.variables[1],
                                    marker='.',markersize=15)
        ax = fig.get_axes()[0]
        ax.plot(dataset.data[:,0],
                result['fitFunc'](dataset.data[:,0],result['fit']),'r',
                linewidth=3)
        ax.grid()
        fig.show()
    if update:
        Resonator['calT1']=result['T1']

def testDelay(Sample, measure, measureR, resName, startTime=st.r[-80:80:2,ns], pulseLength=8*ns,
              amp=0.2, stats=1200L, save=True, collect=False, noisy=True, plot=True, update=True,
              name='Resonator test delay'):
    """Two widely spaced half cycle outo of phase pulses on the resonator with detection in between"""
    sample, qubits = util.loadDeviceType(Sample, 'phaseQubit')
    sample, resonators, Resonators = util.loadDeviceType(Sample, 'resonator', write_access=True)
    q = qubits[measure]
    r = resonators[measureR]
    R = Resonators[measureR]
    axes = [(startTime, 'uWave Start Time')]
    kw = {'stats':stats, 'uwaveAmplitude':amp}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server,start):
        r.xy=eh.mix(
                    r,env.gaussian(start-2.0*pulseLength, pulseLength, amp),freq=r'freq')+eh.mix(r,env.gaussian(start+2.0*pulseLength, pulseLength, amp=-amp), freq='freq')
        q.z = env.rect(-q[resName+'SwapTime']/2, q[resName+'SwapTime'], q[resName+'SwapAmp'])+eh.measurePulse(q,q[resName+'SwapTime']/2+100*ns)
        q['readout']=True
        return runQubits(server,qubits+resonators, stats, probs=[1])
    data = sweeps.grid(func,axes,dataset=save and dataset, noisy=noisy)
    topLen = 5*pulseLength
    transLen = pulseLength/2.0
    if plot or update:
        with labrad.connect() as cxn:
            dv = cxn.data_vault
            dataset = dstools.getOneDeviceDataset(dv,-1,session=sample._dir,
                                                  deviceName='r',averaged=False)
        result = fitting.squarePulse(dataset, topLen, transLen,
                                     timeRange=(-80*ns,80*ns))
        offset = result['horzOffset']
        print 'uwave lag: %f ns'%-offset['ns']
    if plot:
        fig = ds.plotDataset1D(dataset.data,dataset.variables[0],dataset.variables[1])
        ax = fig.get_axes()[0]
        ax.plot(dataset.data[:,0],result['fitFunc'](dataset.data[:,0],result['fit']),
                'g',linewidth=4)
        ax.grid()
        fig.show()
    if update:
        print 'uwave lag corrected by %f ns'%offset['ns']
        R['timingLagUwave']+=offset
    if collect:
        return data


def fockScan(Sample, measure, measureR, paramName, n=1, scanLen=st.r[0:100:1,ns], detuneAmp=0.0,
             excited=False, scanOS=0.0, stats=1500L, buffer=0*ns, piDelay=0.0*ns, measureState=1,
             name=None, save=True, collect=False, noisy=True, probe=True):
    sample, qubits = util.loadDeviceType(Sample, 'phaseQubit')
    #sample, resonators = util.loadDeviceType(Sample, 'resonator')
    q = qubits[measure]

    utilMultilevels.setMultiKeys(q,2)
    utilMultilevels.setMultiKey(q,paramName+'FockTime',1)
    utilMultilevels.setMultiKey(q,paramName+'FockOvershoot',1)

    fockTimes = q['multiLevels'][paramName+'FockTime']
    fockOvershoots = q['multiLevels'][paramName+'FockOvershoot']
    swapAmp = q[paramName+'SwapAmp']
    if name is None:
        name = '%s Fock state |%d> swap length' %(paramName,n)

    axes = [(detuneAmp,'Idle detune amp'),(scanLen, 'Swap length adjust'),(scanOS, 'Amplitude overshoot')]
    kw = {'stats': stats, 'measureState':measureState}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server,idleAmp,probeTime,overshoot):
        t=0*ns
        q.xy=env.NOTHING
        q.z=env.NOTHING
        #Put the resonator in the |n> state
        for fockState in range(n):
            q.xy += eh.boostState(q,t+q['piLen']/2,state=1)
            t += q['piLen']+buffer+piDelay
            q.z  += env.rect(t, fockTimes[fockState],swapAmp)#,overshoot=fockOvershoots[fockState])
            t += fockTimes[fockState]+buffer+4*ns
        #Probe the resonator with the qubit for variable time and with variable overshoot)
        if excited:
            q.xy += eh.mix(q, eh.piPulseHD(q, t+q.piLen/2))
            t += q.piLen+buffer
        if probe:
            q.z += env.rect(t, probeTime, swapAmp, overshoot=overshoot)
            t += probeTime+buffer+4*ns
        else:
            q.z += env.rect(t,probeTime,idleAmp)
            t += probeTime
        #Measure the qubit
        q.z += eh.measurePulse(q, t, state=measureState)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy, collect=collect)

def coherentScan(Sample, measure, measureR, deviceName, pulseAmp=None, pulseTime=None,
                 swapTime=st.r[0:100:1,ns], excited=False,
                 stats=1200, name=None, save=True, collect=True, noisy=True):
    sample,qubits = util.loadDeviceType(Sample, 'phaseQubit')
    sample,resonators = util.loadDeviceType(Sample, 'resonator')
    q,r = qubits[measure],resonators[measureR]
    q['readout']=True
    if pulseAmp is None:
        pulseAmp = r['coherentPulseAmp']
    if pulseTime is None:
        pulseTime = r['coherentPulseTime']
    if name is None:
        name = 'Coherent state swap, q=|%s>' %('e' if excited else 'g')
    axes = [(swapTime, 'Swap Time'),(pulseTime, 'Pulse Time'),(pulseAmp, 'Pulse Amplitude')]
    kw = {'stats':stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)
    def func(server,swapTime,pulseTime,pulseAmp):
        t=0*ns
        r.xy = eh.mix(r,env.flattop(0, pulseTime, r['piFWHM'], pulseAmp), freq=r['freq'])
        t += pulseTime+r['piFWHM']
        q.z = env.rect(t, swapTime, q[deviceName+'SwapAmp'])
        t += swapTime+10*ns
        q.z += eh.measurePulse(q, t, state=1)
        return runQubits(server, qubits+resonators, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if noisy:
        return data

#Yi's function. Need to massage for use in main code
def arbitraryWigner(Sample, measure, measureR, paramName, swapTimes=None,
                    offresTimes=None, pulseAngles=None, pulsePhases=None, alpha=[0.5], probePulseLength=st.r[0:300:2,ns],
                    stats=1200L, SZ=0.0, probePulseDelay=0*ns,
                    name='arbitrary Wigner', save=True, collect=False, noisy=True):
    """Make an arbitrary state in the resonator and measure the resonator Wigner function

    offresTimes - [num,...]: Times to wait with qubit off resonance from resonator before each swap event
    pulseAmplitudes - [num,...]: Pulse amplitudes to apply to the qubit for each swap even. Given in DAC units, as with a['piAmp'].
    """
    sample, qubits = util.loadDeviceType(Sample, 'phaseQubit')
    sample, resonators = util.loadDeviceType(Sample, 'resonator')
    q = qubits[measure]
    r = resonators[measureR]

    if swapTimes is None:
        swapTimes = [q[paramName+'SwapTime']]
    nPulses = len(swapTimes)
    delay = q.piLen/2
    rdelay = r.piLen/2
    qf = q['f10']   #Qubit frequency
    rf = r['freq']  #Resonator frequency
    sa = float(q[paramName+'SwapAmp'])
    os = float(q[paramName+'SwapOvershoot'])+np.zeros(nPulses)

    #sweepPara = [[d,sT] for d in np.array(alpha) for sT in probePulseLength]
    def sweep():
        for d in np.array(alpha):
            for ppl in probePulseLength:
                yield (d,ppl)

    kw = {'stats': stats}

    if offresTimes is None:
        offresTimes = np.resize(2*delay,nPulses)
    else:
        kw['times off resonance'] = offresTimes
    if pulseAngles is None:
        pulseAngles = np.resize(np.pi,nPulses)
    if pulsePhases is None:
        pulsePhases = np.resize(0.0, nPulses)
    if probePulseDelay is not None:
        kw['probe swap pulse delay'] = probePulseDelay
    else:
        probePulseDelay = 0*ns

    axes = [('rm displacement', 're'),('rm displacement', 'im'),('swap pulse length', 'ns')]
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)
    def func(server, args):
        displacementAmp = args[0]
        currLen = args[1]
        time = 0.0
        q.xy = env.NOTHING
        q.z=env.NOTHING
        #Put the resonator in an arbitrary state with a series of qubit pulses and swaps
        for i in np.arange(len(swapTimes)):
            #q.xy += eh.mix(q, eh.rotPulseHD(q, time-delay, angle=pulseAngles[i], phase=pulsePhases[i]+2.0j*np.pi*(qf-rf)*time, alpha=0.5, state=1))
            q.xy += eh.boostState(q, time, 1)
            time += q['piLen']
            q.z += env.rect(time, swapTimes[i], sa,overshoot = os[i])
            time += swapTimes[i]
            time += offresTimes[i]
        resstart = time + probePulseDelay + 2*rdelay
        #Try to dump the qubit population in case there is any
        ##REMOVED##
        #Drive the resonator
        #The conjugate on the displacement amplitude and drive phase is there because you're driving FROM displacementAmp
        #back to the origin!
        r.xy = eh.mix(r, env.gaussian(resstart - rdelay, r.piFWHM, np.conjugate(displacementAmp*r['timingPhase'])), freq='freq')
        time += 4
        time = max(time, resstart)
        #Probe the resonator with the qubit
        q.z += env.rect(time+12, currLen, sa, overshoot=q[paramName+'SwapOvershoot'])
        time += currLen + 4+12

        q.z += eh.measurePulse(q, time)

        q['readout'] = True
        data = yield runQubits(server, qubits, stats=stats, probs=[1])
        data = np.hstack(([displacementAmp.real, displacementAmp.imag, currLen], data))
        returnValue(data)
    result = sweeps.run(func, sweep(), dataset=save and dataset, noisy=noisy)
    if collect:
        return result

def resdrivephase(Sample, measure, measureR, paramName, tf,points=400, stats=1500, unitAmpl=0.3,
                  tuneOS=False, name='resonator drive phase', save=True, collect=True, noisy=True):

    sample, qubits, Qubits = util.loadDeviceType(Sample, 'phaseQubit', write_access=True)
    sample, resonators, Resonators = util.loadDeviceType
    q = qubits[measure]
    r = resonators[measureR]
    R = Resonators[measureR]

    angle = np.linspace(0,2*np.pi, points, endpoint=False)
    displacement=unitAmpl*np.exp(1j*angle)
    if tuneOS:
        os = q[paramName+'SwapAmpOvershoot']
    else:
        os = 0.0

    # kw = {'stats': stats}
    # dataset = sweeps.prepDataset(sample, name, axes=[('displacement','re'),('displacement','im')], measure=measure, kw=kw)

    # def func(server, curr):
        # start = 0
        # q.xy = eh.mix(q, env.gaussian(start, q.piFWHM, amp = q.piAmp/2), freq = 'f10')
        # start += q.piLen/2
        # q.z = env.rect(start, q['swapLen'+paraName], q['swapAmp'+paraName],overshoot = os)
        # start += q['swapLen'+paraName]+r.piLen/2
        # r.xy = eh.mix(r, env.gaussian(start,r.piFWHM, amp = np.conjugate(curr*r.drivePhase)), freq = 'fRes0')
        # start += r.piLen/2
        # q.z += env.rect(start, q['swapLen'+paraName], q['swapAmp'+paraName],overshoot = os)
        # start += q['swapLen'+paraName]
        # q.z += eh.measurePulse(q, start)
        # q['readout'] = True
        # data = yield runQubits(server, qubits, stats, probs=[1])
        # data = np.hstack(([curr.real, curr.imag], data))
        # returnValue(data)

    # result = sweeps.run(func, displacement, dataset=save and dataset, noisy=noisy)
    result = arbitraryWigner(sample, measure, measureR, paramName, np.array([tf(1.0)])*ns, pulseAmplitudes=[0.5*q.piAmp],
              probePulseLength = st.r[tf(1.0):tf(1.0):1,ns], alpha=displacement,
              stats=stats, save=False, collect=True, noisy=False, name='Resonator drive phase')
    result = result[:, [0,3]]
    result[:,0] = angle
    def fitfunc(angle,p):
        return p[0]+p[1]*np.cos(angle-p[2])
    def errfunc(p):
        return result[:,1]-fitfunc(result[:,0],p)
    p,ok = leastsq(errfunc, [0.0,100.0,0.0])
    if p[1] < 0:
        p[1] = -p[1]
        p[2] = p[2]+np.pi
    p[2] = (p[2]+np.pi)%(2*np.pi)-np.pi
    plt.plot(result[:,0],result[:,1])
    plt.plot(angle, fitfunc(angle,p))
    a = r.drivePhase*np.exp(1.0j*p[2])
    print 'Resonator drive Phase correction: %g' % p[2]
    R.drivePhase = a/abs(a)
    return
