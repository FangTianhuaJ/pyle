import numpy as np
import matplotlib.pyplot as plt

from labrad.units import Unit
V, mV, sec, us, ns, GHz, MHz = [Unit(s) for s in ('V', 'mV', 's', 'us', 'ns', 'GHz', 'MHz')]

import pyle.envelopes as env
from pyle.pipeline import returnValue, FutureList
from pyle.dataking import envelopehelpers as eh
from pyle.dataking import measurement
from pyle.dataking import adjust
from pyle.dataking.fpgaseq import runQubits
from pyle.util import sweeptools as st
from pyle.dataking import multiqubitPQ as mq
from pyle.dataking import utilMultilevels as ml
from pyle.dataking import dephasingSweeps as dp
import time

import random

import qubitpulsecal as qpc
import sweeps
import util

import labrad

def bringup(s,meas,initial=False):
    device = s[s['config'][meas]]
    if initial:
        mq.squidsteps(s, measure=meas)
        mq.stepedge(s,measure=meas)
        mq.findSquidEdges(s,meas,plot=True)
        return
    mq.scurve(s,measure=meas)
    mq.spectroscopy(s,measure=meas)
    mq.spectroscopy_two_state(s, measure=meas)
    mq.rabihigh(s, measure=meas)
    mq.freqtuner(s,measure=meas,save=True)
    mq.pituner(s, measure=meas, save=False, update=True, noisy=True, state=1)
    mq.testdelay(s,measure=meas)
    mq.find_mpa_func(s,measure=meas,plot=True)
    mq.find_flux_func(s,measure=meas,plot=True)
    mq.find_zpa_func(s,measure=meas,plot=True)
    mq.findDfDv(s, meas, freqScan=None, measAmplFunc=None,
                fluxBelow=2*mV, fluxAbove=2*mV, fluxStep=0.1*mV, sb_freq=0*GHz, stats=300L,
                plot=True, save=True, name='Find dfreq/dV', collect=False, update=True, noisy=True)
    #Bring up |2>
    mq.find_mpa(s, stats=60, target=0.05, mpa_range=(-2.0, 2.0), state=2,
                measure=meas, pulseFunc=None, resolution=0.005, blowup=0.05,
                falling=None, statsinc=1.25,
                save=False, name='SCurve Search MQ', collect=False, update=True, noisy=True)
    mq.rabihigh(s, measure=meas, state=2)
    mq.freqtuner(s, measure=meas, save=True, update=True, state=2)
    mq.pituner(s, measure=meas, save=False, update=True, noisy=True, state=2)
    mpa1 = device['measureAmp']
    mpa2 = device['measureAmp2']
    mpa = np.hstack((np.linspace(mpa1-0.05,mpa1+0.05,80),np.linspace(mpa2-0.05,mpa2+0.05,80)))
    mq.visibility(s, mpa=mpa, stats=1500, calstats=12000, measure=meas, states=[1,2], save=True, collect=False, noisy=True, update=True)
    mq.ramseyFilter(s, measure=meas, save=True, collect=False, noisy=False)


def bringup2State(s, measure):
    """Bring up 2 state assuming you already have a reasonable operating point"""
    #Tune up pi pulse
    mq.rabihigh(s, measure=measure)
    mq.freqtuner(s, measure=measure)
    #Get rough |2> mpa
    mq.find_mpa(s, stats=60, target=0.05, mpa_range(-2.0,2.0), state=2,
                measure=measure, pulsefunc=None, resolution=0.005, blowup = 0.05,
                falling = None, statsinc = 1.25,
                save=False, name='SCurve Search MQ', collect=False, update=True, noisy=True)
    #Get rough |2> pi pulse
    mq.rabihigh(s, measure=measure, state=2)
    mq.freqtuner(s, measure=measure, save=True, update=True, state=2)
    mq.pituner(s, measure=measure, save=False, update=True, noisey=True, state=2)




def dephasingSuite(s,meas,username='Daniel'):
    mq.t1(s, delay=st.r[-10:1500:50,ns], stats=12000L, measure=meas,
          name='T1', save=True, collect=False, noisy=False, state=1,
          update=True, plot=True)
    dp.ramsey(s, meas, delay=st.r[0:400:4,ns], stats=600, save=True, noisy=False, collect=False,
              randomize=False, averages=20, tomo=True, state=1, plot=True, update=True)
    dp.spinEcho(s, measure=meas, delay=st.r[0:1000:10,ns], df=50*MHz,
                stats=600L, name='Spin Echo', save=True,
                collect=False, noisy=False, randomize=False, averages=20, tomo=True)
    rabiParams = [(0.1, 10*ns,  5.0*ns),
                  (0.15,15*ns,  4.0*ns),
                  (0.2, 15*ns,  4.0*ns),
                  (0.25,15*ns,  3.0*ns),
                  (0.3, 15*ns,  3.0*ns),
                  (0.35,20*ns,  3.0*ns),
                  (0.4, 20*ns,  3.0*ns),
                  (0.45,20*ns,  2.5*ns),
                  (0.5, 20*ns,  2.0*ns),
                  (0.55,35*ns,  1.5*ns),
                  (0.6, 35*ns,  1.5*ns)
                  ]
    for amp,turnOnWidth, dt in rabiParams:
        dp.rabi(s, length=st.r[0:1200:dt,ns], amplitude=amp, measure=meas, stats=600, save=True, collect=False,
                noisy=False, useHd=False, averages=10, check2State=False, turnOnWidth=turnOnWidth)
    if username is not None:
        with labrad.connect() as cxn:
            try:
                cxn.telecomm_server.send_sms('Scan complete','Iteration of scans complete',username)
            except Exception:
                print 'Failed to send text message'

def rabiSuite(s,meas,iterations,username='Daniel'):
    rabiParams = [(0.1, 10*ns,  5.0*ns),
                  (0.15,15*ns,  4.0*ns),
                  (0.2, 15*ns,  4.0*ns),
                  (0.25,15*ns,  3.0*ns),
                  (0.3, 15*ns,  3.0*ns),
                  (0.35,20*ns,  3.0*ns),
                  (0.4, 20*ns,  3.0*ns),
                  (0.45,20*ns,  2.5*ns),
                  (0.5, 20*ns,  2.0*ns),
                  (0.55,35*ns,  1.5*ns),
                  (0.6, 35*ns,  1.5*ns)
                  ]
    def doT1():
        mq.t1(s, delay=st.r[-10:1500:50,ns], stats=12000L, measure=meas,
              name='T1', save=True, collect=False, noisy=False, state=1,
              update=True, plot=True)
    doT1()
    dp.ramsey(s, meas, delay=st.r[0:400:4,ns], stats=600, save=True, noisy=False, collect=False,
              randomize=False, averages=20, tomo=True, state=1, plot=True, update=True)
    dp.spinEcho(s, measure=meas, delay=st.r[0:1000:10,ns], df=50*MHz,
                stats=600L, name='Spin Echo', save=True,
                collect=False, noisy=False, randomize=False, averages=20, tomo=True)
    for i in range(iterations):
        doT1()
        for amp,turnOnWidth,dt in rabiParams:
            dp.rabi(s, length=st.r[0:1200:dt,ns], amplitude=amp, measure=meas, stats=600, save=True, collect=False,
                    noisy=False, useHd=False, averages=10, check2State=False, turnOnWidth=turnOnWidth)
        if username is not None:
            try:
                s._cxn.telecomm_server.send_sms('Scan complete', 'Iteration %d complete'%(i+1),username)
            except Exception:
                print 'Failed to send SMS'

    #dp.ramsey_oscilloscope(s, measure=meas, holdTime = 100*ns, fringeFreq = 50*MHz, timeStep = 1*sec,
    #                       stats=600, name='RamseyScope', save = True)

#DEPRICATED
#def measureCrosstalk(Sample, control, target, delay=st.r[-20:20:1,ns], stats=600,
#                     name='Measurement Crosstalk',
#                     save=True, collect=False, noisy=False):
#    sample,qubits = util.loadDeviceType(Sample,'phaseQubit')
#    qC = qubits[control]
#    qT = qubits[target]
#    measure=[control,target]
#    axes = [(delay,'Delay')]
#    deps = [('Probability','|'+s+'>','') for s in ['00','01','10','11']]
#    kw = {'stats':stats}
#    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)
#    def func(server,time):
#        qC['readout']=True
#        qT['readout']=True
#        t=0.0*ns
#        qC['xy']=eh.boostState(qC, qC['piLen']/2.0, 1)
#        t+=qC['piLen']
#        qC['z']=eh.measurePulse(qC, t, 1)
#        qT['z']=eh.measurePulse(qT, time, 1)
#        return runQubits(server, qubits, stats)
#    data = sweeps.grid(func,axes,dataset=save and dataset, collect=collect, noisy=noisy)
#    if collect:
#        return data

def uWaveCrosstalk(Sample, control, target, delay=st.r[-100:100:1,ns],stats=600,
                   name=None, save=True, collect=False, noisy=False):
    """ Pi pulse on control qubit followed by measure pulse on target """
    sample,qubits = util.loadDeviceType(Sample,'phaseQubit')
    qC,qT = qubits[control],qubits[target]
    measure=[control,target]
    axes = [(delay,'Delay')]
    deps = [('Probability','|'+s+'>','') for s in ['00','01','10','11']]
    kw = {'stats':stats}
    if name is None:
        controlName = Sample['config'][control]
        targetName = Sample['config'][target]
        name = 'uWave Crosstalk, Control=|%s>, Target=|%s>' %(controlName,targetName)
    dataset = sweeps.prepDataset(sample,name,axes,deps,measure=measure,kw=kw)
    def func(server, time):
        qT['readout']=True
        qC['readout']=True
        t=0.0*ns
        qC['xy']=eh.boostState(qC, qC['piLen']/2.0, state=1)
        t+=qC['piLen']+time
        qT['z']=eh.measurePulse(qT,t,state=1)
        return runQubits(server,qubits,stats)
    data = sweeps.grid(func,axes,dataset=save and dataset, collect=collect, noisy=noisy)
    if collect:
        return data

def measureCrosstalk(Sample, control, target, measureAmpControl, measureAmpTarget, delay=st.r[-100:100:1,ns], stats=600,
                     name=None, save=True, collect=False, noisy=False):
    sample,qubits = util.loadDeviceType(Sample,'phaseQubit')
    qC,qT = qubits[control],qubits[target]
    measure=[control,target]
    qC['measureAmp']=measureAmpControl
    qT['measureAmp']=measureAmpTarget
    axes = [(delay,'Delay')]
    deps = [('Probability','|'+s+'>','') for s in ['00','01','10','11']]
    kw = {'stats':stats}
    if name is None:
        controlName = Sample['config'][control]
        targetName = Sample['config'][target]
        name = 'Measurement crosstalk, Control=|%s>, Target=|%s>' %(controlName,targetName)
    dataset = sweeps.prepDataset(sample,name,axes,deps,measure=measure,kw=kw)
    def func(server,time):
        qT['readout']=True
        qC['readout']=True
        qC['z']=eh.measurePulse(qC, 0*ns, state=1)
        qT['z']=eh.measurePulse(qT, time, state=1)
        return runQubits(server,qubits,stats)
    data = sweeps.grid(func,axes, dataset=save and dataset, collect=collect, noisy=noisy)
    if collect:
        return data


def swapSpectros(sample,measures):
    if not isinstance(measures,list):
        measures=[measures]
    swapLen = st.r[0:100:2,ns]
    swapAmp = np.arange(-0.4,0.4,0.004)
    for meas in measures:
        mq.swapSpectroscopy(sample, swapLen=swapLen, swapAmp=swapAmp, measure=meas, stats=300L,
                            name='Swap Spectroscopy', save=True, collect=False, noisy=True, state=1, piPulse=True)
        time.sleep(5)
        mq.swapSpectroscopy(sample, swapLen=swapLen, swapAmp=swapAmp, measure=meas, stats=300L,
                            name='Swap Spectroscopy', save=True, collect=False, noisy=True, state=1, piPulse=False,
                            username='Daniel')

def t1_at_zpa(sample, zpa=None, delay=st.r[0:1000:20, ns], stats=3000, save=True,
              measure=0, name='T1 With Z pulse', noisy=True):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    if zpa is None:
        raise Exception('Need to enter a zpa!')

    axes = [(delay, 'Measure pulse delay')]
    kw = {'stats': stats, 'zpa': zpa}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, delay):
        start = 0*ns
        q.xy = eh.mix(q, eh.piPulse(q, start))
        start+=q.piLen
        q.z = env.rect(start, delay+q['measureLenTop']+q['measureLenFall'], zpa)
        start+=delay
        q.z += env.trapezoid(start, 0, q['measureLenTop'], q['measureLenFall'], q['measureAmp']-zpa)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    return sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)

