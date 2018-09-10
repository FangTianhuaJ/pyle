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

#import qubitpulsecal as qpc
from pyle.dataking import sweeps
from pyle.dataking import util

import labrad


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


def swapSpectros(sample,measures):
    if not isinstance(measures,list):
        measures=[measures]
    swapLen = st.r[0:500:5,ns]
    #Qubit 0
    #6.0 Ghz = 0.275
    #6.6 GHz = -0.171
    #Qubit 1
    #6.0 GHz = 0.4025
    #6.6 GHz = -0.09
    swapAmps = [np.linspace(-0.171, 0.275, 250), np.linspace(-0.09, 0.4025, 250)]
    for meas, swapAmp in zip(measures, swapAmps):
        mq.swapSpectroscopy(sample, swapLen=swapLen, swapAmp=swapAmp, measure=meas, stats=300L,
                            name='Swap Spectroscopy', save=True, collect=False, noisy=True, state=1, piPulse=True)
        time.sleep(5)
        sample._cxn.telecomm_server.send_sms('experiment done','the experiment on qubit %d is finished' %meas,'Daniel')

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

