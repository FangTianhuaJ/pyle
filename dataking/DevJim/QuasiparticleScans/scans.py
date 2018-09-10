import numpy as np
from scipy.optimize import leastsq

from labrad.units import Unit
mK, V, mV, us, ns, GHz, MHz, dBm = [Unit(s) for s in ('mK', 'V', 'mV', 'us', 'ns', 'GHz', 'MHz', 'dBm')]

import pyle.envelopes as env
from pyle.dataking import envelopehelpers as eh
from pyle.dataking.fpgaseq import runQubits
from pyle.util import sweeptools as st
from pyle.dataking import multiqubitPQ as mq
from pyle.dataking.quasiparticle import multiqubit_swap as mqs
from pyle.workflow import switchSession as pss
from pyle import registry
from pyle.dataking import util
import labrad
import time
from pyle.dataking.quasiparticle import setbias
from pyle.dataking.quasiparticle import twostate as ts
from pyle.dataking import dephasingSweeps
from pyle.dataking.quasiparticle import dephasingSweeps_swap
from pyle.dataking import utilMultilevels as ml
from pyle.dataking import sweeps


def singlehittite(cxn, sample, target=0.05, measure=0, maxiter=10, noisy=False):

    # Get step edge
    ts.getMixTemp(cxn, sample, measure=measure)
    mq.find_step_edge(sample,measure=measure,noisy=noisy)
    mq.stepedge(sample,measure=measure,noisy=noisy,update=False)
    mq.rabihigh(sample,measure=measure,noisy=noisy,update=False,name='Rabi Raw')
    mq.scurve(sample, states=[0,1,2], mpa=st.r[0:0.55:0.001], visibility=True, update=False, measure=measure, noisy=noisy,name='SCurve Raw')
    mq.t1(sample,delay=st.r[20:3000:20,ns],stats=3000L,measure=measure,noisy=noisy, plot=False,name='T1 Raw')
    ts.measure2(sample,measstate=2,stats=18000L,measure=measure,noisy=noisy,zeroing=0,name=' Raw')

    #Tune up |1) state: MPA, Pi-Pulse, Frequency
    mq.find_mpa(sample,stats=300,state=1,resolution=0.005,target=target,measure=measure,noisy=noisy, mpa_range=(0,2.0))
    mq.spectroscopy_two_state(sample, measure=measure, noisy=noisy)

    #Tune up |1) state: MPA, Pi-Pulse, Frequency
    freq1=[ts.getFreq(sample, measure=measure, state=1)]
    freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
    mq.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
    freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
    mq.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
    count = 2
    while (count<maxiter) and ((abs(freq1[-1]-freq1[-2])>0.0006*GHz) or (abs(freq1[-2]-freq1[-3])>0.0006*GHz)):
        # Iteratively optimize Pi-Pulse and Frequency
        print ('f01: '+str([freq['GHz'] for freq in freq1]))
        freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
        mq.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
        count = count+1
    print ('f01: '+str([freq['GHz'] for freq in freq1]))
    mq.rabihigh(sample,measure=measure,noisy=noisy,update=False)
    mpa1 = mq.find_final_mpa(sample,stats=300,state=1,resolution=0.005,target=target,measure=measure,noisy=noisy)

    #Tune up |2) state: MPA, Pi-Pulse, Frequency
    mq.find_mpa(sample,stats=300,state=2,resolution=0.005,target=target,measure=measure,noisy=noisy,mpa_range=(0,2.0))
    freq2=[ts.getFreq(sample, measure=measure, state=2)]
    freq2.append(mq.freqtuner(sample,state=2,measure=measure,noisy=noisy))
    mq.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
    freq2.append(mq.freqtuner(sample,state=2,measure=measure,noisy=noisy))
    mq.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
    count = 2
    while (count<maxiter) and ((abs(freq2[-1]-freq2[-2])>0.0006*GHz) or (abs(freq2[-2]-freq2[-3])>0.0006*GHz)):
        # Iteratively optimize Pi-Pulse and Frequency
        print ('f12: '+str([freq['GHz'] for freq in freq2]))
        freq2.append(mq.freqtuner(sample,state=2,measure=measure,noisy=noisy))
        mq.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
        count = count+1
    print ('f12: '+str([freq['GHz'] for freq in freq2]))
    mpa2 = mq.find_final_mpa(sample,stats=300,state=2,resolution=0.005,target=target,measure=measure,noisy=noisy)

    # Measure T1, T2, Visibility
    mq.t1(sample,delay=st.r[20:3000:20,ns],stats=3000L,measure=measure,noisy=noisy, plot=False)
    mq.visibility(sample, states=[1,2], mpa=st.r[max(mpa2-0.15,0):min(mpa1+0.1,2):0.001], calstats=12000, update=False, measure=measure, noisy=noisy)

    # Measure P1
    ts.measure2(sample,measstate=2,stats=18000L,measure=measure,noisy=noisy,zeroing=0)


def multihittite(cxn, sample, amps, user='Jim', target=0.05, measure=0, maxiter=10, noisy=False):

    hit = cxn.hittite_t2100_server
    hit.select_device('Vince GPIB Bus - GPIB1::20')

    for amp in amps:
        print amp
        hit.amplitude(amp+57)
        newSession = 'Power'+str(int(10*amp[dBm]))+'ResFreq'
        sample2,oldSession = pss(cxn,user,session=newSession)
        mq.s_scanning(sample2,power=-50*dBm,noisy=noisy,measure=measure,update=False)
        mq.phase_arc(sample2,bias=st.r[-.1:1:.005,V],noisy=noisy,measure=measure,update=False)
        singlehittite(cxn,sample2,target=target,measure=measure,maxiter=maxiter,noisy=noisy)
        pss(cxn,user,session=oldSession)


def singlehittitebarebones(cxn, sample, target=0.05, measure=0, maxiter=10, noisy=False,updateFreq=False):

    # Get step edge
    ts.getMixTemp(cxn, sample, measure=measure)
    mq.find_step_edge(sample,measure=measure,noisy=noisy)
    mq.stepedge(sample,measure=measure,noisy=noisy,update=False)
    mq.rabihigh(sample,measure=measure,noisy=noisy,update=False,name='Rabi Raw')
    mq.visibility(sample, states=[1,2], mpa=st.r[0:0.55:0.001], calstats=12000, update=False, measure=measure, noisy=noisy, name='Raw Visibility')
    mq.t1(sample,delay=st.r[20:3000:20,ns],stats=3000L,measure=measure,noisy=noisy, plot=False,name='T1 Raw')
    ts.measure2(sample,measstate=2,stats=18000L,measure=measure,noisy=noisy,zeroing=0,name=' Raw')

    #Tune up |1) state: MPA, Pi-Pulse, Frequency
    mq.find_mpa(sample,stats=300,state=1,resolution=0.005,target=target,measure=measure,noisy=noisy, mpa_range=(0,2.0))
    mq.spectroscopy_two_state(sample, measure=measure, noisy=noisy, update=updateFreq)

    #Tune up |1) state: MPA, Pi-Pulse, Frequency
    freq1=[ts.getFreq(sample, measure=measure, state=1)]
    freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
    mq.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
    count = 1
    while (count<maxiter) and (abs(freq1[-1]-freq1[-2])>0.0006*GHz):
        # Iteratively optimize Pi-Pulse and Frequency
        print ('f01: '+str([freq['GHz'] for freq in freq1]))
        freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
        mq.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
        count = count+1
    print ('f01: '+str([freq['GHz'] for freq in freq1]))
    mq.rabihigh(sample,measure=measure,noisy=noisy,update=False)
    mpa1 = mq.find_final_mpa(sample,stats=300,state=1,resolution=0.005,target=target,measure=measure,noisy=noisy)

    #Tune up |2) state: MPA, Pi-Pulse, Frequency
    mq.find_mpa(sample,stats=300,state=2,resolution=0.005,target=target,measure=measure,noisy=noisy, mpa_range=(0,2.0))
    freq2=[ts.getFreq(sample, measure=measure, state=2)]
    freq2.append(mq.freqtuner(sample,state=2,measure=measure,noisy=noisy))
    mq.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
    count = 1
    while (count<maxiter) and (abs(freq2[-1]-freq2[-2])>0.0006*GHz):
        # Iteratively optimize Pi-Pulse and Frequency
        print ('f12: '+str([freq['GHz'] for freq in freq2]))
        freq2.append(mq.freqtuner(sample,state=2,measure=measure,noisy=noisy))
        mq.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
        count = count+1
    print ('f12: '+str([freq['GHz'] for freq in freq2]))
    mpa2 = mq.find_final_mpa(sample,stats=300,state=2,resolution=0.005,target=target,measure=measure,noisy=noisy)

    # Measure T1, T2, Visibility
    mq.t1(sample,delay=st.r[20:3000:20,ns],stats=3000L,measure=measure,noisy=noisy, plot=False)
    mq.visibility(sample, states=[1,2], mpa=st.r[max(mpa2-0.15,0):min(mpa1+0.1,2):0.001], calstats=12000, update=False, measure=measure, noisy=noisy)

    # Tune up swap to TLS
    swapLen, swapAmp = mq.swap10tuner(sample, noisy=noisy, measure=measure)
    mq.swapSpectroscopy(sample, state=1, swapLen=st.arangePQ(0,2.5*swapLen,2,ns), swapAmp=np.arange(swapAmp-0.01,swapAmp+0.01,0.0005), measure=measure, noisy=noisy)
    mq.swap10tuner(sample, stats=1800L, noisy=noisy, measure=measure)
    ts.measure2(sample,measstate=2,stats=18000L,measure=measure,noisy=noisy)

    #Tune up |1) state: MPA, Pi-Pulse, Frequency
    mqs.find_mpa(sample,stats=300,state=1,resolution=0.005,target=target,measure=measure,noisy=noisy, mpa_range=(0,2.0))
    freq1=[ts.getFreq(sample, measure=measure, state=1)]
    freq1.append(mqs.freqtuner(sample,state=1,measure=measure,noisy=noisy))
    mqs.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
    count = 1
    while (count<maxiter) and (abs(freq1[-1]-freq1[-2])>0.0006*GHz):
        # Iteratively optimize Pi-Pulse and Frequency
        print ('f01: '+str([freq['GHz'] for freq in freq1]))
        freq1.append(mqs.freqtuner(sample,state=1,measure=measure,noisy=noisy))
        mqs.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
        count = count+1
    print ('f01: '+str([freq['GHz'] for freq in freq1]))
    mqs.rabihigh(sample,measure=measure,noisy=noisy,update=False,name='Swap Rabi')
    mpa1 = mqs.find_final_mpa(sample,stats=300,state=1,resolution=0.005,target=target,measure=measure,noisy=noisy)

    #Tune up |2) state: MPA, Pi-Pulse, Frequency
    mqs.find_mpa(sample,stats=300,state=2,resolution=0.005,target=target,measure=measure,noisy=noisy, mpa_range=(0,2.0))
    freq2=[ts.getFreq(sample, measure=measure, state=2)]
    freq2.append(mqs.freqtuner(sample,state=2,measure=measure,noisy=noisy))
    mqs.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
    count = 1
    while (count<maxiter) and (abs(freq2[-1]-freq2[-2])>0.0006*GHz):
        # Iteratively optimize Pi-Pulse and Frequency
        print ('f12: '+str([freq['GHz'] for freq in freq2]))
        freq2.append(mqs.freqtuner(sample,state=2,measure=measure,noisy=noisy))
        mqs.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
        count = count+1
    print ('f12: '+str([freq['GHz'] for freq in freq2]))
    mpa2 = mqs.find_final_mpa(sample,stats=300,state=2,resolution=0.005,target=target,measure=measure,noisy=noisy)

    # Measure T1, T2, Visibility
    mqs.t1(sample,delay=st.r[20:3000:20,ns],stats=3000L,measure=measure,noisy=noisy, plot=False, name='Swap T1')
    mqs.visibility(sample, states=[1,2], mpa=st.r[max(mpa2-0.15,0):min(mpa1+0.1,2):0.001], calstats=12000, name='Swap Visibility', update=False, measure=measure, noisy=noisy)

    # Measure P1
    ts.measure2(sample,measstate=2,stats=18000L,measure=measure,noisy=noisy,zeroing=1,name=' Swap')


def singlehittiteswap(cxn, sample, target=0.05, measure=0, maxiter=10, noisy=False,updateFreq=False):

    # Get step edge
    ts.getMixTemp(cxn, sample, measure=measure)
    mq.find_step_edge(sample,measure=measure,noisy=noisy)
    mq.stepedge(sample,measure=measure,noisy=noisy,update=False)
    mq.rabihigh(sample,measure=measure,noisy=noisy,update=False,name='Rabi Raw')
    mq.visibility(sample, states=[1,2], mpa=st.r[0:0.55:0.001], calstats=12000, update=False, measure=measure, noisy=noisy, name='Raw Visibility')
    mq.t1(sample,delay=st.r[20:3000:20,ns],stats=3000L,measure=measure,noisy=noisy, plot=False,name='T1 Raw')
    ts.measure2(sample,measstate=2,stats=18000L,measure=measure,noisy=noisy,zeroing=0,name=' Raw')

    #Tune up |1) state: MPA, Pi-Pulse, Frequency
    mq.find_mpa(sample,stats=300,state=1,resolution=0.005,target=target,measure=measure,noisy=noisy, mpa_range=(0,2.0))
    mq.spectroscopy_two_state(sample, measure=measure, noisy=noisy, update=updateFreq)

    #Tune up |1) state: MPA, Pi-Pulse, Frequency
    freq1=[ts.getFreq(sample, measure=measure, state=1)]
    freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
    mq.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
    freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
    mq.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
    count = 2
    while (count<maxiter) and ((abs(freq1[-1]-freq1[-2])>0.0006*GHz) or (abs(freq1[-2]-freq1[-3])>0.0006*GHz)):
        # Iteratively optimize Pi-Pulse and Frequency
        print ('f01: '+str([freq['GHz'] for freq in freq1]))
        freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
        mq.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
        count = count+1
    print ('f01: '+str([freq['GHz'] for freq in freq1]))
    mq.rabihigh(sample,measure=measure,noisy=noisy,update=False)
    mpa1 = mq.find_final_mpa(sample,stats=300,state=1,resolution=0.005,target=target,measure=measure,noisy=noisy)

    #Tune up |2) state: MPA, Pi-Pulse, Frequency
    mq.find_mpa(sample,stats=300,state=2,resolution=0.005,target=target,measure=measure,noisy=noisy, mpa_range=(0,2.0))
    freq2=[ts.getFreq(sample, measure=measure, state=2)]
    freq2.append(mq.freqtuner(sample,state=2,measure=measure,noisy=noisy))
    mq.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
    freq2.append(mq.freqtuner(sample,state=2,measure=measure,noisy=noisy))
    mq.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
    count = 2
    while (count<maxiter) and ((abs(freq2[-1]-freq2[-2])>0.0006*GHz) or (abs(freq2[-2]-freq2[-3])>0.0006*GHz)):
        # Iteratively optimize Pi-Pulse and Frequency
        print ('f12: '+str([freq['GHz'] for freq in freq2]))
        freq2.append(mq.freqtuner(sample,state=2,measure=measure,noisy=noisy))
        mq.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
        count = count+1
    print ('f12: '+str([freq['GHz'] for freq in freq2]))
    mpa2 = mq.find_final_mpa(sample,stats=300,state=2,resolution=0.005,target=target,measure=measure,noisy=noisy)

    # Get swap spectroscopy and calibration functions
    freqRange = st.r[(freq2[-1]['GHz']-.05):(freq1[-1]['GHz']+.05):.005,GHz]
    mq.find_zpa_func(sample, stats=120L, freqScan=freqRange, measure=measure, noisy=noisy)
    zpa1 = mq.freq2zpa(sample,0,freq1[-1])
    zpa2 = mq.freq2zpa(sample,0,freq2[-1])
    mq.swapSpectroscopy(sample, swapLen=st.arangePQ(10,500,10,ns), swapAmp=np.arange((min(zpa1,zpa2)-0.02),(max(zpa1,zpa2)+0.02),0.002), measure=measure, noisy=noisy)
    mq.find_mpa_func(sample, measure=measure, noisy=noisy)
    mq.find_flux_func(sample, stats=120L, freqScan=freqRange, measure=measure, noisy=noisy)

    # Measure T1, T2, Visibility
    mq.t1(sample,delay=st.r[20:3000:20,ns],stats=3000L,measure=measure,noisy=noisy, plot=False)
    dephasingSweeps.ramsey(sample, delay=st.r[0:300:1, ns], plot=False, update=False, measure=measure, noisy=noisy)
    mq.visibility(sample, states=[1,2], mpa=st.r[max(mpa2-0.15,0):min(mpa1+0.1,2):0.001], calstats=12000, update=False, measure=measure, noisy=noisy)

    # Tune up swap to TLS
    swapLen, swapAmp = mq.swap10tuner(sample, noisy=noisy, measure=measure)
    mq.swapSpectroscopy(sample, state=1, swapLen=st.arangePQ(0,2.5*swapLen,2,ns), swapAmp=np.arange(swapAmp-0.01,swapAmp+0.01,0.0005), measure=measure, noisy=noisy)
    mq.swap10tuner(sample, stats=1800L, noisy=noisy, measure=measure)
    ts.measure2(sample,measstate=2,stats=18000L,measure=measure,noisy=noisy)

    #Tune up |1) state: MPA, Pi-Pulse, Frequency
    mqs.find_mpa(sample,stats=300,state=1,resolution=0.005,target=target,measure=measure,noisy=noisy, mpa_range=(0,2.0))
    freq1=[ts.getFreq(sample, measure=measure, state=1)]
    freq1.append(mqs.freqtuner(sample,state=1,measure=measure,noisy=noisy))
    mqs.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
    freq1.append(mqs.freqtuner(sample,state=1,measure=measure,noisy=noisy))
    mqs.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
    count = 2
    while (count<maxiter) and ((abs(freq1[-1]-freq1[-2])>0.0006*GHz) or (abs(freq1[-2]-freq1[-3])>0.0006*GHz)):
        # Iteratively optimize Pi-Pulse and Frequency
        print ('f01: '+str([freq['GHz'] for freq in freq1]))
        freq1.append(mqs.freqtuner(sample,state=1,measure=measure,noisy=noisy))
        mqs.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
        count = count+1
    print ('f01: '+str([freq['GHz'] for freq in freq1]))
    mqs.rabihigh(sample,measure=measure,noisy=noisy,update=False,name='Swap Rabi')
    mpa1 = mqs.find_final_mpa(sample,stats=300,state=1,resolution=0.005,target=target,measure=measure,noisy=noisy)

    #Tune up |2) state: MPA, Pi-Pulse, Frequency
    mqs.find_mpa(sample,stats=300,state=2,resolution=0.005,target=target,measure=measure,noisy=noisy, mpa_range=(0,2.0))
    freq2=[ts.getFreq(sample, measure=measure, state=2)]
    freq2.append(mqs.freqtuner(sample,state=2,measure=measure,noisy=noisy))
    mqs.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
    freq2.append(mqs.freqtuner(sample,state=2,measure=measure,noisy=noisy))
    mqs.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
    count = 2
    while (count<maxiter) and ((abs(freq2[-1]-freq2[-2])>0.0006*GHz) or (abs(freq2[-2]-freq2[-3])>0.0006*GHz)):
        # Iteratively optimize Pi-Pulse and Frequency
        print ('f12: '+str([freq['GHz'] for freq in freq2]))
        freq2.append(mqs.freqtuner(sample,state=2,measure=measure,noisy=noisy))
        mqs.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
        count = count+1
    print ('f12: '+str([freq['GHz'] for freq in freq2]))
    mpa2 = mqs.find_final_mpa(sample,stats=300,state=2,resolution=0.005,target=target,measure=measure,noisy=noisy)

    # Measure T1, T2, Visibility
    mqs.t1(sample,delay=st.r[20:3000:20,ns],stats=3000L,measure=measure,noisy=noisy, plot=False, name='Swap T1')
    dephasingSweeps_swap.ramsey(sample, delay=st.r[0:300:1,ns], plot=False, update=False, measure=measure, noisy=noisy, name='Swap Ramsey')
    mqs.visibility(sample, states=[1,2], mpa=st.r[max(mpa2-0.15,0):min(mpa1+0.1,2):0.001], calstats=12000, name='Swap Visibility', update=False, measure=measure, noisy=noisy)

    # Measure P1
    ts.measure2(sample,measstate=2,stats=18000L,measure=measure,noisy=noisy,zeroing=1,name=' Swap')


def multihittiteswap(cxn, sample, amps, user='Jim', target=0.05, measure=0, maxiter=10, noisy=False, updateFreq=False):

    hit = cxn.hittite_t2100_server
    hit.select_device('Vince GPIB Bus - GPIB1::20')

    for amp in amps:
        #Tune up case without additional power
        hit.output(False)
        tuneup2barebones(cxn, sample, target=target, measure=measure, maxiter=maxiter, noisy=noisy, updateFreq=False)

        #Apply microwaves from 2nd Hittite
        print amp
#        hit.amplitude(amp+57)
        hit.amplitude(amp+23)
        hit.output(True)
        newSession = 'Power'+str(int(10*amp[dBm]))+'Freq55'
        sample2,oldSession = pss(cxn,user,session=newSession)
        oldParams = getParams(sample2,measure=measure,state=1)
        setParams(cxn,sample2,oldParams,measure=measure,state=1)
        mq.s_scanning(sample2,power=-50*dBm,noisy=noisy,measure=measure,update=False)
        mq.phase_arc(sample2,bias=st.r[-.1:1:.005,V],noisy=noisy,measure=measure,update=False)
        singlehittitebarebones(cxn,sample2,target=target,measure=measure,maxiter=maxiter,noisy=noisy,updateFreq=updateFreq)
        pss(cxn,user,session=oldSession)


def multihittitefreq(cxn, sample, freqs, user='Jim', target=0.05, measure=0, maxiter=10, noisy=False, updateFreq=False):

    hit = cxn.hittite_t2100_server
    hit.select_device('Vince GPIB Bus - GPIB1::20')

    for freq in freqs:
        #Tune up case without additional power
        hit.output(False)
        tuneup2barebones(cxn, sample, target=target, measure=measure, maxiter=maxiter, noisy=noisy, updateFreq=False)

        #Apply microwaves from 2nd Hittite
        print freq
        hit.frequency(freq)
        hit.output(True)
        newSession = 'Power-30Freq'+str(int(amp[MHz]))
        sample2,oldSession = pss(cxn,user,session=newSession)
        oldParams = getParams(sample2,measure=measure,state=1)
        setParams(cxn,sample2,oldParams,measure=measure,state=1)
        mq.s_scanning(sample2,power=-50*dBm,noisy=noisy,measure=measure,update=False)
        mq.phase_arc(sample2,bias=st.r[-.1:1:.005,V],noisy=noisy,measure=measure,update=False)

        # Get uncalibrated P1
        ts.getMixTemp(cxn, sample, measure=measure)
        mq.find_step_edge(sample,measure=measure,noisy=noisy)
        mq.stepedge(sample,measure=measure,noisy=noisy,update=False)
        mq.rabihigh(sample,measure=measure,noisy=noisy,update=False,name='Rabi Raw')
        mq.visibility(sample, states=[1,2], mpa=st.r[0:0.55:0.001], calstats=12000, update=False, measure=measure, noisy=noisy, name='Raw Visibility')
        mq.t1(sample,delay=st.r[20:3000:20,ns],stats=3000L,measure=measure,noisy=noisy, plot=False,name='T1 Raw')
        ts.measure2(sample,measstate=2,stats=18000L,measure=measure,noisy=noisy,zeroing=0,name=' Raw')
        pss(cxn,user,session=oldSession)


def tuneup2(cxn, sample, target=0.05, measure=0, maxiter=10, noisy=False,updateFreq=False):

    #Tune up |1) state: MPA, Pi-Pulse, Frequency
    mq.find_mpa(sample,stats=300,state=1,resolution=0.005,target=target,measure=measure,noisy=noisy, mpa_range=(0,2.0))
    freq1=[ts.getFreq(sample, measure=measure, state=1)]
    freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
    mq.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
    freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
    mq.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
    count = 2
    while (count<maxiter) and ((abs(freq1[-1]-freq1[-2])>0.0006*GHz) or (abs(freq1[-2]-freq1[-3])>0.0006*GHz)):
        # Iteratively optimize Pi-Pulse and Frequency
        print ('f01: '+str([freq['GHz'] for freq in freq1]))
        freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
        mq.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
        count = count+1
    print ('f01: '+str([freq['GHz'] for freq in freq1]))
    mpa1 = mq.find_final_mpa(sample,stats=300,state=1,resolution=0.005,target=target,measure=measure,noisy=noisy)

    #Tune up |2) state: MPA, Pi-Pulse, Frequency
    mq.find_mpa(sample,stats=300,state=2,resolution=0.005,target=target,measure=measure,noisy=noisy, mpa_range=(0,2.0))
    freq2=[ts.getFreq(sample, measure=measure, state=2)]
    freq2.append(mq.freqtuner(sample,state=2,measure=measure,noisy=noisy))
    mq.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
    freq2.append(mq.freqtuner(sample,state=2,measure=measure,noisy=noisy))
    mq.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
    count = 2
    while (count<maxiter) and ((abs(freq2[-1]-freq2[-2])>0.0006*GHz) or (abs(freq2[-2]-freq2[-3])>0.0006*GHz)):
        # Iteratively optimize Pi-Pulse and Frequency
        print ('f12: '+str([freq['GHz'] for freq in freq2]))
        freq2.append(mq.freqtuner(sample,state=2,measure=measure,noisy=noisy))
        mq.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
        count = count+1
    print ('f12: '+str([freq['GHz'] for freq in freq2]))
    mpa2 = mq.find_final_mpa(sample,stats=300,state=2,resolution=0.005,target=target,measure=measure,noisy=noisy)

   # Tune up swap to TLS
    mq.swap10tuner(sample, stats=1800L, noisy=noisy, measure=measure)


def tuneup2barebones(cxn, sample, target=0.05, measure=0, maxiter=10, noisy=False,updateFreq=False):

    #Tune up |1) state: MPA, Pi-Pulse, Frequency
    mq.find_mpa(sample,stats=300,state=1,resolution=0.005,target=target,measure=measure,noisy=noisy, mpa_range=(0,2.0))
    freq1=[ts.getFreq(sample, measure=measure, state=1)]
    freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
    mq.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
    count = 1
    while (count<maxiter) and (abs(freq1[-1]-freq1[-2])>0.0006*GHz):
        # Iteratively optimize Pi-Pulse and Frequency
        print ('f01: '+str([freq['GHz'] for freq in freq1]))
        freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
        mq.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
        count = count+1
    print ('f01: '+str([freq['GHz'] for freq in freq1]))
    mpa1 = mq.find_final_mpa(sample,stats=300,state=1,resolution=0.005,target=target,measure=measure,noisy=noisy)

    #Tune up |2) state: MPA, Pi-Pulse, Frequency
    mq.find_mpa(sample,stats=300,state=2,resolution=0.005,target=target,measure=measure,noisy=noisy, mpa_range=(0,2.0))
    freq2=[ts.getFreq(sample, measure=measure, state=2)]
    freq2.append(mq.freqtuner(sample,state=2,measure=measure,noisy=noisy))
    mq.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
    count = 1
    while (count<maxiter) and (abs(freq2[-1]-freq2[-2])>0.0006*GHz):
        # Iteratively optimize Pi-Pulse and Frequency
        print ('f12: '+str([freq['GHz'] for freq in freq2]))
        freq2.append(mq.freqtuner(sample,state=2,measure=measure,noisy=noisy))
        mq.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
        count = count+1
    print ('f12: '+str([freq['GHz'] for freq in freq2]))
    mpa2 = mq.find_final_mpa(sample,stats=300,state=2,resolution=0.005,target=target,measure=measure,noisy=noisy)

   # Tune up swap to TLS
    mq.swap10tuner(sample, stats=1800L, noisy=noisy, measure=measure)


def findFreq(cxn, sample, target=0.05, measure=0, maxiter=10, noisy=False):
    #Tune up |1) state: MPA, Pi-Pulse, Frequency
    mq.find_mpa(sample,stats=300,state=1,resolution=0.005,target=target,measure=measure,noisy=noisy, mpa_range=(0,2.0))
    freq1=[ts.getFreq(sample, measure=measure, state=1)]
    freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
    mq.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
    count = 1
    while (count<maxiter) and (abs(freq1[-1]-freq1[-2])>0.0006*GHz):
        # Iteratively optimize Pi-Pulse and Frequency
        print ('f01: '+str([freq['GHz'] for freq in freq1]))
        freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
        mq.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
        count = count+1
    print ('f01: '+str([freq['GHz'] for freq in freq1]))
    return freq1[-1]


def findFreq2(cxn, sample, target=0.05, measure=0, maxiter=10, noisy=False):
    #Tune up |1) state: MPA, Pi-Pulse, Frequency
    mq.find_mpa(sample,stats=300,state=1,resolution=0.005,target=target,measure=measure,noisy=noisy, mpa_range=(0,2.0))
    if updateFreq: mq.spectroscopy_two_state(sample, measure=measure, noisy=noisy, update=True)
    freq1=[ts.getFreq(sample, measure=measure, state=1)]
    freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
    mq.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
    freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
    mq.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
    count = 2
    while (count<maxiter) and (abs(freq1[-1]-freq1[-2])>0.0006*GHz):
        # Iteratively optimize Pi-Pulse and Frequency
        print ('f01: '+str([freq['GHz'] for freq in freq1]))
        freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
        mq.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
        count = count+1
    print ('f01: '+str([freq['GHz'] for freq in freq1]))

    #Tune up |2) state: MPA, Pi-Pulse, Frequency
    mq.find_mpa(sample,stats=300,state=2,resolution=0.005,target=target,measure=measure,noisy=noisy, mpa_range=(0,2.0))
    freq2=[ts.getFreq(sample, measure=measure, state=2)]
    freq2.append(mq.freqtuner(sample,state=2,measure=measure,noisy=noisy))
    mq.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
    freq2.append(mq.freqtuner(sample,state=2,measure=measure,noisy=noisy))
    mq.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
    count = 2
    while (count<maxiter) and (abs(freq2[-1]-freq2[-2])>0.0006*GHz):
        # Iteratively optimize Pi-Pulse and Frequency
        print ('f01: '+str([freq['GHz'] for freq in freq1]))
        freq2.append(mq.freqtuner(sample,state=21,measure=measure,noisy=noisy))
        mq.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
        count = count+1
    print ('f12: '+str([freq['GHz'] for freq in freq2]))
    return freq1[-1],freq2[-1]


def getParams(sample, measure=0, state=1):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    ml.setMultiKeys(q,state)
    return ml.getMultiLevels(q,'frequency',state),ml.getMultiLevels(q,'measureAmp',state),ml.getMultiLevels(q,'piAmp',state)


def getParams(sample, measure=0, state=1):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    ml.setMultiKeys(q,state)
    return ml.getMultiLevels(q,'frequency',state),ml.getMultiLevels(q,'measureAmp',state),ml.getMultiLevels(q,'piAmp',state)


def setParams(cxn, sample, params, measure=0, state=1):
    sample, qubits, Qubits = util.loadQubits(sample,write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    ml.setMultiKeys(q,state)
    Q['f'+str(state)+str(state-1)] = params[0]
    Q[ml.saveKeyNumber('measureAmp',state)] = params[1]
    Q[ml.saveKeyNumber('piAmp',state)] = params[2]
    hit = cxn.hittite_t2100_server
    hit.select_device('Vince GPIB Bus - GPIB1::20')
    Q['constantPower']=hit.amplitude()['dBm']-51
    Q['constantFreq']=hit.frequency()


def multihittitequbitfreq(cxn, sample, amps, user='Jim', target=0.05, measure=0, maxiter=10, qubititer=10, noisy=False, updateFreq=False):

    hit = cxn.hittite_t2100_server
    hit.select_device('Vince GPIB Bus - GPIB1::20')

    for amp in amps:
        #Tune up case without additional power
        tuneup2(cxn, sample, target=target, measure=measure, maxiter=maxiter, noisy=noisy, updateFreq=False)

        #Apply microwaves from 2nd Hittite
        print amp
        hit.amplitude(amp+46)
        hit.output(True)
        newSession = 'Power'+str(int(10*amp[dBm]))+'QubitFreq'
        sample2,oldSession = pss(cxn,user,session=newSession)
        oldParams = getParams(sample2,measure=measure,state=1)
        qubitFreq = [0*GHz,oldParams[0]]
        hit.frequency(qubitFreq[-1])

        # Set Hittite to qubit frequency. Iterate to ensure that at true qubit frequency.
        count=0
        while (abs(qubitFreq[-1]-qubitFreq[-2])>0.001*GHz) and (count<qubititer):
            print ('Qubit Freq: '+str([freq['GHz'] for freq in qubitFreq]))
            findFreq(cxn,sample2,target=target,measure=measure,maxiter=maxiter,noisy=noisy)
            newParams = getParams(sample2,measure=measure,state=1)
            qubitFreq.append(newParams[0])
            hit.frequency(qubitFreq[-1])
            count = count+1

        #Return to original parameters
        setParams(cxn,sample2,oldParams,measure=measure,state=1)
        mq.s_scanning(sample2,power=-50*dBm,noisy=noisy,measure=measure,update=False)
        mq.phase_arc(sample2,bias=st.r[-.1:1:.005,V],noisy=noisy,measure=measure,update=False)
        singlehittiteswap(cxn,sample2,target=target,measure=measure,maxiter=maxiter,noisy=noisy,updateFreq=updateFreq)

        #Return to no power case
        pss(cxn,user,session=oldSession)
        hit.output(False)


def multihittitequbit02freq(cxn, sample, amps, user='Jim', target=0.05, measure=0, maxiter=10, qubititer=10, noisy=False, updateFreq=False):

    hit = cxn.hittite_t2100_server
    hit.select_device('Vince GPIB Bus - GPIB1::20')

    for amp in amps:
        #Tune up case without additional power
        hit.output(False)
        tuneup2barebones(cxn, sample, target=target, measure=measure, maxiter=maxiter, noisy=noisy, updateFreq=False)

        #Apply microwaves from 2nd Hittite
        print amp
        hit.amplitude(amp+46)
        hit.output(True)
        newSession = 'Power'+str(int(10*amp[dBm]))+'Qubit02Freq'
        sample2,oldSession = pss(cxn,user,session=newSession)
        oldParams = getParams(sample2,measure=measure,state=1)
        oldParams2 = getParams(sample2,measure=measure,state=2)
        qubitFreq = [0*GHz,(oldParams[0]+oldParams2[0])/2]
        hit.frequency(qubitFreq[-1])

        # Set Hittite to qubit frequency. Iterate to ensure that at true qubit frequency.
        count=0
        while (abs(qubitFreq[-1]-qubitFreq[-2])>0.001*GHz) and (count<qubititer):
            print ('Qubit Freq: '+str([freq['GHz'] for freq in qubitFreq]))
            findFreq2(cxn,sample2,target=target,measure=measure,maxiter=maxiter,noisy=noisy,updateFreq=updateFreq)
            newParams = getParams(sample2,measure=measure,state=1)
            newParams2 = getParams(sample2,measure=measure,state=2)
            qubitFreq.append((newParams[0]+newParams2[0])/2)
            hit.frequency(qubitFreq[-1])
            count = count+1

        #Return to original parameters
        setParams(cxn,sample2,oldParams,measure=measure,state=1)
        setParams(cxn,sample2,oldParams2,measure=measure,state=2)
        mq.s_scanning(sample2,power=-50*dBm,noisy=noisy,measure=measure,update=False)
        mq.phase_arc(sample2,bias=st.r[-.1:1:.005,V],noisy=noisy,measure=measure,update=False)
        singlehittitebarebones(cxn,sample2,target=target,measure=measure,maxiter=maxiter,noisy=noisy,updateFreq=updateFreq)

        #Return to no power case
        pss(cxn,user,session=oldSession)


def saveSweepData(cxn,sample,user,squidxname,squidx,t1data,f01data,state2data,state1data):
    userPath = ['', user]
    reg = registry.RegistryWrapper(cxn, userPath)
    samplePath = reg['sample'][:-1]
    dvPath = userPath + samplePath

    dv = cxn.data_vault
    dv.cd(dvPath)
    dv.new('Squid Heating T1',[squidxname],[('T1','', 'ns')])
    dv.add(np.array([squidx,t1data]).transpose())
    dv.new('Squid Heating df01',[squidxname],[('df01','', 'MHz')])
    dv.add(np.array([squidx,1000*f01data]).transpose())
    dv.new('Squid Heating Measure |2>',[squidxname],[('Probability', '|2>', ''),('Probability', '|1>', ''),(' Corrected Probability', '|1>-|2>', '')])
    dv.add(np.array([squidx,state2data,state1data,state1data-state2data]).transpose())

    gamma = 1000./np.array(t1data)
    df = 1000*np.array(f01data)
    dv.new('Squid Heating QP',[('1/T1', '1/us')],[('df01','', 'MHz')])
    dv.add(np.array([gamma,df]).transpose())

def saveFreqs(cxn,sample,user,rawf01,heatedf01):
    userPath = ['', user]
    reg = registry.RegistryWrapper(cxn, userPath)
    samplePath = reg['sample'][:]
    dvPath = userPath + samplePath

    dv = cxn.data_vault
    dv.cd(dvPath)
    dv.new('Squid Heating f01',[('Squid Heating Pulse','')],[('f01','', 'GHz')])
    dv.add(np.array([[0,1],[rawf01,heatedf01]]).transpose())

def readSquidHeat(sample, squidbias, squidduration, squidsettling, measure=0):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]

    # Read squid heating values if not provided
    if squidbias is None: squidbias = q['squidheatBias']
    if squidduration is None: squidduration = q['squidheatDuration']
    if squidsettling is None: squidsettling = q['squidheatSettling']
    return squidbias, squidduration, squidsettling

def writeSquidHeat(sample, squidbias, squidduration, squidsettling, measure=0):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    Q['squidheatBias'] = squidbias
    Q['squidheatDuration'] = squidduration
    Q['squidheatSettling'] = squidsettling

def readMpas(sample, measure=0):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    return q['measureAmp'], q['measureAmp2']


def singlebias(cxn, sampleOld, user='Jim', measure=0, maxiter=10, squidbias=None, squidduration=None, squidsettling=None, noisy=False, swapName='res', findMpa=False):

    # Read squid heating values if not provided
    squidbias, squidduration, squidsettling = readSquidHeat(sampleOld, squidbias, squidduration, squidsettling, measure=measure)

    #Tune up |1) state: MPA, Pi-Pulse, Frequency
#    ts.squidcutoff(sampleOld)
#    freq1Old=[ts.getFreq(sampleOld, measure=measure, state=1)]
#    freq1Old.append(mq.freqtuner(sampleOld,state=1,measure=measure,noisy=noisy))
#    mq.pituner(sampleOld,state=1,measure=measure,noisy=noisy,diff=0.4)
#    freq1Old.append(mq.freqtuner(sampleOld,state=1,measure=measure,noisy=noisy))
#    mq.pituner(sampleOld,state=1,measure=measure,noisy=noisy,diff=0.4)
#    count = 2
#    while (count<maxiter) and ((abs(freq1Old[-1]-freq1Old[-2])>0.0006*GHz) or (abs(freq1Old[-2]-freq1Old[-3])>0.0006*GHz)):
#        # Iteratively optimize Pi-Pulse and Frequency
#        freq1Old.append(mq.freqtuner(sampleOld,state=1,measure=measure,noisy=noisy))
#        mq.pituner(sampleOld,state=1,measure=measure,noisy=noisy,diff=0.4)
#        count = count+1
#    print ('f01: '+str([freq['GHz'] for freq in freq1Old]))
    ts.squidcutoff(sampleOld)

    # change to a new registry folder
    newSession = 'SH'+format(squidbias,'0.3f')+'SD'+format(squidduration,'0.1f')+'SS'+format(squidsettling,'0.1f')
    sample,oldSession = pss(cxn,user,session=newSession)

    # Set squid heating values
    writeSquidHeat(sample, squidbias, squidduration, squidsettling, measure=measure)
    ts.getMixTemp(cxn, sample, measure=measure)
    mq.squidsteps(sample,update=False)

    #Tune up |1) state: MPA, Pi-Pulse, Frequency
    freq1=[ts.getFreq(sample, measure=measure, state=1)]
    freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
    mq.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
    freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
    mq.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
    count = 2
    while (count<maxiter) and ((abs(freq1[-1]-freq1[-2])>0.0006*GHz) or (abs(freq1[-2]-freq1[-3])>0.0006*GHz)):
        # Iteratively optimize Pi-Pulse and Frequency
        freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
        mq.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
        count = count+1
    print ('f01: '+str([freq['GHz'] for freq in freq1]))
    if findMpa:
        mpa1 = mq.find_final_mpa(sample,stats=300,state=1,resolution=0.005,target=target,measure=measure,noisy=noisy)
    else:
        mpa1 = readMpas(sample, measure=measure)[0]

    #Tune up |2) state: MPA, Pi-Pulse, Frequency
    freq2=[ts.getFreq(sample, measure=measure, state=2)]
    freq2.append(mq.freqtuner(sample,state=2,measure=measure,noisy=noisy))
    mq.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
    freq2.append(mq.freqtuner(sample,state=2,measure=measure,noisy=noisy))
    mq.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
    count = 2
    while (count<maxiter) and ((abs(freq2[-1]-freq2[-2])>0.0006*GHz) or (abs(freq2[-2]-freq2[-3])>0.0006*GHz)):
        # Iteratively optimize Pi-Pulse and Frequency
        freq2.append(mq.freqtuner(sample,state=2,measure=measure,noisy=noisy))
        mq.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
        count = count+1
    print ('f12: '+str([freq['GHz'] for freq in freq2]))
    if findMpa:
        mpa2 = mq.find_final_mpa(sample,stats=300,state=2,resolution=0.005,target=target,measure=measure,noisy=noisy)
    else:
        mpa2 = readMpas(sample, measure=measure)[1]

#    # Get swap spectroscopy and calibration functions
#    freqRange = st.r[(freq2[-1]['GHz']-.05):(freq1[-1]['GHz']+.05):.005,GHz]
#    mq.find_zpa_func(sample, stats=120L, freqScan=freqRange, measure=measure, noisy=noisy)
#    zpa1 = mq.freq2zpa(sample,0,freq1[-1])
#    zpa2 = mq.freq2zpa(sample,0,freq2[-1])
#    mq.swapSpectroscopy(sample, swapLen=st.arangePQ(10,200,10,ns), swapAmp=np.arange((min(zpa1,zpa2)-0.01),(max(zpa1,zpa2)+0.01),0.002), measure=measure, noisy=noisy)
#    mq.find_mpa_func(sample, measure=measure, noisy=noisy)
#    mq.find_flux_func(sample, stats=120L, freqScan=freqRange, measure=measure, noisy=noisy)

    # Measure T1, Visibility
    mq.t1(sample,delay=[time*ns for time in np.append(np.arange(10,100,2),np.arange(100,2000,20))],stats=3000L,measure=measure,noisy=noisy, plot=False)
    mq.visibility(sample, states=[1,2], mpa=st.r[max(mpa2-0.15,0):min(mpa1+0.1,2):0.001], calstats=30000, update=False, measure=measure, noisy=noisy)

#    # Tune up swap to TLS
#    swapLen, swapAmp = mq.swapTuner(sample, swapName, noisy=noisy, measure=measure, ampBound=0.01, timeBound=10.0*ns)
#    mq.swapSpectroscopy(sample, state=1, swapLen=st.arangePQ(0.25*swapLen,1.75*swapLen,0.05*swapLen,ns), swapAmp=np.arange(swapAmp-0.01,swapAmp+0.01,0.001), measure=measure, noisy=noisy)

    # Measure P1
#    ts.measure2(sample,swapName,measstate=1,stats=30000L,measure=measure,noisy=noisy)
#    ts.measure2(sample,swapName,measstate=2,stats=30000L,measure=measure,noisy=noisy)
    measp1(sample,measstate=1,stats=30000L,measure=measure,noisy=noisy)
    measp1(sample,measstate=2,stats=30000L,measure=measure,noisy=noisy)

#    saveFreqs(cxn,sample,user,float(freq1[-1]['GHz']),float(freq1Old[-1]['GHz']))
    # Return to original registry folder and tune up f01
    pss(cxn,user,session=oldSession)


def multibias(cxn, sample, user='Jim', measure=0, maxiter=10, squidbias=None, squidduration=None, squidsettling=None, noisy=False, swapName='res', findMpa=False):
    biaslen = np.size(np.array(squidbias))
    durationlen = np.size(np.array(squidduration))
    settlinglen = np.size(np.array(squidsettling))

    if biaslen>1 and durationlen is 1 and settlinglen is 1:
        for bias in squidbias:
            print('\n\n\n')
            print('New bias: '+str(bias))
            print('\n\n\n')
            time.sleep(10)
            singlebias(cxn,sample,squidbias=bias,squidduration=squidduration,squidsettling=squidsettling,user=user,measure=measure,maxiter=maxiter,noisy=noisy,findMpa=findMpa,swapName=swapName)
    if biaslen is 1 and durationlen>1 and settlinglen is 1:
        for duration in squidduration:
            print('\n\n\n')
            print('New pulse duration: '+str(duration))
            print('\n\n\n')
            time.sleep(10)
            singlebias(cxn,sample,squidbias=squidbias,squidduration=duration,squidsettling=squidsettling,user=user,measure=measure,maxiter=maxiter,noisy=noisy,findMpa=findMpa,swapName=swapName)
    if biaslen is 1 and durationlen is 1 and settlinglen>1:
        for settling in squidsettling:
            print('\n\n\n')
            print('New settling time: '+str(settling))
            print('\n\n\n')
            time.sleep(10)
            singlebias(cxn,sample,squidbias=squidbias,squidduration=squidduration,squidsettling=settling,user=user,measure=measure,maxiter=maxiter,noisy=noisy,findMpa=findMpa,swapName=swapName)
    if biaslen is 1 and durationlen is 1 and settlinglen is 1:
        singlebias(cxn,sample,squidbias=bias,squidduration=squidduration,squidsettling=squidsettling,user=user,measure=measure,maxiter=maxiter,noisy=noisy,findMpa=findMpa,swapName=swapName)


def datascan(cxn,sample,squidbias=None,squidduration=None,squidsettling=None):
    for bias in squidbias:
        for duration in squidduration:
            for settling in squidsettling:
                singlebias(cxn,sample,squidbias=bias,squidduration=duration,squidsettling=settling)


def measp1(sample, stats=3000L, measstate=2, pi=st.r[0:1:1], measure=0, save=True, collect=True, noisy=True, name=''):
    """A single pi-pulse on one qubit, with other qubits also operated."""
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    ml.setMultiKeys(q,max([2,measstate]))

    axes = [(pi, '|1>-to-|2> Pi Pulse')]
    kw = {'stats': stats}
    name='Measure |1) with |'+str(measstate)+')'+name
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, pi):
        q.z = env.NOTHING
        startPi = q['piFWHM']
        if pi:
            q.xy = eh.mix(q, eh.piPulse(q, startPi, state=2), state=2)
        q.z += eh.measurePulse(q, startPi+2*q['piFWHM'], state=measstate)
        q['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, noisy=noisy)
    if noisy:
        print data
    if collect:
        return data

def tuneup(cxn, sample, user='Jim', measure=0, iterations=4, name='test', noisy=False, find_center=False, findmpa=False, changesession=True):
#    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
#    q, Q = qubits[measure], Qubits[measure]

    # change to a new registry folder
    if changesession:
        newSession = name
        sample2,oldSession = pss(cxn,user,session=newSession)
    else:
        sample2 = sample
#    newsample = sample2
#    newsample, qubits, Qubits = util.loadQubits(newsample, write_access=True)
#    q, Q = qubits[measure], Qubits[measure]

    # Set squid heating values
    ts.getMixTemp(cxn, sample2, measure=measure)
    mq.squidsteps(sample2,update=False)
    ts.squidcutoff(sample2)

    #Tune up |1) state: MPA, Pi-Pulse, Frequency
    freq01=[]
    piamp1=[]
    for _ in xrange(iterations):
        # Iteratively optimize Pi-Pulse and Frequency
        freq01.append(mq.freqtuner(sample2,state=1,measure=measure,noisy=False))
        piamp1.append(mq.pitunerHD(sample2,state=1,measure=measure,noisy=False,diff=0.5))
    print ('f01: '+str(freq01))
    print ('|1} Pi-Pulse Amplitude: '+str(piamp1))
    if findmpa: mpa1 = mq.find_final_mpa(sample2,stats=300,state=1,resolution=0.005,measure=measure,noisy=noisy,find_center=find_center)
    ts.squidcutoff(sample2)

    #Tune up |2) state: MPA, Pi-Pulse, Frequency
    freq12=[]
    piamp2=[]
    for _ in xrange(iterations):
        # Iteratively optimize Pi-Pulse and Frequency
        freq12.append(mq.freqtuner(sample2,state=2,measure=measure,noisy=False))
        piamp2.append(mq.pitunerHD(sample2,state=2,measure=measure,noisy=False,diff=0.5))
    print ('f12: '+str(freq12))
    print ('|2} Pi-Pulse Amplitude: '+str(piamp2))
    if findmpa: mpa2 = mq.find_final_mpa(sample2,stats=300,state=2,resolution=0.005,measure=measure,noisy=noisy,find_center=find_center)
    ts.squidcutoff(sample2)

    #Tune up |3) state: MPA, Pi-Pulse, Frequency
    freq23=[]
    piamp3=[]
    for _ in xrange(iterations):
        # Iteratively optimize Pi-Pulse and Frequency
        freq23.append(mq.freqtuner(sample2,state=3,measure=measure,noisy=False))
        piamp3.append(mq.pitunerHD(sample2,state=3,measure=measure,noisy=False,diff=0.5))
    print ('f23: '+str(freq12))
    print ('|3} Pi-Pulse Amplitude: '+str(piamp2))
    if findmpa: mpa3 = mq.find_final_mpa(sample2,stats=300,state=3,resolution=0.005,measure=measure,noisy=noisy,find_center=find_center)

    ts.squidcutoff(sample2)

    # Measure T1 and |2) (w/ and w/o Pi{1-2})
    t1data = mq.t1(sample2,delay=st.r[20:2000:20,ns],stats=6000L,measure=measure,noisy=noisy)
    def t1fit(x,p):
        return p[0]+p[1]*np.exp(-x/p[2])
    t1fitresult = leastsq(lambda p: t1fit(t1data[:,0],p)-t1data[:,1], [0,1,500])

    mq.find_flux_func(sample,freqScan=st.r[5.8:5.9:0.005,GHz])

    ts.squidcutoff(sample2)

    mq.swapSpectroscopy(sample2,swapLen=st.arangePQ(0,150,5,ns),swapAmp=np.arange(0,.05,.002))
    mq.swap10tuner(sample2)

    mq.swapSpectroscopy(sample2,swapLen=st.arangePQ(0,100,5,ns),swapAmp=np.arange(.32,.4,.002))
    mq.swap21tuner(sample2,damp=0.15)

    ts.squidcutoff(sample2)

    ts.measure2(sample2,stats=30000L,measure=measure,noisy=noisy,measstate=1)
    state2probs = ts.measure2(sample2,stats=30000L,measure=measure,noisy=noisy,measstate=2)
    ts.measure2(sample2,stats=30000L,measure=measure,noisy=noisy,measstate=3)
    mq.scurve(sample2, mpa=st.r[-.65:-.1:0.001], stats=300, measure=measure, state=3,noisy=noisy, update=True)

    saveFreqs(cxn,sample2,user,float(freq01[-1]),0*float(freq01[-1]))
    # Return to original registry folder and tune up f01
    if changesession:
        pss(cxn,user,session=oldSession)

#    return [t1fitresult[0][2],state2probs[0][1],state2probs[1][1],float(freq01[-1])] # [ T1 , Prob.|2) without Pi12 , Prob.|2) with Pi12 , df01 ]

def singleconstant(cxn, sample, user='Jim', measure=0, iterations=5, squidbias=None, squidchannel=None, squiddac=None, noisy=False, find_center=True):
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    # Read squid heating values if not provided
    if squidbias is None: squidbias = q['constantHeatBias']
    if squidchannel is None: squidchannel = q['constantHeatChannel']
    if squiddac is None: squiddac = q['constantHeatDAC']
    ts.squidcutoff(sample)

    # Tune up base f01
    freq01base=[]
    piamp1base=[]
    for _ in xrange(iterations-1):
        # Iteratively optimize Pi-Pulse and Frequency
        freq01base.append(mq.freqtuner(sample,state=1,measure=measure,noisy=False))
        piamp1base.append(mq.pitunerHD(sample,state=1,measure=measure,noisy=False,diff=0.5))
    print ('f01: '+str(freq01base))
    print ('|1} Pi-Pulse Amplitude: '+str(piamp1base))

    # change to a new registry folder
    newSession = 'CH'+format(squidbias,'0.3f')
    sample2,oldSession = pss(cxn,user,session=newSession)
    newsample = sample2
    newsample, qubits, Qubits = util.loadQubits(newsample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]

    # Set squid heating values
    Q['constantHeatBias'] = squidbias
    Q['constantHeatChannel'] = squidchannel
    Q['constantHeatDAC'] = squiddac
    setbias.set_fb(squiddac,cxn,squidbias,squidchannel)
    time.sleep(15) #Giving system time to equilibrate

    mq.squidsteps(sample2,update=False)
    ts.squidcutoff(sample2)

    #Tune up |1) state: MPA, Pi-Pulse, Frequency
#    if find_center: mq.find_mpa(sample2,stats=300,state=1,resolution=0.005,measure=measure,noisy=noisy)
    freq01=[]
    piamp1=[]
    for _ in xrange(iterations):
        # Iteratively optimize Pi-Pulse and Frequency
        freq01.append(mq.freqtuner(sample2,state=1,measure=measure,noisy=False))
        piamp1.append(mq.pitunerHD(sample2,state=1,measure=measure,noisy=False,diff=0.5))
    print ('f01: '+str(freq01))
    print ('|1} Pi-Pulse Amplitude: '+str(piamp1))
#    mpa1 = mq.find_final_mpa(sample2,stats=300,state=1,resolution=0.005,measure=measure,noisy=noisy,find_center=find_center)
    ts.squidcutoff(sample2)

    #Tune up |2) state: MPA, Pi-Pulse, Frequency
#    if find_center: mq.find_mpa(sample2,stats=300,state=2,resolution=0.005,measure=measure,noisy=noisy)
    freq12=[]
    piamp2=[]
    for _ in xrange(iterations):
        # Iteratively optimize Pi-Pulse and Frequency
        freq12.append(mq.freqtuner(sample2,state=2,measure=measure,noisy=False))
        piamp2.append(mq.pitunerHD(sample2,state=2,measure=measure,noisy=False,diff=0.5))
    print ('f12: '+str(freq12))
    print ('|2} Pi-Pulse Amplitude: '+str(piamp2))
#    mpa2 = mq.find_final_mpa(sample2,stats=300,state=2,resolution=0.005,measure=measure,noisy=noisy,find_center=find_center)
    ts.squidcutoff(sample2)

    #Tune up |3) state: MPA, Pi-Pulse, Frequency
#    if find_center: mq.find_mpa(sample2,stats=300,state=3,resolution=0.005,measure=measure,noisy=noisy)
    freq23=[]
    piamp3=[]
    for _ in xrange(iterations):
        # Iteratively optimize Pi-Pulse and Frequency
        freq23.append(mq.freqtuner(sample2,state=3,measure=measure,noisy=False))
        piamp3.append(mq.pitunerHD(sample2,state=3,measure=measure,noisy=False,diff=0.5))
    print ('f23: '+str(freq12))
    print ('|3} Pi-Pulse Amplitude: '+str(piamp2))
#    mpa3 = mq.find_final_mpa(sample2,stats=300,state=3,resolution=0.005,measure=measure,noisy=noisy,find_center=find_center)

    ts.squidcutoff(sample2)

    # Measure T1 and |2) (w/ and w/o Pi{1-2})
    t1data = mq.t1(sample2,delay=st.r[20:2000:20,ns],stats=6000L,measure=measure,noisy=noisy)
    def t1fit(x,p):
        return p[0]+p[1]*np.exp(-x/p[2])
    t1fitresult = leastsq(lambda p: t1fit(t1data[:,0],p)-t1data[:,1], [0,1,500])

    ts.measure2(sample2,stats=30000L,measure=measure,noisy=noisy,measstate=1)
    state2probs = ts.measure2(sample2,stats=30000L,measure=measure,noisy=noisy,measstate=2)
    ts.measure2(sample2,stats=30000L,measure=measure,noisy=noisy,measstate=3)
    mq.scurve(sample2, mpa=st.r[-1.05:0:0.001], stats=600, measure=measure, state=3,noisy=noisy,update=True)

    saveFreqs(cxn,sample2,user,float(freq01[-1]),float(freq01base[-1]))
    # Return to original registry folder and tune up f01
    pss(cxn,user,session=oldSession)
    setbias.set_fb(squiddac,cxn,0*V,squidchannel) # Resetting constant bias to 0 V
    time.sleep(15) # Giving system time to equilibrate

    return [t1fitresult[0][2],state2probs[0][1],state2probs[1][1],float(freq01[-1])-float(freq01base[-1])]

def multiconstant(cxn, sample, user='Jim', measure=0, iterations=5, squidbias=None, squidchannel=None, squiddac=None, noisy=False):
    biaslen = np.size(np.array(squidbias))

    t1list = np.array([])
    state2list = np.array([])
    state1list = np.array([])
    f01list = np.array([])

    if biaslen>1:
        for bias in squidbias:
            t1val,state2val,state1val,f01val = singleconstant(cxn,sample,squidbias=bias,squidchannel=squidchannel,squiddac=squiddac,user=user,measure=measure,iterations=iterations,noisy=noisy)
            t1list = np.append(t1list,t1val)
            state2list = np.append(state2list,state2val)
            state1list = np.append(state1list,state1val)
            f01list = np.append(f01list,f01val)
            print('\n\n\n')
        saveSweepData(cxn,sample,user=user,squidxname=('Constant Heating Bias','V'),squidx=squidbias,t1data=t1list,f01data=f01list,state2data=state2list,state1data=state1list)
    if biaslen is 1 and durationlen is 1 and settlinglen is 1:
        t1val,state2val,state1val,f01val = singlebias(cxn,sample,squidbias=bias,squidchannel=squidchannel,squiddac=squiddac,user=user,measure=measure,iterations=iterations,noisy=noisy)
        return t1val,state2val,state1val,f01val
