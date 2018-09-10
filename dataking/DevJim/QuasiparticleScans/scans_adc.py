import numpy as np
from scipy.optimize import leastsq

from labrad.units import Unit
mK, V, mV, us, ns, GHz, MHz = [Unit(s) for s in ('mK', 'V', 'mV', 'us', 'ns', 'GHz', 'MHz')]

from pyle.util import sweeptools as st
from pyle.dataking import multiqubitPQ as mq
from pyle.workflow import switchSession as pss
from pyle import registry
from pyle.dataking import util
from pyle.dataking import dephasingSweeps
import labrad
import time
from pyle.dataking.quasiparticle import setbias
from pyle.dataking.quasiparticle import twostate as ts

def saveFreqs(cxn,sample,user,rawf01,heatedf01):
    userPath = ['', user]
    reg = registry.RegistryWrapper(cxn, userPath)
    samplePath = reg['sample'][:]
    dvPath = userPath + samplePath

    dv = cxn.data_vault
    dv.cd(dvPath)
    dv.new('Squid Heating f01',[('Squid Heating Pulse','')],[('f01','', 'GHz')])
    dv.add(np.array([[0,1],[rawf01,heatedf01]]).transpose())


def getFreq(sample, measure=0, state=1):
    sample, qubits = util.loadQubits(sample)
    q = qubits[measure]
    fkey = 'f'+str(state)+str(state-1)
    return q[fkey]


def tuneup(cxn, sample, target=0.05, measure=0, maxiter=10, noisy=False):
#    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
#    q, Q = qubits[measure], Qubits[measure]

    # Set squid heating values
    ts.getMixTemp(cxn, sample, measure=measure)
    mq.spectroscopy_two_state(sample, measure=measure, noisy=noisy)

    #Tune up |1) state: MPA, Pi-Pulse, Frequency
    mq.find_mpa(sample,stats=300,state=1,resolution=0.005,target=target,measure=measure,noisy=noisy)
    freq1=[getFreq(sample, measure=measure, state=1)]
    freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
    mq.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
    freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
    mq.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
    count = 2
    while (count<maxiter) and (abs(freq1[-1]-freq1[-2])>0.0006*GHz) and (abs(freq1[-2]-freq1[-3])>0.0006*GHz):
        # Iteratively optimize Pi-Pulse and Frequency
        print ('f01: '+str([freq['GHz'] for freq in freq1]))
        freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
        mq.pituner(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
        count = count+1
    print ('f01: '+str([freq['GHz'] for freq in freq1]))
    mpa1 = mq.find_final_mpa(sample,stats=300,state=1,resolution=0.005,target=target,measure=measure,noisy=noisy)

    #Tune up |2) state: MPA, Pi-Pulse, Frequency
    mq.find_mpa(sample,stats=300,state=2,resolution=0.005,target=target,measure=measure,noisy=noisy)
    freq2=[getFreq(sample, measure=measure, state=2)]
    freq2.append(mq.freqtuner(sample,state=2,measure=measure,noisy=noisy))
    mq.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
    freq2.append(mq.freqtuner(sample,state=2,measure=measure,noisy=noisy))
    mq.pituner(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
    count = 2
    while (count<maxiter) and (abs(freq2[-1]-freq2[-2])>0.0006*GHz) and (abs(freq2[-2]-freq2[-3])>0.0006*GHz):
        # Iteratively optimize Pi-Pulse and Frequency
        print ('f12: '+str([freq['GHz'] for freq in freq2]))
        freq21.append(mq.freqtuner(sample,state=2,measure=measure,noisy=noisy))
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
    mq.t1(sample,delay=st.r[20:3000:20,ns],stats=3000L,measure=measure,noisy=noisy)
    dephasingSweeps.ramsey(sample, delay=st.r[0:300:1, ns], plot=False, update=False, measure=measure, noisy=noisy)
    mq.scurve2(sample, state=[0,1,2], mpa=st.r[max(mpa2-0.15,0):min(mpa1+0.1,2):0.001], visibility=True, update=False, measure=measure, noisy=noisy)
    mq.scurve2(sample, state=[0,1,2], calstate=[1,2], stats=12000, measure=measure, noisy=noisy)

    # Tune up swap to TLS
    swapLen, swapAmp = mq.swap10tuner(sample, noisy=noisy, measure=measure)
    mq.swapSpectroscopy(sample, state=1, swapLen=st.arangePQ(0,2.5*swapLen,2,ns), swapAmp=np.arange(swapAmp-0.01,swapAmp+0.01,0.0005), measure=measure, noisy=noisy)
    swapLen, swapAmp = mq.swap10tuner(sample, stats=1800L, noisy=noisy, measure=measure)

    # Measure P1
    ts.measure2(sample,measstate=1,stats=18000L,measure=measure,noisy=noisy)
    ts.measure2(sample,measstate=2,stats=18000L,measure=measure,noisy=noisy)


def tuneup_old(cxn, sample, target=0.05, measure=0, maxiter=10, noisy=False):
#    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
#    q, Q = qubits[measure], Qubits[measure]

    # Set squid heating values
    ts.getMixTemp(cxn, sample, measure=measure)
    mq.spectroscopy_two_state(sample, measure=measure, noisy=noisy)

    #Tune up |1) state: MPA, Pi-Pulse, Frequency
    mq.find_mpa(sample,stats=300,state=1,resolution=0.005,target=target,measure=measure,noisy=noisy)
    freq1=[getFreq(sample, measure=measure, state=1)]
    freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
    mq.pitunerHD(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
    freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
    mq.pitunerHD(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
    count = 2
    while (count<maxiter) and (abs(freq1[-1]-freq1[-2])>0.0006*GHz) and (abs(freq1[-2]-freq1[-3])>0.0006*GHz):
        # Iteratively optimize Pi-Pulse and Frequency
        print ('f01: '+str([freq['GHz'] for freq in freq1]))
        freq1.append(mq.freqtuner(sample,state=1,measure=measure,noisy=noisy))
        mq.pitunerHD(sample,state=1,measure=measure,noisy=noisy,diff=0.4)
        count = count+1
    print ('f01: '+str([freq['GHz'] for freq in freq1]))
    mpa1 = mq.find_final_mpa(sample,stats=300,state=1,resolution=0.005,target=target,measure=measure,noisy=noisy)

    #Tune up |2) state: MPA, Pi-Pulse, Frequency
    mq.find_mpa(sample,stats=300,state=2,resolution=0.005,target=target,measure=measure,noisy=noisy)
    freq2=[getFreq(sample, measure=measure, state=2)]
    freq2.append(mq.freqtuner(sample,state=2,measure=measure,noisy=noisy))
    mq.pitunerHD(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
    freq2.append(mq.freqtuner(sample,state=2,measure=measure,noisy=noisy))
    mq.pitunerHD(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
    count = 2
    while (count<maxiter) and (abs(freq2[-1]-freq2[-2])>0.0006*GHz) and (abs(freq2[-2]-freq2[-3])>0.0006*GHz):
        # Iteratively optimize Pi-Pulse and Frequency
        print ('f12: '+str([freq['GHz'] for freq in freq2]))
        freq21.append(mq.freqtuner(sample,state=2,measure=measure,noisy=noisy))
        mq.pitunerHD(sample,state=2,measure=measure,noisy=noisy,diff=0.4)
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
    mq.t1(sample,delay=st.r[20:3000:20,ns],stats=3000L,measure=measure,noisy=noisy)
    dephasingSweeps_adc.ramsey(sample, delay=st.r[0:300:1,ns], plot=False, update=False, measure=measure, noisy=noisy)
    mq.scurve2(sample, state=[0,1,2], mpa=st.r[max(mpa2-0.15,0):min(mpa1+0.1,2):0.001], visibility=True, update=False, measure=measure, noisy=noisy)
    mq.scurve2(sample, state=[0,1,2], calstate=[1,2], stats=12000, measure=measure, noisy=noisy)

    # Tune up swap to TLS
    swapLen, swapAmp = mq.swap10tuner(sample, noisy=noisy, measure=measure)
    mq.swapSpectroscopy(sample, state=1, swapLen=st.arangePQ(0,2.5*swapLen,2,ns), swapAmp=np.arange(swapAmp-0.01,swapAmp+0.01,0.0005), measure=measure, noisy=noisy)
    swapLen, swapAmp = mq.swap10tuner(sample, stats=1800L, noisy=noisy, measure=measure)

    # Measure P1
    ts.measure2(sample,measstate=1,stats=18000L,measure=measure,noisy=noisy)
    ts.measure2(sample,measstate=2,stats=18000L,measure=measure,noisy=noisy)
