import numpy as np
import matplotlib.pyplot as plt

from pyle.util import sweeptools as st
from pyle.dataking import multiqubitPQ as mq
from pyle.dataking import dephasingSweeps
from pyle.plotting import dstools as ds
from pyle.fitting import fitting
import pyle.dataking.util as util
from pyle.dataking import crosstalk as ct

from pyle.dataking import hadamard
from pyle.dataking.benchmarking import danBench as db

from pyle.dataking import utilMultilevels as ml

from labrad.units import Unit, Value
us, ns, MHz, GHz, V, mV = (Unit(un) for un in ['us','ns','MHz','GHz','V','mV'])

#TODO
#
#Generate sensible report of all data automatically. Currently we can find most
#of the relevant tune up parameters from the registry, but certain things aren't
#recorded, for example the quality of the resonator swaps. This should actually
#be in the characterization section.

LOG_PATH = r'N:\dev\pyle\pyle\dataking\bringuplogs'

####################################
#### AUTOMATED BRINGUP ROUTINES ####
####################################

### GHZ DAC BRINGUP ###

def smsNotify(cxn, msg, username):
    """
    Send as SMS to notify the user of an event

    Returns True if the message is sent successfully, False otherwise.
    """
    server = cxn.get('telecomm_server')
    if server:
        return cxn.telecomm_server.send_sms('automate daily', msg, username)
    else:
        return False

def getBoardGroup(cxn, sample):
    """ Get the board group used by the experiment associated to sample"""
    fpgas = cxn.ghz_fpgas
    boardGroups = fpgas.list_board_groups()
    def getAnyBoard():
        for dev in sample.values():
            try:
                #Look in channels to see if we can find any FPGA board
                return dict(dev['channels'])['uwave'][1][0]
            except (KeyError,TypeError):
                #Don't do anything, just try the next one
                pass
    board = getAnyBoard()
    if board is None:
        return board
    for bg in boardGroups:
        if board in [name[1] for name in fpgas.list_devices(bg)]:
            return bg
    return None

def bringupBoards(cxn, boardGroup):
    ok = True
    resultWords = {True:'ok',False:'failed'}
    fpgas = cxn.ghz_fpgas
    try:
        successDict = GHz_DAC_bringup.bringupBoardGroup(fpgas, boardGroup)
        for board, successes in successDict.items():
            for item,success in successes.items():
                if not success:
                    print 'board %s %s failed'%(board,item)
                    ok = False
    except Exception:
        ok = False
    return ok

def doBoardBringup(s):
    cxn = s._cxn
    username = s._dir[1]
    sample, qubits = util.loadQubits(s)
    boardGroup = getBoardGroup(cxn, sample)
    #Bring up FPGA boards. If it fails, send SMS to the user and end daily bringup
    if not bringupBoards(cxn, boardGroup):
        smsNotify(cxn, 'board bringup failed', username)
        return False


### QUBIT BRINGUP ###

def initializeSample(s, measure):
    mq.testdelay(s, measure=measure, plot=True)
    mq.pulseShape(s, measure=measure)

def bringupSingle(s, measure, auto=True, dFluxFreq=None, dZpaFreq=None, dSwapFreq=None, dSwapFreq2=None):
    """One qubit bringup"""
    #Set default values
    if dFluxFreq is None:
        dFluxFreq = (400*MHz, 400*MHz)
    if dZpaFreq is None:
        dZpaFreq = (500*MHz, 500*MHz)
    if dSwapFreq is None:
        dSwapFreq  = (50*MHz, 50*MHz)
    if dSwapFreq2 is None:
        dSwapFreq2 = (50*MHz, 50*MHz)
    #Load qubit information from registry
    sample, qubits, Qubits = util.loadQubits(s, write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]
    #BEGIN BRINGUP ROUTINES
    #Bringup single qubit pi and measure pulses
    bringupPulses(s, measure, auto=auto)
    #Bringup calibration curves for single qubit
    bringupCals(s, measure, dFluxFreq, dZpaFreq, dSwapFreq, dSwapFreq2)
    #Measure T1, T2
    coherenceFactors(s, measure)

def bringupSingleWithBus(s, measure, busParamName, auto=True, dFluxFreq=None, dZpaFreq=None, dSwapFreq=None, dSwapFreq2=None):
    """One qubit bringup"""
    #Set default values
    if dFluxFreq is None:
        dFluxFreq = (400*MHz, 400*MHz)
    if dZpaFreq is None:
        dZpaFreq = (500*MHz, 500*MHz)
    if dSwapFreq is None:
        dSwapFreq  = (50*MHz, 50*MHz)
    if dSwapFreq2 is None:
        dSwapFreq2 = (50*MHz, 50*MHz)
    #Load qubit information from registry
    sample, qubits, Qubits = util.loadQubits(s, write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]
    #BEGIN BRINGUP ROUTINES
    #Bringup single qubit pi and measure pulses
    bringupPulses(s, measure, auto=auto)
    #Bringup calibration curves for single qubit
    bringupCals(s, measure, dFluxFreq, dZpaFreq, dSwapFreq, dSwapFreq2)
    #Measure T1, T2
    coherenceFactors(s, measure)
    #Tune up swaps to bus
    resonatorSwap(s, measure, busParamName)

def bringupMulti(s, measure, timingMaster, paramName):
    sample, qubits = util.loadQubits(s)
    #Tune inter-qubit timing
    for qubitUpdate in measure:
        if not qubitUpdate == timingMaster:
            mq.testdelayBusSwapTuner(s, qubitUpdate, timingMaster, paramName)
    ct.zpulseCrosstalk(s)

def bringupPulses(s, measure, auto=True):
    """Bring up 2 state assuming you already have a reasonable operating point

    TODO:
    Detect large changed in qubit parameters and react accordingly
    Use more than 1 pulse in rabihigh to amplify errors
    """
    sample,qubits,Qubits = util.loadQubits(s, write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]
    #Tune 0->1 transition
    tunePiPulse(s, measure, 1)
    mpas1 = tuneVisibility(s, measure, 1, auto=auto)
    #Get rough |2> mpa
    mq.find_mpa(s, stats=60, target=0.05, mpa_range=(0,2.0), state=2,
                measure=measure, resolution=0.005, blowup=0.05,
                falling=None, statsinc=1.25, save=False,
                name='SCurve Search', collect=False, update=True, noisy=True)
    #Tune 1->2 transition
    tunePiPulse(s, measure, 2)
    mpas2 = tuneVisibility(s, measure, 2, auto=auto)
    if auto:
        mq.pituner(s, measure=measure, state=1,numPulses=3,save=True)
        mq.pituner(s, measure=measure, state=1,numPulses=3,save=True)
        mq.pituner(s, measure=measure, state=2,numPulses=5,save=True)
        mq.pituner(s, measure=measure, state=2,numPulses=5,save=True)
    else:
        mq.rabihigh(s, measure=measure, state=1)
        mq.rabihigh(s, measure=measure, state=2)
    #Get mpa and tunelling rates now that pulses are tuned
    for whichScurve, mpaRange in zip([1,2],[mpas1,mpas2]):
        mq.maxVisibility(s, measure, whichScurve, mpaRange)
    mq.calscurve(s, measure=measure, state=2)
    #Tune up z pulses
    if not auto:
        tuneZPulse(s, measure)


###################################
#### TUNE UPS FOR SINGLE QUBIT ####
###################################

def tunePiPulse(s, measure, state, auto=True):
    """Tune up pi amplitude and frequency"""
    for save in [False,True]:
        if auto:
            mq.pituner(s, measure=measure, numPulses=1, save=save, state=state)
        else:
            mq.rabihigh(s, measure=measure, save=save, state=state)
        mq.freqtuner(s, measure=measure, save=save, state=state)

def tuneZPulse(s, measure):
    mq.ramseyZPulse(s, measure=measure, zAmp=np.linspace(0,0.1,40), stats=1500L, update=True)
    mq.ramseyZPulse(s, measure=measure, numPulses=5, stats=3000, update=True)

def tuneVisibility(s, measure, whichMpa, auto=False):
    """Maximize measurement visibility using a reasonable starting point"""
    sample, qubits = util.loadQubits(s)
    q = qubits[measure]
    ml.setMultiKeys(q, whichMpa)
    mpa = ml.getMultiLevels(q,'measureAmp',whichMpa)
    #Get a visibility scan
    minMpa, maxMpa = 0.90*mpa, 1.10*mpa
    mq.scurve(s, measure=measure, visibility=True, states=[whichMpa-1,whichMpa],
              mpa=np.linspace(minMpa,maxMpa,100), stats=900, update=False)
    if not auto:
        minMpa = float(raw_input('minMpa: '))
        maxMpa = float(raw_input('maxMpa: '))
        mq.maxVisibility(s, measure, whichMpa, mpaRange = (minMpa, maxMpa))
    else:
        mq.maxVisibility(s, measure, whichMpa, mpaFactor=0.05)
    return (minMpa, maxMpa)

def bringupCals(s, measure, dFluxFreq, dZpaFreq, dSwapFreq, dSwapFreq2):
    """
    Calibration curves: find_mpa_func, find_flux_func, find_DfDv, find_zpa_func, swapSpectroscopy

    dFluxFreq =
    dZpaFreq =
    dSwapFreq: tuple of (minFrequency, maxFrequency)
    """
    sample, qubits, Qubits = util.loadQubits(s, write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]
    #Do the cal scans
    mq.find_mpa_func(s, measure=measure, plot=True)
    mq.find_flux_func(s, measure=measure, plot=True, freqScan=(q['f10']-dFluxFreq[0], q['f10']+dFluxFreq[1]))
    mq.findDfDv(s, measure, plot=True)
    mq.find_zpa_func(s, measure=measure, plot=True, freqScan=(q['f10']-dZpaFreq[0], q['f10']+dZpaFreq[1]))
    #Find zpa for reasonable frequency range and do swapSpectroscopy on both transitions
    upperZpa = mq.freq2zpa(s, measure, q['f10']+dSwapFreq[1])
    lowerZpa = mq.freq2zpa(s, measure, q['f10']-dSwapFreq[0])
    zpaStep = 0.002 #This is roughly 2MHz per zpa step
    #To actually calculate a zpa step, do this:
    #zpaStep = df['GHz'] * 4 * Q['calZpaFunc'][0] * Q['f10']['GHz']**3
    #This comes from looking at how calZpaFunc is defined. See multiqubit.find_zpa_func
    mq.swapSpectroscopy(s, measure=measure,
                        swapAmp = np.arange(min(upperZpa,lowerZpa), max(upperZpa, lowerZpa), zpaStep),
                        swapLen = st.r[0:300:5,ns],stats=300L)
    #Compute zpa range for 1->2 swapSpectroscopy
    zpaOffset = mq.freq2zpa(s, measure, q['f21'])
    upperZpa = mq.freq2zpa(s, measure, q['f21']+dSwapFreq2[1]) - zpaOffset
    lowerZpa = mq.freq2zpa(s, measure, q['f21']-dSwapFreq2[0]) - zpaOffset
    zpaStep = 0.002 #This is roughly 2MHz per zpa step
    mq.swapSpectroscopy(s, measure=measure, state=2,
                        swapAmp = np.arange(min(upperZpa,lowerZpa), max(upperZpa, lowerZpa), 0.002),
                        swapLen = st.r[0:300:5,ns],stats=300L)

def resonatorSwap(s, measure, busParamName):
    #Tune up |1> swap
    swapTime1, swapAmp1 = mq.swapTuner(s, busParamName, measure=measure, iterations=3, save=True,
                                      ampBound=0.025,timeBound=10.0*ns, state=1, noisy=True)
    #Tune up |2> swap
    swapTime2, swapAmp2 = mq.swapTuner(s, busParamName, measure=measure, iterations=3, save=True,
                                      ampBound=0.025,timeBound=10.0*ns, state=2, noisy=True)
    #Get qubit population after swap to make sure the swaps are good.
    swapProb1 = mq.swapSpectroscopy(s, measure=measure, swapLen=[swapTime1],
                                    swapAmp=swapAmp1, state=1,
                                    collect=True, stats=12000)
    swapProb2 = mq.swapSpectroscopy(s, measure=measure, swapLen=[swapTime2],
                                    swapAmp=swapAmp2, state=2,
                                    collect=True, stats=12000)
    swapProb1 = mq.swapSpectroscopy(s, measure=measure, swapLen=st.r[0:2*swapTime1:1,ns],
                                    swapAmp=np.arange(swapAmp1-0.04, swapAmp1+0.04, 0.002),
                                    state=1, collect=True, stats=300L)
    swapProb1 = mq.swapSpectroscopy(s, measure=measure, swapLen=st.r[0:2*swapTime2:1,ns],
                                    swapAmp=np.arange(swapAmp2-0.04, swapAmp2+0.04, 0.002),
                                    state=2, collect=True, stats=300L)

def frequency(s, measure):
    sample, qubits, Qubits = util.loadQubits(s, write_access=True)
    q = qubits[measure]
    Q = Qubits[measure]
    fraction = 0.5
    detunings = st.r[-2:2:0.2, MHz]
    db.pulseTrajectory(s, measure, fraction, detuning = detunings,
                    stats=3000L, tBuf=40*ns, tBufMeasure = 5.0*ns)
    #Find best detuning from dataset
    dataset = db.loadLastDataset(s)
    #Get angle errors in CYCLES
    theta,phi,length = db.analyzePulseTrajectory(dataset, 'x').T
    bestIdx = np.argmin(np.abs(phi))
    detuning = detunings[bestIdx]
    Q['f10'] = q['f10'] + detuning
    print 'Qubit frequency adjusted by %s' % detuning


##############################################
#### TUNE UP ONE THING ON MULTIPLE QUBITS ####
##############################################

def freqtunerAll(s, measures, states):
    for measure in measures:
        for state in states:
            mq.freqtuner(s, measure=measure, state=state)

def tunePiPulseAll(s, measures, states):
    for measure in measures:
        for state in states:
            tunePiPulse(s, measure, state)

def resonatorSwapAll(s, measures):
    """Tune |1> and |2> swap to resonator for several qubits"""
    for measure in measures:
        mq.swapTuner(s, busParamName, measure=measure, iterations=3,
                     ampBound=0.025,timeBound=10.0*ns, state=1, noisy=True)
        mq.swapTuner(s, busParamName, measure=measure, iterations=3,
                     ampBound=0.025,timeBound=10.0*ns, state=2, noisy=True)


##########################
#### CHARACTERIZATION ####
##########################

def coherenceFactors(s, measure):
    mq.t1(s, measure=measure, stats=3000, delay=st.r[-10:2000:20,ns],
          save=True, update=True, plot=True)
    dephasingSweeps.ramsey(s, measure=measure, delay=st.r[0:450:5, ns],
                           stats=1800L, name='Ramsey', plot=True, update=True)

