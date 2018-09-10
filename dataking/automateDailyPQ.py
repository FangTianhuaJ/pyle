from datetime import datetime
import itertools

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import leastsq, fsolve
import time
import random

import labrad

from labrad.units import Unit
ns, us, GHz, MHz = [Unit(s) for s in ('ns', 'us', 'GHz', 'MHz')]
from scripts import GHz_DAC_bringup


from pyle.dataking import measurement
from pyle.dataking import multiqubitPQ as mq
from pyle.dataking import util as util
from pyle.util import sweeptools as st

from pyle.dataking import noonPQ

from pyle.dataking import hadamard as hadi

from pyle.plotting import dstools as ds
from pyle import tomo

# TODO: Big dream here. Create a "report" or a dictionary of all the calibrations for each qubit
# then print that to the screen/save it to registry/ save fits to dataset
# TODO add fits to e.g. T1 and T2
# TODO separate datataking from analysis (so can rerun analysis on datasets directly from data vault)
# TODO save all fitting params with datasets
# TODO retune measure pulse amplitude for maximum visibility


def smsNotify(cxn, msg, username):
    return cxn.telecomm_server.send_sms('automate daily', msg, username)

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

def getReadoutType(sample, device):
    pass

def daily_bringup(s, pause=False):
    cxn = s._cxn
    username = s._dir[1]
    sample, qubits = util.loadQubits(s)
    boardGroup = getBoardGroup(cxn, sample)

    #Bring up FPGA boards. If it fails, send SMS to the user and end daily bringup
    if not bringupBoards(cxn, boardGroup):
        smsNotify(cxn, 'board bringup failed', username)
        return False
    #Set up DC bias (SQUID steps, resonator readout, etc.)
    bringup_dcBias(s, pause=pause)
    return
    bringup_stepedge(s, pause=pause)    #run stepedge and find stepedge for each qubit
    bringup_scurve(s, pause=pause)      #measure scurve over reasonable range. find_mpa_func
    bringup_sample(s, pause=pause)      #spectroscopy, tune pi freq/amp, visibility
                                        #fluxFunc/zpaFunc on reasonable frequency ranges
    #todo the pulshape tune-up
    single_qubit_scans(s)               #Coherence factors
    qubit_coupling_resonator_scans(s)   #For each qubit: swapTuner,fockTuner. Then inter-qubit timing.
    #qubit_memory_resonator_scans(s)
    gate_bringup(s)
    create_bell_state_iswap(s,zSweep=False)

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

def bringup_dcBias(s, pause=True):
    pass

def bringup_squidsteps(s, pause=True):
    N = len(s['config'])
    for i in range(N):
        print 'measuring squidsteps for qubit %d...' % i,
        mq.squidsteps(s, measure=i, noisy=False, update=pause)
        print 'done.'


def bringup_stepedge(s, pause=True):
    N = len(s['config'])
    for i in range(N):
        print 'measuring step edge, qubit %d...' % i,
        mq.stepedge(s, measure=i, noisy=False, update=pause)
        print 'done.'

    for i in range(N):
        print 'binary searching to find step edge %d...' % i
        mq.find_step_edge(s, measure=i, noisy=False)
        print 'done.'


def bringup_scurve(s, pause=True):
    N = len(s['config'])
    for i in range(N):
        print 'measuring scurve, qubit %d...' % i
        mpa05 = mq.find_mpa(s, measure=i, target=0.05, noisy=False, update=False)
        print '5% tunneling at mpa =', mpa05
        mpa95 = mq.find_mpa(s, measure=i, target=0.95, noisy=False, update=False)
        print '95% tunneling at mpa =', mpa95
        low = st.nearest(mpa05 - (mpa95 - mpa05) * 1.0, 0.002)
        high = st.nearest(mpa95 + (mpa95 - mpa05) * 1.0, 0.002)
        step = 0.002 * np.sign(high - low)
        mpa_range = st.r[low:high:step]
        mq.scurve(s, mpa_range, measure=i, stats=1200, noisy=False, update=pause)
        print 'done.'

    for i in range(N):
        print 'binary searching to find mpa %d...' % i
        mq.find_mpa(s, measure=i, noisy=False, update=True)
        mq.find_mpa_func(s, measure=i, noisy=False, update=True)
        print 'done.'


def bringup_spectroscopy(s, freq_range=(6.0*GHz, 6.8*GHz)):
    qubits = s['config']
    N = len(qubits)
    for i in range(N):
        mq.spectroscopy(s, st.r[freq_range[0]:freq_range[1]:0.005*GHz], measure=i, update=True)


def bringup_sample(s, pause=False, fine_tune=True):
    N = len(s['config'])

    bringup_pi_pulses(s, pause=pause)

    if fine_tune:
        for i in range(N):
            # choose frequency range to cover all qubits
            fmin = min(s[qubit]['f10'] for qubit in s['config']) - 0.1*GHz
            fmax = max(s[qubit]['f10'] for qubit in s['config']) + 0.1*GHz

            print 'measuring flux func, qubit %d...' % i,
            mq.find_flux_func(s, (fmin, fmax), measure=i, noisy=False)
            print 'done.'

            print 'measuring zpa func, qubit %d...' % i,
            mq.find_zpa_func(s, (fmin, fmax), measure=i, noisy=False)
            print 'done.'

            # update the calibrated ratio of DAC amplitudes to detuning and rabi freqs
            update_cal_ratios(s)


def update_cal_ratios(s):
    s, _qubits, Qubits = util.loadQubits(s, write_access=True)

    # single-qubit bringup
    for Q in Qubits:
        # convert microwave amplitude to rabi frequency
        fwhm = Q['piFWHM'][ns]
        A = float(Q['piAmp'])
        Q['calRabiOverUwa'] = 2*np.sqrt(np.log(2)/np.pi)/(A*fwhm)*GHz # total area is 1 cycle

        # convert z amplitude to detuning frequency
        a = float(Q['calZpaFunc'][0])
        f = Q['f10'][GHz]
        Q['calDfOverZpa'] = 1/(4*a*f**3)*GHz


def bringup_pi_pulses(s, pause=False):
    N = len(s['config'])

    for i in range(N):
        print 'measuring spectroscopy, qubit %d...' % i,
        mq.spectroscopy(s, measure=i, noisy=False, update=pause) # zoom in on resonance peak
        mq.spectroscopy_two_state(s, measure=i, noisy=False, update=pause)
        print 'done.'

    for i in range(N):
        print 'calibrating pi pulse, qubit %d...' % i,
        mq.pitunerHD(s, measure=i, noisy=False)
        print 'done.'

        print 'fine-tuning frequency, qubit %d...' % i,
        mq.freqtuner(s, iterations=1, measure=i, save=True)
        print 'done.'

        print 'redoing pi pulse calibration, qubit %d...' % i,
        mq.pitunerHD(s, measure=i, noisy=False)
        print 'done.'

        print 'checking visibility, qubit %d...' % i
        mpa1_05 = mq.find_mpa(s, measure=i, pi_pulse=True, target=0.05, noisy=False, update=False)
        print '5% tunneling of 1 at mpa =', mpa1_05
        mpa0_95 = mq.find_mpa(s, measure=i, pi_pulse=False, target=0.95, noisy=False, update=False)
        print '95% tunneling of 0 at mpa =', mpa0_95
        low = max(st.nearest(mpa1_05 - (mpa0_95 - mpa1_05) * 0.5, 0.002), 0)
        high = min(st.nearest(mpa0_95 + (mpa0_95 - mpa1_05) * 0.5, 0.002), 2)
        step = 0.002 * np.sign(high - low)
        mpa_range = st.r[low:high:step]
        mq.visibility(s, mpa_range, stats=1200, measure=i, noisy=False)
        print 'done.'

        # TODO adjust measurePulse_amplitude for maximum visibility

        # measure e0, e1 and visibility very carefully at the correct measure-pulse amplitude
        print 'measuring visibility at calibrated mpa %d...' % i,
        Q = s[s['config'][i]]
        data = mq.visibility(s, [Q['measureAmp']]*100, stats=600, measure=i, noisy=False, name='Measurement Fidelity', collect=True)
        e0, f1 = np.mean(data[:,1]), np.mean(data[:,2])
        print 'done.'
        print '  e0: %g, f0: %g' % (e0, 1-e0)
        print '  e1: %g, f1: %g' % (1-f1, f1)
        Q['measureE0'] = e0
        Q['measureF0'] = 1-e0
        Q['measureE1'] = 1-f1
        Q['measureF1'] = f1


def bringup_timing(s):
    N = len(s['config'])
    for i in range(N):
        print 'measuring timing delay on qubit %d...' % i
        mq.testdelay(s, measure=i, update=True, plot=True, noisy=False)
        print 'done.'
    for i in range(1,N):
        print 'measuring timing delay between qubit 0 and %d...' % i
        mq.testdelay_x(s, measure=0, z_pulse=i, update=True, plot=True, noisy=False)
        print 'done.'


def bringup_xtalk(s):
    """Measure the z-pulse crosstalk between each pair of qubits.

    We then create the crosstalk matrix and store it in the registry.
    In addition, we invert the crosstalk matrix, since this is needed
    to correct the z-pulse signals if desired.

    Assumes you have already run spectroscopy2DZauto, so that the
    cal_zpa_func has already been set, as xtalk is relative to this.
    """
    s, qubits, Qubits = util.loadQubits(s, write_access=True)
    A = np.eye(len(qubits))
    for i, qi in enumerate(qubits):
        for j, _qj in enumerate(qubits):
            if i == j:
                continue
            print 'measuring crosstalk on %s from z-pulse on %s' % (i, j)
            xtfunc = mq.spectroscopy2DZxtalk(s, measure=i, z_pulse=j, noisy=False)
            aii = float(qi['calZpaFunc'][0])
            aij = float(xtfunc[0])
            print 'crosstalk =', aii/aij
            A[i,j] = aii/aij
    Ainv = np.linalg.inv(A)
    print
    print 'xtalk matrix:\n', A
    print
    print 'inverse xtalk matrix:\n', Ainv
    for i, Qi in enumerate(Qubits):
        Qi['calZpaXtalk'] = A[i]
        Qi['calZpaXtalkInv'] = Ainv[i]


def test_xtalk(s):
    s, qubits = util.loadQubits(s, write_access=False)

    readouts = [(0,1,2,3), (0,1,3,2), (3,2,1,0), (1,2,0,3), (2,0,1,3), (2,1,3,0)]
    for readout_order in readouts:
        for q, order in zip(qubits, readout_order):
            q['squidReadoutDelay'] = (order+1) * 10*us
        for i in range(len(qubits)):
            mq.meas_xtalk(s, name='meas-xtalk simult readout order %s' % (readout_order,), drive=i, simult=True, stats=1200, noisy=False)

    for readout_order in readouts:
        for q, order in zip(qubits, readout_order):
            q['squidReadoutDelay'] = (order+1) * 10*us
        for i in range(len(qubits)):
            mq.meas_xtalk(s, name='meas-xtalk readout order %s' % (readout_order,), drive=i, simult=False, stats=1200, noisy=False)


#def test_measurements(s):
#    """Test the various ways of measuring qubits."""
#    s, qubits = util.loadQubits(s)
#    q0, _q1, q2 = qubits
#
#    zpa0 = q0['wZpulseAmp']
#    zpa2 = q2['wZpulseAmp']
#
#    # couple all three qubits together
#    kw = dict(pi_pulse_on=1, t_couple=32*ns, name='w-state meas_test', delay=st.r[30:36:1,ns], zpas=[zpa0, 0, zpa2], stats=3000)
#
#    werner.w_state(s, measure=0, **kw)
#    werner.w_state(s, measure=[1], **kw)
#    werner.w_state(s, measure=[0,1,2], **kw)
#    werner.w_state(s, measure=measurement.Null(3, [0,1]), **kw)
#    werner.w_state(s, measure=measurement.Null(3, [0,1,2]), **kw)
#    werner.w_state(s, measure=measurement.Tomo(3, [0]), **kw)
#    werner.w_state(s, measure=measurement.Tomo(3, [0,1]), **kw)
#    werner.w_state(s, measure=measurement.Tomo(3, [0,1,2]), **kw)
#    werner.w_state(s, measure=measurement.TomoNull(3, [0]), pipesize=2, **kw)
#    werner.w_state(s, measure=measurement.TomoNull(3, [0,1]), pipesize=2, **kw)
#    werner.w_state(s, measure=measurement.TomoNull(3, [0,1,2]), pipesize=2, **kw)
#    werner.w_state(s, measure=measurement.Octomo(3, [0]), pipesize=2, **kw)
#    werner.w_state(s, measure=measurement.Octomo(3, [0,1]), pipesize=2, **kw)
#    werner.w_state(s, measure=measurement.Octomo(3, [0,1,2]), pipesize=2, **kw)
#    werner.w_state(s, measure=measurement.OctomoNull(3, [0,1,2]), pipesize=2, **kw)



def single_qubit_scans(s):
    N = len(s['config'])
    for i in range(N):
        print 'measuring T1, qubit %d' % i,
        mq.t1(s, stats=1800, measure=i, noisy=False)
        #TODO add T1 fits
        print 'done.'

        print 'measuring ramsey fringe, qubit %d' % i,
        #TODO bring T1 fit from above and turn on T2 fit
        mq.ramsey(s, stats=1800, measure=i, noisy=False)
        print 'done.'

        print 'measuring spin_echo, qubit %d' % i,
        mq.spinEcho(s, stats=1800, measure=i, noisy=False)
        print 'done.'

def qubit_coupling_resonator_scans(s):
    start = datetime.now()

    N = len(s['config'])
    for i in range(N):
        print 'measuring SWAP10 Spectroscopy, qubit %d' % i,
        swap10Len, swap10Amp = mq.swap10tuner(s, measure=i, stats=1800, noisy=False, whichRes='Coupler')
        print 'measuring 2D-SWAP Spec around Coupling resonator, for qubit %d' % i,
        swapAmpBND = 0.2
        swapAmpSteps = 0.001
        coarseSet = np.arange(0,swap10Amp*(1-swapAmpBND),swapAmpSteps*5)
        fineSet = np.arange(swap10Amp*(1-swapAmpBND),swap10Amp*(1+swapAmpBND), swapAmpSteps)
        swap10Amp = np.hstack((coarseSet,fineSet))
        mq.swapSpectroscopy(s, state=1, swapLen=st.arangePQ(0,75,2,ns), swapAmp=swap10Amp, measure=i, save=True, noisy=False)
        #run focktuner level =1
        print 'fock tuner for fine calibratin of cZControlLen'
        mq.fockTuner(s, n=1, iteration=3, tuneOS=False, stats=1800, measure=i, save=True, noisy=False)
        print 'done. Calibrated Control qubits'

        print 'Tuning up pi-pulse for |2> of qubit %d' % i,
        noonPQ.pituner21(s, stats = 1800, measure=i, noisy=False, findMPA=True)
        print 'done'

        print 'measuring SWAP21 Spectroscopy'
        swap21Len, swap21Amp = mq.swap21tuner(s, measure=i, stats=1800, noisy=False)
        print 'measuring 2D-SWAP Spec around resonator, for qubit %d' % i,
        mq.swapSpectroscopy(s, state=2, swapLen=st.arangePQ(0,60,2,ns), swapAmp=st.r[swap21Amp*(1-0.2):swap21Amp*(1+0.2):0.001], measure=i, save=True, noisy=False)
        mq.fockTuners21(s, n=1, iteration=3, tuneOS=False, stats=1800, measure=i, save=True, noisy=False)
        print 'done. Calibrated Target qubits'

    print 'now starting qubit-qubit timing calibrations...'
    print 'measuring qubit-qubit delay via the resonator'
    for j,k in [(0,1),(1,0), (0,2),(2,0), (1,2),(2,1), (0,3),(3,0), (1,3),(3,1), (2,3),(3,2)]: #( add back in when all 4 qubits work!
        mq.testQubResDelayCmp(s,measureC=j, measureT=k)
    print 'now measuring resonator T1 using q0 for photon exchange'
    noonPQ.resonatorT1(s, stats=1800, measure=0, whichRes='Coupler')
    end = datetime.now()
    print 'start:', start
    print 'end:', end
    print 'elapsed time for qubit-resonator scans:', (end-start)

def qubit_memory_resonator_scans(s, stats=1800):
    start = datetime.now()

    N = len(s['config'])
    for i in range(N):
        print 'measuring SWAP10 Spectroscopy, qubit %d' % i,
        swap10Len, swap10Amp = mq.swap10tuner(s, measure=i, stats=stats, noisy=False, whichRes='Memory')

        print 'measuring 2D-SWAP Spec around Memory resonator, for qubit %d' % i,
        swapAmpBND = 0.2
        swapAmpSteps = 0.001
        coarseSet = np.arange(0,swap10Amp*(1-swapAmpBND),swapAmpSteps*5)
        fineSet = np.arange(swap10Amp*(1-swapAmpBND),swap10Amp*(1+swapAmpBND), swapAmpSteps)
        swap10Amp = np.hstack((coarseSet,fineSet))
        mq.swapSpectroscopy(s, swapLen=st.arangePQ(0,300,5,ns), swapAmp=swap10Amp, measure=i,
                  save=True, noisy=False, stats=stats, whichRes='Memory')
        #run focktuner level =1
        print 'fock tuner for fine calibratin of memoryReadWriteLen'
        mq.fockTuner(s, n=1, iteration=3, tuneOS=False, stats=stats, measure=i, save=True, noisy=False, whichRes='Memory')
        print 'done. Memory resonator tuned up'
        print 'now measuring memory resonator T1 for resonator %d' %i,
        noonPQ.resonatorT1(s, stats=stats, measure=i, whichRes='Memory')

    end = datetime.now()
    print 'start:', start
    print 'end:', end
    print 'elapsed time for qubit-mem-resonator scans:', (end-start)

def gate_bringup(s):
    start = datetime.now()

    N = len(s['config'])
    for i in range(N):
        print 'Begin Calibrating Single Qubit Hadamard Gates'
        print 'Z-pi pulse tuner'
        mq.pitunerZ(s, measure=i, save=True, stats = 1800, update=True, noisy=False)
        print 'done tuning Z-pi amplitude for qubit %d' %i,
        hadi.hadamardTrajectory(s, measure=i, stats=1500, useHD=True, useTomo=True, tBuf=5*ns, save=True, noisy=False)
        print 'plotting hadamard trajectory on Bloch Sphere'
        print 'correcting for visibilities...generating pretty plots'
        hadi.plotTrajectory(path=s._dir, dataset=None, state=None) #grabs the most recent dataset in the current session
        hadi.plotDensityArrowPlot(path=s._dir, dataset = None) #grabs most recent dataset in the current session

    end = datetime.now()
    print 'start:', start
    print 'end:', end
    print 'elapsed time for single qubit gate bringups:', (end-start)

def create_bell_state_iswap(s,zSweep=False):
    start = datetime.now()

    for j,k in [(0,1),(0,2),(1,2)]: #(0,3),(1,3),(2,3) add back in when all 4 qubits work!
        Qj = s[s['config'][j]]
        print 'measuring SWAPs between q%d and q%d via Rc' %(j,k)
        shor.iSwap(s, measure=[j,k], stats=1500, noisy=False)
        if zSweep:
            bellPhase = Qj['piAmpZ']
            bellPhases = np.arange(-1.0,1.0,0.1)*bellPhase
            for phase in bellPhases:
                print 'Preparing Bell-States via SQRT(iSWAP) between q%d and q%d via Rc' %(j,k)
                shor.bellStateiSwap(s, reps=5, measure=[j,k], stats=1800, corrAmp=phase)
        else:
            print 'Preparing Bell-States via SQRT(iSWAP) between q%d and q%d via Rc' %(j,k)
            shor.bellStateiSwap(s, reps=5, measure=[j,k], stats=1800, corrAmp=0.0)
    end = datetime.now()
    print 'start:', start
    print 'end:', end
    print 'elapsed time for single qubit gate bringups:', (end-start)

def cPhase_bringup(s):
    start = datetime.now()
    N = len(s['config'])
    for i in range(N):
        print 'done with COWS'


    end = datetime.now()
    print 'start:', start
    print 'end:', end
    print 'elapsed time for c-phase bringups:', (end-start)

def full_run(s):
    bringup_multiqubits(s)
    measure_w(s) # do tomography


def bringup_multiqubits(s):
    start = datetime.now()

    test_coupling(s, guess_zpa=True, use_overshoots=False)
    tune_swaps(s)
    #test_coupling(s, guess_zpa=False, use_overshoots=True) # try the swap again with correct overshoot

    tune_phases(s) # tune microwave phases between channels
    check_phase_vs_time(s) # tune microwave phase between channels as a function of time
    tune_swap_dphases(s) # tune phase change due to a swap z-pulse
    tune_dphases(s) # tune phase change due to z-pulses of any length

    end = datetime.now()
    print 'start:', start
    print 'end:', end
    print 'elapsed:', (end-start)


# TODO save hyperbolic fit to coupling strength, so we can adjust for unequal coupling strengths
def test_coupling(s, guess_zpa=True, use_overshoots=False):
    """Determine the z-pulse amplitude needed to bring qubits into resonance.

    Also, measure coupling strength between qubits.

    sets: w_zpulse_amp, w_swap_amp
    """
    s, qubits, Qubits = util.loadQubits(s, write_access=True)
    q0, q1, q2 = qubits
    Q0, _Q1, Q2 = Qubits

    zpafunc0 = mq.get_zpa_func(q0)
    zpafunc1 = mq.get_zpa_func(q1)
    zpafunc2 = mq.get_zpa_func(q2)

    S = 0.015 * GHz # expected coupling strength

    if guess_zpa:
        # guess the required zpa
        zpa0 = q0['wZpulseAmp'] = zpafunc0(q1['f10'])
        zpa2 = q2['wZpulseAmp'] = zpafunc2(q1['f10'])

        # calculate zpa limits to give a reasonable range based on the expected coupling strength
        zpalims0 = sorted([zpafunc0(q1['f10'] - S*2), zpafunc0(q1['f10'] + S*2)])
        zpalims2 = sorted([zpafunc2(q1['f10'] - S*2), zpafunc2(q1['f10'] + S*2)])
    else:
        # use calibrated zpa
        zpa0 = q0['wZpulseAmp']
        zpa2 = q2['wZpulseAmp']

        # calculate zpa limits based on calibrated coupling change with zpa
        dzpa0 = abs(S[GHz]*2 / q0['coupling1DsByDzpa'][GHz])
        zpalims0 = [zpa0 - dzpa0, zpa0 + dzpa0]

        dzpa2 = abs(S[GHz]*2 / q2['coupling1DsByDzpa'][GHz])
        zpalims2 = [zpa2 - dzpa2, zpa2 + dzpa2]

    if not use_overshoots:
        q0['wZpulseOvershoot'] = 0.0
        q2['wZpulseOvershoot'] = 0.0

    from pyle.fitting import fourierplot

    opts = {
        'collect': True,
        'noisy': False,
    }
    null012 = measurement.Null(3, [0,1,2])

    if 1:
        # couple q0 with q1
        rng0 = st.r[zpalims0[0]:zpalims0[1]:(zpalims0[1] - zpalims0[0]) / 25]
        data0 = werner.w_state(s, name='coupling 0 and 1 2D', pi_pulse_on=1, measure=[0],
                           t_couple=1000*ns, delay=st.r[0:200:4,ns], zpas=[rng0, 0, 0], **opts)
        S0, zpa0, ds_by_dzpa0 = fourierplot.fitswap(data0, return_fit=True) # find swap frequency and optimal z-pulse
        print S0, zpa0, ds_by_dzpa0
        Q0['swapAmp'] = Q0['wZpulseAmp'] = zpa0
        Q0['coupling1'] = S0*MHz
        Q0['coupling1DsByDzpa'] = ds_by_dzpa0*MHz

        # do a 1D scan with the optimal pulse amplitude
        data0 = werner.w_state(s, name='coupling 0 and 1', pi_pulse_on=1, measure=null012,
                           t_couple=1000*ns, delay=st.r[0:100:2,ns], zpas=[zpa0, 0, 0], stats=3000, **opts)

    if 1:
        # couple q2 with q1
        rng2 = st.r[zpalims2[0]:zpalims2[1]:(zpalims2[1] - zpalims2[0]) / 25]
        data2 = werner.w_state(s, name='coupling 1 and 2 2D', pi_pulse_on=1, measure=[2],
                           t_couple=1000*ns, delay=st.r[0:200:4,ns], zpas=[0, 0, rng2], **opts)
        S2, zpa2, ds_by_dzpa2 = fourierplot.fitswap(data2, return_fit=True) # find swap frequency and optimal z-pulse
        print S2, zpa2, ds_by_dzpa2
        Q2['swapAmp'] = Q2['wZpulseAmp'] = zpa2
        Q2['coupling1'] = S2*MHz
        Q2['coupling1DsByDzpa'] = ds_by_dzpa2*MHz

        # do a 1D scan with the optimal pulse amplitude
        data2 = werner.w_state(s, name='coupling 1 and 2', pi_pulse_on=1, measure=null012,
                           t_couple=1000*ns, delay=st.r[0:100:2,ns], zpas=[0, 0, zpa2], stats=3000, **opts)

    if 1:
        # couple q0 with q2, moving q1 to negative detuning
        zpa1 = zpafunc1(q2['f10']) # move q1 out of the way
        rng2 = st.r[zpalims2[0]:zpalims2[1]:(zpalims2[1] - zpalims2[0]) / 25]
        data2 = werner.w_state(s, name='coupling 0 and 2 2D', pi_pulse_on=0, measure=[2],
                           t_couple=1000*ns, delay=st.r[0:200:4,ns], zpas=[zpa0, zpa1, rng2], **opts)
        S2, zpa2, ds_by_dzpa2 = fourierplot.fitswap(data2, return_fit=True) # find swap frequency and optimal z-pulse
        print S2, zpa2, ds_by_dzpa2
        #Q2['swapAmp'] = Q2['wZpulseAmp'] = zpa2
        Q2['coupling0'] = S2*MHz # save this coupling value, but not in the standard place
        Q2['coupling0DsByDzpa'] = ds_by_dzpa2*MHz # save fit, but not in the standard place

        # do a 1D scan with the optimal pulse amplitude
        data2 = werner.w_state(s, name='coupling 0 and 2', pi_pulse_on=0, measure=null012,
                           t_couple=1000*ns, delay=st.r[0:100:2,ns], zpas=[zpa0, zpa1, zpa2], stats=3000, **opts)


def tune_swaps(s):
    """Adjust overshoot and pulse length to get the best swap."""

    # overshoots don't seem to help
    s.q0['swapOvershoot'] = 0.0
    s.q2['swapOvershoot'] = 0.0

    werner.swaptuner(s, measure=0, pi_pulse_on=1, noisy=False, update=True, save=False, stats=3000, tune_overshoot=False)
    werner.swaptuner(s, measure=2, pi_pulse_on=1, noisy=False, update=True, save=False, stats=3000, tune_overshoot=False)

    # set overshoots for w-state to be equal to calibrated swap overshoots
    s.q0['wZpulseOvershoot'] = s.q0['swapOvershoot']
    s.q2['wZpulseOvershoot'] = s.q2['swapOvershoot']


def tune_phases(s, t0=None, calibrated_amp=True, stats=3000L, res=50, plot=True):
    s, qubits, Qubits = util.loadQubits(s, write_access=True)
    q0, q1, q2 = qubits

    if calibrated_amp:
        zpa0 = q0['swapAmp']
        zpa2 = q2['swapAmp']
    else:
        zpafunc0 = mq.get_zpa_func(q0)
        zpafunc2 = mq.get_zpa_func(q2)
        zpa0 = zpafunc0(q1['f10'])
        zpa2 = zpafunc2(q1['f10'])

    f_couple = 0.015*GHz
    t_couple = (1/f_couple/4)[ns]*ns

    phase = st.r[-np.pi:np.pi:np.pi/res]

    data0 = werner.uwave_phase_adjust(s, phase=phase, t0=t0, t_couple=t_couple, adjust=0, ref=1, zpas=[zpa0, 0.0, 0.0], collect=True, noisy=False, stats=stats)
    data2 = werner.uwave_phase_adjust(s, phase=phase, t0=t0, t_couple=t_couple, adjust=2, ref=1, zpas=[0.0, 0.0, zpa2], collect=True, noisy=False, stats=stats)

    def fitfunc(x, c):
        return -np.sin(x - c[0]) * c[1] + c[2]

    ph, _p00, p01, p10, _p11 = data0.T
    fit0, _ = leastsq(lambda c: fitfunc(ph, c) - p10, [q0['uwavePhase'], (max(p10)-min(p10))/2.0, (max(p10)+min(p10))/2.0])
    if fit0[1] < 0:
        fit0[0] = (fit0[0] + 2*np.pi) % (2*np.pi) - np.pi
        fit0[1] *= -1
    fit0[0] = (fit0[0] + np.pi) % (2*np.pi) - np.pi

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(ph, p10, 'b.', label='|10>')
        ax.plot(ph, p01, 'g.', label='|01>')
        ax.plot(ph, fitfunc(ph, fit0), 'r-')
        ax.axvline(fit0[0], linestyle='--', color='gray')
        ax.set_title('microwave phase adjustment, qubit 0, ref 1: phase = %0.5g' % fit0[0])
        ax.legend()
    print 'old phase:', q0['uwavePhase']
    print 'new phase:', fit0[0]
    Qubits[0]['uwavePhase'] = fit0[0]

    ph, _p00, p01, p10, _p11 = data2.T
    fit2, _ = leastsq(lambda c: fitfunc(ph, c) - p01, [q2['uwavePhase'], (max(p01)-min(p01))/2.0, (max(p01)+min(p01))/2.0])
    if fit2[1] < 0:
        fit2[0] = (fit2[0] + 2*np.pi) % (2*np.pi) - np.pi
        fit2[1] *= -1
    fit2[0] = (fit2[0] + np.pi) % (2*np.pi) - np.pi

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(ph, p10, 'b.', label='|10>')
        ax.plot(ph, p01, 'g.', label='|01>')
        ax.plot(ph, fitfunc(ph, fit2), 'r-')
        ax.axvline(fit2[0], linestyle='--', color='gray')
        ax.set_title('microwave phase adjustment, qubit 2, ref 1: phase = %0.5g' % fit2[0])
        ax.legend()
    print 'old phase:', q2['uwavePhase']
    print 'new phase:', fit2[0]
    Qubits[2]['uwavePhase'] = fit2[0]

    return fit0[0], fit2[0]


def check_phase_vs_time(s, plot=True):
    s, qubits, Qubits = util.loadQubits(s, write_access=True)

    phases0 = []
    phases2 = []
    t0s = st.r[0:12:1,ns]

    for t0 in t0s:
        ph0, ph2 = tune_phases(s, t0, stats=1200, res=20, plot=False)
        phases0.append(ph0)
        phases2.append(ph2)

    phases0 = np.unwrap(phases0)
    phases2 = np.unwrap(phases2)

    fit0 = np.polyfit(t0s, phases0, 1)
    fit2 = np.polyfit(t0s, phases2, 1)

    df0 = (s.q1['f10'] - s.q0['f10'])[GHz]
    df2 = (s.q1['f10'] - s.q2['f10'])[GHz]

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(t0s, phases0, 'b.', label='measured phase')
        ax.plot(t0s, np.polyval(fit0, t0s), 'r-', label='phase fit')
        ax.plot(t0s, np.polyval([-2*np.pi*df0, 0], t0s), 'c-', label='detuning')
        ax.legend()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(t0s, phases2, 'b.', label='measured phase')
        ax.plot(t0s, np.polyval(fit2, t0s), 'r-', label='phase fit')
        ax.plot(t0s, np.polyval([-2*np.pi*df2, 0], t0s), 'c-', label='detuning')
        ax.legend()

    print 'qubit 0:'
    print '  detuning:', df0
    print '  phase fit:', fit0[0]/(2*np.pi)
    print '  phase offset:', fit0[1]/(2*np.pi)
    print
    Qubits[0]['uwavePhaseSlope'] = fit0[0]/(2*np.pi) * GHz
    Qubits[0]['uwavePhaseOfs'] = fit0[1]
    Qubits[0]['uwavePhaseFit'] = fit0


    print 'qubit 2:'
    print '  detuning q2:', df2
    print '  phase fit:', fit2[0]/(2*np.pi)
    print '  phase offset:', fit2[1]/(2*np.pi)
    print
    Qubits[2]['uwavePhaseSlope'] = fit2[0]/(2*np.pi) * GHz
    Qubits[2]['uwavePhaseOfs'] = fit2[1]
    Qubits[2]['uwavePhaseFit'] = fit2


def tune_swap_dphases(s, calibrated_amp=True):
    s, qubits, Qubits = util.loadQubits(s, write_access=True)
    q0, q1, q2 = qubits
    Q0, _Q1, Q2 = Qubits

    if not calibrated_amp:
        zpafunc0 = mq.get_zpa_func(q0)
        zpafunc2 = mq.get_zpa_func(q2)
        q0['swapAmp'] = zpafunc0(q1['f10'])
        q2['swapAmp'] = zpafunc2(q1['f10'])

    def fitfunc(x, c):
        return np.cos(x - c[0]) * c[1] + c[2]

    def fit_dphase(i, q):
        print 'measuring qubit', i
        phase = st.r[-np.pi:np.pi:np.pi/20]
        data = werner.swap_dphase_adjust(s, phase, adjust=i, ref=1, stats=600, noisy=False, collect=True, save=False)

        ph, p1 = data.T
        fit, _ = leastsq(lambda c: fitfunc(ph, c) - p1, [ph[np.argmax(p1)], (max(p1)-min(p1))/2.0, (max(p1)+min(p1))/2.0])
        if fit[1] < 0:
            fit[0] = (fit[0] + 2*np.pi) % (2*np.pi) - np.pi
            fit[1] *= -1

        print '  dphase =', fit[0]
        dphase = fit[0]
        return dphase

    dphase0 = fit_dphase(0, q0)
    dphase2 = fit_dphase(2, q2)

    print 'qubit 0:'
    print '  swapDphase:', dphase0
    print
    Q0['swapDphase'] = dphase0


    print 'qubit 2:'
    print '  swapDphase:', dphase2
    print
    Q2['swapDphase'] = dphase2


def tune_dphases(s, calibrated_amp=True):
    s, qubits, Qubits = util.loadQubits(s, write_access=True)
    q0, q1, q2 = qubits
    Q0, _Q1, Q2 = Qubits

    if not calibrated_amp:
        zpafunc0 = mq.get_zpa_func(q0)
        zpafunc2 = mq.get_zpa_func(q2)
        q0['wZpulseAmp'] = zpafunc0(q1['f10'])
        q2['wZpulseAmp'] = zpafunc2(q1['f10'])

    def fitfunc(x, c):
        return np.cos(x - c[0]) * c[1] + c[2]

    zp_rng = st.r[0:25:1,ns]
    ts = np.array([zp_len[ns] for zp_len in zp_rng])

    def fit_phases(i, q):
        print 'measuring qubit', i
        dphases = []
        for zp_len in zp_rng:
            q['wZpulseLen'] = zp_len
            phase = st.r[-np.pi:np.pi:np.pi/20]
            data = werner.w_dphase_adjust(s, phase, adjust=i, ref=1, stats=600,
                                          noisy=False, collect=True, save=True)
            ph, p1 = data.T
            fit, _ = leastsq(lambda c: fitfunc(ph, c) - p1, [ph[np.argmax(p1)], (max(p1)-min(p1))/2.0, (max(p1)+min(p1))/2.0])
            if fit[1] < 0:
                fit[0] = (fit[0] + 2*np.pi) % (2*np.pi) - np.pi
                fit[1] *= -1

            print '  t =', zp_len[ns], '  dphase =', fit[0]
            dphases.append(fit[0])
        print
        return np.unwrap(dphases)

    dphases0 = fit_phases(0, q0)
    dphases2 = fit_phases(2, q2)

    fit0 = np.polyfit(ts, dphases0, 1)
    fit2 = np.polyfit(ts, dphases2, 1)

    df0 = (q1['f10'] - q0['f10'])[GHz]
    df2 = (q1['f10'] - q2['f10'])[GHz]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ts, dphases0, 'b.', label='measured phase')
    ax.plot(ts, np.polyval(fit0, ts), 'r-', label='phase fit')
    ax.plot(ts, np.polyval([-2*np.pi*df0, 0], ts), 'c-', label='detuning')
    ax.legend()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ts, dphases2, 'b.')
    ax.plot(ts, np.polyval(fit2, ts), 'r-')
    ax.plot(ts, np.polyval([-2*np.pi*df2, 0], ts), 'c-', label='detuning')
    ax.legend()

    print 'qubit 0:'
    print '  detuning:', df0
    print '  phase fit:', fit0[0]/(2*np.pi)
    print '  phase offset:', fit0[1]/(2*np.pi)
    print
    Q0['wDphaseSlope'] = fit0[0]/(2*np.pi) * GHz
    Q0['wDphaseFit'] = fit0


    print 'qubit 2:'
    print '  detuning q2:', df2
    print '  phase fit:', fit2[0]/(2*np.pi)
    print '  phase offset:', fit2[1]/(2*np.pi)
    print
    Q2['wDphaseSlope'] = fit2[0]/(2*np.pi) * GHz
    Q2['wDphaseFit'] = fit2


def measure_w(s, with_tomo=True):
    s, qubits = util.loadQubits(s)
    q0, _q1, q2 = qubits

    t_swap = (q0['swapLen'] + q2['swapLen']) / 2
    t_couple = t_swap * 4.0/9.0

    for _i in itertools.count():
        # couple all three qubits together
        null012 = measurement.Null(3, [0,1,2])
        werner.w_state(s, pi_pulse_on=1, t_couple=1000*ns, delay=st.r[0:50:1,ns], measure=null012, stats=1200)
        werner.w_state(s, pi_pulse_on=1, t_couple=t_couple, delay=st.r[0:50:1,ns], measure=null012, stats=1200)

        if with_tomo:
            # do tomography
            tomo012 = measurement.TomoNull(3, [0,1,2])
            opts = {
                'pi_pulse_on': 1,
                'measure': tomo012,
                'stats': 600,
                'pipesize': 1,
            }
            werner.w_state(s, t_couple=1000*ns, delay=st.r[0:30:1,ns], **opts)
            werner.w_state(s, t_couple=t_couple, delay=st.r[0:30:1,ns], **opts)
            werner.w_state(s, t_couple=1000*ns, delay=st.r[15:20:0.25,ns], **opts)
            werner.w_state(s, t_couple=t_couple, delay=st.r[15:20:0.25,ns], **opts)


def tweak_detunings(s):
    s, qubits = util.loadQubits(s)
    q0, _q1, q2 = qubits

    zpa0 = q0['swapAmp']
    zpa2 = q2['swapAmp']

    def sfunc(q):
        p0 = q['coupling1'][MHz]
        p1 = q['coupling1DsByDzpa'][MHz]
        zpa0 = q['swap_amp']
        return lambda zpa: np.sqrt(p0**2 + p1**2*(zpa - zpa0))

    sfunc0 = sfunc(q0)
    sfunc2 = sfunc(q2)

    smin0 = sfunc0(q0['swapAmp'])
    smin2 = sfunc2(q2['swapAmp'])

    print 'minimum splittings:'
    print '  q0 <-> q1: %g MHz' % smin0
    print '  q2 <-> q1: %g MHz' % smin2
    print

    if smin0 < smin2:
        # adjust zpa 0
        zpa0opt = fsolve(lambda zpa: sfunc0(zpa) - smin2, zpa0)
        det0 = q0['coupling1DsByDzpa'] * (zpa0opt - q0['swapAmp'])
        print 'qubit0 optimal zpa=%g, s=%g, det=%g' % (zpa0opt, sfunc0(zpa0opt), det0)

        zpas = sorted([zpa0opt, 2*zpa0 - zpa0opt])
        print 'trying', zpas
        for zpa0 in zpas:
            q0['swapAmp'] = zpa0
            measure_w(s, with_tomo=False)

    else:
        # adjust zpa 0
        zpa2opt = fsolve(lambda zpa: sfunc2(zpa) - smin0, zpa2)
        det2 = q2['coupling1DsByDzpa'] * (zpa2opt - q2['swapAmp'])
        print 'qubit2 optimal zpa=%g, s=%g, det=%g' % (zpa2opt, sfunc2(zpa2opt), det2)

        zpas = sorted([zpa2opt, 2*zpa2 - zpa2opt])
        print 'trying', zpas
        for zpa2 in zpas:
            q2['swapAmp'] = zpa2
            measure_w(s, with_tomo=False)


def measure_ghz(s, with_tomo=True, with_ghz=True):
    s, qubits = util.loadQubits(s)
    q0, _q1, q2 = qubits

    for _i in [0]: #itertools.count():
        # couple all three qubits together
        null012 = measurement.Null(3, [0,1,2])
        #mq.w_state(s, pi_pulse_on=1, t_couple=1000*ns, delay=st.r[0:50:1,ns], measure=null012, stats=1200)
        #mq.w_state(s, pi_pulse_on=1, t_couple=17.5*ns, delay=st.r[0:50:1,ns], measure=null012, stats=1200)

        ghz.ghz_simult(s, stage=st.r[0:3:0.05], measure=measurement.Null(3), stats=1800)
        ghz.ghz_iswap(s, stage=st.r[0:4:0.05], measure=measurement.Null(3), stats=1800)

        if with_ghz:
            ghz.ghz_simult(s, stage=st.r[0:3:0.1], measure=ghz.GHZ(), stats=1200)
            ghz.ghz_iswap(s, stage=st.r[0:4:0.1], measure=ghz.GHZ(), stats=1200)

        if with_tomo:
            # do tomography
            tomo012 = measurement.TomoNull(3, [0,1,2])
            opts = {
                'pi_pulse_on': 1,
                'measure': tomo012,
                'stats': 600,
                'pipesize': 1,
            }
            #mq.w_state(s, t_couple=1000*ns, delay=st.r[0:30:1,ns], **opts)
            #mq.w_state(s, t_couple=19*ns, delay=st.r[0:30:1,ns], **opts)
            #mq.w_state(s, t_couple=1000*ns, delay=st.r[15:25:0.25,ns], **opts)
            #mq.w_state(s, t_couple=19*ns, delay=st.r[15:25:0.25,ns], **opts)

            ghz.ghz_simult(s, stage=st.r[0:3:0.1], measure=measurement.TomoNull(3), pipesize=1, stats=1200)
            ghz.ghz_iswap(s, stage=st.r[0:4:0.1], measure=measurement.TomoNull(3), pipesize=1, stats=1200)


def measure_ghz_iswap(s, with_tomo=True, with_ghz=True):
    s, qubits = util.loadQubits(s)
    q0, _q1, q2 = qubits

    while True:
        #for sf, ef, es in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
        #                   (2, 0, 0), (2, 0, 1), (2, 1, 0), (2, 1, 1)]:
        for sf, ef, es in [(0, 1, 0), (0, 1, 1),
                           (2, 0, 0), (2, 0, 1), (2, 1, 0), (2, 1, 1)]:
            # couple all three qubits together
            null012 = measurement.Null(3, [0,1,2])
            #mq.w_state(s, pi_pulse_on=1, t_couple=1000*ns, delay=st.r[0:50:1,ns], measure=null012, stats=1200)
            #mq.w_state(s, pi_pulse_on=1, t_couple=17.5*ns, delay=st.r[0:50:1,ns], measure=null012, stats=1200)

            opts = {
                'swap_first': sf,
                'swap_second': 2-sf,
                'echo_first': ef,
                'echo_second': es,
            }
            opts2 = {
                'swap_first': sf,
                'swap_second': 2-sf,
            }

            #ghz.ghz_simult(s, stage=st.r[0:3:0.05], measure=measurement.Null(3), stats=1800)
            #ghz.ghz_iswap(s, stage=st.r[0:4:0.05], measure=null012, stats=1200, **opts)

            #if with_ghz:
            #    ghz.ghz_simult(s, stage=st.r[0:3:0.1], measure=ghz.GHZ(), stats=1200)
            #    ghz.ghz_iswap(s, stage=st.r[0:4:0.1], measure=ghz.GHZ(), stats=1200)

            if with_tomo:
                # do tomography
                tomo012 = measurement.TomoNull(3, [0,1,2])
                #opts = {
                #    'pi_pulse_on': 1,
                #    'measure': tomo012,
                #    'stats': 600,
                #    'pipesize': 1,
                #}
                #mq.w_state(s, t_couple=1000*ns, delay=st.r[0:30:1,ns], **opts)
                #mq.w_state(s, t_couple=19*ns, delay=st.r[0:30:1,ns], **opts)
                #mq.w_state(s, t_couple=1000*ns, delay=st.r[15:25:0.25,ns], **opts)
                #mq.w_state(s, t_couple=19*ns, delay=st.r[15:25:0.25,ns], **opts)

                #ghz.ghz_simult(s, stage=st.r[0:3:0.1], measure=measurement.TomoNull(3), pipesize=1, stats=600)
                #ghz.ghz_iswap(s, stage=[4], measure=tomo012, pipesize=1, stats=6000, **opts)
                ghz.ghz_iswap_tight(s, stage=[4], measure=tomo012, pipesize=1, stats=6000, **opts2)
                #ghz.ghz_iswap(s, stage=st.r[0:4:0.2], measure=tomo012, pipesize=1, stats=600, **opts)


def measure_ghz_iswap_tight(s, with_tomo=True, with_ghz=True):
    s, qubits = util.loadQubits(s)
    q0, _q1, q2 = qubits

    for _i in range(1):
        for sf in [0, 2]:
            opts = {
                'swap_first': sf,
                'swap_second': 2-sf,
            }

            #null012 = measurement.Null(3, [0,1,2])
            #ghz.ghz_simult(s, stage=st.r[0:3:0.05], measure=null012, stats=1800)
            #ghz.ghz_iswap(s, stage=st.r[0:4:0.05], measure=null012, stats=1200, **opts)

            #if with_ghz:
            #    ghz.ghz_simult(s, stage=st.r[0:3:0.1], measure=ghz.GHZ(), stats=1200)
            #    ghz.ghz_iswap(s, stage=st.r[0:4:0.1], measure=ghz.GHZ(), stats=1200)

            if with_tomo:
                tomo012 = measurement.TomoNull(3, [0,1,2])
                ghz.ghz_iswap_tight(s, stage=[0,1,2,3,4], measure=tomo012, pipesize=1, stats=6000, **opts)


def measure_ghz_simult(s, with_tomo=True, with_ghz=True):
    s, qubits = util.loadQubits(s)
    q0, _q1, q2 = qubits

    for _i in range(1):
        for sf in [0, 2]:
            opts = {
                'swap_first': sf,
                'swap_second': 2-sf,
            }

            #null012 = measurement.Null(3, [0,1,2])
            #ghz.ghz_simult(s, stage=st.r[0:3:0.05], measure=null012, stats=1800)
            #ghz.ghz_iswap(s, stage=st.r[0:4:0.05], measure=null012, stats=1200, **opts)

            #if with_ghz:
            #    ghz.ghz_simult(s, stage=st.r[0:3:0.1], measure=ghz.GHZ(), stats=1200)
            #    ghz.ghz_iswap(s, stage=st.r[0:4:0.1], measure=ghz.GHZ(), stats=1200)

            if with_tomo:
                tomo012 = measurement.TomoNull(3, [0,1,2])
                ghz.ghz_iswap_tight(s, stage=[0,1,2,3,4], measure=tomo012, pipesize=1, stats=6000, **opts)



