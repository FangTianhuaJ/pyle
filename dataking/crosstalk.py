import numpy as np
from msvcrt import getch,kbhit

import labrad
from labrad.units import Unit
V, mV, sec, us, ns, GHz, MHz = [Unit(s) for s in ('V', 'mV', 's', 'us', 'ns', 'GHz', 'MHz')]

import pyle.envelopes as env
from pyle.pipeline import returnValue, FutureList
from pyle.dataking import envelopehelpers as eh
from pyle.dataking import multiqubitPQ as mq
from pyle.dataking.fpgaseq import runQubits
from pyle.util import sweeptools as st
from pyle.plotting import dstools
from pyle.fitting import fitting
from pyle.dataking import utilMultilevels as ml
import sweeps
import util
from pyle.dataking import fpgaseq
fpgaseq.PREPAD = 350

#stuff related to crosstalk goes here: measure it, correct for it, etc.


def zpulseCrosstalk(sample, measure, drops=None, adds=None, zpas=st.r[-1.0:1.0:0.5],
                    delay=st.r[0:100:1,ns], fringeFreq=50*MHz, nfftpoints=4000,
                    stats=1200L, saveFreqtuner=False, noisy=False, plot=True, collect=False):
    """Measure the z-pulse crosstalk between all pairs of qubits.

    We create the crosstalk matrix and its inverse and store them in
    the registry to correct the z-pulse signals if desired.

    Assumes you have already run mq.find_zpa_func, so that the
    calZpaFunc has already been set, as xtalk is relative to this.

    The elements of the crosstalk matrix are defined in this way:
    A[i,j] is the frequency shift of qubit i due to a z pulse on qubit j.

    PARAMETERS
    sample: object defining qubits to measure, loaded from registry
    measure - list of ints: Qubits between which to measure zpa crosstalk.
    drops - list of list of ints: Pairs that should not be scanned. If None, do
        not drop out any pairs.
    adds - list of list of ints: Pairs to be scanned. If None, scan all pairs.
    zpas - list of floats: Z-pulse ampitudes.
    delay - list of times: Delay times between pi/2 pulses.
    fringeFreq - value [Mhz]: Desired frequency of Ramsey fringes
    nfftpoints - int: Number of frequencies to use in FFT.
    stats - scalar: number of times a point will be measured.
    saveFreqtuner - bool: Whether to save data from each freqtuner to Data Vault.
    noisy - bool: Whether to print out probabilities while the scan runs.
    plot - bool: Whether to plot freqs vs z-pulse amplitudes.
    collect - bool: Whether to return matrices to the local scope.
    """
    zpas = [zpas[index] for index in np.argsort(abs(np.array(zpas)))]
    s, qubits, Qubits = util.loadQubits(sample, write_access=True)
    dim = len(qubits)
    dimConfig = len(s['config'])
    if adds is None:
        if drops is None:
            drops = []
        # Pairs is all pairs of elements in measure excluding those in drops.
        pairs = [[i,j] for i in measure for j in measure if [i,j] not in drops]
    else:
        if drops is None:
            pairs = adds
        else:
            raise Exception('Specified both adds and drops.')
    for i in range(dimConfig):
        if len(qubits[i]['calZpaXtalk']) != dimConfig:
            xtalkRow = dimConfig*[0]
            xtalkRow[i] = 1
            Qubits[i]['calZpaXtalk'] = xtalkRow
    for pair in pairs:
        i,j = pair
        rowMat = Qubits[i]['calZpaXtalk']
        print 'Old row:\n',rowMat
        if i == j:
            rowMat[i] = 1.0
        else:
            print 'measuring crosstalk on %s from z-pulse on %s' % (i, j)
            xtfunc = freqTunerZCrosstalk(s, measure=i, control=j, zpas=zpas, delay=delay,
                                            fringeFreq=fringeFreq, nfftpoints=nfftpoints,
                                            stats=stats, saveFreqtuner=saveFreqtuner, plot=plot)
            aii = float(qubits[i]['calZpaFunc'][0])
            aij = float(xtfunc[0])
            print 'crosstalk =', aii/aij
            print 'Old row:\n',rowMat
            rowMat[j] = aii/aij
        Qubits[i]['calZpaXtalk'] = rowMat
        print 'New row:\n',Qubits[i]['calZpaXtalk']
    A = [Q['calZpaXtalk'] for Q in Qubits]
    Ainv = np.linalg.inv(A)
    print
    print 'xtalk matrix:\n', A
    print
    print 'inverse xtalk matrix:\n', Ainv
    for i, Qi in enumerate(Qubits):
        Qi['calZpaXtalk'] = A[i]
        Qi['calZpaXtalkInv'] = Ainv[i]
    if collect:
        return A,Ainv


def freqTunerZCrosstalk(sample, control, measure, zpas=[0.0], tBuf = 5.0*ns,
                        delay=st.r[0:100:1,ns], fringeFreq = 50*MHz, nfftpoints=4000,
                        stats=1200L, name='Z-Pulse Crosstalk', saveFreqtuner = False,
                        save=True, noisy=True, plot=True):
    """Measure z-pulse crosstalk between two qubits.

    This scan uses freqtuner (with func explicitly given here) on measure while
    applying a z-pulse of different amplitudes to control. The fit function is
    then returned.

    PARAMETERS
    sample: object defining qubits to measure, loaded from registry
    control - int: Qubit on which to apply z-pulse.
    measure - int: Qubit on which to measure freq shift.
    tBuf - time: Time to wait between pulses.
    zpas - list of floats: Z-pulse ampitudes.
    delay - list of times: Delay times between pi/2 pulses.
    fringeFreq - value [Mhz]: Desired frequency of Ramsey fringes
    nfftpoints - int: Number of frequencies to use in FFT.
    stats - scalar: number of times a point will be measured.
    name - string: Name of dataset.
    saveFreqtuner - bool: Whether to save data from each freqtuner to Data Vault.
    save - bool: Whether to save data to the Data Vault.
    noisy - bool: Whether to print out probabilities while the scan runs.
    plot - bool: Whether to plot freqs vs z-pulse amplitudes.
    """
    sample, qubits = util.loadQubits(sample)
    qC = qubits[control]
    qM = qubits[measure]
    if control == measure:
        raise Exception('Control and measure qubits must not be the same.')

    axes = [(delay, 'Delay')]
    zpaFreqs = []

    def func(server, delay):
        t = 0.5*qM['piLen'] + tBuf
        qM.xy = eh.mix(qM, eh.piHalfPulse(qM, t, phase=0.0, state=1), state=1)
        t += delay
        qM.xy += eh.mix(qM, eh.piHalfPulse(qM, t, phase = 2*np.pi*(fringeFreq['GHz']*delay['ns']), state=1), state=1)
        t += 0.5*qM['piLen'] + tBuf
        qC.z = env.rect(0, t, zpa)
        t += tBuf
        qM.z = eh.measurePulse(qM, t, state=1)
        qM['readout'] = True
        return runQubits(server, qubits, stats, probs=[1])

    for zpa in zpas:
        #Run the scan and save data
        kw = {'stats': stats, 'fringeFrequency': fringeFreq, 'controlQubit': control, 'measureQubit': measure, 'controlZpa':zpa}
        dataset = sweeps.prepDataset(sample, name+' Freqtuner', axes, measure=measure, kw=kw)
        data = sweeps.grid(func, axes, dataset = saveFreqtuner and dataset, collect=True, noisy=noisy)
        fringe = fitting.maxFreq(data, nfftpoints, plot=False)
        delta_freq = fringeFreq - fringe
        print 'At zpa of %s, Qubit frequency shifted by %s' %(zpa,delta_freq)
        zpaFreq = qM['f10']-delta_freq
        zpaFreqs.append(zpaFreq)

    if save:
        axes2 = [(zpas, 'Z-Pulse Amplitude')]
        deps2 = [('Frequency', '', 'GHz')]
        kw = {'stats': stats, 'fringeFrequency': fringeFreq, 'controlQubit': control, 'measureQubit': measure}
        dataset2 = sweeps.prepDataset(sample, name, axes2, deps2, measure=measure, kw=kw)
        dataset2.connect()
        dataset2._create()
        for index in range(len(zpas)):
            dataset2.add([zpas[index],zpaFreqs[index]])
    zpaFreqs = np.array([zpaFreq['GHz'] for zpaFreq in zpaFreqs])
    zpas = np.array(zpas)
    crosstalkFunc = np.polyfit(zpaFreqs**4,zpas,1)
    if plot:
        data = np.vstack([zpas,zpaFreqs]).T
        fig=dstools.plotDataset1D(data, [('Z-Pulse Amplitude','')], [('Frequency','','GHz')], style='.',
                                  legendLocation=None, show=False, markersize=15,
                                  title='Z-Pulse Crosstalk, Control: %d, Measure %d' %(control,measure))
        fig.get_axes()[0].plot(np.polyval(crosstalkFunc,data[:,1]**4),data[:,1],'r',linewidth=3)
        fig.show()
    return crosstalkFunc


def iSwap(sample, paramName, swapLen=st.r[-20:1000:1,ns], drive=0, swap=1,
         stats=1500L, name='iSwap MQ', save=True, collect=False, noisy=True):
    """Measures an iSwap^x gate.

    Excite DRIVE qubit in measure and swap the excitation into the resonator
    for a variable time. Then swaps whatever excitation is in the resonator
    into the SWAP qubit. Then measure all 4 probs.

    PARAMETERS
    sample: Object defining qubits to measure, loaded from registry.
    paramName - string: Name of swap pulses (e.g., 'swapAmp'+paramName,
        'swapTime'+paramName)
    swapLen - list of times: How long to swap 1st qubit into resonator.
    drive - int: Which qubit to initially exicte.
    swap - int: Which qubit to swap excitation from resonator into.
    stats - scalar: Number of times a point will be measured.
    name - string: Name of dataset.
    save - bool: Whether to save to data vault.
    collect - bool: Whether to return data.
    noisy - bool: Whether to print out probabilities while the scan runs.
    """
    sample, qubits = util.loadQubits(sample)
    q0 = qubits[drive]
    q1 = qubits[swap]
    measure = [drive,swap]
    #nameEx = [' q0->q1',' q1->q0']
    axes = [(swapLen, 'swap pulse length')]
    kw = {'stats': stats}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)#+nameEx[measure[0]]

    def func(server, curr):
        for q in qubits:
            q.z = env.NOTHING
        start = 0
        q0.xy = eh.mix(q0, eh.piPulseHD(q0, start))
        start += q0['piLen']/2
        q0.z += env.rect(start, curr, q0['swapAmp'+paramName])#amplitude for an iSWAP
        start += curr
        q1.z += env.rect(start, q1['swapTime'+paramName], q1['swapAmp'+paramName])#amplitude & length for an iSWAP
        start += q1['swapTime'+paramName]
        q0.z += eh.measurePulse(q0, start)
        q1.z += eh.measurePulse(q1, start)
        q0['readout'] = True
        q1['readout'] = True
        eh.correctCrosstalkZ(qubits)
        return runQubits(server, qubits, stats=stats)
    data = sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)
    if collect:
        return data

def rabiLinearity(sample, length=st.r[0:500:2,ns], amps=st.r[0:2:0.25], nfftpoints = 4001, measure=0,
                  name ='Rabi Frequency vs Amplitude', stats=1500L, plot=False, saveRabis=False, save=True):
    """Measures linearity of Rabi frequency vs amplitude.

    This function runs Rabis of different amplitudes and FFTs them to determine
    their frequency. The results are then plotted as a check of the linearity
    of Rabi frequencies vs amplitude.

    PARAMETERS
    sample: Object defining qubits to measure, loaded from registry.
    length - list of times: Duration of Rabis.
    amps - list of floats: Amplitudes of Rabis
    nfftpoints - int: Number of freqs to use in FFT.
    measure - int: Qubit number (as denoted in sample['config']).
    name - string: Title of plot and of dataset in data vault
    stats - scalar: Number of times a point will be measured.
    plot - bool: Whether to plot Freqs vs Amps.
    saveRabis - bool: Whether to save each Rabi to the data vault.
    save - bool: Whether to save plot of Freqs vs Amps to data vault.
    """
    rabifreqs = np.array([])
    for amp in amps:
        data = mq.rabilong(sample, length=length, amplitude=amp, measure=measure,
                           stats=stats, save=saveRabis, collect=True)
        rabifreq = fitting.maxFreq(data,nfftpoints)
        rabifreqs = np.append(rabifreqs,rabifreq)
    amps = np.array(amps)
    if plot:
        p = np.polyfit(amps, rabifreqs, 1)
        data = np.vstack((amps,rabifreqs)).T
        fig=dstools.plotDataset1D(data, [('Rabi Amps','')],[('Rabi Freqs.','','MHz')], style='.',
                                  markersize=20, legendLocation=None, show=False, title=name)
        fig.get_axes()[0].plot(amps, np.polyval(p,amps), 'r', linewidth=4)
        fig.show()
    if save:
        sample, qubits = util.loadQubits(sample)
        axes = [(amps, 'Rabi Amps')]
        deps = [('Rabi Freqs', '', 'MHz')]
        kw = {'stats': stats, 'lengthMin': min(length), 'lengthMax': max(length), 'lengthStep': length[1]-length[0]}
        dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)
        dataset.connect()
        dataset._create()
        for amp,rabifreq in zip(amps,rabifreqs):
            dataset.add([amp,rabifreq])


def uwaveCrosstalk(sample, drive, measure, length=st.r[0:1000:2,ns], amp=None,
               stats=1500L, name='Uwave Crosstalk', save=True, collect=False, noisy=True):
    """Measures microwave crosstalk. Drives Rabis on qubit DRIVE at frequency of
        qubit MEASURE and measures P1 for qubit MEASURE.

    rabiLinearity should be run first to determine the max Rabi amplitude (for maximum
    signal) while ensuring that Rabi frequencies vary linearly with amplitude.

    PARAMETERS
    sample: Object defining qubits to measure, loaded from registry.
    drive - int: Number of the qubit on which to drive Rabis (as denoted in
        sample['config']).
    measure - int: Number of the qubit on which to measure Rabis (as denoted in
        sample['config']).
    length - list of times: duration of Rabis
    amp - float: amplitude of Rabis
    stats - scalar: Number of times a point will be measured.
    name - string: Name of dataset.
    save - bool: Whether to save data to the datavault.
    collect - bool: Whether or not to return data.
    noisy - bool: Whether to print out probabilities while the scan runs
    """
    sample, qubits = util.loadDeviceType(sample,'phaseQubit')
    qD = qubits[drive]
    measure = sorted(measure)

    if amp is None: amp = qD['piAmp']
    fD = qD['f10']
    axes = [(length, 'pulse length')]
    deps = [('Probability','|'+sample['config'][numQubit]+'>','') for numQubit in measure]
    name = name + ' Drive |%s>' %sample['config'][drive]
    kw = {'stats': stats, 'amplitude':amp}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=measure, kw=kw)

    def func(server, len):
        reqs = []
        for measQubit in measure:
            for numQubit in measure:
                qubits[numQubit]['readout'] = False
                qubits[numQubit]['z'] = env.NOTHING
            qubits[measQubit]['readout'] = True
            qD['f10'] = qubits[measQubit]['f10']
            if measQubit == drive:
                qD['f10'] = fD
            qD.xy = eh.mix(qD, env.flattop(0, len, w=qD['piFWHM'], amp=amp))
            qubits[measQubit]['z'] = eh.measurePulse(qubits[measQubit], len+qD['piLen']/2.0)
            reqs.append(runQubits(server, qubits, stats, probs=[1]))
        probs = yield FutureList(reqs)
        problist = [p[0] for p in probs]
        returnValue(problist)
    data = sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)
    if collect:
        return data


def newUwaveCrosstalk(sample, drive, measure, length=st.r[0:1000:2,ns], amp=None, periodsCutoff=5, ampSteps = 20,expandAmp=0.25,
               nfftpoints=4000, stats=1500L, name='Uwave Crosstalk', save=True, collect=False, noisy=True):
    data = rabilong(sample, drive, measure, length=length, amp=amp,
                    stats=stats, name=name, save=True, collect=True, noisy=noisy)

    print 'Type in how many periods there are.'
    print 'Round to nearest integer.'
    print 'If there are no oscillations, type 0.'
    numPeriods = -1
    while (numPeriods == -1):
        try:
            numPeriods = int(raw_input('Enter an int: '))
        except ValueError:
            pass

    sample2, qubits = util.loadDeviceType(sample,'phaseQubit')
    qM = qubits[measure]
    qD = qubits[drive]
    if numPeriods > periodsCutoff:
        rabiFreq = fitting.maxFreq(data, nfftpoints, plot=plot)
        crosstalkAmp = 1000*qM['piAmp'] * (rabiFreq['MHz']*2.0*qM['piFWHM']['ns'])
        newRabi = mq.rabilong(sample, length=length, amplitude=crosstalkAmp, measure=measure,
                              stats=stats, save=save, collect=True, noisy=noisy)
        newRabiFreq = fitting.maxFreq(data, nfftpoints, plot=plot)
        crosstalk = newRabiFreq['MHz'] / rabiFreq['MHz']
    else:
        timeStep = max(length)['ns']/periodsCutoff/10.0
        lengthShort = st.r[(min(length)['ns']):(max(length)['ns']):timeStep,ns]
        maxAmp = (1+expandAmp)*qM['piAmp'] * (2.0 * qM['piFWHM']['ns']) / (max(length)['ns']/(numPeriods+1))
        if numPeriods>1:
            minAmp = (1-expandAmp)*qM['piAmp'] * (2.0 * qM['piFWHM']['ns']) / (max(length)['ns']/max(0,numPeriods-1))
        else:
            minAmp = 0.0
        ampStep = (maxAmp-minAmp) / ampSteps
        def rabiDist(amplitude,crosstalkData):
            rabiData = mq.rabilong(sample, length=lengthShort, amplitude=amplitude, measure=measure,
                                   stats=stats, save=save, collect=True, noisy=noisy)
            return np.sum((rabiData[:,1]-crosstalkData[:,1])**2)
        oldRabi = rabilong(sample, drive, measure, length=lengthShort, amp=amp,
                           stats=5*stats, name=name, save=save, collect=True, noisy=noisy)
        ampDists = []
        amps = st.r[minAmp:maxAmp:ampStep]
        for ampNum,ampMeas in enumerate(amps):
            ampDists.append(rabiDist(ampMeas,oldRabi))
            if noisy:
                print 'Amp %d complete' %ampNum
                print ampMeas,ampDists[-1]
        crosstalkAmp = amps[np.argmin(ampDists)]
        if amp is None:
            amp = qD['piAmp']
        crosstalk = crosstalkAmp/amp
        if noisy:
            print [ampVal for ampVal in amps]
            print ampDists
    print 'Microwave crosstalk between drive=%d and measure=%d is %f' %(drive,measure,crosstalk)
    return crosstalk





def rabilong(sample, drive, measure, length=st.r[0:1000:2,ns], amp=None,
               stats=1500L, name='Uwave Crosstalk', save=True, collect=False, noisy=True):
    """Measures microwave crosstalk. Drives Rabis on qubit DRIVE at frequency of
        qubit MEASURE and measures P1 for qubit MEASURE.

    rabiLinearity should be run first to determine the max Rabi amplitude (for maximum
    signal) while ensuring that Rabi frequencies vary linearly with amplitude.

    PARAMETERS
    sample: Object defining qubits to measure, loaded from registry.
    drive - int: Number of the qubit on which to drive Rabis (as denoted in
        sample['config']).
    measure - int: Number of the qubit on which to measure Rabis (as denoted in
        sample['config']).
    length - list of times: duration of Rabis
    amp - float: amplitude of Rabis
    stats - scalar: Number of times a point will be measured.
    name - string: Name of dataset.
    save - bool: Whether to save data to the datavault.
    collect - bool: Whether or not to return data.
    noisy - bool: Whether to print out probabilities while the scan runs
    """
    sample, qubits = util.loadDeviceType(sample,'phaseQubit')
    listQubits = range(len(sample['config']))
    qD = qubits[drive]
    qM = qubits[measure]
    qD['f10'] = qM['f10']

    if amp is None: amp = qD['piAmp']
    axes = [(length, 'pulse length')]
    name = name + ' Drive |%s>' %sample['config'][drive]
    kw = {'stats': stats, 'amplitude':amp, 'drive':drive}
    dataset = sweeps.prepDataset(sample, name, axes, measure=measure, kw=kw)

    def func(server, len):
        for numQubit in listQubits:
            qubits[numQubit]['z'] = env.NOTHING
        qM['readout'] = True
        qD.xy = eh.mix(qD, env.flattop(0, len, w=qD['piFWHM'], amp=amp))
        qM['z'] += eh.measurePulse(qM, len+qD['piLen']/2.0)
        return runQubits(server, qubits, stats, probs=[1])
    data = sweeps.grid(func, axes, dataset=save and dataset, collect=collect, noisy=noisy)
    if collect:
        return data

def measureCrosstalkTunnel(Sample, measure, mpas, stats=600, save=True,
                     name='Measurement Tunnel Crosstalk', collect=False, noisy=False):
    """Measures measurement crosstalk (tunneling), where tunneling one qubit
    causes the other to tunnel.

    PARAMETERS
    sample: Object defining qubits to measure, loaded from registry.
    measure - list of ints: Numbers of the qubits to measure (as denoted in
        sample['config']).
    mpas - list of floats: MPAs to force tunneling. Qubits in ascending order
        wrt placement in config. (i.e., config=['q0','q1','q2'], measure=[2,0],
        mpa=[1.5,0.5] gives 1.5 for 'q0' and 0.5 for 'q2').
    stats - scalar: Number of times a point will be measured.
    save - bool: Whether to save data to the datavault.
    name - string: Name of dataset.
    collect - bool: Whether or not to return data.
    noisy - bool: Whether to print out probabilities while the scan runs
    """

    sample,qubits,Qubits = util.loadDeviceType(Sample,'phaseQubit',write_access=True)
    measure = sorted(measure)
    axes = [(measure,'Tunnel Qubit')]
    deps = [('Probability','|'+sample['config'][numQubit]+'>','') for numQubit in measure]
    kw = {'stats':stats}
    dataset = sweeps.prepDataset(sample,name,axes,deps,measure=measure,kw=kw)

    def func(server,tunnelQubit):
        reqs = []
        for measQubit in measure:
            for numQubit in measure:
                qubits[numQubit]['readout'] = False
                qubits[numQubit]['z'] = env.NOTHING
            qubits[measQubit]['readout'] = True
            qt = qubits[tunnelQubit]
            qt['measureAmp'] = mpas[measure.index(tunnelQubit)]
            qt['readout'] = True
            if measQubit == tunnelQubit:
                probNum = 1
            else:
                probNum = 3
            reqs.append(runQubits(server, qubits, stats, probs=[probNum]))
            qt['z'] = eh.measurePulse(qt, 0*ns, state=1)
            reqs.append(runQubits(server, qubits, stats, probs=[probNum]))
        probs = yield FutureList(reqs)
        probdiff = [probs[2*q+1][0]-probs[2*q][0] for q in range(len(measure))]
        returnValue(probdiff)
    data = sweeps.grid(func,axes, dataset=save and dataset, collect=collect, noisy=noisy)
    if collect:
        return data


def measureCrosstalkExcite(Sample, tunnel, readout, mpaTunnel, delay=st.r[-150:150:10,ns], stats=30000,
                     correctZ=False, name='Measurement Xtalk Excite', save=True, collect=False, noisy=False):
    """Measures measurement crosstalk (excite), where tunneling one qubit
    causes the other to be excited.

    Applies large enough mpa to tunnel qubit TUNNEL. Wait time delay and then
    apply normal mpa to qubit READOUT. Measure prob READOUT excited.

    PARAMETERS
    sample: Object defining qubits to measure, loaded from registry.
    tunnel - int: Qubit to tunnel.
    readout - int: Qubit to readout (and determine if excited).
    mpaTunnel - float: MPA to force tunneling for the tunnel qubit.
    delay - iterable: Time to delay the second measure pulse wrt the first.
    stats - scalar: Number of times a point will be measured.
    save - bool: Whether to save data to the datavault.
    name - string: Name of dataset.
    collect - bool: Whether or not to return data.
    noisy - bool: Whether to print out probabilities while the scan runs
    """

    sample,qubits = util.loadDeviceType(Sample,'phaseQubit')
    qT,qR = qubits[tunnel],qubits[readout]
    measure = sorted([tunnel,readout])

    qT['measureAmp'] = mpaTunnel
    axes = [(delay,'Delay')]
    deps = [('Probability','|'+s+'>','') for s in ['g','e']]
    kw = {'stats':stats, 'tunnel': tunnel, 'correctZ':correctZ}
    dataset = sweeps.prepDataset(sample,name,axes,deps,measure=readout,kw=kw)
    def func(server,time):
        reqs = []
        qR['readout'] = True
        for q in qubits:
            q['z'] = env.NOTHING
        qR['z'] += eh.measurePulse(qR, 0*ns, state=1)
#        probNum = 1+int(readout<tunnel)
        if correctZ:
            eh.correctCrosstalkZ(qubits)
        reqs.append(runQubits(server, qubits, stats, probs=[1]))
        for q in qubits:
            q['z'] = env.NOTHING
        if time<0:
            t = -time
        else:
            t = 0*ns
        qT['z'] += eh.measurePulse(qT, t, state=1)
        qR['z'] += eh.measurePulse(qR, t+time, state=1)
        if correctZ:
            eh.correctCrosstalkZ(qubits)
        reqs.append(runQubits(server, qubits, stats, probs=[1]))
        probs = yield FutureList(reqs)
        probdiff = [p[0] for p in probs]
        returnValue(probdiff)
    data = sweeps.grid(func,axes, dataset=save and dataset, collect=collect, noisy=noisy)
    if collect:
        return data

def measureCrosstalkExciteMatrix(Sample, measure, mpa, delay=st.r[-150:150:10,ns], stats=3000, statsZero=30000, correctZ=False, calProbs=True,
                     saveMatrix=True, name='Measurement Xtalk Excite', save=True, collect=False, noisy=False):
    """Measures measurement crosstalk (excite), where tunneling one qubit
    causes the other to be excited.

    OUTPUTS:
    1) Matrix with max probs of exciting one qubit given other qubit tunneling.
    2) Matrix with prob of exciting qubit w/ real mpa if both qubits' measure
        pulses occur simultaneously.

    PARAMETERS
    Sample: Object defining qubits to measure, loaded from registry.
    measure - list of ints: Qubits to measure crosstalk between.
    mpa - float: MPAs to force tunneling (same order as for measure).
    delay - list of times: Time to delay the second measure pulse wrt the first.
    stats - scalar: Number of times a point will be measured.
    statsZero - scalar: Number of times a point will be measured for data where
        pulses at same time.
    correctZ - bool: Whether to apply zpa-crosstalk correction.
    saveMatrix - bool: Whether to save output matrices to data vault.
    name - string: Name of datasets.
    save - bool: Whether to save data to the datavault.
    collect - bool: Whether or not to return data.
    noisy - bool: Whether to print out probabilities while the scan runs
    """

    sample, qubits = util.loadDeviceType(Sample,'phaseQubit')
    maxQubit = max(measure)
    maxMatrix = np.zeros((maxQubit+1,maxQubit+1))
    crosstalkMatrix = np.zeros((maxQubit+1,maxQubit+1))
    for tunnelQubit, mpaTunnel in zip(measure,mpa):
        measureTunnel = [q for q in measure]
        measureTunnel.remove(tunnelQubit)
        for measureQubit in measureTunnel:
            if calProbs:
                scurveProbs = qubits[measureQubit]['calScurve1']
            data = measureCrosstalkExcite(sample, tunnelQubit, measureQubit, mpaTunnel, delay=delay,
                                                correctZ = correctZ, name=name,
                                                stats=stats, save = True, noisy=noisy, collect=True)
            data2 = measureCrosstalkExcite(sample, tunnelQubit, measureQubit, mpaTunnel, delay=0*ns,
                                                correctZ = correctZ, name=name,
                                                stats=stats, save = False, noisy=noisy, collect=True)
            maxElem = max(data[:,2]-data[:,1])
            crosstalkElem = data2[0,1]-data2[0,0]
            if calProbs:
                scurveProbs = qubits[measureQubit]['calScurve1']
                visFactor = 1/(float(scurveProbs[1])-float(scurveProbs[0]))
                maxElem *= visFactor
                crosstalkElem *= visFactor
            maxMatrix[tunnelQubit][measureQubit] = maxElem
            crosstalkMatrix[tunnelQubit][measureQubit] = crosstalkElem
    if saveMatrix:
        sample, qubits = util.loadQubits(sample)
        axes = [(measure, 'Tunnel Qubit'),(measure,'Measure Qubit')]
        deps = [('Measurement Crosstalk', '', 'MHz')]
        kw = {'stats': stats, 'correctZ': correctZ, 'minDelay': min(delay), 'maxDelay': max(delay), 'delayStep': delay[1]-delay[0]}
        dataset = sweeps.prepDataset(sample, name+' Matrix', axes, deps, measure=measure, kw=kw)
        dataset.connect()
        dataset._create()
        for tunnelQubit in measure:
            for measureQubit in measure:
                dataset.add([tunnelQubit,measureQubit,crosstalkMatrix[tunnelQubit][measureQubit]])
        axes2 = [(measure, 'Tunnel Qubit'),(measure,'Measure Qubit')]
        deps2 = [('Measurement Crosstalk', '', 'MHz')]
        kw2 = {'stats': stats, 'correctZ': correctZ, 'minDelay': min(delay), 'maxDelay': max(delay), 'delayStep': delay[1]-delay[0]}
        dataset2 = sweeps.prepDataset(sample, name+' Max Matrix', axes2, deps2, measure=measure, kw=kw2)
        dataset2.connect()
        dataset2._create()
        for tunnelQubit in measure:
            for measureQubit in measure:
                dataset2.add([tunnelQubit,measureQubit,maxMatrix[tunnelQubit][measureQubit]])
    if collect:
        return crosstalkMatrix,maxMatrix


def whichTunnel(measure,numTunnel):
    """Determines which qubits to tunnel based on decimal representation.

    PARAMETERS
    measure - list of ints: All qubits to be measured.
    numTunnel - int: Decimal representation of which qubits to tunnel. Here, 12=1100
        represents tunneling qubits 0 and 1 if measure=[0,1,2,3].
    """
    numQubits = len(measure)-1
    # For all elements in the binary representation of numTunnel, determine if they are '1'.
    #     This returns a list of bools, with True if the element is '1' and False otherwise.
    qubitsEqualTo1 = [index=='1' for index in bin(numTunnel)[2:]]
    # Return indices where the elements in the binary representation of numTunnel are '1'.
    location1s = np.where(np.flipud(np.array(qubitsEqualTo1)))[0]
    # Flip the order of the indices to convert into elements of measure. This is because the
    #     first qubit to be tunneled is the last qubit in measure (as the order is
    #     000->001->010->etc.).
    whichElems = numQubits - location1s
    # Convert these indices into the qubit numbers in config (pick out elements in measure).
    return [measure[elem] for elem in whichElems]


def identifyProbs(dim,whichProb):
    """Determines which probabilities contain the binary representation of whichProb
    in their own binary representation (e.g., 0101 contained within 1101 but not
    vice versa).

    PARAMETERS
    dim - int: Hilbert space dimension. Equals 2**len(measure).
    whichProb - int: Number whose binary representation all numbers returned should
        contain.
    """
    allProbs = np.array(range(dim))
    # Determine which numbers in range(dim) contain whichProb in their binary representation.
    #    This first does a bitwise and between each element and whichProb (done in binary)
    #    and then checks which of these ands equals whichProb
    containingWhichProb = [np.array([(num & whichProb)==whichProb for num in range(dim)])]
    #Convert the indices into the numbers of the probs. Returns all elements excluding the
    #    first (whichProb itself).
    return allProbs[containingWhichProb][1:]



def measureCrosstalkAll(Sample, measure, mpas, stats=30000, correctZ=True, save=True,
                        name='Measure Xtalk', collect=False, noisy=False):
    """Measures measurement crosstalk probabilities.

    Data saved where row is which qubits were tunneled (in binary). For instance,
    if len(measure)=4 and indep. varible (row) is 12=1100, 1st and 2nd qubits
    are tunneled. Probs returned in order 000,001,010,011,100,101,110,111.

    PARAMETERS
    Sample: Object defining qubits to measure, loaded from registry.
    measure - list of ints: Qubits to measure measurement crosstalk between.
    mpas - list of floats: MPAs to force tunneling.
    stats - scalar: Number of times a point will be measured.
    correctZ - bool: Whether to apply zpa-crosstalk correction.
    save - bool: Whether to save data to the datavault.
    name - string: Name of dataset.
    collect - bool: Whether or not to return data.
    noisy - bool: Whether to print out probabilities while the scan runs
    """

    sample,qubits = util.loadDeviceType(Sample,'phaseQubit')
    N = len(measure)
    tunnelProbs = range(2**N)
    mpasOld = [qubits[num]['measureAmp'] for num in measure]


    axes = [(tunnelProbs,'Tunneling Probs (in binary)')]
    deps = [('Probability','|'+s+'>', '') for s in [bin(i)[2:].rjust(N,'0') for i in xrange(2**N)]]
    kw = {'stats':stats, 'correctZ':correctZ, 'tunnelMpas': mpas}
    dataset = sweeps.prepDataset(sample,name,axes,deps,measure=measure,kw=kw)
    def func(server,whichTunnelProb):
        reqs = []
        # Determines which qubits to tunnel.
        qubitsTunnel = whichTunnel(measure,whichTunnelProb)
        for q in qubits:
            q['z'] = env.NOTHING
        for num,qubit in enumerate(measure):
            if qubit in qubitsTunnel:
                # Force these qubits to tunnel.
                qubits[qubit]['measureAmp'] = mpas[num]
            else:
                # Measure these qubits.
                qubits[qubit]['measureAmp'] = mpasOld[num]
            qubits[qubit]['z'] += eh.measurePulse(qubits[qubit], 20*ns, state=1)
            qubits[qubit]['readout'] = True
        if correctZ:
            eh.correctCrosstalkZ(qubits)
        return runQubits(server, qubits, stats)
    data = sweeps.grid(func,axes, dataset=save and dataset, collect=collect, noisy=noisy)
    if collect:
        return data

def measureCrosstalkMatrix(path, xtalkDatanum, correctVis=True, pairwise=False, calScurveKey = 'calScurve1'):
    """Gets measurement crosstalk matrix.

    Requires running measureCrosstalkAll on the qubits for which to build the
    crosstalk matrix first to get data for matrix. Output is the crosstalk matrix.

    PARAMETERS
    path: Location of crosstalk data from measureCrosstalkAll.
    xtalkDatanumt - ints: Dataset number for data from measureCrosstalkAll.
    correctVis - bool: Whether to correct raw probs for visibility.
    pairwise - bool: Whether to return only pairwise matrix. If False (default),
        returns full crosstalk matrix. If True, returns matrix with only pairwise
        elements (column - tunneled qubit, row - measured qubit)
    calScurveKey - string: Registry key containing visibilities.
    """
    with labrad.connect() as cxn:
        # Get crosstalk probs from data vault.
        dv = cxn.data_vault
        dataset = dstools.getDeviceDataset(dv,xtalkDatanum,path)
    data = dataset.data
    parameters = dataset.parameters
    measure = parameters['measure']
    dim = 2**len(measure)
    if correctVis:
        for row,probs in enumerate(data):
            # For each row (set of probs for given set of qubits that tunnel),
            #    create visibility matrix (Kronecker product of identity if
            #    qubit tunneled or visibility matrix if qubit not tunneled).
            #    Multiply inverse of visibility matrix by raw probs and save
            #    over the original data.
            sMatrices = []
            qubitsTunnel = whichTunnel(measure,row)
            for m in measure:
                if m in qubitsTunnel:
                    S = np.eye(2)
                else:
                    # Just like benchmarking.danBench.FMatrix.
                    qubit = parameters[parameters.config[m]]
                    s10 = qubit[calScurveKey][0]
                    s11 = qubit[calScurveKey][1]
                    S = np.array([[1-s10,1-s11],[s10,s11]])
                sMatrices.append(S)
            F = reduce(np.kron, sMatrices)
            Finv = np.linalg.inv(F)
            data[row] = np.hstack((probs[0],np.dot(Finv,probs[1:])))
    matrix = np.eye(dim)
    for col in range(1,dim-1):
        rows = identifyProbs(dim,col)
        for row in rows:
            # Create the elements of the crosstalk matrix.
            matrix[row][col] = data[col][row+1]
    lower = np.tril(matrix)
    # Normalize the crosstalk matrix (by varying the diagonal elements from 1)
    #    so that the sum of each column is 1.
    normMatrix = lower + np.diag(np.diag(matrix)-np.sum(lower,axis=0))
    if pairwise:
        dim = len(measure)
        pairMatrix = np.eye(dim)
        for row in range(dim):
            for col in range(dim):
                if row != col:
                    pairMatrix[row][col] = normMatrix[2**(dim-1-row)+2**(dim-1-col)][2**(dim-1-col)]
        returnMatrix = pairMatrix
    else:
        returnMatrix = normMatrix
    return returnMatrix


def stateDependentFreqShift(sample, control, measure, controlState = 1, measureState = 1,
                            delay=st.r[0:100:1,ns], fringeFreq = 50*MHz, nfftpoints=4000,
                            stats=2400L, name='State Dependent Freq Shift', save = True,
                            noisy=True, collect = False, plot=True):
    """Determine frequency shift on one qubit (via Ramsey) from excitation on
    a second qubit.

    PARAMETERS
    sample: object defining qubits to measure, loaded from registry
    control - int: Qubit to excite (or not) with pi-pulse.
    measure - int: Qubit on which to measure freq shift.
    controlState - int: State to excite control into.
    measureState - int: Which state's frequency for which to measure shift.
    delay - list of times: Delay times between pi/2 pulses.
    fringeFreq - value [Mhz]: Desired frequency of Ramsey fringes
    nfftpoints - int: Number of frequencies to use in FFT.
    stats - scalar: number of times a point will be measured.
    name - string: Name of dataset.
    save - bool: Whether to save data to the Data Vault.
    noisy - bool: Whether to print out probabilities while the scan runs.
    collect - bool: Whether to return data to the local scope.
    plot - bool: Whether to plot the FFT.
    """
    sample, qubits, Qubits = util.loadQubits(sample, write_access=True)
    qC = qubits[control]
    QC = Qubits[control]
    qM = qubits[measure]
    QM = Qubits[measure]
    if control == measure:
        raise Exception('Control and measure qubits must not be the same.')
    ml.setMultiKeys(qC,controlState) #Creates q['multiKeys']
    ml.setMultiKeys(qM,measureState) #Creates q['multiKeys']

    axes = [(delay, 'Delay')]
    deps = [('Probability', 'Control:NoPi,Measure:|1>', ''),('Probability', 'Control:|1>,Measure:|0>', ''),('Probability', 'Control:|1>,Measure:|1>', '')]

    kw = {'stats': stats, 'fringeFrequency': fringeFreq, 'controlQubit': control, 'measureQubit': measure}
    dataset = sweeps.prepDataset(sample, name, axes, deps, measure=sorted([control,measure]), kw=kw)

    def func(server, delay):
        reqs = []
        for controlPi in [False,True]:
            t = 0
            qC.xy = qC.z = env.NOTHING
            if controlPi:
                qC.xy += eh.boostState(qC, t, controlState)
                t += (controlState-0.5)*qC['piLen'] + 0.5*qM['piLen']
            qM.xy = eh.boostState(qM, t, measureState-1)
            t += (measureState-1)*qM['piLen']
            qM.xy += eh.mix(qM, eh.piHalfPulse(qM, t, phase=0.0, state=measureState), state=measureState)
            t += delay
            qM.xy += eh.mix(qM, eh.piHalfPulse(qM, t, phase = 2*np.pi*(fringeFreq['GHz']*delay['ns']), state=measureState), state=measureState)
            t += 0.5*qM['piLen']
            qM.z = eh.measurePulse(qM, t, state=measureState)
            if controlPi:
                qM['readout'] = True
                qC['readout'] = True
                qC.z += eh.measurePulse(qC, t, state=controlState)
                if control < measure:
                    probsReadout = [2,3]
                else:
                    probsReadout = [1,3]
            else:
                probsReadout = [1]
                qM['readout'] = True
                qC['readout'] = False
            reqs.append(runQubits(server, qubits, stats, probs=probsReadout))
        probs = yield FutureList(reqs)
        data = []
        for probset in probs:
            data += probset
        problist = [data[0],data[1],data[2]]
        returnValue(data)
    #Run the scan and save data
    data = sweeps.grid(func, axes, dataset = save and dataset, collect=True, noisy=noisy)
    controlGFreqShift = fitting.maxFreq(np.vstack([data[:,0],data[:,1]]).T, nfftpoints, plot=plot) - 50.0*MHz
    controlEFreqShift = fitting.maxFreq(np.vstack([data[:,0],data[:,3]/(data[:,2]+data[:,3])]).T, nfftpoints, plot=plot) - 50.0*MHz

    print 'Frequency shift, |Control = g>: %s' % controlGFreqShift
    print 'Frequency shift, |Control = e>: %s' % controlEFreqShift
    print 'Frequency shift due to Control Excitation: %s' % (controlEFreqShift-controlGFreqShift)

    if collect:
        return controlGFreqShift, controlEFreqShift, data
    else:
        return controlGFreqShift, controlEFreqShift, None


def stateDependentFreqShiftMatrix(sample, measure, controlState = 1, measureState = 1,
                            delay=st.r[0:100:1,ns], fringeFreq = 50*MHz, nfftpoints=4000,
                            stats=2400L, name='State Dep. Freq Shift', save = False,
                            noisy=False, collect = False, plot=True, saveMatrix = True):
    """Determine frequency shift matrix for all qubits, indicating frequency shift on
    one qubit (via Ramsey) from excitation on a second qubit.

    PARAMETERS
    sample: object defining qubits to measure, loaded from registry
    measure - list of ints: Qubits to measure freq shift between.
    controlState - int: State to excite control into.
    measureState - int: Which state's frequency for which to measure shift.
    delay - list of times: Delay times between pi/2 pulses.
    fringeFreq - value [Mhz]: Desired frequency of Ramsey fringes
    nfftpoints - int: Number of frequencies to use in FFT.
    stats - scalar: number of times a point will be measured.
    name - string: Name of dataset.
    save - bool: Whether to save data to the Data Vault.
    noisy - bool: Whether to print out probabilities while the scan runs.
    collect - bool: Whether to return data to the local scope.
    plot - bool: Whether to plot the FFT.
    saveMatrix - bool: Whether to save frequency shift matrix to Data Vault.
    """
    maxQubit = max(measure)
    freqShifts = np.zeros((maxQubit+1,maxQubit+1))
    for controlQubit in measure:
        measureControl = [q for q in measure]
        measureControl.remove(controlQubit)
        for measureQubit in measureControl:
            freqShift = stateDependentFreqShift(sample, controlQubit, measureQubit, controlState = controlState,
                                                measureState = measureState, delay=delay, fringeFreq=fringeFreq,
                                                nfftpoints=nfftpoints, name=name,
                                                stats=stats, save = True, noisy=noisy, collect = False, plot=False)
            freqShifts[controlQubit][measureQubit] = freqShift[1]-freqShift[0]
    if saveMatrix:
        sample, qubits = util.loadQubits(sample)
        axes = [(measure, 'Control Qubit'),(measure,'Measure Qubit')]
        deps = [('Frequency Shift', '|Control=e>-|Control=g>', 'MHz')]
        kw = {'stats': stats, 'fringeFrequency': fringeFreq}
        dataset = sweeps.prepDataset(sample, name+' Matrix', axes, deps, measure=measure, kw=kw)
        dataset.connect()
        dataset._create()
        for controlQubit in measure:
            for measureQubit in measure:
                dataset.add([controlQubit,measureQubit,freqShifts[controlQubit][measureQubit]])
    if collect:
        return freqShifts
