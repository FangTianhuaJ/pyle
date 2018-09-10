import numpy as np
from scipy import special
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import itertools
from collections import OrderedDict
from pyle.dataking import utilMultilevels as ml
from scipy.cluster import vq

###########################################
# The assumption here is that we use the ADC V7.
# If ADC V1 is used, first you should transform
# the data from V1 to V7
#
# In the demodulate mode:
# ADC V7: (demod channel, stats, retrigger, I/Q),
# e.g. (12, 3000, 3, 2)
# ADC V1: (demod channel, I/Q, stats)
# e.g. (4, 2, 3000)
#
# In the average mode:
# ADC V1 (board in time order, I/Q, waveform)
# e.g. (1, 2, 8192)
# ADC V7 (board in timg order, I/Q, waveform)
# e.g. (1, 2, 4096)
############################################


######## Data Processing #########
def parseDataFormat(data, dataFormat='iq'):
    if dataFormat == 'iqRaw':
        return data
    elif dataFormat == 'iq':
        return averageIQSingleDemod(data)

def averageStats(data):
    """ average along stats """
    return np.mean(data, axis=1)

def averageIQSingleDemod(data):
    """
    Returns average I/Q for single demod, n channel case.
    """
    data = averageStats(data)
    data = np.squeeze(data)
    return data

def iqToPolar(data):
    """ from IQ to mag and phase """
    I = data[..., 0]
    Q = data[..., 1]
    mag = np.sqrt(I**2+Q**2)
    phase = np.arctan2(Q, I)
    return mag, phase

def iqToZ(data):
    """ Convert IQ to complex Z """
    z = data[..., 0]  + 1.0j*data[..., 1]
    return z

def heraldStates(statesAllChannels, heraldState=0, noisy=False, warningRate=0.75):
    """
    herald the states, return only properly heralded data

    This function finds all stats in which the first demod
    found the desired heraldState for every channel, and returns the data
    for only those stats.

    This will reduce the size of your data set by approximately
    initializationFidelity**numChannels.

    @param statesAllChannels: the result of zsToStates (or iqToStates),
           in the shape of (nChannels, stats, retriggers)
    @param heraldState: int, the state we herald
    @param noisy: if True, herald information will be printed
    @param warningRate: the rate the herald process is safe.
    @return: heralded data, index of heralded data
    """
    stats = statesAllChannels.shape[1]
    nDemods = statesAllChannels.shape[2]
    assert nDemods > 1, "You Need more than 1 demod to herald"

    idxOneChannels = [np.arange(stats, dtype=int)]
    for statesOneChannel in statesAllChannels:
        # statesOneChannel in the shape (stats, retriggers)
        # we use the first demod as herald
        idxOneChannel = np.where(statesOneChannel[..., 0]==heraldState)[0]
        idxOneChannels.append(idxOneChannel)
    idxFullHerald = reduce(np.intersect1d, idxOneChannels)

    successRate = len(idxFullHerald)/float(stats)

    if noisy:
        if successRate >= warningRate:
            print("(readout) Herald success rate: %s" %successRate)
        else:
            print("Warning, (readout) Herald success rate %s, smaller than %s" %(successRate, warningRate))
    elif successRate < warningRate:
            print("Warning, (readout) Herald success rate %s, smaller than %s" %(successRate, warningRate))

    statesAllChannelsHerald = statesAllChannels[:, idxFullHerald, 1:]
    return statesAllChannelsHerald, idxFullHerald

def iqToReadoutCenter(IQs):
    """
    calculate the center of IQ
    @param IQs: the last axis of IQ is for I and Q
    @return: center (complex value, I+1j*Q), mask, std
    """
    Is, Qs = np.squeeze(IQs[..., 0]), np.squeeze(IQs[..., 1])
    Zs = Is + 1j*Qs
    zmid = np.median(Is) + 1j*np.median(Qs)
    std = np.std(Zs)
    mask = np.abs(Zs-zmid) <= std
    center = np.mean(Zs[mask])
    state_std = np.std(Zs[mask])
    return center, mask, state_std

def iqToReadoutAxis(IQ0, IQ1):
    """
    calculate the axis of IQ
    @param IQ0: IQ data for lower state,
    @param IQ1: IQ data for higher state
    @return: axis, a complex value with radial=1
    """
    center0, _, std0 = iqToReadoutCenter(IQ0)
    center1, _, std1 = iqToReadoutCenter(IQ1)
    axis = center1 - center0
    axis /= np.abs(axis)
    return axis

##############################################################################
# two methods can be used to determine the state according to IQ
# 1. find the minial distance to some specifical points, and use it as criteria of different states.
#    this method can distinguish two or more states.
# 2. IQ scatter rotate an angle, and find a seperation point. the right side and the left
#    side represent two states. In this method, only two states can be distinguished.
##############################################################################


#######################################################
#######  functions for method 1
#######  all the functions for method 1 are in class RM1
#######################################################

class RM1(object):
    """
    functions for Readout Method 1,
    for not confusing namespace
    In this method, two registry keys are used: "readoutCenters", "readoutCenterStates"
    "readoutCenters", a list of complex value I+1j*Q, is used to record the centers of
        IQ of specific states.
    "readoutCenterStates", a list of states to record the states in the readoutCenters.
    e.g.  q["readoutCenters"] = [0+0j, 1+1j, 2+2j]
        q["readoutCenterStates"] = [0, 1, 2]  represents |0>, |1>, |2> states
    """
    @staticmethod
    def genCentersList(qubits, states=None):
        """
        generate center list together with the state
        @param qubits: qubits should contain readoutOrder
        @param states: states need be distinguished, default is None, representing all states in the registry
        @return: the center list of the qubit measured in the config order
            the centers is a dict like {0: I0+1j*Q0, 1: I1+1j*Q1, ...}
            centers is an OrderedDict (from collections) in the ascending order of the states
        """
        if not isinstance(qubits, list):
            qubits = [qubits]

        tmp_qubits = [q for q in qubits if 'readoutOrder' in q] # filter unread qubits or readoutDevices
        centersList = [None for i in range(len(tmp_qubits))]
        for q in tmp_qubits:
            idx = q['readoutOrder'] # readoutOrder represent the order in config
            center = OrderedDict()
            readoutCentersUnsorted = q['readoutCenters']
            readoutCenterStatesUnsorted = q['readoutCenterStates']
            arg = np.argsort(q['readoutCenterStates'])
            readoutCenters = [readoutCentersUnsorted[i] for i in arg]
            readoutCenterStates = [readoutCenterStatesUnsorted[i] for i in arg]
            # readoutCenters is a complex list, e.g. [I0+1j*Q0, I1+1j*Q1, ...]
            for rc, rcs in zip(readoutCenters, readoutCenterStates):
                if states:
                    if rcs in states:
                        center[rcs] = rc
                else:
                    center[rcs] = rc
            centersList[idx] = center

        return centersList

    @staticmethod
    def zsToStates(zs, centersList):
        """
        from zs to states

        centers is a dict: {0: z0, 1: z1, ...}, z0 = I0 + 1j*Q0, represents the center,
            the key of the center represents the state
        zs, centerList is given in the order of config

        @return states, same shape with zs
        """
        if zs.shape[0] != len(centersList):
            raise RuntimeError("zs has the %s channels, while centersList has %s channels"
                               %(zs.shape[0], len(centersList)))
        results = []
        for zThisChannel, centersThisChannel in zip(zs, centersList):
            # the shape of zThisChannel is (stats, retrigger)
            states, centers = zip(*centersThisChannel.items())
            distanceArray = np.array([np.abs(zThisChannel-center) for center in centers])
            # now the shape of distanceArray is ( len(centers), stats, retrigger )
            distanceIdx = np.argmin(distanceArray, axis=0)
            thisChannelStates = np.array(states, dtype=int)[distanceIdx]
            results.append(thisChannelStates)
        results = np.array(results)

        return results

    @staticmethod
    def iqToStates(data, qubits, states=None, herald=False, noisy=False, centersList=None):
        if centersList is None:
            centersList = RM1.genCentersList(qubits, states)
        zs = iqToZ(data)
        states = RM1.zsToStates(zs, centersList)
        if herald:
            states, idxFullHerald = heraldStates(states, heraldState=0, noisy=noisy)
        return states

    @staticmethod
    def centersListToMQStatesDict(centersList):
        """
        centersList is for multiple qubits
        we need dict[(1,0,0)] = idx
        and inverseDict[idx] = (1,0,0)
        @return: dict and inverseDict
        """
        stateToIdx = {}
        idxToState = {}

        states = [c.keys() for c in centersList]
        mqStates = itertools.product(*states)
        for idx, state in enumerate(mqStates):
            stateToIdx[state] = idx
            idxToState[idx] = state

        return stateToIdx, idxToState

    @staticmethod
    def statesToProbs(states, centersList, correlated=False):
        """
        from states to probs,
        if correlated, the probs is correlated for multiple qubits, P(000), P(001), ...

        @param states, may be the result of zsToStates, the shape is ( nChannel, stats, nDemod)
        @param centersList, may be the result of genCentersList, the centers should be in the ascending order.
        @param correlated, bool value, if false, the probs is for each qubit along.
        @return if correlated, probs is in the shape (nDemod, nStates)
                otherwise, probs is in the shape (nChannel, nDemod, nStates)
        """

        mqStatesKeys = [centers.keys() for centers in centersList]

        if correlated:
            # the states is in the shape (nChannel (or n qubits) , stats, nDemod)
            # the shape of output probs should be ( N, nDemod )
            # N presents the number of all the possible states in these qubits, e.g. |000>, |001>, ...
            stateToIdx, idxToState = RM1.centersListToMQStatesDict(centersList)
            nStates = len(stateToIdx.keys())
            nChannel, stats, nDemod = states.shape
            mqProbs = np.zeros((nStates, nDemod), dtype=float)
            for sidx in range(stats):
                for nidx in range(nDemod):
                    # tolist here is for speed, do not remove it
                    idx = stateToIdx[tuple(states[:, sidx, nidx].tolist())]
                    mqProbs[idx, nidx] += 1
            # now we have calculated the number of each state
            # and then we get probs
            mqProbs /= float(stats)
            probs = mqProbs.T
            # the follow code may be faster for a few number of qubit (below 6)
            # but not too much when stats is not very large (about x0.5 faster)
            # for about 9 or more qubits, the follow coding is much slower
            # mqStates = np.zeros((stats, nDemod), dtype=int)
            # for sidx in range(states):
            #     for nidx in range(nDemod):
            #         idx = stateToIdx[tuple(states[:, sidx, nidx].tolist())]
            #         mqStates[sidx, nidx] = idx
            # keys = sorted(idxToState.keys())
            # for demod in mqStates.T:
            #     subProbs = np.array([np.mean(demod==idx) for idx in keys])
            #     probs.append(subProbs)
            # probs = np.array(probs)
        else:
            probs = []
            for channelStates, channelStateKeys in zip(states, mqStatesKeys):
                # channelStates in the shape (stats, nDemod)
                stats, nDemod = channelStates.shape
                nStates = len(channelStateKeys)
                sqProbs = np.zeros((nStates, nDemod), dtype=float)
                for sidx in range(stats):
                    for nidx in range(nDemod):
                        idx = channelStates[sidx, nidx]
                        sqProbs[idx, nidx] += 1
                sqProbs /= float(stats)
                probs.append(sqProbs.T)
            probs = np.array(probs)

        return probs

    @staticmethod
    def iqToProbs(data, qubits, centersList=None, herald=False, correlated=False,
                  noisy=False, states=None):
        """
        from iq to probs, if correlated, return correlated probs, e.g. P(000), P(001) .....

        @param data: data is the raw data from ADC, the shape of data should be (nChannel, stats, nDemod)
        @param qubits: list of directory for qubit, the order should be the same as config
        @param centersList: default is None, see genCentersList for the details of the format of centersList
        @param herald: bool, herald qubit state or not
        @param correlated: bool, the probs is correlated or not
        @param noisy: bool, if noisy, herald information will be printed
        @param states: list of state, the states need be measured, default is None, representing all the
                       states listed in the registry key "readoutCenters", "readoutCenterStates" will be
                       taken into considerasion. You can pass states=[0,1] to ensure only |0> or |1> can
                       be distinguished from the IQ data.
        @return: probs,
                if correlated, the shape of probs is (nDemod, nStates) for not heralding states
                and (nDemod-1, nStates) for heralding states
                if not correlated, the shape of probs is (nChannel, nDemod, nStates) for not heralding states
                and (nChannel, nDemod-1, nStates) for heralding states.

        """
        if centersList is None:
            centersList = RM1.genCentersList(qubits, states=states)
        zs = iqToZ(data)
        states = RM1.zsToStates(zs, centersList)
        if herald:
            states, idxFullHerald = heraldStates(states, heraldState=0, noisy=noisy)
        probs = RM1.statesToProbs(states, centersList, correlated=correlated)
        return probs

    @staticmethod
    def genProbDeps(qubits, measure, correlated=False, states=None, centersList=None):
        """
        generate dependence for saving data to data vault
        @param qubits: list of directory for qubits, should be given in the order of config
        @param measure: list of index for the qubits measured
        @param correlated: the probs is correlated for mulitiple qubits or not
        @param states: list of the states we want to get from the IQ data
        @param centersList: default is None, see genCentersList for the details of the centersList
        @return: deps for saving data to datavault
        """

        if centersList is None:
            centersList = RM1.genCentersList(qubits, states)

        if correlated:
            stateToIdx, idxToState = RM1.centersListToMQStatesDict(centersList)
            index = range(len(stateToIdx.keys()))
            mqStates = [idxToState[idx] for idx in index]
            # the elements of deps: (name, label, unit)
            deps = [("Probability |%s>" %("".join(str(elem) for elem in state)), "",  "")
                    for state in mqStates]
        else:
            if not isinstance(measure, (list, tuple)) and (measure is not None):
                measure = [measure]

            sqStates = [centers.keys() for centers in centersList]
            name = lambda q: q.__name__
            qNames = [name(qubits[idx]) for idx in sorted(measure)]

            deps = []
            for qName, sqState in zip(qNames, sqStates):
                deps.extend( [ ("Probability |%s>" %(str(state)), str(qName), "") for state in sqState ])

        return deps

    @staticmethod
    def iqToReadoutFidelity(IQsList, states, centers=None, k_means=False):
        """
        @param IQsList: a list of IQsList, each IQ is for one states,
                the shape of each IQ should be (stats, 2)
        @param states: the states for IQsList
        @param centers: centers of IQ for different states
        @param k_means: use k-means method to generate centers (more accurate), default is False.
        @return: fids, probs, centers, stds
        """
        stds = OrderedDict()
        if centers is None:
            centers = OrderedDict()
            for IQs, state in zip(IQsList, states):
                center, mask, std = iqToReadoutCenter(IQs)
                centers[state] = center
                stds[state] = std
            if k_means:
                k_guess = np.array([[c.real, c.imag] for k, c in centers.items()])
                new_centers = vq.kmeans(np.vstack(IQsList).astype('float'), k_guess)[0]
                # convert to complex form
                k_guess = k_guess[:, 0] + 1j * k_guess[:, 1]
                new_centers = new_centers[:, 0] + 1j * new_centers[:, 1]
                # the center returned by k-means may not be in the order
                # we should determine the order of the center according to our origin center
                # by computing the distance.
                order = []
                for kg in k_guess:
                    dis_arr = np.abs(kg - new_centers)
                    order.append(np.argmin(dis_arr, axis=0))
                new_centers = new_centers[order]
                for center, state in zip(new_centers, states):
                    centers[state] = center

        centersList = [centers]
        probs = [RM1.iqToProbs(IQs.reshape((1, -1, 1, 2)), [], centersList) for IQs in IQsList]
        fids = [np.squeeze(prob)[state] for prob, state in zip(probs, states)]
        return fids, np.squeeze(probs), centers, stds


#######################################################
#######  functions for method 2
#######  all the functions for method 2 are in class RM2
#######################################################

class RM2(object):
    """
    functions for Readout Method 2,
    for not confusing namespace
    In this method, two registry keys are used: "readoutCenterZ", "readoutAxis"
    for readout different states, e.g. |2>, "readoutCenterZ2", "readoutAxis2" are used
    similar keys are used for |3>, "readoutCenterZ3", "readoutAxis3", and so on.
    """

    @staticmethod
    def genCenterAxisList(qubits, measure=None, states=None):
        """
        generate a list of (center, axis)
        @param qubits: qubits should be given in the order of config!!!
        @param measure: the index or a list of index the qubit(s) measured, if None is given,
                       all the qubits is measured.
        @param states: a list of the states need to be measured. if not None,
                       The order of states is the same with measure, and the length of states
                       is the same with measure. e.g. states = [1, 1, 2] is for three qubits.
                       the third qubit measured |2>, and the first and the seconds measured |1>,
                       if None is given, states = [1]*len(measure)
        @return: list of (center, axis) pair in the order of config
        """
        if not isinstance(qubits, list):
            qubits = [qubits]
        if measure is None:
            measure = range(len(qubits))
        if not isinstance(measure, (list, tuple)):
            measure = [measure]
        if states is None:
            states = [1]*len(measure)
        assert len(states) == len(measure)

        # keep the order of config, not the order of measure
        arg = np.argsort(measure)
        measure = [measure[i] for i in arg]
        states = [states[i] for i in arg]

        centerAxisList = []
        for m, state in zip(measure, states):
            q = qubits[m]
            center = ml.getMultiLevels(q, 'readoutCenterZ', state)
            axis = ml.getMultiLevels(q, 'readoutAxis', state)
            centerAxisList.append((center, axis))

        return centerAxisList

    @staticmethod
    def zsToStates(zs, centerAxisList):
        """
        from zs to states, zs, centerList is given in the order of config

        @param zs, I+1j*Q the shape should be (nChannel, stats, nDemod)
        @param centerAxisList: a list of (center, axis) pair
        @return states, same shape with zs
        """
        if zs.shape[0] != len(centerAxisList):
            raise RuntimeError("zs has the %s channels, while centerAxisList has %s channels"
                               %(zs.shape[0], len(centerAxisList)))
        results = []
        for zThisChannel, centerAxisThisChannel in zip(zs, centerAxisList):
            # the shape of zThisChannel is (stats, retrigger)
            center, axis =  centerAxisThisChannel
            thisChannelStates = RM2.singleQubitIqToState(zThisChannel, center, axis)
            # zRotated = (zThisChannel - center) * np.conj(axis)
            # thisChannelStates = np.array(zRotated.real>0, dtype=int)
            results.append(thisChannelStates)
        results = np.array(results)

        return results

    @staticmethod
    def singleQubitIqToState(zs, center, axis):
        """
        from zs to state, with the input of readoutCenter and readoutAxis
        @param zs: represent IQ, z=I+1j*Q, the shape of z may be (stats, nDemod)
        @param center: complex value
        @param axis: complex value with radial=1
        @return:
        """
        zRotated = (zs - center) * np.conj(axis)
        return np.array(zRotated.real>0, dtype=int)

    @staticmethod
    def iqToStates(data, qubits, measure, states=None, herald=False,
                   noisy=False, centerAxisList=None):
        """
        from IQ data to states, the states here are for each qubit

        @param data: data is the raw data from ADC, the shape of data should be (nChannel, stats, nDemod)
        @param qubits: list of directory for qubit, the order should be the same as config
        @param measure: list of idx for the qubits measured
        @param states: a list of the states need to be measured. if not None,
                       The order of states is the same with measure, and the length of states
                       is the same with measure. e.g. states = [1, 1, 2] is for three qubits.
                       the third qubit measured |2>, and the first and the seconds measured |1>,
                       if None is given, states = [1]*len(measure)
        @param herald: bool, herald qubit state or not
        @param noisy: bool, if noisy, herald information will be printed
        @param centerAxisList: default is None, see genCentersList for the details of the format of centersList
        @return the states for each qubit, each experiment.
                the shape is (nChannel, stats, nDemod), the same as the input data if not herald
        """
        if centerAxisList is None:
            centerAxisList = RM2.genCenterAxisList(qubits, measure, states)
        zs = iqToZ(data)
        states = RM2.zsToStates(zs, centerAxisList)
        if herald:
            states, idxFullHerald = heraldStates(states, heraldState=0, noisy=noisy)
        return states

    @staticmethod
    def mqStatesDict(Nqubits):
        """
        we need dict[(1,0,0)] = idx
        and inverseDict[idx] = (1,0,0)
        @param Nqubits, the number of qubits
        @return: dict and inverseDict
        """
        stateToIdx = OrderedDict()
        idxToState = OrderedDict()

        mqStates = itertools.product((0,1), repeat=Nqubits)
        for idx, state in enumerate(mqStates):
            stateToIdx[state] = idx
            idxToState[idx] = state

        return stateToIdx, idxToState

    @staticmethod
    def statesToProbs(states, centerAxisList, correlated=False):
        """
        from states to probs,
        if correlated, the probs is correlated for multiple qubits, P(000), P(001), ...

        @param states, may be the result of zsToStates, the shape is ( nChannel, stats, nDemod)
        @param centerAxisList, may be the result of genCenterAxisList
        @param correlated, bool value, if false, the probs is for each qubit along.
        @return if correlated, probs is in the shape (nDemod, nStates)
                otherwise, probs is in the shape (nChannel, nDemod, nStates)
        """
        assert states.shape[0] == len(centerAxisList)

        mqStatesKeys = [ (0,1) ]*len(centerAxisList)

        if correlated:
            # the states is in the shape (nChannel (or n qubits) , stats, nDemod)
            # the shape of output probs should be ( N, nDemod )
            # N presents the number of all the possible states in these qubits, e.g. |000>, |001>, ...
            stateToIdx, idxToState = RM2.mqStatesDict(len(centerAxisList))
            nStates = len(stateToIdx.keys())
            nChannel, stats, nDemod = states.shape
            mqProbs = np.zeros((nStates, nDemod), dtype=float)
            for sidx in range(stats):
                for nidx in range(nDemod):
                    # tolist here is for speed, do not remove it
                    idx = stateToIdx[tuple(states[:, sidx, nidx].tolist())]
                    mqProbs[idx, nidx] += 1
            # now we have calculated the number of each state
            # and then we get probs
            mqProbs /= float(stats)
            probs = mqProbs.T
        else:
            probs = []
            for channelStates, channelStateKeys in zip(states, mqStatesKeys):
                # channelStates in the shape (stats, nDemod)
                stats, nDemod = channelStates.shape
                nStates = len(channelStateKeys)
                sqProbs = np.zeros((nStates, nDemod), dtype=float)
                # channelStates here must be in channelStateKeys
                # then we just sum it along axis0 (stats) to get prob
                sqProbs = np.array([np.sum(channelStates==idx, axis=0) for idx in channelStateKeys])
                sqProbs /= float(stats)
                probs.append(sqProbs.T)
            probs = np.array(probs)

        return probs

    @staticmethod
    def iqToProbs(data, qubits, measure, centerAxisList=None, herald=False, correlated=False,
                  noisy=False, states=None):
        """
        from iq to probs, if correlated, return correlated probs, e.g. P(000), P(001) .....

        @param data: data is the raw data from ADC, the shape of data should be (nChannel, stats, nDemod)
        @param qubits: list of directory for qubit, the order should be the same as config
        @param measure: list of idx for the qubits measured
        @param centerAxisList: default is None, see genCentersList for the details of the format of centersList
        @param herald: bool, herald qubit state or not
        @param correlated: bool, the probs is correlated or not
        @param noisy: bool, if noisy, herald information will be printed
        @param states: list of state, the states need be measured, default is None, representing all the
                       states listed in the registry key "readoutCenters", "readoutCenterStates" will be
                       taken into considerasion. You can pass states=[0,1] to ensure only |0> or |1> can
                       be distinguished from the IQ data.
        @return: probs,
                if correlated, the shape of probs is (nDemod, nStates) for not heralding states
                and (nDemod-1, nStates) for heralding states
                if not correlated, the shape of probs is (nChannel, nDemod, nStates) for not heralding states
                and (nChannel, nDemod-1, nStates) for heralding states.

        """
        if centerAxisList is None:
            centerAxisList = RM2.genCenterAxisList(qubits, measure, states=states)
        zs = iqToZ(data)
        states = RM2.zsToStates(zs, centerAxisList)
        if herald:
            states, idxFullHerald = heraldStates(states, heraldState=0, noisy=noisy)
        probs = RM2.statesToProbs(states, centerAxisList, correlated=correlated)
        return probs

    @staticmethod
    def genProbDeps(qubits, measure, correlated=False, Nqubits=None):
        """
        generate dependence for saving data to data vault
        @param qubits: list of directory for qubits, should be given in the order of config
        @param measure: list of index for the qubits measured
        @param correlated: the probs is correlated for mulitiple qubits or not
        @return: deps for saving data to datavault
        """
        if measure is None:
            measure = range(qubits)
        if not isinstance(measure, (list, tuple)):
            measure = [measure]
        if Nqubits is None:
            Nqubits = len(measure)

        if correlated:
            stateToIdx, idxToState = RM2.mqStatesDict(Nqubits)
            index = range(len(stateToIdx.keys()))
            mqStates = [idxToState[idx] for idx in index]
            # the elements of deps: (name, label, unit)
            deps = [("Probability |%s>" %("".join(str(elem) for elem in state)), "",  "")
                    for state in mqStates]
        else:
            sqStates = [(0,1)] * Nqubits
            name = lambda q: q.__name__
            qNames = [name(qubits[idx]) for idx in sorted(measure)]

            deps = []
            for qName, sqState in zip(qNames, sqStates):
                deps.extend( [ ("Probability |%s>" %(str(state)), str(qName), "") for state in sqState ])

        return deps

    @staticmethod
    def iqToReadoutFidelity(IQsList, states, centers=None):
        """
        @param IQsList: a list of IQsList, each IQs is for one states
        @param states: the states for IQsList
        @param centers: centers of IQ for different states
        @return: fids, probs, centers
        """
        raise NotImplementedError
        if centers is None:
            centers = {}
            stds = {}
            for IQs, state in zip(IQsList, states):
                center, mask, std = iqToReadoutCenter(IQs)
                centers[state] = center
                stds[state] = std

        centersList = [centers]
        probs = [RM2.iqToProbs(IQs, [], [], centersList) for IQs in IQsList]
        fids = [np.squeeze(prob)[state] for prob, state in zip(probs, states)]

        return fids, probs, centers

    @staticmethod
    def singleQubitIqToCDF(IQ, q, state=1):
        """
        from IQ data to cumulative distribution,
        we shift and rotate the IQ data, according to the readoutCenterZ and readoutAxis
        @param IQ: IQ data, the shape (stats, 2)
        @param q: qubit dict
        @param state: the state to discrimination, default is 1
        @return: cumulative distribution function (CDF)
        """
        axis = ml.getMultiLevels(q, key='readoutAxis', state=state)
        center = ml.getMultiLevels(q, key='readoutCenterZ', state=state)
        return iqToCDF(IQ, center, axis)


def iqToCDF(IQ, center, axis):
    """
    from IQ data to cumulative distribution function
    @param IQ: IQ data, the shape is (stats, 2)
    @param center: the center of the shift
    @param axis: rotation axis
    @return: CDF
    """
    N = IQ.shape[0]
    z = iqToZ(IQ)
    zR = (z-center)*np.conj(axis)
    p = 1.0 * np.arange(N)/(N-1)
    zR = np.sort(zR.real)
    cdf_func = interp1d(zR, p, bounds_error=False, fill_value=(0.0, 1.0), assume_sorted=True)
    return cdf_func


def iqDiscrimination(IQ0, IQ1):
    """
    from the input IQs, calculate the readoutAxis and readoutCenterZ

    From IQ0 and IQ1, we determine the readoutAxis, and the CDF can be calculated.
    We find the maximum value of the different of CDF between the two states.
    @param IQ0: IQ data for lower state
    @param IQ1: IQ data for higher state
    @return: center, axis, visibity, fidelity lower state, fidelity higher state
    """
    # 1. get axis
    axis = iqToReadoutAxis(IQ0, IQ1)

    # 2. calculaate the center
    cdf0_func = iqToCDF(IQ0, 0+0j, axis)
    cdf1_func = iqToCDF(IQ1, 0+0j, axis)
    z0 = iqToZ(IQ0)
    z1 = iqToZ(IQ1)
    z0r = z0*np.conj(axis)
    x0_min = np.min(z0r.real)
    x0_max = np.max(z0r.real)
    z1r = z1*np.conj(axis)
    x1_min = np.min(z1r.real)
    x1_max = np.max(z1r.real)
    xmin = min([x0_min, x1_min])
    xmax = max([x0_max, x1_max])
    N = IQ0.shape[0] + IQ1.shape[0]
    x = np.linspace(xmin, xmax, N)
    cdf0 = cdf0_func(x)
    cdf1 = cdf1_func(x)
    idx = np.argmax((np.abs(cdf0-cdf1)))
    xc = x[idx]
    center = xc*axis

    # 3. calculate fidelity
    F0 = cdf0[idx]
    F1 = 1 - cdf1[idx]
    S10 = cdf0[idx] - cdf1[idx]

    return center, axis, S10, F0, F1

# we use readout method 1 for default
genCentersList = RM1.genCentersList
zsToStates = RM1.zsToStates
iqToStates = RM1.iqToStates
centersListToMQStatesDict = RM1.centersListToMQStatesDict
statesToProbs = RM1.statesToProbs
iqToProbs = RM1.iqToProbs
genProbDeps = RM1.genProbDeps
iqToReadoutFidelity = RM1.iqToReadoutFidelity
