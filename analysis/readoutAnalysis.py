# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.cluster import vq

from pyle.analysis import readout
from pyle.plotting import dstools

def readoutHeraldAnalysis(dataset):

    parameters = dataset.parameters
    stats = parameters['stats']
    qName = dstools.getMeasureNames(dataset)[0]
    qubit = parameters[qName]
    centersList = readout.genCentersList([qubit], states=[0,1])

    # the dataformat, see readoutHerald in singleQubitTransmon
    r0 = np.array(dataset[:, (1,2,3,4)])
    r1 = np.array(dataset[:, (5,6,7,8)])
    r0 = r0.reshape(1, -1, 2, 2) # (nChannel, stats, nTrigger, I/Q)
    r1 = r1.reshape(1, -1, 2, 2) # (nChannel, stats, nTrigger, I/Q)

    zs = readout.iqToZ(r0)
    states = readout.zsToStates(zs, centersList)
    heraldStates, allChannelIndices = readout.heraldStates(states, heraldState=0)
    probs0 = readout.statesToProbs(heraldStates, centersList)
    zsFirst0 = zs[0, :, 0]
    zsSecond0 = zs[0, :, 1]
    zsHerald0 = zs[0, allChannelIndices, 1]
    heraldRate0 = (heraldStates.shape[1] / float(stats))

    print("Preparing state |0> state")
    print("Avg pops for first measurement: %s" %(np.mean(states[0, :, 0])))
    print("Avg pops for second measurement: %s" %(np.mean(states[0, :, 1])))
    print("Heralding Success rate: %s" %(heraldStates.shape[1] / float(stats)))
    print("Heralded probs for second measurement: %s" %np.squeeze(probs0))
    print("")


    zs = readout.iqToZ(r1)
    states = readout.zsToStates(zs, centersList)
    heraldedStates, allChannelIndices = readout.heraldStates(states, heraldState=0)
    probs1 = readout.statesToProbs(heraldedStates, centersList)
    zsFirst1 = zs[0, :, 0]
    zsSecond1 = zs[0, :, 1]
    zsHeralded0Prepare1 = zs[0, allChannelIndices, 1]
    heraldRate1 = (heraldedStates.shape[1] / float(stats))

    print("Preparing state |1> state")
    print("Avg pops for first measurement: %s" % np.mean(states[0, :, 0]))
    print("Avg pops for second measurement: %s" % np.mean(states[0, :, 1]))
    print("Heralding Success rate: %s" % (heraldedStates.shape[1] / float(stats)))
    print("Heralded probs for second measurement: %s" % np.squeeze(probs1))
    print("")

    p00 = np.squeeze(probs0)[0] # P(0|0)
    p10 = np.squeeze(probs1)[1] # P(1|0)
    p0_0 = (heraldRate0 + heraldRate1) * 0.5

    print("Raw Readout Fidelity for |0>, F0 : %s" %p00)
    print("Raw Readout Fidelity for |1>, F1 : %s" %p10)
    print("Raw Avg Readout Fidelity: %s" %(0.5*(p00+p10)))
    print("")


    # the following analysis is taken the state preparation into account
    # by Bayes' theorem
    def err_func(x, y):
        """
        x = (p, F0, F1)
        y = (P(0), P(0|0), P(1|0))
        """
        p, F0, F1 = x
        p0, p00, p10 = y
        a = p0
        b = p00*p0
        c = p10*p0
        e1 = (p*F0 + (1-p)*(1-F1) - a)**2
        e2 = (p*F0*F0 + (1-p)*(1-F1)*(1-F1) - b)**2
        e3 = (p*F0*F1 + (1-p)*(1-F0)*(1-F1) - c)**2
        return e1 + e2 + e3

    ans = opt.fmin_l_bfgs_b(err_func, x0=(p0_0, p00, p10), args=((p0_0, p00, p10), ), approx_grad=True)
    ans = ans[0]
    print("Analysis by Bayes' theorem: ")
    print("Thermal population : %s" %(1-ans[0]))
    print("Readout Fidelity for |0>, F0 : %s" %(ans[1]))
    print("Readout Fidelity for |1>, F1 : %s" %(ans[2]))
    print("Avg Readout Fidelity: %s" %(0.5*(ans[1]+ans[2])))

    ans_dict = {'first prep 0': zsFirst0, 'second prep 0': zsSecond0,
                'herald 0 prep 0': zsHerald0,
                'first prep 1': zsFirst1, 'second prep 1': zsSecond1,
                'herald 0 prep 1': zsHeralded0Prepare1,
                'raw F0': p00, 'raw F1': p10,
                'F0': ans[1], 'F1': ans[2],
                "avg thermal population": 1-ans[0]}

    return ans_dict

def readoutHeraldAnalysis2(dataset):

    parameters = dataset.parameters
    stats = parameters['stats']
    qName = dstools.getMeasureNames(dataset)[0]
    qubit = parameters[qName]
    centersList = readout.genCentersList([qubit], states=[0,1,2])

    # the dataformat, see readoutHerald in singleQubitTransmon
    r0 = np.array(dataset[:, (1,2,3,4)])
    r1 = np.array(dataset[:, (5,6,7,8)])
    r2 = np.array(dataset[:, (9,10,11,12)])
    r0 = r0.reshape(1, -1, 2, 2) # (nChannel, stats, nTrigger, I/Q)
    r1 = r1.reshape(1, -1, 2, 2) # (nChannel, stats, nTrigger, I/Q)
    r2 = r2.reshape(1, -1, 2, 2) # (nChannel, stats, nTrigger, I/Q)

    zs = readout.iqToZ(r0)
    states = readout.zsToStates(zs, centersList)
    heraldStates, allChannelIndices = readout.heraldStates(states, heraldState=0)
    probs0 = readout.statesToProbs(heraldStates, centersList)
    zsFirst0 = zs[0, :, 0]
    zsSecond0 = zs[0, :, 1]
    zsHerald0 = zs[0, allChannelIndices, 1]
    heraldRate0 = (heraldStates.shape[1] / float(stats))

    print("Preparing state |0> state")
    print("Avg pops for first measurement: %s" %(np.mean(states[0, :, 0])))
    print("Avg pops for second measurement: %s" %(np.mean(states[0, :, 1])))
    print("Heralding Success rate: %s" %(heraldStates.shape[1] / float(stats)))
    print("Heralded probs for second measurement: %s" %np.squeeze(probs0))
    print("")


    zs = readout.iqToZ(r1)
    states = readout.zsToStates(zs, centersList)
    heraldedStates, allChannelIndices = readout.heraldStates(states, heraldState=0)
    probs1 = readout.statesToProbs(heraldedStates, centersList)
    zsFirst1 = zs[0, :, 0]
    zsSecond1 = zs[0, :, 1]
    zsHeralded0Prepare1 = zs[0, allChannelIndices, 1]
    heraldRate1 = (heraldedStates.shape[1] / float(stats))

    print("Preparing state |1> state")
    print("Avg pops for first measurement: %s" % np.mean(states[0, :, 0]))
    print("Avg pops for second measurement: %s" % np.mean(states[0, :, 1]))
    print("Heralding Success rate: %s" % (heraldedStates.shape[1] / float(stats)))
    print("Heralded probs for second measurement: %s" % np.squeeze(probs1))
    print("")

    zs = readout.iqToZ(r2)
    states = readout.zsToStates(zs, centersList)
    heraldedStates, allChannelIndices = readout.heraldStates(states, heraldState=0)
    probs2 = readout.statesToProbs(heraldedStates, centersList)
    zsFirst2 = zs[0, :, 0]
    zsSecond2 = zs[0, :, 1]
    zsHeralded0Prepare2 = zs[0, allChannelIndices, 1]
    heraldRate2 = (heraldedStates.shape[1] / float(stats))

    print("Preparing state |2> state")
    print("Avg pops for first measurement: %s" % np.mean(states[0, :, 0]))
    print("Avg pops for second measurement: %s" % np.mean(states[0, :, 1]))
    print("Heralding Success rate: %s" % (heraldedStates.shape[1] / float(stats)))
    print("Heralded probs for second measurement: %s" % np.squeeze(probs2))
    print("")

    p00 = np.squeeze(probs0)[0] # P(0|0)
    p10 = np.squeeze(probs1)[1] # P(1|0)
    p20 = np.squeeze(probs2)[2] # P(2|0)
    p0_0 = (heraldRate0 + heraldRate1 + heraldRate2) / 3.0

    print("Raw Readout Fidelity for |0>, F0 : %s" %p00)
    print("Raw Readout Fidelity for |1>, F1 : %s" %p10)
    print("Raw Readout Fidelity for |2>, F2 : %s" %p20)
    print("Raw Avg Readout Fidelity: %s" %((p00+p10+p20)/3.0))
    print("")

