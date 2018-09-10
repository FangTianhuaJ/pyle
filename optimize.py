import copy
from collections import OrderedDict
import numpy as np
from pyle.dataking import sweeps
import labrad.units as U
import pyle.gateCompiler as gc


def makeFunctionWrapper(average, func, axes, deps, measure, sample, noisy=True):
    """
    wrapper function, for optimize a loss function
    the loss function should return an error.

    @param average: average time to get error
    @param func: the loss function
    @param axes: sweep axes
    @param deps: dependences
    @param measure: the qubit(s) measured
    @param sample: sample dict
    @param noisy: print average information
    @return: wrappered function, return average error
            and standard deviation among the average times.
    """

    def funcWrapper(args):
        args = list(args)

        def sweepGen():
            for currAvg in range(average):
                yield args + [currAvg]

        sweep_gen = sweepGen()

        dummy_dataset = sweeps.prepDataset(sample, 'dummy dataset', axes, deps, measure=measure)
        data = sweeps.run(func, sweep_gen, dataset=dummy_dataset, save=False, noisy=False)
        avgError = np.mean(data)
        std = np.std(data)

        if noisy:
            print("average error: {avg}, std: {std}".format(avg=avgError, std=std))

        return avgError, std

    return funcWrapper


class Parameters(object):
    def __init__(self, devs, measure, qubitAndParamNames=None, paramNames=None):
        """

        @param devs: devices
        @param measure: qubit(s) measured
        @param qubitAndParamNames: tell the parameter names of each qubit, should be a list in the form:
            [[qubit Name, paramter Name], ... ]
            if given None, the object will build this automatically according to paramName, devs, and measure
        @param paramNames: if qubitAndParams is given (not None), this parameter will be ignored
                a list of parameter names: [param0 name, param1 name, ...]
        """
        if not isinstance(measure, (list, tuple)):
            measure = [measure]
        self.measure = list(measure)

        if (qubitAndParamNames is None) and (paramNames is None):
            raise Exception("qubitAndParams and paramNames can not be None at the same time")

        if qubitAndParamNames is None:
            qubitAndParamNames = []
            tmpQNames = [devs[idx].__name__ for idx in measure]
            for q in tmpQNames:
                for pName in paramNames:
                    qubitAndParamNames.append([q, pName])

        self.qubitAndParamNames = qubitAndParamNames

        qubitNames = [q for q, p in qubitAndParamNames]

        nameToIdx = {}
        for idx in measure:
            nameToIdx[devs[idx].__name__] = idx
        self.nameToIdx = nameToIdx  # qubit name to idx in devs

        paramDict = {}
        for qName in qubitNames:
            if not qName in paramDict.keys():
                tmp_dict = {}
                params = [p for q, p in qubitAndParamNames if q == qName]
                idx = nameToIdx[qName]
                for param in params:
                    tmp_dict[param] = devs[idx][param]
                paramDict[qName] = tmp_dict
        self.paramDict = paramDict

    def makeInputsAxesDeps(self, nelderMead=True, sweepRange=None):
        """
        build axes and deps, inputs
        the inputs are the intial value for Nelder-Mead method
        @param nelderMead: bool, build axes and deps for Nelder-Mead method
        @param sweepRange: list of sweepRange according to qubitAndParamNames
        @return: axes, deps
        """
        if nelderMead:
            axes = [("Function Calls", "")]
            deps = [("Sequence Error", "", ""), ("Sequence Error STD", "", "")]
        else:
            axes = []
            deps = []
        inputs = []
        for index, (qName, paramName) in enumerate(self.qubitAndParamNames):
            value = self.paramDict[qName][paramName]
            if not np.iterable(value):
                unit = str(value.unit) if hasattr(value, 'unit') else ""
                inputs.append(value[value.unit] if hasattr(value, 'unit') else value)
                if nelderMead:
                    deps.append((qName + "." + str(paramName), "", unit))
                else:
                    axes.append((sweepRange[index], qName + "." + str(paramName)))
            else:
                for idx, val in value:
                    unit = str(val.unit) if hasattr(val, 'unit') else ""
                    inputs.append(val[val.unit] if hasattr(val, 'unit') else val)
                    if nelderMead:
                        deps.append((qName + "." + str(paramName) + "[%d]" % idx, "", unit))
                    else:
                        axes.append((sweepRange[index], qName + "." + str(paramName)))
        return axes, deps, inputs

    def args2Params(self, args):
        """
        convert arguments to parameter dictionary, the order of args should be the same with qubitAndParamNames
        @param args: the arguments need to be transformed
        @return: parameter dictionary
        """
        largs = copy.deepcopy(list(args))
        currParams = copy.deepcopy(self.paramDict)
        for qName, param in self.qubitAndParamNames:
            preValue = currParams[qName][param]
            if not np.iterable(preValue):
                val = largs.pop(0)
                if hasattr(preValue, 'unit'):
                    val = U.Value(val, preValue.unit)
                currParams[qName][param] = val
            else:
                value = list()
                for idx, preVal in enumerate(preValue):
                    val = largs.pop(0)
                    if hasattr(preVal, "unit"):
                        val = U.Value(val, preVal.unit)
                    value.append(val)
                currParams[qName][param] = value
        return currParams

    def updateQubits(self, qubits, currParams, noisy=False):
        """
        update qubits with a dictionary of parameters
        """
        for qName, qDict in currParams.items():
            idx = self.nameToIdx[qName]
            qubit = qubits[idx]
            for key, value in qDict.items():
                if noisy:
                    print("{qName}[{key}]: {oldValue} => {newValue}".format(
                        qName=qName, key=key, oldValue=qubit[key], newValue=value
                    ))
                qubit[key] = copy.copy(value)


def makeQubitAndParamNames(qubit, paramNames):
    """
    @param qubit: dictionary of single qubit
    @param paramNames: a list of parameters, e.g. ["piAmp", "f10"]
    @return: qubitAndParamNames, [ [qName, param0Name], [qName, param0Name], ..., ]
    """
    ans = [[qubit.__name__, param] for param in paramNames]
    return ans


def updateNM(Sample, dvwData, method='min'):
    """
    update registry according to the Nelder-Mead result

    @param Sample: RegistryWrapper
    @param dvwData: LocalDataset, raw data from data_vault
    @param method: default is "min",
                "min", find minimum value
                "max", find maximum value
                None, the last one
                int N, the N-th one (indice start at 0 )
    """

    if method is None:
        optimalParams = list(dvwData[-1, 3:])
    elif method == 'min':
        bestIdx = np.argmin(dvwData[:, 1])
        optimalParams = list(dvwData[bestIdx, 3:])
        print('best IDX is: %s' % (bestIdx + 1))
    elif method == 'max':
        bestIdx = np.argmax(dvwData[:, 1])
        optimalParams = list(dvwData[bestIdx, 3:])
        print('best IDX is: %s' % (bestIdx + 1))
    else:
        optimalParams = list(dvwData[method, 3:])

    measure = [int(i) for i in dvwData.parameters['measure']]
    paramNames = dvwData.parameters['paramNames']
    sample, devs, qubits, Qubits = gc.loadQubits(Sample, measure, write_access=True)
    param = Parameters(devs, measure, paramNames=paramNames)
    currParamDict = param.args2Params(optimalParams)
    param.updateQubits(Qubits, currParamDict, noisy=True)
