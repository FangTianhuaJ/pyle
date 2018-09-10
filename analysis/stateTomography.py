# Copyright (C) 2012  Daniel Sank
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
from pyle import tomo
import labrad

def fMatrix(dataset, calScurveKey = 'calScurve1'):
    parameters = dataset.parameters
    sMatrices = []
    for qubit in [parameters[parameters.config[m]] for m in parameters.measure]:
        s10 = qubit[calScurveKey][0]
        s11 = qubit[calScurveKey][1]
        S = np.array([[1-s10,1-s11],[s10,s11]])
        sMatrices.append(S)
    return reduce(np.kron, sMatrices)

def measfMatrix(dataset, key='measureF'):
    """
    similar with fMatrix, but use measureF0, measureF1 as correction key
    @return: fidelity matrix
    """
    parameters = dataset.parameters
    sMatrices = []
    key0 = key + '0'
    key1 = key + '1'
    for qubit in [parameters[parameters.config[m]] for m in parameters.measure]:
        f0 = qubit[key0]
        f1 = qubit[key1]
        fM = np.array([[f0, 1-f1], [1-f0, f1]])
        sMatrices.append(fM)
    return reduce(np.kron, sMatrices)

def correctVisibility(F, tomoData, N):
    """
    Correct one row of tomography data for measurement visibility

    INTPUTS

    F: Fidelity matrix for the multiqubit system

    tomoData: Each row contains the probabilities of the various
    multiqubit states. One row for each tomography operation.
    Each row should have length 2**N where N is the number of
    (measured) qubits in the system

    N: number of qubits measured

    NOTES

    dataset.data will contain several columns in the following form
    (assuming two qubits):
                    |   Tomo rotation 0   |   Tomo rotation 1   |
    indep_x indep_y | P_00 P_01 P_10 P_11 | P_00 P_01 P_10 P_11 |
     x_0     y_0    | p_00 p_01 p_10 p_11 | p_00 p_01 p_10 p_11 |
     x_0     y_1    | p_00 p_01 p_10 p_11 | p_00 p_01 p_10 p_11 |
     x_0     y_2    | p_00 p_01 p_10 p_11 | p_00 p_01 p_10 p_11 |
     etc.

    In the mathematical treatment of measurement visibility correction
    we say that the measured probabilities M are related to the true
    probabilities P via the equation
     M = F P
    where F is the visibility matrix and matrix multiplication is
    implied. M and P are column vectors of probabilities in "kronecker"
    order, ie |00> |01> |10> |11> for two qubits.
    The problem is solved by P = F^-1 M.

    As shown above, for each tomo operation the probabilities are
    in a row, so we do the matrix multiplication by transposing the
    equation:

    P^T = M^T (F^-1)^T

    where here M^T is a row of probabilities.

    We can do the inversion for many different column vectors M
    simultaneously by simply stacking them next to one another and
    effecting the same matrix multiplication. Just write it out to
    see that it works!

    In our case we indeed have several vectors of probabilities to
    correct; one for each tomo operation. Therefore, we take our
    row of probabilities, reshape it into a matrix in which each row
    corresponds to a single tomography rotation, and then use the idea
    of simultaneous correction to fix all the rows at once. We then
    reshape the matrix back into a single row and return it to the user.

    Author: DTS
    """
    #Switch data from a single row to matrix for which each row
    #corresponds to a single tomography rotation.
    reshaped = tomoData.reshape((-1,2**N))
    #Do the matrix multiplication
    result = np.dot(reshaped,np.linalg.inv(F).T)
    #Turn data back into a single row
    result = result.reshape((-1,))
    return result


def correctQPT(dataset, savePath, name='CZ QPT', correct=True, crosstalkMatrix=None):
    """Correct QPT data for measurement visibility and crosstalk and saves result
    to the data vault.

    PARAMETERS
    dataset: Dataset with QPT data/parameters/etc.
    path: Location to save corrected data to in the data vault.
    name - str: File name for corrected data .
    correct - bool: Whether to correct for measurement visibility.
    crosstalkMatrix - array: Crosstalk matrix. Load matrix prior to running with
        pyle.dataking.crosstalk.measureCrosstalkMatrix.
    """
    newdata = dataset.data.copy()
    params = dataset.parameters
    if correct:
        name = name+' Corrected'
        F = fMatrix(dataset)
        if crosstalkMatrix is None:
            # Only correct for measurement visibility
            correctMat = F
        else:
            # Correct for measurement visibility and crosstalk
            correctMat = np.dot(crosstalkMatrix,F)
        N = len(dataset.parameters.measure)
        for row,datarow in enumerate(dataset.data):
            newdata[row] = np.append(datarow[0],correctVisibility(correctMat,datarow[1:],N))
    # Now save corrected data to the data vault
    with labrad.connect() as cxn:
        dv = cxn.data_vault
        dv.cd(savePath)
        dv.new(name,dataset.variables[0],dataset.variables[1])
        dv.add(newdata)
        for key in params.keys():
            if key in params.config:
                for qubitkey in params[key].keys():
                    dv.add_parameter(key+'.'+qubitkey,params[key][qubitkey])
            else:
                dv.add_parameter(key,params[key])
        dv.add_parameter('correctVisibility',correct)
        dv.add_parameter('crosstalkMatrix',crosstalkMatrix is not None)


def getQstRho(dataset, phases=None, correct=True, row=None):
    """
    Extract a density matrix (or list of them) from a dataset object

    dataset is an attribute dictionary, probably generated by
    pyle.plotting.dstools.getDeviceDataset

    Returns a numpy array of density matrices.

    Note that if there aren't any indepenedent variables in
    dataset.variables, the returned data will still be an array with one
    element, and that element is the density matrix you measured.
    """
    numQubits = len(dataset.parameters['measure'])
    tomoType = dataset.parameters['measureType']
    numIndeps = len(dataset.variables[0])
    if row is None:
        probs = dataset.data[:,numIndeps:]
    else:
        probs = dataset.data[row,numIndeps:]
    indeps = dataset.data[:,0:numIndeps]
    numTomoOps = {'Tomo': 3, 'Octomo': 6}
    if correct:
        F = fMatrix(dataset)
    else:
        F = None
    #Get maximum likelihood fit density matrices
    pxmsArray = probs.reshape((-1, numTomoOps[tomoType]**numQubits, 2**numQubits))
    rhos = []
    for num,pxms in enumerate(pxmsArray):
        if phases is None:
            tomoNum = str(numQubits) if (numQubits > 1) else ''
            Us, U = tomo._qst_transforms[tomoType.lower()+tomoNum]
        else:
            if tomoType == 'Tomo':
                tomoOps = tomo.tomo_ops
            if tomoType == 'Octomo':
                tomoOps = tomo.octomo_ops
            Us = tomo.tensor_combinations_phases(tomoOps,numQubits,phases[num])
        rhos.append(tomo.qst_mle(pxms, Us, F))
    return np.array(rhos)
