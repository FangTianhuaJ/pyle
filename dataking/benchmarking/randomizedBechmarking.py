import numpy as np
from numpy.linalg import eigvals
import random
import itertools
import copy
from scipy.optimize import curve_fit

from pyle import tomo
from pyle.math import tensor, dot3

from labrad import units as U

ns = U.Unit("ns")


def gate2OpList(numQubits, gateIn):
    """
    This is for a *single* clifford gate (C1 or C2)
    transform the gate to operation, and pad IGNs to align CZ gate
    example 1 (for single qubit)
        gate = ( ("X", "Y"), )
        =>
        op = [ ("X",), ("Y",) ]
    example 2 (for two qubits without 2-qubit gate)
        gate = ( ("X", "Y"), ("X/2", "Y/2") )
        =>
        op = [ ("X", "X/2"), ("Y", "Y/2") ]
    example 3 (for two qubits with 2-qubit gate)
        gate = ( ("X", "CZ", "X/2"), ("CZ", "Y/2") )
        =>
        op = [ ("X", "IGN"), ("CZ", "CZ"), ("X/2", "Y/2") ]
        add an "IGN"
    @param numQubits: the number of qubits, 1 or 2 is implmented now
    @param gateIn: list or tuple,
    @return: list of operation
    """
    gateIn = copy.deepcopy(gateIn)
    if numQubits == 1:
        # single qubit gate, only 1 element in gateIn
        opList = [(elem,) for elem in gateIn[0]]
    elif numQubits == 2:
        opList = []
        gateOut = list(gateIn)
        numCZ = gateIn[0].count("CZ")
        if numCZ != (gateIn[1].count("CZ")):
            raise Exception("Different CZ number in two qubits")
        if numCZ > 0:
            idx0 = gateIn[0].index("CZ")  # first index of CZ in q0 gates
            idx1 = gateIn[1].index("CZ")  # first index of CZ in q1 gates
            # align CZ according the first CZ
            # from the four class of C2, I find when the first CZ is aligned
            # the followed CZ is aligned too
            if idx0 > idx1:
                gateOut[1] = ("IGN",) * (idx0 - idx1) + gateIn[1]
            elif idx0 < idx1:
                gateOut[0] = ("IGN",) * (idx1 - idx0) + gateIn[0]
        # in gateOut, CZ(s) is aligned, and the number of gates can also be different
        # I add IGN(s) to keep the same number of gates for two qubits.
        len0 = len(gateOut[0])
        len1 = len(gateOut[1])
        if len0 > len1:
            gateOut[1] += ("IGN",) * (len0 - len1)
        elif len0 < len1:
            gateOut[0] += ("IGN",) * (len1 - len0)
        # now all the CZs are aligned and the gate numbers are the same.
        for g1, g2 in zip(gateOut[0], gateOut[1]):
            opList.append((g1, g2))
    else:
        raise Exception("numQubit <= 2 are implemented")
    return opList

class RBClifford(object):
    """
    Randomized Bechmarking Clifford Group based.
    Generate Clifford gate list for RB, 1 and 2 qubits are implemented with CZ-based two-qubit gates.
    For 1 qubit : generates cliffords from the 1 qubit clifford set C1, with |C1|=24
    For 2 qubits: generates cliffords from the 2 qubit clifford set C2, with |C2|=11520 from 4 classes:
    (CZ-based clifford set C2)
        single qubit class (no CZs): 576
        CNOT-like class (1 CZ): 5184
        ISWAP-like class (2 CZs): 5184
        SWAP-like class (3 CZs): 576

    gatelist format:
    [ 1st clifford: ((q0 gates), (q1 gates)),
      2nd clifford: ((q0 gates), (q1 gates)),
      ...
    ]
    (q0 gates) may be ("X", "Y"), means first "X" on q0 and then "Y" on q0

    for example:
    gateList = [
        ( ("X", "X/2"), ("Y", "I") ),
        ( ("-Y", "Y/2"), ("X/2", "-X") )
    ]
    1st: X on q0, Y on q1, then X/2 on q0, I on q1
    2nd: -Y on q0, Y/2 on q1, then Y/2 on q0, -X on q1

    for 2-qubit gates, the first one is control, the second one is target
    gateList = [
        ( ("X", "CZ", "X/2"), ("Y", "CZ", "Y/2") )
    ]
    1st: X on q0, Y on q1
    2nd: q0 control, q1 target, CZ
    3rd: X/2 on q0, Y/2 on q1
    """

    def __init__(self, numQubits, randomizedOverIswapAndSwap=False):
        """
        @param numQubits: number of qubits, 1 or 2 is implemented
        @param randomizedOverIswapAndSwap: default is False,
        We use CZ as our base operation for 2-qubit, and there are 8 ways to generate
        iSwap with CZ, and 24 ways to generate Swap with CZ. this flag is whether we
        choose one of 8(24) ways to generate iSwap(Swap) or we just use the fixed one
        described in Appendix B of Julian Kelly's thesis (2015)
        """
        self.numQubits = numQubits
        self.randomizedOverIswapAndSwap = randomizedOverIswapAndSwap
        # this is the elementary gate to generate clifford operation for single qubits
        # for 2qubit, CZ is used
        self.cliffordGeneratorSet = ("I", "X", "Y", "X/2", "Y/2", "-X/2", "-Y/2", "-X", "-Y")

        # single qubit cliffords C1 and S1 (From Appendix B of Julian Kelly's thesis)
        # average single qubit gates per clifford: 45/24 = 1.875
        self.gateSetC1 = (
            # Paulis
            ("I",), ("X",), ("Y",), ("Y", "X"),
            # 2pi/3 rotation
            ("X/2", "Y/2"), ("X/2", "-Y/2"), ("-X/2", "Y/2"), ("-X/2", "-Y/2"),
            ("Y/2", "X/2"), ("Y/2", "-X/2"), ("-Y/2", "X/2"), ("-Y/2", "-X/2"),
            # pi/2 rotations
            ("X/2",), ("-X/2",), ("Y/2",), ("-Y/2",),
            ("-X/2", "Y/2", "X/2"), ("-X/2", "-Y/2", "X/2"),
            # Hadamard-like
            ("X", "Y/2"), ("X", "-Y/2"), ("Y", "X/2"), ("Y", "-X/2"),
            ("X/2", "Y/2", "X/2"), ("-X/2", "Y/2", "-X/2")
        )

        # these S1 gates are elements for the single qubit Clifford group
        # for details, just go to see Appendix B of Julian Kelly's thesis (2015)
        self.gateSetS1 = [ ("I",), ("Y/2", "X/2"), ("-X/2", "-Y/2") ]
        self.gateSetS1_X2 = [ ("X/2",), ("X/2", "Y/2", "X/2"), ("-Y/2",) ]
        self.gateSetS1_Y2 = [ ("Y/2",), ("Y", "X/2"), ("-X/2", "-Y/2", "X/2") ]

        # 8 ways to decompose ISWAP into CZ and single qubit operations
        all_iswaps = [
            (('X/2', 'CZ', 'X/2', 'CZ', '-X/2'), ('Y/2', 'CZ', '-Y/2', 'CZ', '-Y/2')),
            (('X/2', 'CZ', '-X/2', 'CZ', '-X/2'), ('-Y/2', 'CZ', '-Y/2', 'CZ', 'Y/2')),
            (('Y/2', 'CZ', '-Y/2', 'CZ', '-Y/2'), ('X/2', 'CZ', 'X/2', 'CZ', '-X/2')),
            (('Y/2', 'CZ', 'Y/2', 'CZ', '-Y/2'), ('-X/2', 'CZ', 'X/2', 'CZ', 'X/2')),
            (('-X/2', 'CZ', 'X/2', 'CZ', 'X/2'), ('Y/2', 'CZ', 'Y/2', 'CZ', '-Y/2')),
            (('-X/2', 'CZ', '-X/2', 'CZ', 'X/2'), ('-Y/2', 'CZ', 'Y/2', 'CZ', 'Y/2')),
            (('-Y/2', 'CZ', '-Y/2', 'CZ', 'Y/2'), ('X/2', 'CZ', '-X/2', 'CZ', '-X/2')),
            (('-Y/2', 'CZ', 'Y/2', 'CZ', 'Y/2'), ('-X/2', 'CZ', '-X/2', 'CZ', 'X/2'))
        ]

        # 24 ways to decompose SWAP int CZ and single qubit operations
        all_swaps= [
            (('X/2', 'CZ', '-X/2', 'CZ', 'X/2', 'CZ'), ('-X/2', 'CZ', 'X/2', 'CZ', '-X/2', 'CZ')),
            (('Y/2', 'CZ', '-Y/2', 'CZ', 'Y/2', 'CZ'), ('-Y/2', 'CZ', 'Y/2', 'CZ', '-Y/2', 'CZ')),
            (('-X/2', 'CZ', 'X/2', 'CZ', '-X/2', 'CZ'), ('X/2', 'CZ', '-X/2', 'CZ', 'X/2', 'CZ')),
            (('-Y/2', 'CZ', 'Y/2', 'CZ', '-Y/2', 'CZ'), ('Y/2', 'CZ', '-Y/2', 'CZ', 'Y/2', 'CZ')),
            (('I', 'CZ', 'X/2', 'CZ', '-X/2', 'CZ', 'X/2'), ('I', 'CZ', '-X/2', 'CZ', 'X/2', 'CZ', '-X/2')),
            (('I', 'CZ', 'Y/2', 'CZ', '-Y/2', 'CZ', 'Y/2'), ('I', 'CZ', '-Y/2', 'CZ', 'Y/2', 'CZ', '-Y/2')),
            (('I', 'CZ', '-X/2', 'CZ', 'X/2', 'CZ', '-X/2'), ('I', 'CZ', 'X/2', 'CZ', '-X/2', 'CZ', 'X/2')),
            (('I', 'CZ', '-Y/2', 'CZ', 'Y/2', 'CZ', '-Y/2'), ('I', 'CZ', 'Y/2', 'CZ', '-Y/2', 'CZ', 'Y/2')),
            (('I', 'CZ', 'X/2', 'CZ', '-X/2', 'CZ', 'I'), ('X/2', 'CZ', '-X/2', 'CZ', 'X/2', 'CZ', '-X/2')),
            (('I', 'CZ', 'Y/2', 'CZ', '-Y/2', 'CZ', 'I'), ('Y/2', 'CZ', '-Y/2', 'CZ', 'Y/2', 'CZ', '-Y/2')),
            (('I', 'CZ', '-X/2', 'CZ', 'X/2', 'CZ', 'I'), ('-X/2', 'CZ', 'X/2', 'CZ', '-X/2', 'CZ', 'X/2')),
            (('I', 'CZ', '-Y/2', 'CZ', 'Y/2', 'CZ', 'I'), ('-Y/2', 'CZ', 'Y/2', 'CZ', '-Y/2', 'CZ', 'Y/2')),
            (('X/2', 'CZ', '-X/2', 'CZ', 'X/2', 'CZ', '-X/2'), ('I', 'CZ', 'X/2', 'CZ', '-X/2', 'CZ', 'I')),
            (('X/2', 'CZ', 'Y/2', 'CZ', '-Y/2', 'CZ', '-X/2'), ('Y/2', 'CZ', 'X/2', 'CZ', '-X/2', 'CZ', '-Y/2')),
            (('X/2', 'CZ', '-X/2', 'CZ', 'X/2', 'CZ', 'I'), ('-X/2', 'CZ', 'X/2', 'CZ', '-X/2', 'CZ', 'I')),
            (('X/2', 'CZ', '-Y/2', 'CZ', 'Y/2', 'CZ', '-X/2'), ('-Y/2', 'CZ', 'X/2', 'CZ', '-X/2', 'CZ', 'Y/2')),
            (('Y/2', 'CZ', '-Y/2', 'CZ', 'Y/2', 'CZ', '-Y/2'), ('I', 'CZ', 'Y/2', 'CZ', '-Y/2', 'CZ', 'I')),
            (('Y/2', 'CZ', 'X/2', 'CZ', '-X/2', 'CZ', '-Y/2'), ('X/2', 'CZ', 'Y/2', 'CZ', '-Y/2', 'CZ', '-X/2')),
            (('Y/2', 'CZ', '-X/2', 'CZ', 'X/2', 'CZ', '-Y/2'), ('-X/2', 'CZ', 'Y/2', 'CZ', '-Y/2', 'CZ', 'X/2')),
            (('Y/2', 'CZ', '-Y/2', 'CZ', 'Y/2', 'CZ', 'I'), ('-Y/2', 'CZ', 'Y/2', 'CZ', '-Y/2', 'CZ', 'I')),
            (('-X/2', 'CZ', 'X/2', 'CZ', '-X/2', 'CZ', 'X/2'), ('I', 'CZ', '-X/2', 'CZ', 'X/2', 'CZ', 'I')),
            (('-X/2', 'CZ', 'X/2', 'CZ', '-X/2', 'CZ', 'I'), ('X/2', 'CZ', '-X/2', 'CZ', 'X/2', 'CZ', 'I')),
            (('-X/2', 'CZ', 'Y/2', 'CZ', '-Y/2', 'CZ', 'X/2'), ('Y/2', 'CZ', '-X/2', 'CZ', 'X/2', 'CZ', '-Y/2')),
            (('-X/2', 'CZ', '-Y/2', 'CZ', 'Y/2', 'CZ', 'X/2'), ('-Y/2', 'CZ', '-X/2', 'CZ', 'X/2', 'CZ', 'Y/2')),
            (('-Y/2', 'CZ', 'Y/2', 'CZ', '-Y/2', 'CZ', 'Y/2'), ('I', 'CZ', '-Y/2', 'CZ', 'Y/2', 'CZ', 'I')),
            (('-Y/2', 'CZ', 'X/2', 'CZ', '-X/2', 'CZ', 'Y/2'), ('X/2', 'CZ', '-Y/2', 'CZ', 'Y/2', 'CZ', '-X/2')),
            (('-Y/2', 'CZ', 'Y/2', 'CZ', '-Y/2', 'CZ', 'I'), ('Y/2', 'CZ', '-Y/2', 'CZ', 'Y/2', 'CZ', 'I')),
            (('-Y/2', 'CZ', '-X/2', 'CZ', 'X/2', 'CZ', 'Y/2'), ('-X/2', 'CZ', '-Y/2', 'CZ', 'Y/2', 'CZ', 'X/2'))
        ]

        if numQubits == 2:
            twoQubitS1 = tuple(itertools.product(self.gateSetS1, self.gateSetS1))
            twoQubitS1_I_Y2 = tuple(itertools.product(self.gateSetS1, self.gateSetS1_Y2))
            twoQubitS1_Y2_X2 = tuple(itertools.product(self.gateSetS1_Y2, self.gateSetS1_X2))

            # build single qubit class (no CZs), 576 elements
            gateSetC2_sq = list(itertools.product(self.gateSetC1, self.gateSetC1))

            # build CNOT-like class (1 CZ), 5184 elements
            # first element is control, second element is target
            two_qubit_part = ( ("CZ",), ("CZ",) )
            gateSetC2_cnot = list()
            for c2_q0, c2_q1 in gateSetC2_sq:
                for s1, s1_y2 in twoQubitS1_I_Y2:
                    g = (c2_q0 + two_qubit_part[0] + s1, c2_q1 + two_qubit_part[1] + s1_y2)
                    gateSetC2_cnot.append(g)

            if randomizedOverIswapAndSwap:
                # Build ISWAP-like class (2 CZs), 5184 elements
                gateSetC2_iswap = list()
                two_qubit_part = random.sample(all_iswaps, 1)[0]
                for c2_q0, c2_q1 in gateSetC2_sq:
                    for s1_q0, s1_q1 in twoQubitS1:
                        g = (c2_q0 + two_qubit_part[0] + s1_q0, c2_q1 + two_qubit_part[1] +  s1_q1)
                        gateSetC2_iswap.append(g)

                # Build SWAP-like class (3 CZs), 576 elements
                gateSetC2_swap = list()
                two_qubit_part = random.sample(all_swaps, 1)[0]
                for c2_q0, c2_q1 in gateSetC2_sq:
                    g = (c2_q0+two_qubit_part[0], c2_q1+two_qubit_part)
                    gateSetC2_swap.append(g)
            else:
                # Build ISWAP-like class (2 CZs), 5184 elements
                gateSetC2_iswap = list()
                two_qubit_part = ( ("CZ", "Y/2", "CZ"), ("CZ", "-X/2", "CZ") )
                for c2_q0, c2_q1 in gateSetC2_sq:
                    for s1_y2, s1_x2 in twoQubitS1_Y2_X2:
                        g = (c2_q0 + two_qubit_part[0] + s1_y2, c2_q1 + two_qubit_part[1] + s1_x2)
                        gateSetC2_iswap.append(g)

                # Build SWAP-like class (3 CZs), 576 elements
                gateSetC2_swap = list()
                two_qubit_part = ( ("CZ", "-Y/2", "CZ", "Y/2", "CZ"), ("CZ", "Y/2", "CZ", "-Y/2", "CZ", "Y/2"))
                for c2_q0, c2_q1 in gateSetC2_sq:
                    g = (c2_q0+two_qubit_part[0], c2_q1+two_qubit_part[1])
                    gateSetC2_swap.append(g)

            # gateSetC2 contains all the operations, 11520 elements
            gateSetC2 = tuple(gateSetC2_sq + gateSetC2_cnot + gateSetC2_iswap + gateSetC2_swap)
        else:
            gateSetC2 = tuple()

        self.gateSetC2 = gateSetC2

        # build full gate set, 1-qubit and 2-qubit share the same format
        if numQubits == 1:
            self.cliffordGateSet = tuple((g,) for g in self.gateSetC1)
        elif numQubits == 2:
            self.cliffordGateSet = copy.deepcopy(self.gateSetC2)
        else:
            raise Exception("RB Clifford is not implemented for numQubit >2 ")

        self.singleQubitGateNames = ("I", "X", "Y", "X/2", "Y/2", "-X", "-Y", "-X/2", "-Y/2",
                                     "IGN", "IW", "IWSE", "Zpi", "Zpi/2")
        self.singleQubitGateUnitary = {
            "I": tomo.sigmaI,
            "IW": tomo.sigmaI,
            "IWSE": tomo.sigmaI,
            "SE": tomo.Rmat(tomo.sigmaX, np.pi),
            "IGN": tomo.sigmaI,
            "X": tomo.Rmat(tomo.sigmaX, np.pi),
            "Y": tomo.Rmat(tomo.sigmaY, np.pi),
            "X/2": tomo.Rmat(tomo.sigmaX, np.pi/2),
            "Y/2": tomo.Rmat(tomo.sigmaY, np.pi/2),
            "-X": tomo.Rmat(tomo.sigmaX, -np.pi),
            "-Y": tomo.Rmat(tomo.sigmaY, -np.pi),
            "-X/2": tomo.Rmat(tomo.sigmaX, -np.pi/2),
            "-Y/2": tomo.Rmat(tomo.sigmaY, -np.pi/2),
            "H": np.array([[1,1],[1,-1]], dtype=np.complex)/np.sqrt(2),
            "Zpi": tomo.Rmat(tomo.sigmaZ, np.pi),
            "Zpi/2": tomo.Rmat(tomo.sigmaZ, np.pi/2),
            "Zpi/4": tomo.Rmat(tomo.sigmaZ, np.pi/4),
            "2T": tomo.Rmat(tomo.sigmaZ, np.pi/2),
            "Z": np.eye(2, dtype=np.complex),
            "ZW": np.eye(2, dtype=np.complex),
            "SWAP": np.eye(2, dtype=np.complex),
        }

        if numQubits == 1:
            self.baseGateUnitary = self.singleQubitGateUnitary
        if numQubits == 2:
            self.baseGateUnitary = {}
            twoQubit = itertools.product(self.singleQubitGateNames, self.singleQubitGateNames)
            for gate in twoQubit:
                self.baseGateUnitary[gate] = tensor([self.singleQubitGateUnitary[g] for g in gate])
            self.baseGateUnitary[("CZ", "CZ")] = np.diag([1, 1, 1, -1]).astype('complex')
            self.baseGateUnitary[("CNOT", "CNOT")] = np.array([[1, 0, 0, 0],
                                                               [0, 1, 0, 0],
                                                               [0, 0, 0, 1],
                                                               [0, 0, 1, 0]], dtype=np.complex)

        # compute the matrix for each gate in cliffordGateSet
        cliffordUnitary = {}
        for gate in self.cliffordGateSet:
            cliffordUnitary[gate] = self.computeUnitary([gate])
        self.cliffordUnitary = cliffordUnitary

    def computeUnitary(self, gateList):
        """
        compute the total unitary matrix of the *entire* sequence of gateList
        @param gateList: the gateList should be in the format bellow:
            gatelist format (elements of gateList can be tuple or list):
                [ 1st gate: [[q0 gates], [q1 gates]],
                  2nd gate: [[q0 gates], [q1 gates]],
                  ...
                ]
        @return: the unitary matrix
        """
        U = np.eye(2**self.numQubits)
        for gate in gateList:
            opList = self.gate2OpList(gate)
            for op in opList:
                if len(op) > 1 and isinstance(op, (list, tuple)): # two-qubit operation
                    currU = self.baseGateUnitary[tuple(op)]
                    U = np.dot(currU, U)
                elif len(op) == 1: # single qubit operation
                    # op is a tuple, e.g. ("X",), so we should convert it to "X"
                    currU = self.baseGateUnitary[op[0]]
                    U = np.dot(currU, U)
                else: # may be like "X", single qubit operation
                    currU = self.baseGateUnitary[op]
                    U = np.dot(currU, U)
        return U

    def gate2OpList(self, gateIn):
        """
        This is for a *single* clifford gate (C1 or C2)
        transform the gate to operation, and pad IGNs to align CZ gate
        example 1 (for single qubit)
            gate = ( ("X", "Y"), )
            =>
            op = [ ("X",), ("Y",) ]
        example 2 (for two qubits without 2-qubit gate)
            gate = ( ("X", "Y"), ("X/2", "Y/2") )
            =>
            op = [ ("X", "X/2"), ("Y", "Y/2") ]
        example 3 (for two qubits with 2-qubit gate)
            gate = ( ("X", "CZ", "X/2"), ("CZ", "Y/2") )
            =>
            op = [ ("X", "IGN"), ("CZ", "CZ"), ("X/2", "Y/2") ]
            add an "IGN"
        @param gateIn: list or tuple,
        @return: list of operation
        """
        return gate2OpList(self.numQubits, gateIn)

    @staticmethod
    def howCloseToI(U):
        """
        calculate how close to I-matrix
        @param U: the matrix need to be check
        @return: float value, if close to 1, the matrix U is close to I
        """
        assert U.shape[0] == U.shape[1]
        N = U.shape[0]
        vals = eigvals(U)
        ang = np.angle(vals[0])
        vals = np.exp(-1j*ang) * vals
        return np.sum(vals).real/N

    def finishGateList(self, gateList):
        """
        add an final gate from Clifford group to make the unitary of the entire sequence
        be the identity, neglecting a global phase.
        @param gateList, list of gate, the format is decribed in the docstring of this Class
        @return full gate list
        """
        U = self.computeUnitary(gateList)
        finalGate = None
        for gate in self.cliffordGateSet:
            Uclifford=self.cliffordUnitary[gate] # use precomputed unitaries, faster
            Uf = np.dot(Uclifford, U)
            dis2I = RBClifford.howCloseToI(Uf)
            if dis2I > 0.9999:
                print dis2I
                finalGate = gate
                break
        if finalGate is None:
            raise Exception('Final gate could not be found (when interleaving: use a Clifford generator).')
        fullGateList = gateList + [finalGate]
        return fullGateList

    def randGen(self, numClifffordGates, interleaved=None, finish=True):
        """
        This generates a random list of strings that correspond to gates
        MAKE SURE IT IS A CLIFFORD GENERATOR, otherwise there may not be a reversal gate
        @param numClifffordGates: the number of Clifford gates. ***Clifford gate*** !!!
        @param interleaved: describe the gate to be interleaved, default is None
                e.g. interleaved = "X" for single qubit
                     interleaved = ("X", "Y/2") for single qubit
                     interleaved = (("CZ",), ("CZ",)) for two qubits
                     interleaved = ("CZ", "CZ") for two qubits
        @param finish: whether adding a finish gate to make the total sequence to be identity,
                       default is True
        """

        if self.numQubits == 1 and interleaved:
            # parser the interleaved gate for single qubit
            # e.g. "X" => (("X", ), ), ("X","Y/2") => ( ("X", "Y/2"), )
            if isinstance(interleaved, basestring):
                interleaved = ((interleaved, ), )
            elif isinstance(interleaved, (tuple, list)):
                interleaved = (tuple(interleaved), )
        elif self.numQubits == 2 and interleaved:
            # parser the interleaved gate for two qubits
            # e.g. ( ("CZ", ), ("CZ,) ) => ( ("CZ", ), ("CZ,) )
            # e.g. [ ["CZ"],["CZ"] ] => ( ("CZ", ), ("CZ,) )
            # e.g. ("CZ", "CZ") => ( ("CZ", ), ("CZ,) )
            if isinstance(interleaved[0], basestring):
                # this is for the format ("CZ", "CZ"), 1D tuple or 1D list
                interleaved = ( (interleaved[0], ), (interleaved[1], ) )
            else:
                # this if for the format of 2D tuple or 2D list
                interleaved = tuple((tuple(x) for x in interleaved))

        U = np.eye(2**self.numQubits)
        if interleaved:
            Uinterleaved = self.computeUnitary([interleaved])  # get unitary of interleaved
            gateList = []
            N = len(self.cliffordGateSet)
            for k in range(numClifffordGates):
                idx = random.randint(0, N - 1)
                gate = self.cliffordGateSet[idx]
                gateList.append(gate)
                U = np.dot(self.cliffordUnitary[gate], U)
                gateList.append(interleaved)
                U = np.dot(Uinterleaved, U)
        else:
            gateList = []
            N = len(self.cliffordGateSet)
            for k in range(numClifffordGates):
                idx = random.randint(0, N - 1)
                gate = self.cliffordGateSet[idx]
                gateList.append(gate)
                U = np.dot(self.cliffordUnitary[gate], U)

        if finish:
            finalGate = None
            for gate in self.cliffordGateSet:
                Uclifford = self.cliffordUnitary[gate]  # use precomputed unitaries, faster
                Uf = np.dot(Uclifford, U)
                dis2I = RBClifford.howCloseToI(Uf)
                if dis2I > 0.9999:
                    finalGate = gate
                    break
            if finalGate is None:
                raise Exception(('Final gate could not be found '
                                 '(when interleaving: use a Clifford generator).'))
            gateList = gateList + [finalGate]

        return gateList

    def setGateLength(self, devs):
        """
        set the length of each gate according to the devices (devs)
        the gateLength is a dictionary in the format:
        {"X": [12*ns, 14*ns], "Y": ....}
        @param devs: the devices e.g. qubit
        """
        if self.numQubits == 1:
            gateSet = self.singleQubitGateNames
            gateSet += ("H", "2T", )
        elif self.numQubits == 2:
            gateSet = self.singleQubitGateNames
            gateSet += ('CZ', "H", "2T")
        gateLength = {}
        for gate in gateSet:
            for devidx, dev in enumerate(devs):
                padding = dev.get('xyPadding', 0.0 * ns)
                if np.alen(padding) == 1:
                    padding = [padding, padding]
                    # padding before and after the pulse, default symmetric padding if not specified
                padding = sum(padding)
                addtoDic = False
                if gate is 'I':
                    gate_length = [dev.get('identityLen', np.min([dev['piLen'], dev['piHalfLen']]) * ns)]
                    addtoDic = True
                if gate in ['IW', 'IWSE', 'SE']:
                    gate_length = [dev.get('identityWaitLen', 0. * ns)]
                    addtoDic = True
                if gate is 'IGN':
                    gate_length = [0. * ns]
                    addtoDic = True
                if gate in ['Z', 'ZW']:
                    gate_length = [dev['detuneLen'] + 2. * dev['detuneW'] + 2. * dev.get('detunePadding', 0.0)]
                    addtoDic = True
                elif gate in ['X/2', 'Y/2', '-X/2', '-Y/2']:
                    gate_length = [dev['piHalfLen'] + padding]
                    addtoDic = True
                elif gate in ['X', 'Y', '-X', '-Y']:
                    gate_length = [dev['piLen'] + padding]
                    addtoDic = True
                elif gate in ['CZ'] and devidx == 0:
                    cztime = dev[devs[1].__name__ + 'aczTime'] + 2. * dev[devs[1].__name__ + 'aczW']
                    gate_length = [cztime, cztime]
                    addtoDic = True
                elif gate in ['H']:
                    gate_length = [dev['piHalfLen'] + dev['piLen'] + 2. * padding]
                    addtoDic = True
                elif gate in ['Zpi']:
                    gate_length = [dev.get('piLenZ', 0.0 * ns)]
                    addtoDic = True
                elif gate in ['Zpi/2']:
                    gate_length = [dev.get('piHalfLenZ', 0.0 * ns)]
                    addtoDic = True
                elif gate in ['Zpi/4']:
                    gate_length = [dev.get('piQuarterLenZ', 0.0 * ns)]
                    addtoDic = True
                elif gate in ['2T']:
                    gate_length = [2. * dev.get('piQuarterLenZ', 0.0 * ns)]
                    addtoDic = True
                if addtoDic:
                    if gate in gate_length:
                        gateLength[gate] += gate_length
                    else:
                        gateLength[gate] = gate_length
        self.gateLength = gateLength

    def computeGateLength(self, gateList):
        """
        compute the total length of gateSequence roughly (not exactly, and overestimated)
        @param gateList: the sequence of gate
        @return: the length of all total gates
        """
        totalTime = 0*ns
        for singleGate in gateList:
            operations = self.gate2OpList(singleGate)
            currGateTime = 0*ns
            for ops in operations:
                currTime = 0*ns
                for idx, op in enumerate(ops):
                    currTime = max(currTime, self.gateLength[op][idx])
                currGateTime += currTime
            totalTime += currGateTime
        return totalTime

def getlength(rbClass, maxtime, interleaved=None, mstart=1, mstep=5):
    """
    get the maximum m (number of clifford gates) which fits in maxtime.
    Make sure to set the gate length first (RBClass.setGateLength)
    @param rbClass: the object of RB
    @param maxtime: the max time of the sequence
    @param interleaved: default is None. the interleaved is given in the same format of the rbClass.randGen
    @param mstart: start value of m (number of clifford gates)
    @param mstep: increase step of m
    @return: the max value of m
    """
    averageover = 5
    stayinloop = True
    m = mstart
    while stayinloop:
        seqLengths = []
        for av in range(averageover):
            seqf = rbClass.randGen(m, interleaved=interleaved, finish=True)
            seqLength = rbClass.computeGateLength(seqf)
            seqLengths.append(seqLength)
        if np.max(seqLengths) > maxtime:
            # sequence too long, decrease m and get out
            m -= mstep
            stayinloop = False
        else:
            m += mstep
    return m


def fitFunc(A=None, B=None):
    """
    the function is in the form:
        Prob = A*p**m + B
    when A or B is given with a value, the parameter is fixed in the function
    @param A: default is None
    @param B: defaulit is None
    @return: function, default p0 for fitting
    """

    p0 = [0.99, np.pi / 4., np.pi / 4]  # free A and B
    p0AorB = [0.99, np.pi / 4.]  # A set
    p0AB = [0.99]  # A and B set

    if A and B:
        return lambda m, p: A*p**m + B, p0AB
    elif A:
        return lambda m, p, B: A*p**m + B, p0AorB
    elif B:
        return lambda m, p, A: A*p**m + B, p0AorB
    else:
        return lambda m, p, A, B: A*p**m + B, p0


def fitData(ms, sequence_fidelities, A=None, B=None, p0=None):
    """
    fitting the RB data, return a parameter dictionary with the error of the fitting parameter
    the function is in the form:
        Prob = A * p**m + B
    if A or B is given with a value, it will be fixed during the fitting procedure.
    @param ms: array of m
    @param sequence_fidelities: array of Prob (or fidelity) corresponding to ms
    @param A: default is None, parameter A
    @param B: default is None, parameter B
    @param p0: default is None, starting p0 in the fitting procedure
    @return: a dictionary with fitting information,
        the full parameters are
        { "A": A, "B": B, "p": p, "Aerr": error of A, "Berr": error of B", "perr": error of p }
    """
    func, p0default = fitFunc(A, B)
    if p0 is None:
        p0 = p0default
    p1, cov = curve_fit(func, ms, sequence_fidelities, p0=p0, maxfev=int(1e5))
    p1err = np.sqrt(np.abs(np.diag(cov)))  # var[element x] = covariance matrix [x,x]
    if A and B:
        params = {'A': A, 'B': B, 'p': p1[0], 'Aerr': 0., 'Berr': 0., 'perr': p1err[0]}
    elif A:
        params = {'A': A, 'B': p1[1], 'p': p1[0], 'Aerr': 0., 'Berr': p1err[1], 'perr': p1err[0]}
    elif B:
        params = {'A': p1[1], 'B': B, 'p': p1[0], 'Aerr': p1err[1], 'Berr': 0., 'perr': p1err[0]}
    else:
        params = {'A': p1[1], 'B': p1[2], 'p': p1[0], 'Aerr': p1err[1], 'Berr': p1err[2], 'perr': p1err[0]}

    return params

def pToErr(p, d):
    """
    from p to error in RB
    err = (1-p)*(1-1/d)
    @param p: float value, p in formular  A*p**m + B
    @param d: d=2 for single qubit, d=4 for two qubits,
              d should 2**numQubit
    @return: error
    """
    r = (1. - 1. / d) * (1. - p)
    return r


def ppgToErr(pref, pgate, d):
    """
    given reference p, gate p and d, compute the gate error
    err = (1 - 1/d) * (1 - Pgate/Pref)
    @param pref: p of reference
    @param pgate: p of gate
    @param d: d=2 for single qubit, d=4 for two qubits, d=2**numQubits
    @return: gate error
    """
    rc = (1 - 1./d) * (1. - pgate / pref)
    return rc

if __name__ == '__main__':
    """ test program """
    rbClass = RBClifford(2)
    gateList = rbClass.randGen(6, interleaved=("CZ","CZ"), finish=True)
    print len(rbClass.cliffordGateSet)
    U = np.eye(4, 4, dtype=np.complex)
    miss = 0
    for idx, gate in enumerate(gateList):
        print("=" * 10 + " No. %s " % (idx+1) + "=" * 10)
        print("\tgate: %s" %str(gate))
        if gate in rbClass.cliffordUnitary:
            U = np.dot(rbClass.cliffordUnitary[gate], U)
        else:
            miss += 1
            print("gate %s not in cliffordUnitary" %(str(gate)))
            U = np.dot(rbClass.computeUnitary([gate]), U)
        # print("%s" % repr(U))
    print miss
    print np.round(U,5)
