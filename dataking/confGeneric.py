import os

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'Your drive path here')

import labrad
from labrad.units import Unit,Value

rad, ns, us, s, MHz, GHz, mV, V, dBm = (Unit(s) for s in ['rad','ns','us','s','MHz','GHz','mV','V','dBm'])

from pyle import registry
from pyle import envelopes as env
from pyle.dataking import dephasingSweeps
from pyle.dataking import envelopehelpers as eh
from pyle.dataking import fpgaseq
from pyle.dataking import multiqubitPQ as mq
from pyle.util import sweeptools as st
from pyle.workflow import switchSession as pss
from pyle.plotting import dstools as ds

def switchSession(session=None, user=None):
    """Switch the current session, using the global connection object"""
    global s
    if user is None:
        user = s._dir[1]
    s = pss(cxn, user, session)


def pipe_filling_factor(board_group=('DR Direct Ethernet', 1)):
    """Download performance data from the GHz DACs and calculate
    the fraction of time that the GHz DAC pipeline is full.
    """
    with labrad.connect() as cxn:
        perf = dict(cxn.ghz_dacs.performance_data())
    times = perf[board_group][3].asarray
    error_rate = sum(times == 0) / float(len(times))
    return 1 - error_rate


# connect to labrad and setup a wrapper for the current sample
cxn = labrad.connect()
switchSession(user='Username')

