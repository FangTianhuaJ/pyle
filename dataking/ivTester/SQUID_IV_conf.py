import os

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'G:\\labrad\\dev\\pyle\\')

import labrad
from labrad.units import V, mV, us, ns, GHz, MHz

from pyle import registry
from pyle.workflow import switchSession as pss


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
switchSession(user='Daniel')

