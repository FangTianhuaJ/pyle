import os
import numpy as np
import matplotlib.pyplot as plt
import sys

import labrad
from labrad.units import Unit,Value

s,Hz,MHz = (Unit(u) for u in ['s','Hz','MHz'])

from pyle.workflow import switchSession as pss #(P)yle(S)witch(S)ession
from pyle.util import sweeptools as st
import noiseTest

def switchSession(session=None, user=None):
    """Switch the current session, using the global connection object"""
    global s
    if user is None:
        user = s._dir[1]
    s = pss(cxn, user, session, useDataVault=True)

# connect to labrad and setup a wrapper for the current sample
cxn = labrad.connect()
user = raw_input('User name: ')
switchSession(user=user)

sr = cxn.signal_analyzer_sr770

