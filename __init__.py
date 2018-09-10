# bring in basic stuff to the main pyle namespace

__version__ = '1.0'

import datasaver
import envelopes
import examples
import math
import pipeline
import plotting
import tomo
import util
import workflow
import analysis
import dataking
import fitting
from util import registry_wrapper2 as registry

from datasaver import Dataset
from dataking.qubitsequencer import QubitSequencer

#TODO: get rid of this. Imprt * is verboten!
from math import *
