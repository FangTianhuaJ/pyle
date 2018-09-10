import labrad
import numpy as np
import averageScript as av
import SNTJFit_dataVault as fit
from matplotlib import pyplot as plt
import SNTJDataCode as sn
cxn = labrad.connect()

 
def todB(x):
    num = float(x)
    db = 10*np.log10(num)
    return db
    
def toNum(x):
    dB = float(x)
    num = 10**(dB/10.0)
    return num

    
sa = cxn.spectrum_analyzer_server
sa.select_device('Vince GPIB Bus - GPIB1::18')

fn = cxn.agilent_33120a_generator
fn.select_device('Vince GPIB Bus - GPIB1::13')