import labrad
import numpy as np
from matplotlib import pyplot as plt
cxn = labrad.connect()

 
def todB(x):
    num = float(x)
    db = 10*np.log10(num)
    return db
    
def toNum(x):
    dB = float(x)
    num = 10**(dB/10.0)
    return num

    
