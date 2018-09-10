import numpy as np
from scipy.optimize import fminbound
import matplotlib.pyplot as plt
from math import *

from labrad.units import Unit
V, mV, us, ns, GHz, MHz = [Unit(s) for s in ('V', 'mV', 'us', 'ns', 'GHz', 'MHz')]

from pyle.plotting import dstools
from pyle import registry
from pyle.dataking import util

def fitspectroscopy(ds, datasetNumber=None, session=None, order=2, frange=None):
    everything = dstools.getDeviceDataset(ds, datasetId=datasetNumber, session=session)
    data = everything['data']
    if frange is not None:
        freqs = data[:,1]
        data = data[np.logical_and(freqs>min(frange),freqs<max(frange))]
    freqflux = np.empty([0,2])
    uniquefreqs = np.unique(data[:,1])
    for freq in uniquefreqs:
        datafreq = data[data[:,1]==freq]
        freqflux = np.vstack((freqflux,[freq,datafreq[np.argmax(datafreq[:,2]),0]]))
    poly = np.polyfit(freqflux[:,0]**4,freqflux[:,1],order)    
    return poly


def getQBParams(cxn, Vp=None, Vl=None, L=720e-12, measure=0, session=None, dir=None, dv=None, datasetNumber=None, order=2, frange=None):
    """Get qubit parameters (Ic, C, potential well height dU, delta).
    Inputs are Vp - period of squid steps, Vl - length of squid step,
    both in Volts. L - design inductance.
    """
    
    phi0=2.067833636e-15
    hbar=1.05457148e-34
    
    """References are to Martinis, 'Mapping of Flux-Bias to Current-
    Bias Circuits in a Phase Qubit' (maptocbjj.pdf) and to Cooper,
    'Deriving Junction Capacitance from Spectroscopy and Steps'
    (CapDerivation.pdf), both in T:\Notes_Derivations.
    """
    
    # Get copy of registry (without changing session).
    if session is None and dir is None:
        raise Exception('Dummy, you must specify either the session or the directory.')
    if session is not None and dir is not None:
        raise Exception('You can only specify one of session or dir.')
    if session is not None:
        dir = session._dir
    reg = registry.RegistryWrapper(cxn,dir)
    q = [reg.copy()[q] for q in reg['config']][measure] #Returns registry in usual format.
    
    #Find del0 that satisfies eq 10 in John's current to flux bias paper.
    #del0 = pi/2 for current biased junctions
    #phic is the critical flux
    phic = (Vl/2)/Vp*phi0 #Cooper Eqns4,12
    del0 = fminbound(lambda x: abs(phic-(phi0/2/pi)*(x-tan(x))),0,2*pi) #Cooper Eq4
    
    #Calculate inductances and critical current
    LJ0 = -L*cos(del0) #Cooper Eq3
    Ic = phi0 / (2*pi*LJ0)
    
    #Determine critical voltage from 2D spectroscopy
    if datasetNumber is None:
        Vse = q['biasStepEdge'][V]
        fluxFunc = q['calFluxFunc']
        Vc = Vse+float(fluxFunc[-1])
    else:
        if dv is None: dv=cxn.data_vault
        fluxFunc = fitspectroscopy(dv,datasetNumber=datasetNumber, session=dir, order=order, frange=frange)
        Vc = fluxFunc[-1]
    
    #Calculate capacitance
    m = (2*pi*1e9/0.9)**4 / float(fluxFunc[-2]) #Cooper Eq19 - Convert f to wp
    C = (1/L)*sqrt(abs((4*pi*tan(del0))/(m*Vp))) #Cooper Eqs10,14,16
    
    #Calculate flux
    Vop = q['biasOperate'][V]
    phi = phic + phi0*(Vop - Vc)/Vp * np.sign(tan(del0)*m*Vp) #Cooper Eqs9,11
    
    #Determine current parameters for Martinis
    I1 = (phi-del0*phi0/(2*pi))/L #Cooper Eq2 - I'
    Ic1 = Ic*sin(del0) #Cooper Eq1 - I0'
    
    #Determine barrier height. This is done by taking Martinis Eq8 (giving
    #   U(del)) and finding del such that U'(del)=0 @ cos(del)=I1/Ic1.
    #   Then, substitute this back into U(del). Factor of two comes from
    #   +/- solutions (or max/min solutions).
    delU = 2*((Ic1*phi0)/(2*pi))*(sqrt(1-(I1/Ic1)**2)-(I1/Ic1)*acos(I1/Ic1))
    
    #Solve for plasma frequency
    wp = sqrt((2*pi*Ic1)/(C*phi0))*sqrt(sqrt(1-(I1/Ic1)**2)) #Cooper Eqs5,6
    
    # Solve for phase delta (del1,del2 two ways to find delta')
    del1 = del0*asin(I1/Ic1)/(pi/2)
    del2 = fminbound(lambda x: (-Ic*phi0*cos(x))/(2*pi) + (phi-x*phi0/(2*pi))**2/(2*L) , 0, del0)
    
    #Grab other qubit parameters from registry
    f01 = q['f10'][GHz]
    
    #Return qubit parameters as dictionary
    qubitParams = {'Phi_c':phic,'delta_0':del0,'LJ0':LJ0,'Ic':Ic,'C':C,'Phi':phi,'Iprime':I1,'Icprime':Ic1,'omega_p':(wp/2*pi),'delta1':del1,'delta2':del2,'DeltaU':delU,'f10':f01,'delU/hwp':(delU/(hbar*wp))}
    return qubitParams