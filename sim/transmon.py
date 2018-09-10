import numpy as np
import scipy.interpolate as interpolate

EJ_OVER_EC_RANGE = (50, 200)

def hamiltonian_realSpace(EjOverEc, N=500, alpha=0.0):
    """The discretized real space (phase) Hamiltonian for the transmon"""
    delta = np.linspace(-np.pi, np.pi, N) #includes endpoints
    ddelta = delta[1] - delta[0]
    
    derivTerm = -4 * 1.0/(ddelta**2) * ( np.diag([1]*(N-1), -1) + np.diag([-2]*N) + np.diag([1]*(N-1), 1) )
    potentialTerm = -1.0 * EjOverEc * np.diag(np.cos(delta))

    if alpha:
        potentialTerm += np.diag([-alpha*Ej*np.sqrt(1-(np.sin(d)/alpha)**2) for d in delta])
    
    H = derivTerm + potentialTerm
    return delta, H
    
def analyzeH(H):
    """Get ordered eigenvalues and eigenvectors of a Hamiltonian"""
    eval,evec = np.linalg.eig(H)
    #Sort eigenvalues and eigenvectors by increasing energy
    inds=np.argsort(eval)
    eval=eval[inds]
    evec=evec[:,inds]
    
    return eval, evec

def anharmVsRatio(ratios, N, noisy=False):
    """Calculate transmon parameters as Ej/Ec is varied"""
    anharms = np.array([])
    E10s = np.array([])
    for i, ratio in enumerate(ratios):
        if noisy:
            print 'Ratio %d' %i
        deltas, H = hamiltonian_realSpace(ratio, N=N)
        eval, evec = analyzeH(H)
        E10 = eval[1] - eval[0]
        E21 = eval[2] - eval[1]
        anharm = (E21 - E10)/E10
        anharms = np.hstack((anharms, anharm))
        E10s = np.hstack((E10s, E10))
    return np.vstack((ratios, E10s, anharms)).T

#One time computation of interpolating functions
result = anharmVsRatio(np.linspace(EJ_OVER_EC_RANGE[0], EJ_OVER_EC_RANGE[1], 50), 300)
_anharm2EjOverEc = interpolate.interp1d(result[:,2], result[:,0], kind='cubic')
_EjOverEc2E10OverEc = interpolate.interp1d(result[:,0], result[:,1], kind='cubic')

def anharm2EjOverEc(anharm):
    data = _anharm2EjOverEc(anharm)
    return data
    
def EjOverEc2E10OverEc(EjOverEc):
    data = _EjOverEc2E10OverEc(EjOverEc)
    return data
        
def freqAnharm2EjEc(freq, anharm):
    """Junction and capacitive characterisitic frequencies as a function of f10 and relative anharmonicity"""
    EjOverEc = anharm2EjOverEc(anharm)
    E10OverEc = EjOverEc2E10OverEc(EjOverEc)
    Fc = freq / E10OverEc
    Fj = EjOverEc * Fc
    return Fj, Fc