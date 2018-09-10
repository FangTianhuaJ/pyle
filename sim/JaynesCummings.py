# states determined by two indeces, qubit and resonator
# qubit: 0 is ground state
# resonator: n is n photons,
from scipy.linalg import expm

from numpy import zeros, arange, dot,sqrt, ones, array, asarray, shape, kron, identity, reshape, pi, diag, trace, transpose, conjugate, sqrt, exp, log, copy, sum, newaxis, real, size, sort
from numpy.linalg import eigvals
from pylab import iterable, subplot, plot, ion, ioff, xlim, ylim, clf, xlabel, ylabel, title, setp, subplots_adjust, show, draw
from scipy.special import erfc

from pyle.analysis.myPlotUtils import plot2d


def gauss(x,x0,sigma):
    return 1.0/sigma/sqrt(2*pi) * exp(-0.5*((x-x0)/sigma)**2)


def pulse(t0,fwhm=8.0,rotations=0.5,phase=0.0):
    sigma = fwhm/sqrt(8*log(2))
    return lambda t: rotations*exp(1.0j*phase)*gauss(t,t0,sigma)



def fzeros(s):
    return zeros(s,dtype=float)

def dagger(m):
    return(transpose(conjugate(m)))

def _a(nmax):
    result = fzeros((nmax+1,nmax+1))
    n=arange(1,nmax+1)
    result[n-1,n] = sqrt(n)
    return result

def _aDagger(nmax):
    result = fzeros((nmax+1,nmax+1))
    n=arange(0,nmax)
    result[n+1,n] = sqrt(n+1)
    return result

def _sigmaP(nqubit=1):
    result = fzeros((nqubit+1,nqubit+1))
    n=arange(0,nqubit)
    result[n+1,n] = 1.0
    return result

def _sigmaM(nqubit=1):
    result = fzeros((nqubit+1,nqubit+1))
    n=arange(0,nqubit)
    result[n,n+1] = 1.0
    return result

def _phaseflip():
    return array([[1.0,0.0],[0.0,-1.0]])

def stdfreq(t):
    return 6.0

def fzero(t):
    return 0.0

def Hjc(nmax=10,nqubit=2,fHO=stdfreq, fTLS=stdfreq, fDrive=stdfreq,
        driveTLS=fzero, driveHO=fzero,
        coupling = fzero,nonlin=0.1):
    a       = kron(identity(nqubit+1), _a(nmax))
    aDagger = kron(identity(nqubit+1), _aDagger(nmax))
    Hres = dot(aDagger,a)
    sigmaP = kron(_sigmaP(nqubit=nqubit), identity(nmax+1))
    sigmaM = kron(_sigmaM(nqubit=nqubit), identity(nmax+1))
    Hcoupling = (dot(aDagger,sigmaM) + dot(a,sigmaP))
    Htls1 = fzeros((nqubit+1,nqubit+1))
    Htls1[1,1] = 1.0
    if nqubit>1:
        Htls2 = fzeros((nqubit+1,nqubit+1))
        Htls2[2,2] = 1.0
    Htls1 = kron(Htls1,identity(nmax+1))
    Htls2 = kron(Htls2,identity(nmax+1))
    return lambda t: 2.0 * pi * ( \
            (fHO(t)-fDrive(t))  * Hres     + \
            (fTLS(t)-fDrive(t)) * Htls1 + \
            (fTLS(t)*2-nonlin-fDrive(t)*2) * Htls2 + \
            0.5*coupling(t)    * Hcoupling + \
            0.5j*driveTLS(t) * sigmaP - 0.5j*conjugate(driveTLS(t))*sigmaM) + \
            1.0j*driveHO(t)  * aDagger  - 1.0j*conjugate(driveHO(t))*a
        #factor 0.5*2pi on qubit drive so that driveTLS is rabi freq
        #no factor 0.5*2pi on resonator drive so that integral driveHO dt is
        #displacement alpha




def spectrum(nmax=3,nqubit=2,fmin=5.8,fmax=6.2,fres=6.,coupling=0.040):
    H = Hjc(nmax,nqubit,fHO=lambda t: fres, fTLS = lambda x: fmin*(1-x)+fmax*x,
            coupling=lambda t: coupling)
    x = arange(1000)
    result = zeros((size(x),(nmax+1)*(nqubit+1)),dtype=float)

    for y in x:
        result[y,:] = eigvals(H(0.001*y))
    result = sort(result,axis=1)
    ioff()
    for y in arange((nmax+1)*(nqubit+1)):
        plot(x,result[:,y])
    show()

def spectroscopy(nmax=3,nqubit=2,fmin=5.8,fmax=6.2,fres=6.,coupling=0.040):
    H = Hjc(nmax,nqubit,fHO=lambda t: fres, fTLS = lambda x: fmin*(1-x)+fmax*x,
            coupling=lambda t: coupling)
    x = arange(1000)
    n = (nmax+1)*(nqubit+1)
    result = zeros((size(x),n),dtype=float)

    for y in x:
        result[y,:] = eigvals(H(0.001*y))
    result = sort(result,axis=1)
    result = abs(result[:,:,None] - result[:,None,:])
    result = reshape(result,(size(x),n**2))
    ioff()
    for y in arange(n**2):
        plot(x,result[:,y])
    show()




# def nonlintest():
#     H = Hjc2(nmax,fHO=lambda t: fres, fTLS1 = lambda t: fres,
#              fTLS2 = lambda x: fmin*(1-x)+fmax*x+nonlin, coupling1=lambda t: coupling1, coupling2 = lambda x: coupling2)
#     x = arange(0,1,0.001)
#     ioff()
#     for y in x:
#         plot([y]*(nmax+1)*4,eigvals(H(y)),'k.')
#     show()

def Ujc(t=arange(0,5000,1), fHO=6.564, fTLS=6.314, fDrive=6.31256, driveTLS=0.0, driveHO=0.0, coupling=38e-3, T1TLS = 6e2, T2TLS=1.5e2, T1HO=1e3, initial=10):
    if not iterable(fTLS):
        fTLS *= ones(len(t))
    if not iterable(fDrive):
        fDrive *= ones(len(t))
    if not iterable(fHO):
        fHO *= ones(len(t))
    if not iterable(driveTLS):
        driveTLS *= ones(len(t))
    if not iterable(driveHO):
        driveHO *= ones(len(t))
    if not iterable(coupling):
        coupling *= ones(len(t))


    if iterable(initial):
        rho = asarray(copy(initial))
    else:
        rho = zeros((2*(initial+1),2*(initial+1)),dtype=complex)
        rho[0,0] = 1.0

    s = shape(rho)
    if s[0] != s[1] or s[0] % 2:
        print "Can't understand density matrix"
        return
    nmax = s[1]/2-1

    a       = kron(identity(2), _a(nmax))
    aDagger = kron(identity(2), _aDagger(nmax))
    Hres = dot(aDagger,a)
    sigmaP = kron(_sigmaP(), identity(nmax+1))
    sigmaM = kron(_sigmaM(), identity(nmax+1))
    Hcoupling = (dot(aDagger,sigmaM) + dot(a,sigmaP))
    Htls = dot(sigmaP,sigmaM)

    phaseflip = kron(_phaseflip(), identity(nmax+1))
    result = zeros((len(t),2*(nmax+1),2*(nmax+1)),dtype=complex)

    result[0] = rho
    for i in arange(len(t)-1):
        dt = t[i+1] - t[i]
        # Jaynes Cummings Hamilitonian with YY coupling
        Hjc = 2.0 * pi * ( \
            (fHO[i]-fDrive[i])  * Hres     + \
            (fTLS[i]-fDrive[i]) * Htls + \
            0.5*coupling[i]    * Hcoupling + \
            0.5j*driveTLS[i] * sigmaP - 0.5j*conjugate(driveTLS[i])*sigmaM) + \
            1.0j*driveHO[i]  * aDagger  - 1.0j*conjugate(driveHO[i])*a
        #factor 0.5*2pi on qubit drive so that driveTLS is rabi freq
        #no factor 0.5*2pi on resonator drive so that integral driveHO dt is
        #displacement alpha


        U = expm(-1.0j*Hjc*dt)
        # Hamiltonian evolution


        rho = dot(dot(U,rho),dagger(U))

        rho -=  (0.5/T1TLS * (dot(Htls,rho) + dot(rho,Htls) \
                          - 2 * dot(dot(sigmaM,rho),sigmaP)) \
             + 0.5/T2TLS * (rho - dot(dot(phaseflip,rho),phaseflip)) \
             + 0.5/T1HO * (dot(Hres,rho) + dot(rho,Hres) - \
                           2 * dot(dot(a,rho),aDagger))) * dt

        result[i+1] = rho


    return result

def plotdensity(t,rho):
    ioff()
    clf()
    s = shape(rho)
    n = s[1]/2
    i = range(s[1])
    # get the probability (diagonal)
    p = real(reshape(rho[:,i,i],(s[0],2,n)))
    print 'P1 at end: %g' % sum(p[-1,1,:])


    ax0=subplot(411)
    plot(t,sum(p[:,1,:],axis=-1))
    plot(t,sum(sum(p,axis=1)*arange(n)[newaxis,:],axis=1))
    ylim(-0.05,1.05)
    ylabel('photon count')
    title('photon number in qubit and resonator')
    setp(ax0.get_xticklabels(), visible=False)
    ax1=subplot(412,sharex=ax0)
    plot2d(transpose(sum(p,axis=1)),(t[0],t[-1],0,n-1),vmin=0,vmax=1)
    ylabel('photons')
    title('probability of resonator states')
    setp(ax1.get_xticklabels(), visible=False)
    ax2=subplot(413,sharex=ax0,sharey=ax1)
    plot2d(transpose(p[:,0,:]),(t[0],t[-1],0,n-1),vmin=0,vmax=1)
    ylabel('photons')
    title('probability of resonator states and empty qubit')
    setp(ax2.get_xticklabels(), visible=False)
    ax3=subplot(414,sharex=ax0,sharey=ax1)
    plot2d(transpose(p[:,1,:]),(t[0],t[-1],0,n-1),vmin=0,vmax=1)
    ylabel('photons')
    xlabel('t (ns)')
    title('probability of resonator states and filled qubit')
    subplots_adjust(hspace=0.2)
    xlim(t[0],t[-1])
    draw()
    ion()


def ramsey(tmax=500, tstep=1.0,undersample=10,fHO=6.564, fTLS=6.314, fDrive=6.31256, coupling=38e-3, T1TLS = 6e2, T2TLS=1.5e2, T1HO=1e3, nmax=10):
    #only the first pulse:
    t = arange(0,tmax,tstep)
    drive = pulse(20,rotations=0.25)
    onepulse = Ujc(t, fHO = fHO, fTLS=fTLS, fDrive=fDrive, driveTLS=drive(t),
                   coupling = coupling, T1TLS = T1TLS, T2TLS=T2TLS,T1HO=T1HO, initial=nmax)
    t2=arange(0,40,tstep)
    m = arange(0,len(t),undersample)
    result = zeros((len(m),2*(nmax+1),2*(nmax+1)),dtype=complex)
    for n,i in enumerate(m):
        result[n,:,:] = Ujc(t2, fHO = fHO, fTLS=fTLS, fDrive=fDrive,
                            driveTLS=drive(t2), coupling = coupling,
                            T1TLS = T1TLS, T2TLS=T2TLS,T1HO=T1HO,
                            initial = onepulse[i,:,:])[-1,:,:]

    return t[m], result








def fockstate(swapTimes, swapAmplitudes, probetime=100,fHO=6.564, fTLS=6.314, fDrive=6.31256, coupling=38e-3, T1TLS = 6e2, T2TLS=1.5e2, T1HO=1e3, initial=10,fwhm=2.0,rotations=0.5):

    nPulses = len(swapTimes)
    if len(swapAmplitudes) != nPulses:
        print 'swapTimes, swapAmplitudes must have same length!'
        return

    indeps = []
    sweep = [[]]

    def addSweeps(paramList, sweep):
        result = []
        for i,p in enumerate(paramList):
            if iterable(p):
                result += [(i, len(indeps))]
                sweep = [s + [ns] for ns in p for s in sweep]
        return result, sweep

    timeScan, sweep = addSweeps(swapTimes, sweep)
    amplScan, sweep = addSweeps(swapAmplitudes, sweep)

    t = arange(-20,probetime,0.2)

    delay = 10



    def stepFunc(curr):
        for i in timeScan:
            swapTimes[i[0]] = curr[i[1]]
        for i in amplScan:
            swapAmplitudes[i[0]] = curr[i[1]]

        start = [0]
        for st in swapTimes:
            start += [start[-1] + 2*delay + st]


        def uwave(t):
            result = 0.0
            for i in range(len(swapTimes)):
                result += pulse(start[i] + delay,rotations=rotations)(t)
            return result

        def bias(t):
            result = fTLS
            for i,st in enumerate(swapTimes):
                result += erf_tophat(start[i] + 2 * delay, st, fwhm=fwhm,amplitude=swapAmplitudes[i])(t)
            return result
        clf()
        plot(t,bias(t))
        plot(t,uwave(t))
        return Ujc(t, fHO=fHO, fTLS=bias(t), fDrive=fDrive, driveTLS=uwave(t), driveHO=0.0, coupling=coupling, T1TLS = T1TLS, T2TLS=T2TLS, T1HO=T1HO, initial=initial)
    result = array([stepFunc(curr) for curr in sweep])
    return result


# gaussian pulse trace creation functions
def nounits(x):
    if hasattr(x,'value'):
        return x.value
    else:
        return x


def gaussian_envelope(t0, w, phase=0.0, amplitude=1.0, sbfreq=0.0):
    """Create an envelope function for a gaussian pulse.

    The pulse is centered at t0, has a FWHM of w, and a specified phase.
    """
    return lambda t: exp(-(t-t0)**2*log(2.0)/(w/2.0)**2) * nounits(amplitude) * exp(1.0j*phase-2.0j*pi*sbfreq*t)


def erf_tophat(start, length, fwhm=2.0, amplitude=1.0, force=False):
    #gaussbandwidth is -3dB frequency of corresponding Gaussian
    #a = pi * gaussbandwidth / sqrt(log(2.0)/2.0)
    #fwhm is FWHM of corresponding Gaussian
    a = 2 * sqrt(log(2.0))/fwhm
    amplitude = nounits(amplitude)
    if force:
        #force the maximum height to be as specified
        if abs(length) < 1e-6:
            return gaussian_envelope(0,fwhm, amplitude = amplitude)
        amplitude /= erfc(- 0.5 * a * length) - erfc(0.5 * a * length)
    else:
        #amplitude is only reached for very long pulses
        amplitude /= 2.0
    return lambda t: amplitude * (erfc(a*(start-t)) - erfc(a*(start+length-t)))

