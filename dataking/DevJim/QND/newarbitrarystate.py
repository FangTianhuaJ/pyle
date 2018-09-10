from numpy import zeros, arange, shape, real, imag, sin, cos, conjugate, reshape, dot, sqrt, array, size, newaxis, arctan, arctan2, pi, exp, sign, log, argmax, sum, asarray
import newstates


def swappulse(state,angle,phase=1.0,conj=False):
    N = shape(state)[1]
    matrix = zeros((2,N,2,N),dtype=complex)
    n=arange(N)
    matrix[1,n,1,n] = cos(0.5*angle*sqrt(n+1))
    matrix[0,n,0,n] = cos(0.5*angle*sqrt(n))
    n=n[:-1]
    matrix[1,n,0,n+1] = -1j * phase * sin(0.5*angle*sqrt(n+1))
    matrix[0,n+1,1,n] = -1j * conjugate(phase) * sin(0.5*angle*sqrt(n+1))
    matrix=reshape(matrix,(2*N,2*N))
    if conj:
        return reshape(conjugate(dot(conjugate(reshape(state,2*N)),matrix)),
                       (2,N))
    else:
        return reshape(dot(matrix,reshape(state,2*N)),(2,N))

def drivepulse(state,angle,phase=1.0,conj=False):
    N = shape(state)[1]
    matrix = zeros((2,N,2,N),dtype=complex)
    n = arange(N)
    phase *= angle/abs(angle)
    angle = abs(angle)
    matrix[0,n,0,n] = cos(0.5*angle)
    matrix[1,n,1,n] = cos(0.5*angle)
    matrix[0,n,1,n] = -1j * conjugate(phase) * sin(0.5*angle)
    matrix[1,n,0,n] = -1j * phase * sin(0.5*angle)
    matrix=reshape(matrix,(2*N,2*N))
    if conj:
        return reshape(conjugate(dot(conjugate(reshape(state,2*N)),matrix)),
                       (2,N))
    else:
        return reshape(dot(matrix,reshape(state,2*N)),(2,N))

        
def phasepulse(state,angle,conj=False):
    result = 1.0*state
    if conj:
        angle = -angle
    result[1,:] *= exp(1j*angle)
    return result

def sequence(state, verbose=False, qubit=None, driveAdjust=-1.0j, swapAdjust=0,
             visualization=newstates.printstate, filenameiter=None):
    """Calculate the sequence to produce an arbitrary state of an HO according to PRL 76 1055. The algorithm is adapted to our system: We can't adjust the phase of the qubit resonator coupling but we can dephase ground and excited state of the qubit with a z pulse. If qubit is a qubit object, calculate the excact pulse sequence, given the qubit parameters."""
    def NoneIter():
        while True:
            yield None
            
    if filenameiter is None:
        filenameiter = NoneIter()
    state=asarray(state).astype(complex)
    state/=sqrt(sum(abs(state**2)))
    n=size(state)-1
    while state[n]==0:
        n-=1
    state = state[newaxis,:] * array([1,0])[:,newaxis]
    swaptimes=zeros(n,dtype=float)
    zpulses = 1.0*swaptimes
    drives = 1.0j*swaptimes
    if verbose:
        visualization(state,'',filename=filenameiter.next())
    while n > 0:
        n-=1

        # phase shift between |g,n> and |e,n-1> and swap
        if state[1,n] == 0:
            time=pi/sqrt(n+1)
            phase = 0
        else:
            time = 1j*state[0,n+1]/state[1,n]
            phase = -arctan2(imag(time),real(time))
            oldstate=state
            state = phasepulse(state,phase,conj=True)
            if verbose:
                visualization(state,'phase', phase, oldstate=oldstate,
                              filename=filenameiter.next())
                visualization(state,filename=filenameiter.next())
            time = arctan(abs(time))
            time = (time + pi * (time<0)) * 2 / sqrt(n+1)
        oldstate=state
        state = swappulse(state, time, conj=True)
        swaptimes[n] = time
        zpulses[n] = phase
        
        if verbose:
            visualization(state,'swap', time, oldstate=oldstate,
                          filename=filenameiter.next())
            visualization(state,filename=filenameiter.next())
        #drive
        if state[0,n] == 0:
            if state[1,n] == 0:
                drive = 0
            else:
                drive=pi * 1j*state[1,n]/abs(state[1,n])
        else:
            ratio = state[1,n]/state[0,n]
            drive = arctan(abs(ratio))
            drive = (drive + pi * (drive < 0)) * 1j*ratio/abs(ratio)*2
        oldstate=state
        state = drivepulse(state, drive, conj=True)
        drives[n] = drive
        if verbose:
            visualization(state,'drive',drive, oldstate=oldstate,
                          filename=filenameiter.next())
            visualization(state,filename=filenameiter.next())
    zpulses[1:] = 1.0*zpulses[:-1]
    zpulses[0] = 0
    if qubit is None:
        return drives, swaptimes, zpulses
    # else:
        # ##modify below
        # detuning = 2*pi*(qubit.resonanceFrequency['GHz'] - \
            # qubit.resonatorFrequency['GHz'])
        # sbfreq = 2*pi*(qubit.resonanceFrequency['GHz'] - \
            # qubit.anritsuFrequency['GHz'])
        
        # swapTimes = (qubit.swapLength['ns']+swapAdjust)*swaptimes/pi

        # offresTimes = 2*qubit.delayAfterPi['ns'] # minimum time off resonance
        # offresTimes = offresTimes + \
            # ((-sign(detuning)*(zpulses - offresTimes * detuning)) % (2*pi))/abs(detuning)
        # print offresTimes
        # swapStarts = 1.0 * offresTimes
        # swapStarts[1:] += (offresTimes + swapTimes)[:-1].cumsum()
        # drivepulseCenter = swapStarts - 0.5*offresTimes
        # n = arange(size(swapTimes))
        # def flux(t):
            # result = 0
            # for i in n:
                # result = result + erf_tophat(swapStarts[i],swapTimes[i],
                                             # amplitude=qubit.swapAmplitude,fwhm=0.0)(t)
            # return result
        # def uwave(t):
            # result = 0
            # for i in n:
                # result = result + gaussian_envelope(drivepulseCenter[i],
                                                    # qubit.piLength['ns'])(t) * \
                    # qubit.piAmplitude/pi*drives[i] * \
                    # driveAdjust * exp(-1.0j*sbfreq*(t-swapStarts[i]))      
            # return result
        # return uwave, flux



    

