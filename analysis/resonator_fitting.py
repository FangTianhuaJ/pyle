# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import labrad
from pyle.datavault import DataVaultWrapper
from scipy import optimize
from scipy import interpolate

class S21Func(object):
    """
    S21 trace has the form of:
        1 - Q/Qc*exp(i*phi0)* 1/(1 + 2j*Q*(f-f0)/f0)
    invS21 (S21^-1) has the form of:
        1 + Qi/Qc*exp(i*phi1)* 1/(1 + 2j*Qi*(f-f0)/f0)
    From XiangLiang
    """
    def __init__(self, target='invS21'):
        assert target in ["invS21", "S21"]
        self.target = target
        self.func = self.get_func()

    def get_func(self):
        if self.target == 'invS21':
            return self.invS21Func
        elif self.target == 'S21':
            return self.s21Func

    def __call__(self, x, p):
        return self.func(x, *p)

    @staticmethod
    def invS21Func(x, x0, Qi, Qc, phi):
        val = 1 + Qi/Qc*np.exp(1j*phi)/(1+1j*2*Qi*(x-x0)/x0)
        return val

    @staticmethod
    def s21Func(x, x0, Q, Qc, phi):
        val = 1 - Q/Qc*np.exp(1j*phi)/(1+2j*Q*(x-x0)/x0)
        return val

class S21Curve(object):
    def __init__(self, freq, mag, phase, dB=True):
        self.freq = np.array(freq)
        self.phase = np.unwrap(np.array(phase))
        mag = np.array(mag)
        if dB:
            self.mag = 10**(mag/20.0)
        else:
            self.mag = mag
        self.S21 = self.mag*np.exp(1j*self.phase)
        self.invS21 = 1./self.S21

    def plotS21mag(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.plot(self.freq, self.mag, '.-')
        ax.set_xlabel("Freq")
        ax.set_ylabel("Mag")

    def plotS21phase(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.plot(self.freq, self.phase, '.-')
        ax.set_xlabel("Freq")
        ax.set_ylabel("Phase")

    def plotinvS21(self, ax=None, cycle=True, *arg, **kw):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        if cycle:
            ax.plot(self.invS21.real, self.invS21.imag, *arg, **kw)
            ax.plot(self.invS21.real[0], self.invS21.imag[0], '*', markersize=10, *arg, **kw)
            ax.set_xlabel(r"$\mathrm{Re}[S_{21}^{-1}]$")
            ax.set_ylabel(r"$\mathrm{Im}[S_{21}^{-1}]$")
            ax.set_aspect("equal")
        else:
            ax.plot(self.freq, self.invS21.real, label='Re', *arg, **kw)
            ax.plot(self.freq, self.invS21.imag, label='Im', *arg, **kw)
            ax.set_xlabel("Freq")

    def fittinginvS21(self, p0):
        """
        p0 = (Qi, Qc, phi)
        """
        s21_func = S21Func('invS21')
        idx = np.argmin(self.mag)
        f0 = self.freq[idx]
        p0 = [f0, p0[0], p0[1], p0[2]]
        errFunc = lambda p: np.abs(self.invS21-s21_func(self.freq, p))
        res = optimize.leastsq(errFunc, p0, full_output=True)
        popt, pcov, infodict, errmsg, ier = res
        # estimate the error of parameters
        s_sq = (infodict['fvec']**2).sum()/(len(self.freq)-len(p0))
        pcov = pcov * s_sq
        if ier not in [1,2,3,4]:
            raise RuntimeError("Optimal parameters not found: " + errmsg)
        return popt, pcov

    def fittingS21(self, p0):
        """
        p0 = (Q, Qc, phi)
        """
        s21_func = S21Func('S21')
        idx = np.argmin(self.mag)
        f0 = self.freq[idx]
        p0 = [f0, p0[0], p0[1], p0[2]]
        errFunc = lambda p: np.abs(self.S21-s21_func(self.freq, p))
        res = optimize.leastsq(errFunc, p0, full_output=True)
        popt, pcov, infodict, errmsg, ier = res
        # estimate the error of parameters
        s_sq = (infodict['fvec']**2).sum()/(len(self.freq)-len(p0))
        pcov = pcov * s_sq
        if ier not in [1,2,3,4]:
            raise RuntimeError("Optimal parameters not found: " + errmsg)
        return popt, pcov

    def fitting(self, invFit=True):
        """
        fitting the parameter of S21 curve
        When invFit is True, we fit the data S21^(-1), else we fit the data S21.
        It is recommended to fit S21^(-1)
        @param invFit: default is True
        @return: the fitting parameter (f0, Qi, Qc, phi) and the covariance matrix
        """
        if invFit:
            p0 = self._guess_para(target='invS21')
            return self.fittinginvS21(p0)
        else:
            p0 = self._guess_para(target='S21')
            return self.fittingS21(p0)

    def _guess_para(self, target='invS21'):
        if target == 'invS21':
            z = self.invS21 - 1.0
        elif target == 'S21':
            z = 1.0 - self.S21
        # The following calculates the center of the resonance in the complex plane.
        # The basic idea is to find the mean of the max and the min of both the realz
        # and imagz.  This would give the center of the circle if this were a true
        # circle.  However, since the resonance is not a circle we find the center by
        # rotating the resonance by an angle, finding the mean of the max and the min
        # of both the realz and imagz of the rotated circle, then rotating this new
        # point back to the original orientation. Finally, the middle of the resonance
        # is given by finding the mean of all these rotated back ave max min values.
        # Note: we only need to rotate a quarter of a turn because anything over
        # that would be redundant.
        steps = 100
        centerpoints = np.array(range(steps), dtype=complex)
        for ang in range(steps):
            rotation = np.exp((2j * np.pi * (ang+1) / steps) / 4)
            # the 4 here is for a quarter turn
            zrot = rotation*z
            re = (zrot.real.max() + zrot.real.min()) / 2.
            im = (zrot.imag.max() + zrot.imag.min()) / 2.
            centerpoints[ang] = complex(re,im) / rotation
            # here the new center point is rotated back
        center=centerpoints.mean()
        # Finding an estimate for the diameter of a circle that would fit the data
        diameter = 2 * np.mean(np.abs(z-center))
        # Find the rotated angle from the none symmetry
        # We assumed the curve is normalized, which means the rotate point is
        # fixed at complex(0, 0)
        arrow = (center - np.complex(0, 0))
        angle_rot = np.angle(arrow)
        # This finds an approximation to the resonant frequncy located at an angle of zero
        # the frequency of the 3dB points which are located at pi/2 and -pi/2
        angles = np.angle((z - center) / arrow)
        angleindx = np.logical_and(angles>-2,angles<2)
        freqinterp = self.freq[angleindx]
        anginterp = angles[angleindx]
        freqL = np.median(interpolate.sproot(
                          interpolate.splrep(freqinterp,anginterp-np.pi/2) ) )
        freqR = np.median(interpolate.sproot(
                          interpolate.splrep(freqinterp,anginterp+np.pi/2) ) )
        f0 = np.median(interpolate.sproot(interpolate.splrep(freqinterp,anginterp)))
        Qx = f0 / (freqR - freqL)
        phi = angle_rot
        Qc = Qx/diameter
        return (Qx, Qc, phi)


    def plotFittingResult(self, p0=None):
        """
        plot invS21 and the fitting result
        fitting with invS21, p0: [Qi, Qc, phi], default is None.
        """
        s21_func = S21Func('invS21')
        if p0 is None:
            p, cov = self.fitting(invFit=True)
        else:
            p, cov = self.fittinginvS21(p0)
        fig = plt.figure(figsize=(10, 5))
        f = np.linspace(np.min(self.freq), np.max(self.freq), 5*len(self.freq)+1)
        invS21 = s21_func(f, p)
        S21 = 1./invS21
        f0 = p[0]
        mag = 20*np.log10(np.abs(S21))
        ang = np.angle(S21)

        ax = fig.add_subplot(121)
        ax.plot(self.invS21.real, self.invS21.imag, '.')
        ax.plot(invS21.real, invS21.imag)
        ax.plot(self.invS21.real[0], self.invS21.imag[0], 'r*', markersize=10)
        ax.set_xlabel(r"$\mathrm{Re}[S_{21}^{-1}]$")
        ax.set_ylabel(r"$\mathrm{Im}[S_{21}^{-1}]$")
        ax.set_aspect("equal")

        ax2 = fig.add_subplot((222))
        ax2.plot(self.freq-f0, 20*np.log10(self.mag), '.')
        ax2.plot(f-f0, mag)
        ax2.set_xlabel(r"$f-f_0$ [MHz]")
        ax2.set_ylabel(r"$|S_{21}|$ [dB]")
        ax2.grid()

        ax3 = fig.add_subplot((224))
        ax3.plot(self.freq-f0, self.phase/np.pi, '.')
        ax3.plot(f-f0, ang/np.pi)
        ax3.set_xlabel(r"$f-f_0$ [MHz]")
        ax3.set_ylabel(r"Angle [$\pi$]")
        ax3.grid()

        fig.tight_layout()
        print(self.formatFittingResult(p, cov))
        return p, cov

    @staticmethod
    def formatFittingResult(p, cov):
        cov = np.sqrt(np.diag(cov))
        res = "f0 = %.2f +/- %.2f\n" %(p[0], cov[0])
        res += "Qi = %.3e +/- %.3e\n" %(p[1], cov[1])
        res += "Qc = %.3e +/- %.3e\n" %(p[2], cov[2])
        res += "phi = %.4f +/- %.4f\n" %(p[3], cov[3])
        return res

def calibrate(dataS21, caliS21, magOrder=0, phaseOrder=1):
    """
    calibrate S21 curve
    @param dataS21: the S21Curve that needs be calibrated
    @param caliS21: the calibration S21Curve
    @param magOrder: the fitting order of magnitude, default=0
    @param phaseOrder: the fitting order of phase, default=1
    @return: 
    """
    assert (magOrder>=0) and (phaseOrder>=1)
    f_dat = dataS21.freq
    f_cal = caliS21.freq
    mask_left = f_cal<=np.min(f_dat)
    mask_right = f_cal>=np.max(f_dat)
    mask = np.logical_or( mask_left, mask_right )

    # phase fitting
    # p_left = np.polyfit(f_cal[mask_left], caliS21.phase[mask_left], 1)
    # p_right = np.polyfit(f_cal[mask_right], caliS21.phase[mask_right], 1)

    p_phase = np.polyfit(f_cal[mask], caliS21.phase[mask], phaseOrder)
    phase_corr = dataS21.phase - np.poly1d(p_phase)(f_dat)
    a = phase_corr[0]
    b = np.mod(a+np.pi, 2*np.pi) - np.pi
    shift = a - b
    phase_corr = phase_corr - shift

    # mag calibrate
    if magOrder == 0:
        mean_mag = np.mean(caliS21.mag[mask])
        mag_corr = dataS21.mag / mean_mag
    else:
        p_mag = np.polyfit(f_cal[mask], caliS21.mag[mask], magOrder)
        mag_corr = dataS21.mag / (np.poly1d(p_mag)(f_dat))

    return S21Curve(f_dat, mag_corr, phase_corr, dB=False)

def resonator_fitting(session, dataId, caliId, magOrder=0, phaseOrder=1):
    """
    @param session, the directory in the datavault
    @param dataId, datasetId of the data
    @param caliId, datasetId of the calibration data
    @param magOrder, the order of fitting for magnitude
    @param phaseOrder, the order of fitting for phase
    return calibrated S21Curve.
    """
    with labrad.connect() as cxn:
        dvw = DataVaultWrapper(session, cxn)
        dataS21 = S21Curve(dvw[dataId][:,0], dvw[dataId][:,1], dvw[dataId][:,2])
        caliS21 = S21Curve(dvw[caliId][:,0], dvw[caliId][:,1], dvw[caliId][:,2])
    S21 = calibrate(dataS21, caliS21, magOrder, phaseOrder)
    return S21

def plotS21(dataset, session):
    """
    plot S21 more conveniently
    """
    with labrad.connect() as cxn:
        dvw = DataVaultWrapper(session, cxn)
        dat = dvw[dataset]
        S21 = S21Curve(dat[:,0], dat[:,1], dat[:,2])
    fig = plt.figure(figsize=(10,4.5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    S21.plotS21mag(ax1)
    S21.plotS21phase(ax2)
    ax1.set_xlabel("Freq")
    ax1.set_ylabel("Mag")
    ax1.grid()
    ax2.set_xlabel("Freq")
    ax2.set_ylabel("Phase")
    ax2.grid()
    fig.tight_layout()
    return fig

if __name__ == '__main__':
    session = ['', 'test', 'xmon4qubit', '170603']
    S21 = resonator_fitting(session, dataId=4, caliId=3)
    S21.plotS21mag()
    S21.plotS21phase()
    # S21.plotinvS21(cycle=True)
    S21.plotFittingResult()
    plt.show()