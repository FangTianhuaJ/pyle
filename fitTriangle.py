import numpy as np
from scipy.optimize import leastsq
import pylab as pl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def fit_cos_curve(data, para_guess=[1.0,1.0,0,0.0], plot=True, fourier=False, nfftpoints=10000):

    def func(x,p):
        amp, freq, phi, offset = p
        return amp*np.sin(2*np.pi*freq*x+phi)+offset

    def residuals(p,y,x):
        return y-func(x,p)

    def maxFreq(data, nfftpoints):
	    ts, ps = data.T
	    y = ps - np.polyval(np.polyfit(ts, ps, 1), ts) # detrend
	    timestep = ts[1] - ts[0]
	    freq = np.fft.fftfreq(nfftpoints, timestep)
	    fourier = abs(np.fft.fft(y, nfftpoints))
	    Freq_fit = abs(freq[np.argmax(fourier)]) # GHz
	    FFTx = 1000*np.fft.fftshift(freq)
	    FFTy = np.fft.fftshift(fourier)
	    return Freq_fit,FFTx,FFTy,1000*Freq_fit

    indeps = data[0]
    deps = data[1]
    indeps_new = np.arange(np.min(indeps),np.max(indeps),0.001*np.abs(indeps[1]-indeps[0]))
    deps_new = interp1d(indeps,deps)(indeps_new)
    indeps_new_2pi, indeps_new_2pi_int, div = indeps_new/(1.0), np.round(indeps_new/(1.0)), []

    for i in range(len(indeps_new_2pi)):
    	if indeps_new_2pi_int[i] != 0:
	        divi = indeps_new_2pi[i]/indeps_new_2pi_int[i]
	        div.append(divi)
    div = np.array(div)
    deps_new_2pi = deps_new[np.where(np.abs(div-1)<0.001)]

    deps_sort = np.sort(deps)
    select_num = int(len(deps_sort)/50.0)+1
    c = np.column_stack((indeps,deps))
    data_amp = (np.mean(deps_sort[-select_num:])-np.mean(deps_sort[:select_num]))/2.0
    data_freq = maxFreq(np.column_stack((indeps,deps)), nfftpoints)[0]
    data_phi = np.mean(deps_new_2pi)
    data_offset = (np.mean(deps_sort[:select_num])+np.mean(deps_sort[-select_num:]))/2.0

    if para_guess == None:
        para_guess = [data_amp,data_freq,data_phi,data_offset]
    else:
        para_guess = para_guess

    para_fit = leastsq(residuals,para_guess,args=(deps,indeps))[0]
    deps_fit = func(x,para_fit)
    print 'fitting parameters is: \ndata_amp={}, data_freq={}\ndata_phi={}, data_offset={}'.format(\
    	para_fit[0],para_fit[1],para_fit[2],para_fit[3])

    if plot:
        plt.figure(figsize=(8,6))
        plt.scatter(indeps,deps,color="red",label="Init data",linewidth=3) 
        plt.plot(indeps,deps_fit,color="orange",label="Fitting line",linewidth=2) 
        plt.legend(loc=1)
        plt.show()
    if fourier:
    	maxFreq = maxFreq(np.column_stack((indeps,deps)), nfftpoints)
    	FFTx, FFTy, Freq_fit = maxFreq[1], maxFreq[2], maxFreq[3]
    	fig = plt.figure(figsize=(8,6))
    	ax = fig.add_subplot(1,1,1)
    	ax.plot(FFTx, FFTy)
    	ax.set_xlabel('FFT Frequency [MHz]')
    	ax.set_ylabel('FFT Amplitude')
    	ax.set_title('Freq_fit = '+str(Freq_fit)+'MHz')
    	plt.show()
    return para_fit

if __name__ == '__main__':
    x = np.linspace(0,10,100)
    y = 5*np.sin(2*np.pi*1.0*x+19)+27+0.5*np.random.rand(len(x))
    data = [x,y]
    fit_cos_curve(data, para_guess=None, plot=True, fourier=True, nfftpoints=10000)