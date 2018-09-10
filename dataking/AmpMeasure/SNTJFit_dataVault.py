
import labrad
from pylab import*
import time
import math
import numpy
import scipy
from scipy import optimize
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

k = 1.381e-23    # boltzmann's constant
e = 1.602e-19    # charge of an electron
h = 6.62e-34      # plancks constant



def fitSNTJ(cxn, dir, fileNumber, plotBool = True,
            voltEnf = False, voltVar = False, freqVar = False,
            gainEnf = False, tempEnf = False, noiseEnf = False,
            truncate = False, limit = [1,1]):
    voltages = np.array([])
    finalData = np.array([])
    params = np.array([])
    voltDiff = 1.1



    dv = cxn.data_vault
    dv.cd(dir)
    dv.open(fileNumber)
    data = np.asarray(dv.get())

    suggestedGain = dv.get_parameter('guessGain')
    suggestedNoiseTemp = dv.get_parameter('guessNoise')
    suggestedTemp = dv.get_parameter('guessTemp')
    f = dv.get_parameter('freq')*1e6

    points = len(data[:,0])
    if truncate:
        for i in range(points):
            if data[i,0]>limit[0] and data[i,0]<limit[1]:
                voltages = np.append(voltages, float(data[i,0]))
                finalData = np.append(finalData, float(data[i,1]))
    else:
        for i in range(points):
            voltages = np.append(voltages, float(data[i,0]))
            finalData = np.append(finalData, float(data[i,1]))
    suggestedGain
    suggestedNoiseTemp
    suggestedTemp
    vOffset = 0.00001
    amplitude = 2*dv.get_parameter('inputAttenuation')*dv.get_parameter('voltageAmplitude')
    if gainEnf:
        guess = [suggestedNoiseTemp, suggestedTemp, vOffset]

        def fitfunc( V, b, c, d):
            y = suggestedGain*(b + (((h*f-e*(V-d))/(4*k))*((np.cosh((h*f-e*(V-d))/(2.0*k*c)))/(np.sinh((h*f-e*(V-d))/(2.0*k*c)))))+
                               (((h*f+e*(V-d))/(4*k))*((np.cosh((h*f+e*(V-d))/(2.0*k*c)))/(np.sinh((h*f+e*(V-d))/(2.0*k*c))))))
            return y


        popt, pcov = curve_fit(fitfunc, voltages, finalData, p0=guess)
        params = popt

        print 'noise = ', popt[0]
        print 'temp = ', popt[1]
        print 'offset = ', popt[2]

        x1 = np.linspace(-amplitude, amplitude, 5000)
        y1 = fitfunc(x1, popt[0], popt[1], popt[2])/suggestedGain

        if plotBool:
            plt.plot(voltages, finalData/suggestedGain, 'o', label='data')
            plt.plot(x1,y1, label='fit')
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.show()

    elif tempEnf:
        guess = [suggestedGain, suggestedNoiseTemp, vOffset]

        def fitfunc( V, a, b, d):
            y = a*(b + ((h*f-e*(V-d))/(4*k))*((np.cosh((h*f-e*(V-d))/(2.0*k*suggestedTemp)))/(np.sinh((h*f-e*(V-d))/(2.0*k*suggestedTemp))))+
                   ((h*f+e*(V-d))/(4*k))*((np.cosh((h*f+e*(V-d))/(2.0*k*suggestedTemp)))/(np.sinh((h*f+e*(V-d))/(2.0*k*suggestedTemp)))))
            return y


        popt, pcov = curve_fit(fitfunc, voltages, finalData, p0=guess)
        params = popt

        print 'gain = ', 10*np.log10(popt[0])
        print 'noise = ', popt[1]
        print 'offset = ', popt[2]

        x1 = np.linspace(-amplitude, amplitude, 5000)
        y1 = fitfunc(x1, 1, popt[1], popt[2])

        if plotBool:
            plt.plot(voltages, finalData/popt[0], 'o', label='data')
            plt.plot(x1,y1, label='fit')
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.show()

    elif noiseEnf:
        guess = [suggestedGain, suggestedTemp, vOffset]

        def fitfunc( V, a, c, d):
            y = a*(suggestedNoiseTemp + ((h*f-e*(V-d))/(4*k))*((np.cosh((h*f-e*(V-d))/(2.0*k*c)))/(np.sinh((h*f-e*(V-d))/(2.0*k*c))))+
                   ((h*f+e*(V-d))/(4*k))*((np.cosh((h*f+e*(V-d))/(2.0*k*c)))/(np.sinh((h*f+e*(V-d))/(2.0*k*c)))))
            return y


        popt, pcov = curve_fit(fitfunc, voltages, finalData, p0=guess)
        params = popt

        print 'gain = ', 10*np.log10(popt[0])
        print 'temp = ', popt[1]
        print 'offset = ', popt[2]

        x1 = np.linspace(-amplitude, amplitude, 5000)
        y1 = fitfunc(x1, 1, popt[1], popt[2])

        if plotBool:
            plt.plot(voltages, finalData/popt[0], 'o', label='data')
            plt.plot(x1,y1, label='fit')
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.show()

    elif freqVar:
        guess = [suggestedGain, suggestedNoiseTemp, suggestedTemp, vOffset, f]

        def fitfunc( V, a, b, c, d, fr):
            y = a*(b + ((h*fr-e*(V-d))/(4*k))*((np.cosh((h*fr-e*(V-d))/(2.0*k*c)))/(np.sinh((h*fr-e*(V-d))/(2.0*k*c))))+
                   ((h*fr+e*(V-d))/(4*k))*((np.cosh((h*fr+e*(V-d))/(2.0*k*c)))/(np.sinh((h*fr+e*(V-d))/(2.0*k*c)))))
            return y


        popt, pcov = curve_fit(fitfunc, voltages, finalData, p0=guess)
        params = popt

        print 'gain = ', 10*np.log10(popt[0])
        print 'noise = ', popt[1]
        print 'temp = ', popt[2]
        print 'offset = ', popt[3]
        print 'freq = ', popt[4]

        x1 = np.linspace(-amplitude, amplitude, 5000)
        y1 = fitfunc(x1, 1, popt[1], popt[2], popt[3], popt[4])

        if plotBool:
            plt.plot(voltages, finalData/popt[0], 'o', label='data')
            plt.plot(x1,y1, label='fit')
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.show()

        return popt[0]
    elif voltVar:
        guess = [suggestedGain, suggestedNoiseTemp, suggestedTemp, vOffset, voltDiff]

        def fitfunc( V, a, b, c, d, diff):
            y = a*(b + ((h*f-e*diff*(V-d))/(4*k))*((np.cosh((h*f-e*diff*(V-d))/(2.0*k*c)))/(np.sinh((h*f-e*diff*(V-d))/(2.0*k*c))))+
                   ((h*f+e*diff*(V-d))/(4*k))*((np.cosh((h*f+e*diff*(V-d))/(2.0*k*c)))/(np.sinh((h*f+e*diff*(V-d))/(2.0*k*c)))))
            return y


        popt, pcov = curve_fit(fitfunc, voltages, finalData, p0=guess)
        params = popt

        print 'gain = ', 10*np.log10(popt[0])
        print 'noise = ', popt[1]
        print 'temp = ', popt[2]
        print 'offset = ', popt[3]
        print 'Voltage factor = ', popt[4]

        x1 = np.linspace(-amplitude, amplitude, 5000)
        y1 = fitfunc(x1, 1, popt[1], popt[2], popt[3], popt[4])

        if plotBool:
            plt.plot(voltages, finalData/popt[0], 'o', label='data')
            plt.plot(x1,y1, label='fit')
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.show()

        return popt[0], popt[1]
    elif voltEnf:
        guess = [suggestedGain, suggestedNoiseTemp, suggestedTemp, vOffset]

        def fitfunc( V, a, b, c, d):
            y = a*(b + ((h*f-e*voltDiff*(V-d))/(4*k))*((np.cosh((h*f-e*voltDiff*(V-d))/(2.0*k*c)))/(np.sinh((h*f-e*voltDiff*(V-d))/(2.0*k*c))))+
                   ((h*f+e*voltDiff*(V-d))/(4*k))*((np.cosh((h*f+e*voltDiff*(V-d))/(2.0*k*c)))/(np.sinh((h*f+e*voltDiff*(V-d))/(2.0*k*c)))))
            return y


        popt, pcov = curve_fit(fitfunc, voltages, finalData, p0=guess)
        params = popt

        print 'gain = ', 10*np.log10(popt[0])
        print 'noise = ', popt[1]
        print 'temp = ', popt[2]
        print 'offset = ', popt[3]

        x1 = np.linspace(-amplitude, amplitude, 5000)
        y1 = fitfunc(x1, 1, popt[1], popt[2], popt[3])

        if plotBool:
            plt.plot(voltages, finalData/popt[0], 'o', label='data')
            plt.plot(x1,y1, label='fit')
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.show()

        return popt[0], popt[1]
    else:
        guess = [suggestedGain, suggestedNoiseTemp, suggestedTemp, vOffset]

        def fitfunc( V, a, b, c, d):
            y = a*(b + ((h*f-e*(V-d))/(4*k))*((np.cosh((h*f-e*(V-d))/(2.0*k*c)))/(np.sinh((h*f-e*(V-d))/(2.0*k*c))))+((h*f+e*(V-d))/(4*k))*((np.cosh((h*f+e*(V-d))/(2.0*k*c)))/(np.sinh((h*f+e*(V-d))/(2.0*k*c)))))
            return y


        popt, pcov = curve_fit(fitfunc, voltages, finalData, p0=guess)
        params = popt

        print 'gain = ', 10*np.log10(popt[0])
        print 'noise = ', popt[1]
        print 'temp = ', popt[2]
        print 'offset = ', popt[3]

        x1 = np.linspace(-amplitude, amplitude, 5000)
        y1 = fitfunc(x1, 1, popt[1], popt[2], popt[3])

        if plotBool:
            plt.plot(voltages, finalData/popt[0], 'o', label='data')
            plt.plot(x1,y1, label='fit')
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.show()

        return popt[0], popt[1]


def manualFitSNTJ(cxn, date, fileNumber, plotBool = True,
            voltEnf = False, voltVar = False, freqVar = False,
            gainEnf = False, tempEnf = False, noiseEnf = False,
            truncate = False, limit = [1,1], suggestedGain = 300000000,
            suggestedNoiseTemp = 2.5, suggestedTemp = 0.050 ):
    voltages = np.array([])
    finalData = np.array([])
    params = np.array([])
    voltDiff = 0.9



    dv = cxn.data_vault
    dv.cd(['', 'Ted', 'SNTJ', date])
    dv.open(fileNumber)
    data = np.asarray(dv.get())

    f = dv.get_parameter('freq')*1e6

    points = len(data[:,0])
    if truncate:
        for i in range(points):
            if data[i,0]>limit[0] and data[i,0]<limit[1]:
                voltages = np.append(voltages, float(data[i,0]))
                finalData = np.append(finalData, float(data[i,1]))
    else:
        for i in range(points):
            voltages = np.append(voltages, float(data[i,0]))
            finalData = np.append(finalData, float(data[i,1]))
    suggestedGain
    suggestedNoiseTemp
    suggestedTemp
    vOffset = 0.00001
    amplitude = 2*dv.get_parameter('inputAttenuation')*dv.get_parameter('voltageAmplitude')

    if voltVar:
        guess = [suggestedGain, suggestedNoiseTemp, suggestedTemp, vOffset, voltDiff]

        def fitfunc( V, a, b, c, d, diff):
            y = a*(b + ((h*f-e*diff*(V-d))/(4*k))*((np.cosh((h*f-e*diff*(V-d))/(2.0*k*c)))/(np.sinh((h*f-e*diff*(V-d))/(2.0*k*c))))+
                   ((h*f+e*diff*(V-d))/(4*k))*((np.cosh((h*f+e*diff*(V-d))/(2.0*k*c)))/(np.sinh((h*f+e*diff*(V-d))/(2.0*k*c)))))
            return y


        popt, pcov = curve_fit(fitfunc, voltages, finalData, p0=guess)
        params = popt

        print 'gain = ', 10*np.log10(popt[0])
        print 'noise = ', popt[1]
        print 'temp = ', popt[2]
        print 'offset = ', popt[3]
        print 'Voltage factor = ', popt[4]

        x1 = np.linspace(-amplitude, amplitude, 5000)
        y1 = fitfunc(x1, 1, popt[1], popt[2], popt[3], popt[4])

        if plotBool:
            plt.plot(voltages, finalData/popt[0], 'o', label='data')
            plt.plot(x1,y1, label='fit')
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.show()

        return popt[0], popt[1]

    else:
        guess = [suggestedGain, suggestedNoiseTemp, suggestedTemp, vOffset]

        def fitfunc( V, a, b, c, d):
            y = a*(b + ((h*f-e*(V-d))/(4*k))*((np.cosh((h*f-e*(V-d))/(2.0*k*c)))/(np.sinh((h*f-e*(V-d))/(2.0*k*c))))+((h*f+e*(V-d))/(4*k))*((np.cosh((h*f+e*(V-d))/(2.0*k*c)))/(np.sinh((h*f+e*(V-d))/(2.0*k*c)))))
            return y


        popt, pcov = curve_fit(fitfunc, voltages, finalData, p0=guess)
        params = popt

        print 'gain = ', 10*np.log10(popt[0])
        print 'noise = ', popt[1]
        print 'temp = ', popt[2]
        print 'offset = ', popt[3]

        x1 = np.linspace(-amplitude, amplitude, 5000)
        y1 = fitfunc(x1, 1, popt[1], popt[2], popt[3])

        if plotBool:
            plt.plot(voltages, finalData/popt[0], 'o', label='data')
            plt.plot(x1,y1, label='fit')
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.show()

        return popt[0], popt[1]

def fitSLUGSNTJ(cxn, dir, fileNumber, plotBool = True,
            voltEnf = False, voltVar = False, freqVar = False,
            gainEnf = False, tempEnf = False, noiseEnf = False,
            truncate = False, limit = [1,1]):
    voltages = np.array([])
    finalData = np.array([])
    params = np.array([])
    voltDiff = 0.53



    dv = cxn.data_vault
    dv.cd(dir)
    dv.open(fileNumber)
    data = np.asarray(dv.get())

    suggestedGain = dv.get_parameter('guessGain')
    suggestedNoiseTemp = dv.get_parameter('guessNoise')
    suggestedTemp = dv.get_parameter('guessTemp')
    f = dv.get_parameter('freq')*1e6

    points = len(data[:,0])
    if truncate:
        for i in range(points):
            if data[i,0]>limit[0] and data[i,0]<limit[1]:
                voltages = np.append(voltages, float(data[i,0]))
                finalData = np.append(finalData, float(data[i,1]))
    else:
        for i in range(points):
            voltages = np.append(voltages, float(data[i,0]))
            finalData = np.append(finalData, float(data[i,1]))
    suggestedGain
    suggestedNoiseTemp
    suggestedTemp
    vOffset = 0.00001
    amplitude = 2*dv.get_parameter('inputAttenuation')*dv.get_parameter('voltageAmplitude')
    if gainEnf:
        guess = [suggestedNoiseTemp, suggestedTemp, vOffset]

        def fitfunc( V, b, c, d):
            y = suggestedGain*(b + (((h*f-e*(V-d))/(4*k))*((np.cosh((h*f-e*(V-d))/(2.0*k*c)))/(np.sinh((h*f-e*(V-d))/(2.0*k*c)))))+
                               (((h*f+e*(V-d))/(4*k))*((np.cosh((h*f+e*(V-d))/(2.0*k*c)))/(np.sinh((h*f+e*(V-d))/(2.0*k*c))))))
            return y


        popt, pcov = curve_fit(fitfunc, voltages, finalData, p0=guess)
        params = popt

        print 'noise = ', popt[0]
        print 'temp = ', popt[1]
        print 'offset = ', popt[2]

        x1 = np.linspace(-amplitude, amplitude, 5000)
        y1 = fitfunc(x1, popt[0], popt[1], popt[2])/suggestedGain

        if plotBool:
            plt.plot(voltages, finalData/suggestedGain, 'o', label='data')
            plt.plot(x1,y1, label='fit')
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.show()

    elif tempEnf:
        guess = [suggestedGain, suggestedNoiseTemp, vOffset]

        def fitfunc( V, a, b, d):
            y = a*(b + ((h*f-e*(V-d))/(4*k))*((np.cosh((h*f-e*(V-d))/(2.0*k*suggestedTemp)))/(np.sinh((h*f-e*(V-d))/(2.0*k*suggestedTemp))))+
                   ((h*f+e*(V-d))/(4*k))*((np.cosh((h*f+e*(V-d))/(2.0*k*suggestedTemp)))/(np.sinh((h*f+e*(V-d))/(2.0*k*suggestedTemp)))))
            return y


        popt, pcov = curve_fit(fitfunc, voltages, finalData, p0=guess)
        params = popt

        print 'gain = ', 10*np.log10(popt[0])
        print 'noise = ', popt[1]
        print 'offset = ', popt[2]

        x1 = np.linspace(-amplitude, amplitude, 5000)
        y1 = fitfunc(x1, 1, popt[1], popt[2])

        if plotBool:
            plt.plot(voltages, finalData/popt[0], 'o', label='data')
            plt.plot(x1,y1, label='fit')
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.show()

    elif noiseEnf:
        guess = [suggestedGain, suggestedTemp, vOffset]

        def fitfunc( V, a, c, d):
            y = a*(suggestedNoiseTemp + ((h*f-e*(V-d))/(4*k))*((np.cosh((h*f-e*(V-d))/(2.0*k*c)))/(np.sinh((h*f-e*(V-d))/(2.0*k*c))))+
                   ((h*f+e*(V-d))/(4*k))*((np.cosh((h*f+e*(V-d))/(2.0*k*c)))/(np.sinh((h*f+e*(V-d))/(2.0*k*c)))))
            return y


        popt, pcov = curve_fit(fitfunc, voltages, finalData, p0=guess)
        params = popt

        print 'gain = ', 10*np.log10(popt[0])
        print 'temp = ', popt[1]
        print 'offset = ', popt[2]

        x1 = np.linspace(-amplitude, amplitude, 5000)
        y1 = fitfunc(x1, 1, popt[1], popt[2])

        if plotBool:
            plt.plot(voltages, finalData/popt[0], 'o', label='data')
            plt.plot(x1,y1, label='fit')
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.show()

    elif freqVar:
        guess = [suggestedGain, suggestedNoiseTemp, suggestedTemp, vOffset, f]

        def fitfunc( V, a, b, c, d, fr):
            y = a*(b + ((h*fr-e*(V-d))/(4*k))*((np.cosh((h*fr-e*(V-d))/(2.0*k*c)))/(np.sinh((h*fr-e*(V-d))/(2.0*k*c))))+
                   ((h*fr+e*(V-d))/(4*k))*((np.cosh((h*fr+e*(V-d))/(2.0*k*c)))/(np.sinh((h*fr+e*(V-d))/(2.0*k*c)))))
            return y


        popt, pcov = curve_fit(fitfunc, voltages, finalData, p0=guess)
        params = popt

        print 'gain = ', 10*np.log10(popt[0])
        print 'noise = ', popt[1]
        print 'temp = ', popt[2]
        print 'offset = ', popt[3]
        print 'freq = ', popt[4]

        x1 = np.linspace(-amplitude, amplitude, 5000)
        y1 = fitfunc(x1, 1, popt[1], popt[2], popt[3], popt[4])

        if plotBool:
            plt.plot(voltages, finalData/popt[0], 'o', label='data')
            plt.plot(x1,y1, label='fit')
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.show()

        return popt[0]
    elif voltVar:
        guess = [suggestedGain, suggestedNoiseTemp, suggestedTemp, vOffset, voltDiff]

        def fitfunc( V, a, b, c, d, diff):
            y = a*(b + ((h*f-e*diff*(V-d))/(4*k))*((np.cosh((h*f-e*diff*(V-d))/(2.0*k*c)))/(np.sinh((h*f-e*diff*(V-d))/(2.0*k*c))))+
                   ((h*f+e*diff*(V-d))/(4*k))*((np.cosh((h*f+e*diff*(V-d))/(2.0*k*c)))/(np.sinh((h*f+e*diff*(V-d))/(2.0*k*c)))))
            return y


        popt, pcov = curve_fit(fitfunc, voltages, finalData, p0=guess)
        params = popt

        print 'gain = ', 10*np.log10(popt[0])
        print 'noise = ', popt[1]
        print 'temp = ', popt[2]
        print 'offset = ', popt[3]
        print 'Voltage factor = ', popt[4]

        x1 = np.linspace(-amplitude, amplitude, 5000)
        y1 = fitfunc(x1, 1, popt[1], popt[2], popt[3], popt[4])

        if plotBool:
            plt.plot(voltages, finalData/popt[0], 'o', label='data')
            plt.plot(x1,y1, label='fit')
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.show()

        return popt[0], popt[1]
    elif voltEnf:
        guess = [suggestedGain, suggestedNoiseTemp, suggestedTemp, vOffset]

        def fitfunc( V, a, b, c, d):
            y = a*(b + ((h*f-e*voltDiff*(V-d))/(4*k))*((np.cosh((h*f-e*voltDiff*(V-d))/(2.0*k*c)))/(np.sinh((h*f-e*voltDiff*(V-d))/(2.0*k*c))))+
                   ((h*f+e*voltDiff*(V-d))/(4*k))*((np.cosh((h*f+e*voltDiff*(V-d))/(2.0*k*c)))/(np.sinh((h*f+e*voltDiff*(V-d))/(2.0*k*c)))))
            return y


        popt, pcov = curve_fit(fitfunc, voltages, finalData, p0=guess)
        params = popt

        print 'gain = ', 10*np.log10(popt[0])
        print 'noise = ', popt[1]
        print 'temp = ', popt[2]
        print 'offset = ', popt[3]

        x1 = np.linspace(-amplitude, amplitude, 5000)
        y1 = fitfunc(x1, 1, popt[1], popt[2], popt[3])

        if plotBool:
            plt.plot(voltages, finalData/popt[0], 'o', label='data')
            plt.plot(x1,y1, label='fit')
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.show()

        return popt[0], popt[1]
    else:
        guess = [suggestedGain, suggestedNoiseTemp, suggestedTemp, vOffset]

        def fitfunc( V, a, b, c, d):
            y = a*(b + ((h*f-e*(V-d))/(4*k))*((np.cosh((h*f-e*(V-d))/(2.0*k*c)))/(np.sinh((h*f-e*(V-d))/(2.0*k*c))))+((h*f+e*(V-d))/(4*k))*((np.cosh((h*f+e*(V-d))/(2.0*k*c)))/(np.sinh((h*f+e*(V-d))/(2.0*k*c)))))
            return y


        popt, pcov = curve_fit(fitfunc, voltages, finalData, p0=guess)
        params = popt

        print 'gain = ', 10*np.log10(popt[0])
        print 'noise = ', popt[1]
        print 'temp = ', popt[2]
        print 'offset = ', popt[3]

        x1 = np.linspace(-amplitude, amplitude, 5000)
        y1 = fitfunc(x1, 1, popt[1], popt[2], popt[3])

        if plotBool:
            plt.plot(voltages, finalData/popt[0], 'o', label='data')
            plt.plot(x1,y1, label='fit')
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.show()

        return popt[0], popt[1]

def manualFitSLUGSNTJ(cxn, dev, date, fileNumber, plotBool = True,
            voltEnf = False, voltVar = False, freqVar = False,
            gainEnf = False, tempEnf = False, noiseEnf = False,
            truncate = False, limit = [1,1], suggestedGain = 300000000,
            suggestedNoiseTemp = 2.5, suggestedTemp = 0.050 ):
    voltages = np.array([])
    finalData = np.array([])
    params = np.array([])
    voltDiff = 0.66



    dv = cxn.data_vault
    dv.cd(['', 'Ted', 'SLUG', dev, date])
    dv.open(fileNumber)
    data = np.asarray(dv.get())

    f = dv.get_parameter('freq')*1e6

    points = len(data[:,0])
    if truncate:
        for i in range(points):
            if data[i,0]>limit[0] and data[i,0]<limit[1]:
                voltages = np.append(voltages, float(data[i,0]))
                finalData = np.append(finalData, float(data[i,1]))
    else:
        for i in range(points):
            voltages = np.append(voltages, float(data[i,0]))
            finalData = np.append(finalData, float(data[i,1]))
    suggestedGain
    suggestedNoiseTemp
    suggestedTemp
    vOffset = 0.00001
    amplitude = 2*dv.get_parameter('inputAttenuation')*dv.get_parameter('voltageAmplitude')

    if voltVar:
        guess = [suggestedGain, suggestedNoiseTemp, suggestedTemp, vOffset, voltDiff]

        def fitfunc( V, a, b, c, d, diff):
            y = a*(b + ((h*f-e*diff*(V-d))/(4*k))*((np.cosh((h*f-e*diff*(V-d))/(2.0*k*c)))/(np.sinh((h*f-e*diff*(V-d))/(2.0*k*c))))+
                   ((h*f+e*diff*(V-d))/(4*k))*((np.cosh((h*f+e*diff*(V-d))/(2.0*k*c)))/(np.sinh((h*f+e*diff*(V-d))/(2.0*k*c)))))
            return y


        popt, pcov = curve_fit(fitfunc, voltages, finalData, p0=guess)
        params = popt

        print 'gain = ', 10*np.log10(popt[0])
        print 'noise = ', popt[1]
        print 'temp = ', popt[2]
        print 'offset = ', popt[3]
        print 'Voltage factor = ', popt[4]

        x1 = np.linspace(-amplitude, amplitude, 5000)
        y1 = fitfunc(x1, 1, popt[1], popt[2], popt[3], popt[4])

        if plotBool:
            plt.plot(voltages, finalData/popt[0], 'o', label='data')
            plt.plot(x1,y1, label='fit')
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.show()

        return popt[0], popt[1]

    elif voltEnf:
        guess = [suggestedGain, suggestedNoiseTemp, suggestedTemp, vOffset]

        def fitfunc( V, a, b, c, d):
            y = a*(b + ((h*f-e*voltDiff*(V-d))/(4*k))*((np.cosh((h*f-e*voltDiff*(V-d))/(2.0*k*c)))/(np.sinh((h*f-e*voltDiff*(V-d))/(2.0*k*c))))+
                   ((h*f+e*voltDiff*(V-d))/(4*k))*((np.cosh((h*f+e*voltDiff*(V-d))/(2.0*k*c)))/(np.sinh((h*f+e*voltDiff*(V-d))/(2.0*k*c)))))
            return y


        popt, pcov = curve_fit(fitfunc, voltages, finalData, p0=guess)
        params = popt

        print 'gain = ', 10*np.log10(popt[0])
        print 'noise = ', popt[1]
        print 'temp = ', popt[2]
        print 'offset = ', popt[3]

        x1 = np.linspace(-amplitude, amplitude, 5000)
        y1 = fitfunc(x1, 1, popt[1], popt[2], popt[3])

        if plotBool:
            plt.plot(voltages, finalData/popt[0], 'o', label='data')
            plt.plot(x1,y1, label='fit')
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.show()

        return popt[0], popt[1]

    else:
        guess = [suggestedGain, suggestedNoiseTemp, suggestedTemp, vOffset]

        def fitfunc( V, a, b, c, d):
            y = a*(b + ((h*f-e*(V-d))/(4*k))*((np.cosh((h*f-e*(V-d))/(2.0*k*c)))/(np.sinh((h*f-e*(V-d))/(2.0*k*c))))+((h*f+e*(V-d))/(4*k))*((np.cosh((h*f+e*(V-d))/(2.0*k*c)))/(np.sinh((h*f+e*(V-d))/(2.0*k*c)))))
            return y


        popt, pcov = curve_fit(fitfunc, voltages, finalData, p0=guess)
        params = popt

        print 'gain = ', 10*np.log10(popt[0])
        print 'noise = ', popt[1]
        print 'temp = ', popt[2]
        print 'offset = ', popt[3]

        x1 = np.linspace(-amplitude, amplitude, 5000)
        y1 = fitfunc(x1, 1, popt[1], popt[2], popt[3])

        if plotBool:
            plt.plot(voltages, finalData/popt[0], 'o', label='data')
            plt.plot(x1,y1, label='fit')
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.show()

        return popt[0], popt[1]
