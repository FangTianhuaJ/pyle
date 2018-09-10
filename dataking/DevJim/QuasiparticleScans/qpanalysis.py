import numpy as np
from numpy import linalg
from scipy.optimize import fminbound, leastsq
import matplotlib.pyplot as plt
from math import *

from labrad.units import Unit
V, mV, us, ns, GHz, MHz = [Unit(s) for s in ('V', 'mV', 'us', 'ns', 'GHz', 'MHz')]

from pyle.fitting import qubitparams
from pyle import registry

FOLDERLIST=[['4q5r','w101116A','r7c10','110422','SH1.500SD200.0SS20.0'],
            ['4q5r','w101116A','r7c10','110410','SH1.500SD200.0SS38.0'],
            ['4q5r','w101116A','r7c10','110410','SH1.500SD200.0SS72.5'],
            ['4q5r','w101116A','r7c10','110410','SH1.500SD200.0SS138.0'],
            ['4q5r','w101116A','r7c10','110410','SH1.500SD200.0SS262.7'],
            ['4q5r','w101116A','r7c10','110410','SH1.500SD200.0SS500.0'],
            ['4q5r','w101116A','r7c10','110410','test']]
BIASTIMES=np.array([20,38,72.5,138,262.7,500,1000])


def switchSession(cxn, user, samplePath=None):
    """Switch the current session."""
    userPath = ['', user]
    reg = registry.RegistryWrapper(cxn, userPath)
    oldSession = reg['sample']
    reg['sample'] = samplePath
    for dir in samplePath:
        reg = reg[dir]
    # change data vault directory, creating new directories as needed
    cxn.data_vault.cd(userPath + samplePath, True)
    return reg, oldSession


def getFileData(dv,fileName):
    # Get file number for data with desired fileName
    files = dv.dir()[1]
    filenames = np.array([])
    filenums = np.array([])
    for i in range(len(files)):
        filenames = np.append(filenames,files[i][18:])
        filenums = np.append(filenums,int(files[i][:5]))
    datanum = int(filenums[filenames==fileName][-1])
    
    # Open data set
    dv.open(datanum)
    data = np.array(dv.get())
    return data


def getT1(dv,flag=''):
    filename = 'T1 MQ'+flag
    t1data = getFileData(dv,filename)
    def t1fit(x,p):
        return p[0]+p[1]*np.exp(-x/p[2])
    t1fitresult = leastsq(lambda p: t1fit(t1data[:,0],p)-t1data[:,1], [0,1,500])
    return t1fitresult[0][2]


def getFreqShift(dv):
    fdata = getFileData(dv,'ing f01')
    fnoheat = fdata[1][1]
    fheat = fdata[0][1]
    shift = fheat-fnoheat
    return shift


def getProbs(dv):
    prob1data = getFileData(dv,'Measure |1) with |1)')
    prob2data = getFileData(dv,'Measure |1) with |2)')
    prob3data = getFileData(dv,'Measure |1) with |3)')
    probnopi = np.array([prob3data[0,1],prob2data[0,1],prob1data[0,1],1])
    probpi = np.array([prob3data[0,1],prob2data[0,1],prob1data[0,1],1])
    return probnopi,probpi


#def getProbs(dv):
#    prob1data = getFileData(dv,'Measure |1) with |1)')
#    prob2data = getFileData(dv,'Measure |1) with |2)')
#    prob3data = getFileData(dv,'Measure |1) with |3)')
#    prob1 = prob1data[:,1]
#    prob2 = prob2data[:,1]
#    prob3 = prob3data[:,1]
#    return prob1, prob2, prob3


def getCorrections(dv):
    prob1data = getFileData(dv,'Measure Multi-States w/ |1)')
    prob2data = getFileData(dv,'Measure Multi-States w/ |2)')
    prob3data = getFileData(dv,'Measure Multi-States w/ |3)')
    prob0 = np.array([1,1,1,1])
    prob1 = prob1data[:,1]
    prob2 = prob2data[:,1]
    prob3 = prob3data[:,1]
    vismat = np.vstack((prob3,prob2,prob1,prob0))
    return vismat


def correctProbs(dv):
#    cor1,cor2 = getCorrections(dv)
    corMatrix = np.linalg.inv(getCorrections(dv))
    measNoPulse,measPiPulse = getProbs(dv)
    probNoPulse = np.dot(corMatrix,measNoPulse)
    probPiPulse = np.dot(corMatrix,measPiPulse)
    return probNoPulse, probPiPulse


def dataCompile(cxn,session,user='Jim',measure=0):
    dv=cxn.data_vault
    for folder in FOLDERLIST:
        session,oldSession = switchSession(cxn,user,folder)
        print folder
        if folder is ['4q5r','w101116A','r7c10','110410','test']: flagname=''
        else: flagname=' for |1>,meas=|1>'
        print getT1(dv,flag=flagname)
        print getFreqShift(dv)
        print correctProbs(dv)
        session,finishedSession = switchSession(cxn,user,oldSession)


#def correctProbs(dv):
#    cor1,cor2 = getCorrections(dv)
#    corMatrix = np.linalg.inv(np.array([[cor2[2]-cor2[0],cor2[1]-cor2[0]],[cor1[2]-cor1[0],cor1[1]-cor1[0]]]))
#    meas1,meas2 = getProbs(dv)
#    measNoPulse = np.array([meas2[0]-cor2[0],meas1[0]-cor1[0]])
#    measPiPulse = np.array([meas2[1]-cor2[0],meas1[1]-cor1[0]])
#    probNoPulse = np.dot(corMatrix,measNoPulse)[0]
#    probPiPulse = np.dot(corMatrix,measPiPulse)[0]
#    return probNoPulse, probPiPulse, meas2[0], meas2[1]
#
#
#def qpDensity(dv,session,Vp,Vl,measure=0):
#    t1val = getT1(dv)
#    qubitdata = qubitparams.getQBParams(session, Vp, Vl,measure=measure)
#    print qubitdata
#    I1 = qubitdata['Iprime']
#    Ic1 = qubitdata['Icprime']
#    delta = qubitdata['delta']
#    f10 = qubitdata['f10']
#    print([I1,Ic1])
#    prefactor = (1+cos(delta))*f10*sqrt(Ic1/(Ic1-I1))*sqrt(41.1/f10)
#    mikeratio = -.25*(1+2*1.2)*sqrt(41.1/f10)*sqrt(Ic1/(Ic1-I1))/(1+cos(delta))
#    print(prefactor,mikeratio)
#    density = 1./(78.5*t1val)
#    return t1val, density
#
#
#def dataCompile(cxn,session,Vp,Vl,user='Jim',measure=0):
#    dv=cxn.data_vault
#    t1array=np.array([])
#    densityarray=np.array([])
#    fshiftarray=np.array([])
#    probNoPulsearray=np.array([])
#    probPiPulsearray=np.array([])
#    meas2NoPulsearray=np.array([])
#    meas2PiPulsearray=np.array([])
#    for folder in FOLDERLIST:
#        session,oldSession = switchSession(cxn,user,folder)
##        t1,density=qpDensity(dv,session,Vp,Vl,measure=measure)
#        t1=getT1(dv)
#        t1array=np.append(t1array,t1)
#        densityarray=np.append(densityarray,1./(78.5*t1))
##        densityarray=np.append(densityarray,density)
#        fshift=getFreqShift(dv)
#        fshiftarray=np.append(fshiftarray,fshift)
#        probNoPulse,probPiPulse,meas2NoPulse,meas2PiPulse=correctProbs(dv)
#        probNoPulsearray=np.append(probNoPulsearray,probNoPulse)
#        probPiPulsearray=np.append(probPiPulsearray,probPiPulse)
#        meas2NoPulsearray=np.append(meas2NoPulsearray,meas2NoPulse)
#        meas2PiPulsearray=np.append(meas2PiPulsearray,meas2PiPulse)
#        session,finishedSession = switchSession(cxn,user,oldSession)
#    return t1array,densityarray,fshiftarray,probNoPulsearray,probPiPulsearray,meas2NoPulsearray,meas2PiPulsearray
#
#def qpPlots(cxn,session,Vp,Vl,user='Jim',measure=0):
#    t1,density,fshift,probNoPulse,probPiPulse,meas2NoPulse,meas2PiPulse=dataCompile(cxn,session,Vp,Vl,user,measure)
#    plt.subplot(3,3,1)
#    plt.plot(BIASTIMES,t1,'b.')
#    plt.xlabel('SQUID Settling Time (us)')
#    plt.ylabel('T1 (ns)')
#    plt.subplot(3,3,4)
#    plt.plot(BIASTIMES,fshift,'b.')
#    plt.xlabel('SQUID Settling Time (us)')
#    plt.ylabel('$f-f_0/f_0$')
#    plt.subplot(3,3,7)
#    plt.plot(BIASTIMES,meas2NoPulse,'b.')
#    plt.plot(BIASTIMES,meas2PiPulse,'r.')
#    plt.plot(BIASTIMES,meas2PiPulse-meas2NoPulse,'k.')
#    plt.xlabel('SQUID Settling Time (us)')
#    plt.ylabel('Uncorrected Prob.')
#    plt.subplot(3,3,2)
#    plt.plot(BIASTIMES,1/t1,'b.')
#    plt.xlabel('SQUID Settling Time (us)')
#    plt.ylabel('1/T1 (1/ns)')
#    plt.subplot(3,3,5)
#    plt.plot(BIASTIMES,probNoPulse,'b.')
#    plt.plot(BIASTIMES,probPiPulse,'r.')
#    plt.plot(BIASTIMES,probPiPulse-probNoPulse,'k.')
#    plt.xlabel('SQUID Settling Time (us)')
#    plt.ylabel('Corrected Prob.')
#    plt.subplot(3,3,3)
#    plt.plot(1/t1,fshift,'b.')
#    plt.xlabel('1/T1 (1/ns)')
#    plt.ylabel('$f-f_0$ (GHz)')
#    plt.subplot(3,3,6)
#    plt.plot(1/t1,meas2PiPulse-meas2NoPulse,'b.')
#    plt.xlabel('1/T1 (1/ns)')
#    plt.ylabel('Uncorrected Prob.')
#    plt.subplot(3,3,9)
#    plt.plot(density,probPiPulse-probNoPulse,'b.')
#    plt.xlabel('QP Density $n_{qp}/n_{cp}$')
#    plt.ylabel('Corrected Prob.')
#    
#def qpFreqPlots(cxn,session,Vp,Vl,user='Jim',measure=0):
#    t1,density,fshift,probNoPulse,probPiPulse,meas2NoPulse,meas2PiPulse=dataCompile(cxn,session,Vp,Vl,user,measure)
#    plt.rcParams["font.size"]=48
#    plt.subplot(2,2,1)
#    plt.plot(BIASTIMES,t1,'b.',markersize=30)
#    plt.xlabel('SQUID Settling Time (us)')
#    plt.ylabel('$T_1$ (ns)')
#    plt.axis([0,1000,0,900])
#    plt.yticks([0,200,400,600,800])
#    ax=plt.gca()
#    ax.spines["bottom"].set_linewidth(5)
#    ax.spines["right"].set_linewidth(5)
#    ax.spines["top"].set_linewidth(5)
#    ax.spines["left"].set_linewidth(5)
#    for line in (ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines()):
#        line.set_markeredgewidth(5)
#        line.set_markersize(10)
#    plt.subplot(2,2,3)
#    plt.plot(BIASTIMES,1000*fshift,'b.',markersize=30)
#    plt.xlabel('SQUID Settling Time (us)')
#    plt.ylabel('$f-f_0$ (MHz)')
#    ax=plt.gca()
#    ax.spines["bottom"].set_linewidth(5)
#    ax.spines["right"].set_linewidth(5)
#    ax.spines["top"].set_linewidth(5)
#    ax.spines["left"].set_linewidth(5)
#    for line in (ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines()):
#        line.set_markeredgewidth(5)
#        line.set_markersize(10)
#    plt.subplot(2,2,2)
#    plt.plot(1000/t1,1000*fshift,'b.',markersize=30)
#    plt.xlabel('1/T1 (1/us)')
#    plt.ylabel('$f-f_0$ (MHz)')
#    pfit=np.polyfit(1/t1,fshift,1)
#    print(pfit)
#    pdata=np.polyval(pfit,np.linspace(0,9,1000))
#    plt.plot(np.linspace(0,9,1000),pdata,'k',linewidth=5)
#    plt.axis([0,9,-6,0])
#    ax=plt.gca()
#    ax.spines["bottom"].set_linewidth(5)
#    ax.spines["right"].set_linewidth(5)
#    ax.spines["top"].set_linewidth(5)
#    ax.spines["left"].set_linewidth(5)
#    for line in (ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines()):
#        line.set_markeredgewidth(5)
#        line.set_markersize(10)
#        
#def qpUncorrectedPlot(cxn,session,Vp,Vl,user='Jim',measure=0):
#    t1,density,fshift,probNoPulse,probPiPulse,meas2NoPulse,meas2PiPulse=dataCompile(cxn,session,Vp,Vl,user,measure)
#    plt.rcParams["font.size"]=48
#    plt.rcParams["legend.numpoints"]=3
#    plt.plot(BIASTIMES,meas2NoPulse,'b.',markersize=30,label='$P_2$')
#    plt.plot(BIASTIMES,meas2PiPulse,'r.',markersize=30,label='$P_1+P_2$')
#    plt.plot(BIASTIMES,meas2PiPulse-meas2NoPulse,'k.',markersize=30,label='$P_1$')
#    plt.xlabel('SQUID Settling Time (us)')
#    plt.ylabel('Probability')
#    plt.axis([0,1050,0,.18])
#    ax=plt.gca()
#    ax.spines["bottom"].set_linewidth(5)
#    ax.spines["right"].set_linewidth(5)
#    ax.spines["top"].set_linewidth(5)
#    ax.spines["left"].set_linewidth(5)
#    for line in (ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines()):
#        line.set_markeredgewidth(5)
#        line.set_markersize(10)
#    leg=plt.legend(loc='upper right',labelspacing=0,borderpad=0.1,handletextpad=0.1)   
#    leg.get_frame().set_lw(5)
#    no,pi,sub=leg.get_texts()
#    no.set_color('b')
#    pi.set_color('r')
#    sub.set_color('k')
#        
#def qpCorrectedPlot(cxn,session,Vp,Vl,user='Jim',measure=0):
#    t1,density,fshift,probNoPulse,probPiPulse,meas2NoPulse,meas2PiPulse=dataCompile(cxn,session,Vp,Vl,user,measure)
#    plt.rcParams["font.size"]=48
#    plt.rcParams["legend.numpoints"]=3
#    plt.plot(BIASTIMES,probNoPulse,'b.',markersize=30,label='$P_2$')
#    plt.plot(BIASTIMES,probPiPulse,'r.',markersize=30,label='$P_1+P_2$')
#    plt.plot(BIASTIMES,probPiPulse-probNoPulse,'k.',markersize=30,label='$P_1$')
#    plt.xlabel('SQUID Settling Time (us)')
#    plt.ylabel('Corrected Probability')
#    plt.axis([0,1050,-.02,.27])
#    ax=plt.gca()
#    ax.spines["bottom"].set_linewidth(5)
#    ax.spines["right"].set_linewidth(5)
#    ax.spines["top"].set_linewidth(5)
#    ax.spines["left"].set_linewidth(5)
#    for line in (ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines()):
#        line.set_markeredgewidth(5)
#        line.set_markersize(10)
#    leg=plt.legend(loc='upper right',labelspacing=0,borderpad=0.1,handletextpad=0.1)   
#    leg.get_frame().set_lw(5)
#    no,pi,sub=leg.get_texts()
#    no.set_color('b')
#    pi.set_color('r')
#    sub.set_color('k')
#    
#    
#    
#def qpDensityPlot(cxn,session,Vp,Vl,user='Jim',measure=0):
#    t1,density,fshift,probNoPulse,probPiPulse,meas2NoPulse,meas2PiPulse=dataCompile(cxn,session,Vp,Vl,user,measure)
#    plt.rcParams["font.size"]=48
#    plt.plot(density,probPiPulse-probNoPulse,'b.',markersize=30)
#    plt.plot(density[:9],probPiPulse[:9]-probNoPulse[:9],'r.',markersize=30)
#    plt.xlabel('QP Density $n_{qp}/n_{cp}$')
#    plt.ylabel('Corrected $P_1$')
#    plt.axis([0,.00011,0,.25])
#    plt.xticks([0,5e-5,10e-5])
#    ax=plt.gca()
#    ax.spines["bottom"].set_linewidth(5)
#    ax.spines["right"].set_linewidth(5)
#    ax.spines["top"].set_linewidth(5)
#    ax.spines["left"].set_linewidth(5)
#    for line in (ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines()):
#        line.set_markeredgewidth(5)
#        line.set_markersize(10)
#    pfit=np.polyfit(density[:9],probPiPulse[:9],1)
#    print(pfit)
#    
#    
#    
#def qpDensityPlotTheory(cxn,session,Vp,Vl,user='Jim',measure=0):
#    t1,density,fshift,probNoPulse,probPiPulse,meas2NoPulse,meas2PiPulse=dataCompile(cxn,session,Vp,Vl,user,measure)
#    plt.rcParams["font.size"]=48
#    plt.plot(density,probPiPulse-probNoPulse,'b.',markersize=30)
#    theoryx=np.linspace(0,.00011,1000)
#    plt.plot(density[:9],probPiPulse[:9]-probNoPulse[:9],'r.',markersize=30)
#    plt.plot(theoryx,3000*theoryx,'k',linewidth=5)
#    plt.xlabel('QP Density $n_{qp}/n_{cp}$')
#    plt.ylabel('Corrected $P_1$')
#    plt.axis([0,.00011,0,.25])
#    plt.xticks([0,5e-5,10e-5])
#    ax=plt.gca()
#    ax.spines["bottom"].set_linewidth(5)
#    ax.spines["right"].set_linewidth(5)
#    ax.spines["top"].set_linewidth(5)
#    ax.spines["left"].set_linewidth(5)
#    for line in (ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines()):
#        line.set_markeredgewidth(5)
#        line.set_markersize(10)
#    pfit=np.polyfit(density[:9],probPiPulse[:9],1)
#    print(pfit)
#    
#    
#    
#def qpDensityLogPlot(cxn,session,Vp,Vl,user='Jim',measure=0):
#    t1,density,fshift,probNoPulse,probPiPulse,meas2NoPulse,meas2PiPulse=dataCompile(cxn,session,Vp,Vl,user,measure)
#    plt.rcParams["font.size"]=48
#    plt.loglog(density,probPiPulse-probNoPulse,'b.',markersize=30)
##    plt.plot(density[:9],probPiPulse[:9]-probNoPulse[:9],'r.',markersize=30)
#    plt.xlabel('QP Density $n_{qp}/n_{cp}$')
#    plt.ylabel('Corrected $P_1$')
##    plt.xticks([0,5e-5,10e-5])
#    ax=plt.gca()
#    ax.spines["bottom"].set_linewidth(5)
#    ax.spines["right"].set_linewidth(5)
#    ax.spines["top"].set_linewidth(5)
#    ax.spines["left"].set_linewidth(5)
#    for line in (ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines()):
#        line.set_markeredgewidth(5)
#        line.set_markersize(10)
#    theoryx = np.logspace(-5,-3,1000)
#    plt.loglog(theoryx,3000*theoryx,'r',linewidth=5)
#    plt.axis([1e-5,1e-3,1e-2,1e0])    