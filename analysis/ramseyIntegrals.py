import numpy as np
from numpy import sin, cos, pi, exp
import scipy as sp
from scipy import integrate
import scipy.interpolate as interpolate

import matplotlib.pyplot as plt
import matplotlib.font_manager

MARKERS = ['k.','r^','bs','g*']
MARKER_SIZES=[35,25,20,25]
LINE_STYLES = ['-','--',':','-.']
LINEWIDTH = 10
GRIDLINE_WIDTH = 1
GRIDLINE_STYLE = '--'
MARKER_EDGE_WIDTH = 7
MARKERSIZE = 10
FONT_SIZE = 50

def showIntegralsVsAlpha(alphaEnds,ts,fMin):
    alphas = np.linspace(alphaEnds[0],alphaEnds[1],20)
    fig = plt.figure()
    fig.subplots_adjust(bottom=0.12)
    plt.rcParams['font.size']=FONT_SIZE
    plt.rcParams['xtick.labelsize']=FONT_SIZE
    plt.rcParams['ytick.labelsize']=FONT_SIZE
    #markers=['bs','r^','k.']
    markers=['b','r']
    markersizes = [15,15,25]
    linestyles=['-','--']
    for i,t in enumerate(ts):
        integrals = [integrate.quad(integrand_1(alpha),0,-np.log(fMin*t))[0] for alpha in alphas]
        plt.plot(alphas,integrals,markers[i],linewidth=7,linestyle=linestyles[i],label='t='+str(t)[0:4]+'ns')
    plt.grid(linestyle=GRIDLINE_STYLE,linewidth=GRIDLINE_WIDTH)
    plt.xlabel('$\\alpha$',fontsize=75)
    plt.ylabel('$I$',fontsize=75)
    ax = fig.get_axes()[0]
    for line in ax.xaxis.get_ticklines():
        line.set_markeredgewidth(MARKER_EDGE_WIDTH)
        line.set_markersize(MARKERSIZE)
    for line in ax.yaxis.get_ticklines():
        line.set_markeredgewidth(MARKER_EDGE_WIDTH)
        line.set_markersize(MARKERSIZE)
    prop=matplotlib.font_manager.FontProperties(size=48)
    plt.legend(loc='upper left',prop=prop)

    
def showIntegrands(alphas,x):
    integrands = [integrand_1(alpha) for alpha in alphas]
    fig=plt.figure()
    fig.subplots_adjust(bottom=0.12)
    plt.rcParams['xtick.labelsize']=FONT_SIZE
    plt.rcParams['ytick.labelsize']=FONT_SIZE
    plt.rcParams['font.size']=FONT_SIZE
    for i,integrand in enumerate(integrands):
        plt.plot(x,integrand(x),MARKERS[i],markersize=MARKER_SIZES[i],markeredgewidth=0.0,label='$\\alpha=$ '+str(alphas[i])[0:4])
        plt.grid()
        #plt.title('Ramsey integrand'+' '+'$e^{x(\\alpha-1)}\sin(\pi e^{-x})^2 / (\pi e^{-x})^2$')
        plt.xlabel('$x$',fontsize=75)
        plt.ylabel('Integrand',fontsize=FONT_SIZE)
    prop=matplotlib.font_manager.FontProperties(size=FONT_SIZE)
    plt.legend(loc='upper left',prop=prop,numpoints=1)
    plt.grid(linestyle=GRIDLINE_STYLE,linewidth=GRIDLINE_WIDTH)
    ax = fig.get_axes()[0]
    for line in ax.xaxis.get_ticklines():
        line.set_markeredgewidth(MARKER_EDGE_WIDTH)
        line.set_markersize(MARKERSIZE)
    for line in ax.yaxis.get_ticklines():
        line.set_markeredgewidth(MARKER_EDGE_WIDTH)
        line.set_markersize(MARKERSIZE)

def showEchoIntegrands(alphas,x):
    integrands = [echoIntegrand_1(alpha) for alpha in alphas]
    fig=plt.figure()
    fig.subplots_adjust(bottom=0.12)
    plt.rcParams['xtick.labelsize']=42
    plt.rcParams['ytick.labelsize']=42
    plt.rcParams['font.size']=50
    for i,integrand in enumerate(integrands):
        plt.plot(x,integrand(x),linewidth=5,label='$\\alpha=$ '+str(alphas[i])[0:4])
        plt.grid()
        #plt.title('Echo integrand'+' '+'$e^{x(\\alpha-1)}\sin(\pi e^{-x})^2 / (\pi e^{-x})^2$')
        plt.xlabel('$x$',fontsize=90)
        plt.ylabel('Integrand',fontsize=52)
    prop=matplotlib.font_manager.FontProperties(size=42)
    plt.legend(loc='upper left',prop=prop)
    
def showIntegrals(integrals,ts):
    fig=plt.figure()
    fig.subplots_adjust(bottom=0.13)
    plt.rcParams['xtick.labelsize']=FONT_SIZE
    plt.rcParams['ytick.labelsize']=FONT_SIZE
    plt.rcParams['font.size']=FONT_SIZE
    i=-1
    for alpha,fMin,fMax,integral in integrals:
        i+=1
        plt.plot(ts,integral(ts),MARKERS[i],markersize=MARKER_SIZES[i],markeredgewidth=0.0,label='$\\alpha=$ '+str(alpha)[0:4])
        #plt.title('Ramsey integrals - $f_{\mathrm{min}}=$'+str(1.0E-9/fMin)+' seconds')
        plt.xlabel('$t$'+' [ns]',fontsize=60)
        plt.ylabel('$I$',fontsize=75)
    prop=matplotlib.font_manager.FontProperties(size=FONT_SIZE)
    plt.legend(loc='upper right',prop=prop,numpoints=1)
    plt.grid(linestyle=GRIDLINE_STYLE,linewidth=GRIDLINE_WIDTH)
    ax = fig.get_axes()[0]
    for line in ax.xaxis.get_ticklines():
        line.set_markeredgewidth(MARKER_EDGE_WIDTH)
        line.set_markersize(MARKERSIZE)
    for line in ax.yaxis.get_ticklines():
        line.set_markeredgewidth(MARKER_EDGE_WIDTH)
        line.set_markersize(MARKERSIZE)
    
def ramseyIntegrand(x,alpha):
    """The expression that must be integrated to give <\phi^2> for the Ramsey

    PARAMETERS
    x - scalar: value of t*f_{min}
    alpha - scalar: noise exponent according to S = 1/f^\\alpha

    OUTPUT
    Returns a single scalar, the value of the integrand evaluated at
    x and alpha
    """
    return ((sin(pi * exp(-x))**2)/((pi*exp(-x))**2))*exp(x*(alpha-1))

integrand = ramseyIntegrand

def ramseyIntegrand_1(alpha):
    """Ramsey integrand for chosen alpha but with x unevaluated.

    PARAMETERS
    alpha - scalar: noise exponent according to S = 1/f^\\alpha

    OUTPUT
    Returns a function of x, representing the Ramsey integrand for fixed
    alpha
    """
    def func(x):
        return ((sin(pi * exp(-x))**2)/((pi*exp(-x))**2))*exp(x*(alpha-1))
    return func

integrand_1 = ramseyIntegrand_1
    
def echoIntegrand(x,alpha):
    """The expression which must be integrated to give <\phi^2> for the spin Echo"""
    return (1.0/(2.0**alpha))*(sin(pi * exp(-x))**2)/((pi*exp(-x))**2)*(1-cos(2*pi*exp(-x)))*exp(x*(alpha-1))

def echoIntegrand_1(alpha):
    """The echo integrand function with a specific alpha, but unevaluated x"""
    def func(x):
        return (1.0/(2.0**alpha))*(sin(pi * exp(-x))**2)/((pi*exp(-x))**2)*(1-cos(2*pi*exp(-x)))*exp(x*(alpha-1))
    return func
        
def ramseyInterpForAlpha(alpha,ts,fMin,fMax):
    """Interpolating function giving the result of integrating the Ramsey
    integrand for a fixed alpha but different values of t.
    """
    #If no maximum frequency is assigned, just us a 
    integrationResults = np.array([])
    for t in ts:
        integral1 = integrate.quad(integrand_1(alpha),0,-np.log(fMin*t))[0]
        integral2 = integrate.quad(integrand_1(alpha),-np.log(fMax*t),0)[0]
        integral = integral1+integral2
        integrationResults = np.hstack((integrationResults,integral))
    interp = interpolate.interp1d(ts,integrationResults,kind='cubic')
    return interp

interpForAlpha = ramseyInterpForAlpha

def echoInterpForAlpha(alpha,ts,fMin,fMax):
    """Interpolating function giving the result of integrating the Echo
    integrand for a fixed alpha over a range of values of t
    """
    integrationResults = np.array([])
    for t in ts:
        integral = integrate.quad(echoIntegrand_1(alpha),-np.log(fMax*t)/2.0,-np.log(fMin*t)/2.0)[0]
        integrationResults = np.hstack((integrationResults,integral))
    interp = interpolate.interp1d(ts,integrationResults,kind='cubic')
    return interp

def getRamseyInterpolationFunctions(alphas,ts,fMin,fMax):
    """List of interpolating functions representing the Ramsey integral.
    Each element in the list is for a different value of alpha
    """
    results=[]
    for alpha in alphas:
        results.append((alpha,fMin,fMax,interpForAlpha(alpha,ts,fMin,fMax)))
    return results

getInterpolationFunctions = getRamseyInterpolationFunctions

def getEchoInterpolationFunctions(alphas,ts,fMin,fMax):
    """List of interpolating functions representing the Echo integral.
    Each element in the list is for a different alpha
    """
    results = []
    for alpha in alphas:
        results.append((alpha,fMin,fMax,echoInterpForAlpha(alpha,ts,fMin,fMax)))
    return results