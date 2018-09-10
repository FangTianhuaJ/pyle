from pyle.plotting import dstools
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import itertools

#################################################
### High level functions to get and plot data ###
#################################################
    
def plotSwapFreq(cxn,datasetNumber=None,session=None,dir=None):
    dv=cxn.data_vault
    if session is None and dir is None:
        raise Exception('You must specify either the session or the directory with the swap spectroscopy.')
    if session is not None and dir is not None:
        raise Exception('You can only specify one of session or dir.')
    if session is not None:
        dir = session._dir
    if datasetNumber is None and session is None:
        raise Exception('You must specify a datasetNumber if providing the directory and not the session.')
    if datasetNumber is None:
        dv.cd(dir)
        datasetNumber=int(dv.dir()[1][-1][:5])
    dstools.plotSwapSpectro(dstools.getSwapSpectroscopy(dv,datasetNumber,dir))
    
    
def plot1DScans(dataset, colors=None, title='', **kw):
    """Plot a 1D scan with an arbitrary number of curves
    
    **kw: keyword dictionary to be passed to plotting functions
    """
    if colors is None:
        colors = itertools.cycle(['b','r','g','k'])
    plt.rcParams['font.size']=20
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    indepLabel, indepUnit = dataset.variables[0][0]
    indepData = dataset.data[:,0]
    nIndeps = dataset.data.shape[1]-1
    for i in range(nIndeps):
        depData = dataset.data[:,i+1]
        depLabel,depLegend,depUnit = dataset.variables[1][i]
        ax.plot(indepData, depData, label=depLegend, color=colors.next(), marker='.', **kw)
    plt.legend()
    depStr = depLabel+' [%s]'%depUnit if depUnit else depLabel
    indepStr = indepLabel+' [%s]'%indepUnit if indepUnit else indepLabel
    plt.xlabel(indepStr)
    plt.ylabel(depStr)
    plt.title(title+': '+str(dataset.path))
    plt.grid()
    fig.show()
    
def addData(ax):
    pass
    
def makePresentable(ax, dataset=None):
    xaxis = ax.xaxis
    yaxis = ax.yaxis
    #Set tick line size
    for line in xaxis.get_ticklines():
        line.set_markeredgewidth(10)
        line.set_markersize(15)
    for line in yaxis.get_ticklines():
        line.set_markeredgewidth(10)
        line.set_markersize(15)
    for label in xaxis.get_ticklabels():
        label.set_fontsize(60)
    for label in yaxis.get_ticklabels():
        label.set_fontsize(60)
    ax.grid(linewidth=2)
    prop = fm.FontProperties(size=35)
    ax.legend(loc='lower left', numpoints=1, prop=prop)
    if dataset is not None:
        ax.set_title(dataset.path, size=40)
