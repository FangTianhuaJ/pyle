# dataserver tools
from __future__ import with_statement

import numpy as N
import scipy as S
from pylab import *
from matplotlib import rcParams
from contextlib import contextmanager
from scipy.optimize import leastsq

def addParam(ds, name, value):
    """Add a parameter to a dataset and set its value."""
    ds.add_parameter(name, value)

def addParamDict(ds, params, prefix=''):
    """Add and set a dict of parameters to a dataset."""
    for name in params:
        if name[0] == '_':
            continue
        if isinstance(params[name], dict):
            addParamDict(ds, params[name], prefix = prefix + name + '.')
        else:
            addParam(ds, prefix + name, params[name])

def addParamList(ds, params, prefix=''):
    """Add and set a dict of parameters to a dataset."""
    for name, value in params:
        addParam(ds, prefix + name, value)
            

def _openDataset(ds, dataset=None, session=None):
    if session: ds.cd(session)
    if dataset is None:
        dataset = 0
    if dataset <= 0:
        dataset+= len(ds.dir()[1])
    ds.open(dataset)
    

def getDataset(ds, dataset=None, session=None):
    _openDataset(ds, dataset, session)
    return ds.get().asarray


def getParameters(ds, dataset=None, session=None):
    _openDataset(ds, dataset, session)
    paramlist = ds.get_parameters()
    pars = {}
    for name,value in paramlist:
        pars[name] = value
    return pars


def getMatrix(ds, dataset=None, session=None, independentColumns=None, dependent=-1,default=N.NaN,steps=None):
    if isinstance(dataset, list):
        session = [session]*len(dataset)
        pts = []
        for i,d in enumerate(dataset):
            pts += [getDataset(ds, d, session[i])]
        pts = vstack(pts)
    else:
        pts = getDataset(ds, dataset, session)
    ind,dep = ds.variables()
    ind = extractAttrs(['name', 'units'], ind)
    dep = extractAttrs(['name', 'legend', 'units'], dep)
    #calculate column index in
    dependent = asarray(dependent)
    column = dependent + len(dep)*(dependent<0) + len(ind)
    if independentColumns is None:
        independentColumns = arange(len(ind))
    data, info = columns2Matrix(pts, independentColumns, column,default=default, steps=steps)
    if len(independentColumns) == 2:
        info['Xname'] = ind[independentColumns[0]]['name']
        info['Xunits'] = ind[independentColumns[0]]['units']
        info['Yname'] = ind[independentColumns[1]]['name']
        info['Yunits'] = ind[independentColumns[1]]['units']
    info['Inames'] = [ind[i]['name'] for i in independentColumns]
    info['Iunits'] = [ind[i]['units'] for i in independentColumns]
    
    if iterable(dependent):
        info['Dname'] = [dep[d]['name'] for d in dependent]
        info['Dunits'] = [dep[d]['units'] for d in dependent]
    else:  
        info['Dname'] = dep[dependent]['name']
        info['Dunits'] = dep[dependent]['units']
    return data, info
    




@contextmanager
def rcTemp(*a, **kw):
    try:
        prev = dict(rcParams)
        rcParams.update(*a, **kw)
        yield
    finally:
        rcParams.update(prev)

def extractAttrs(attrs,  varList):
    return [dict(zip(attrs, map(str,v))) for v in varList]





def plotDataset(ds, dataset=None, session=None, style='.-',
                legendLocation=None, cmap = None,
                Xindep=0, Yindep=1, dependent=-1,
                exact2D=False, default=0,
                vmin=None, vmax=None, interpolation='nearest', steps=None):

    pts = getDataset(ds, dataset, session)
    if not len(pts):
        print 'No datapoints.'
        return
    ind,dep  = ds.variables()
    ind = extractAttrs(['name', 'units'], ind)
    dep = extractAttrs(['name', 'legend', 'units'], dep)
    if len(ind) < 2:
        plotDataset1D(pts, ind, dep, style, legendLocation)
    else:
        if exact2D:
            plotDataset2Dexact(pts, ind, dep, cmap, legendLocation)
        else:
            plotDataset2D(pts, ind, dep, cmap = cmap,
                      Xindep=Xindep, Yindep=Yindep,
                      dependent=dependent, default=default,
                      vmin=vmin, vmax=vmax, legendLocation=legendLocation,
                      interpolation=interpolation, steps=steps)

def plotDataset1D(pts, ind, dep, style='.-', legendLocation=None):
  
    if len(ind) > 1:
        raise Exception('PlotDataset1D only plots 1D datasets.')


    # for a given name, we may have multiple traces with their own
    # legend entries we'll create a subplot for each of these names,
    # and a legend in each subplot
    # BUT, for now, just plot everything on the same axis (using name and
    # units of the first dependent variable) and create a legend

    with rcTemp(interactive=False):
        figure()
        hold(True)
        for i, d in enumerate(dep):
            plot(pts[:,0], pts[:,i+1], style)

        x = ind[0]
        lbl = x['name']
        if x['units']:
            lbl += ' (%s)' % x['units']
        xlabel(lbl)

        y = dep[0]
        lbl = y['name']
        if y['units']:
            lbl += u' (%s)' % y['units']
        ylabel(lbl)

        entries = [d['legend'] for d in dep]
        if legendLocation:
            legend(entries, loc=legendLocation)
        else:
            legend(entries)
        show()


def plotDataset2Dexact(pts, ind, dep, cmap=None, legendLocation=None,
                       vmin=None,vmax=None):

    """plots a 2D dataset.
    No special order of the datapoints is required.
    Creates a pixel for every X value in the dataset
    at every Y value in the dataset.
    Can handle non-uniform spacing, but this requires to use pcolor
    to plot, which gives sluggish display for big images.
    See: pylab.pcolor, plotDataset2Dfast
    """
    if not len(ind) == 2:
        raise Exception('Can only plot 2D datasets')
    n = len(pts[:,0])
    print '%d data points' % n

    #create sorted array of X values (1st indepenent variable)
    argsorted = argsort(pts[:,0])

    #changes contains the indices where X changes
    changes = ones(n,dtype=int)
    changes[1:] = diff(pts[argsorted,0]) > 1e-10*(pts[-1,0]-pts[0,0])

    #now create a sorted array of all X values
    #where each value occurs only once
    Xvalues = pts[argsorted[nonzero(changes)],0]
    print '%d different X values' % len(Xvalues)

    #for plotting we do not need the actual X values of the pixel
    #center but the borders between pixels
    Xborders = zeros(len(Xvalues) + 1, dtype = float)
    Xborders[1:-1] = 0.5 * (Xvalues[0:-1] + Xvalues[1:])
    Xborders[0] = Xvalues[0]
    Xborders[-1] = Xvalues[-1]

    #For each line in pts make Xindex contain the corresponding
    #index into the Xvalues array. That will be the pixel number in X
    #of this line
    changes[0] = 0
    changes = cumsum(changes)
    Xsize = len(Xvalues)
    Xindex=zeros(n,dtype=int)
    Xindex[argsorted] = changes

    #Do the same for Y
    argsorted = argsort(pts[:,1])

    changes = ones(n,dtype=int)
    changes[1:] = diff(pts[argsorted,1]) > 1e-10*(pts[-1,0]-pts[0,0])

    Yvalues = pts[argsorted[nonzero(changes)],1]
    print '%d different Y values' % len(Yvalues)

    Yborders = zeros(len(Yvalues)+1,dtype=float)
    Yborders[1:-1] = 0.5 * (Yvalues[0:-1] + Yvalues[1:])
    Yborders[0] = Yvalues[0]
    Yborders[-1] = Yvalues[-1]

    changes[0]=0
    changes = cumsum(changes)
    Ysize = len(Yvalues)
    Yindex=zeros(n,dtype=int)
    Yindex[argsorted] = changes

    #Create the 2D image array
    data = zeros((Ysize,Xsize),dtype=float)
    data[Yindex,Xindex] = pts[:,2]

    #plot it
    with rcTemp(interactive=False):
        figure()
        hold(True)
        Xborders,Yborders = meshgrid(Xborders, Yborders)
        pcolor(Xborders,Yborders,data,shading='flat',cmap=cmap,
               vmin=vmin,vmax=vmax)

        x = ind[0]
        lbl = x[0]
        if len(x)>1:
            lbl += ' [%s]' % x[1]
        xlabel(lbl)

        y = ind[1]
        lbl = y[0]
        if len(y)>1:
            lbl += ' [%s]' % y[1]
        ylabel(lbl)
        show()
        return Xvalues,Yvalues,data

def columns2Matrix(pts, independentColumns=[0,1], dependentColumn=-1,default=N.NaN, steps=None):
    
    """converts data in columns format into a Matrix.
    No special order of the datapoints is required.
    The spacing between pixels is the median of nozero changes
    of indep. variables in neigboring lines.
    """

    #n = len(pts[:,0])
    dims  = size(independentColumns)
    mins = N.min(pts[:,independentColumns],axis=0)
    maxs = N.max(pts[:,independentColumns],axis=0)
    if steps is None:
        steps = ones(dims,dtype=float)
        for i in arange(dims):
            colSteps = N.diff(N.sort(pts[:,independentColumns[i]]))
            colSteps = colSteps[N.argwhere(colSteps>1e-8*(maxs[i]-mins[i]))]
            if len(colSteps):
                steps[i] = N.median(abs(colSteps))
    sizes = (N.round((maxs - mins) / steps)).astype(int) + 1
    indices = tuple([N.round((pts[:,independentColumns[i]]-mins[i]) / steps[i]).astype(long) for i in arange(dims)])
    pts = pts[:,dependentColumn]
    if iterable(dependentColumn):
        data = N.resize(default,sizes.tolist()+[len(dependentColumn)])
    else:
        data = N.resize(default,sizes)
    #Create the 2D image array
    Vmin = N.min(pts,axis=0)
    Vmax = N.max(pts,axis=0)
    data[indices] = pts
    info = {'Dmin': Vmin, 'Dmax': Vmax,
            'Imin': mins, 'Imax': maxs, 'Isteps': steps}
    if dims == 2:
        info['Xmin'] = mins[0]
        info['Xmax'] = maxs[0]
        info['Xstep'] = steps[0]
        info['Ymin'] = mins[1]
        info['Ymax'] = maxs[1]
        info['Ystep'] = steps[1]
        
    return data, info
           


def plotDataset2D(pts, ind, dep, cmap = None,
                      Xindep=0, Yindep=1, dependent=-1, default=0,
                      vmin=None, vmax=None, legendLocation=None,
                      interpolation='nearest', steps=None):
    
    """plots a 2D dataset.
    No special order of the datapoints is required.
    The spacing between pixels is the median of nozero changes
    of indep. variables in neigboring lines.
    Not assigned elements in the matrix are NaN. By default they are plotted like 0. If you want them transparent, provide a cmap where you called
    cmap.set_under(alpha=0.0) before
    """
    dependent = asarray(dependent)
    column = dependent + len(dep)*(dependent<0) + len(ind)
    data, info = columns2Matrix(pts, [Xindep,Yindep], column, steps=steps,
                                default=0)
    data = transpose(data)

    #plot it
    with rcTemp(interactive=False):
        figure()
        hold(True)
        #bug in imshow. Won't plot if data[0,0] is nan
        #if N.isnan(data[0,0]):
        #    data[0,0]=0
            
        imshow(data, cmap=cmap, aspect='auto', origin='lower',
               interpolation = interpolation,
               extent = (info['Xmin'] - 0.5 * info['Xstep'],
                         info['Xmax'] + 0.5 * info['Xstep'],
                         info['Ymin'] - 0.5 * info['Ystep'],
                         info['Ymax'] + 0.5 * info['Ystep']),
               vmin=vmin, vmax=vmax)
        x = ind[Xindep]
        lbl = x['name']
        if x['units']:
            lbl += ' [%s]' % x['units']
        xlabel(lbl)

        y = ind[Yindep]
        lbl = y['name']
        if y['units']:
            lbl += ' [%s]' % y['units']
        ylabel(lbl)
        show()
