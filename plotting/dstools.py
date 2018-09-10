# dataserver tools
from __future__ import with_statement

from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
import re
from matplotlib import rcParams
from pyle.util import structures #Used to be: from util import structures

from labrad.units import Unit,Value

GHz,V = (Unit(s) for s in ['GHz','V'])

#####################################
### Low level datavault functions ###
#####################################

def name2number(name):
    match = re.match('[0-9]*',name)
    number = name[0:match.end()]
    return int(number)

def addParam(ds, name, value):
    """Add a parameter to a dataset and set its value."""
    ds.add_parameter(name, value)

def addParamDict(ds, params, prefix='', ignore='_'):
    """Add and set a dict of parameters to a dataset."""
    def makeList(params, prefix):
        plist = []
        for key, val in sorted(params.items()):
            if ignore and key.startswith(ignore):
                continue
            if isinstance(val, dict):
                plist.extend(makeList(val, prefix + key + '.'))
            else:
                plist.append((prefix + key, val))
        return plist
    ds.add_parameters(makeList(params, prefix))

def addParamList(ds, params, prefix=''):
    """Add and set a dict of parameters to a dataset."""
    ds.add_parameters([(prefix + key, val) for key, val in params])

def _openDataset(ds, dataset=None, session=None):
    if session:
        ds.cd(session)
    if dataset is None:
        dataset = 0
    if dataset <= 0:
        #dataset += len(ds.dir()[1]) #This doesn't work because the list returned by
        dataset = ds.dir()[1][dataset]
    ds.open(dataset)

def getDataset(ds, dataset=None, session=None):
    _openDataset(ds, dataset, session)
    return np.asarray(ds.get())

def getParameters(ds, dataset=None, session=None):
    _openDataset(ds, dataset, session)
    return dict(ds.get_parameters())

def getMeasureNames(dataset):
    parameters = dataset.parameters
    try:
        ans = parameters['measureNames']
    except KeyError:
        measure = parameters['measure']
        config = parameters['config']
        ans = [config[m] for m in measure]

    return ans

def getMatrix(ds, dataset=None, session=None, independentColumns=None, dependent=-1, default=np.NaN, steps=None):
    if isinstance(dataset, list):
        pts = np.vstack(getDataset(ds, d, session) for d in dataset)
    else:
        pts = getDataset(ds, dataset, session)
    ind, dep = ds.variables()
    ind = [{'name': i[0], 'units': i[1]} for i in ind]
    dep = [{'name': i[0], 'legend': i[1], 'units': i[2]} for i in dep]
    #calculate column index in
    dependent = np.asarray(dependent)
    column = len(ind) + dependent + len(dep)*(dependent<0)
    if independentColumns is None:
        independentColumns = np.arange(len(ind))
    data, info = columns2Matrix(pts, independentColumns, column, default=default, steps=steps)
    if len(independentColumns) == 2:
        info['Xname'] = ind[independentColumns[0]]['name']
        info['Xunits'] = ind[independentColumns[0]]['units']
        info['Yname'] = ind[independentColumns[1]]['name']
        info['Yunits'] = ind[independentColumns[1]]['units']
    info['Inames'] = [ind[i]['name'] for i in independentColumns]
    info['Iunits'] = [ind[i]['units'] for i in independentColumns]

    if np.iterable(dependent):
        info['Dname'] = [dep[d]['name'] for d in dependent]
        info['Dunits'] = [dep[d]['units'] for d in dependent]
    else:
        info['Dname'] = dep[dependent]['name']
        info['Dunits'] = dep[dependent]['units']
    return data, info

#######################################################
### Higher level dataset retrieval and construction ###
#######################################################

def getOneDeviceDataset(ds, datasetNumber, session, deviceName=None,
                        averaged=False, correctVisibility=False):
    """Gets a dataset and gives the measured device a special name
    in the list of dataset parameters.

    PARAMETERS:
        session: path to dataset, ie ['','Person','Wafer','die','session',setNumber]
    KEYWORD ARGUMENTS:
        deviceName - str: Name to give to the measured device
    """
    if deviceName is None:
        deviceName = 'q'

    dataset = getDeviceDataset(ds, datasetNumber, session)
    measure = dataset.parameters.measure
    if isinstance(measure,list):
        if len(measure)>1:
            raise Exception('More than one device listed as measured')
        deviceIndex = measure[0]
    elif isinstance(measure,int):
        deviceIndex = measure
    else:
        raise Exception('measure must be a list or an integer')
    deviceTag=dataset['parameters']['config'][deviceIndex]
    dataset.parameters[deviceName]=dataset.parameters[deviceTag]
    if averaged:
        dataset.data = compressAveragedScan(dataset.data, dataset.parameters.averages)
    if correctVisibility:
        correctOneDeviceOneDimensionalData(dataset)
    return dataset

def getDeviceDataset(dv, datasetId, session):
    """Gets a dataset from the data vault and parses the parameters in a nice way.
    INPUTS
        dv - server object: Data Vault server object
        datasetId - number or string: numerical tag of dataset, or full string name
        session - list of strings: path to the datavault directory containing the dataset
    OUTPUTS
        dataset - AttrDict:
            path -  data vault path for dataset
            data - numpy array: data
            parameters - AttrDict:
                All parameters from the data run
                One sub-AttrDict for each device's parameters
    """
    #Read numerical data as ndarray
    data = getDataset(dv, dataset=datasetId, session=session)
    #Read parameters as AttrDict
    params = _parseDeviceParameters(getParameters(dv, dataset=datasetId,
                                    session=session))
    #Read axis data. This comes back as a tuple of lists. See LabRAD docstring.
    variables = tuple([list(variable) for variable in dv.variables()])
    datasetName = dv.get_name()
    datasetId = name2number(datasetName)
    dataset = structures.AttrDict({'path':session+[datasetId],'data':data,
                                   'parameters':params,'variables':variables})
    return dataset

def _parseDeviceParameters(params):
    """Parses the result of the data vault get_parameters() call into an
    PARAMETERS:
        params, ((name,value),...,(name,value)): result of data_vault.get_parameters().
    RETURNS:
        AttrDict of parameters. All parameters with names with leading characters followed
        by a '.' are assumed to be parameters pertaining to a particular device.
        For example 'qubit0.biasStepEdge' is assumed to be a parameter for a device
        called "qubit0". These parameters are stored in sub AttrDicts with names determined
        by whatever precedes the '.', in this case "qubit0".

    COMMENTS
    LabRAD lists come back to python as labrad.types.List which is an un-unflattened proxy for a python
    list. This is annoying for later processing, so here we convert all list proxies to real lists. This
    is ok here because we don't have huge arrays of numerical data, so unflattening the list proxies is
    ok.
    """

    parsed=structures.AttrDict()

    for name,value in params.items():
        p = re.compile('.*\\.') #Anything followed by a dot
        m = p.match(name) #Match the compiled string search to the BEGINNING of name

        if m is not None:
            #Found a device parameter
            device = m.group()[:-1] #Chop off trailing '.' in parameter name
            parameter = name[m.end():] #Rest of the string is the parameter name
            if device not in parsed.keys():
                #Device hasn't been created yet, so create it
                parsed[device] = structures.AttrDict()
            parsed[device][parameter] = value.aslist if isinstance(value,list) else value
        else:
            #This parameter is not a device parameter so store it in top level
            parsed[name] = value.aslist if isinstance(value,list) else value
    return parsed

###############################################
### Builders for specific types of datasets ###
###############################################

def buildFluxNoiseDataset(ds, datasetNumber, session, averaged=False,
                          getFluxSensitivity=True, randomized=False, deviceName='q'):
    if randomized and averaged:
        raise Exception("Dude, you can't randomize and average at the same time")
    dataset=getOneDeviceDataset(ds,datasetNumber,session,deviceName=deviceName,
                                averaged=averaged)
    if randomized:
        zipped = zip(dataset.data[:,0],dataset.data[:,1:])
        srtd = sorted(zipped)
        unzipped = zip(*srtd)
        data = np.vstack((np.array(unzipped[0]),np.array(unzipped[1]).T)).T
        dataset.data = data
    if getFluxSensitivity:
        dataset['parameters']['dfdPhi0'] = getdfdPhi0(dataset.parameters[deviceName])
    return dataset

def swapSpectroscopyDataset(ds, datasetNumber, session):
    dataset = getOneDeviceDataset(ds, datasetNumber, session)

    p = dataset['parameters']['q']['calZpaFunc']
    p = np.array([float(elem) for elem in p])
    zpas = dataset.data[:,0]
    frequencies = ((zpas - p[1])/p[0])**(0.25)
    probs, info = columns2Matrix(dataset.data)
    probs = np.transpose(probs)
    xAxis = frequencies
    yAxis = np.arange(info['Ymin'],info['Ymax']+info['Ystep'],info['Ystep'])
    dataset.data = structures.AttrDict({'probs':probs,
                    'frequency':xAxis,
                    'time':yAxis})
    return dataset

def getSwapSpectroscopy(dv, datasetNumber, session, p=None):
    dataset = getOneDeviceDataset(dv, datasetNumber, session)
    if p is None:
        p = dataset['parameters']['q']['calZpaFunc']
        p = np.array([float(elem) for elem in p])
    zpas = dataset.data[:,0]
    frequencies = ((zpas - p[1])/p[0])**(0.25)
    dataset.data[:,0] = frequencies
    dataset.variables[0][0] = ('frequency', 'GHz')
    return dataset

def plotSwapSpectro(dataset, reverse=False):
    xAxis, yAxis, dataMatrix = arrange2Dscan(dataset.data)
    print len(xAxis)
    print len(yAxis)
    print dataMatrix.shape
    plt.figure()
    if reverse:
        xAxis = xAxis[-1::-1]
        dataMatrix = dataMatrix[:,-1::-1]
    plt.pcolor(xAxis, yAxis, dataMatrix)
    plt.xlabel('%s [%s]' %(dataset.variables[0][0][0],dataset.variables[0][0][1]))
    plt.ylabel('%s [%s]' %(dataset.variables[0][1][0],dataset.variables[0][1][1]))
    titleString = dataset.path.__repr__()
    titleString = titleString + ' ' + dataset.parameters.config[dataset.parameters.measure[0]]
    plt.title('Swap Spectroscopy: '+titleString)

def arrange2Dscan(data):
    """Rearranges a 2D scan to facilitate plotting

    INPUTS
    data - numpy.array: dataset you want to process
        data[:,0] = x values of data points
        data[:,1] = y values of data points
        data[:,2] = z values of data points, ie measured value

    Columns of data, ie. data points for fixed x value, must occur
    in a continuous series. It is assumed that x is iterated more slowly
    than y. In other words, as you iterate i in data[i,:], you should have
    y values changing fast, and x values changing only when each y range
    finishes.

    The value of x does not have to be in order, ie. columns may have been
    recorded in any order, as long as each column was completed before
    the next was started.
    """
    xs = data[:,0]
    ys = data[:,1]
    zs = data[:,2]

    #Find out how many x and y values we have
    uniqueXs = set()
    uniqueYs = set()
    for x in xs:
        uniqueXs.add(x)
    for y in ys:
        uniqueYs.add(y)
    nCols = len(uniqueXs)
    nRows = len(uniqueYs)

    #Zero padd the data if points are missing
    expectedPoints = nCols*nRows
    if len(zs) != expectedPoints: #If a the dataset is missing some points
        print 'Missing points detected. Adding zeros'
        pads = np.zeros(expectedPoints-len(zs))
        zs.shape= (zs.shape[0],1)
        pads.shape = (pads.shape[0],1)
        zs = np.vstack((zs,pads)) #Add zeros
        zs.shape = (zs.shape[0],)

    dataMatrix = np.zeros((nRows,nCols))

    xAxis = np.zeros(nCols)
    for col in range(nCols):
        #Keep track of what order we go through x values
        xAxis[col] = xs[col*nRows]
        dataMatrix[:,col] = zs[col*nRows:((col+1)*nRows)]

    #Columns are created, but now we need to sort them by x value
    indices = np.argsort(xAxis)
    dataMatrix = dataMatrix[:,indices]

    y = np.array(list(uniqueYs))
    y = y[np.argsort(y)]
    x = np.array(list(uniqueXs))
    x = x[np.argsort(x)]

    return x,y,dataMatrix


def format2D(data):

    dataSize = len(mlab.find(data[:,0]==data[0,0]))
    dataLen = len(data)
    numColumns = dataLen/dataSize
    newDataLen = numColumns*dataSize
    data = data[:newDataLen, :] # we take off any unfinished rows
    xParameter = data[::dataSize, 0]
    yParameter = data[:dataSize,1]
    reshapedData = np.reshape(data[:,2], (numColumns, dataSize)).T

    return xParameter, yParameter, reshapedData

###############
### Utility ###
###############

@contextmanager
def rcTemp(*a, **kw):
    try:
        prev = dict(rcParams)
        rcParams.update(*a, **kw)
        yield
    finally:
        rcParams.update(prev)

def compressAveragedScan(data, averages):
    """
    INPUTS:
    data - ndarray: height is averages times the length of the time axis
    averages - int: number of times the scan is repeated
    """
    L = data.shape[0]
    dataOut = np.array([])
    for index in range(data.shape[1]):
        col = data[:,index]
        col.shape = (L,1)
        row = np.transpose(col)
        reshaped = np.reshape(row,(averages,-1))
        averaged = np.transpose(np.mean(reshaped,0))
        averaged.shape = (averaged.shape[0],1)
        if index ==0:
            dataOut = averaged
        else:
            dataOut = np.hstack((dataOut,averaged))
    return dataOut

def correctOneDeviceOneDimensionalData(dataset, deviceName='q', p=None):
    """Computes corrected one state population for a one device dataset with
    one dimensional data"""
    if p is None:
        p = dataset['parameters'][deviceName]['calScurve1']
    indep = dataset.data[:,0]
    dep = dataset.data[:,1:]
    dep = correctVisibility(dep,p)
    result = np.vstack((indep,dep.T)).T
    dataset.data=result

def correctVisibility(data, p):
    """Takes measured qubit data and corrects for the measurement visibility

    PARAMETERS
    data: ndarray - numpy array of measurement probabilities. ALL entries in this
                    array will be corrected for the measurement visibility.
                    We're assuming that the data here are all one state populations.
    p: tuple - The two numbers that characterize visibility. p[0] is the scurve value for the
               zero state, and p[1] is the scurve value for the one state.
    """
    p0=p[0]
    p1=p[1]
    dataCorrected = (data-p0)/(p1-p0)
    return dataCorrected

def getdfdPhi0(qubit):
    dVdPhi0 = (qubit.squidEdges[1]-qubit.squidEdges[0])*Value(1.0,'1/PhiZero')
    if 'calDfOverDVbias' in qubit.keys():
        dfdV = qubit['calDfOverDVbias']
    elif 'calDfDV' in qubit.keys():
        dfdV = qubit['calDfDV']
    else:
        raise Exception('qubit frequency sensitivity not found')
    if dfdV.isDimensionless():
        print 'dfdV had no units. Assuming GHz/V'
        dfdV = dfdV * GHz/V
    dfdPhi0 = dfdV*dVdPhi0
    return dfdPhi0['GHz/PhiZero']*Value(1.0,'GHz/PhiZero')

################################
### Generic dataset plotting ###
################################

def plotDataset(ds, dataset=None, session=None, style='.-',
                legendLocation=None, cmap = None,
                Xindep=0, Yindep=1, dependent=-1,
                exact2D=False, default=0,
                vmin=None, vmax=None, interpolation='nearest', steps=None):

    pts = getDataset(ds, dataset, session)
    if not len(pts):
        print 'No datapoints.'
        return
    ind, dep  = ds.variables()
    ind = [{'name': i[0], 'units': i[1]} for i in ind]
    dep = [{'name': i[0], 'legend': i[1], 'units': i[2]} for i in dep]
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

def plotDataset1D(pts, ind, dep, style='.-',legendLocation=None,show=False,title='',**kw):

    if len(ind) > 1:
        raise Exception('PlotDataset1D only plots 1D datasets.')

    # for a given name, we may have multiple traces with their own
    # legend entries we'll create a subplot for each of these names,
    # and a legend in each subplot
    # BUT, for now, just plot everything on the same axis (using name and
    # units of the first dependent variable) and create a legend

    with rcTemp(interactive=False):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # plot data
        for i, var in enumerate(dep):
            ax.plot(pts[:,0], pts[:,i+1], style, label=var[1],**kw)

        # make axis labels and title
        ax.set_title(title)
        ax.set_xlabel(makeIndepLabel(ind[0]))
        ax.set_ylabel(makeDepLabel(dep[0]))

        # add a legend
        if legendLocation is not None:
            ax.legend(loc=legendLocation)
        else:
            ax.legend()
        if show:
            plt.show()
        return fig

def makeIndepLabel(var):
    label = var[0]
    if var[1]:
        label+=' [%s]' %var[1]
    return label

def makeDepLabel(var):
    label = var[0]
    if var[2]:
        label+=' [%s]' %var[2]
    return label

def plotDataset2D(pts, ind, dep, cmap = None,
                  Xindep=0, Yindep=1, dependent=-1, default=0,
                  vmin=None, vmax=None, legendLocation=None,
                  interpolation='nearest', steps=None):
    """Plots a 2D dataset.

    No special order of the datapoints is required.  The spacing between
    pixels is the median of nozero changes of indep. variables in neigboring lines.
    Not assigned elements in the matrix are NaN. By default they are plotted like 0.
    If you want them transparent, provide a cmap where you called cmap.set_under(alpha=0.0)
    """
    dependent = np.asarray(dependent)
    column = dependent + len(dep)*(dependent<0) + len(ind)
    data, info = columns2Matrix(pts, [Xindep, Yindep], column, steps=steps, default=0)
    data = np.transpose(data)

    #plot it
    with rcTemp(interactive=False):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        #bug in imshow. Won't plot if data[0,0] is nan
        #if np.isnan(data[0,0]):
        #    data[0,0]=0

        # plot data
        ax.imshow(data, cmap=cmap, aspect='auto', origin='lower',
                  interpolation=interpolation,
                  extent=(info['Xmin'] - 0.5 * info['Xstep'],
                          info['Xmax'] + 0.5 * info['Xstep'],
                          info['Ymin'] - 0.5 * info['Ystep'],
                          info['Ymax'] + 0.5 * info['Ystep']),
                  vmin=vmin, vmax=vmax)

        # add axis labels
        ax.set_xlabel(makeIndepLabel(ind[Xindep]))
        ax.set_ylabel(makeIndepLabel(ind[Yindep]))

        plt.show()

def plotDataset2Dexact(pts, ind, dep, cmap=None, legendLocation=None,
                       vmin=None,vmax=None):
    """Plots a 2D dataset.

    No special order of the datapoints is required. Creates a pixel for every X value in the dataset
    at every Y value in the dataset. Can handle non-uniform spacing, but this requires to use pcolor
    to plot, which gives sluggish display for big images.
    See: pylab.pcolor, plotDataset2Dfast
    """
    if not len(ind) == 2:
        raise Exception('Can only plot 2D datasets')
    n = len(pts[:,0])
    print '%d data points' % n

    #create sorted array of X values (1st indepenent variable)
    argsorted = np.argsort(pts[:,0])

    #changes contains the indices where X changes
    changes = np.ones(n, dtype=int)
    changes[1:] = np.diff(pts[argsorted,0]) > 1e-10*(pts[-1,0] - pts[0,0])

    #now create a sorted array of all X values
    #where each value occurs only once
    Xvalues = pts[argsorted[np.nonzero(changes)],0]
    print '%d different X values' % len(Xvalues)

    #for plotting we do not need the actual X values of the pixel
    #center but the borders between pixels
    Xborders = np.zeros(len(Xvalues) + 1, dtype=float)
    Xborders[1:-1] = 0.5 * (Xvalues[0:-1] + Xvalues[1:])
    Xborders[0] = Xvalues[0]
    Xborders[-1] = Xvalues[-1]

    #For each line in pts make Xindex contain the corresponding
    #index into the Xvalues array. That will be the pixel number in X
    #of this line
    changes[0] = 0
    changes = np.cumsum(changes)
    Xsize = len(Xvalues)
    Xindex = np.zeros(n, dtype=int)
    Xindex[argsorted] = changes

    #Do the same for Y
    argsorted = np.argsort(pts[:,1])

    changes = np.ones(n,dtype=int)
    changes[1:] = np.diff(pts[argsorted,1]) > 1e-10*(pts[-1,0]-pts[0,0])

    Yvalues = pts[argsorted[np.nonzero(changes)],1]
    print '%d different Y values' % len(Yvalues)

    Yborders = np.zeros(len(Yvalues)+1, dtype=float)
    Yborders[1:-1] = 0.5 * (Yvalues[0:-1] + Yvalues[1:])
    Yborders[0] = Yvalues[0]
    Yborders[-1] = Yvalues[-1]

    changes[0]=0
    changes = np.cumsum(changes)
    Ysize = len(Yvalues)
    Yindex = np.zeros(n,dtype=int)
    Yindex[argsorted] = changes

    #Create the 2D image array
    data = np.zeros((Ysize, Xsize), dtype=float)
    data[Yindex,Xindex] = pts[:,2]

    #plot it
    with rcTemp(interactive=False):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        Xborders, Yborders = np.meshgrid(Xborders, Yborders)
        ax.pcolor(Xborders, Yborders, data, edgecolors=None, cmap=cmap,
                  vmin=vmin, vmax=vmax)


        ax.set_xlabel(makeIndepLabel(ind[0]))
        ax.set_ylabel(makeIndepLabel(ind[1]))

        plt.show()
        return Xvalues, Yvalues, data

def columns2Matrix(pts, independentColumns=[0,1], dependentColumn=-1, default=np.NaN, steps=None):
    """Converts data in columns format into a 2D array.

    No special order of the datapoints is required.
    The spacing between pixels is the median of nonzero changes
    of independent variables in neigboring lines.
    """

    #n = len(pts[:,0])
    dims = np.size(independentColumns)
    mins = np.min(pts[:,independentColumns], axis=0)
    maxs = np.max(pts[:,independentColumns], axis=0)
    if steps is None:
        steps = np.ones(dims, dtype=float)
        for i in np.arange(dims):
            colSteps = np.diff(np.sort(pts[:,independentColumns[i]]))
            colSteps = colSteps[np.argwhere(colSteps > 1e-8*(maxs[i]-mins[i]))]
            if len(colSteps):
                steps[i] = np.median(abs(colSteps))
    sizes = (np.round((maxs - mins) / steps)).astype(int) + 1
    indices = tuple([np.round((pts[:,independentColumns[i]]-mins[i]) / steps[i]).astype(long) for i in np.arange(dims)])
    pts = pts[:,dependentColumn]
    if np.iterable(dependentColumn):
        data = np.resize(default, sizes.tolist() + [len(dependentColumn)])
    else:
        data = np.resize(default, sizes)
    #Create the 2D image array
    Vmin = np.min(pts, axis=0)
    Vmax = np.max(pts, axis=0)
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

