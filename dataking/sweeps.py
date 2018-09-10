import numpy as np

import pyle
from pyle.pipeline import pmap, returnValue
from pyle.util import getch
import datetime
from pyle.datavault import LocalDataSet


def prepDataset(sample, name, axes=None, dependents=None, measure=None, cxn=None, kw=None):
    """Prepare dataset for a sweep.

    This function builds a dictionary of keyword arguments to be used to create
    a Dataset object for a sweep.  Sample should be a dict-like object
    (usually a copy of the sample as returned by loadQubits) that contains current
    parameter settings.  Name is the name of the dataset, which will get prepended
    with a list indicating which qubits are in the sample config, and which of them
    are to be measured.  kw is a dictionary of additional parameters that should
    be added to the dataset.

    Axes can be specified explicitly as a tuple of (<name>, <unit>), or else
    by value for use with grid sweeps.  In the latter case (grid sweep), you should
    specify axes as (<value>, <name>).  If value is iterable, the axis will be
    added to the list of dependent variables so the value can be swept (we look
    at element [0] to get the units); if value is not iterable it will be added
    to the dictionary of static parameters.

    Dependents is either a list of (<name>, <label>, <unit>) designations for the
    dependent variables, or None.  If no list is provided, then the dependents are
    assumed to be probabilities.  In this case, the measure variable is used to
    determine the appropriate set of probabilities: for one qubit, we assume only
    P1 will be measured, while for N qubits all 2^N probabilities are assumed to
    be measured, in the order P00...00, P00...01, P00...10,..., P11...11.  If this
    is not what you want, you must specify the independent variables explicitly.

    Note that measure can be None (all qubits assumed to be measured), an integer
    (just one qubit measured, identified by index in sample['config']) or a list
    of integers (multiple qubits measured).
    """
    conf = list(sample['config'])

    # copy parameters
    kw = {} if kw is None else dict(kw)
    kw.update(sample) # copy all sample data

    kw['_datetime'] = datetime.datetime.now().strftime("%Y-%m-%d, %H:%M")

    if measure is None:
        measure = range(len(conf))
    elif isinstance(measure, (int, long)):
        measure = [measure]

    if hasattr(measure, 'params'):
        # this is a Measurer
        kw.update(measure.params())
    else:
        kw['measure'] = measure

    # update dataset name to reflect which qubits are measured
    for i, q in enumerate(conf):
        if i in kw['measure']:
            conf[i] = '|%s>' % q
    name = '%s: %s' % (', '.join(conf), name)

    # create list of independent vars
    independents = []
    for param, label in axes:
        if isinstance(param, str):
            # param specified as string name
            independents.append((param, label))
        elif np.iterable(param):
            # param value will be swept
            try:
                units = param[0].units
            except Exception:
                units = ''
            independents.append((label, units))
        else:
            # param value is static
            kw[label] = param

    # create list of dependent vars
    if dependents is None:
        if hasattr(measure, 'dependents'):
            # this is a Measurer
            dependents = measure.dependents()
        else:
            n = len(measure)
            if n == 1:
                labels = ['|1>']
            else:
                labels = ['|%s>' % bin(i)[2:].rjust(n,'0') for i in xrange(2**n)]
            dependents = [('Probability', s, '') for s in labels]

    return pyle.Dataset(
        path=list(sample._dir),
        name=name,
        independents=independents,
        dependents=dependents,
        params=kw,
        cxn=cxn
    )


def run(func, sweep, save=True, dataset=None,
        abortable=True, abortPrefix=[],
        collect=True, noisy=True, pipesize=10):
    """Run a function pipelined over an iterable sweep.

    func: function that will be called once for each value in the sweep.
          should be written to work with pipelining, and should
          return sequence objects ready to be sent to the qubit sequencer.
    sweep: an iterable which returns successive values to be
           passed to func

    abortable: if True, check for keypresses to allow the sweep to be aborted cleanly
    save: if True, create a new dataset (using ds_info) and save all data to it
    collect: if True, collect the data into an array and return it
    noisy: if True, print each row of data as it comes in

    dataset: a dataset that will be called with the iterable of data to be saved
    pipesize: the number of pipelined calls to func that should be run in parallel


    The following additional parameters usually only need to be modified for defining
    new types of sweep, such as grid:

    abortPrefix: passed along to checkAbort for abortable sweeps
    """
    with pyle.QubitSequencer() as sequencer:
        # wrap the sweep iterator to handle keypresses
        if abortable:
            sweep = checkAbort(sweep, prefix=abortPrefix)

        # wrap the function to pass the qubit sequencer as the first param
        def wrapped(val):
            ans = yield func(sequencer, val)
            ans = np.asarray(ans)
            if noisy:
                if len(ans.shape) == 1:
                    rows = [ans]
                else:
                    rows = ans
                for row in rows:
                    cont = list()
                    for v in row:
                        if hasattr(v, 'unit'):
                            cont.append(("%0.6g %s" %(v[v.unit], v.unit)).ljust(12))
                        else:
                            cont.append(("%0.6g" %v).ljust(12))
                    print " ".join(cont)
                    # print ' '.join(('%0.3g' % v).ljust(8) for v in row)
            returnValue(ans)

        # Build the generator that returns results of func. Note that this generator
        # doesn't execute any code yet, and won't until .next() is called.
        iter = pmap(wrapped, sweep, size=pipesize)
        #Massage iter so that the datavault can catch incoming data
        if save and dataset:
            iter = dataset.capture(iter)

        # run the iterable, and either collect or discard
        if collect:
            data = np.vstack(iter)
            if all( [ hasattr(dataset, attr) for attr in
                     ('dependents', 'independents', 'params', 'name') ] ):
                #Pack data as a dataset object. This unifies data processing
                #from within data acquisition scripts and from other processing
                #done later.
                dependents = dataset.dependents
                independents = dataset.independents
                params = dataset.params
                name = dataset.name
                return LocalDataSet(data, independents, dependents, params, name)
            else:
                return data
        else:
            for _ in iter: pass


def grid(func, axes, **kw):
    """Run a pipelined sweep on a grid over the given list of axes.

    The axes should be specified as a list of (value, label) tuples.
    We iterate over each axis that is iterable, leaving others constant.
    Func should be written to return only the dependent variable data
    (e.g. probabilities), and the independent variables that are being
    swept will be prepended automatically before the data is passed along.

    All other keyword arguments to this function are passed directly to run.
    """
    def gridSweep(axes):
        if not len(axes):
            yield (), ()
        else:
            (param, _label), rest = axes[0], axes[1:]
            if np.iterable(param): # TODO: different way to detect if something should be swept
                for val in param:
                    for all, swept in gridSweep(rest):
                        yield (val,) + all, (val,) + swept
            else:
                for all, swept in gridSweep(rest):
                    yield (param,) + all, swept

    # pass in all params to the function, but only prepend swept params to data
    def wrapped(server, args):
        all, swept = args
        ans = yield func(server, *all)
        ans = np.asarray(getValue(ans))
        pre = np.asarray(getValue(swept))
        # ans = np.asarray(ans)
        # pre = np.asarray(swept)
        if len(ans.shape) != 1:
            pre = np.tile([pre], (ans.shape[0], 1))
        returnValue(np.hstack((pre, ans)))

    return run(wrapped, gridSweep(axes), abortPrefix=[1], **kw)


def getValue(dataIn):
    # if dataIn is a ndarray, just do nothing
    if isinstance(dataIn, np.ndarray):
        return dataIn
    # if dataIn has a unit, return the raw data
    if hasattr(dataIn, 'unit'):
        dataIn = dataIn[dataIn.unit]
        return dataIn
    # in this case, elements of dataIn may have different units
    # or only parts of them are Values
    if np.iterable(dataIn):
        dataOut = [getValue(d) for d in dataIn]
        return dataOut
    else:
        if hasattr(dataIn, 'unit'):
            dataIn = dataIn[dataIn.unit]
        return dataIn

def checkAbort(iterable, labels=[], prefix=[]):
    """Wrap an iterator to allow it to be aborted during iteration.

    Pressing ESC will cause the iterable to abort immediately.
    Alternately, pressing a number key (1, 2, 3, etc.) will abort
    the next time there is a change at a specific index in the value
    produced by the iterable.  This assumes that the source iterable
    returns values at each step that are indexable (e.g. tuples) so
    that we can grab a particular element and check if it has changed.

    In addition, the optional prefix parameter allows to specify a part
    of the value at each step to be monitored for changes.  For example,
    grid sweeps produce two tuples, a tuple of all current values,
    and a second tuple of the current values of only the swept parameters
    (the tuple of all values is what gets passed to the sweep function,
    while the second tuple of just swept parameters is what gets passed
    to the data vault).  In this case, the prefix would be set to [1]
    so that we only check the second tuple for changes.
    """
    idx = -1
    last = None
    for val in iterable:
        curr = val
        for i in prefix:
            curr = curr[i]
        key = getch.getch()
        if key is not None:
            if key == '\x1b':
                print 'Abort scan'
                break
            elif hasattr(curr, '__len__') and key in [str(i+1) for i in xrange(len(curr))]:
                idx = int(key) - 1
                if labels:
                    print 'Abort scan on next change of %s' % labels[idx]
                else:
                    print 'Abort scan on next change at index %d' % idx
            elif key == '\r':
                if idx >= 0:
                    idx = -1
                    print 'Abort canceled'
        if (idx >= 0) and (last is not None):
            if curr[idx] != last[idx]:
                break
        yield val
        last = curr

def wrap_function(func, args):
    ncalls = [0]
    def function_wrapper(x):
        ncalls[0] += 1
        return func(x, *args)
    return ncalls, function_wrapper

def fmin(func, x0, dataset, args=(), xtol=1e-4, ftol=1e-4, nonzdelt=0.05, zdelt=0.00025,
         maxiter=None, maxfun=None, full_output=0, disp=1, retall=0, callback=None):
    """
    Minimize a function using the downhill simplex algorithm.

    This algorithm only uses function values, not derivatives or second
    derivatives.

    This function is mainly copied from scipy.optimize.fmin. dataset argument is added
    to save data to the datavault

    Parameters
    ----------
    func : callable func(x,*args)
        The objective function to be minimized.
    x0 : ndarray
        Initial guess.
    dataset: Dataset object
        add intermediate data to datavault
    args : tuple
        Extra arguments passed to func, i.e. ``f(x,*args)``.
    callback : callable
        Called after each iteration, as callback(xk), where xk is the
        current parameter vector.

    Returns
    -------
    xopt : ndarray
        Parameter that minimizes function.
    fopt : float
        Value of function at minimum: ``fopt = func(xopt)``.
    iter : int
        Number of iterations performed.
    funcalls : int
        Number of function calls made.
    warnflag : int
        1 : Maximum number of function evaluations made.
        2 : Maximum number of iterations reached.
    allvecs : list
        Solution at each iteration.

    Other parameters
    ----------------
    xtol : float
        Relative error in xopt acceptable for convergence.
    ftol : number
        Relative error in func(xopt) acceptable for convergence.
    maxiter : int
        Maximum number of iterations to perform.
    maxfun : number
        Maximum number of function evaluations to make.
    full_output : bool
        Set to True if fopt and warnflag outputs are desired.
    disp : bool
        Set to True to print convergence messages.
    retall : bool
        Set to True to return list of solutions at each iteration.

    Notes
    -----
    Uses a Nelder-Mead simplex algorithm to find the minimum of function of
    one or more variables.

    This algorithm has a long history of successful use in applications.
    But it will usually be slower than an algorithm that uses first or
    second derivative information. In practice it can have poor
    performance in high-dimensional problems and is not robust to
    minimizing complicated functions. Additionally, there currently is no
    complete theory describing when the algorithm will successfully
    converge to the minimum, or how fast it will if it does.

    References
    ----------
    Nelder, J.A. and Mead, R. (1965), "A simplex method for function
    minimization", The Computer Journal, 7, pp. 308-313
    Wright, M.H. (1996), "Direct Search Methods: Once Scorned, Now
    Respectable", in Numerical Analysis 1995, Proceedings of the
    1995 Dundee Biennial Conference in Numerical Analysis, D.F.
    Griffiths and G.A. Watson (Eds.), Addison Wesley Longman,
    Harlow, UK, pp. 191-208.

    """

    dataset.connect()
    fcalls, func = wrap_function(func, args)
    x0 = np.asfarray(x0).flatten()
    N = len(x0)
    rank = len(x0.shape)
    if not -1 < rank < 2:
        raise ValueError("Initial guess must be a scalar or rank-1 sequence.")
    if maxiter is None:
        maxiter = N * 200
    if maxfun is None:
        maxfun = N * 200

    rho, chi, psi, sigma = 1, 2, 0.5, 0.5
    one2np1 = list(range(1, N + 1))
    if rank == 0:
        sim = np.zeros((N + 1,), dtype=x0.dtype)
    else:
        sim = np.zeros((N + 1, N), dtype=x0.dtype)
    fsim = np.zeros((N + 1,), float)

    sim[0] = x0
    if retall:
        allvecs = [sim[0]]

    fsim[0], stdx0 = func(x0)
    dataset.add(np.hstack((fcalls[0], fsim[0], stdx0, x0)))
    print("starting value")

    for k in range(0, N):
        y = np.array(x0, copy=True)
        if y[k] != 0:
            y[k] = (1 + nonzdelt) * y[k]
        else:
            y[k] = zdelt
        sim[k + 1] = y
        f, stdy = func(y)
        dataset.add(np.hstack((fcalls[0], f, stdy, y)))
        print("initial %s value" % k)
        fsim[k + 1] = f

    ind = np.argsort(fsim)
    fsim = np.take(fsim, ind, 0)
    sim = np.take(sim, ind, 0)  # sim[0,:] is the best one among sim

    iterations = 1

    while fcalls[0] < maxfun and iterations < maxiter:
        if (np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= xtol and
                    np.max(np.abs(fsim[0] - fsim[1:])) <= ftol):
            break

        xbar = np.add.reduce(sim[:-1], 0) / N

        # reflection
        xr = (1 + rho) * xbar - rho * sim[-1]
        fxr, stdxr = func(xr)
        dataset.add(np.hstack((fcalls[0], fxr, stdxr, xr)))
        print("reflection")
        doshrink = 0

        if fxr < fsim[0]:
            # expansion
            xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
            fxe, stdxe = func(xe)
            dataset.add(np.hstack((fcalls[0], fxe, stdxe, xe)))
            print("expansion")
            if fxe < fxr:
                sim[-1] = xe
                fsim[-1] = fxe
            else:
                sim[-1] = xr
                fsim[-1] = fxr
        else:  # fsim[0] <= fxr
            if fxr < fsim[-2]:
                sim[-1] = xr
                fsim[-1] = fxr
            else:  # fxr >= fsim[-2]
                # Perform contraction
                if fxr < fsim[-1]:
                    xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                    fxc, stdxc = func(xc)
                    dataset.add(np.hstack((fcalls[0], fxc, stdxc, xc)))
                    print("contraction")

                    if fxc <= fxr:
                        sim[-1] = xc
                        fsim[-1] = fxc
                    else:
                        doshrink = 1
                else:
                    # Perform an insider contraction
                    xcc = (1 - psi) * xbar + psi * sim[-1]
                    fxcc, stdxcc = func(xcc)
                    dataset.add(np.hstack((fcalls[0], fxcc, stdxcc, xcc)))
                    print("inside contraction")

                    if fxcc < fsim[-1]:
                        sim[-1] = xcc
                        fsim[-1] = fxcc
                    else:
                        doshrink = 1

                if doshrink:
                    for j in one2np1:
                        sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                        fsim[j], stdsimj = func(sim[j])
                        dataset.add(np.hstack((fcalls[0], fsim[j], stdsimj, sim[j])))
                        print("shrink")

        ind = np.argsort(fsim)
        sim = np.take(sim, ind, 0)
        fsim = np.take(fsim, ind, 0)

        if callback is not None:
            callback(sim[0])
        iterations += 1
        if retall:
            allvecs.append(sim[0])

    x = sim[0]
    fval = min(fsim)
    warnflag = 0

    if fcalls[0] >= maxfun:
        warnflag = 1
        if disp:
            print "Warning: Maximum number of function evaluations has " \
                  "been exceeded."
    elif iterations >= maxiter:
        warnflag = 2
        if disp:
            print "Warning: Maximum number of iterations has been exceeded"
    else:
        if disp:
            print "Optimization terminated successfully."
            print "         Current function value: %f" % fval
            print "         Iterations: %d" % iterations
            print "         Function evaluations: %d" % fcalls[0]

    if full_output:
        retlist = x, fval, iterations, fcalls[0], warnflag
        if retall:
            retlist += (allvecs,)
    else:
        retlist = x
        if retall:
            retlist = (x, allvecs)

    return retlist
