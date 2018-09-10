import pyle.registry as registry
try:
    import pyle.util.registry_wrapper2 as registry2
except:
    registry2 = registry

import numpy as np
import labrad
import labrad.units
import pyle.util.cache
import pyle.util.job_queue
import copy

import ConfigParser
import os
import contextlib
try:
    import tkFileDialog as tkFD
except:
    pass

# this function is used for print numpy array
@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)

class LocalDataSet(np.ndarray):
    '''
    This wraps a dataset as a numpy array with extra attributes.  This is just following
    the example from http://docs.scipy.org/doc/numpy/user/basics.subclassing.html#basics-subclassing

    Since we inheret from numpy array, all the normal numpy stuff works, but
    you can also access the variable names, parameters, and comments as attributes.

    There is also a column accessor that can index by column name.  So if you have
    a data set of I, Q vs. time, you can say:

    plt.plot(ds.get_column('time'), ds.column(('I', 'Q'))

    This is the base class that represets just a local dataset.  It can be returned
    by methods like sweeps.grid() and is compatible with datasets returned by the
    datavault wrapper, just without comments and tags.
    '''
    def __new__(cls, data, indep, dep, parameters, name):
        '''
        This magic is copied from the numpy docs.  It is needed
        to work properly when constructing views.

        This probably doesn't actually work right if you try to
        make a view, then index it by column name -- somewhere
        we have to change the variable names for a view that
        only contains some of the columns.
        '''
        obj = np.asarray(data).view(cls)
        obj.indep = list(indep) # Convert labrad LazyList to regular list
        obj.dep = list(dep)
        obj.name = name     # name and number in data vault
        if isinstance(parameters, dict):
            obj.parameters = copy.deepcopy(parameters)
        else:
            obj.parameters = parse_parameters(parameters)
        return obj

    def __reduce__(self):
        '''
        Support for pickling.

        By default, subclasses will be pickled as a LocalDataSet
        '''
        return (LocalDataSet, (np.array(self), self.indep, self.dep,
                               self.parameters, self.name), None, None, None)

    def __array_finalize__(self, obj):
        '''
        Like __new__ this is copied from numpy docs.
        This is responsible for fixing up a view or slice.
        Ideally it really needs to figure out which columns are in
        the view and only include those.
        '''
        if obj is None: return
        self.indep = getattr(obj, 'indep', None)
        self.dep = getattr(obj, 'dep', None)
        self.name = getattr(obj, 'name', None)
        self.parameters = getattr(obj, 'parameters', None)
    def __array_wrap__(self, obj):
        '''
        This function is called at the end of uops and similar functions.
        ndarray has some weird logic where reductions like sum() that result
        in zero-rank arrays automatically convert to numpy scalars
        if-and-only-if the type is a ndarray base class.  If we want the same
        we have to do it here
        '''
        if obj.shape == ():
            return obj[()]
        else:
            return np.ndarray.__array_wrap__(self,obj)
    def __getitem__(self, idx):
        '''
        I am not sure if this is a good idea.  It overloads the []
        operator to work on strings.  If you try to index a
        LocalDataSet with a string or a tuple containing at least
        one string, it calls column:

        dsw['x'] or dsw['x', 'y'] or dsw['t', 1, 2]

        will all index columns of the data set.  However:

        dsw[0, 1, 2]

        Will call through to the numpy [] operator.
        '''
        if isinstance(idx, basestring):
            return self.get_column(idx)
        if isinstance(idx, tuple) and any([isinstance(x, basestring) for x in idx]):
            return self.column(idx)
        return np.ndarray.__getitem__(self, idx)

    def unit(self, col_idx):
        if col_idx < len(self.indep):
            return labrad.units.Unit(self.indep[col_idx][1])
        else:
            return labrad.units.Unit(self.dep[col_idx - len(self.indep)][2])

    def get_column(self, column):
        '''
        Extract a single column by name or index.
        '''
        column_names = [x[0] for x in self.indep] + [x[1] or x[0] for x in self.dep]
        if isinstance(column, (long, int)):
            return self[:,column].view(np.ndarray)
        elif isinstance(column, basestring):
            try:
                idx = column_names.index(column)
            except ValueError:
                raise IndexError('Column (%s) does not exist in array' % column)
            return self[:,idx].view(np.ndarray)

    def column(self, columns):
        '''
        Extract columns by name and return a 2-D array

        ds.column(['time']) or ds.column(['I', 'Q']).

        Numerical indicies can also be freely intermixed with text labels.

        Currently, this returns a bare ndarray, not a DataSetWrapper.  Ideally
        it would return an DataSetWrapper, but update the dep and indep labels
        correctly.  Numpy doesn't seem to have hooks that make that simple,
        so we would have to completely re-implement numpy indexing which seems
        complicated.
        '''
        data = []
        for c in columns:
            data.append(self.get_column(c))
        return np.column_stack(data)

    def __repr__(self):
        header = "LocalDataSet( name='%s',\n" % (self.name)
        with printoptions(precision=5, threshold=100):
            val = header + np.array2string(self) + ")"
        return val

class DataSetWrapper(LocalDataSet):
    '''
    This wraps a dataset retreived from the datavault.  Set LocalDataSet above
    for most of the details, we just fill in the bits about talking to the datavault

    XXX: Currently, we don't subscribe to notifications for updated data or parameters.
         That wouldn't work through the old datavault proxy anyway, but with the new
         multi-head datavault that would prevent stale data.
    '''
    def __new__(cls, data, dvwrapper, indep, dep, parameters, name, number):
        '''
        Delegate to LocalDataSet except for the datavault reference and number
        '''
        obj = super(DataSetWrapper, DataSetWrapper).__new__(DataSetWrapper, data, indep, dep, parameters, name)
        obj.dvw = dvwrapper # dv wrappers are static once created, so this is OK
        obj.number = number
        return obj
    def __array_finalize__(self, obj):
        '''
        See LocalDataSet
        '''
        super(DataSetWrapper, self).__array_finalize__(obj)
        if obj is None: return
        self.dvw = getattr(obj, 'dvw', None)
        self.number = getattr(obj, 'number', None)

    @property
    def comments(self):
        return self.dvw.get_comments(self.name)
    # Parameters are only loaded once, when the data set is loaded.
    # The data vault allows parameters to be changed, but this is considered
    # poor form so it not allowed here.  This is also good for efficiency,
    # as we don't have to talk to the server for every access.
    # @property
    # def parameters(self):
    #     return self.dvw.get_parameters(self.name)
    def comment(self, *comments):
        self.dvw.add_comments(self.name, *comments)
    def tag(self, tag, value=True):
        self.dvw.set_tag(self.name, tag, value)
    def star(self, value=True):
        self.tag('star', value)
    def trash(self, value=True):
        self.tag('trash', value)
    @property
    def path(self):
        return self.dvw._dir
    @property
    def tags(self):
        return self.dvw.get_tags(self.name)
    def __repr__(self):
        header = "DataSetWrapper( path=%s,\n\tname='%s',\n" % (self.path, self.name)
        with printoptions(precision=5, threshold=100):
            val = header + np.array2string(self) + ")"
        return val

def parameter_dict_to_list(p, prefix=''):
    rv = []
    for k,v in p.items:
        if isinstance(v, dict):
            rv.extend(parameter_dict_to_list(v, prefix=prefix+k+'.'))
        else:
            rv.append((prefix+k, v))
    return rv

def parse_parameters(items):
    rv = {}
    if items is None:
        return {}
    def assign_in(subdict, k, v):
        split_key=k.split('.', 1)
        if len(split_key)==1:
            subdict[split_key[0]]=v
        else:
            if split_key[0] not in subdict:
                subdict[split_key[0]]={}
            assign_in(subdict[split_key[0]], split_key[1], v)

    for (key,val) in items:
        assign_in(rv, key, val)
    return rv

class DataVaultWrapper(object):
    '''
    Wrap a data vault directory, allowing direct indexing with [] operator
    to safely fetch a dataset.

    DataVaultWrappers are 'immutable'  -- cd() returns a new instance,
    it never changes the directory of the current instance.  This is needed
    so that datasets can keep a reference to the wrapper to fetch/set tags,
    comments, and properties.
    '''

    def __init__(self, dir_or_registry, cxn=None, ctx=None):
        '''
        Usage: DataVaultWrapper(dir_or_registry, cxn)

        This constructs a wrapper around a specific directory of the datavault.
        You need to provide a connection and the target path.  The path may be
        specified as a list i.e., ['', 'Evan', 'Transmon', ...] or as a
        RegistryWrapper object.
        '''
        if isinstance(dir_or_registry, basestring):
            dir = [dir_or_registry]
        if isinstance(dir_or_registry, (registry.RegistryWrapper, registry2.RegistryWrapper)):
            dir = dir_or_registry._dir
            if cxn is None:
                cxn = dir_or_registry._cxn
        else:
            dir = list(dir_or_registry)
        if cxn is None:
            raise RuntimeError('Connection not specified')

        if '' in dir[1:]:
            raise Exception('Empty string is invalid subdirectory name')
        if dir[0] != '':
            dir = [''] + dir
        if ctx is None:
            ctx = cxn.context()
        srv = cxn.data_vault

        object.__setattr__(self, '_dir', dir)
        object.__setattr__(self, '_cxn', cxn)
        object.__setattr__(self, '_srv', srv)
        object.__setattr__(self, '_ctx', ctx)
        self._packet().cd(dir).send()

    def _packet(self):
        """Create a packet with the correct context and directory."""
        return self._srv.packet(context=self._ctx)

    def _load_file(self, item):
        pkt_return = self._packet().open(item).send()
        item_path, item_name = pkt_return.open
        item_number = int(item_name[0:5])

        data = []
        # There is a problem where the data vault or the data vault proxy
        # crashes if you try to request too big a data file all at once.
        # 50,000 points per packet seems to be safe.
        indep, dep = self._packet().variables().send().variables

        # The limit appears to be around 85000 floats at once.
        # 50000 seems safe, and avoids too much overhead.
        rows_at_once = 50000 // (len(indep)+len(dep))
        if rows_at_once == 0: rows_at_once = 1
        while True:
            new_data = self._packet().get(rows_at_once).send().get
            new_data = np.asarray(new_data)
            if new_data.size:
                data.append(new_data)
            else:
                break
        if data:
            data = np.vstack(data)
        else:
            data=np.empty((0,))
        parameters = self._packet().get_parameters().send().get_parameters
        comments = self._packet().get_comments().send().get_comments

        return DataSetWrapper(data, self, indep, dep, parameters, item_name, item_number)

    def ds_metadata(self, item):
        pkt = self._packet()
        result = pkt.open(item).get_parameters().get_comments().send()
        tag_result = self._packet().get_tags([], result['open'][1]).send()

        param = parse_parameters(result['get_parameters'])
        comments = result['get_comments']
        tags = tag_result['get_tags'][1][0][1]

        return (param, comments, tags)

    def hashkey(self, item):
        k = (tuple(self._dir), item)
        return k

    job_queue = pyle.util.job_queue.JobQueue()
    def _load_file_async(self, item):
        return self.job_queue.run_in_thread(self._load_file, item)

    cache = pyle.util.cache.LRU_Cache(_load_file_async, hashkey, 32)
    def __getitem__(self, item, refresh=False, wait=True):
        '''
        Allows index-like retreival of data sets:

        dvw[2] returns data set 2.  Data sets will be cached globally, so
        subsequent calls don't need to retreive the data again.

        Warning: DataSetWrapper has no way to know if a data set is incomplete.
        If you retreive an incomplete data set (because the acquisition is still
        running) the incomplete data set will be cached.

        If you run into this problem, call __getitem__ explicitly with
        refresh=True.
        '''
        if isinstance(item, (long, int)) and item < 0:
            # This is a bit weird.  Data vault numbers start at one,
            # not zero.  Positive numbers are treated as dataset numbers,
            # but negative numbers are python style indicies.  Zero is invalid.
            item=self.keys()[item]
        if refresh:
            self.cache.expire_item(self, item)
        try:
            ds = self.cache(self, item)
        except labrad.types.Error:
            raise KeyError("Dataset %s doesn't exist (Hint: datasets are numbered from 1, not 0")
        if wait:
            return np.array(ds.get_result(), copy=True, subok=True)

    def prefetch(self, item, refresh=False):
        self.__getitem__(item, refresh=refresh, wait=False)

    def __contains__(self, name):
        '''
        We allow (one based) numbers of strings, following the datavault API.
        '''
        dirs, keys = self._get_list()
        if isinstance(name, (long, int)):
            return name in [int(key[0:5]) for key in keys]
        else:
            return name in keys

    def dir(self, tagFilters=['-trash']):
        '''
        The list of subdirectories.  Note that this is not returned by keys()
        which only returns data sets.
        '''
        dirs, keys = self._packet().dir(tagFilters, False).send().dir
        return dirs

    def cd(self, newdir, *args):
        '''
        Change current working directory.  Does not change the current
        context but instead returns a copy with the new directory.
        The reason is that we want DataVault wrappers to be immutable
        so that DatasetWrappers can keep a handle to them to access
        tags and comments.

        dvw.cd(['dir1, dir2'])
        *or*
        dvw.cd('dir1', 'dir2', ...)
        '''
        if isinstance(newdir, basestring):
            newdir=[newdir] + list(args)

        old_path = list(self._dir)
        newdir = list(newdir)

        # Handle three cases:  new path is absolute, new path is relative,
        # or newpath starts with '..' entries.
        if(newdir and newdir[0] == ''): # absolute path
            path = newdir
        else:
            # data vault doesn't handle '..', so we do it manually
            while(newdir and newdir[0] == '..'):
                del newdir[0]
                del old_path[-1]
            path = old_path + newdir

        return DataVaultWrapper(path, self._cxn)

    def keys(self, tagFilters=['-trash']):
        '''
        The list of data sets only (not subdirectories), so that you can
        enumerate over all data sets in a vault directory
        '''
        dirs, keys = self._packet().dir(tagFilters, False).send().dir
        return list(keys)

    def add_comments(self, item, *comments):
        ''' Wrapper around the datavault method '''
        p=self._packet()
        p.open(item)
        for c in comments:
            p.add_comment(c)
        p.send()
    def set_tag(self, item, tag, value):
        '''
        Note that the data vault only accepts string item names for
        update_tags.  We could implement int->string conversion here,
        but normally this is only implicitly used by the DatasetWrapper
        '''
        p = self._packet()
        tag_code = ("" if value else "-") + tag
        p.update_tags(tag_code, [], item)
        p.send()

    def get_comments(self, item):
        p = self._packet()
        p.open(item)
        p.get_comments()
        rv = p.send()
        return rv.get_comments
    def get_tags(self, item):
        '''
        Note that the data vault only accepts string item names for
        tags.  We could implement int->string conversion here,
        but normally this is only implicitly used by the DatasetWrapper
        '''
        p = self._packet()
        p.get_tags([], item)
        rv = p.send()
        dataset_tags = rv.get_tags[1]
        tags = dataset_tags[0][1]
        #tags = [ t[1] for t in dataset_tags]
        return tags
    def pretty_list(self):
        keys = self.keys()
        maxlen = max([ len(k) for k in keys] )
        for k in keys:
            tags = self.get_tags(k)
            if 'trash' in tags:
                continue
            if 'star' in tags:
                prefix = ' * '
            else:
                prefix = '   '
            padlen = 2 + maxlen - len(k)
            tags = filter(lambda x: x not in ['trash', 'star'], tags)
            if tags:
                tag_str = str(tags)
            else:
                tag_str = ''
            print prefix, k, (" "*padlen), tag_str
    def __iter__(self):
        return iter(self.keys())

    def iteritems(self):
        for k in self.keys():
            yield (k, self[k])

    def itervalues(self):
        for k in self.keys():
            yield self[k]

    def deepiter(self, prefix=[]):
        '''
        Recursive iterator.
        '''
        for d in self.dir():
            for k,v in self[d].deepiter(prefix + [d]):
                yield k,v
        for k in self.keys():
            yield (prefix + [k], self[k])
    def __repr__(self):
        return '<DataVault: %r>' % (self._dir,)


def DataSetCSVLoader(basename=None, **kw):
    '''
    Load a dataset from the CSV/INI files stored on skynet
    basename should be the file path to the .ini or .csv file.
    If basename=None, open a dialog box to select a file.

    Extra keyword arguments will be passed to loadtxt.  The most
    important choices are dtype, but usecols and skiprows may
    also be useful.  They will not fix
    '''
    if basename is None:
        if os.name == 'posix':
            starting_path = os.path.join(os.environ["HOME"], '_LabRAD_data_/')
        else:
            starting_path = 'V:\\_LabRAD_data_\\'
        basename = tkFD.askopenfilename(initialdir=starting_path)

    if basename.lower().endswith('.csv') or basename.lower().endswith('.ini'):
        basename = basename[:-4]
    kw.set_default('ndmin', 2) # Force 2D array
    data = np.loadtxt(basename+'.csv', delimiter=',', **kw)
    cp = ConfigParser.SafeConfigParser()
    cp.read(basename+".ini")
    name = cp.get('General', 'title')
    ndep = int(cp.get('General', 'dependent'))
    nindep = int(cp.get('General', 'independent'))
    nparam = int(cp.get('General', 'parameters'))

    parameters = []
    for x in range(nparam):
        key = cp.get('Parameter %d' % (x+1,), 'label')
        value = labrad.types.evalLRData(cp.get('Parameter %d' % (x+1), 'data'))
        parameters.append((key, value))

    indep = []
    for x in range(nindep):
        units = cp.get('Dependent %d' % (x+1,), 'units')
        label = cp.get('Dependent %d' % (x+1,), 'label')
        indep.append((label, units))

    dep = []
    for x in range(ndep):
        units = cp.get('Dependent %d' % (x+1,), 'units')
        label = cp.get('Dependent %d' % (x+1,), 'label')
        category = cp.get('Dependent %d' % (x+1,), 'category')
        dep.append((category, label, units))

    ds = LocalDataSet(data, indep, dep, parameters, name)
    ds.filename = basename+'.csv'
    return ds
