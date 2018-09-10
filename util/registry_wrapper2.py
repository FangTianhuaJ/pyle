from pyle.registry import AttrDict
from twisted.internet.defer import inlineCallbacks
import labrad.units
import copy
import warnings
import types
import logging
log = logging.getLogger(__name__)
log.setLevel('INFO')

###
#
# This nonsense is needed in order to use python's copy module to copy
# qubit dictionaries.  The default implementation of copy methods
# does not work on classes that have overridden __new__ methods that take
# arguments.  For the cases below, we have an easy fix because we don't
# need (or want, in the case of units) to actually copy the objects.
#
# Ideally, we would fix labrad.units directly, but I don't want to mess
# with pushing those changes yet, so we monkey patch.  We can't use
# use the AttrDict copy() method directly because it has no way to handle
# multiple AttrDicts with internal references to each other.
#
###

def monkey_patch_immutable(cls):
    '''
    Make the target class "immutable" for (deep-)copy operations.
    '''
    def do_copy(self):
        return self
    def do_deepcopy(self, memo):
        return self
    if '__copy__' not in cls.__dict__:
        cls.__copy__ = types.MethodType(do_copy, None, cls)
    if '__deepcopy__' not in cls.__dict__:
        cls.__deepcopy__ = types.MethodType(do_deepcopy, None, cls)

monkey_patch_immutable(labrad.units.Unit)
monkey_patch_immutable(labrad.units.Value)
monkey_patch_immutable(labrad.units.Complex)

def copy_labrad_object(obj):
    '''
    Labrad types don't support normal copying via a copy contructor.  i.e.:
    type(x)(x) won't work.  But we need to copy lists returned by the registry
    wrapper so that mutating them doesn't change the in-memory cache.
    This includes lists nested inside tuples.

    The other labrad types are immutable (I hope!), so we just reference them.

    This should be changed to use copy.deepcopy().  See above monkey patching
    notes
    '''
    if isinstance(obj, list):
        return [copy_labrad_object(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple([copy_labrad_object(x) for x in obj])
    else:
        return obj

class RegistryWrapper(object):
    """Access the labrad registry with a dictionary based interface
    and an auto-updated cache.

    This differs from the original registry wrapper in its use of a
    cache.  This dramatically improves performance, but there is a
    cost.  In order to avoid stale data, we use labrad messages to get
    registry updates.  However, in order to do that, we need to
    maintain an open context in every cached directory.  This prevents
    those directories from being deleted.

    As a compromise: empty directories will never be cached.  Any
    non-empty directory that becomes empty will cause the *entire
    cache* to be invalidated.  The (potentially) recursive delete will
    fail, but a second try will succeed."""
    cache = {}
    notification_ID = 42321

    @classmethod
    def _invalidate_cache(cls):
        '''
        invalidate all cache items
        '''
        for c in cls.cache:
            cls.cache[c]._invalidate()

    def __new__(cls, cxn, path=['']):
        '''
        Figure out if there is already an existing registry wrapper for the
        given directory.
        '''
        key = tuple(path)
        if key in cls.cache:
            return cls.cache[key]
        else:
            obj = object.__new__(cls)
            object.__setattr__(obj, "_cxn", cxn)
            object.__setattr__(obj, "_dir", path)
            object.__setattr__(obj, "_ctx", cxn.context())
            object.__setattr__(obj, "_srv", cxn.registry)
            object.__setattr__(obj, "_valid", False)
            cls.cache[key] = obj
            # We add the listener here once and forever, then turn notification
            # on and off in _invalidate() and initialize()
            if "_backend" in cxn.__dict__:
                cxn._backend.cxn.addListener(obj.update_notification, source=obj._srv.ID, context=obj._ctx, ID=cls.notification_ID)
            else:
                cxn._cxn.addListener(obj.update_notification, source=obj._srv.ID, context=obj._ctx, ID=cls.notification_ID)

            obj.initialize()
            return obj
    def __init__(self, cxn, path=['']):
        '''
        Constructs a new registry wrapper with the specified labrad connection
        and path.  Actually, registry wrappers are singletons, so __new__()nd
        initialize() do the work.  This is here so that ipython shows the
        signature.
        '''
        pass

    def initialize(self, force=False):
        '''
        This is called when creating a RegistryWrapper for a new
        directory, or when the cache has been invalidated.
        '''
        try:
            result = self._packet().cd(self._dir).dir().send()
        except:
            raise KeyError('Registry path %s not found' % self._dir)
        subdirs, keys = result.dir
        if not (force or len(subdirs) or len(keys)):
            # Directory is empty!  Cd to the root directory and stay invalid
            self._packet().cd(['']).send()
            object.__setattr__(self, "_subdirs", set())
            object.__setattr__(self, "_updated_keys", set())
            object.__setattr__(self, "_keys", {})
            #print "Empty directory %s.  Not caching" % self._dir
            log.info("Empty directory %s.  Not caching", self._dir)
            return
        object.__setattr__(self, "_subdirs", set(subdirs))
        object.__setattr__(self, "_updated_keys", set(keys))
        object.__setattr__(self, "_keys", {})
        p = self._packet()
        p.notify_on_change(self.notification_ID, True)
        p.send()
        object.__setattr__(self, '_valid', True)
        self.update_keys()

    def update_keys(self):
        '''
        Handle a batch of updates for this directory, getting all
        new/modified keys in a single packet.  Should only be called
        in the valid state!
        '''
        if not self._updated_keys:
            return
        p = self._packet()
        for k in self._updated_keys:
            p.get(k, key=k)
        result=p.send()
        for k in self._updated_keys:
            self._keys[k] = result[k]
        self._updated_keys.clear()

    def update_notification(self, message_ctx, args):
        '''
        Handle a notification.  Added or removed directories are
        handled immediately, as are deleted keys.  For performance
        reasons, new/modified keys are deferred so they can all be
        retreived in a single update.
        '''
        if message_ctx.ID != self._ctx:
            raise RuntimeError('Got message for wrong context')
        name, isdir, addorchange = args
        if isdir:
            if addorchange:
                self._subdirs.add(name)
            else:
                self._subdirs.discard(name)
        else:
            if addorchange:
                if name in self._keys:
                    del self._keys[name]
                self._updated_keys.add(name)
            else:
                if name in self._keys:
                    del self._keys[name]
                elif name in self._updated_keys:
                    self._updated_keys.remove(name)
                else:
                    #print "warning, unknown key %s removed" % name
                    log.warn("warning, unknown key %s removed" % name)
        if not addorchange:
            if not (self._keys or self._subdirs or self._updated_keys):
                # We invalidate the whole cache here.  If the user is doing a recursive
                # delete it will fail (nothing we can do), but at least if they try
                # a second time it will succeed.
                RegistryWrapper._invalidate_cache()
    def _packet(self):
        """Create a packet with the correct context and directory."""
        return self._srv.packet(context=self._ctx)

    def _setitems(self, dict_like):
        '''
        Batch set items in a single packet for efficiency.
        '''
        self._validate(force=True)
        keys = dict_like.keys()
        p = self._packet()
        for k in keys:
            if isinstance(dict_like[k], (dict, RegistryWrapper)):
                self[k] = dict_like[k]
            else:
                p.set(k, dict_like[k])
        p.send()
    def __getitem__(self, name):
        '''
        Standard dictionary type access.  We take this opportunity to grab any
        new or modified keys we have found out about recently.
        '''
        self._validate()
        self.update_keys()
        if name == '..' and len(self._dir)>1:
            return RegistryWrapper(self._cxn, path=self._dir[:-1])
        if name in self._subdirs:
            return RegistryWrapper(self._cxn, path=self._dir + [name])
        elif name in self._keys:
            v = self._keys[name]
            return copy_labrad_object(v)
        else:
            raise KeyError(name)
    def __setitem__(self, name, value):
        '''
        setitem calls are passed directly to the registry.  We rely on notifications
        to update the cache.
        '''
        if self.__contains__(name):
            print "__setitem__ %s to %s from %s in path %s" % (name, value, self[name], self._dir)
        else:
            print "__setitem__(%s, %s) in path %s" % (name, value, self._dir)
        self._validate(force=True)
        if isinstance(value, (dict, RegistryWrapper)):
            # recursive copy
            if name in self:
                raise KeyError('Attempt to assign dict over existing key or subdirectory')
            p = self._packet()
            p.mkdir(name)
            p.send()
            target = self[name]
            target._setitems(value)
        else:
            p = self._packet()
            p.set(name, value)
            p.send()
    def _del_contents(self):
        '''
        Delete the entire directory contents.  We disable
        notifications first to avoid unnecessary notifications.
        '''
        self._validate()
        for d in self._subdirs:
            self[d]._del_contents()
        p = self._packet()
        p.notify_on_change(self.notification_ID, False)
        for d in self._subdirs:
            p.rmdir(d)
        for k in self._keys:
            p['del'](k)
        for k in self._updated_keys:
            p['del'](k)
        p.cd([''])
        p.send()

        object.__setattr__(self, "_subdirs", set())
        object.__setattr__(self, "_updated_keys", set())
        object.__setattr__(self, "_keys", {})
        object.__setattr__(self, "_valid", False)

    def __delitem__(self, name):
        print "__delitem__(%s) in path %s" % (name, self._dir)

        self._validate()
        if name in self._subdirs:
            # Recursive delete.  This is tricky as we need to make sure to
            # clear cached contexts before removing directories.
            print "RECURSIVE delete of: %s/%s" % (self._dir, name)
            self[name]._del_contents()
            p = self._packet()
            p.rmdir(name)
            p.send()
        elif name in self._keys or name in self._updated_keys:
            p = self._packet()
            p['del'](name)
            p.send()
        else:
            raise KeyError(name)

    # The following 3 methods implement attribute access for the
    # registry wrapper.  This is a terrible idea.  The namespace for
    # registry entries conflicts with the methods and attributes of
    # the wrapper and it makes debugging a giant pain as mispelled
    # attribute access get converted into registry calls 'for free'.
    # It is included here for compatibility with existing AttrDict and
    # RegistryWrapper code.  At some point it will be removed, and you
    # will be sorry.
    def __getattr__(self, name):
        try:
            item = self.__getitem__(name)
            warnings.warn('Attribute access to registry wrappers is deprecated (get attr: %s.)  See comment in registry_wrapper2.py.' % name, DeprecationWarning, stacklevel=2)
            return item
        except KeyError:
            raise AttributeError(name)
    def __setattr__(self, name, value):
        warnings.warn('Attribute access to registry wrappers is deprecated (trying to set attr: %s.)  See comment in registry_wrapper2.py.' % name, DeprecationWarning, stacklevel=2)
        self.__setitem__(name, value)

    def __delattr__(self, name):
        try:
            return self.__delitem__(name)
        except KeyError:
            raise AttributeError(name)

    def copy(self, readonly=False):
        '''
        Make a recursive AttrDict copy.  The copy can be mutated without
        screwing up the registry wrapper.
        '''
        target = AttrDict()
        object.__setattr__(target, '_dir', self._dir)
        object.__setattr__(target, '__name__', self._dir[-1])
        for k in self.keys():
            val = self[k]
            if isinstance(val, (dict, RegistryWrapper)):
                target[k] = val.copy(readonly)
            else:
                target[k] = val
        if readonly:
            target.set_readonly()
        return target
    # These are used by the copy module.  Since RegistryWrappers just
    # point to the underlying registry server, we don't actually do a
    # copy.
    def __copy__(self):
        return self
    def __deepcopy__(self, memo):
        return self
    def _validate(self, force=False):
        '''
        Should be called at the begining of any operations that use the cache.
        If the cache has been invalidated but the directory is still valid,
        refill the cache
        '''
        if not self._valid:
            self.initialize(force)

    def _invalidate(self):
        '''
        Invalidate the cache.  Most importantly, this does a 'cd' back into the
        root directory to allow directories to be deleted.
        '''
        print "Invalidating cache for %s" % self._dir
        p = self._packet()
        p.notify_on_change(self.notification_ID, False)
        p.cd([''])
        rv = p.send(wait=False)  # we are going to ignore the return value of this deferred because
                                 # I can't get the async stuff to work properly -- ERJ
        object.__setattr__(self, "_subdirs", set())
        object.__setattr__(self, "_updated_keys", set())
        object.__setattr__(self, "_keys", {})
        object.__setattr__(self, "_valid", False)

    def __del__(self):
        '''
        Expire the context created for this session.  Currently,
        this should never be called as the global cache holds on
        to every object.  TODO: change cache to use weakrefs.
        '''
        self._invalidate().wait()
        p = self._cxn.manager.packet(context=self._ctx)
        p.expire_context()
        p.send()
    def __contains__(self, name):
        return name in self.keys()

    # Dictionary protocol implementation of keys, values, and items.
    # This may or may not be as useful as you like because it will
    # return subdirectories along side data keys.
    #
    # These lists are always returned in order sorted by keys (unlike
    # regular dicts that have an arbitrary but repeatable order).
    def keys(self):
        self._validate()
        return sorted(list(self._subdirs)) + sorted(list(self._keys) + list(self._updated_keys))
    def values(self):
        keys = self.keys()
        return [self[k] for k in keys]
    def items(self):
        keys = self.keys()
        return [ (k, self[k]) for k in keys ]

    # More dict-like accessors
    def update(self, *other, **kw):
        '''
        Updates registry directory from 'other' (can be a dict,
        registry wrapper, or list of (k,v) pairs, or from keyword
        arguments.  Existing keys are overwritten.
        '''
        for x in other:
            # If x is dict-like, extract its items()
            if hasattr(x, 'items'):
                x = x.items()
            for k,v in x:
                self[k] = v
        for k,v in kw.items():
            self[k] = v

    def setdefault(self, key, default=None):
        '''
        If key is not present, set it to the provided default value.
        In either case, return the value corresponding to key.
        '''
        try:
            return self[key]
        except KeyError:
            self[key] = default
            return default

    def get(self, key, default=None):
        '''
        get with an default value.  Never raises KeyError.
        '''
        try:
            return self[key]
        except KeyError:
            return default

    def dir(self):
        '''
        Returns names of only subdirectories, not registry keys.
        '''
        self._validate()
        return sorted(list(self._subdirs))
    def __iter__(self):
        return iter(self.keys())
    def __repr__(self):
        return '<RegistryWrapper: %r>' % (self._dir,)

def dictDiff(dict1,dict2):
    """ Determines the difference between two dictionaries
        Can be used to compare registry parameters across different datasets
    """

    print "KEYS NOT COMMON BETWEEN DICTS:\n ", list(set(dict1.keys())^ set(dict2.keys()))
    print "KEYS THAT HAVE CHANGED: "
    for key in dict1.keys():
        try:
            if dict1[key]!=dict2[key]:
                print "KEY ", key, " CHANGED FROM", dict1[key], " TO " ,dict2[key]
        except: pass

def regDiff(cxn,path1,path2):
    """ Takes two paths and plugs the regsitry wrapper dictionaries
        for those paths into dictDiff to compare registry parameters
        for different registry directories
        
        example input:
            regDiff(cxn,['','Josh','qubit','LGBI','140526','q0'],['','Josh','qubit','LGBI','140527','q0'])
    """

    dict1 = RegistryWrapper(cxn,path1)
    dict2 = RegistryWrapper(cxn,path2)
    dictDiff(dict1,dict2)
