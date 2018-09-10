#
# This module implements a LFU cache.  There are many modules like it,
# but this one is mine.
#
# The basic idea of a cache is that you have some function f(x) that is
# expensive (in CPU or IO) to compute, and for which we often evaluate for
# the same x many times.  If f(x) depends only on x and has no side effects,
# you can just remember (memoize) the result after the first call, and subsequent
# calls just return the cache.
#
# There are a couple of problems with that.  First, a generic problem: caches
# need a mechanism to discard entries or they grow without bound.  In our
# case, the cache values may be relatively large arrays: 100 KB - 1 MB.  If
# we store 1000 entries, that can take up to 1 GB of memory.  So real caches
# eventually need a mechanism to discard entries.  The most common algorithms
# are least-frequently-used and least-recently-used.  When the cache reaches
# a fixed size, the indicated elements are discarded.  Often it is computationally
# easier to find some approximation of the least-recently-used element.
#
# The second problem is that in our case 'x' itself may be quite large (a numpy
# array of similar size to f(x).  Locating an item x by value may be excessively
# expensive.  numpy arrays are not hashable, and not immutable, complicating
# matters further.  Therefore, it is the clients responsibility to calculate a
# unique key that represents the data.  A common pattern will be to require that
# x be an array of equally spaced elements, which can be uniquely described by the
# the first and last elements, and the length.

import numpy as np
import collections
import itertools
import traceback
import threading

class fast_str(str):
    '''
    String class with a hash function that is fast for very long
    strings.  Note that str(a) and fast_str(a) will compare equal but
    have different hash values.  Do not use unless you understand the
    ramifications of that.
    '''
    def __hash__(self):
        if len(self) > 64:
            return str.__hash__(self[0:32:2] + self[-32::2])
        else:
            return str.__hash__(self)

def keyfunc_ndarray(a):
    '''
    Key function for a numpy array.  Uses the string representation of the
    raw data, and uses the fast_str hash function for maximum performance.
    '''
    a = np.asarray(a)
    shape = a.shape
    data = a.tostring()
    return (a.shape, a.dtype, fast_str(a.tostring()))

class LRU_Cache(object):
    '''
    LRU cache implementation for memoizing functions.  It takes a keyfunc
    parameter which is a function that turns an argument into a hashable
    key.  keyfunc should raise TypeError if the input argument is unsuitable
    for caching (in which case it will just pass-through with no caching)

    This is adapted from the version found at
    http://code.activestate.com/recipes/498245-lru-and-lfu-cache-decorators/
    but uses a specified keyfunc and works if the arguments are not
    directly hashable.
    '''
    def __init__(self, func, keyfunc=None, N=512):
        self.N=N
        self.func = func
        self.keyfunc = keyfunc or (lambda x: x)
        self.cache = {}
        self.refcount = {}
        self.order = collections.deque()
        self.miss = 0
        self.hit = 0
        self.uncacheable = 0
        self.hash_failure = 0

    def __call__(self, *args, **kwargs):
        '''
        Return a cached value or invoke the wrapped function
        '''
        try:
            key = self.keyfunc(*args, **kwargs)
        except TypeError as e: # Uncachable argument.
            self.uncacheable += 1
            return self.func(*args, **kwargs)

        try:
            return self.get_from_cache(key)
        except KeyError: # not in cache.  Add it.
            return self.add_to_cache(key, args, kwargs)
        except TypeError: # unhashable value returned by keyfunc.
            print "warning, keyfunc returned unhashable value"
            return self.func(*args, **kwargs)

    def get_from_cache(self, key):
        '''
        Get an item from the cache.
        '''
        val = self.cache[key]
        self.hit += 1
        self.order.append(key)
        self.refcount[key] += 1
        if len(self.order) > self.N*4:
            self.cleanup_lru()
        return self.cache[key]

    def add_to_cache(self, key, args, kwargs):
        '''
        Evaluate the underlying function and add it to the cache
        '''
        val = self.func(*args, **kwargs)
        self.miss += 1
        self.cache[key] = val
        self.order.append(key)
        self.refcount[key] = 1
        if len(self.cache) > self.N:
            self.expire_cache()
        return val

    def cleanup_lru(self):
        '''
        Removes duplicate entries from the LRU list.  We traverse the
        list oldest to newest, dropping entires with a refcount less
        than one.  That leaves only the most recent access in the queue.
        '''
        new_order = collections.deque()
        for k in self.order:
            if self.refcount[k] > 1:
                self.refcount[k] -= 1
            else:
                new_order.append(k)
        self.order = new_order

    def expire_item(self, *args, **kwargs):
        '''
        Remove a specific item from the cache.  Note this is O(N).
        '''
        try:
            key = self.keyfunc(*args, **kwargs)
        except TypeError: # Uncachable, don't bother
            return
        try:
            del self.cache[key]
            del self.refcount[key]
        except (KeyError, TypeError): # item not in cache or unhashable
            return
        it = itertools.ifilter(lambda x: x!=key, self.order)
        self.order = collections.deque(it)

    def clear_cache(self):
        '''
        clears the entire cache
        '''
        self.cache.clear()
        self.order = collections.deque()
        self.refcount = {}

    def expire_cache(self):
        '''
        Expire one item from the cache.  Traverse the LRU list oldest
        to newest, dropping elements until we find one with a refcount
        of 1, which is then deleted from the cache.
        '''
        while True:
            k = self.order.popleft()
            self.refcount[k] -= 1
            if self.refcount[k] == 0:
                del self.refcount[k]
                del self.cache[k]
                return

