# In the registry directory for a single qubit, we have certain important
# parameters such as pi-pulse amplitude, pi-pulse length, and measure
# amplitude. These parameters are state dependent; the amplitude for a pi
# pulse from the zero to one state will not be the same as the amplitude
# for a pi pulse from the one to two state. We store the parameters for
# different levels as separate keys. For example, the different pi amplitudes
# show up in the registry like this:
#
# piAmp
# piAmp2
# piAmp3
#
# This is nice when you're looking at the registry editor, but it's annoying
# for writing scripts where you want to be able to loop over pi amplitudes
# for different levels. To deal with this problem, you want some kind of list
# that contains all of the pi amplitudes, i.e. [piAmp, piAmp2, piAmp3,...]
# The code we've written here builds this list, and other such lists.
#
# This used to create a list in the qubit dictionary for all multi-level
# parameters.  This is error prone, as the information is now duplicated.
# A scan that tries to change f10 may or may not work.
#
# Instead, we just have a set of helper functions that let you access
# the canonical names.  so getMultiLevels(q, 'frequency', 1) will
# return q['f10'].  If you want to use the old lists, see
# getMultiKeyList.

def multiLevelKeyName(key,state):
    """Create key name for higher state pulses for saving into registry.

    Inputs the registry key base name and the state. Outputs the
    corresponding registry key referring to that state.

    Examples: multiLevelKeyName('piAmp',1) returns 'piAmp'
              multiLevelKeyName('piAmp',3) returns 'piAmp3'
              multiLevelKeyName('frequency', 2) returns 'f21'
    """
    if key == 'frequency':
        newkey = 'f%d%d' % (state, state-1)
    else:
        statenum = str(state) if state>1 else ''
        newkey = key + statenum
    return newkey

# Old and confusing name/alias
saveKeyNumber=multiLevelKeyName

def getMultiLevels(q,key,state):
    """Get registry key for a given state."""
    return q[multiLevelKeyName(key, state)]

def setMultiLevels(q,key,val,state):
    """Set input value in local registry for higher state pulses.

    Inputs the registry key name (piLen or measureAmp), the value
    to be written, and the state for the value.
    """
    q[multiLevelKeyName(key, state)] = val

def setMultiKeys(q,max_state):
    '''Do nothing for compatibility with old code'''
    pass

def getMultiKeyList(q, key):
    '''
    Get the entire list of keys for a given parameter.  For instance,
    getMultiKeyList(q, 'frequency') would return:

    [q['f10'], q['f21'], ...]
    '''
    l = []
    state = 1
    try:
        while True:
            l.append(q[multiLevelKeyName(key, state)])
            state += 1
    except KeyError: # Finished
        return l
