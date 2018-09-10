import numpy as np
from labrad.types import Value
from labrad.units import Unit


class AttrDict(dict):
    """Subclass of dict with attribute style access. Allows for simpler access
    in interactive sessions."""
    def __init__(self,initDict={}):
        dict.__init__(self,initDict)

    def __repr__(self):
        string =''
        for k in self.keys():
            string = string + str(k) +'\n'
        return string

    def show(self):
        for k in self.keys():
            print k, self[k]

    def __setitem__(self,key,value):
        return super(AttrDict, self).__setitem__(key,value)

    def __getitem__(self,name):
        return super(AttrDict, self).__getitem__(name)

    __getattr__ = __getitem__
    __setattr__ = __setitem__
        

        
class ValueArray(object):
    #TODO:
        # Make sure iteration in loops works
        # Test behavior formally
        
    def __init__(self,dataIn,unit):
        #Copy the input data to avoid problems with mutability
        #and cast to numpy array
        data = np.array(dataIn[:])
        self.data=data
        if isinstance(unit,Unit):
            pass
        elif isinstance(unit,str):
            unit = Unit(unit)
        else:
            raise Exception('Units must be specified as str or labrad.units.Unit')
        self.unit = unit

    def __repr__(self):
        dat=''
        for d in self.data:
            dat+= str(d)+','
        s = "ValueArray([%s],Unit('%s'))" %(dat[:-1], str(self.unit)) #chop off final comma in data list
        return s
    
    def value(self):
        return self.data
        
    def units(self):
        return self.unit.name
        
    def inUnitsOf(self,targetUnit):
        scale = Value(1.0,self.unit).inUnitsOf(targetUnit)
        return self.data[:]*scale

    def inBaseUnits(self):
        raise Exception('Not implemented')
        
    def __iter__(self):
        return iter([Value(x,self.unit) for x in self.data])
        
    def __getitem__(self,key):
        if isinstance(key,int):
            return Value(self.data[key],self.unit)
        elif isinstance(key,str) or isinstance(key,Unit):
            return self.inUnitsOf(key)
        elif isinstance(key,slice):
            return ValueArray(self.data[key],self.unit)

    def __add__(self,other):
        if not isinstance(other,ValueArray):
            raise Exception('Can only add ValueArray to another ValueArray')
        if self.unit.isCompatible(other.unit):
            data = self[self.unit]+other[self.unit]
            result = ValueArray(data,self.unit)
        else:
            raise Exception('Cannot add array with unit %s to array with unit %s' %(str(other.unit),str(self.unit)))
        return result
            
    def __mul__(self,other):
        if isinstance(other,ValueArray) or isinstance(other,Value):
            return ValueArray(self.value()*other.value(), self.unit*other.unit)
        else:
            return ValueArray(self.data*other,self.unit)
            
    def __div__(self,other):
        if isinstance(other,ValueArray) or isinstance(other,Value):
            result = ValueArray(self.value()/other.value(),self.unit/other.unit)
        else:
            result = ValueArray(self.value()/other, self.unit)
        return result
