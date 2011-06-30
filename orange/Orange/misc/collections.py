import bisect
import array
from itertools import izip

class TypedDict(object):
    """ An space efficient dictionary like object with typed keys and
    values and O(log(n)) item lookup.
    
    Example ::
    
        >>> d = TypedDict({1:'a', 2:'b', 3:'c'}, keytype="i", valuetype="c")
        
    """
    __slots__ = ["keytype", "valuetype", "_key_array", "_value_array"]
    def __init__(self, mapping=None, keytype="I", valuetype="B", _key_array=None, _value_array=None):
        """
        :param mapping: If given initialize the TypedDict object from this
            dict like object
        :param keytype: A string type code for keys (see `array` module for
            details)
        :param valuetype: A string type code for values (see `array` module
            for details)
          
        """
        self.keytype = keytype
        self.valuetype = valuetype
        if _key_array is not None and _value_array is not None:
            assert(len(_key_array) == len(_value_array))
            self._key_array = _key_array
            self._value_array = _value_array
        elif mapping:
            items = []
            for key in mapping:
                if isinstance(key, tuple) and len(key) == 2:
                    items.append(key)
                else:
                    items.append((key, mapping[key]))
            items.sort()
            
            self._key_array = array.array(self.keytype, [i[0] for i in items])
            self._value_array = array.array(self.valuetype, [i[1] for i in items])
        else:
            self._key_array = array.array(keytype)
            self._value_array = array.array(valuetype)
        
    def __getitem__(self, key):
        i = bisect.bisect_left(self._key_array, key)
        if i == len(self._key_array):
            raise KeyError(key)
        elif self._key_array[i] == key:
            return self._value_array[i]
        else:
            raise KeyError(key)
        
    def __setitem__(self, key, value):
        i = bisect.bisect_left(self._key_array, key)
        if i == len(self._key_array):
            self._key_array.insert(i, key)
            self._value_array.insert(i, value)
        elif self._key_array[i] == key:
            self._value_array[i] = value
        else:
            self._key_array.insert(i, key)
            self._value_array.insert(i, value)
        
    def keys(self):
        return self._key_array.tolist()
    
    def values(self):
        return self._value_array.tolist()
    
    def items(self):
        return zip(self.iterkeys(), self.itervalues())
    
    def iterkeys(self):
        return iter(self._key_array)
    
    def itervalues(self):
        return iter(self._value_array)
    
    def iteritems(self):
        return izip(self.iterkeys(), self.itervalues())
    
    def get(self, key, default=None):
        i = bisect.bisect_left(self._key_array, key)
        if i == len(self._key_array):
            return default
        elif self._key_array[i] == key:
            return self._value_array[i]
        else:
            return default
        
    def has_key(self, key):
        return self.__contains__(key)
        
    def update(self, mapping):
        raise NotImplementedError

    def __len__(self):
        return len(self._key_array)
    
    def __iter__(self):
        return self.iterkeys()
    
    def __contains__(self, key):
        i = bisect.bisect_left(self._key_array, key)
        if i == len(self._key_array) or self._key_array[i] != key:
            return False
        else:
            return True
    
    def __delitem__(self, key):
        raise NotImplementedError
    
    def clear(self):
        raise NotImplementedError
    
    def todict(self):
        """ Return a regular dict object initialized from this TypedDict.
        """
        return dict(self.iteritems())
    
    def __repr__(self):
        return "TypedDict({0!r})".format(self.todict())
    
    def __reduce_ex__(self, protocol):
        return TypedDict, (), self.__getstate__()
    
    def __getstate__(self):
        return [getattr(self, slot) for slot in self.__slots__]
    
    def __setstate__(self, state):
        for slot, value in zip(self.__slots__, state):
            setattr(self, slot, value)
     