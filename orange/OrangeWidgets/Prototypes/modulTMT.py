# This file was created automatically by SWIG 1.3.27.
# Don't modify this file, modify the SWIG interface instead.

import _modulTMT

# This file is compatible with both classic and new-style classes.
def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "this"):
        if isinstance(value, class_type):
            self.__dict__[name] = value.this
            if hasattr(value,"thisown"): self.__dict__["thisown"] = value.thisown
            del value.thisown
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static) or hasattr(self,name) or (name == "thisown"):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError,name

import types
try:
    _object = types.ObjectType
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0
del types



tokenize = _modulTMT.tokenize
class Lemmatization(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Lemmatization, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Lemmatization, name)
    def __init__(self): raise RuntimeError, "No constructor defined"
    def __repr__(self):
        return "<%s.%s; proxy of C++ Lemmatization instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    WTStopword = _modulTMT.Lemmatization_WTStopword
    WTIgnoreword = _modulTMT.Lemmatization_WTIgnoreword
    WTNormal = _modulTMT.Lemmatization_WTNormal
    __swig_setmethods__["stopwords"] = _modulTMT.Lemmatization_stopwords_set
    __swig_getmethods__["stopwords"] = _modulTMT.Lemmatization_stopwords_get
    if _newclass:stopwords = property(_modulTMT.Lemmatization_stopwords_get, _modulTMT.Lemmatization_stopwords_set)
    __swig_setmethods__["ignorewords"] = _modulTMT.Lemmatization_ignorewords_set
    __swig_getmethods__["ignorewords"] = _modulTMT.Lemmatization_ignorewords_get
    if _newclass:ignorewords = property(_modulTMT.Lemmatization_ignorewords_get, _modulTMT.Lemmatization_ignorewords_set)
    def __del__(self, destroy=_modulTMT.delete_Lemmatization):
        try:
            if self.thisown: destroy(self)
        except: pass

    def isStopword(*args): return _modulTMT.Lemmatization_isStopword(*args)
    def isIgnoreword(*args): return _modulTMT.Lemmatization_isIgnoreword(*args)
    def containsLemma(*args): return _modulTMT.Lemmatization_containsLemma(*args)
    def containsWordForm(*args): return _modulTMT.Lemmatization_containsWordForm(*args)
    def getWordForm(*args): return _modulTMT.Lemmatization_getWordForm(*args)
    def getLemma(*args): return _modulTMT.Lemmatization_getLemma(*args)
    def getWordForms(*args): return _modulTMT.Lemmatization_getWordForms(*args)
    def getLemmas(*args): return _modulTMT.Lemmatization_getLemmas(*args)
    def getMSDs(*args): return _modulTMT.Lemmatization_getMSDs(*args)
    def getDerNorms(*args): return _modulTMT.Lemmatization_getDerNorms(*args)
    def getInfNorms(*args): return _modulTMT.Lemmatization_getInfNorms(*args)

class LemmatizationPtr(Lemmatization):
    def __init__(self, this):
        _swig_setattr(self, Lemmatization, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, Lemmatization, 'thisown', 0)
        self.__class__ = Lemmatization
_modulTMT.Lemmatization_swigregister(LemmatizationPtr)

class FSALemmatization(Lemmatization):
    __swig_setmethods__ = {}
    for _s in [Lemmatization]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, FSALemmatization, name, value)
    __swig_getmethods__ = {}
    for _s in [Lemmatization]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, FSALemmatization, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ FSALemmatization instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, FSALemmatization, 'this', _modulTMT.new_FSALemmatization(*args))
        _swig_setattr(self, FSALemmatization, 'thisown', 1)
    def __del__(self, destroy=_modulTMT.delete_FSALemmatization):
        try:
            if self.thisown: destroy(self)
        except: pass

    def containsLemma(*args): return _modulTMT.FSALemmatization_containsLemma(*args)
    def containsWordForm(*args): return _modulTMT.FSALemmatization_containsWordForm(*args)
    def getWordForm(*args): return _modulTMT.FSALemmatization_getWordForm(*args)
    def getLemma(*args): return _modulTMT.FSALemmatization_getLemma(*args)
    def getWordForms(*args): return _modulTMT.FSALemmatization_getWordForms(*args)
    def getLemmas(*args): return _modulTMT.FSALemmatization_getLemmas(*args)
    def getMSDs(*args): return _modulTMT.FSALemmatization_getMSDs(*args)
    def getDerNorms(*args): return _modulTMT.FSALemmatization_getDerNorms(*args)
    def getInfNorms(*args): return _modulTMT.FSALemmatization_getInfNorms(*args)

class FSALemmatizationPtr(FSALemmatization):
    def __init__(self, this):
        _swig_setattr(self, FSALemmatization, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, FSALemmatization, 'thisown', 0)
        self.__class__ = FSALemmatization
_modulTMT.FSALemmatization_swigregister(FSALemmatizationPtr)

class NOPLemmatization(Lemmatization):
    __swig_setmethods__ = {}
    for _s in [Lemmatization]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, NOPLemmatization, name, value)
    __swig_getmethods__ = {}
    for _s in [Lemmatization]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, NOPLemmatization, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ NOPLemmatization instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, NOPLemmatization, 'this', _modulTMT.new_NOPLemmatization(*args))
        _swig_setattr(self, NOPLemmatization, 'thisown', 1)
    def __del__(self, destroy=_modulTMT.delete_NOPLemmatization):
        try:
            if self.thisown: destroy(self)
        except: pass


class NOPLemmatizationPtr(NOPLemmatization):
    def __init__(self, this):
        _swig_setattr(self, NOPLemmatization, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, NOPLemmatization, 'thisown', 0)
        self.__class__ = NOPLemmatization
_modulTMT.NOPLemmatization_swigregister(NOPLemmatizationPtr)

class vectorStr(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, vectorStr, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, vectorStr, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ std::vector<std::string > instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def empty(*args): return _modulTMT.vectorStr_empty(*args)
    def size(*args): return _modulTMT.vectorStr_size(*args)
    def clear(*args): return _modulTMT.vectorStr_clear(*args)
    def swap(*args): return _modulTMT.vectorStr_swap(*args)
    def get_allocator(*args): return _modulTMT.vectorStr_get_allocator(*args)
    def pop_back(*args): return _modulTMT.vectorStr_pop_back(*args)
    def __init__(self, *args):
        _swig_setattr(self, vectorStr, 'this', _modulTMT.new_vectorStr(*args))
        _swig_setattr(self, vectorStr, 'thisown', 1)
    def push_back(*args): return _modulTMT.vectorStr_push_back(*args)
    def front(*args): return _modulTMT.vectorStr_front(*args)
    def back(*args): return _modulTMT.vectorStr_back(*args)
    def assign(*args): return _modulTMT.vectorStr_assign(*args)
    def resize(*args): return _modulTMT.vectorStr_resize(*args)
    def reserve(*args): return _modulTMT.vectorStr_reserve(*args)
    def capacity(*args): return _modulTMT.vectorStr_capacity(*args)
    def __nonzero__(*args): return _modulTMT.vectorStr___nonzero__(*args)
    def __len__(*args): return _modulTMT.vectorStr___len__(*args)
    def pop(*args): return _modulTMT.vectorStr_pop(*args)
    def __getslice__(*args): return _modulTMT.vectorStr___getslice__(*args)
    def __setslice__(*args): return _modulTMT.vectorStr___setslice__(*args)
    def __delslice__(*args): return _modulTMT.vectorStr___delslice__(*args)
    def __delitem__(*args): return _modulTMT.vectorStr___delitem__(*args)
    def __getitem__(*args): return _modulTMT.vectorStr___getitem__(*args)
    def __setitem__(*args): return _modulTMT.vectorStr___setitem__(*args)
    def append(*args): return _modulTMT.vectorStr_append(*args)
    def __del__(self, destroy=_modulTMT.delete_vectorStr):
        try:
            if self.thisown: destroy(self)
        except: pass


class vectorStrPtr(vectorStr):
    def __init__(self, this):
        _swig_setattr(self, vectorStr, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, vectorStr, 'thisown', 0)
        self.__class__ = vectorStr
_modulTMT.vectorStr_swigregister(vectorStrPtr)

class setStr(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, setStr, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, setStr, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ std::set<std::string > instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, setStr, 'this', _modulTMT.new_setStr(*args))
        _swig_setattr(self, setStr, 'thisown', 1)
    def empty(*args): return _modulTMT.setStr_empty(*args)
    def size(*args): return _modulTMT.setStr_size(*args)
    def clear(*args): return _modulTMT.setStr_clear(*args)
    def swap(*args): return _modulTMT.setStr_swap(*args)
    def get_allocator(*args): return _modulTMT.setStr_get_allocator(*args)
    def erase(*args): return _modulTMT.setStr_erase(*args)
    def count(*args): return _modulTMT.setStr_count(*args)
    def __nonzero__(*args): return _modulTMT.setStr___nonzero__(*args)
    def __len__(*args): return _modulTMT.setStr___len__(*args)
    def append(*args): return _modulTMT.setStr_append(*args)
    def __contains__(*args): return _modulTMT.setStr___contains__(*args)
    def __getitem__(*args): return _modulTMT.setStr___getitem__(*args)
    def __del__(self, destroy=_modulTMT.delete_setStr):
        try:
            if self.thisown: destroy(self)
        except: pass


class setStrPtr(setStr):
    def __init__(self, this):
        _swig_setattr(self, setStr, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, setStr, 'thisown', 0)
        self.__class__ = setStr
_modulTMT.setStr_swigregister(setStrPtr)



