# This file was created automatically by SWIG.
# Don't modify this file, modify the SWIG interface instead.
# This file is compatible with both classic and new-style classes.
import _statisticsc
def _swig_setattr(self,class_type,name,value):
    if (name == "this"):
        if isinstance(value, class_type):
            self.__dict__[name] = value.this
            if hasattr(value,"thisown"): self.__dict__["thisown"] = value.thisown
            del value.thisown
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    self.__dict__[name] = value

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


ks2 = _statisticsc.ks2

pks2 = _statisticsc.pks2

lngamma = _statisticsc.lngamma

log_nCr = _statisticsc.log_nCr

ks1 = _statisticsc.ks1

ks2_asympt = _statisticsc.ks2_asympt

alnorm = _statisticsc.alnorm

gammad = _statisticsc.gammad

PPND = _statisticsc.PPND

POLY = _statisticsc.POLY

chi_squared = _statisticsc.chi_squared

cvar = _statisticsc.cvar
EPSILON = cvar.EPSILON
INF = cvar.INF

