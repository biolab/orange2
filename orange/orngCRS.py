# This file was created automatically by SWIG.
# Don't modify this file, modify the SWIG interface instead.
# This file is compatible with both classic and new-style classes.
import _orngCRS
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


MCluster = _orngCRS.MCluster

HCluster = _orngCRS.HCluster

DMCluster = _orngCRS.DMCluster

DHCluster = _orngCRS.DHCluster

FCluster = _orngCRS.FCluster

DFCluster = _orngCRS.DFCluster

LogReg = _orngCRS.LogReg

svm_destroy_model = _orngCRS.svm_destroy_model

SVMClassifier = _orngCRS.SVMClassifier

SVMLearn = _orngCRS.SVMLearn

SVMClassify = _orngCRS.SVMClassify

SVMClassifyM = _orngCRS.SVMClassifyM


