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


svm_destroy_model = _orngCRS.svm_destroy_model

SVMClassifier = _orngCRS.SVMClassifier

SVMLearnS = _orngCRS.SVMLearnS

SVMLearn = _orngCRS.SVMLearn

SVMClassify = _orngCRS.SVMClassify

SVMClassifyP = _orngCRS.SVMClassifyP

SVMClassifyM = _orngCRS.SVMClassifyM

SVMClassifyS = _orngCRS.SVMClassifyS

SVMClassifyPS = _orngCRS.SVMClassifyPS

SVMClassifyMS = _orngCRS.SVMClassifyMS

MCluster = _orngCRS.MCluster

HCluster = _orngCRS.HCluster

DMCluster = _orngCRS.DMCluster

DHCluster = _orngCRS.DHCluster

FCluster = _orngCRS.FCluster

DFCluster = _orngCRS.DFCluster

LogReg = _orngCRS.LogReg

Computer = _orngCRS.Computer

NBprepare = _orngCRS.NBprepare

NBkill = _orngCRS.NBkill

NBcleanup = _orngCRS.NBcleanup

NBquality = _orngCRS.NBquality

TANquality = _orngCRS.TANquality

NBdivergence = _orngCRS.NBdivergence

NBqualityW = _orngCRS.NBqualityW

NBsaveScores = _orngCRS.NBsaveScores

NBrememberScores = _orngCRS.NBrememberScores

NBcompareScores = _orngCRS.NBcompareScores

NBexportScores = _orngCRS.NBexportScores

NBexportProbabilities = _orngCRS.NBexportProbabilities

NBcompareLists = _orngCRS.NBcompareLists

NBstoreModel = _orngCRS.NBstoreModel

NBclassify = _orngCRS.NBclassify

NBclassifyW = _orngCRS.NBclassifyW

NBupdate = _orngCRS.NBupdate

Ksetmodel = _orngCRS.Ksetmodel

Kaddmodel = _orngCRS.Kaddmodel

Ktestaddition = _orngCRS.Ktestaddition

Kdie = _orngCRS.Kdie

Kuse = _orngCRS.Kuse

Kvalidate = _orngCRS.Kvalidate

Kremember = _orngCRS.Kremember

Klearn = _orngCRS.Klearn

KgetDOF = _orngCRS.KgetDOF

Kcheckreversal = _orngCRS.Kcheckreversal

Ksetensemble = _orngCRS.Ksetensemble

Kuseensemble = _orngCRS.Kuseensemble

Ktestmodels = _orngCRS.Ktestmodels


