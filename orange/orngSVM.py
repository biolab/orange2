# ORANGE Support Vector Machines
# This module was written by Ales Erjavec
# and supersedes an earlier one written by Alex Jakulin (jakulin@acm.org),
# based on: Chih-Chung Chang and Chih-Jen Lin's
# LIBSVM : a library for support vector machines
#  (http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.ps.gz)


import orange, orngTest, orngStat, sys

try:
    import orngSVM_Jakulin
    BasicSVMLearner=orngSVM_Jakulin.BasicSVMLearner
    BasicSVMClassifier=orngSVM_Jakulin.BasicSVMClassifier
except:
    pass

def SVMLearner(examples=None, weightID=0, **kwds):
    l=apply(SVMLearnerClass, (), kwds)
    if examples:
        l=l(examples)
    return l

class SVMLearnerClass:
    def __init__(self, **kwds):
        self.learner=orange.SVMLearner()
        self.svm_type=0
        self.kernel_type=2
        self.kernelFunc=None
        self.C=1.0
        self.nu=0.4
        self.p=0.1
        self.gamma=0.2
        self.degree=3
        self.coef0=0
        self.shrinking=1
        self.probability=0
        self.cache_size=100
        self.eps=0.001
        self.__dict__.update(kwds)
        self.learner=orange.SVMLearner(**kwds)

    def __setattr__(self, name, value):
        if name in ["svm_type", "kernel_type", "kernelFunc", "C", "nu", "p", "gamma", "degree",
                    "coef0", "shrinking", "probability", "cache_size", "eps"]:
            self.learner.__dict__[name]=value
        self.__dict__[name]=value

    def __call__(self, examples, weight=0):
        if self.svm_type in [0,1] and examples.domain.classVar.varType!=orange.VarTypes.Discrete:
            raise AttributeError, "Cannot learn a discrete classifier from non descrete class data. Use EPSILON_SVR or NU_SVR for regression"
        if self.svm_type in [3,4] and examples.domain.classVar.varType==orange.VarTypes.Discrete:
            raise AttributeError, "Cannot do regression on descrete class data. Use C_SVC or NU_SVC for classification"
        if self.kernel_type==4 and not self.kernelFunc:
            raise AttributeError, "Custom kernel function not supplied"
        ##################################################
        if self.kernel_type==4:     #There is a bug in svm
            self.probability=True
        ##################################################

        for name in ["svm_type", "kernel_type", "kernelFunc", "C", "nu", "p", "gamma", "degree",
                "coef0", "shrinking", "probability", "cache_size", "eps"]:
            self.learner.__dict__[name]=getattr(self, name)
        return self.learner(examples)


def parameter_selection(learner, data, folds=4, parameters={}, best={}, callback=None):
    """parameter selection tool: uses cross validation to find the optimal parameters.
    parameters argument is a dictionary containing ranges for parameters
    return value is a dictionary with optimal parameters and error
    the callback function takes two arguments, a 0.0-1.0 float(progress), and the current best parameters
    >>>params=parameter_selection(learner, data, 10, {"C":range(1,10,2), "gama":range(0.5,2.0,0.25)})"""
    global steps, curStep
    steps=1
    for c in parameters.values():
        steps*=len(c)
    curStep=1
    def mysetattr(obj, name, value):
        names=name.split(".")
        for name in names[:-1]:
            obj=getattr(obj, name)
        setattr(obj, name, value)
        
    def search(learner, data, folds, keys, ranges, current, best={}, callback=None):
        global steps, curStep
        if len(keys)==1:
            for p in ranges[0]:
                mysetattr(learner, keys[0], p)
                current[keys[0]]=p
                te=orngTest.crossValidation([learner], data, folds)
                if data.domain.classVar.varType==orange.VarTypes.Discrete:
                    [res]=orngStat.CA(te)
                    res=1-res
                else:
                    [res]=orngStat.MSE(te)
                if res<best["error"]:
                    best.update(current)
                    best["error"]=res
                curStep+=1
                if callback:
                    callback(curStep/float(steps), best)
        else:
            for p in ranges[0]:
                mysetattr(learner, keys[0], p)
                current[keys[0]]=p
                search(learner, data, folds, keys[1:], ranges[1:], current, best, callback)
                
    keys=parameters.keys()
    ranges=[parameters[key] for key in keys]
    best["error"]=sys.maxint
    current={}
    for key in keys:
        best[key]=parameters[key][0]
        current[key]=parameters[key][0]
    search(learner, data, folds, keys, ranges, current, best, callback)
    return best

import math
class KernelWrapper:
    def __init__(self, wrapped):
        self.wrapped=wrapped
    def __call__(self, example1, example2):
        return self.wrapped(example1, example2)

class DualKernelWrapper(KernelWrapper):
    def __init__(self, wrapped1, wrapped2):
        self.wrapped1=wrapped1
        self.wrapped2=wrapped2
        
class RBFKernelWrapper(KernelWrapper):
    gamma=0.5
    def __call__(self, example1, example2):
        return math.exp(-math.pow(self.wrapped(example1, example2),2)/self.gamma)

class PolyKernelWrapper(KernelWrapper):
    degree=3.0
    def __call__(self, example1, example2):
        return math.pow(self.wrapped(example1, example2), self.degree)

class AdditionKernelWrapper(DualKernelWrapper):
    def __call__(self, example1, example2):
        return self.wrapped1(example1, example2)+self.wrapped2(example1, example2)

class MultiplicationKernelWrapper(DualKernelWrapper):
    def __call__(self, example1, example2):
        return self.wrapped1(example1, example2)*self.wrapped2(example1, example2)
