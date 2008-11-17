# ORANGE Support Vector Machines
# This module was written by Ales Erjavec
# and supersedes an earlier one written by Alex Jakulin (jakulin@acm.org),
# based on: Chih-Chung Chang and Chih-Jen Lin's
# LIBSVM : a library for support vector machines
#  (http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.ps.gz)


import orange, orngTest, orngStat, sys, math

try:
    import orngSVM_Jakulin
    BasicSVMLearner=orngSVM_Jakulin.BasicSVMLearner
    BasicSVMClassifier=orngSVM_Jakulin.BasicSVMClassifier
except:
    pass

class SVMLearner(orange.SVMLearner):
    def __new__(cls, examples=None, weightID=0, **kwargs):
        self = orange.SVMLearner.__new__(cls, **kwargs)
        if examples:
            self.__init__(**kwargs)
            return self.__call__(examples, weightID)
        else:
            return self
        
    def __init__(self, **kwargs):
        self.svm_type=orange.SVMLearner.Nu_SVC
        self.kernel_type=2
        self.kernelFunc=None
        self.C=1.0
        self.nu=0.5
        self.p=0.1
        self.gamma=0.0
        self.degree=3
        self.coef0=0
        self.shrinking=1
        self.probability=1
        self.cache_size=100
        self.eps=0.001
        self.normalization = True
        self.__dict__.update(kwargs)
        self.learner=orange.SVMLearner(**kwargs)

    def __call__(self, examples, weight=0):
        if self.svm_type in [0,1] and examples.domain.classVar.varType!=orange.VarTypes.Discrete:
            self.svm_type+=3
            #raise AttributeError, "Cannot learn a discrete classifier from non descrete class data. Use EPSILON_SVR or NU_SVR for regression"
        if self.svm_type in [3,4] and examples.domain.classVar.varType==orange.VarTypes.Discrete:
            self.svm_type-=3
            #raise AttributeError, "Cannot do regression on descrete class data. Use C_SVC or NU_SVC for classification"
        if self.kernel_type==4 and not self.kernelFunc:
            raise AttributeError, "Custom kernel function not supplied"
        ##################################################
        if self.kernel_type==4:     #There is a bug in svm. For some unknown reason only the probability model works with custom kernels
            self.probability=True
        ##################################################
        if self.svm_type == orange.SVMLearner.Nu_SVC: #check nu feasibility
            dist = list(orange.Distribution(examples.domain.classVar, examples))
            def pairs(seq):
                for i, n1 in enumerate(seq):
                    for n2 in seq[i+1:]:
                        yield n1, n2
            maxNu = min(2.0 * min(n1, n2) / (n1 + n2) for n1, n2 in pairs(dist))
            if self.nu > maxNu:
                import warnings
                warnings.warn("Specified nu %.3f is infeasible. Setting nu to %.3f" % (self.nu, maxNu))
                self.nu = maxNu
            
        for name in ["svm_type", "kernel_type", "kernelFunc", "C", "nu", "p", "gamma", "degree",
                "coef0", "shrinking", "probability", "cache_size", "eps"]:
            self.learner.__dict__[name]=getattr(self, name)
        return self.learnClassifier(examples)

    def learnClassifier(self, examples):
        if self.normalization:
            dc = orange.DomainContinuizer()
            dc.classTreatment = orange.DomainContinuizer.Ignore
            dc.continuousTreatment = orange.DomainContinuizer.NormalizeBySpan
            dc.multinomialTreatment = orange.DomainContinuizer.NValues
            newdomain = dc(examples)
            examples = examples.translate(newdomain)
            return SVMClassifier(self.learner(examples), newdomain)
        return self.learner(examples)

class SVMClassifier(object):
    def __init__(self, classifier, domain):
        self.classifier = classifier
        self.domain = domain

    def __getattr__(self, name):
        try:
            return getattr(self.classifier, name)
        except AttributeError:
            raise AttributeError(name)

    def __call__(self, example, what=orange.GetValue):
        example = orange.Example(self.domain, example)
        return self.classifier(example, what)
        
class SVMLearnerSparse(SVMLearner):
    def __init__(self, **kwds):
        SVMLearner.__init__(self, **kwds)
        self.learner=orange.SVMLearnerSparse(**kwds)

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

    
class SVMLearnerEasy(SVMLearner):
    def __init__(self, **kwds):
        self.folds=5
        SVMLearner.__init__(self, **kwds)
        
    def learnClassifier(self, examples):
        transformer=orange.DomainContinuizer()
        transformer.multinominalTreatment=orange.DomainContinuizer.NValues
        transformer.continuousTreatment=orange.DomainContinuizer.NormalizeBySpan
        transformer.classTreatment=orange.DomainContinuizer.Ignore
        newdomain=transformer(examples)
        newexamples=examples.translate(newdomain)
        #print newexamples[0]
        params={}
        if self.svm_type in [1,4]:
            numOfNuValues=9
            params["nu"]=map(lambda a: float(a)/numOfNuValues, range(1,1+numOfNuValues))
        else:
            params["C"]=map(lambda a: 2**a, range(-5,15,2))
        if self.kernel_type==2:
            params["gamma"]=map(lambda a: 2**a, range(-3,15,2))+[0]
        best=parameter_selection(self.learner, newexamples, self.folds, params)
        #print best["error"]
        del best["error"]
        for name, val in best.items():
            setattr(self.learner, name, val)
        return SVMClassifierClassEasyWrapper(self.learner(newexamples), newdomain, examples)

class SVMLearnerSparseClassEasy(SVMLearnerEasy, SVMLearnerSparse):
    def __init__(self, **kwds):
        SVMLearnerSparse.__init__(self, **kwds)
        
class SVMClassifierClassEasyWrapper:
    def __init__(self, classifier, domain=None, oldexamples=None):
        self.classifier=classifier
        self.domain=domain
        self.oldexamples=oldexamples
    def __call__(self,example, getBoth=orange.GetValue):
        example=orange.ExampleTable([example]).translate(self.domain)[0] #orange.Example(self.domain, example)
        return self.classifier(example, getBoth)
    def __getattr__(self, name):
        if name in ["supportVectors", "nSV", "coef", "rho", "examples", "kernelFunc"]:
            return getattr(self.__dict__["classifier"], name)
        else:
            raise AttributeError(name)
    def __setstate__(self, state):
        print state
        self.__dict__.update(state)
        transformer=orange.DomainContinuizer()
        transformer.multinominalTreatment=orange.DomainContinuizer.NValues
        transformer.continuousTreatment=orange.DomainContinuizer.NormalizeBySpan
        transformer.classTreatment=orange.DomainContinuizer.Ignore
        print self.examples
        self.domain=transformer(self.oldexamples)

def getLinearSVMWeights(classifier):
    """returns list of weights for linear class vs. class classifiers for the linear multiclass svm classifier. The list is in the order of 1vs2, 1vs3 ... 1vsN, 2vs3 ..."""
    def updateWeights(w, key, val, mul):
        if key in w:
            w[key]+=mul*val
        else:
            w[key]=mul*val
            
    SVs=classifier.supportVectors
    weights=[]
    classes=classifier.supportVectors.domain.classVar.values
    classSV=dict([(value, filter(lambda sv: sv.getclass()==value, classifier.supportVectors)) for value in classes])
    svRanges=[(0, classifier.nSV[0])]
    for n in classifier.nSV[1:]:
        svRanges.append((svRanges[-1][1], svRanges[-1][1]+n))
    for i in range(len(classes)-1):
        for j in range(i+1, len(classes)):
            w={}
            coefInd=j-1
            for svInd in apply(range, svRanges[i]):
                for attr in SVs.domain.attributes+SVs[svInd].getmetas(False, orange.Variable).keys():
                    if attr.varType==orange.VarTypes.Continuous:
                        updateWeights(w, attr, float(SVs[svInd][attr]), classifier.coef[coefInd][svInd])
            coefInd=i
            for svInd in apply(range, svRanges[j]):
                for attr in SVs.domain.attributes+SVs[svInd].getmetas(False, orange.Variable).keys():
                    if attr.varType==orange.VarTypes.Continuous:
                        updateWeights(w, attr, float(SVs[svInd][attr]), classifier.coef[coefInd][svInd])
            weights.append(w)
    return weights

def exampleWeightedSum(example, weights):
    sum=0
    for attr, w in weights.items():
        sum+=float(example[attr])*w
    return sum

import math
class KernelWrapper(object):
    def __init__(self, wrapped):
        self.wrapped=wrapped
    def __call__(self, example1, example2):
        return self.wrapped(example1, example2)

class DualKernelWrapper(KernelWrapper):
    def __init__(self, wrapped1, wrapped2):
        self.wrapped1=wrapped1
        self.wrapped2=wrapped2
        
class RBFKernelWrapper(KernelWrapper):
    def __init__(self, wrapped, gamma=0.5):
        KernelWrapper.__init__(self, wrapped)
        self.gamma=gamma
    def __call__(self, example1, example2):
        return math.exp(-math.pow(self.wrapped(example1, example2),2)/self.gamma)

class PolyKernelWrapper(KernelWrapper):
    def __init__(self, wrapped, degree=3.0):
        KernelWrapper.__init__(self, wrapped)
        self.degree=degree
    def __call__(self, example1, example2):
        return math.pow(self.wrapped(example1, example2), self.degree)

class AdditionKernelWrapper(DualKernelWrapper):
    def __call__(self, example1, example2):
        return self.wrapped1(example1, example2)+self.wrapped2(example1, example2)

class MultiplicationKernelWrapper(DualKernelWrapper):
    def __call__(self, example1, example2):
        return self.wrapped1(example1, example2)*self.wrapped2(example1, example2)

class CompositeKernelWrapper(DualKernelWrapper):
    def __init__(self, wrapped1, wrapped2, l=0.5):
        DualKernelWrapper.__init__(self, wrapped1, wrapped2)
        self.l=l
    def __call__(self, example1, example2):
        return self.l*self.wrapped1(example1, example2) + (1-self.l)*self.wrapped2(example1,example2)

class SparseLinKernel(object):
    """Computes a linear kernel function using the examples meta attributes (need to be floats)"""
    def __call__(self, example1, example2):
        s=set(example1.getmetas().keys()+example2.getmetas().keys())
        sum=0
        getmeta=lambda e: e.hasmeta(key) and float(e[key]) or 0.0
        for key in s:
            sum+=pow(getmeta(example2)-getmeta(example1), 2)
        return pow(sum, 0.5)

class BagOfWords(object):
    """Computes a BOW kernel function (sum_i(example1[i]*example2[i])) using the examples meta attributes (need to be floats)"""
    def __call__(self, example1, example2):
        s=Set(example1.getmetas().keys()).intersection(Set(example2.getmetas().keys()))
        sum=0
        for key in s:
            sum+=float(example2[key])*float(example1[key])
        return sum

class RFE(object):
    def __init__(self, learner=None):
        self.learner = learner or SVMLearner(kernel_type=orange.SVMLearner.Linear)

    def getAttrScores(self, data, stopAt=0):
        iter = 1
        attrs = data.domain.attributes
        attrScores = {}
        while len(attrs)>stopAt:
            weights = getLinearSVMWeights(self.learner(data))
            print iter, "Remaining:", len(attrs)
            score = dict.fromkeys(attrs, 0)
            for w in weights:
                for attr, wAttr in w.items():
                    score[attr] += wAttr**2
            score = score.items()
            score.sort(lambda a,b:cmp(a[1],b[1]))
            numToRemove = max(int(len(attrs)*1.0/(iter+1)), 1)
            for attr, s in  score[:numToRemove]:
                attrScores[attr] = len(attrScores)
            attrs = [attr for attr, s in score[numToRemove:]]
            if attrs:
                data = data.select(attrs + [data.domain.classVar])
            iter += 1
        return attrScores
        
    def __call__(self, data, numSelected=20):
        scores = self.getAttrScores(data)
        scores = scores.items()
        scores.sort(lambda a,b:cmp(a[1],b[1]))
        scores = dict(scores[-numSelected:])
        attrs = [attr for attr in data.domain.attributes if attr in scores]
        data = data.select(attrs + [data.domain.classVar])
        return data

def exampleTableToSVMFormat(examples, file):
    attrs = examples.domain.attributes + examples.domain.getmetas().values()
    attrs = [attr for attr in attrs if attr.varType in [orange.VarTypes.Continuous, orange.VarTypes.Discrete]]
    cv = examples.domain.classVar
    
    for ex in examples:
        file.write(str(int(ex[cv])))
        for i, attr in enumerate(attrs):
            if not ex[attr].isSpecial():
                file.write(" "+str(i+1)+":"+str(ex[attr]))
        file.write("\n")
            