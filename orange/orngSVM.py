# ORANGE Support Vector Machines
# This module was written by Ales Erjavec
# and supersedes an earlier one written by Alex Jakulin (jakulin@acm.org),
# based on: Chih-Chung Chang and Chih-Jen Lin's
# LIBSVM : a library for support vector machines
#  (http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.ps.gz)


import orange, orngTest, orngStat, orngWrap, sys, math

try:
    import orngSVM_Jakulin
    BasicSVMLearner=orngSVM_Jakulin.BasicSVMLearner
    BasicSVMClassifier=orngSVM_Jakulin.BasicSVMClassifier
except:
    pass

def maxNu(examples):
    """ Given example table compute the maximum nu parameter for Nu_SVC
    """
    nu = 1.0
    dist = list(orange.Distribution(examples.domain.classVar, examples))
    def pairs(seq):
        for i, n1 in enumerate(seq):
            for n2 in seq[i+1:]:
                yield n1, n2
    return min([2.0 * min(n1, n2) / (n1 + n2) for n1, n2 in pairs(dist) if n1 != 0 and n2 !=0] + [nu])

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
        self.verbose = False
        self.cache_size=100
        self.eps=0.001
        self.normalization = True
        self.__dict__.update(kwargs)
        self.learner=orange.SVMLearner(**kwargs)
        self.weight = []

    maxNu = staticmethod(maxNu)

    def __call__(self, examples, weight=0):
        examples = orange.Preprocessor_dropMissingClasses(examples)
        if len(examples) == 0:
            raise ValueError("Example table is without any defined classes")
        print examples, len(examples), len(examples.domain)
        if self.svm_type in [0,1] and examples.domain.classVar.varType!=orange.VarTypes.Discrete:
            self.svm_type+=3
            #raise AttributeError, "Cannot learn a discrete classifier from non descrete class data. Use EPSILON_SVR or NU_SVR for regression"
        if self.svm_type in [3,4] and examples.domain.classVar.varType==orange.VarTypes.Discrete:
            self.svm_type-=3
            #raise AttributeError, "Cannot do regression on descrete class data. Use C_SVC or NU_SVC for classification"
        if self.kernel_type==4 and not self.kernelFunc:
            raise AttributeError, "Custom kernel function not supplied"
        ##################################################
#        if self.kernel_type==4:     #There is a bug in svm. For some unknown reason only the probability model works with custom kernels
#            self.probability=True
        ##################################################
        nu = self.nu
        if self.svm_type == orange.SVMLearner.Nu_SVC: #check nu feasibility
            maxNu = self.maxNu(examples)
            if self.nu > maxNu:
                if getattr(self, "verbose", 0):
                    import warnings
                    warnings.warn("Specified nu %.3f is infeasible. Setting nu to %.3f" % (self.nu, maxNu))
                nu = max(maxNu - 1e-7, 0.0)
            
        for name in ["svm_type", "kernel_type", "kernelFunc", "C", "nu", "p", "gamma", "degree",
                "coef0", "shrinking", "probability", "verbose", "cache_size", "eps"]:
            self.learner.__dict__[name]=getattr(self, name)
        self.learner.nu = nu
        self.learner.setWeights(self.weight)
        return self.learnClassifier(examples)

    def learnClassifier(self, examples):
        if self.normalization:
            examples = self._normalize(examples)
            return SVMClassifierWrapper(self.learner(examples), examples.domain)
        return self.learner(examples)

    def tuneParameters(self, examples, parameters=None, folds=5, verbose=0, progressCallback=None):
        """ Tune the parameters of the SVMLearner on given examples using cross validation.
        Parameters:
            * *examples* ExampleTable on which to tune the parameters 
            * *parameters* if not set defaults to ["nu", "C", "gamma"]
            * *folds* number of folds used for cross validation
            * *verbose* 
            * *progressCallback* a callback function to report progress
            
        Example::
            >>> svm = orngSVM.SVMLearner()
            >>> svm.tuneParameters(examples, parameters=["gamma"], folds=3)
        This code tunes the *gamma* parameter on *examples* using 3-fold cross validation  
        
        """
        parameters = ["nu", "C", "gamma"] if parameters == None else parameters
        searchParams = []
        normalization = self.normalization
        if normalization:
            examples = self._normalize(examples)
            self.normalization = False
        if self.svm_type == SVMLearner.Nu_SVC and "nu" in parameters:
            numOfNuValues=9
            maxNu = max(self.maxNu(examples) - 1e-7, 0.0)
            searchParams.append(("nu", [i/10.0 for i in range(1, 9) if i/10.0 < maxNu] + [maxNu]))
        elif "C" in parameters:
            searchParams.append(("C", [2**a for a in  range(-5,15,2)]))
        if self.kernel_type==2 and "gamma" in parameters:
            searchParams.append(("gamma", [2**a for a in range(-5,5,2)]+[0]))
        tunedLearner = orngWrap.TuneMParameters(object=self, parameters=searchParams, folds=folds, 
                                                returnWhat=orngWrap.TuneMParameters.returnLearner, 
                                                progressCallback=progressCallback if progressCallback else lambda i:None)
        tunedLearner(examples, verbose=verbose)
        if normalization:
            self.normalization = normalization

    def _normalize(self, examples):
        dc = orange.DomainContinuizer()
        dc.classTreatment = orange.DomainContinuizer.Ignore
        dc.continuousTreatment = orange.DomainContinuizer.NormalizeBySpan
        dc.multinomialTreatment = orange.DomainContinuizer.NValues
        newdomain = dc(examples)
        return examples.translate(newdomain)

class SVMClassifierWrapper(object):
    def __init__(self, classifier, domain):
        self.classifier = classifier
        self.domain = domain
        
    def __getattr__(self, name):
        try:
            return getattr(self.__dict__["classifier"], name)
        except (KeyError, AttributeError):
            raise AttributeError(name)

    def __call__(self, example, what=orange.GetValue):
        example = orange.Example(self.domain, example)
        return self.classifier(example, what)
        
class SVMLearnerSparse(SVMLearner):
    def __init__(self, **kwds):
        SVMLearner.__init__(self, **kwds)
        self.learner=orange.SVMLearnerSparse(**kwds)

    
class SVMLearnerEasy(SVMLearner):
    def __init__(self, **kwds):
        self.folds=4
        self.verbose=0
        SVMLearner.__init__(self, **kwds)
        self.learner = SVMLearner(**kwds)
        
    def learnClassifier(self, examples):
        transformer=orange.DomainContinuizer()
        transformer.multinomialTreatment=orange.DomainContinuizer.NValues
        transformer.continuousTreatment=orange.DomainContinuizer.NormalizeBySpan
        transformer.classTreatment=orange.DomainContinuizer.Ignore
        newdomain=transformer(examples)
        newexamples=examples.translate(newdomain)
        #print newexamples[0]
        params={}
        parameters = []
        self.learner.normalization = False ## Normalization already done
        
        if self.svm_type in [1,4]:
            numOfNuValues=9
            if self.svm_type == SVMLearner.Nu_SVC:
                maxNu = max(self.maxNu(newexamples) - 1e-7, 0.0)
            else:
                maxNu = 1.0
            parameters.append(("nu", [i/10.0 for i in range(1, 9) if i/10.0 < maxNu] + [maxNu]))
        else:
            parameters.append(("C", [2**a for a in  range(-5,15,2)]))
        if self.kernel_type==2:
            parameters.append(("gamma", [2**a for a in range(-5,5,2)]+[0]))
        tunedLearner = orngWrap.TuneMParameters(object=self.learner, parameters=parameters, folds=self.folds)
        
        return SVMClassifierClassEasyWrapper(tunedLearner(newexamples, verbose=self.verbose), newdomain, examples)

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
    """ Returns a list of weights for linear class vs. class classifiers for the linear svm classifier.
    The list is in the order of 1vs2, 1vs3 ... 1vsN, 2vs3 ... e.g. a return value for a classifier trained
    on the iris dataset would contain three weights lists [Iris-setosa vs Iris-versicolor, Iris-setosa vs Iris-virginica, 
    Iris-versicolor vs Iris-virginica]"""
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
    """ Recursive feature elimination using linear svm derived attribute weights.
    Example::
        >>> rfe = RFE(SVMLearner(kernel_type=SVMLearner.Linear))
        >>> rfe.getAttrScores(data) #returns a dictionary of attribute scores
        {...}
        >>> data_with_removed_features = rfe(data, 5) # retruns an example table with only 5 best attributes
    """
    def __init__(self, learner=None):
        self.learner = learner or SVMLearner(kernel_type=orange.SVMLearner.Linear)

    def getAttrScores(self, data, stopAt=0):
        """ Return a dict mapping attributes to scores (scores are not scores in a general
        meaning they represent the step number at which they were removed from the recursive
        evaluation).  
        """
        iter = 1
        attrs = data.domain.attributese
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
        """ Return a new dataset with only *numSelected* best scoring attributes. 
        """
        scores = self.getAttrScores(data)
        scores = scores.items()
        scores.sort(lambda a,b:cmp(a[1],b[1]))
        scores = dict(scores[-numSelected:])
        attrs = [attr for attr in data.domain.attributes if attr in scores]
        data = data.select(attrs + [data.domain.classVar])
        return data

def exampleTableToSVMFormat(examples, file):
    """ Save an example table in .svm format used by libSVM
    """
    attrs = examples.domain.attributes + examples.domain.getmetas().values()
    attrs = [attr for attr in attrs if attr.varType in [orange.VarTypes.Continuous, orange.VarTypes.Discrete]]
    cv = examples.domain.classVar
    
    for ex in examples:
        file.write(str(int(ex[cv]) if cv.varType == orange.VarTypes.Discrete else float(ex[cv])))
        for i, attr in enumerate(attrs):
            if not ex[attr].isSpecial():
                file.write(" "+str(i+1)+":"+str(ex[attr]))
        file.write("\n")
            
class LinearLearner(orange.LinearLearner):
    """ A wrapper around orange.LinearLearner with a default
    solver_type == L2Loss_SVM_Dual (the default in orange.LinearLearner
    is L2_LR)
    """
    def __new__(cls, data=None, weightId=0, **kwargs):
        self = orange.LinearLearner.__new__(cls, **kwargs)
        if data:
            self.__init__(**kwargs)
            return self.__call__(data, weightId)
        else:
            return self
        
    def __init__(self, **kwargs):
        if kwargs.get("solver_type", None) in [orange.LinearLearner.L2_LR, None]:
            kwargs = dict(kwargs)
            kwargs["solver_type"] = orange.LinearLearner.L2Loss_SVM_Dual
        for name, val in kwargs.items():
            setattr(self, name, val)
