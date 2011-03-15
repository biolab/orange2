"""
.. index: logistic regression
.. index:
   single: classification; logistic regression

*******************
Logistic regression
*******************

Implements `logistic regression
<http://en.wikipedia.org/wiki/Logistic_regression>`_ with an extension for
proper treatment of discrete features.  The algorithm can handle various
anomalies in features, such as constant variables and singularities, that
could make fitting of logistic regression almost impossible. Stepwise
logistic regression, which iteratively selects the most informative
features, is also supported.


.. autoclass:: LogRegLearner
.. autoclass:: StepWiseFSS
.. autofunction:: dump

Examples
========

The first example shows a very simple induction of a logistic regression
classifier (`logreg-run.py`_, uses `titanic.tab`_).

.. literalinclude:: code/logreg-run.py

Result::

    Classification accuracy: 0.778282598819

    class attribute = survived
    class values = <no, yes>

        Attribute       beta  st. error     wald Z          P OR=exp(beta)

        Intercept      -1.23       0.08     -15.15      -0.00
     status=first       0.86       0.16       5.39       0.00       2.36
    status=second      -0.16       0.18      -0.91       0.36       0.85
     status=third      -0.92       0.15      -6.12       0.00       0.40
        age=child       1.06       0.25       4.30       0.00       2.89
       sex=female       2.42       0.14      17.04       0.00      11.25

The next examples shows how to handle singularities in data sets
(`logreg-singularities.py`_, uses `adult_sample.tab`_).

.. literalinclude:: code/logreg-singularities.py

The first few lines of the output of this script are::

    <=50K <=50K
    <=50K <=50K
    <=50K <=50K
    >50K >50K
    <=50K >50K

    class attribute = y
    class values = <>50K, <=50K>

                               Attribute       beta  st. error     wald Z          P OR=exp(beta)

                               Intercept       6.62      -0.00       -inf       0.00
                                     age      -0.04       0.00       -inf       0.00       0.96
                                  fnlwgt      -0.00       0.00       -inf       0.00       1.00
                           education-num      -0.28       0.00       -inf       0.00       0.76
                 marital-status=Divorced       4.29       0.00        inf       0.00      72.62
            marital-status=Never-married       3.79       0.00        inf       0.00      44.45
                marital-status=Separated       3.46       0.00        inf       0.00      31.95
                  marital-status=Widowed       3.85       0.00        inf       0.00      46.96
    marital-status=Married-spouse-absent       3.98       0.00        inf       0.00      53.63
        marital-status=Married-AF-spouse       4.01       0.00        inf       0.00      55.19
                 occupation=Tech-support      -0.32       0.00       -inf       0.00       0.72

If :obj:`removeSingular` is set to 0, inducing a logistic regression
classifier would return an error::

    Traceback (most recent call last):
      File "logreg-singularities.py", line 4, in <module>
        lr = classification.logreg.LogRegLearner(table, removeSingular=0)
      File "/home/jure/devel/orange/Orange/classification/logreg.py", line 255, in LogRegLearner
        return lr(examples, weightID)
      File "/home/jure/devel/orange/Orange/classification/logreg.py", line 291, in __call__
        lr = learner(examples, weight)
    orange.KernelException: 'orange.LogRegLearner': singularity in workclass=Never-worked

We can see that the attribute workclass is causing a singularity.

The example below shows, how the use of stepwise logistic regression can help to
gain in classification performance (`logreg-stepwise.py`_, uses `ionosphere.tab`_):

.. literalinclude:: code/logreg-stepwise.py

The output of this script is::

    Learner      CA
    logistic     0.841
    filtered     0.846

    Number of times attributes were used in cross-validation:
     1 x a21
    10 x a22
     8 x a23
     7 x a24
     1 x a25
    10 x a26
    10 x a27
     3 x a28
     7 x a29
     9 x a31
     2 x a16
     7 x a12
     1 x a32
     8 x a15
    10 x a14
     4 x a17
     7 x a30
    10 x a11
     1 x a10
     1 x a13
    10 x a34
     2 x a19
     1 x a18
    10 x a3
    10 x a5
     4 x a4
     4 x a7
     8 x a6
    10 x a9
    10 x a8

.. _logreg-run.py: code/logreg-run.py
.. _logreg-singularities.py: code/logreg-singularities.py
.. _logreg-stepwise.py: code/logreg-stepwise.py

.. _ionosphere.tab: code/ionosphere.tab
.. _adult_sample.tab: code/adult_sample.tab
.. _titanic.tab: code/titanic.tab

"""

#from Orange.core import LogRegLearner, LogRegClassifier, LogRegFitter, LogRegFitter_Cholesky

import Orange
import math, os
import warnings
from numpy import *
from numpy.linalg import *


##########################################################################
## Print out methods

def dump(classifier):
    """ Formatted string of all major features in logistic
    regression classifier. 

    :param classifier: logistic regression classifier
    """

    # print out class values
    out = ['']
    out.append("class attribute = " + classifier.domain.classVar.name)
    out.append("class values = " + str(classifier.domain.classVar.values))
    out.append('')
    
    # get the longest attribute name
    longest=0
    for at in classifier.continuizedDomain.attributes:
        if len(at.name)>longest:
            longest=len(at.name);

    # print out the head
    formatstr = "%"+str(longest)+"s %10s %10s %10s %10s %10s"
    out.append(formatstr % ("Feature", "beta", "st. error", "wald Z", "P", "OR=exp(beta)"))
    out.append('')
    formatstr = "%"+str(longest)+"s %10.2f %10.2f %10.2f %10.2f"    
    out.append(formatstr % ("Intercept", classifier.beta[0], classifier.beta_se[0], classifier.wald_Z[0], classifier.P[0]))
    formatstr = "%"+str(longest)+"s %10.2f %10.2f %10.2f %10.2f %10.2f"    
    for i in range(len(classifier.continuizedDomain.attributes)):
        out.append(formatstr % (classifier.continuizedDomain.attributes[i].name, classifier.beta[i+1], classifier.beta_se[i+1], classifier.wald_Z[i+1], abs(classifier.P[i+1]), math.exp(classifier.beta[i+1])))

    return '\n'.join(out)
        

def has_discrete_values(domain):
    for at in domain.attributes:
        if at.varType == Orange.core.VarTypes.Discrete:
            return 1
    return 0

class LogRegLearner(Orange.classification.Learner):
    """ Logistic regression learner.

    Implements logistic regression. If data instances are provided to
    the constructor, the learning algorithm is called and the resulting
    classifier is returned instead of the learner.

    :param table: data table with either discrete or continuous features
    :type table: Orange.data.Table
    :param weightID: the ID of the weight meta attribute
    :type weightID: int
    :param removeSingular: set to 1 if you want automatic removal of disturbing features, such as constants and singularities
    :type removeSingular: bool
    :param fitter: the fitting algorithm (by default the Newton-Raphson fitting algorithm is used)
    :param stepwiseLR: set to 1 if you wish to use stepwise logistic regression
    :type stepwiseLR: bool
    :param addCrit: parameter for stepwise feature selection
    :type addCrit: float
    :param deleteCrit: parameter for stepwise feature selection
    :type deleteCrit: float
    :param numFeatures: parameter for stepwise feature selection
    :type numFeatures: int
    :rtype: :obj:`LogRegLearner` or :obj:`LogRegClassifier`

    """
    def __new__(cls, instances=None, weightID=0, **argkw):
        self = Orange.classification.Learner.__new__(cls, **argkw)
        if instances:
            self.__init__(**argkw)
            return self.__call__(instances, weightID)
        else:
            return self

    def __init__(self, removeSingular=0, fitter = None, **kwds):
        self.__dict__.update(kwds)
        self.removeSingular = removeSingular
        self.fitter = None

    def __call__(self, examples, weight=0):
        imputer = getattr(self, "imputer", None) or None
        if getattr(self, "removeMissing", 0):
            examples = Orange.core.Preprocessor_dropMissing(examples)
##        if hasDiscreteValues(examples.domain):
##            examples = createNoDiscTable(examples)
        if not len(examples):
            return None
        if getattr(self, "stepwiseLR", 0):
            addCrit = getattr(self, "addCrit", 0.2)
            removeCrit = getattr(self, "removeCrit", 0.3)
            numFeatures = getattr(self, "numFeatures", -1)
            attributes = StepWiseFSS(examples, addCrit = addCrit, deleteCrit = removeCrit, imputer = imputer, numFeatures = numFeatures)
            tmpDomain = Orange.core.Domain(attributes, examples.domain.classVar)
            tmpDomain.addmetas(examples.domain.getmetas())
            examples = examples.select(tmpDomain)
        learner = Orange.core.LogRegLearner()
        learner.imputerConstructor = imputer
        if imputer:
            examples = self.imputer(examples)(examples)
        examples = Orange.core.Preprocessor_dropMissing(examples)
        if self.fitter:
            learner.fitter = self.fitter
        if self.removeSingular:
            lr = learner.fitModel(examples, weight)
        else:
            lr = learner(examples, weight)
        while isinstance(lr, Orange.core.Variable):
            if isinstance(lr.getValueFrom, Orange.core.ClassifierFromVar) and isinstance(lr.getValueFrom.transformer, Orange.core.Discrete2Continuous):
                lr = lr.getValueFrom.variable
            attributes = examples.domain.attributes[:]
            if lr in attributes:
                attributes.remove(lr)
            else:
                attributes.remove(lr.getValueFrom.variable)
            newDomain = Orange.core.Domain(attributes, examples.domain.classVar)
            newDomain.addmetas(examples.domain.getmetas())
            examples = examples.select(newDomain)
            lr = learner.fitModel(examples, weight)
        return lr



class UnivariateLogRegLearner(Orange.classification.Learner):
    def __new__(cls, instances=None, **argkw):
        self = Orange.classification.Learner.__new__(cls, **argkw)
        if instances:
            self.__init__(**argkw)
            return self.__call__(instances)
        else:
            return self

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __call__(self, examples):
        examples = createFullNoDiscTable(examples)
        classifiers = map(lambda x: LogRegLearner(Orange.core.Preprocessor_dropMissing(examples.select(Orange.core.Domain(x, examples.domain.classVar)))), examples.domain.attributes)
        maj_classifier = LogRegLearner(Orange.core.Preprocessor_dropMissing(examples.select(Orange.core.Domain(examples.domain.classVar))))
        beta = [maj_classifier.beta[0]] + [x.beta[1] for x in classifiers]
        beta_se = [maj_classifier.beta_se[0]] + [x.beta_se[1] for x in classifiers]
        P = [maj_classifier.P[0]] + [x.P[1] for x in classifiers]
        wald_Z = [maj_classifier.wald_Z[0]] + [x.wald_Z[1] for x in classifiers]
        domain = examples.domain

        return Univariate_LogRegClassifier(beta = beta, beta_se = beta_se, P = P, wald_Z = wald_Z, domain = domain)

class UnivariateLogRegClassifier(Orange.core.Classifier):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __call__(self, example, resultType = Orange.core.GetValue):
        # classification not implemented yet. For now its use is only to provide regression coefficients and its statistics
        pass
    

class LogRegLearnerGetPriors(object):
    def __new__(cls, instances=None, weightID=0, **argkw):
        self = object.__new__(cls, **argkw)
        if instances:
            self.__init__(**argkw)
            return self.__call__(instances, weightID)
        else:
            return self

    def __init__(self, removeSingular=0, **kwds):
        self.__dict__.update(kwds)
        self.removeSingular = removeSingular
    def __call__(self, examples, weight=0):
        # next function changes data set to a extended with unknown values 
        def createLogRegExampleTable(data, weightID):
            setsOfData = []
            for at in data.domain.attributes:
                # za vsak atribut kreiraj nov newExampleTable newData
                # v dataOrig, dataFinal in newData dodaj nov atribut -- continuous variable
                if at.varType == Orange.core.VarTypes.Continuous:
                    atDisc = Orange.core.FloatVariable(at.name + "Disc")
                    newDomain = Orange.core.Domain(data.domain.attributes+[atDisc,data.domain.classVar])
                    newDomain.addmetas(data.domain.getmetas())
                    newData = Orange.core.ExampleTable(newDomain,data)
                    altData = Orange.core.ExampleTable(newDomain,data)
                    for i,d in enumerate(newData):
                        d[atDisc] = 0
                        d[weightID] = 1*data[i][weightID]
                    for i,d in enumerate(altData):
                        d[atDisc] = 1
                        d[at] = 0
                        d[weightID] = 0.000001*data[i][weightID]
                elif at.varType == Orange.core.VarTypes.Discrete:
                # v dataOrig, dataFinal in newData atributu "at" dodaj ee  eno  vreednost, ki ima vrednost kar  ime atributa +  "X"
                    atNew = Orange.core.EnumVariable(at.name, values = at.values + [at.name+"X"])
                    newDomain = Orange.core.Domain(filter(lambda x: x!=at, data.domain.attributes)+[atNew,data.domain.classVar])
                    newDomain.addmetas(data.domain.getmetas())
                    newData = Orange.core.ExampleTable(newDomain,data)
                    altData = Orange.core.ExampleTable(newDomain,data)
                    for i,d in enumerate(newData):
                        d[atNew] = data[i][at]
                        d[weightID] = 1*data[i][weightID]
                    for i,d in enumerate(altData):
                        d[atNew] = at.name+"X"
                        d[weightID] = 0.000001*data[i][weightID]
                newData.extend(altData)
                setsOfData.append(newData)
            return setsOfData
                  
        learner = LogRegLearner(imputer = Orange.core.ImputerConstructor_average(), removeSingular = self.removeSingular)
        # get Original Model
        orig_model = learner(examples,weight)
        if orig_model.fit_status:
            print "Warning: model did not converge"

        # get extended Model (you should not change data)
        if weight == 0:
            weight = Orange.core.newmetaid()
            examples.addMetaAttribute(weight, 1.0)
        extended_set_of_examples = createLogRegExampleTable(examples, weight)
        extended_models = [learner(extended_examples, weight) \
                           for extended_examples in extended_set_of_examples]

##        print examples[0]
##        printOUT(orig_model)
##        print orig_model.domain
##        print orig_model.beta
##        print orig_model.beta[orig_model.continuizedDomain.attributes[-1]]
##        for i,m in enumerate(extended_models):
##            print examples.domain.attributes[i]
##            printOUT(m)
            
        
        # izracunas odstopanja
        # get sum of all betas
        beta = 0
        betas_ap = []
        for m in extended_models:
            beta_add = m.beta[m.continuizedDomain.attributes[-1]]
            betas_ap.append(beta_add)
            beta = beta + beta_add
        
        # substract it from intercept
        #print "beta", beta
        logistic_prior = orig_model.beta[0]+beta
        
        # compare it to bayes prior
        bayes = Orange.core.BayesLearner(examples)
        bayes_prior = math.log(bayes.distribution[1]/bayes.distribution[0])

        # normalize errors
##        print "bayes", bayes_prior
##        print "lr", orig_model.beta[0]
##        print "lr2", logistic_prior
##        print "dist", Orange.core.Distribution(examples.domain.classVar,examples)
##        print "prej", betas_ap

        # error normalization - to avoid errors due to assumption of independence of unknown values
        dif = bayes_prior - logistic_prior
        positives = sum(filter(lambda x: x>=0, betas_ap))
        negatives = -sum(filter(lambda x: x<0, betas_ap))
        if not negatives == 0:
            kPN = positives/negatives
            diffNegatives = dif/(1+kPN)
            diffPositives = kPN*diffNegatives
            kNegatives = (negatives-diffNegatives)/negatives
            kPositives = positives/(positives-diffPositives)
    ##        print kNegatives
    ##        print kPositives

            for i,b in enumerate(betas_ap):
                if b<0: betas_ap[i]*=kNegatives
                else: betas_ap[i]*=kPositives
        #print "potem", betas_ap

        # vrni originalni model in pripadajoce apriorne niclele
        return (orig_model, betas_ap)
        #return (bayes_prior,orig_model.beta[examples.domain.classVar],logistic_prior)

class LogRegLearnerGetPriorsOneTable:
    def __init__(self, removeSingular=0, **kwds):
        self.__dict__.update(kwds)
        self.removeSingular = removeSingular
    def __call__(self, examples, weight=0):
        # next function changes data set to a extended with unknown values 
        def createLogRegExampleTable(data, weightID):
            finalData = Orange.core.ExampleTable(data)
            origData = Orange.core.ExampleTable(data)
            for at in data.domain.attributes:
                # za vsak atribut kreiraj nov newExampleTable newData
                # v dataOrig, dataFinal in newData dodaj nov atribut -- continuous variable
                if at.varType == Orange.core.VarTypes.Continuous:
                    atDisc = Orange.core.FloatVariable(at.name + "Disc")
                    newDomain = Orange.core.Domain(origData.domain.attributes+[atDisc,data.domain.classVar])
                    newDomain.addmetas(newData.domain.getmetas())
                    finalData = Orange.core.ExampleTable(newDomain,finalData)
                    newData = Orange.core.ExampleTable(newDomain,origData)
                    origData = Orange.core.ExampleTable(newDomain,origData)
                    for d in origData:
                        d[atDisc] = 0
                    for d in finalData:
                        d[atDisc] = 0
                    for i,d in enumerate(newData):
                        d[atDisc] = 1
                        d[at] = 0
                        d[weightID] = 100*data[i][weightID]
                        
                elif at.varType == Orange.core.VarTypes.Discrete:
                # v dataOrig, dataFinal in newData atributu "at" dodaj ee  eno  vreednost, ki ima vrednost kar  ime atributa +  "X"
                    atNew = Orange.core.EnumVariable(at.name, values = at.values + [at.name+"X"])
                    newDomain = Orange.core.Domain(filter(lambda x: x!=at, origData.domain.attributes)+[atNew,origData.domain.classVar])
                    newDomain.addmetas(origData.domain.getmetas())
                    temp_finalData = Orange.core.ExampleTable(finalData)
                    finalData = Orange.core.ExampleTable(newDomain,finalData)
                    newData = Orange.core.ExampleTable(newDomain,origData)
                    temp_origData = Orange.core.ExampleTable(origData)
                    origData = Orange.core.ExampleTable(newDomain,origData)
                    for i,d in enumerate(origData):
                        d[atNew] = temp_origData[i][at]
                    for i,d in enumerate(finalData):
                        d[atNew] = temp_finalData[i][at]                        
                    for i,d in enumerate(newData):
                        d[atNew] = at.name+"X"
                        d[weightID] = 10*data[i][weightID]
                finalData.extend(newData)
            return finalData
                  
        learner = LogRegLearner(imputer = Orange.core.ImputerConstructor_average(), removeSingular = self.removeSingular)
        # get Original Model
        orig_model = learner(examples,weight)

        # get extended Model (you should not change data)
        if weight == 0:
            weight = Orange.core.newmetaid()
            examples.addMetaAttribute(weight, 1.0)
        extended_examples = createLogRegExampleTable(examples, weight)
        extended_model = learner(extended_examples, weight)

##        print examples[0]
##        printOUT(orig_model)
##        print orig_model.domain
##        print orig_model.beta

##        printOUT(extended_model)        
        # izracunas odstopanja
        # get sum of all betas
        beta = 0
        betas_ap = []
        for m in extended_models:
            beta_add = m.beta[m.continuizedDomain.attributes[-1]]
            betas_ap.append(beta_add)
            beta = beta + beta_add
        
        # substract it from intercept
        #print "beta", beta
        logistic_prior = orig_model.beta[0]+beta
        
        # compare it to bayes prior
        bayes = Orange.core.BayesLearner(examples)
        bayes_prior = math.log(bayes.distribution[1]/bayes.distribution[0])

        # normalize errors
        #print "bayes", bayes_prior
        #print "lr", orig_model.beta[0]
        #print "lr2", logistic_prior
        #print "dist", Orange.core.Distribution(examples.domain.classVar,examples)
        k = (bayes_prior-orig_model.beta[0])/(logistic_prior-orig_model.beta[0])
        #print "prej", betas_ap
        betas_ap = [k*x for x in betas_ap]                
        #print "potem", betas_ap

        # vrni originalni model in pripadajoce apriorne niclele
        return (orig_model, betas_ap)
        #return (bayes_prior,orig_model.beta[data.domain.classVar],logistic_prior)


######################################
#### Fitters for logistic regression (logreg) learner ####
######################################

def pr(x, betas):
    k = math.exp(dot(x, betas))
    return k / (1+k)

def lh(x,y,betas):
    llh = 0.0
    for i,x_i in enumerate(x):
        pr = pr(x_i,betas)
        llh += y[i]*log(max(pr,1e-6)) + (1-y[i])*log(max(1-pr,1e-6))
    return llh


def diag(vector):
    mat = identity(len(vector), Float)
    for i,v in enumerate(vector):
        mat[i][i] = v
    return mat
    
class SimpleFitter(Orange.core.LogRegFitter):
    def __init__(self, penalty=0, se_penalty = False):
        self.penalty = penalty
        self.se_penalty = se_penalty
    def __call__(self, data, weight=0):
        ml = data.native(0)
        for i in range(len(data.domain.attributes)):
          a = data.domain.attributes[i]
          if a.varType == Orange.core.VarTypes.Discrete:
            for m in ml:
              m[i] = a.values.index(m[i])
        for m in ml:
          m[-1] = data.domain.classVar.values.index(m[-1])
        Xtmp = array(ml)
        y = Xtmp[:,-1]   # true probabilities (1's or 0's)
        one = reshape(array([1]*len(data)), (len(data),1)) # intercept column
        X=concatenate((one, Xtmp[:,:-1]),1)  # intercept first, then data

        betas = array([0.0] * (len(data.domain.attributes)+1))
        oldBetas = array([1.0] * (len(data.domain.attributes)+1))
        N = len(data)

        pen_matrix = array([self.penalty] * (len(data.domain.attributes)+1))
        if self.se_penalty:
            p = array([pr(X[i], betas) for i in range(len(data))])
            W = identity(len(data), Float)
            pp = p * (1.0-p)
            for i in range(N):
                W[i,i] = pp[i]
            se = sqrt(diagonal(inverse(matrixmultiply(transpose(X), matrixmultiply(W, X)))))
            for i,p in enumerate(pen_matrix):
                pen_matrix[i] *= se[i]
        # predict the probability for an instance, x and betas are vectors
        # start the computation
        likelihood = 0.
        likelihood_new = 1.
        while abs(likelihood - likelihood_new)>1e-5:
            likelihood = likelihood_new
            oldBetas = betas
            p = array([pr(X[i], betas) for i in range(len(data))])

            W = identity(len(data), Float)
            pp = p * (1.0-p)
            for i in range(N):
                W[i,i] = pp[i]

            WI = inverse(W)
            z = matrixmultiply(X, betas) + matrixmultiply(WI, y - p)

            tmpA = inverse(matrixmultiply(transpose(X), matrixmultiply(W, X))+diag(pen_matrix))
            tmpB = matrixmultiply(transpose(X), y-p)
            betas = oldBetas + matrixmultiply(tmpA,tmpB)
#            betaTemp = matrixmultiply(matrixmultiply(matrixmultiply(matrixmultiply(tmpA,transpose(X)),W),X),oldBetas)
#            print betaTemp
#            tmpB = matrixmultiply(transpose(X), matrixmultiply(W, z))
#            betas = matrixmultiply(tmpA, tmpB)
            likelihood_new = lh(X,y,betas)-self.penalty*sum([b*b for b in betas])
            print likelihood_new

            
            
##        XX = sqrt(diagonal(inverse(matrixmultiply(transpose(X),X))))
##        yhat = array([pr(X[i], betas) for i in range(len(data))])
##        ss = sum((y - yhat) ** 2) / (N - len(data.domain.attributes) - 1)
##        sigma = math.sqrt(ss)
        p = array([pr(X[i], betas) for i in range(len(data))])
        W = identity(len(data), Float)
        pp = p * (1.0-p)
        for i in range(N):
            W[i,i] = pp[i]
        diXWX = sqrt(diagonal(inverse(matrixmultiply(transpose(X), matrixmultiply(W, X)))))
        xTemp = matrixmultiply(matrixmultiply(inverse(matrixmultiply(transpose(X), matrixmultiply(W, X))),transpose(X)),y)
        beta = []
        beta_se = []
        print "likelihood ridge", likelihood
        for i in range(len(betas)):
            beta.append(betas[i])
            beta_se.append(diXWX[i])
        return (self.OK, beta, beta_se, 0)

def pr_bx(bx):
    if bx > 35:
        return 1
    if bx < -35:
        return 0
    return exp(bx)/(1+exp(bx))

class BayesianFitter(Orange.core.LogRegFitter):
    def __init__(self, penalty=0, anch_examples=[], tau = 0):
        self.penalty = penalty
        self.anch_examples = anch_examples
        self.tau = tau

    def create_array_data(self,data):
        if not len(data):
            return (array([]),array([]))
        # convert data to numeric
        ml = data.native(0)
        for i,a in enumerate(data.domain.attributes):
          if a.varType == Orange.core.VarTypes.Discrete:
            for m in ml:
              m[i] = a.values.index(m[i])
        for m in ml:
          m[-1] = data.domain.classVar.values.index(m[-1])
        Xtmp = array(ml)
        y = Xtmp[:,-1]   # true probabilities (1's or 0's)
        one = reshape(array([1]*len(data)), (len(data),1)) # intercept column
        X=concatenate((one, Xtmp[:,:-1]),1)  # intercept first, then data
        return (X,y)
    
    def __call__(self, data, weight=0):
        (X,y)=self.create_array_data(data)

        exTable = Orange.core.ExampleTable(data.domain)
        for id,ex in self.anch_examples:
            exTable.extend(Orange.core.ExampleTable(ex,data.domain))
        (X_anch,y_anch)=self.create_array_data(exTable)

        betas = array([0.0] * (len(data.domain.attributes)+1))

        likelihood,betas = self.estimate_beta(X,y,betas,[0]*(len(betas)),X_anch,y_anch)

        # get attribute groups atGroup = [(startIndex, number of values), ...)
        ats = data.domain.attributes
        atVec=reduce(lambda x,y: x+[(y,not y==x[-1][0])], [a.getValueFrom and a.getValueFrom.whichVar or a for a in ats],[(ats[0].getValueFrom and ats[0].getValueFrom.whichVar or ats[0],0)])[1:]
        atGroup=[[0,0]]
        for v_i,v in enumerate(atVec):
            if v[1]==0: atGroup[-1][1]+=1
            else:       atGroup.append([v_i,1])
        
        # compute zero values for attributes
        sumB = 0.
        for ag in atGroup:
            X_temp = concatenate((X[:,:ag[0]+1],X[:,ag[0]+1+ag[1]:]),1)
            if X_anch:
                X_anch_temp = concatenate((X_anch[:,:ag[0]+1],X_anch[:,ag[0]+1+ag[1]:]),1)
            else: X_anch_temp = X_anch
##            print "1", concatenate((betas[:i+1],betas[i+2:]))
##            print "2", betas
            likelihood_temp,betas_temp=self.estimate_beta(X_temp,y,concatenate((betas[:ag[0]+1],betas[ag[0]+ag[1]+1:])),[0]+[1]*(len(betas)-1-ag[1]),X_anch_temp,y_anch)
            print "finBetas", betas, betas_temp
            print "betas", betas[0], betas_temp[0]
            sumB += betas[0]-betas_temp[0]
        apriori = Orange.core.Distribution(data.domain.classVar, data)
        aprioriProb = apriori[0]/apriori.abs
        
        print "koncni rezultat", sumB, math.log((1-aprioriProb)/aprioriProb), betas[0]
            
        beta = []
        beta_se = []
        print "likelihood2", likelihood
        for i in range(len(betas)):
            beta.append(betas[i])
            beta_se.append(0.0)
        return (self.OK, beta, beta_se, 0)

     
        
    def estimate_beta(self,X,y,betas,const_betas,X_anch,y_anch):
        N,N_anch = len(y),len(y_anch)
        r,r_anch = array([dot(X[i], betas) for i in range(N)]),\
                   array([dot(X_anch[i], betas) for i in range(N_anch)])
        p    = array([pr_bx(ri) for ri in r])
        X_sq = X*X

        max_delta      = [1.]*len(const_betas)
        likelihood     = -1.e+10
        likelihood_new = -1.e+9
        while abs(likelihood - likelihood_new)>0.01 and max(max_delta)>0.01:
            likelihood = likelihood_new
            print likelihood
            betas_temp = [b for b in betas]
            for j in range(len(betas)):
                if const_betas[j]: continue
                dl = matrixmultiply(X[:,j],transpose(y-p))
                for xi,x in enumerate(X_anch):
                    dl += self.penalty*x[j]*(y_anch[xi] - pr_bx(r_anch[xi]*self.penalty))

                ddl = matrixmultiply(X_sq[:,j],transpose(p*(1-p)))
                for xi,x in enumerate(X_anch):
                    ddl += self.penalty*x[j]*pr_bx(r[xi]*self.penalty)*(1-pr_bx(r[xi]*self.penalty))

                if j==0:
                    dv = dl/max(ddl,1e-6)
                elif betas[j] == 0: # special handling due to non-defined first and second derivatives
                    dv = (dl-self.tau)/max(ddl,1e-6)
                    if dv < 0:
                        dv = (dl+self.tau)/max(ddl,1e-6)
                        if dv > 0:
                            dv = 0
                else:
                    dl -= sign(betas[j])*self.tau
                    dv = dl/max(ddl,1e-6)
                    if not sign(betas[j] + dv) == sign(betas[j]):
                        dv = -betas[j]
                dv = min(max(dv,-max_delta[j]),max_delta[j])
                r+= X[:,j]*dv
                p = array([pr_bx(ri) for ri in r])
                if N_anch:
                    r_anch+=X_anch[:,j]*dv
                betas[j] += dv
                max_delta[j] = max(2*abs(dv),max_delta[j]/2)
            likelihood_new = lh(X,y,betas)
            for xi,x in enumerate(X_anch):
                try:
                    likelihood_new += y_anch[xi]*r_anch[xi]*self.penalty-log(1+exp(r_anch[xi]*self.penalty))
                except:
                    likelihood_new += r_anch[xi]*self.penalty*(y_anch[xi]-1)
            likelihood_new -= sum([abs(b) for b in betas[1:]])*self.tau
            if likelihood_new < likelihood:
                max_delta = [md/4 for md in max_delta]
                likelihood_new = likelihood
                likelihood = likelihood_new + 1.
                betas = [b for b in betas_temp]
        print "betas", betas
        print "init_like", likelihood_new
        print "pure_like", lh(X,y,betas)
        return (likelihood,betas)
    
############################################################
#  Feature subset selection for logistic regression

def get_likelihood(fitter, examples):
    res = fitter(examples)
    if res[0] in [fitter.OK]: #, fitter.Infinity, fitter.Divergence]:
       status, beta, beta_se, likelihood = res
       if sum([abs(b) for b in beta])<sum([abs(b) for b in beta_se]):
           return -100*len(examples)
       return likelihood
    else:
       return -100*len(examples)
        


class StepWiseFSS(Orange.classification.Learner):
  """Implementation of algorithm described in [Hosmer and Lemeshow, Applied Logistic Regression, 2000].

  Perform stepwise logistic regression and return a list of the
  most "informative" features. Each step of the algorithm is composed
  of two parts. The first is backward elimination, where each already
  chosen feature is tested for a significant contribution to the overall
  model. If the worst among all tested features has higher significance
  than is specified in :obj:`deleteCrit`, the feature is removed from
  the model. The second step is forward selection, which is similar to
  backward elimination. It loops through all the features that are not
  in the model and tests whether they contribute to the common model
  with significance lower that :obj:`addCrit`. The algorithm stops when
  no feature in the model is to be removed and no feature not in the
  model is to be added. By setting :obj:`numFeatures` larger than -1,
  the algorithm will stop its execution when the number of features in model
  exceeds that number.

  Significances are assesed via the likelihood ration chi-square
  test. Normal F test is not appropriate, because errors are assumed to
  follow a binomial distribution.

  If :obj:`table` is specified, stepwise logistic regression implemented
  in :obj:`StepWiseFSS` is performed and a list of chosen features
  is returned. If :obj:`table` is not specified an instance of
  :obj:`StepWiseFSS` with all parameters set is returned.

  :param table: data set
  :type table: Orange.data.Table

  :param addCrit: "Alpha" level to judge if variable has enough importance to be added in the new set. (e.g. if addCrit is 0.2, then features is added if its P is lower than 0.2)
  :type addCrit: float

  :param deleteCrit: Similar to addCrit, just that it is used at backward elimination. It should be higher than addCrit!
  :type deleteCrit: float

  :param numFeatures: maximum number of selected features, use -1 for infinity.
  :type numFeatures: int
  :rtype: :obj:`StepWiseFSS` or list of features

  """

  def __new__(cls, instances=None, **argkw):
      self = Orange.classification.Learner.__new__(cls, **argkw)
      if instances:
          self.__init__(**argkw)
          return self.__call__(instances)
      else:
          return self


  def __init__(self, addCrit=0.2, deleteCrit=0.3, numFeatures = -1, **kwds):
    self.__dict__.update(kwds)
    self.addCrit = addCrit
    self.deleteCrit = deleteCrit
    self.numFeatures = numFeatures
  def __call__(self, examples):
    if getattr(self, "imputer", 0):
        examples = self.imputer(examples)(examples)
    if getattr(self, "removeMissing", 0):
        examples = Orange.core.Preprocessor_dropMissing(examples)
    continuizer = Orange.core.DomainContinuizer(zeroBased=1,continuousTreatment=Orange.core.DomainContinuizer.Leave,
                                           multinomialTreatment = Orange.core.DomainContinuizer.FrequentIsBase,
                                           classTreatment = Orange.core.DomainContinuizer.Ignore)
    attr = []
    remain_attr = examples.domain.attributes[:]

    # get LL for Majority Learner 
    tempDomain = Orange.core.Domain(attr,examples.domain.classVar)
    #tempData  = Orange.core.Preprocessor_dropMissing(examples.select(tempDomain))
    tempData  = Orange.core.Preprocessor_dropMissing(examples.select(tempDomain))

    ll_Old = get_likelihood(Orange.core.LogRegFitter_Cholesky(), tempData)
    ll_Best = -1000000
    length_Old = float(len(tempData))

    stop = 0
    while not stop:
        # LOOP until all variables are added or no further deletion nor addition of attribute is possible
        worstAt = None
        # if there are more than 1 attribute then perform backward elimination
        if len(attr) >= 2:
            minG = 1000
            worstAt = attr[0]
            ll_Best = ll_Old
            length_Best = length_Old
            for at in attr:
                # check all attribute whether its presence enough increases LL?

                tempAttr = filter(lambda x: x!=at, attr)
                tempDomain = Orange.core.Domain(tempAttr,examples.domain.classVar)
                tempDomain.addmetas(examples.domain.getmetas())
                # domain, calculate P for LL improvement.
                tempDomain  = continuizer(Orange.core.Preprocessor_dropMissing(examples.select(tempDomain)))
                tempData = Orange.core.Preprocessor_dropMissing(examples.select(tempDomain))

                ll_Delete = get_likelihood(Orange.core.LogRegFitter_Cholesky(), tempData)
                length_Delete = float(len(tempData))
                length_Avg = (length_Delete + length_Old)/2.0

                G=-2*length_Avg*(ll_Delete/length_Delete-ll_Old/length_Old)

                # set new worst attribute                
                if G<minG:
                    worstAt = at
                    minG=G
                    ll_Best = ll_Delete
                    length_Best = length_Delete
            # deletion of attribute
            
            if worstAt.varType==Orange.core.VarTypes.Continuous:
                P=lchisqprob(minG,1);
            else:
                P=lchisqprob(minG,len(worstAt.values)-1);
            if P>=self.deleteCrit:
                attr.remove(worstAt)
                remain_attr.append(worstAt)
                nodeletion=0
                ll_Old = ll_Best
                length_Old = length_Best
            else:
                nodeletion=1
        else:
            nodeletion = 1
            # END OF DELETION PART
            
        # if enough attributes has been chosen, stop the procedure
        if self.numFeatures>-1 and len(attr)>=self.numFeatures:
            remain_attr=[]
         
        # for each attribute in the remaining
        maxG=-1
        ll_Best = ll_Old
        length_Best = length_Old
        bestAt = None
        for at in remain_attr:
            tempAttr = attr + [at]
            tempDomain = Orange.core.Domain(tempAttr,examples.domain.classVar)
            tempDomain.addmetas(examples.domain.getmetas())
            # domain, calculate P for LL improvement.
            tempDomain  = continuizer(Orange.core.Preprocessor_dropMissing(examples.select(tempDomain)))
            tempData = Orange.core.Preprocessor_dropMissing(examples.select(tempDomain))
            ll_New = get_likelihood(Orange.core.LogRegFitter_Cholesky(), tempData)

            length_New = float(len(tempData)) # get number of examples in tempData to normalize likelihood

            # P=PR(CHI^2>G), G=-2(L(0)-L(1))=2(E(0)-E(1))
            length_avg = (length_New + length_Old)/2
            G=-2*length_avg*(ll_Old/length_Old-ll_New/length_New);
            if G>maxG:
                bestAt = at
                maxG=G
                ll_Best = ll_New
                length_Best = length_New
        if not bestAt:
            stop = 1
            continue
        
        if bestAt.varType==Orange.core.VarTypes.Continuous:
            P=lchisqprob(maxG,1);
        else:
            P=lchisqprob(maxG,len(bestAt.values)-1);
        # Add attribute with smallest P to attributes(attr)
        if P<=self.addCrit:
            attr.append(bestAt)
            remain_attr.remove(bestAt)
            ll_Old = ll_Best
            length_Old = length_Best

        if (P>self.addCrit and nodeletion) or (bestAt == worstAt):
            stop = 1

    return attr


class StepWiseFSSFilter(object):
    def __new__(cls, instances=None, **argkw):
        self = object.__new__(cls, **argkw)
        if instances:
            self.__init__(**argkw)
            return self.__call__(instances)
        else:
            return self
    
    def __init__(self, addCrit=0.2, deleteCrit=0.3, numFeatures = -1):
        self.addCrit = addCrit
        self.deleteCrit = deleteCrit
        self.numFeatures = numFeatures

    def __call__(self, examples):
        attr = StepWiseFSS(examples, addCrit=self.addCrit, deleteCrit = self.deleteCrit, numFeatures = self.numFeatures)
        return examples.select(Orange.core.Domain(attr, examples.domain.classVar))
                

####################################
##  PROBABILITY CALCULATIONS

def lchisqprob(chisq,df):
    """
    Return the (1-tailed) probability value associated with the provided
    chi-square value and df.  Adapted from chisq.c in Gary Perlman's |Stat.
    """
    BIG = 20.0
    def ex(x):
    	BIG = 20.0
    	if x < -BIG:
    	    return 0.0
    	else:
    	    return math.exp(x)
    if chisq <=0 or df < 1:
    	return 1.0
    a = 0.5 * chisq
    if df%2 == 0:
    	even = 1
    else:
    	even = 0
    if df > 1:
    	y = ex(-a)
    if even:
    	s = y
    else:
        s = 2.0 * zprob(-math.sqrt(chisq))
    if (df > 2):
        chisq = 0.5 * (df - 1.0)
        if even:
            z = 1.0
        else:
            z = 0.5
        if a > BIG:
            if even:
            	e = 0.0
            else:
            	e = math.log(math.sqrt(math.pi))
            c = math.log(a)
            while (z <= chisq):
            	e = math.log(z) + e
            	s = s + ex(c*z-a-e)
            	z = z + 1.0
            return s
        else:
            if even:
                e = 1.0
            else:
                e = 1.0 / math.sqrt(math.pi) / math.sqrt(a)
            c = 0.0
            while (z <= chisq):
                e = e * (a/float(z))
                c = c + e
                z = z + 1.0
            return (c*y+s)
    else:
        return s


def zprob(z):
    """
    Returns the area under the normal curve 'to the left of' the given z value.
    Thus:: 

    for z<0, zprob(z) = 1-tail probability
    for z>0, 1.0-zprob(z) = 1-tail probability
    for any z, 2.0*(1.0-zprob(abs(z))) = 2-tail probability

    Adapted from z.c in Gary Perlman's |Stat.
    """
    Z_MAX = 6.0    # maximum meaningful z-value
    if z == 0.0:
	x = 0.0
    else:
	y = 0.5 * math.fabs(z)
	if y >= (Z_MAX*0.5):
	    x = 1.0
	elif (y < 1.0):
	    w = y*y
	    x = ((((((((0.000124818987 * w
			-0.001075204047) * w +0.005198775019) * w
		      -0.019198292004) * w +0.059054035642) * w
		    -0.151968751364) * w +0.319152932694) * w
		  -0.531923007300) * w +0.797884560593) * y * 2.0
	else:
	    y = y - 2.0
	    x = (((((((((((((-0.000045255659 * y
			     +0.000152529290) * y -0.000019538132) * y
			   -0.000676904986) * y +0.001390604284) * y
			 -0.000794620820) * y -0.002034254874) * y
		       +0.006549791214) * y -0.010557625006) * y
		     +0.011630447319) * y -0.009279453341) * y
		   +0.005353579108) * y -0.002141268741) * y
		 +0.000535310849) * y +0.999936657524
    if z > 0.0:
	prob = ((x+1.0)*0.5)
    else:
	prob = ((1.0-x)*0.5)
    return prob

   
