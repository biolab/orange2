"""
.. index: logreg

===================
Logistic Regression
===================

Module :obj:`Orange.classification.logreg` is a set of wrappers around
classes LogisticLearner and LogisticClassifier, that are implemented
in core Orange. This module extends the use of logistic regression
to discrete attributes, it helps handling various anomalies in
attributes, such as constant variables and singularities, that make
fitting logistic regression almost impossible. It also implements a
function for constructing a stepwise logistic regression, which is a
good technique for prevent overfitting, and is a good feature subset
selection technique as well.

Functions
---------

.. autofunction:: LogRegLearner
.. autofunction:: StepWiseFSS
.. autofunction:: printOUT

Class
-----

.. autoclass:: StepWiseFSS_class

Examples
--------

First example shows a very simple induction of a logistic regression
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

Next examples shows how to handle singularities in data sets
(`logreg-singularities.py`_, uses `adult_sample.tab`_).

.. literalinclude:: code/logreg-singularities.py

Result::

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
                 occupation=Craft-repair       0.37       0.00        inf       0.00       1.45
                occupation=Other-service       2.68       0.00        inf       0.00      14.61
                        occupation=Sales       0.22       0.00        inf       0.00       1.24
               occupation=Prof-specialty       0.18       0.00        inf       0.00       1.19
            occupation=Handlers-cleaners       1.29       0.00        inf       0.00       3.64
            occupation=Machine-op-inspct       0.86       0.00        inf       0.00       2.37
                 occupation=Adm-clerical       0.30       0.00        inf       0.00       1.35
              occupation=Farming-fishing       1.12       0.00        inf       0.00       3.06
             occupation=Transport-moving       0.62       0.00        inf       0.00       1.85
              occupation=Priv-house-serv       3.46       0.00        inf       0.00      31.87
              occupation=Protective-serv       0.11       0.00        inf       0.00       1.12
                 occupation=Armed-Forces       0.59       0.00        inf       0.00       1.81
                       relationship=Wife      -1.06      -0.00        inf       0.00       0.35
                  relationship=Own-child      -1.04      60.00      -0.02       0.99       0.35
              relationship=Not-in-family      -1.94  532845.00      -0.00       1.00       0.14
             relationship=Other-relative      -2.42       0.00       -inf       0.00       0.09
                  relationship=Unmarried      -1.92       0.00       -inf       0.00       0.15
                 race=Asian-Pac-Islander      -0.19       0.00       -inf       0.00       0.83
                 race=Amer-Indian-Eskimo       2.88       0.00        inf       0.00      17.78
                              race=Other       3.93       0.00        inf       0.00      51.07
                              race=Black       0.11       0.00        inf       0.00       1.12
                              sex=Female       0.30       0.00        inf       0.00       1.36
                            capital-gain      -0.00       0.00       -inf       0.00       1.00
                            capital-loss      -0.00       0.00       -inf       0.00       1.00
                          hours-per-week      -0.04       0.00       -inf       0.00       0.96

In case we set removeSingular to 0, inducing logistic regression
classifier would return an error::

    Traceback (most recent call last):
      File "logreg-singularities.py", line 4, in <module>
        lr = classification.logreg.LogRegLearner(table, removeSingular=0)
      File "/home/jure/devel/orange/Orange/classification/logreg.py", line 255, in LogRegLearner
        return lr(examples, weightID)
      File "/home/jure/devel/orange/Orange/classification/logreg.py", line 291, in __call__
        lr = learner(examples, weight)
    orange.KernelException: 'orange.LogRegLearner': singularity in workclass=Never-worked


We can see that attribute workclass=Never-worked is causeing singularity. The
issue of this is that we should remove Never-worked manually or leave it to
function LogRegLearner to remove it automatically. 

In the last example it is shown, how the use of stepwise logistic
regression can help us in achieving better classification
(`logreg-stepwise.py`_, uses `ionosphere.tab`_):

.. literalinclude:: code/logreg-stepwise.py

Result::

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

References
----------

David W. Hosmer, Stanley Lemeshow. Applied Logistic Regression - 2nd ed. Wiley, New York, 2000 


.. _logreg-run.py: code/logreg-run.py
.. _logreg-singularities.py: code/logreg-singularities.py
.. _logreg-stepwise.py: code/logreg-stepwise.py

.. _ionosphere.tab: code/ionosphere.tab
.. _adult_sample.tab: code/adult_sample.tab
.. _titanic.tab: code/titanic.tab

"""

from orange import LogRegLearner, LogRegClassifier, LogRegFitter, LogRegFitter_Cholesky

import orange
import orngCI
import math, os
from numpy import *
from numpy.linalg import *


#######################
## Print out methods ##
#######################

def printOUT(classifier):
    """ Formatted print to console of all major attributes in logistic
    regression classifier. Parameter classifier is a logistic regression
    classifier.

    :param examples: data set
    :type examples: :obj:`Orange.data.table`
    """

    # print out class values
    print
    print "class attribute = " + classifier.domain.classVar.name
    print "class values = " + str(classifier.domain.classVar.values)
    print
    
    # get the longest attribute name
    longest=0
    for at in classifier.continuizedDomain.attributes:
        if len(at.name)>longest:
            longest=len(at.name);

    # print out the head
    formatstr = "%"+str(longest)+"s %10s %10s %10s %10s %10s"
    print formatstr % ("Attribute", "beta", "st. error", "wald Z", "P", "OR=exp(beta)")
    print
    formatstr = "%"+str(longest)+"s %10.2f %10.2f %10.2f %10.2f"    
    print formatstr % ("Intercept", classifier.beta[0], classifier.beta_se[0], classifier.wald_Z[0], classifier.P[0])
    formatstr = "%"+str(longest)+"s %10.2f %10.2f %10.2f %10.2f %10.2f"    
    for i in range(len(classifier.continuizedDomain.attributes)):
        print formatstr % (classifier.continuizedDomain.attributes[i].name, classifier.beta[i+1], classifier.beta_se[i+1], classifier.wald_Z[i+1], abs(classifier.P[i+1]), math.exp(classifier.beta[i+1]))
        

def hasDiscreteValues(domain):
    for at in domain.attributes:
        if at.varType == orange.VarTypes.Discrete:
            return 1
    return 0

def LogRegLearner(table=None, weightID=0, **kwds):
    """ Logistic regression learner

    :obj:`LogRegLearner` implements logistic regression. If data
    instances are provided to the constructor, the learning algorithm
    is called and the resulting classifier is returned instead of the
    learner.

    :param table: data set with either discrete or continuous features
    :type table: Orange.table.data
    :param weightID: the ID of the weight meta attribute
    :param removeSingular: set to 1 if you want automatic removal of disturbing attributes, such as constants and singularities
    :param fitter: alternate the fitting algorithm (currently the Newton-Raphson fitting algorithm is used)
    :param stepwiseLR: set to 1 if you wish to use stepwise logistic regression
    :param addCrit: parameter for stepwise attribute selection
    :param deleteCrit: parameter for stepwise attribute selection
    :param numAttr: parameter for stepwise attribute selection
        
    """
    lr = LogRegLearnerClass(**kwds)
    if table:
        return lr(table, weightID)
    else:
        return lr

class LogRegLearnerClass(orange.Learner):
    def __init__(self, removeSingular=0, fitter = None, **kwds):
        self.__dict__.update(kwds)
        self.removeSingular = removeSingular
        self.fitter = None

    def __call__(self, examples, weight=0):
        imputer = getattr(self, "imputer", None) or None
        if getattr(self, "removeMissing", 0):
            examples = orange.Preprocessor_dropMissing(examples)
##        if hasDiscreteValues(examples.domain):
##            examples = createNoDiscTable(examples)
        if not len(examples):
            return None
        if getattr(self, "stepwiseLR", 0):
            addCrit = getattr(self, "addCrit", 0.2)
            removeCrit = getattr(self, "removeCrit", 0.3)
            numAttr = getattr(self, "numAttr", -1)
            attributes = StepWiseFSS(examples, addCrit = addCrit, deleteCrit = removeCrit, imputer = imputer, numAttr = numAttr)
            tmpDomain = orange.Domain(attributes, examples.domain.classVar)
            tmpDomain.addmetas(examples.domain.getmetas())
            examples = examples.select(tmpDomain)
        learner = orange.LogRegLearner()
        learner.imputerConstructor = imputer
        if imputer:
            examples = self.imputer(examples)(examples)
        examples = orange.Preprocessor_dropMissing(examples)
        if self.fitter:
            learner.fitter = self.fitter
        if self.removeSingular:
            lr = learner.fitModel(examples, weight)
        else:
            lr = learner(examples, weight)
        while isinstance(lr, orange.Variable):
            if isinstance(lr.getValueFrom, orange.ClassifierFromVar) and isinstance(lr.getValueFrom.transformer, orange.Discrete2Continuous):
                lr = lr.getValueFrom.variable
            attributes = examples.domain.attributes[:]
            if lr in attributes:
                attributes.remove(lr)
            else:
                attributes.remove(lr.getValueFrom.variable)
            newDomain = orange.Domain(attributes, examples.domain.classVar)
            newDomain.addmetas(examples.domain.getmetas())
            examples = examples.select(newDomain)
            lr = learner.fitModel(examples, weight)
        return lr



def Univariate_LogRegLearner(examples=None, **kwds):
    learner = apply(Univariate_LogRegLearner_Class, (), kwds)
    if examples:
        return learner(examples)
    else:
        return learner

class Univariate_LogRegLearner_Class(orange.Learner):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __call__(self, examples):
        examples = createFullNoDiscTable(examples)
        classifiers = map(lambda x: LogRegLearner(orange.Preprocessor_dropMissing(examples.select(orange.Domain(x, examples.domain.classVar)))), examples.domain.attributes)
        maj_classifier = LogRegLearner(orange.Preprocessor_dropMissing(examples.select(orange.Domain(examples.domain.classVar))))
        beta = [maj_classifier.beta[0]] + [x.beta[1] for x in classifiers]
        beta_se = [maj_classifier.beta_se[0]] + [x.beta_se[1] for x in classifiers]
        P = [maj_classifier.P[0]] + [x.P[1] for x in classifiers]
        wald_Z = [maj_classifier.wald_Z[0]] + [x.wald_Z[1] for x in classifiers]
        domain = examples.domain

        return Univariate_LogRegClassifier(beta = beta, beta_se = beta_se, P = P, wald_Z = wald_Z, domain = domain)

class Univariate_LogRegClassifier(orange.Classifier):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __call__(self, example, resultType = orange.GetValue):
        # classification not implemented yet. For now its use is only to provide regression coefficients and its statistics
        pass
    

def LogRegLearner_getPriors(examples = None, weightID=0, **kwds):
    lr = LogRegLearnerClass_getPriors(**kwds)
    if examples:
        return lr(examples, weightID)
    else:
        return lr

class LogRegLearnerClass_getPriors(object):
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
                if at.varType == orange.VarTypes.Continuous:
                    atDisc = orange.FloatVariable(at.name + "Disc")
                    newDomain = orange.Domain(data.domain.attributes+[atDisc,data.domain.classVar])
                    newDomain.addmetas(data.domain.getmetas())
                    newData = orange.ExampleTable(newDomain,data)
                    altData = orange.ExampleTable(newDomain,data)
                    for i,d in enumerate(newData):
                        d[atDisc] = 0
                        d[weightID] = 1*data[i][weightID]
                    for i,d in enumerate(altData):
                        d[atDisc] = 1
                        d[at] = 0
                        d[weightID] = 0.000001*data[i][weightID]
                elif at.varType == orange.VarTypes.Discrete:
                # v dataOrig, dataFinal in newData atributu "at" dodaj ee  eno  vreednost, ki ima vrednost kar  ime atributa +  "X"
                    atNew = orange.EnumVariable(at.name, values = at.values + [at.name+"X"])
                    newDomain = orange.Domain(filter(lambda x: x!=at, data.domain.attributes)+[atNew,data.domain.classVar])
                    newDomain.addmetas(data.domain.getmetas())
                    newData = orange.ExampleTable(newDomain,data)
                    altData = orange.ExampleTable(newDomain,data)
                    for i,d in enumerate(newData):
                        d[atNew] = data[i][at]
                        d[weightID] = 1*data[i][weightID]
                    for i,d in enumerate(altData):
                        d[atNew] = at.name+"X"
                        d[weightID] = 0.000001*data[i][weightID]
                newData.extend(altData)
                setsOfData.append(newData)
            return setsOfData
                  
        learner = LogRegLearner(imputer = orange.ImputerConstructor_average(), removeSingular = self.removeSingular)
        # get Original Model
        orig_model = learner(examples,weight)
        if orig_model.fit_status:
            print "Warning: model did not converge"

        # get extended Model (you should not change data)
        if weight == 0:
            weight = orange.newmetaid()
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
        bayes = orange.BayesLearner(examples)
        bayes_prior = math.log(bayes.distribution[1]/bayes.distribution[0])

        # normalize errors
##        print "bayes", bayes_prior
##        print "lr", orig_model.beta[0]
##        print "lr2", logistic_prior
##        print "dist", orange.Distribution(examples.domain.classVar,examples)
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

class LogRegLearnerClass_getPriors_OneTable:
    def __init__(self, removeSingular=0, **kwds):
        self.__dict__.update(kwds)
        self.removeSingular = removeSingular
    def __call__(self, examples, weight=0):
        # next function changes data set to a extended with unknown values 
        def createLogRegExampleTable(data, weightID):
            finalData = orange.ExampleTable(data)
            origData = orange.ExampleTable(data)
            for at in data.domain.attributes:
                # za vsak atribut kreiraj nov newExampleTable newData
                # v dataOrig, dataFinal in newData dodaj nov atribut -- continuous variable
                if at.varType == orange.VarTypes.Continuous:
                    atDisc = orange.FloatVariable(at.name + "Disc")
                    newDomain = orange.Domain(origData.domain.attributes+[atDisc,data.domain.classVar])
                    newDomain.addmetas(newData.domain.getmetas())
                    finalData = orange.ExampleTable(newDomain,finalData)
                    newData = orange.ExampleTable(newDomain,origData)
                    origData = orange.ExampleTable(newDomain,origData)
                    for d in origData:
                        d[atDisc] = 0
                    for d in finalData:
                        d[atDisc] = 0
                    for i,d in enumerate(newData):
                        d[atDisc] = 1
                        d[at] = 0
                        d[weightID] = 100*data[i][weightID]
                        
                elif at.varType == orange.VarTypes.Discrete:
                # v dataOrig, dataFinal in newData atributu "at" dodaj ee  eno  vreednost, ki ima vrednost kar  ime atributa +  "X"
                    atNew = orange.EnumVariable(at.name, values = at.values + [at.name+"X"])
                    newDomain = orange.Domain(filter(lambda x: x!=at, origData.domain.attributes)+[atNew,origData.domain.classVar])
                    newDomain.addmetas(origData.domain.getmetas())
                    temp_finalData = orange.ExampleTable(finalData)
                    finalData = orange.ExampleTable(newDomain,finalData)
                    newData = orange.ExampleTable(newDomain,origData)
                    temp_origData = orange.ExampleTable(origData)
                    origData = orange.ExampleTable(newDomain,origData)
                    for i,d in enumerate(origData):
                        d[atNew] = temp_origData[i][at]
                    for i,d in enumerate(finalData):
                        d[atNew] = temp_finalData[i][at]                        
                    for i,d in enumerate(newData):
                        d[atNew] = at.name+"X"
                        d[weightID] = 10*data[i][weightID]
                finalData.extend(newData)
            return finalData
                  
        learner = LogRegLearner(imputer = orange.ImputerConstructor_average(), removeSingular = self.removeSingular)
        # get Original Model
        orig_model = learner(examples,weight)

        # get extended Model (you should not change data)
        if weight == 0:
            weight = orange.newmetaid()
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
        bayes = orange.BayesLearner(examples)
        bayes_prior = math.log(bayes.distribution[1]/bayes.distribution[0])

        # normalize errors
        #print "bayes", bayes_prior
        #print "lr", orig_model.beta[0]
        #print "lr2", logistic_prior
        #print "dist", orange.Distribution(examples.domain.classVar,examples)
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

def Pr(x, betas):
    k = math.exp(dot(x, betas))
    return k / (1+k)

def lh(x,y,betas):
    llh = 0.0
    for i,x_i in enumerate(x):
        pr = Pr(x_i,betas)
        llh += y[i]*log(max(pr,1e-6)) + (1-y[i])*log(max(1-pr,1e-6))
    return llh


def diag(vector):
    mat = identity(len(vector), Float)
    for i,v in enumerate(vector):
        mat[i][i] = v
    return mat
    
class simpleFitter(orange.LogRegFitter):
    def __init__(self, penalty=0, se_penalty = False):
        self.penalty = penalty
        self.se_penalty = se_penalty
    def __call__(self, data, weight=0):
        ml = data.native(0)
        for i in range(len(data.domain.attributes)):
          a = data.domain.attributes[i]
          if a.varType == orange.VarTypes.Discrete:
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
            p = array([Pr(X[i], betas) for i in range(len(data))])
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
            p = array([Pr(X[i], betas) for i in range(len(data))])

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
##        yhat = array([Pr(X[i], betas) for i in range(len(data))])
##        ss = sum((y - yhat) ** 2) / (N - len(data.domain.attributes) - 1)
##        sigma = math.sqrt(ss)
        p = array([Pr(X[i], betas) for i in range(len(data))])
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

def Pr_bx(bx):
    if bx > 35:
        return 1
    if bx < -35:
        return 0
    return exp(bx)/(1+exp(bx))

class bayesianFitter(orange.LogRegFitter):
    def __init__(self, penalty=0, anch_examples=[], tau = 0):
        self.penalty = penalty
        self.anch_examples = anch_examples
        self.tau = tau

    def createArrayData(self,data):
        if not len(data):
            return (array([]),array([]))
        # convert data to numeric
        ml = data.native(0)
        for i,a in enumerate(data.domain.attributes):
          if a.varType == orange.VarTypes.Discrete:
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
        (X,y)=self.createArrayData(data)

        exTable = orange.ExampleTable(data.domain)
        for id,ex in self.anch_examples:
            exTable.extend(orange.ExampleTable(ex,data.domain))
        (X_anch,y_anch)=self.createArrayData(exTable)

        betas = array([0.0] * (len(data.domain.attributes)+1))

        likelihood,betas = self.estimateBeta(X,y,betas,[0]*(len(betas)),X_anch,y_anch)

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
            likelihood_temp,betas_temp=self.estimateBeta(X_temp,y,concatenate((betas[:ag[0]+1],betas[ag[0]+ag[1]+1:])),[0]+[1]*(len(betas)-1-ag[1]),X_anch_temp,y_anch)
            print "finBetas", betas, betas_temp
            print "betas", betas[0], betas_temp[0]
            sumB += betas[0]-betas_temp[0]
        apriori = orange.Distribution(data.domain.classVar, data)
        aprioriProb = apriori[0]/apriori.abs
        
        print "koncni rezultat", sumB, math.log((1-aprioriProb)/aprioriProb), betas[0]
            
        beta = []
        beta_se = []
        print "likelihood2", likelihood
        for i in range(len(betas)):
            beta.append(betas[i])
            beta_se.append(0.0)
        return (self.OK, beta, beta_se, 0)

     
        
    def estimateBeta(self,X,y,betas,const_betas,X_anch,y_anch):
        N,N_anch = len(y),len(y_anch)
        r,r_anch = array([dot(X[i], betas) for i in range(N)]),\
                   array([dot(X_anch[i], betas) for i in range(N_anch)])
        p    = array([Pr_bx(ri) for ri in r])
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
                    dl += self.penalty*x[j]*(y_anch[xi] - Pr_bx(r_anch[xi]*self.penalty))

                ddl = matrixmultiply(X_sq[:,j],transpose(p*(1-p)))
                for xi,x in enumerate(X_anch):
                    ddl += self.penalty*x[j]*Pr_bx(r[xi]*self.penalty)*(1-Pr_bx(r[xi]*self.penalty))

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
                p = array([Pr_bx(ri) for ri in r])
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
####  Feature subset selection for logistic regression  ####
############################################################


def StepWiseFSS(examples = None, **kwds):
    """
    If examples are specified, stepwise logistic regression implemented
    in stepWiseFSS_class is performed and a list of chosen attributes
    is returned. If examples are not specified an instance of
    stepWiseFSS_class with all parameters set is returned. Parameters
    addCrit, deleteCrit and numAttr are explained in the description
    of stepWiseFSS_class.

    :param examples: data set
    :type examples: Orange.data.table

    :param addCrit: "Alpha" level to judge if variable has enough importance to be added in the new set. (e.g. if addCrit is 0.2, then attribute is added if its P is lower than 0.2)
    :type addCrit: float

    :param deleteCrit: Similar to addCrit, just that it is used at backward elimination. It should be higher than addCrit!
    :type deleteCrit: float

    :param numAttr: maximum number of selected attributes, use -1 for infinity
    :type numAttr: int

    """

    """
      Constructs and returns a new set of examples that includes a
      class and attributes selected by stepwise logistic regression. This is an
      implementation of algorithm described in [Hosmer and Lemeshow, Applied Logistic Regression, 2000]

      examples: data set (ExampleTable)     
      addCrit: "Alpha" level to judge if variable has enough importance to be added in the new set. (e.g. if addCrit is 0.2, then attribute is added if its P is lower than 0.2)
      deleteCrit: Similar to addCrit, just that it is used at backward elimination. It should be higher than addCrit!
      numAttr: maximum number of selected attributes, use -1 for infinity

    """

    fss = apply(StepWiseFSS_class, (), kwds)
    if examples is not None:
        return fss(examples)
    else:
        return fss

def getLikelihood(fitter, examples):
    res = fitter(examples)
    if res[0] in [fitter.OK]: #, fitter.Infinity, fitter.Divergence]:
       status, beta, beta_se, likelihood = res
       if sum([abs(b) for b in beta])<sum([abs(b) for b in beta_se]):
           return -100*len(examples)
       return likelihood
    else:
       return -100*len(examples)
        


class StepWiseFSS_class(orange.Learner):
  """ Performs stepwise logistic regression and returns a list of "most"
  informative attributes. Each step of this algorithm is composed
  of two parts. First is backward elimination, where each already
  chosen attribute is tested for significant contribution to overall
  model. If worst among all tested attributes has higher significance
  that is specified in deleteCrit, this attribute is removed from
  the model. The second step is forward selection, which is similar
  to backward elimination. It loops through all attributes that are
  not in model and tests whether they contribute to common model with
  significance lower that addCrit. Algorithm stops when no attribute
  in model is to be removed and no attribute out of the model is to
  be added. By setting numAttr larger than -1, algorithm will stop its
  execution when number of attributes in model will exceed that number.

  Significances are assesed via the likelihood ration chi-square
  test. Normal F test is not appropriate, because errors are assumed to
  follow a binomial distribution.

  """

  def __init__(self, addCrit=0.2, deleteCrit=0.3, numAttr = -1, **kwds):
    self.__dict__.update(kwds)
    self.addCrit = addCrit
    self.deleteCrit = deleteCrit
    self.numAttr = numAttr
  def __call__(self, examples):
    if getattr(self, "imputer", 0):
        examples = self.imputer(examples)(examples)
    if getattr(self, "removeMissing", 0):
        examples = orange.Preprocessor_dropMissing(examples)
    continuizer = orange.DomainContinuizer(zeroBased=1,continuousTreatment=orange.DomainContinuizer.Leave,
                                           multinomialTreatment = orange.DomainContinuizer.FrequentIsBase,
                                           classTreatment = orange.DomainContinuizer.Ignore)
    attr = []
    remain_attr = examples.domain.attributes[:]

    # get LL for Majority Learner 
    tempDomain = orange.Domain(attr,examples.domain.classVar)
    #tempData  = orange.Preprocessor_dropMissing(examples.select(tempDomain))
    tempData  = orange.Preprocessor_dropMissing(examples.select(tempDomain))

    ll_Old = getLikelihood(orange.LogRegFitter_Cholesky(), tempData)
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
                tempDomain = orange.Domain(tempAttr,examples.domain.classVar)
                tempDomain.addmetas(examples.domain.getmetas())
                # domain, calculate P for LL improvement.
                tempDomain  = continuizer(orange.Preprocessor_dropMissing(examples.select(tempDomain)))
                tempData = orange.Preprocessor_dropMissing(examples.select(tempDomain))

                ll_Delete = getLikelihood(orange.LogRegFitter_Cholesky(), tempData)
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
            
            if worstAt.varType==orange.VarTypes.Continuous:
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
        if self.numAttr>-1 and len(attr)>=self.numAttr:
            remain_attr=[]
         
        # for each attribute in the remaining
        maxG=-1
        ll_Best = ll_Old
        length_Best = length_Old
        bestAt = None
        for at in remain_attr:
            tempAttr = attr + [at]
            tempDomain = orange.Domain(tempAttr,examples.domain.classVar)
            tempDomain.addmetas(examples.domain.getmetas())
            # domain, calculate P for LL improvement.
            tempDomain  = continuizer(orange.Preprocessor_dropMissing(examples.select(tempDomain)))
            tempData = orange.Preprocessor_dropMissing(examples.select(tempDomain))
            ll_New = getLikelihood(orange.LogRegFitter_Cholesky(), tempData)

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
        
        if bestAt.varType==orange.VarTypes.Continuous:
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


def StepWiseFSS_Filter(examples = None, **kwds):
    """
        check function StepWiseFSS()
    """

    filter = apply(StepWiseFSS_Filter_class, (), kwds)
    if examples is not None:
        return filter(examples)
    else:
        return filter


class StepWiseFSS_Filter_class(object):
    def __init__(self, addCrit=0.2, deleteCrit=0.3, numAttr = -1):
        self.addCrit = addCrit
        self.deleteCrit = deleteCrit
        self.numAttr = numAttr
    def __call__(self, examples):
        attr = StepWiseFSS(examples, addCrit=self.addCrit, deleteCrit = self.deleteCrit, numAttr = self.numAttr)
        return examples.select(orange.Domain(attr, examples.domain.classVar))
                

####################################
####  PROBABILITY CALCULATIONS  ####
####################################

def lchisqprob(chisq,df):
    """
Returns the (1-tailed) probability value associated with the provided
chi-square value and df.  Adapted from chisq.c in Gary Perlman's |Stat.

Usage:   lchisqprob(chisq,df)
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
Thus, 
    for z<0, zprob(z) = 1-tail probability
    for z>0, 1.0-zprob(z) = 1-tail probability
    for any z, 2.0*(1.0-zprob(abs(z))) = 2-tail probability
Adapted from z.c in Gary Perlman's |Stat.

Usage:   lzprob(z)
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

   
