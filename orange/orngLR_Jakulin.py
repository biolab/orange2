# ORANGE Logistic Regression
#    by Alex Jakulin (jakulin@acm.org)
#
#       based on:
#           Miller, A.J. (1992):
#           Algorithm AS 274: Least squares routines to supplement
#                those of Gentleman.  Appl. Statist., vol.41(2), 458-478.
#
#       and Alan Miller's F90 logistic regression code
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# CVS Status: $Id$
#
# Version 1.8 (14/05/2004)
#   - Normalized calibration
#
# Version 1.7 (11/08/2002)
#   - Support for new Orange (RandomIndices)
#   - Assertion error was resulting because Orange incorrectly compared the
#     attribute values as returned from .getclass() and the values array in
#     the attribute definition. Cast both values to integer before comparison
#   - Extended error messages.
#   - Support for setting the preferred ordinal attribute transformation.
#   - Prevent divide-by-0 errors in discriminant code.
#
# Version 1.6 (31/10/2001)
#
# The procedure used by this implementation of LR is minimization of log-likelihood
# via deviance testing.
#
#
# To Do:
#   - Trivial imputation is used for missing attribute values. You should use
#     something better. Eventually, this code might do something automatically.
#
#   - MarginMetaLearner is not doing anything when the classifier outputs a
#     probability distribution. But even in such cases, the PD could be corrected.
#
#   - CalibrationMetaLearner  -> calibrates the *PD* estimates via CV

import orange
import orng2Array
import orngCRS
import math


# BEWARE: these routines do not work with orange tables and are not orange-compatible
class BLogisticLearner(orange.Learner):
    def getmodel(self, examples):
      errors = ["LogReg: ngroups < 2, ndf < 0 -- not enough examples with so many attributes",
                "LogReg: n[i]<0",
                "LogReg: r[i]<0",
                "LogReg: r[i]>n[i]",
                "LogReg: constant variable",
                "LogReg: singularity",
                "LogReg: infinity in beta",
                "LogReg: no convergence"]
      model = orngCRS.LogReg(examples)
      errorno = model[8]
      if errorno == 5 or errorno == 6:
        # dependencies between variables, remove them
        raise RedundanceException(model[9])
      else:
        if errorno != 0 and errorno != 7:
            # unhandled exception
            raise errors[errorno-1]
      return (model,errorno)
        
    def __call__(self, examples):
      (model,errorno) = self.getmodel(examples)
      if errorno == 7:
        # there exists a perfect discriminant
        return BDiscriminantClassifier(model, examples)
      else:
        return BLogisticClassifier(model)


class BLogisticClassifier(orange.Classifier):
    def __init__(self, model):
        (self.chisq,self.devnce,self.ndf,self.beta,
        self.se_beta,self.fit,self.stdres,
        self.covbeta,errorno,masking) = model
        
    def getmargin(self,example):
        # returns the actual probability which is not to be fudged with
        return self.__call__(example)[1]

    def description(self,attnames,classname,classv):
        print 'Logistic Regression Report'
        print '=========================='
        print '\chi^2',self.chisq
        print 'deviance',self.devnce
        print 'NDF',self.ndf
        print
        print 'Base outcome:',classname[0],'=',classv
        assert(len(attnames)==len(self.beta)-1)
        print 'beta_0:',self.beta[0],'+-',self.se_beta[0]
        for i in range(len(attnames)):
            print attnames[i],self.beta[i+1],'+-',self.se_beta[i+1]

    def __call__(self,example):
        # logistic regression
        sum = self.beta[0]
        for i in range(len(self.beta)-1):
            sum = sum + example[i]*self.beta[i+1]
        # print sum, example
        if sum > 10000:
            return (1,1.0)
        elif sum < -10000:
            return (0,1.0)
        else:
            sum = math.exp(sum)
            p = sum/(1.0+sum) # probability that the class is 1
            if p < 0.5:
                return (0,1-p)
            else:
                return (1,p)


class BDiscriminantClassifier(BLogisticClassifier):
    def __init__(self, model, examples):
        (self.chisq,self.devnce,self.ndf,self.beta,
        self.se_beta,self.fit,self.stdres,
        self.covbeta,errorno,masking) = model

        # set up the parameters for discrimination
        sum = 1.0
        for i in self.beta[1:]:
            if abs(i) > 1e-6:
                sum *= abs(i)
        if sum > 1e100:
            sum = max(self.beta[1:])
        if sum < 1e-6:
            sum = 1e-6
        scale = math.sqrt(sum)
        self.nbeta = [x/scale for x in self.beta]

    def getmargin(self,example):
        sum = self.nbeta[0]
        for i in range(len(self.nbeta)-1):
            sum = sum + example[i]*self.nbeta[i+1]
        return sum

    def __call__(self, example):
        sum = self.getmargin(example)
        # linear discriminant
        if sum < 0.0:
            return (0,1.0)
        else:
            return (1,1.0)


class RedundanceException:
  def __init__(self,redundant_vars):
    self.redundant_vars = redundant_vars

  def __str__(self):
    return "Logistic regression cannot work with constant or linearly dependent variables."



#
# Logistic regression throws an exception upon constant or linearly
# dependent attributes. RobustBLogisticLearner remembers to ignore
# such attributes.
#
# returns None, if all attributes singular
#
class RobustBLogisticLearner(BLogisticLearner):
    def __call__(self, examples):
        skipping = 0
        na = len(examples[0])
        mask = [0]*na
        assert(na > 0)
        # while there are any unmasked variables
        while skipping < na-1: 
            try:
                if skipping != 0:
                    # remove some variables
                    data = []
                    for ex in examples:
                        maskv = []
                        for i in range(len(mask)):
                            if mask[i] == 0:
                                maskv.append(ex[i])
                        data.append(maskv)
                else:
                    data = examples
                classifier = BLogisticLearner.__call__(self,data)
                return RobustBLogisticClassifierWrap(classifier,mask)
            except RedundanceException, exp:
                ext_offs = 0 # offset in the existing mask
                for i in exp.redundant_vars:
                    # skip what's already masked
                    while mask[ext_offs] == 1:
                        ext_offs += 1
                    if i != 0:
                        # new masking
                        mask[ext_offs] = 1
                        skipping += 1
                    ext_offs += 1


# this wrapper transforms the example
#
# it is a wrapper, because it has to work with both
# the discriminant and the LR
class RobustBLogisticClassifierWrap(orange.Classifier):
    def __init__(self, classifier, mask):
        self.classifier = classifier
        self.mask = mask

    def translate(self,example):
        assert(len(example) == len(self.mask) or len(example) == len(self.mask)-1) # note that for classification, the class isn't defined
        maskv = []
        for i in range(len(example)):
            if self.mask[i] == 0:
                maskv.append(example[i])
        return maskv

    def description(self,variablenames,n):
        maskv = []
        for i in range(len(variablenames[0])):
            if self.mask[i] == 0:
                maskv.append(variablenames[0][i])
        self.classifier.description(maskv,variablenames[1],n)

    def getmargin(self, example):
        return self.classifier.getmargin(self.translate(example))

    def __call__(self, example):
        return self.classifier(self.translate(example))


#
# Logistic regression works with arrays and not Orange domains
# This wrapper performs the domain translation
#
class BasicLogisticLearner(RobustBLogisticLearner):
    def __init__(self):
        self.translation_mode = 0 # dummy

    def __call__(self, examples, weight = 0):
        if not(examples.domain.classVar.varType == 1 and len(examples.domain.classVar.values)==2):
            for i in examples.domain.classVar.values:
                print i
            raise "Logistic learner only works with binary discrete class."
        translate = orng2Array.DomainTranslation(self.translation_mode)
        translate.analyse(examples, weight)
        translate.prepareLR()
        mdata = translate.transform(examples)
        r = RobustBLogisticLearner.__call__(self,mdata)
        if r == None:
            if weight != 0:
                return orange.MajorityLearner()(examples, weight)
            else:
                return orange.MajorityLearner()(examples)
        else:
            return BasicLogisticClassifier(r,translate)


class BasicLogisticClassifier(orange.Classifier):
    def __init__(self, classifier, translator):
        self.classifier = classifier
        self.translator = translator
        self._name = 'Basic Logistic Classifier'        

    def getmargin(self,example):
        tex = self.translator.extransform(example)
        r = self.classifier.getmargin(tex)
        return r

    def description(self):
        self.classifier.description(self.translator.description(),self.translator.cv.attr.values[1])

    def __call__(self, example, format = orange.GetValue):
        tex = self.translator.extransform(example)
        r = self.classifier(tex)
        #print example, tex, r
        v = self.translator.getClass(r[0])
        p = [0.0,0.0]
        for i in range(2):
            if int(v) == i:
                p[i] = r[1]
                p[1-i] = 1-r[1]
                break
        if format == orange.GetValue:
            return v
        if format == orange.GetBoth:
            return (v,p)
        if format == orange.GetProbabilities:
            return p


#
# A margin-based Bayesian learner
#
class BasicBayesLearner(orange.Learner):
    def _safeRatio(self,a,b):
        if a*10000.0 < b:
            return -10
        elif b*10000.0 < a:
            return 10
        else:
            return math.log(a)-math.log(b)

    def _process(self,classifier,examples):
        # todo todo - support for loess
        beta = self._safeRatio(classifier.distribution[1],classifier.distribution[0])
        coeffs = []
        for i in range(len(examples.domain.attributes)):
            for j in range(len(examples.domain.attributes[i].values)):
                p1 = classifier.conditionalDistributions[i][j][1]
                p0 = classifier.conditionalDistributions[i][j][0]
                coeffs.append(self._safeRatio(p1,p0)-beta)
        return (beta, coeffs)

    
    def __init__(self):
        self.translation_mode = 1 # binarization

    def __call__(self, examples, weight = 0):
        if not(examples.domain.classVar.varType == 1 and len(examples.domain.classVar.values)==2):
            raise "BasicBayes learner only works with binary discrete class."
        for attr in examples.domain.attributes:
            if not(attr.varType == 1):
                raise "BasicBayes learner does not work with continuous attributes."
        translate = orng2Array.DomainTranslation(self.translation_mode)
        translate.analyse(examples, weight)
        translate.prepareLR()
        (beta, coeffs) = self._process(orange.BayesLearner(examples), examples)
        return BasicBayesClassifier(beta,coeffs,translate)


class BasicBayesClassifier(orange.Classifier):
    def __init__(self, beta, coeffs, translator):
        self.beta = beta
        self.coeffs = coeffs
        self.translator = translator
        self._name = 'Basic Bayes Classifier'        

    def getmargin(self,example):
        tex = self.translator.extransform(example)
        sum = self.beta
        for i in xrange(len(self.coeffs)):
            sum += tex[i]*self.coeffs[i]
        return -sum

    def __call__(self, example, format = orange.GetValue):
        sum = -self.getmargin(example)

        # print sum, example
        if sum > 10000:
            r = (1,1.0)
        elif sum < -10000:
            r = (0,1.0)
        else:
            sum = math.exp(sum)
            p = sum/(1.0+sum) # probability that the class is 1
            if p < 0.5:
                r = (0,1-p)
            else:
                r = (1,p)

        v = self.translator.getClass(r[0])
        p = [0.0,0.0]
        for i in range(2):
            if int(v) == i:
                p[i] = r[1]
                p[1-i] = 1-r[1]
                break
        if format == orange.GetValue:
            return v
        if format == orange.GetBoth:
            return (v,p)
        if format == orange.GetProbabilities:
            return p


class BasicCalibrationLearner(orange.Learner):
    def __init__(self, discr = orange.EntropyDiscretization(), learnr = orange.BayesLearner()):
        self.disc = discr
        self.learner = learnr

    def __call__(self, examples):
        if not(examples.domain.classVar.varType == 1 and len(examples.domain.classVar.values)==2):
            raise "BasicCalibration learner only works with binary discrete class."
        if len(examples.domain.attributes) > 1 or examples.domain.attributes[0].varType != 2:
            raise "BasicCalibration learner only works with a single numerical attribute."

        new_a = self.disc(examples.domain.attributes[0],examples)
        data = examples.select([new_a, examples.domain.classVar])
        c = self.learner(data)

        return BasicCalibrationClassifier(c)

class BasicCalibrationClassifier(orange.Classifier):
    def __init__(self, classifier):
        self.classifier = classifier

    def __call__(self, example, format = orange.GetValue):
        return self.classifier(example,format)
        
#
# Margin Probability Wrap
#
# Margin metalearner attempts to use the margin-based classifiers, such as linear
# discriminants and SVM to return the class probability distribution. Thie metalearner
# only works with binary classes.
#
# Margin classifiers output the distance from the separating hyperplane, not
# the probability distribution. However, the distance from the hyperplane can
# be associated with the probability. This is a regression problem, for which
# we can apply logistic regression.
#
# However, one must note that perfect separating hyperplanes generate trivial
# class distributions. 
#
class MarginMetaLearner(orange.Learner):
    def __init__(self, learner, folds = 10, replications = 1, metalearner = BasicLogisticLearner()):
        self.learner = learner
        self.folds = 10
        self.metalearner = metalearner
        self.replications = replications
        
    def __call__(self, examples, weight = 0):
        if not(examples.domain.classVar.varType == 1 and len(examples.domain.classVar.values)==2):
            raise "Margin metalearner only works with binary discrete class."

        mv = orange.FloatVariable(name="margin")
        estdomain = orange.Domain([mv,examples.domain.classVar])
        mistakes = orange.ExampleTable(estdomain)
        if weight != 0:
            mistakes.addMetaAttribute(1)

        for replication in range(self.replications):
            # perform 10 fold CV, and create a new dataset
            try:
                selection = orange.MakeRandomIndicesCV(examples, self.folds, stratified=0, randomGenerator = orange.globalRandom) # orange 2.2
            except:
                selection = orange.RandomIndicesCVGen(examples, self.folds) # orange 2.1
            for fold in range(self.folds):
              if self.folds != 1: # no folds
                  learn_data = examples.selectref(selection, fold, negate=1)
                  test_data  = examples.selectref(selection, fold)
              else:
                  learn_data = examples
                  test_data  = examples
                  

              if weight!=0:
                  classifier = self.learner(learn_data, weight=weight)
              else:
                  classifier = self.learner(learn_data)
              # normalize the range              
              mi = 1e100
              ma = -1e100
              for ex in learn_data:
                  margin = classifier.getmargin(ex)
                  mi = min(mi,margin)
                  ma = max(ma,margin)
              coeff = 1.0/max(ma-mi,1e-16)
              for ex in test_data:
                  margin = coeff*classifier.getmargin(ex)
                  if type(margin)==type(1.0) or type(margin)==type(1):
                      # ignore those examples which are handled with
                      # the actual probability distribution
                      mistake = orange.Example(estdomain,[float(margin), ex.getclass()])
                      if weight!=0:
                          mistake.setmeta(ex.getMetaAttribute(weight),1)
                      mistakes.append(mistake)

        if len(mistakes) < 1:
            # nothing to learn from
            if weight == 0:
                return self.learner(examples)
            else:
                return self.learner(examples,weight)
        if weight != 0:
            # learn a classifier to estimate the probabilities from margins
            # learn a classifier for the whole training set
            estimate = self.metalearner(mistakes, weight = 1)
            classifier = self.learner(examples, weight)
        else:
            estimate = self.metalearner(mistakes)
            classifier = self.learner(examples)

        # normalize the range              
        mi = 1e100
        ma = -1e100
        for ex in examples:
            margin = classifier.getmargin(ex)
            mi = min(mi,margin)
            ma = max(ma,margin)
        coeff = 1.0/max(ma-mi,1e-16)
        #print estimate.classifier.classifier
        #for x in mistakes:
        #    print x,estimate(x,orange.GetBoth)

        return MarginMetaClassifier(classifier, estimate, examples.domain, estdomain, coeff)


class MarginMetaClassifier(orange.Classifier):
    def __init__(self, classifier, estimator, domain, estdomain, coeff):
        self.classifier = classifier
        self.coeff = coeff
        self.estimator = estimator
        self.domain = domain
        self.estdomain = estdomain
        self.cv = self.estdomain.classVar(0)
        self._name = 'MarginMetaClassifier'

    def __call__(self, example, format = orange.GetValue):
        r = self.coeff*self.classifier.getmargin(example)
        if type(r) == type(1.0) or type(r) == type(1):
            # got a margin
            ex = orange.Example(self.estdomain,[r,self.cv]) # need a dummy class value
            (v,p) = self.estimator(ex,orange.GetBoth)
        else:
            # got a probability distribution, which can happen with LR... easy
            (v,p) = r

        if format == orange.GetValue:
            return v
        if format == orange.GetBoth:
            return (v,p)
        if format == orange.GetProbabilities:
            return p
        