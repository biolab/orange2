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

MAX_EXP = 50

# BEWARE: these routines do not work with orange tables and are not orange-compatible
class BLogisticLearner(orange.Learner):
    def __init__(self, regularization = 0.0):
        self.regul = regularization
        
    def getmodel(self, examples):
      errors = ["LogReg: ngroups < 2, ndf < 0 -- not enough examples with so many attributes",
                "LogReg: n[i]<0",
                "LogReg: r[i]<0",
                "LogReg: r[i]>n[i]",
                "LogReg: constant variable",
                "LogReg: singularity",
                "LogReg: infinity in beta",
                "LogReg: no convergence"]
      model = orngCRS.LogReg(examples,self.regul)
      errorno = model[8]
      #print errors[errorno-1]
      if errorno == 5 or errorno == 6:
        # dependencies between variables, remove them
        raise RedundanceException(model[9])
      elif errorno == 1:
        raise TooManyAttributes()
      else:
        if errorno != 0 and errorno != 7:
            # unhandled exception (0=all ok, 7=perfect separation)
            raise errors[errorno-1]
      return (model,errorno)
        
    def __call__(self, examples):
      (model,errorno) = self.getmodel(examples)
      return BLogisticClassifier(model,examples)


class BLogisticClassifier(orange.Classifier):
    def __init__(self, model,examples):
        (self.chisq,self.devnce,self.ndf,self.beta,
        self.se_beta,self.fit,self.covbeta,
        self.stdres,errorno,masking) = model

        # set up the parameters for discrimination
        sum = 1.0
        for i in self.beta[1:]:
            if abs(i) > 1e-6:
                sum *= abs(i)
        if sum > 1e100:
            sum = max(self.beta[1:])
        if sum < 1e-6:
            sum = 1e-6
        scale = 1.0/math.sqrt(sum)
        self.nbeta = [x*scale for x in self.beta]
        self.prior = 1.0/(len(examples)+2)
        self.iprior = 1-(2*self.prior)
                
    def getmargin(self,example):
        sum = self.nbeta[0]
        for i in xrange(len(self.nbeta)-1):
            sum = sum + example[i]*self.nbeta[i+1]
        return sum

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
        for i in xrange(len(attnames)):
            print attnames[i],self.beta[i+1],'+-',self.se_beta[i+1]

    def geteffects(self,example):
        # logistic regression
        sum = self.beta[0]
        for i in xrange(len(self.beta)-1):
            sum = sum + example[i]*self.beta[i+1]
        return sum

    def __call__(self,example):
        sum = self.geteffects(example)
        # print sum, example
        if sum > MAX_EXP:
            return (1,1-self.prior)
        elif sum < -MAX_EXP:
            return (0,1-self.prior)
        else:
            sum = math.exp(sum)
            p = sum/(1.0+sum) # probability that the class is 1
            if p < 0.5:
                return (0,1-self.prior-self.iprior*p)
            else:
                return (1,self.prior+self.iprior*p)

class MajorityLogClassifier(orange.Classifier):
    def __init__(self,ratio,lent):
        self.beta = [math.log(ratio/(1-ratio))]
        self.nbeta = [1]
        self.ratio = ratio
        self.prior = 1.0/(lent+2)
        self.iprior = 1-(2*self.prior)
        if ratio>0.5:
            self.o = 1
        else:
            self.o = 0
            self.ratio = 1-self.ratio
        self.ratio = self.prior+self.iprior*self.ratio

    def geteffects(self,example):
        return self.beta[0]

    def getmargin(self,example):
        return 0
        
    def __call__(self,example):
        return (self.o,self.ratio)
        

class TooManyAttributes:
  def __init__(self):
    pass

  def __str__(self):
    return "Too many variables."

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
    def __init__(self, regularization=0.0):
        self.regularization = regularization
        
    def __call__(self, examples, translate, importances, classfreq):
        assert(len(classfreq)==2)
        skipping = 0
        na = len(examples[0])
        mask = [0]*na
        last_importance = 0
        assert(na > 0)
        # while there are any unmasked variables
        blearner = BLogisticLearner(self.regularization)
        while skipping < na-1: 
            try:
                if skipping != 0:
                    # remove some variables
                    data = []
                    for ex in examples:
                        maskv = []
                        for i in xrange(len(mask)):
                            if mask[i] == 0:
                                maskv.append(ex[i])
                        data.append(maskv)
                else:
                    data = examples
                classifier = blearner(data)
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
                #print 'r.skipping',skipping
            except TooManyAttributes:
                # remove something
                removed = 0
                while removed == 0 or skipping < na-1:
                    tokill = importances[last_importance][1]
                    i = translate.trans[tokill].idx
                    while i < translate.trans[tokill].nidx:
                        if mask[i] == 0: # if not deleted already
                            removed += 1
                            mask[i] = 1
                            skipping += 1
                        i += 1
                    last_importance += 1
                #print 'i.skipping',skipping
        return RobustBLogisticClassifierWrap(MajorityLogClassifier(classfreq[1],len(examples)),mask)

# this wrapper transforms the example
#
# it is a wrapper, because it has to work with both
# the discriminant and the LR
class RobustBLogisticClassifierWrap(orange.Classifier):
    def __init__(self, classifier, mask):
        self.classifier = classifier
        self.mask = mask

    def translate(self,example):
        #assert(len(example) == len(self.mask) or len(example) == len(self.mask)-1) # note that for classification, the class isn't defined
        maskv = []
        for i in xrange(len(example)):
            if self.mask[i] == 0:
                maskv.append(example[i])
        return maskv

    def description(self,variablenames,n):
        maskv = []
        for i in xrange(len(variablenames[0])):
            if self.mask[i] == 0:
                maskv.append(variablenames[0][i])
        self.classifier.description(maskv,variablenames[1],n)

    def geteffects(self, example):
        return self.classifier.geteffects(self.translate(example))

    def getmargin(self, example):
        return self.classifier.getmargin(self.translate(example))

    def __call__(self, example):
        return self.classifier(self.translate(example))


#
# Logistic regression works with arrays and not Orange domains
# This wrapper performs the domain translation
#
class BasicLogisticLearner(RobustBLogisticLearner):
    def __init__(self,regularization = 0.01):
        self.translation_mode_d = 0 # dummy
        self.translation_mode_c = 1 # standardize
        self.regularization = regularization

    def __call__(self, examples, weight = 0,fulldata=0):
        if examples.domain.classVar.varType != 1:
            raise "Logistic learner only works with discrete class."
        translate = orng2Array.DomainTranslation(self.translation_mode_d,self.translation_mode_c)
        if fulldata != 0:
            translate.analyse(fulldata, weight, warning=0)
        else:
            translate.analyse(examples, weight, warning=0)
        translate.prepareLR()
        mdata = translate.transform(examples)

        # get the attribute importances
        t = examples
        importance = []
        for i in xrange(len(t.domain.attributes)):
            qi = orange.MeasureAttribute_relief(t.domain.attributes[i],t)
            importance.append((qi,i))
        importance.sort()
        freqs = list(orange.Distribution(examples.domain.classVar,examples))
        s = 1.0/sum(freqs)
        freqs = [x*s for x in freqs] # normalize

        rl = RobustBLogisticLearner(regularization=self.regularization)
        if len(examples.domain.classVar.values) > 2:
            ## form several experiments:
            # identify the most frequent class value
            tfreqs = [(freqs[i],i) for i in xrange(len(freqs))]
            tfreqs.sort()
            base = tfreqs[-1][1] # the most frequent class
            classifiers = []
            for i in xrange(len(tfreqs)-1):
                # edit the translation
                alter = tfreqs[i][1]
                cfreqs = [tfreqs[-1][0],tfreqs[i][0]] # 0=base,1=alternative
                # edit all the examples
                for j in xrange(len(mdata)):
                    c = int(examples[j].getclass())
                    if c==alter:
                        mdata[j][-1] = 1
                    else:
                        mdata[j][-1] = 0
                r = rl(mdata,translate,importance,cfreqs)
                classifiers.append(r)
            return ArrayLogisticClassifier(classifiers,translate,tfreqs,examples.domain.classVar,len(mdata))
        else:
            r = rl(mdata,translate,importance,freqs)
            return BasicLogisticClassifier(r,translate)


class ArrayLogisticClassifier(orange.Classifier):
    def __init__(self, classifiers, translator, tfreqs, classVar, allex):
        self.classifiers = classifiers
        self.translator = translator
        self.tfreqs = tfreqs
        self.classVar = classVar
        self.nc = len(classifiers)+1
        self.prior = 1.0/(self.nc+allex)
        self.iprior = 1.0-self.nc*self.prior
        assert(self.nc == len(classVar.values) and self.nc == len(tfreqs))
        self._name = 'Multiclass Logistic Classifier'

    def description(self):
        for x in self.classifiers:
            x.description(self.translator.description(),self.translator.cv.attr.values[1])

    def __call__(self, example, format = orange.GetValue):
        tex = self.translator.extransform(example)
        effects = []
        for i in xrange(self.nc-1):
            idx = self.tfreqs[i][1]
            effect = self.classifiers[i].geteffects(tex)
            if effect > MAX_EXP:
                effect = MAX_EXP
            elif effect < -MAX_EXP:
                effect = -MAX_EXP
            effects.append((idx,math.exp(effect)))
        tfreqs = self.tfreqs

        # aggregate the predictions
        p = [0.0 for i in xrange(self.nc)]
        sum = 0.0
        for (idx,effect) in effects:
            sum += effect
        sum += 1.0
        q = (self.iprior/sum)
        
        # prepare PDF
        p = [0.0 for i in xrange(self.nc)]
        psum = 0.0
        maxp = -1
        maxi = -1
        for (idx,effect) in effects:
            rr = self.prior + effect*q
            p[idx] = rr
            psum += rr
            if rr >= maxp: # most likely class
                maxi = idx
                maxp = rr
        p[tfreqs[-1][1]] = 1.0-psum # base class
        if 1-psum >= maxp:
            maxi = tfreqs[-1][1] # most likely class

        v = self.classVar(maxi)
        if format == orange.GetValue:
            return v
        if format == orange.GetBoth:
            return (v,p)
        if format == orange.GetProbabilities:
            return p

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
        for i in xrange(2):
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
        for i in xrange(len(examples.domain.attributes)):
            for j in xrange(len(examples.domain.attributes[i].values)):
                p1 = classifier.conditionalDistributions[i][j][1]
                p0 = classifier.conditionalDistributions[i][j][0]
                coeffs.append(self._safeRatio(p1,p0)-beta)
        return (beta, coeffs)

    
    def __init__(self):
        self.translation_mode_d = 1 # binarization
        self.translation_mode_c = 1 # standardization

    def __call__(self, examples, weight = 0,fulldata=0):
        if not(examples.domain.classVar.varType == 1 and len(examples.domain.classVar.values)==2):
            raise "BasicBayes learner only works with binary discrete class."
        for attr in examples.domain.attributes:
            if not(attr.varType == 1):
                raise "BasicBayes learner does not work with continuous attributes."
        translate = orng2Array.DomainTranslation(self.translation_mode_d,self.translation_mode_c)
        if fulldata != 0:
            translate.analyse(fulldata, weight)
        else:
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
        if sum > MAX_EXP:
            r = (1,1.0)
        elif sum < -MAX_EXP:
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
        for i in xrange(2):
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
    def __init__(self, learner, folds = 10, replications = 1, normalization=0, fulldata=0, metalearner = BasicLogisticLearner()):
        self.learner = learner
        self.folds = folds
        self.metalearner = metalearner
        self.replications = replications
        self.normalization = normalization
        self.fulldata = fulldata
        
    def __call__(self, examples, weight = 0):
        if not(examples.domain.classVar.varType == 1 and len(examples.domain.classVar.values)==2):
            # failing the assumptions of margin-metalearner...
            return MarginMetaClassifierWrap(self.learner(examples))

        mv = orange.FloatVariable(name="margin")
        estdomain = orange.Domain([mv,examples.domain.classVar])
        mistakes = orange.ExampleTable(estdomain)
        if weight != 0:
            mistakes.addMetaAttribute(1)

        for replication in xrange(self.replications):
            # perform 10 fold CV, and create a new dataset
            try:
                selection = orange.MakeRandomIndicesCV(examples, self.folds, stratified=0, randomGenerator = orange.globalRandom) # orange 2.2
            except:
                selection = orange.RandomIndicesCVGen(examples, self.folds) # orange 2.1
            for fold in xrange(self.folds):
              if self.folds != 1: # no folds
                  learn_data = examples.selectref(selection, fold, negate=1)
                  test_data  = examples.selectref(selection, fold)
              else:
                  learn_data = examples
                  test_data  = examples

              # fulldata removes the influence of scaling on the distance dispersion.                  
              if weight!=0:
                  if self.fulldata:
                      classifier = self.learner(learn_data, weight=weight, fulldata=examples)
                  else:
                      classifier = self.learner(learn_data, weight=weight)
              else:
                  if self.fulldata:
                      classifier = self.learner(learn_data, fulldata=examples)
                  else:
                      classifier = self.learner(learn_data)
              # normalize the range
              if self.normalization:
                  mi = 1e100
                  ma = -1e100
                  for ex in learn_data:
                      margin = classifier.getmargin(ex)
                      mi = min(mi,margin)
                      ma = max(ma,margin)
                  coeff = 1.0/max(ma-mi,1e-16)
              else:
                  coeff = 1.0  
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
        if self.normalization:
            mi = 1e100
            ma = -1e100
            for ex in examples:
                margin = classifier.getmargin(ex)
                mi = min(mi,margin)
                ma = max(ma,margin)
            coeff = 1.0/max(ma-mi,1e-16)
        else:
            coeff = 1.0
        #print estimate.classifier.classifier
        #for x in mistakes:
        #    print x,estimate(x,orange.GetBoth)

        return MarginMetaClassifier(classifier, estimate, examples.domain, estdomain, coeff)


class MarginMetaClassifierWrap(orange.Classifier):
    def __init__(self, classifier):
        self.classifier = classifier
        self._name = 'MarginMetaClassifier'

    def __call__(self, example, format = orange.GetValue):
	try:
		(v,p) = self.classifier(example,orange.GetBoth)
	except:
		v = self.classifier(example)
        if format == orange.GetValue:
            return v
        if format == orange.GetBoth:
            return (v,p)
        if format == orange.GetProbabilities:
            return p

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
        
