"""

.. index:: rule learning

Supervised rule induction algorithms and rule-based classification methods.

First, the commonly used CN2 learner's description is given. That is followed
by documentation of common classes and functions of the module.

===
CN2
===

.. index:: CN2

Several variations of well-known CN2 rule learning algorithms are implemented.
All are implemented by wrapping the
:class:`Orange.classification.rules.RuleLearner` class. Each CN2 learner class
in this module changes some of RuleLearner's replaceable components to reflect
the required behaviour.

Usage is consistent with typical learner usage in Orange::

    # Read some data
    instances =  Orange.data.Table("titanic")

    # construct the learning algorithm and use it to induce a classifier
    cn2_learner = Orange.classifier.rules.CN2Learner()
    cn2_clasifier = cn2_learner(data)

    # ... or, in a single step
    cn2_classifier = Orange.classifier.rules.CN2Learner(instances)

.. autoclass:: Orange.classification.rules.CN2Learner
   :members:
   :show-inheritance:
   
.. autoclass:: Orange.classification.rules.CN2Classifier
   :members:
   :show-inheritance:
   
.. index:: Unordered CN2
   
.. autoclass:: Orange.classification.rules.CN2UnorderedLearner
   :members:
   :show-inheritance:
   
.. autoclass:: Orange.classification.rules.CN2UnorderedClassifier
   :members:
   :show-inheritance:
   
.. index:: CN2-SD
.. index:: Subgroup discovery
   
.. autoclass:: Orange.classification.rules.CN2SDUnorderedLearner
   :members:
   :show-inheritance:
   
References
----------

* Clark, Niblett. `The CN2 Induction Algorithm
  <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.53.9180>`_. Machine
  Learning 3(4):261--284, 1989.
* Clark, Boswell. `Rule Induction with CN2: Some Recent Improvements
  <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.24.1700>`_. In
  Machine Learning - EWSL-91. Proceedings of the European Working Session on
  Learning., pages 151--163, Porto, Portugal, March 1991.
* Lavrac, Kavsek, Flach, Todorovski: `Subgroup Discovery with CN2-SD
  <http://jmlr.csail.mit.edu/papers/volume5/lavrac04a/lavrac04a.pdf>`_. Journal
  of Machine Learning Research 5: 153-188, 2004.

===========
All the rest ...
===========

"""

from Orange.core import \
    AssociationClassifier, \
    AssociationLearner, \
    RuleClassifier, \
    RuleClassifier_firstRule, \
    RuleClassifier_logit, \
    RuleLearner, \
    Rule, \
    RuleBeamCandidateSelector, \
    RuleBeamCandidateSelector_TakeAll, \
    RuleBeamFilter, \
    RuleBeamFilter_Width, \
    RuleBeamInitializer, \
    RuleBeamInitializer_Default, \
    RuleBeamRefiner, \
    RuleBeamRefiner_Selector, \
    RuleClassifierConstructor, \
    RuleCovererAndRemover, \
    RuleCovererAndRemover_Default, \
    RuleDataStoppingCriteria, \
    RuleDataStoppingCriteria_NoPositives, \
    RuleEvaluator, \
    RuleEvaluator_Entropy, \
    RuleEvaluator_LRS, \
    RuleEvaluator_Laplace, \
    RuleEvaluator_mEVC, \
    RuleFinder, \
    RuleBeamFinder, \
    RuleList, \
    RuleStoppingCriteria, \
    RuleStoppingCriteria_NegativeDistribution, \
    RuleValidator, \
    RuleValidator_LRS

import Orange.core
import random
import math
from orngABCN2 import ABCN2


class LaplaceEvaluator(RuleEvaluator):
    def __call__(self, rule, data, weightID, targetClass, apriori):
        if not rule.classDistribution:
            return 0.
        sumDist = rule.classDistribution.cases
        if not sumDist or (targetClass>-1 and not rule.classDistribution[targetClass]):
            return 0.
        # get distribution
        if targetClass>-1:
            return (rule.classDistribution[targetClass]+1)/(sumDist+2)
        else:
            return (max(rule.classDistribution)+1)/(sumDist+len(data.domain.classVar.values))


class WRACCEvaluator(RuleEvaluator):
    def __call__(self, rule, data, weightID, targetClass, apriori):
        if not rule.classDistribution:
            return 0.
        sumDist = rule.classDistribution.cases
        if not sumDist or (targetClass>-1 and not rule.classDistribution[targetClass]):
            return 0.
        # get distribution
        if targetClass>-1:
            pRule = rule.classDistribution[targetClass]/apriori[targetClass]
            pTruePositive = rule.classDistribution[targetClass]/sumDist
            pClass = apriori[targetClass]/apriori.cases
        else:
            pRule = sumDist/apriori.cases
            pTruePositive = max(rule.classDistribution)/sumDist
            pClass = apriori[rule.classDistribution.modus()]/sum(apriori)
        if pTruePositive>pClass:
            return pRule*(pTruePositive-pClass)
        else: return (pTruePositive-pClass)/max(pRule,1e-6)


class CN2Learner(RuleLearner):
    """
    Classical CN2 (see Clark and Niblett; 1988) induces a set of ordered
    rules, which means that classificator must try these rules in the same
    order as they were learned.
    
    If data instances are provided to the constructor, the learning algorithm
    is called and the resulting classifier is returned instead of the learner.

    Constructor can be given the following parameters:
    
    :param evaluator: an object that evaluates a rule from covered instances.
        By default, entropy is used as a measure. 
    :type evaluator: :class:`Orange.classification.rules.RuleEvaluator`
    :param beamWidth: width of the search beam.
    :type beamWidth: int
    :param alpha: significance level of the statistical test to determine
        whether rule is good enough to be returned by rulefinder. Likelihood
        ratio statistics is used that gives an estimate if rule is
        statistically better than the default rule.
    :type alpha: number

    """
    
    def __new__(cls, instances=None, weightID=0, **kwargs):
        self = RuleLearner.__new__(cls, **kwargs)
        if instances is not None:
            self.__init__(**kwargs)
            return self.__call__(instances, weightID)
        else:
            return self
        
    def __init__(self, evaluator = RuleEvaluator_Entropy(), beamWidth = 5, alpha = 1.0, **kwds):
        self.__dict__.update(kwds)
        self.ruleFinder = RuleBeamFinder()
        self.ruleFinder.ruleFilter = RuleBeamFilter_Width(width = beamWidth)
        self.ruleFinder.evaluator = evaluator
        self.ruleFinder.validator = RuleValidator_LRS(alpha = alpha)
        
    def __call__(self, instances, weight=0):
        supervisedClassCheck(instances)
        
        cl = RuleLearner.__call__(self,instances,weight)
        rules = cl.rules
        return CN2Classifier(rules, instances, weight)


class CN2Classifier(RuleClassifier):
    """
    Classical CN2 (see Clark and Niblett; 1988) classifies a new instance
    using an ordered set of rules. Usually the learner
    (:class:`Orange.classification.rules.CN2Learner`) is used to construct the
    classifier.
        
        :param instance: instance to be classifier
        :type instance: :class:`Orange.data.Instance`
        :param result_type: :class:`Orange.core.Classifier.GetValue` or \
              :class:`Orange.core.Classifier.GetProbabilities` or
              :class:`Orange.core.Classifier.GetBoth`
        
        :rtype: :class:`Orange.data.Value`, 
              :class:`Orange.statistics.Distribution` or a tuple with both
    
    """
    def __init__(self, rules=None, instances=None, weightID = 0, **argkw):
        self.rules = rules
        self.instances = instances
        self.weightID = weightID
        self.classVar = None if instances is None else instances.domain.classVar
        self.__dict__.update(argkw)
        if instances is not None:
            self.prior = Orange.core.Distribution(instances.domain.classVar, instances)

    def __call__(self, instance, result_type=Orange.core.Classifier.GetValue):
        classifier = None
        for r in self.rules:
         #   r.filter.domain = instance.domain
            if r(instance) and r.classifier:
                classifier = r.classifier
                classifier.defaultDistribution = r.classDistribution
                break
        if not classifier:
            classifier = Orange.core.DefaultClassifier(instance.domain.classVar, self.prior.modus())
            classifier.defaultDistribution = self.prior

        if result_type == Orange.core.Classifier.GetValue:
          return classifier(instance)
        if result_type == Orange.core.Classifier.GetProbabilities:
          return classifier.defaultDistribution
        return (classifier(instance),classifier.defaultDistribution)

    def __str__(self):
        retStr = ruleToString(self.rules[0])+" "+str(self.rules[0].classDistribution)+"\n"
        for r in self.rules[1:]:
            retStr += "ELSE "+ruleToString(r)+" "+str(r.classDistribution)+"\n"
        return retStr


class CN2UnorderedLearner(RuleLearner):
    """
    CN2 unordered (see Clark and Boswell; 1991) induces a set of unordered
    rules - classification from rules does not assume ordering of rules.
    Learning rules is quite similar to learning in classical CN2, where
    the process of learning of rules is separated to learning rules for each
    class.
    
    If data instances are provided to the constructor, the learning algorithm
    is called and the resulting classifier is returned instead of the learner.

    Constructor can be given the following parameters:
    
    :param evaluator: an object that evaluates a rule from covered instances.
        By default, Laplace's rule of succession is used as a measure. 
    :type evaluator: :class:`Orange.classification.rules.RuleEvaluator`
    :param beamWidth: width of the search beam.
    :type beamWidth: int
    :param alpha: significance level of the statistical test to determine
        whether rule is good enough to be returned by rulefinder. Likelihood
        ratio statistics is used that gives an estimate if rule is
        statistically better than the default rule.
    :type alpha: number
    """
    def __new__(cls, instances=None, weightID=0, **kwargs):
        self = RuleLearner.__new__(cls, **kwargs)
        if instances is not None:
            self.__init__(**kwargs)
            return self.__call__(instances, weightID)
        else:
            return self
            
    def __init__(self, evaluator = RuleEvaluator_Laplace(), beamWidth = 5, alpha = 1.0, **kwds):
        self.__dict__.update(kwds)
        self.ruleFinder = RuleBeamFinder()
        self.ruleFinder.ruleFilter = RuleBeamFilter_Width(width = beamWidth)
        self.ruleFinder.evaluator = evaluator
        self.ruleFinder.validator = RuleValidator_LRS(alpha = alpha)
        self.ruleFinder.ruleStoppingValidator = RuleValidator_LRS(alpha = 1.0)
        self.ruleStopping = RuleStopping_apriori()
        self.dataStopping = RuleDataStoppingCriteria_NoPositives()
        
    def __call__(self, instances, weight=0):
        supervisedClassCheck(instances)
        
        rules = RuleList()
        self.ruleStopping.apriori = Orange.core.Distribution(instances.domain.classVar,instances)
        progress=getattr(self,"progressCallback",None)
        if progress:
            progress.start = 0.0
            progress.end = 0.0
            distrib = Orange.core.Distribution(instances.domain.classVar, instances, weight)
            distrib.normalize()
        for targetClass in instances.domain.classVar:
            if progress:
                progress.start = progress.end
                progress.end += distrib[targetClass]
            self.targetClass = targetClass
            cl = RuleLearner.__call__(self,instances,weight)
            for r in cl.rules:
                rules.append(r)
        if progress:
            progress(1.0,None)
        return CN2UnorderedClassifier(rules, instances, weight)


class CN2UnorderedClassifier(RuleClassifier):
    """
    CN2 unordered (see Clark and Boswell; 1991) induces a set of unordered
    rules. Usually the learner
    (:class:`Orange.classification.rules.CN2UnorderedLearner`) is used to
    construct the classifier.
        
        :param instance: instance to be classifier
        :type instance: :class:`Orange.data.Instance`
        :param result_type: :class:`Orange.core.Classifier.GetValue` or \
              :class:`Orange.core.Classifier.GetProbabilities` or
              :class:`Orange.core.Classifier.GetBoth`
        
        :rtype: :class:`Orange.data.Value`, 
              :class:`Orange.statistics.Distribution` or a tuple with both
    
    """
    def __init__(self, rules = None, instances = None, weightID = 0, **argkw):
        self.rules = rules
        self.instances = instances
        self.weightID = weightID
        self.classVar = instances.domain.classVar if instances is not None else None
        self.__dict__.update(argkw)
        if instances is not None:
            self.prior = Orange.core.Distribution(instances.domain.classVar, instances)

    def __call__(self, instance, result_type=Orange.core.GetValue, retRules = False):
        def add(disc1, disc2, sumd):
            disc = Orange.core.DiscDistribution(disc1)
            sumdisc = sumd
            for i,d in enumerate(disc):
                disc[i]+=disc2[i]
                sumdisc += disc2[i]
            return disc, sumdisc

        # create empty distribution
        retDist = Orange.core.DiscDistribution(self.instances.domain.classVar)
        covRules = RuleList()
        # iterate through instances - add distributions
        sumdisc = 0.
        for r in self.rules:
            if r(instance) and r.classDistribution:
                retDist, sumdisc = add(retDist, r.classDistribution, sumdisc)
                covRules.append(r)
        if not sumdisc:
            retDist = self.prior
            sumdisc = self.prior.abs
            
        if sumdisc > 0.0:
            for c in self.instances.domain.classVar:
                retDist[c] /= sumdisc
        else:
            retDist.normalize()
        
        if retRules:
            if result_type == Orange.core.Classifier.GetValue:
              return (retDist.modus(), covRules)
            if result_type == Orange.core.Classifier.GetProbabilities:
              return (retDist, covRules)
            return (retDist.modus(),retDist,covRules)
        if result_type == Orange.core.Classifier.GetValue:
          return retDist.modus()
        if result_type == Orange.core.Classifier.GetProbabilities:
          return retDist
        return (retDist.modus(),retDist)

    def __str__(self):
        retStr = ""
        for r in self.rules:
            retStr += ruleToString(r)+" "+str(r.classDistribution)+"\n"
        return retStr


class CN2SDUnorderedLearner(CN2UnorderedLearner):
    """
    CN2-SD (see Lavrac et al.; 2004) induces a set of unordered rules, which
    is the same as :class:`Orange.classification.rules.CN2UnorderedLearner`.
    The difference between classical CN2 unordered and CN2-SD is selection of
    specific evaluation function and covering function:
    :class:`Orange.classifier.rules.WRACCEvaluator` is used to implement
    weight-relative accuracy and 
    :class:`Orange.classifier.rules.CovererAndRemover_multWeight` avoids
    excluding covered instances, multiplying their weight by the value of
    mult parameter instead.
    
    If data instances are provided to the constructor, the learning algorithm
    is called and the resulting classifier is returned instead of the learner.

    Constructor can be given the following parameters:
    
    :param evaluator: an object that evaluates a rule from covered instances.
        By default, weighted relative accuracy is used.
    :type evaluator: :class:`Orange.classification.rules.RuleEvaluator`
    :param beamWidth: width of the search beam.
    :type beamWidth: int
    :param alpha: significance level of the statistical test to determine
        whether rule is good enough to be returned by rulefinder. Likelihood
        ratio statistics is used that gives an estimate if rule is
        statistically better than the default rule.
    :type alpha: number
    :param mult: multiplicator for weights of covered instances.
    :type mult: number
    """
    def __new__(cls, instances=None, weightID=0, **kwargs):
        self = CN2UnorderedLearner.__new__(cls, **kwargs)
        if instances is not None:
            self.__init__(**kwargs)
            return self.__call__(instances, weightID)
        else:
            return self
        
    def __init__(self, evaluator = WRACCEvaluator(), beamWidth = 5, alpha = 0.05, mult=0.7, **kwds):
        CN2UnorderedLearnerClass.__init__(self, evaluator = evaluator,
                                          beamWidth = beamWidth, alpha = alpha, **kwds)
        self.coverAndRemove = CovererAndRemover_multWeights(mult=mult)

    def __call__(self, instances, weight=0):        
        supervisedClassCheck(instances)
        
        oldInstances = Orange.data.Table(instances)
        classifier = CN2UnorderedLearnerClass.__call__(self,instances,weight)
        for r in classifier.rules:
            r.filterAndStore(oldInstances,weight,r.classifier.defaultVal)
        return classifier


def ruleToString(rule, showDistribution = True):
    def selectSign(oper):
        if oper == Orange.core.ValueFilter_continuous.Less:
            return "<"
        elif oper == Orange.core.ValueFilter_continuous.LessEqual:
            return "<="
        elif oper == Orange.core.ValueFilter_continuous.Greater:
            return ">"
        elif oper == Orange.core.ValueFilter_continuous.GreaterEqual:
            return ">="
        else: return "="

    if not rule:
        return "None"
    conds = rule.filter.conditions
    domain = rule.filter.domain
    
    ret = "IF "
    if len(conds)==0:
        ret = ret + "TRUE"

    for i,c in enumerate(conds):
        if i > 0:
            ret += " AND "
        if type(c) == Orange.core.ValueFilter_discrete:
            ret += domain[c.position].name + "=" + str([domain[c.position].values[int(v)] for v in c.values])
        elif type(c) == Orange.core.ValueFilter_continuous:
            ret += domain[c.position].name + selectSign(c.oper) + str(c.ref)
    if rule.classifier and type(rule.classifier) == Orange.core.DefaultClassifier and rule.classifier.defaultVal:
        ret = ret + " THEN "+domain.classVar.name+"="+\
        str(rule.classifier.defaultValue)
        if showDistribution:
            ret += str(rule.classDistribution)
    elif rule.classifier and type(rule.classifier) == Orange.core.DefaultClassifier and type(domain.classVar) == Orange.core.EnumVariable:
        ret = ret + " THEN "+domain.classVar.name+"="+\
        str(rule.classDistribution.modus())
        if showDistribution:
            ret += str(rule.classDistribution)
    return ret        


class mEstimate(RuleEvaluator):
    def __init__(self, m=2):
        self.m = m
    def __call__(self, rule, data, weightID, targetClass, apriori):
        if not rule.classDistribution:
            return 0.
        sumDist = rule.classDistribution.abs
        if self.m == 0 and not sumDist:
            return 0.
        # get distribution
        if targetClass>-1:
            p = rule.classDistribution[targetClass]+self.m*apriori[targetClass]/apriori.abs
            p = p / (rule.classDistribution.abs + self.m)
        else:
            p = max(rule.classDistribution)+self.m*apriori[rule.classDistribution.modus()]/apriori.abs
            p = p / (rule.classDistribution.abs + self.m)      
        return p

class RuleStopping_apriori(RuleStoppingCriteria):
    def __init__(self, apriori=None):
        self.apriori =  None
        
    def __call__(self,rules,rule,instances,data):
        if not self.apriori:
            return False
        if not type(rule.classifier) == Orange.core.DefaultClassifier:
            return False
        ruleAcc = rule.classDistribution[rule.classifier.defaultVal]/rule.classDistribution.abs
        aprioriAcc = self.apriori[rule.classifier.defaultVal]/self.apriori.abs
        if ruleAcc>aprioriAcc:
            return False
        return True

class LengthValidator(RuleValidator):
    """ prune rules with more conditions than self.length. """
    def __init__(self, length=-1):
        self.length = length
        
    def __call__(self, rule, data, weightID, targetClass, apriori):
        if self.length >= 0:
            return len(rule.filter.conditions) <= self.length
        return True    
    

def supervisedClassCheck(instances):
    if not instances.domain.classVar:
        raise Exception("Class variable is required!")
    if instances.domain.classVar.varType == Orange.core.VarTypes.Continuous:
        raise Exception("CN2 requires a discrete class!")
    




class RuleClassifier_bestRule(RuleClassifier):
    def __init__(self, rules, instances, weightID = 0, **argkw):
        self.rules = rules
        self.instances = instances
        self.classVar = instances.domain.classVar
        self.__dict__.update(argkw)
        self.prior = Orange.core.Distribution(instances.domain.classVar, instances)

    def __call__(self, instance, result_type=Orange.core.Classifier.GetValue):
        retDist = Orange.core.Distribution(instance.domain.classVar)
        bestRule = None
        for r in self.rules:
            if r(instance) and (not bestRule or r.quality>bestRule.quality):
                for v_i,v in enumerate(instance.domain.classVar):
                    retDist[v_i] = r.classDistribution[v_i]
                bestRule = r
        if not bestRule:
            retDist = self.prior
        else:
            bestRule.used += 1
        sumdist = sum(retDist)
        if sumdist > 0.0:
            for c in self.instances.domain.classVar:
                retDist[c] /= sumdisc
        else:
            retDist.normalize()
        # return classifier(instance, result_type=result_type)
        if result_type == Orange.core.Classifier.GetValue:
          return retDist.modus()
        if result_type == Orange.core.Classifier.GetProbabilities:
          return retDist
        return (retDist.modus(),retDist)

    def __str__(self):
        retStr = ""
        for r in self.rules:
            retStr += ruleToString(r)+" "+str(r.classDistribution)+"\n"
        return retStr    

class CovererAndRemover_multWeights(RuleCovererAndRemover):
    def __init__(self, mult = 0.7):
        self.mult = mult
    def __call__(self, rule, instances, weights, targetClass):
        if not weights:
            weights = Orange.core.newmetaid()
            instances.addMetaAttribute(weights,1.)
            instances.domain.addmeta(weights, Orange.data.feature.Continuous("weights-"+str(weights)), True)
        newWeightsID = Orange.core.newmetaid()
        instances.addMetaAttribute(newWeightsID,1.)
        instances.domain.addmeta(newWeightsID, Orange.data.feature.Continuous("weights-"+str(newWeightsID)), True)
        for instance in instances:
            if rule(instance) and instance.getclass() == rule.classifier(instance,Orange.core.Classifier.GetValue):
                instance[newWeightsID]=instance[weights]*self.mult
            else:
                instance[newWeightsID]=instance[weights]
        return (instances,newWeightsID)

class CovererAndRemover_addWeights(RuleCovererAndRemover):
    def __call__(self, rule, instances, weights, targetClass):
        if not weights:
            weights = Orange.core.newmetaid()
            instances.addMetaAttribute(weights,1.)
            instances.domain.addmeta(weights, Orange.data.feature.Continuous("weights-"+str(weights)), True)
        try:
            coverage = instances.domain.getmeta("Coverage")
        except:
            coverage = Orange.data.feature.Continuous("Coverage")
            instances.domain.addmeta(Orange.core.newmetaid(),coverage, True)
            instances.addMetaAttribute(coverage,0.0)
        newWeightsID = Orange.core.newmetaid()
        instances.addMetaAttribute(newWeightsID,1.)
        instances.domain.addmeta(newWeightsID, Orange.data.feature.Continuous("weights-"+str(newWeightsID)), True)
        for instance in instances:
            if rule(instance) and instance.getclass() == rule.classifier(instance,Orange.core.Classifier.GetValue):
                try:
                    instance[coverage]+=1.0
                except:
                    instance[coverage]=1.0
                instance[newWeightsID]=1.0/(instance[coverage]+1)
            else:
                instance[newWeightsID]=instance[weights]
        return (instances,newWeightsID)

def rule_in_set(rule,rules):
    for r in rules:
        if rules_equal(rule,r):
            return True
    return False

def rules_equal(rule1,rule2):
    if not len(rule1.filter.conditions)==len(rule2.filter.conditions):
        return False
    for c1 in rule1.filter.conditions:
        found=False # find the same condition in the other rule
        for c2 in rule2.filter.conditions:
            try:
                if not c1.position == c2.position: continue # same attribute?
                if not type(c1) == type(c2): continue # same type of condition
                if type(c1) == Orange.core.ValueFilter_discrete:
                    if not type(c1.values[0]) == type(c2.values[0]): continue
                    if not c1.values[0] == c2.values[0]: continue # same value?
                if type(c1) == Orange.core.ValueFilter_continuous:
                    if not c1.oper == c2.oper: continue # same operator?
                    if not c1.ref == c2.ref: continue #same threshold?
                found=True
                break
            except:
                pass
        if not found:
            return False
    return True

class noDuplicates_validator(RuleValidator):
    def __init__(self,alpha=.05,min_coverage=0,max_rule_length=0,rules=RuleList()):
        self.rules = rules
        self.validator = RuleValidator_LRS(alpha=alpha,min_coverage=min_coverage,max_rule_length=max_rule_length)
        
    def __call__(self, rule, data, weightID, targetClass, apriori):
        if rule_in_set(rule,self.rules):
            return False
        return bool(self.validator(rule,data,weightID,targetClass,apriori))
                
class ruleSt_setRules(RuleStoppingCriteria):
    def __init__(self,validator):
        self.ruleStopping = RuleStoppingCriteria_NegativeDistribution()
        self.validator = validator

    def __call__(self,rules,rule,instances,data):        
        ru_st = self.ruleStopping(rules,rule,instances,data)
        if not ru_st:
            self.validator.rules.append(rule)
        return bool(ru_st)
    

# Miscellaneous - utility functions
def avg(l):
    if len(l)==0:
        return 0.
    return sum(l)/len(l)

def var(l):
    if len(l)<2:
        return 0.
    av = avg(l)
    vars=[math.pow(li-av,2) for li in l]
    return sum(vars)/(len(l)-1)

def median(l):
    if len(l)==0:
        return 0.    
    l.sort()
    le = len(l)
    if le%2 == 1:
        return l[(le-1)/2]
    else:
        return (l[le/2-1]+l[le/2])/2

def perc(l,p):
    l.sort()
    return l[int(math.floor(p*len(l)))]

def createRandomDataSet(data):
    newData = Orange.data.Table(data)
    # shuffle data
    cl_num = newData.toNumeric("C")
    random.shuffle(cl_num[0][:,0])
    clData = Orange.data.Table(Orange.data.Domain([newData.domain.classVar]),cl_num[0])
    for d_i,d in enumerate(newData):
        d[newData.domain.classVar] = clData[d_i][newData.domain.classVar]
    return newData

# estimated fisher tippett parameters for a set of values given in vals list (+ deciles)
def compParameters(vals,oldMi=0.5,oldBeta=1.1):                    
    # compute percentiles
    vals.sort()
    N = len(vals)
    percs = [avg(vals[int(float(N)*i/10):int(float(N)*(i+1)/10)]) for i in range(10)]            
    if N<10:
        return oldMi, oldBeta, percs
    beta = math.sqrt(6*var(vals)/math.pow(math.pi,2))
    beta = min(2.0,max(oldBeta, beta))
    mi = max(oldMi, avg(vals) - 0.57721*beta)
    return mi, beta, percs


def computeDists(data, weight=0, targetClass=0, N=100, learner=None):
    """ Compute distributions of likelihood ratio statistics of extreme (best) rules.  """
    if not learner:
        learner = createLearner()

    #########################
    ## Learner preparation ##
    #########################
    oldStopper = learner.ruleFinder.ruleStoppingValidator
    evaluator = learner.ruleFinder.evaluator
    learner.ruleFinder.evaluator = RuleEvaluator_LRS()
    learner.ruleFinder.evaluator.storeRules = True
    learner.ruleFinder.ruleStoppingValidator = RuleValidator_LRS(alpha=1.0)
    learner.ruleFinder.ruleStoppingValidator.max_rule_complexity = 0  

    # loop through N (sampling repetitions)
    maxVals = []
    for d_i in range(N):
        # create data set (remove and randomize)
        tempData = createRandomDataSet(data)
        learner.ruleFinder.evaluator.rules = RuleList()
        # Next, learn a rule
        bestRule = learner.ruleFinder(tempData,weight,targetClass,RuleList())
        maxVals.append(bestRule.quality)
    extremeDists=[compParameters(maxVals,1.0,1.0)]

    #####################
    ## Restore learner ##
    #####################
    learner.ruleFinder.evaluator = evaluator
    learner.ruleFinder.ruleStoppingValidator = oldStopper
    return extremeDists

def createEVDistList(evdList):
    l = Orange.core.EVDistList()
    for el in evdList:
        l.append(Orange.core.EVDist(mu=el[0],beta=el[1],percentiles=el[2]))
    return l

class CovererAndRemover_Prob(RuleCovererAndRemover):
    """ This class impements probabilistic covering. """
    def __init__(self, probAttribute=None, sigAttribute=None):
        self.indices = None
        self.probAttribute = probAttribute
        self.bestRule = []

    def initialize(self, instances, weightID, targetClass, apriori):
        self.bestRule = [None]*len(instances)
        self.probAttribute = Orange.core.newmetaid()
        instances.addMetaAttribute(self.probAttribute,-1.e-6)
        instances.domain.addmeta(self.probAttribute, Orange.data.feature.Continuous("Probs"))
        for instance in instances:
##            if targetClass<0 or (instance.getclass() == targetClass):
            instance[self.probAttribute] = apriori[targetClass]/apriori.abs
        return instances

    def getBestRules(self, currentRules, instances, weightID):
        bestRules = RuleList()
        for r in currentRules:
            if hasattr(r.learner, "argumentRule") and not orngCN2.rule_in_set(r,bestRules):
                bestRules.append(r)
        for r_i,r in enumerate(self.bestRule):
            if r and not rule_in_set(r,bestRules) and instances[r_i].getclass()==r.classifier.defaultValue:
                bestRules.append(r)
        return bestRules

    def remainingInstancesP(self, instances, targetClass):
        pSum, pAll = 0.0, 0.0
        for ex in instances:
            if ex.getclass() == targetClass:
                pSum += ex[self.probAttribute]
                pAll += 1.0
        return pSum/pAll

    def __call__(self, rule, instances, weights, targetClass):
        if targetClass<0:
            for instance_i, instance in enumerate(instances):
                if rule(instance) and rule.quality>instance[self.probAttribute]-0.01:
                    instance[self.probAttribute] = rule.quality+0.01
                    self.bestRule[instance_i]=rule
        else:
            for instance_i, instance in enumerate(instances): #rule.classifier.defaultVal == instance.getclass() and
                if rule(instance) and rule.quality>instance[self.probAttribute]:
                    instance[self.probAttribute] = rule.quality+0.001
                    self.bestRule[instance_i]=rule
##                if rule.classifier.defaultVal == instance.getclass():
##                    print instance[self.probAttribute]
        # compute factor
        return (instances,weights)

def add_sub_rules(rules, instances, weight, learner, dists):
    apriori = Orange.core.Distribution(instances.domain.classVar,instances,weight)
    newRules = RuleList()
    for r in rules:
        newRules.append(r)

    # loop through rules
    for r in rules:
        tmpList = RuleList()
        tmpRle = r.clone()
        tmpRle.filter.conditions = []
        tmpRle.parentRule = None
        tmpRle.filterAndStore(instances,weight,r.classifier.defaultVal)
        tmpList.append(tmpRle)
        while tmpList and len(tmpList[0].filter.conditions) <= len(r.filter.conditions):
            tmpList2 = RuleList()
            for tmpRule in tmpList:
                # evaluate tmpRule
                oldREP = learner.ruleFinder.evaluator.returnExpectedProb
                learner.ruleFinder.evaluator.returnExpectedProb = False
                learner.ruleFinder.evaluator.evDistGetter.dists = createEVDistList(dists[int(r.classifier.defaultVal)])
                tmpRule.quality = learner.ruleFinder.evaluator(tmpRule,instances,weight,r.classifier.defaultVal,apriori)
                learner.ruleFinder.evaluator.returnExpectedProb = oldREP
                # if rule not in rules already, add it to the list
                if not True in [rules_equal(ri,tmpRule) for ri in newRules] and len(tmpRule.filter.conditions)>0 and tmpRule.quality > apriori[r.classifier.defaultVal]/apriori.abs:
                    newRules.append(tmpRule)
                # create new tmpRules, set parent Rule, append them to tmpList2
                if not True in [rules_equal(ri,tmpRule) for ri in newRules]:
                    for c in r.filter.conditions:
                        tmpRule2 = tmpRule.clone()
                        tmpRule2.parentRule = tmpRule
                        tmpRule2.filter.conditions.append(c)
                        tmpRule2.filterAndStore(instances,weight,r.classifier.defaultVal)
                        if tmpRule2.classDistribution.abs < tmpRule.classDistribution.abs:
                            tmpList2.append(tmpRule2)
            tmpList = tmpList2
    for cl in instances.domain.classVar:
        tmpRle = Rule()
        tmpRle.filter = Orange.core.Filter_values(domain = instances.domain)
        tmpRle.parentRule = None
        tmpRle.filterAndStore(instances,weight,int(cl))
        tmpRle.quality = tmpRle.classDistribution[int(cl)]/tmpRle.classDistribution.abs
        newRules.append(tmpRle)
    return newRules

#def CN2EVCUnorderedLearner(instances = None, weightID=0, **kwds):
#    cn2 = CN2EVCUnorderedLearnerClass(**kwds)
#    if instances:
#        return cn2(instances, weightID)
#    else:
#        return cn2
    
class CN2EVCUnorderedLearner(ABCN2):
    """This is implementation of CN2 + EVC as evaluation + LRC classification.
        Main parameters:
          -- ...
    """
    def __init__(self, width=5, nsampling=100, rule_sig=1.0, att_sig=1.0, min_coverage = 1., max_rule_complexity = 5.):
        ABCN2.__init__(self, width=width, nsampling=nsampling, rule_sig=rule_sig, att_sig=att_sig,
                       min_coverage=int(min_coverage), max_rule_complexity = int(max_rule_complexity))