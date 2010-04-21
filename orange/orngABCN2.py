""" This module implements argument based rule learning.
The main learner class is ABCN2. The first few classes are some variants of ABCN2 with reasonable settings.  """

import orange
import orngCN2
from orngABML import *
import operator
import random
import numpy
import math

# Default learner - returns     #
# default classifier with pre-  #
# defined output  class         #
class DefaultLearner(orange.Learner):
    def __init__(self,defaultValue = None):
        self.defaultValue = defaultValue
    def __call__(self,examples,weightID=0):
        return orange.DefaultClassifier(self.defaultValue,defaultDistribution = orange.Distribution(examples.domain.classVar,examples,weightID))


# Main ABCN2 class
class ABCN2(orange.RuleLearner):
    """This is implementation of ABCN2 + EVC as evaluation + LRC classification.
    """
    
    def __init__(self, argumentID=0, width=5, m=2, opt_reduction=2, nsampling=100, max_rule_complexity=5,
                 rule_sig=1.0, att_sig=1.0, postpruning=None, min_quality=0., min_coverage=1, min_improved=1, min_improved_perc=0.0,
                 learn_for_class = None, learn_one_rule = False, evd=None, evd_arguments=None, prune_arguments=False, analyse_argument=-1,
                 alternative_learner = None, min_cl_sig = 0.5, min_beta = 0.0, set_prefix_rules = False, add_sub_rules = False,
                 **kwds):
        """
        Parameters:
            General rule learning:
                width               ... beam width (default 5)
                learn_for_class     ... learner rules for one class? otherwise None
                learn_one_rule      ... learn one rule only ?
                analyse_argument    ... learner only analyses argument with this index; if set to -1, then it learns normally
                
            Evaluator related:
                m                   ... m-estimate to be corrected with EVC (default 2)
                opt_reduction       ... types of EVC correction; 0=no correction, 1=pessimistic, 2=normal (default 2)
                nsampling           ... number of samples in estimating extreme value distribution (for EVC) (default 100)
                evd                 ... pre given extreme value distributions
                evd_arguments       ... pre given extreme value distributions for arguments

            Rule Validation:
                rule_sig            ... minimal rule significance (default 1.0)
                att_sig             ... minimal attribute significance in rule (default 1.0)
                max_rule_complexity ... maximum number of conditions in rule (default 5)
                min_coverage        ... minimal number of covered examples (default 5)

            Probabilistic covering:
                min_improved        ... minimal number of examples improved in probabilistic covering (default 1)
                min_improved_perc   ... minimal percentage of covered examples that need to be improved (default 0.0)

            Classifier (LCR) related:
                add_sub_rules       ... add sub rules ? (default False)
                min_cl_sig          ... minimal significance of beta in classifier (default 0.5)
                min_beta            ... minimal beta value (default 0.0)
                set_prefix_rules    ... should ordered prefix rules be added? (default False)
                alternative_learner ... use rule-learner as a correction method for other machine learning methods. (default None)

        """

        
        # argument ID which is passed to abcn2 learner
        self.argumentID = argumentID
        # learn for specific class only?        
        self.learn_for_class = learn_for_class
        # only analysing a specific argument or learning all at once
        self.analyse_argument = analyse_argument
        # should we learn only one rule?
        self.learn_one_rule = learn_one_rule
        self.postpruning = postpruning
        # rule finder
        self.ruleFinder = orange.RuleBeamFinder()
        self.ruleFilter = orange.RuleBeamFilter_Width(width=width)
        self.ruleFilter_arguments = ABBeamFilter(width=width)
        if max_rule_complexity - 1 < 0:
            max_rule_complexity = 10
        self.ruleFinder.ruleStoppingValidator = orange.RuleValidator_LRS(alpha = 1.0, min_quality = 0., max_rule_complexity = max_rule_complexity - 1, min_coverage=min_coverage)
        self.refiner = orange.RuleBeamRefiner_Selector()
        self.refiner_arguments = SelectorAdder(discretizer = orange.EntropyDiscretization(forceAttribute = 1,
                                                                                           maxNumberOfIntervals = 2))
        self.prune_arguments = prune_arguments
        # evc evaluator
        evdGet = orange.EVDistGetter_Standard()
        self.ruleFinder.evaluator = orange.RuleEvaluator_mEVC(m=m, evDistGetter = evdGet, min_improved = min_improved, min_improved_perc = min_improved_perc)
        self.ruleFinder.evaluator.returnExpectedProb = True
        self.ruleFinder.evaluator.optimismReduction = opt_reduction
        self.ruleFinder.evaluator.ruleAlpha = rule_sig
        self.ruleFinder.evaluator.attributeAlpha = att_sig
        self.ruleFinder.evaluator.validator = orange.RuleValidator_LRS(alpha = 1.0, min_quality = min_quality, min_coverage=min_coverage, max_rule_complexity = max_rule_complexity - 1)

        # learn stopping criteria
        self.ruleStopping = None
        self.dataStopping = orange.RuleDataStoppingCriteria_NoPositives()
        # evd fitting
        self.evd_creator = EVDFitter(self,n=nsampling)
        self.evd = evd
        self.evd_arguments = evd_arguments
        # classifier
        self.add_sub_rules = add_sub_rules
        self.classifier = PILAR(alternative_learner = alternative_learner, min_cl_sig = min_cl_sig, min_beta = min_beta, set_prefix_rules = set_prefix_rules)
        # arbitrary parameters
        self.__dict__.update(kwds)


    def __call__(self, examples, weightID=0):
        # initialize progress bar
        progress=getattr(self,"progressCallback",None)
        if progress:
            progress.start = 0.0
            progress.end = 0.0
            distrib = orange.Distribution(examples.domain.classVar, examples, weightID)
            distrib.normalize()
        
        # we begin with an empty set of rules
        all_rules = orange.RuleList()

        # th en, iterate through all classes and learn rule for each class separately
        for cl_i,cl in enumerate(examples.domain.classVar):
            if progress:
                step = distrib[cl] / 2.
                progress.start = progress.end
                progress.end += step
                
            if self.learn_for_class and not self.learn_for_class in [cl,cl_i]:
                continue

            # rules for this class only
            rules, arg_rules = orange.RuleList(), orange.RuleList()

            # create dichotomous class
            dich_data = self.create_dich_class(examples, cl)

            # preparation of the learner (covering, evd, etc.)
            self.prepare_settings(dich_data, weightID, cl_i, progress)

            # learn argumented rules first ...
            self.turn_ABML_mode(dich_data, weightID, cl_i)
            # first specialize all unspecialized arguments
            # dich_data = self.specialise_arguments(dich_data, weightID)
            # comment: specialisation of arguments is within learning of an argumented rule;
            #          this is now different from the published algorithm
            if progress:
                progress.start = progress.end
                progress.end += step
            
            aes = self.get_argumented_examples(dich_data)
            aes = self.sort_arguments(aes, dich_data)
            while aes:
                if self.analyse_argument > -1 and not dich_data[self.analyse_argument] == aes[0]:
                    aes = aes[1:]
                    continue
                ae = aes[0]
                rule = self.learn_argumented_rule(ae, dich_data, weightID) # target class is always first class (0)
                if not progress:
                    print "learned rule", orngCN2.ruleToString(rule)
                if rule:
                    arg_rules.append(rule)
                    aes = filter(lambda x: not rule(x), aes)
                else:
                    aes = aes[1:]
            if not progress:
                print " arguments finished ... "                    
                   
            # remove all examples covered by rules
##            for rule in rules:
##                dich_data = self.remove_covered_examples(rule, dich_data, weightID)
##            if progress:
##                progress(self.remaining_probability(dich_data),None)

            # learn normal rules on remaining examples
            if self.analyse_argument == -1:
                self.turn_normal_mode(dich_data, weightID, cl_i)
                while dich_data:
                    # learn a rule
                    rule = self.learn_normal_rule(dich_data, weightID, self.apriori)
                    if not rule:
                        break
                    if not progress:
                        print "rule learned: ", orngCN2.ruleToString(rule), rule.quality
                    dich_data = self.remove_covered_examples(rule, dich_data, weightID)
                    if progress:
                        progress(self.remaining_probability(dich_data),None)
                    rules.append(rule)
                    if self.learn_one_rule:
                        break

            for r in arg_rules:
                dich_data = self.remove_covered_examples(r, dich_data, weightID)
                rules.append(r)

            # prune unnecessary rules
            rules = self.prune_unnecessary_rules(rules, dich_data, weightID)

            if self.add_sub_rules:
                rules = self.add_sub_rules_call(rules, dich_data, weightID)

            # restore domain and class in rules, add them to all_rules
            for r in rules:
                all_rules.append(self.change_domain(r, cl, examples, weightID))

            if progress:
                progress(1.0,None)
        # create a classifier from all rules        
        return self.create_classifier(all_rules, examples, weightID)

    def learn_argumented_rule(self, ae, examples, weightID):
        # prepare roots of rules from arguments
        positive_args = self.init_pos_args(ae, examples, weightID)
        if not positive_args: # something wrong
            raise "There is a problem with argumented example %s"%str(ae)
            return None
        negative_args = self.init_neg_args(ae, examples, weightID)

        # set negative arguments in refiner
        self.ruleFinder.refiner.notAllowedSelectors = negative_args
        self.ruleFinder.refiner.example = ae
        # set arguments to filter
        self.ruleFinder.ruleFilter.setArguments(examples.domain,positive_args)

        # learn a rule
        self.ruleFinder.evaluator.bestRule = None
        self.ruleFinder.evaluator.returnBestFuture = True
        self.ruleFinder(examples,weightID,0,positive_args)
##        self.ruleFinder.evaluator.bestRule.quality = 0.8
        
        # return best rule
        return self.ruleFinder.evaluator.bestRule
        
    def prepare_settings(self, examples, weightID, cl_i, progress):
        # apriori distribution
        self.apriori = orange.Distribution(examples.domain.classVar,examples,weightID)
        
        # prepare covering mechanism
        self.coverAndRemove = CovererAndRemover_Prob(examples, weightID, 0, self.apriori)
        self.ruleFinder.evaluator.probVar = examples.domain.getmeta(self.coverAndRemove.probAttribute)

        # compute extreme distributions
        # TODO: why evd and evd_this????
        if self.ruleFinder.evaluator.optimismReduction > 0 and not self.evd:
            self.evd_this = self.evd_creator.computeEVD(examples, weightID, target_class=0, progress = progress)
        if self.evd:
            self.evd_this = self.evd[cl_i]

    def turn_ABML_mode(self, examples, weightID, cl_i):
        # evaluator
        if self.ruleFinder.evaluator.optimismReduction > 0 and self.argumentID:
            if self.evd_arguments:
                self.ruleFinder.evaluator.evDistGetter.dists = self.evd_arguments[cl_i]
            else:
                self.ruleFinder.evaluator.evDistGetter.dists = self.evd_this # self.evd_creator.computeEVD_example(examples, weightID, target_class=0)
        # rule refiner
        self.ruleFinder.refiner = self.refiner_arguments
        self.ruleFinder.refiner.argumentID = self.argumentID
        self.ruleFinder.ruleFilter = self.ruleFilter_arguments

    def create_dich_class(self, examples, cl):
        """ create dichotomous class. """
        (newDomain, targetVal) = createDichotomousClass(examples.domain, examples.domain.classVar, str(cl), negate=0)
        newDomainmetas = newDomain.getmetas()
        newDomain.addmeta(orange.newmetaid(), examples.domain.classVar) # old class as meta
        dichData = examples.select(newDomain)
        if self.argumentID:
            for d in dichData: # remove arguments given to other classes
                if not d.getclass() == targetVal:
                    d[self.argumentID] = "?"
        return dichData

    def get_argumented_examples(self, examples):
        if not self.argumentID:
            return None
        
        # get argumentated examples
        return ArgumentFilter_hasSpecial()(examples, self.argumentID, targetClass = 0)

    def sort_arguments(self, arg_examples, examples):
        if not self.argumentID:
            return None
        evaluateAndSortArguments(examples, self.argumentID)
        if len(arg_examples)>0:
            # sort examples by their arguments quality (using first argument as it has already been sorted)
            sorted = arg_examples.native()
            sorted.sort(lambda x,y: -cmp(x[self.argumentID].value.positiveArguments[0].quality,
                                         y[self.argumentID].value.positiveArguments[0].quality))
            return orange.ExampleTable(examples.domain, sorted)
        else:
            return None

    def turn_normal_mode(self, examples, weightID, cl_i):
        # evaluator
        if self.ruleFinder.evaluator.optimismReduction > 0:
            if self.evd:
                self.ruleFinder.evaluator.evDistGetter.dists = self.evd[cl_i]
            else:
                self.ruleFinder.evaluator.evDistGetter.dists = self.evd_this # self.evd_creator.computeEVD(examples, weightID, target_class=0)
        # rule refiner
        self.ruleFinder.refiner = self.refiner
        self.ruleFinder.ruleFilter = self.ruleFilter
        
    def learn_normal_rule(self, examples, weightID, apriori):
        if hasattr(self.ruleFinder.evaluator, "bestRule"):
            self.ruleFinder.evaluator.bestRule = None
        rule = self.ruleFinder(examples,weightID,0,orange.RuleList())
        if hasattr(self.ruleFinder.evaluator, "bestRule") and self.ruleFinder.evaluator.returnExpectedProb:
            rule = self.ruleFinder.evaluator.bestRule
            self.ruleFinder.evaluator.bestRule = None
        if self.postpruning:
            rule = self.postpruning(rule,examples,weightID,0, aprior)
        return rule

    def remove_covered_examples(self, rule, examples, weightID):
        nexamples, nweight = self.coverAndRemove(rule,examples,weightID,0)
        return nexamples


    def prune_unnecessary_rules(self, rules, examples, weightID):
        return self.coverAndRemove.getBestRules(rules,examples,weightID)

    def change_domain(self, rule, cl, examples, weightID):
        rule.examples = rule.examples.select(examples.domain)
        rule.classDistribution = orange.Distribution(rule.examples.domain.classVar,rule.examples,weightID) # adapt distribution
        rule.classifier = orange.DefaultClassifier(cl) # adapt classifier
        rule.filter = orange.Filter_values(domain = examples.domain,
                                        conditions = rule.filter.conditions)
        if hasattr(rule, "learner") and hasattr(rule.learner, "arg_example"):
            rule.learner.arg_example = orange.Example(examples.domain, rule.learner.arg_example)
        return rule

    def create_classifier(self, rules, examples, weightID):
        return self.classifier(rules, examples, weightID)

    def add_sub_rules_call(self, rules, examples, weightID):
        apriori = orange.Distribution(examples.domain.classVar,examples,weightID)
        newRules = orange.RuleList()
        for r in rules:
            newRules.append(r)

        # loop through rules
        for r in rules:
            tmpList = orange.RuleList()
            tmpRle = r.clone()
            tmpRle.filter.conditions = r.filter.conditions[:r.requiredConditions] # do not split argument
            tmpRle.parentRule = None
            tmpRle.filterAndStore(examples,weightID,r.classifier.defaultVal)
            tmpRle.complexity = 0
            tmpList.append(tmpRle)
            while tmpList and len(tmpList[0].filter.conditions) <= len(r.filter.conditions):
                tmpList2 = orange.RuleList()
                for tmpRule in tmpList:
                    # evaluate tmpRule
                    oldREP = self.ruleFinder.evaluator.returnExpectedProb
                    self.ruleFinder.evaluator.returnExpectedProb = False
                    tmpRule.quality = self.ruleFinder.evaluator(tmpRule,examples,weightID,r.classifier.defaultVal,apriori)
                    self.ruleFinder.evaluator.returnExpectedProb = oldREP
                    # if rule not in rules already, add it to the list
                    if not True in [orngCN2.rules_equal(ri,tmpRule) for ri in newRules] and len(tmpRule.filter.conditions)>0 and tmpRule.quality > apriori[r.classifier.defaultVal]/apriori.abs:
                        newRules.append(tmpRule)
                    # create new tmpRules, set parent Rule, append them to tmpList2
                    if not True in [orngCN2.rules_equal(ri,tmpRule) for ri in newRules]:
                        for c in r.filter.conditions:
                            tmpRule2 = tmpRule.clone()
                            tmpRule2.parentRule = tmpRule
                            tmpRule2.filter.conditions.append(c)
                            tmpRule2.filterAndStore(examples,weightID,r.classifier.defaultVal)
                            tmpRule2.complexity += 1
                            if tmpRule2.classDistribution.abs < tmpRule.classDistribution.abs:
                                tmpList2.append(tmpRule2)
                tmpList = tmpList2
        return newRules


    def init_pos_args(self, ae, examples, weightID):
        pos_args = orange.RuleList()
        # prepare arguments
        for p in ae[self.argumentID].value.positiveArguments:
            new_arg = orange.Rule(filter=ArgFilter(argumentID = self.argumentID,
                                                   filter = self.newFilter_values(p.filter)),
                                                   complexity = 0)
            new_arg.valuesFilter = new_arg.filter.filter
            pos_args.append(new_arg)


        if hasattr(self.ruleFinder.evaluator, "returnExpectedProb"):
            old_exp = self.ruleFinder.evaluator.returnExpectedProb
            self.ruleFinder.evaluator.returnExpectedProb = False
            
        # argument pruning (all or just unfinished arguments)
        # if pruning is chosen, then prune arguments if possible
        for p in pos_args:
            p.filterAndStore(examples, weightID, 0)
            # pruning on: we check on all conditions and take only best
            if self.prune_arguments:
                allowed_conditions = [c for c in p.filter.conditions]
                pruned_conditions = self.prune_arg_conditions(ae, allowed_conditions, examples, weightID)
                p.filter.conditions = pruned_conditions
            else: # prune only unspecified conditions
                spec_conditions = [c for c in p.filter.conditions if not c.unspecialized_condition]
                unspec_conditions = [c for c in p.filter.conditions if c.unspecialized_condition]
                # let rule cover now all examples filtered by specified conditions
                p.filter.conditions = spec_conditions
                p.filterAndStore(examples, weightID, 0)
                pruned_conditions = self.prune_arg_conditions(ae, unspec_conditions, p.examples, p.weightID)
                p.filter.conditions.extend(pruned_conditions)
                p.filter.filter.conditions.extend(pruned_conditions)
                # if argument does not contain all unspecialized reasons, add those reasons with minimum values
                at_oper_pairs = [(c.position, c.oper) for c in p.filter.conditions if type(c) == orange.ValueFilter_continuous]
                for u in unspec_conditions:
                    if not (u.position, u.oper) in at_oper_pairs:
                        # find minimum value
                        u.ref = min([float(e[u.position])-10. for e in p.examples])
                        p.filter.conditions.append(u)
                        p.filter.filter.conditions.append(u)
                

        # set parameters to arguments
        for p_i,p in enumerate(pos_args):
            p.filterAndStore(examples,weightID,0)
            p.filter.domain = examples.domain
            if not p.learner:
                p.learner = DefaultLearner(defaultValue=ae.getclass())
            p.classifier = p.learner(p.examples, p.weightID)
            p.baseDist = p.classDistribution
            p.requiredConditions = len(p.filter.conditions)
            p.learner.setattr("arg_length", len(p.filter.conditions))
            p.learner.setattr("arg_example", ae)
            p.complexity = len(p.filter.conditions)
            
        if hasattr(self.ruleFinder.evaluator, "returnExpectedProb"):
            self.ruleFinder.evaluator.returnExpectedProb = old_exp

        return pos_args

    def newFilter_values(self, filter):
        newFilter = orange.Filter_values()
        newFilter.conditions = filter.conditions[:]
        newFilter.domain = filter.domain
        newFilter.negate = filter.negate
        newFilter.conjunction = filter.conjunction
        return newFilter

    def init_neg_args(self, ae, examples, weightID):
        return ae[self.argumentID].value.negativeArguments

    def remaining_probability(self, examples):
        return self.coverAndRemove.covered_percentage(examples)

    def prune_arg_conditions(self, crit_example, allowed_conditions, examples, weightID):
        if not allowed_conditions:
            return []
        cn2_learner = orngCN2.CN2UnorderedLearner()
        cn2_learner.ruleFinder = orange.RuleBeamFinder()
        cn2_learner.ruleFinder.refiner = SelectorArgConditions(crit_example, allowed_conditions)
        cn2_learner.ruleFinder.evaluator = orngCN2.mEstimate(self.ruleFinder.evaluator.m)
        rule = cn2_learner.ruleFinder(examples,weightID,0,orange.RuleList())
        return rule.filter.conditions


class ABCN2Ordered(ABCN2):
    """ Rules learned by ABCN2 are ordered and used as a decision list. """
    def __init__(self, argumentID=0, **kwds):
        ABCN2.__init__(self, argumentID=argumentID, **kwds)
        self.classifier.set_prefix_rules = True
        self.classifier.optimize_betas = False

class ABCN2M(ABCN2):
    """ Argument based rule learning with m-estimate as evaluation function. """
    def __init__(self, argumentID=0, **kwds):
        ABCN2.__init__(self, argumentID=argumentID, **kwds)
        self.opt_reduction = 0
    

# *********************** #
# Argument based covering #
# *********************** #

class ABBeamFilter(orange.RuleBeamFilter):
    """ ABBeamFilter: Filters beam;
        - leaves first N rules (by quality)
        - leaves first N rules that have only of arguments in condition part 
    """
    def __init__(self,width=5):
        self.width=width
        self.pArgs=None

    def __call__(self,rulesStar,examples,weightID):
        newStar=orange.RuleList()
        rulesStar.sort(lambda x,y: -cmp(x.quality,y.quality))
        argsNum=0
        for r_i,r in enumerate(rulesStar):
            if r_i<self.width: # either is one of best "width" rules
                newStar.append(r)
            elif self.onlyPositives(r):
                if argsNum<self.width:
                    newStar.append(r)
                    argsNum+=1
        return newStar                

    def setArguments(self,domain,positiveArguments):
        self.pArgs = positiveArguments
        self.domain = domain
        self.argTab = [0]*len(self.domain.attributes)
        for arg in self.pArgs:
            for cond in arg.filter.conditions:
                self.argTab[cond.position]=1
        
    def onlyPositives(self,rule):
        if not self.pArgs:
            return False

        ruleTab=[0]*len(self.domain.attributes)
        for cond in rule.filter.conditions:
            ruleTab[cond.position]=1
        return map(operator.or_,ruleTab,self.argTab)==self.argTab


class ruleCoversArguments:
    """ Class determines if rule covers one out of a set of arguments. """
    def __init__(self, arguments):
        self.arguments = arguments
        self.indices = []
        for a in self.arguments:
            indNA = getattr(a.filter,"indices",None)
            if not indNA:
                a.filter.setattr("indices", ruleCoversArguments.filterIndices(a.filter))
            self.indices.append(a.filter.indices)

    def __call__(self, rule):
        if not self.indices:
            return False
        if not getattr(rule.filter,"indices",None):
            rule.filter.indices = ruleCoversArguments.filterIndices(rule.filter)
        for index in self.indices:
            if map(operator.or_,rule.filter.indices,index) == rule.filter.indices:
                return True
        return False

    def filterIndices(filter):
        if not filter.domain:
            return []
        ind = [0]*len(filter.domain.attributes)
        for c in filter.conditions:
            ind[c.position]=operator.or_(ind[c.position],
                                         ruleCoversArguments.conditionIndex(c))
        return ind
    filterIndices = staticmethod(filterIndices)

    def conditionIndex(c):
        if type(c) == orange.ValueFilter_continuous:
            if (c.oper == orange.ValueFilter_continuous.GreaterEqual or
                c.oper == orange.ValueFilter_continuous.Greater):
                return 5# 0101
            elif (c.oper == orange.ValueFilter_continuous.LessEqual or
                  c.oper == orange.ValueFilter_continuous.Less):
                return 3 # 0011
            else:
                return c.oper
        else:
            return 1 # 0001
    conditionIndex = staticmethod(conditionIndex)        

    def oneSelectorToCover(ruleIndices, argIndices):
        at, type = -1, 0
        for r_i, ind in enumerate(ruleIndices):
            if not argIndices[r_i]:
                continue
            if at>-1 and not ind == argIndices[r_i]: # need two changes
                return (-1,0)
            if not ind == argIndices[r_i]:
                if argIndices[r_i] in [1,3,5]:
                    at,type=r_i,argIndices[r_i]
                if argIndices[r_i]==6:
                    if ind==3:
                        at,type=r_i,5
                    if ind==5:
                        at,type=r_i,3
        return at,type
    oneSelectorToCover = staticmethod(oneSelectorToCover)                 

class SelectorAdder(orange.RuleBeamRefiner):
    """ Selector adder, this function is a refiner function:
       - refined rules are not consistent with any of negative arguments. """
    def __init__(self, example=None, notAllowedSelectors=[], argumentID = None,
                 discretizer = orange.EntropyDiscretization(forceAttribute=True)):
        # required values - needed values of attributes
        self.example = example
        self.argumentID = argumentID
        self.notAllowedSelectors = notAllowedSelectors
        self.discretizer = discretizer
        
    def __call__(self, oldRule, data, weightID, targetClass=-1):
        inNotAllowedSelectors = ruleCoversArguments(self.notAllowedSelectors)
        newRules = orange.RuleList()

        # get positive indices (selectors already in the rule)
        indices = getattr(oldRule.filter,"indices",None)
        if not indices:
            indices = ruleCoversArguments.filterIndices(oldRule.filter)
            oldRule.filter.setattr("indices",indices)

        # get negative indices (selectors that should not be in the rule)
        negativeIndices = [0]*len(data.domain.attributes)
        for nA in self.notAllowedSelectors:
            #print indices, nA.filter.indices
            at_i,type_na = ruleCoversArguments.oneSelectorToCover(indices, nA.filter.indices)
            if at_i>-1:
                negativeIndices[at_i] = operator.or_(negativeIndices[at_i],type_na)

        #iterate through indices = attributes 
        for i,ind in enumerate(indices):
            if not self.example[i] or self.example[i].isSpecial():
                continue
            if ind == 1: 
                continue
            if data.domain[i].varType == orange.VarTypes.Discrete and not negativeIndices[i]==1: # DISCRETE attribute
                if self.example:
                    values = [self.example[i]]
                else:
                    values = data.domain[i].values
                for v in values:
                    tempRule = oldRule.clone()
                    tempRule.filter.conditions.append(orange.ValueFilter_discrete(position = i,
                                                                                  values = [orange.Value(data.domain[i],v)],
                                                                                  acceptSpecial=0))
                    tempRule.complexity += 1
                    tempRule.filter.indices[i] = 1 # 1 stands for discrete attribute (see ruleCoversArguments.conditionIndex)
                    tempRule.filterAndStore(oldRule.examples, oldRule.weightID, targetClass)
                    if len(tempRule.examples)<len(oldRule.examples):
                        newRules.append(tempRule)
            elif data.domain[i].varType == orange.VarTypes.Continuous and not negativeIndices[i]==7: # CONTINUOUS attribute
                try:
                    at = data.domain[i]
                    at_d = self.discretizer(at,oldRule.examples)
                except:
                    continue # discretization failed !
                # If discretization makes sense? then:
                if len(at_d.values)>1:
                    for p in at_d.getValueFrom.transformer.points:
                        #LESS
                        if not negativeIndices[i]==3:
                            tempRule = self.getTempRule(oldRule,i,orange.ValueFilter_continuous.LessEqual,p,targetClass,3)
                            if len(tempRule.examples)<len(oldRule.examples) and self.example[i]<=p:# and not inNotAllowedSelectors(tempRule):
                                newRules.append(tempRule)
                        #GREATER
                        if not negativeIndices[i]==5:
                            tempRule = self.getTempRule(oldRule,i,orange.ValueFilter_continuous.Greater,p,targetClass,5)
                            if len(tempRule.examples)<len(oldRule.examples) and self.example[i]>p:# and not inNotAllowedSelectors(tempRule):
                                newRules.append(tempRule)
        for r in newRules:
            r.parentRule = oldRule
            r.valuesFilter = r.filter.filter
        return newRules

    def getTempRule(self,oldRule,pos,oper,ref,targetClass,atIndex):
        tempRule = oldRule.clone()

        tempRule.filter.conditions.append(orange.ValueFilter_continuous(position=pos,
                                                                        oper=oper,
                                                                        ref=ref,
                                                                        acceptSpecial=0))
        tempRule.complexity += 1
        tempRule.filter.indices[pos] = operator.or_(tempRule.filter.indices[pos],atIndex) # from ruleCoversArguments.conditionIndex
        tempRule.filterAndStore(oldRule.examples,tempRule.weightID,targetClass)
        return tempRule

    def setCondition(self, oldRule, targetClass, ci, condition):
        tempRule = oldRule.clone()
        tempRule.filter.conditions[ci] = condition
        tempRule.filter.conditions[ci].setattr("specialized",1)
        tempRule.filterAndStore(oldRule.examples,oldRule.weightID,targetClass)
        return tempRule


# This filter is the ugliest code ever! Problem is with Orange, I had some problems with inheriting deepCopy
# I should take another look at it.
class ArgFilter(orange.Filter):
    """ This class implements AB-covering principle. """
    def __init__(self, argumentID=None, filter = orange.Filter_values()):
        self.filter = filter
        self.indices = getattr(filter,"indices",[])
        if not self.indices and len(filter.conditions)>0:
            self.indices = ruleCoversArguments.filterIndices(filter)
        self.argumentID = argumentID
        self.debug = 0
        self.domain = self.filter.domain
        self.conditions = filter.conditions
        
    def condIn(self,cond): # is condition in the filter?
        condInd = ruleCoversArguments.conditionIndex(cond)
        if operator.or_(condInd,self.indices[cond.position]) == self.indices[cond.position]:
            return True
        return False
    
    def __call__(self,example):
##        print "in", self.filter(example), self.filter.conditions[0](example)
##        print self.filter.conditions[1].values
        if self.filter(example):
            try:
                if example[self.argumentID].value and len(example[self.argumentID].value.positiveArguments)>0: # example has positive arguments
                    # conditions should cover at least one of the positive arguments
                    oneArgCovered = False
                    for pA in example[self.argumentID].value.positiveArguments:
                        argCovered = [self.condIn(c) for c in pA.filter.conditions]
                        oneArgCovered = oneArgCovered or len(argCovered) == sum(argCovered) #argCovered
                        if oneArgCovered:
                            break
                    if not oneArgCovered:
                        return False
                if example[self.argumentID].value and len(example[self.argumentID].value.negativeArguments)>0: # example has negative arguments
                    # condition should not cover neither of negative arguments
                    for pN in example[self.argumentID].value.negativeArguments:
                        argCovered = [self.condIn(c) for c in pN.filter.conditions]
                        if len(argCovered)==sum(argCovered):
                            return False
            except:
                return True
            return True
        else:
            return False

    def __setattr__(self,name,obj):
        self.__dict__[name]=obj
        self.filter.setattr(name,obj)

    def deepCopy(self):
        newFilter = ArgFilter(argumentID=self.argumentID)
        newFilter.filter = orange.Filter_values() #self.filter.deepCopy()
        newFilter.filter.conditions = self.filter.conditions[:]
        newFilter.domain = self.filter.domain
        newFilter.negate = self.filter.negate
        newFilter.conjunction = self.filter.conjunction
        newFilter.domain = self.filter.domain
        newFilter.conditions = newFilter.filter.conditions
        newFilter.indices = self.indices[:]
        if getattr(self,"candidateValues",None):
            newFilter.candidateValues = self.candidateValues[:]
        return newFilter


class SelectorArgConditions(orange.RuleBeamRefiner):
    """ Selector adder, this function is a refiner function:
       - refined rules are not consistent with any of negative arguments. """
    def __init__(self, example, allowed_selectors):
        # required values - needed values of attributes
        self.example = example
        self.allowed_selectors = allowed_selectors

    def __call__(self, oldRule, data, weightID, targetClass=-1):
        if len(oldRule.filter.conditions) >= len(self.allowed_selectors):
            return orange.RuleList()
        newRules = orange.RuleList()
        for c in self.allowed_selectors:
            # normal condition
            if not c.unspecialized_condition:
                tempRule = oldRule.clone()
                tempRule.filter.conditions.append(c)
                tempRule.filterAndStore(oldRule.examples, oldRule.weightID, targetClass)
                if len(tempRule.examples)<len(oldRule.examples):
                    newRules.append(tempRule)
            # unspecified condition
            else:
                # find all possible example values
                vals = {}
                for e in oldRule.examples:
                    if not e[c.position].isSpecial():
                        vals[str(e[c.position])] = 1
                values = vals.keys()
                # for each value make a condition
                for v in values:
                    tempRule = oldRule.clone()
                    tempRule.filter.conditions.append(orange.ValueFilter_continuous(position=c.position,
                                                                                    oper=c.oper,
                                                                                    ref=float(v),
                                                                                    acceptSpecial=0))
                    if tempRule(self.example):
                        tempRule.filterAndStore(oldRule.examples, oldRule.weightID, targetClass)
                        if len(tempRule.examples)<len(oldRule.examples):
                            newRules.append(tempRule)
##        print " NEW RULES "
##        for r in newRules:
##            print orngCN2.ruleToString(r)
        for r in newRules:
            r.parentRule = oldRule
##            print orngCN2.ruleToString(r)
        return newRules


# ********************** #
# Probabilistic covering #
# ********************** #

class CovererAndRemover_Prob(orange.RuleCovererAndRemover):
    """ This class impements probabilistic covering. """

    def __init__(self, examples, weightID, targetClass, apriori):
        self.bestRule = [None]*len(examples)
        self.probAttribute = orange.newmetaid()
        self.aprioriProb = apriori[targetClass]/apriori.abs
        examples.addMetaAttribute(self.probAttribute, self.aprioriProb)
        examples.domain.addmeta(self.probAttribute, orange.FloatVariable("Probs"))

    def getBestRules(self, currentRules, examples, weightID):
        bestRules = orange.RuleList()
##        for r in currentRules:
##            if hasattr(r.learner, "argumentRule") and not orngCN2.rule_in_set(r,bestRules):
##                bestRules.append(r)
        for r_i,r in enumerate(self.bestRule):
            if r and not orngCN2.rule_in_set(r,bestRules) and int(examples[r_i].getclass())==int(r.classifier.defaultValue):
                bestRules.append(r)
        return bestRules

    def __call__(self, rule, examples, weights, targetClass):
        if hasattr(rule, "learner") and hasattr(rule.learner, "arg_example"):
            example = rule.learner.arg_example
        else:
            example = None
        for ei, e in enumerate(examples):
##            if e == example:
##                e[self.probAttribute] = 1.0
##                self.bestRule[ei]=rule
            if example and not (hasattr(self.bestRule[ei], "learner") and hasattr(self.bestRule[ei].learner, "arg_example")):
                can_be_worst = True
            else:
                can_be_worst = False
            if can_be_worst and rule(e) and rule.quality>(e[self.probAttribute]-0.01):
                e[self.probAttribute] = rule.quality+0.001 # 0.001 is added to avoid numerical errors
                self.bestRule[ei]=rule
            elif rule(e) and rule.quality>e[self.probAttribute]:
                e[self.probAttribute] = rule.quality+0.001 # 0.001 is added to avoid numerical errors
                self.bestRule[ei]=rule
        return (examples,weights)

    def covered_percentage(self, examples):
        p = 0.0
        for ei, e in enumerate(examples):
            p += (e[self.probAttribute] - self.aprioriProb)/(1.0-self.aprioriProb)
        return p/len(examples)


# **************************************** #
# Estimation of extreme value distribution #
# **************************************** #

# Miscellaneous - utility functions
def avg(l):
    return sum(l)/len(l) if l else 0.

def var(l):
    if len(l)<2:
        return 0.
    av = avg(l)
    return sum([math.pow(li-av,2) for li in l])/(len(l)-1)

def perc(l,p):
    l.sort()
    return l[int(math.floor(p*len(l)))]

class EVDFitter:
    """ Randomizes a dataset and fits an extreme value distribution onto it. """

    def __init__(self, learner, n=200, randomseed=100):
        self.learner = learner
        self.n = n
        self.randomseed = randomseed
        
    def createRandomDataSet(self, data):
        newData = orange.ExampleTable(data)
        # shuffle data
        cl_num = newData.toNumpy("C")
        random.shuffle(cl_num[0][:,0])
        clData = orange.ExampleTable(orange.Domain([newData.domain.classVar]),cl_num[0])
        for d_i,d in enumerate(newData):
            d[newData.domain.classVar] = clData[d_i][newData.domain.classVar]
        return newData

    def createEVDistList(self, evdList):
        l = orange.EVDistList()
        for el in evdList:
            l.append(orange.EVDist(mu=el[0],beta=el[1],percentiles=el[2]))
        return l

    # estimated fisher tippett parameters for a set of values given in vals list (+ deciles)
    def compParameters(self, vals, oldMi=0.5,oldBeta=1.1):                    
        # compute percentiles
        vals.sort()
        N = len(vals)
        percs = [avg(vals[int(float(N)*i/10):int(float(N)*(i+1)/10)]) for i in range(10)]            
        if N<10:
            return oldMi, oldBeta, percs
        beta = min(2.0, max(oldBeta, math.sqrt(6*var(vals)/math.pow(math.pi,2))))
        mi = max(oldMi,percs[-1]+beta*math.log(-math.log(0.95)))
        return mi, beta, percs

    def prepare_learner(self):
        self.oldStopper = self.learner.ruleFinder.ruleStoppingValidator
        self.evaluator = self.learner.ruleFinder.evaluator
        self.refiner = self.learner.ruleFinder.refiner
        self.validator = self.learner.ruleFinder.validator
        self.ruleFilter = self.learner.ruleFinder.ruleFilter
        self.learner.ruleFinder.validator = None
        self.learner.ruleFinder.evaluator = orange.RuleEvaluator_LRS()
        self.learner.ruleFinder.evaluator.storeRules = True
        self.learner.ruleFinder.ruleStoppingValidator = orange.RuleValidator_LRS(alpha=1.0)
        self.learner.ruleFinder.ruleStoppingValidator.max_rule_complexity = 0
        self.learner.ruleFinder.refiner = orange.RuleBeamRefiner_Selector()
        self.learner.ruleFinder.ruleFilter = orange.RuleBeamFilter_Width(width = 1)


    def restore_learner(self):
        self.learner.ruleFinder.evaluator = self.evaluator
        self.learner.ruleFinder.ruleStoppingValidator = self.oldStopper
        self.learner.ruleFinder.refiner = self.refiner
        self.learner.ruleFinder.validator = self.validator
        self.learner.ruleFinder.ruleFilter = self.ruleFilter

    def computeEVD(self, data, weightID=0, target_class=0, progress=None):
        # initialize random seed to make experiments repeatable
        random.seed(self.randomseed)

        # prepare learned for distribution computation        
        self.prepare_learner()

        # loop through N (sampling repetitions)
        extremeDists=[(0, 1, [])]
        self.learner.ruleFinder.ruleStoppingValidator.max_rule_complexity = self.oldStopper.max_rule_complexity
        maxVals = [[] for l in range(self.oldStopper.max_rule_complexity)]
        for d_i in range(self.n):
            if not progress:
                print d_i,
            else:
                progress(float(d_i)/self.n, None)                
            # create data set (remove and randomize)
            tempData = self.createRandomDataSet(data)
            self.learner.ruleFinder.evaluator.rules = orange.RuleList()
            # Next, learn a rule
            self.learner.ruleFinder(tempData,weightID,target_class, orange.RuleList())
            for l in range(self.oldStopper.max_rule_complexity):
                qs = [r.quality for r in self.learner.ruleFinder.evaluator.rules if r.complexity == l+1]
                if qs:
                    maxVals[l].append(max(qs))
                else:
                    maxVals[l].append(0)

        mu, beta = 1.0, 1.0
        for mi,m in enumerate(maxVals):
            mu, beta, perc = self.compParameters(m,mu,beta)
            extremeDists.append((mu, beta, perc))
            extremeDists.extend([(0,1,[])]*(mi))

        self.restore_learner()
        return self.createEVDistList(extremeDists)

# ************************* #
# Rule based classification #
# ************************* #

class CrossValidation:
    def __init__(self, folds=5, randomGenerator = 150):
        self.folds = folds
        self.randomGenerator = randomGenerator

    def __call__(self, learner, examples, weight):
        res = orngTest.crossValidation([learner], (examples, weight), folds = self.folds, randomGenerator = self.randomGenerator)
        return self.get_prob_from_res(res, examples)

    def get_prob_from_res(self, res, examples):
        probDist = orange.DistributionList()
        for tex in res.results:
            d = orange.Distribution(examples.domain.classVar)
            for di in range(len(d)):
                d[di] = tex.probabilities[0][di]
            probDist.append(d)
        return probDist

class PILAR:
    """ PILAR (Probabilistic improvement of learning algorithms with rules) """
    def __init__(self, alternative_learner = None, min_cl_sig = 0.5, min_beta = 0.0, set_prefix_rules = False, optimize_betas = True):
        self.alternative_learner = alternative_learner
        self.min_cl_sig = min_cl_sig
        self.min_beta = min_beta
        self.set_prefix_rules = set_prefix_rules
        self.optimize_betas = optimize_betas
        self.selected_evaluation = CrossValidation(folds=5)

    def __call__(self, rules, examples, weight=0):
        rules = self.add_null_rule(rules, examples, weight)
        if self.alternative_learner:
            probDist = self.selected_evaluation(self.alternative_learner, examples, weight)
            classifier = self.alternative_learner(examples,weight)
##            probDist = orange.DistributionList()
##            for e in examples:
##                probDist.append(classifier(e,orange.GetProbabilities))
            cl = orange.RuleClassifier_logit(rules, self.min_cl_sig, self.min_beta, examples, weight, self.set_prefix_rules, self.optimize_betas, classifier, probDist)
        else:
            cl = orange.RuleClassifier_logit(rules, self.min_cl_sig, self.min_beta, examples, weight, self.set_prefix_rules, self.optimize_betas)

##        print "result"
        for ri,r in enumerate(cl.rules):
            cl.rules[ri].setattr("beta",cl.ruleBetas[ri])
##            if cl.ruleBetas[ri] > 0:
##                print orngCN2.ruleToString(r), r.quality, cl.ruleBetas[ri]
        cl.all_rules = cl.rules
        cl.rules = self.sortRules(cl.rules)
        cl.ruleBetas = [r.beta for r in cl.rules]
        cl.setattr("data", examples)
        return cl

    def add_null_rule(self, rules, examples, weight):
        for cl in examples.domain.classVar:
            tmpRle = orange.Rule()
            tmpRle.filter = orange.Filter_values(domain = examples.domain)
            tmpRle.parentRule = None
            tmpRle.filterAndStore(examples,weight,int(cl))
            tmpRle.quality = tmpRle.classDistribution[int(cl)]/tmpRle.classDistribution.abs
            rules.append(tmpRle)
        return rules
        
    def sortRules(self, rules):
        newRules = orange.RuleList()
        foundRule = True
        while foundRule:
            foundRule = False
            bestRule = None
            for r in rules:
                if r in newRules:
                    continue
                if r.beta < 0.01 and r.beta > -0.01:
                    continue
                if not bestRule:
                    bestRule = r
                    foundRule = True
                    continue
                if len(r.filter.conditions) < len(bestRule.filter.conditions):
                    bestRule = r
                    foundRule = True
                    continue
                if len(r.filter.conditions) ==  len(bestRule.filter.conditions) and r.beta > bestRule.beta:
                    bestRule = r
                    foundRule = True
                    continue
            if bestRule:
                newRules.append(bestRule)
        return newRules     


class CN2UnorderedClassifier(orange.RuleClassifier):
    """ Classification from rules as in CN2. """
    def __init__(self, rules, examples, weightID = 0, **argkw):
        self.rules = rules
        self.examples = examples
        self.weightID = weightID
        self.prior = orange.Distribution(examples.domain.classVar, examples, weightID)
        self.__dict__.update(argkw)

    def __call__(self, example, result_type=orange.GetValue, retRules = False):
        # iterate through the set of induced rules: self.rules and sum their distributions 
        ret_dist = self.sum_distributions([r for r in self.rules if r(example)])
        # normalize
        a = sum(ret_dist)
        for ri, r in enumerate(ret_dist):
            ret_dist[ri] = ret_dist[ri]/a
##        ret_dist.normalize()
        # return value
        if result_type == orange.GetValue:
          return ret_dist.modus()
        if result_type == orange.GetProbabilities:
          return ret_dist
        return (ret_dist.modus(),ret_dist)

    def sum_distributions(self, rules):
        if not rules:
            return self.prior
        empty_disc = orange.Distribution(rules[0].examples.domain.classVar)
        for r in rules:
            for i,d in enumerate(r.classDistribution):
                empty_disc[i] = empty_disc[i] + d
        return empty_disc

    def __str__(self):
        retStr = ""
        for r in self.rules:
            retStr += orngCN2.ruleToString(r)+" "+str(r.classDistribution)+"\n"
        return retStr


class RuleClassifier_bestRule(orange.RuleClassifier):
    """ A very simple classifier, it takes the best rule of each class and normalizes probabilities. """
    def __init__(self, rules, examples, weightID = 0, **argkw):
        self.rules = rules
        self.examples = examples
        self.apriori = orange.Distribution(examples.domain.classVar,examples,weightID)
        self.aprioriProb = [a/self.apriori.abs for a in self.apriori]
        self.weightID = weightID
        self.__dict__.update(argkw)
        self.defaultClassIndex = -1

    def __call__(self, example, result_type=orange.GetValue, retRules = False):
        example = orange.Example(self.examples.domain,example)
        tempDist = orange.Distribution(example.domain.classVar)
        bestRules = [None]*len(example.domain.classVar.values)

        for r in self.rules:
            if r(example) and not self.defaultClassIndex == int(r.classifier.defaultVal) and \
               (not bestRules[int(r.classifier.defaultVal)] or r.quality>tempDist[r.classifier.defaultVal]):
                tempDist[r.classifier.defaultVal] = r.quality
                bestRules[int(r.classifier.defaultVal)] = r
        for b in bestRules:
            if b:
                used = getattr(b,"used",0.0)
                b.setattr("used",used+1)
        nonCovPriorSum = sum([tempDist[i] == 0. and self.aprioriProb[i] or 0. for i in range(len(self.aprioriProb))])
        if tempDist.abs < 1.:
            residue = 1. - tempDist.abs
            for a_i,a in enumerate(self.aprioriProb):
                if tempDist[a_i] == 0.:
                    tempDist[a_i]=self.aprioriProb[a_i]*residue/nonCovPriorSum
            finalDist = tempDist #orange.Distribution(example.domain.classVar)
        else:
            tempDist.normalize() # prior probability
            tmpExamples = orange.ExampleTable(self.examples)
            for r in bestRules:
                if r:
                    tmpExamples = r.filter(tmpExamples)
            tmpDist = orange.Distribution(tmpExamples.domain.classVar,tmpExamples,self.weightID)
            tmpDist.normalize()
            probs = [0.]*len(self.examples.domain.classVar.values)
            for i in range(len(self.examples.domain.classVar.values)):
                probs[i] = tmpDist[i]+tempDist[i]*2
            finalDist = orange.Distribution(self.examples.domain.classVar)
            for cl_i,cl in enumerate(self.examples.domain.classVar):
                finalDist[cl] = probs[cl_i]
            finalDist.normalize()
                
        if retRules: # Do you want to return rules with classification?
            if result_type == orange.GetValue:
              return (finalDist.modus(),bestRules)
            if result_type == orange.GetProbabilities:
              return (finalDist, bestRules)
            return (finalDist.modus(),finalDist, bestRules)
        if result_type == orange.GetValue:
          return finalDist.modus()
        if result_type == orange.GetProbabilities:
          return finalDist
        return (finalDist.modus(),finalDist)


        