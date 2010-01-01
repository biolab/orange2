import orange
import random, math

def ruleToString(rule, showDistribution = True):
    def selectSign(oper):
        if oper == orange.ValueFilter_continuous.Less:
            return "<"
        elif oper == orange.ValueFilter_continuous.LessEqual:
            return "<="
        elif oper == orange.ValueFilter_continuous.Greater:
            return ">"
        elif oper == orange.ValueFilter_continuous.GreaterEqual:
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
        if type(c) == orange.ValueFilter_discrete:
            ret += domain[c.position].name + "=" + str([domain[c.position].values[int(v)] for v in c.values])
        elif type(c) == orange.ValueFilter_continuous:
            ret += domain[c.position].name + selectSign(c.oper) + str(c.ref)
    if rule.classifier and type(rule.classifier) == orange.DefaultClassifier and rule.classifier.defaultVal:
        ret = ret + " THEN "+domain.classVar.name+"="+\
        str(rule.classifier.defaultValue)
        if showDistribution:
            ret += str(rule.classDistribution)
    elif rule.classifier and type(rule.classifier) == orange.DefaultClassifier and type(domain.classVar) == orange.EnumVariable:
        ret = ret + " THEN "+domain.classVar.name+"="+\
        str(rule.classDistribution.modus())
        if showDistribution:
            ret += str(rule.classDistribution)
    return ret        

class LaplaceEvaluator(orange.RuleEvaluator):
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

class WRACCEvaluator(orange.RuleEvaluator):
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

class mEstimate(orange.RuleEvaluator):
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

class RuleStopping_apriori(orange.RuleStoppingCriteria):
    def __init__(self, apriori=None):
        self.apriori =  None
        
    def __call__(self,rules,rule,examples,data):
        if not self.apriori:
            return False
        if not type(rule.classifier) == orange.DefaultClassifier:
            return False
        ruleAcc = rule.classDistribution[rule.classifier.defaultVal]/rule.classDistribution.abs
        aprioriAcc = self.apriori[rule.classifier.defaultVal]/self.apriori.abs
        if ruleAcc>aprioriAcc:
            return False
        return True

class LengthValidator(orange.RuleValidator):
    """ prune rules with more conditions than self.length. """
    def __init__(self, length=-1):
        self.length = length
        
    def __call__(self, rule, data, weightID, targetClass, apriori):
        if self.length >= 0:
            return len(rule.filter.conditions) <= self.length
        return True    
    
def CN2Learner(examples = None, weightID=0, **kwds):
    cn2 = CN2LearnerClass(**kwds)
    if examples:
        return cn2(examples, weightID)
    else:
        return cn2

def supervisedClassCheck(examples):
    if not examples.domain.classVar:
        raise Exception("Class variable is required!")
    if examples.domain.classVar.varType == orange.VarTypes.Continuous:
        raise Exception("CN2 requires a discrete class!")
    
class CN2LearnerClass(orange.RuleLearner):
    def __init__(self, evaluator = orange.RuleEvaluator_Entropy(), beamWidth = 5, alpha = 1.0, **kwds):
        self.__dict__.update(kwds)
        self.ruleFinder = orange.RuleBeamFinder()
        self.ruleFinder.ruleFilter = orange.RuleBeamFilter_Width(width = beamWidth)
        self.ruleFinder.evaluator = evaluator
        self.ruleFinder.validator = orange.RuleValidator_LRS(alpha = alpha)
        
    def __call__(self, examples, weight=0):
        supervisedClassCheck(examples)
        
        cl = orange.RuleLearner.__call__(self,examples,weight)
        rules = cl.rules
        return CN2Classifier(rules, examples, weight)

class CN2Classifier(orange.RuleClassifier):
    def __init__(self, rules, examples, weightID = 0, **argkw):
        self.rules = rules
        self.examples = examples
        self.__dict__.update(argkw)
        self.prior = orange.Distribution(examples.domain.classVar, examples)

    def __call__(self, example, result_type=orange.GetValue):
        classifier = None
        for r in self.rules:
         #   r.filter.domain = example.domain
            if r(example) and r.classifier:
                classifier = r.classifier
                classifier.defaultDistribution = r.classDistribution
                break
        if not classifier:
            classifier = orange.DefaultClassifier(example.domain.classVar, self.prior.modus())
            classifier.defaultDistribution = self.prior

        if result_type == orange.GetValue:
          return classifier(example)
        if result_type == orange.GetProbabilities:
          return classifier.defaultDistribution
        return (classifier(example),classifier.defaultDistribution)

    def __str__(self):
        retStr = ruleToString(self.rules[0])+" "+str(self.rules[0].classDistribution)+"\n"
        for r in self.rules[1:]:
            retStr += "ELSE "+ruleToString(r)+" "+str(r.classDistribution)+"\n"
        return retStr


def CN2UnorderedLearner(examples = None, weightID=0, **kwds):
    cn2 = CN2UnorderedLearnerClass(**kwds)
    if examples:
        return cn2(examples, weightID)
    else:
        return cn2

# Kako nastavim v c++, da mi ni potrebno dodati imena
class CN2UnorderedLearnerClass(orange.RuleLearner):
    def __init__(self, evaluator = orange.RuleEvaluator_Laplace(), beamWidth = 5, alpha = 1.0, **kwds):
        self.__dict__.update(kwds)
        self.ruleFinder = orange.RuleBeamFinder()
        self.ruleFinder.ruleFilter = orange.RuleBeamFilter_Width(width = beamWidth)
        self.ruleFinder.evaluator = evaluator
        self.ruleFinder.validator = orange.RuleValidator_LRS(alpha = alpha)
        self.ruleFinder.ruleStoppingValidator = orange.RuleValidator_LRS(alpha = 1.0)
        self.ruleStopping = RuleStopping_apriori()
        self.dataStopping = orange.RuleDataStoppingCriteria_NoPositives()
        
    def __call__(self, examples, weight=0):
        supervisedClassCheck(examples)
        
        rules = orange.RuleList()
        self.ruleStopping.apriori = orange.Distribution(examples.domain.classVar,examples)
        progress=getattr(self,"progressCallback",None)
        if progress:
            progress.start = 0.0
            progress.end = 0.0
            distrib = orange.Distribution(examples.domain.classVar, examples, weight)
            distrib.normalize()
        for targetClass in examples.domain.classVar:
            if progress:
                progress.start = progress.end
                progress.end += distrib[targetClass]
            self.targetClass = targetClass
            cl = orange.RuleLearner.__call__(self,examples,weight)
            for r in cl.rules:
                rules.append(r)
        if progress:
            progress(1.0,None)
        return CN2UnorderedClassifier(rules, examples, weight)


class CN2UnorderedClassifier(orange.RuleClassifier):
    def __init__(self, rules = None, examples = None, weightID = 0, **argkw):
        self.rules = rules
        self.examples = examples
        self.weightID = weightID
        self.__dict__.update(argkw)
        if examples:
            self.prior = orange.Distribution(examples.domain.classVar, examples)

    def __call__(self, example, result_type=orange.GetValue, retRules = False):
        def add(disc1, disc2, sumd):
            disc = orange.DiscDistribution(disc1)
            sumdisc = sumd
            for i,d in enumerate(disc):
                disc[i]+=disc2[i]
                sumdisc += disc2[i]
            return disc, sumdisc

        # create empty distribution
        retDist = orange.DiscDistribution(self.examples.domain.classVar)
        covRules = orange.RuleList()
        # iterate through examples - add distributions
        sumdisc = 0.
        for r in self.rules:
            if r(example) and r.classDistribution:
                retDist, sumdisc = add(retDist, r.classDistribution, sumdisc)
                covRules.append(r)
        if not sumdisc:
            retDist = self.prior
            sumdisc = self.prior.abs
        for c in self.examples.domain.classVar:
            retDist[c] /= sumdisc
        if retRules:
            if result_type == orange.GetValue:
              return (retDist.modus(), covRules)
            if result_type == orange.GetProbabilities:
              return (retDist, covRules)
            return (retDist.modus(),retDist,covRules)
        if result_type == orange.GetValue:
          return retDist.modus()
        if result_type == orange.GetProbabilities:
          return retDist
        return (retDist.modus(),retDist)

    def __str__(self):
        retStr = ""
        for r in self.rules:
            retStr += ruleToString(r)+" "+str(r.classDistribution)+"\n"
        return retStr

class RuleClassifier_bestRule(orange.RuleClassifier):
    def __init__(self, rules, examples, weightID = 0, **argkw):
        self.rules = rules
        self.examples = examples
        self.__dict__.update(argkw)
        self.prior = orange.Distribution(examples.domain.classVar, examples)

    def __call__(self, example, result_type=orange.GetValue):
        retDist = orange.Distribution(example.domain.classVar)
        bestRule = None
        for r in self.rules:
            if r(example) and (not bestRule or r.quality>bestRule.quality):
                for v_i,v in enumerate(example.domain.classVar):
                    retDist[v_i] = r.classDistribution[v_i]
                bestRule = r
        if not bestRule:
            retDist = self.prior
        else:
            bestRule.used += 1
        retDist.normalize()
        # return classifier(example, result_type=result_type)
        if result_type == orange.GetValue:
          return retDist.modus()
        if result_type == orange.GetProbabilities:
          return retDist
        return (retDist.modus(),retDist)

    def __str__(self):
        retStr = ""
        for r in self.rules:
            retStr += ruleToString(r)+" "+str(r.classDistribution)+"\n"
        return retStr    

class CovererAndRemover_multWeights(orange.RuleCovererAndRemover):
    def __init__(self, mult = 0.7):
        self.mult = mult
    def __call__(self, rule, examples, weights, targetClass):
        if not weights:
            weights = orange.newmetaid()
            examples.addMetaAttribute(weights,1.)
            examples.domain.addmeta(weights, orange.FloatVariable("weights-"+str(weights)), True)
        newWeightsID = orange.newmetaid()
        examples.addMetaAttribute(newWeightsID,1.)
        examples.domain.addmeta(newWeightsID, orange.FloatVariable("weights-"+str(newWeightsID)), True)
        for example in examples:
            if rule(example) and example.getclass() == rule.classifier(example,orange.GetValue):
                example[newWeightsID]=example[weights]*self.mult
            else:
                example[newWeightsID]=example[weights]
        return (examples,newWeightsID)

class CovererAndRemover_addWeights(orange.RuleCovererAndRemover):
    def __call__(self, rule, examples, weights, targetClass):
        if not weights:
            weights = orange.newmetaid()
            examples.addMetaAttribute(weights,1.)
            examples.domain.addmeta(weights, orange.FloatVariable("weights-"+str(weights)), True)
        try:
            coverage = examples.domain.getmeta("Coverage")
        except:
            coverage = orange.FloatVariable("Coverage")
            examples.domain.addmeta(orange.newmetaid(),coverage, True)
            examples.addMetaAttribute(coverage,0.0)
        newWeightsID = orange.newmetaid()
        examples.addMetaAttribute(newWeightsID,1.)
        examples.domain.addmeta(newWeightsID, orange.FloatVariable("weights-"+str(newWeightsID)), True)
        for example in examples:
            if rule(example) and example.getclass() == rule.classifier(example,orange.GetValue):
                try:
                    example[coverage]+=1.0
                except:
                    example[coverage]=1.0
                example[newWeightsID]=1.0/(example[coverage]+1)
            else:
                example[newWeightsID]=example[weights]
        return (examples,newWeightsID)

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
                if type(c1) == orange.ValueFilter_discrete:
                    if not type(c1.values[0]) == type(c2.values[0]): continue
                    if not c1.values[0] == c2.values[0]: continue # same value?
                if type(c1) == orange.ValueFilter_continuous:
                    if not c1.oper == c2.oper: continue # same operator?
                    if not c1.ref == c2.ref: continue #same threshold?
                found=True
                break
            except:
                pass
        if not found:
            return False
    return True

class noDuplicates_validator(orange.RuleValidator):
    def __init__(self,alpha=.05,min_coverage=0,max_rule_length=0,rules=orange.RuleList()):
        self.rules = rules
        self.validator = orange.RuleValidator_LRS(alpha=alpha,min_coverage=min_coverage,max_rule_length=max_rule_length)
        
    def __call__(self, rule, data, weightID, targetClass, apriori):
        if rule_in_set(rule,self.rules):
            return False
        return bool(self.validator(rule,data,weightID,targetClass,apriori))
                
class ruleSt_setRules(orange.RuleStoppingCriteria):
    def __init__(self,validator):
        self.ruleStopping = orange.RuleStoppingCriteria_NegativeDistribution()
        self.validator = validator

    def __call__(self,rules,rule,examples,data):        
        ru_st = self.ruleStopping(rules,rule,examples,data)
        if not ru_st:
            self.validator.rules.append(rule)
        return bool(ru_st)

def CN2SDUnorderedLearner(examples = None, weightID=0, **kwds):
    cn2 = CN2SDUnorderedLearnerClass(**kwds)
    if examples:
        return cn2(examples, weightID)
    else:
        return cn2
    
class CN2SDUnorderedLearnerClass(CN2UnorderedLearnerClass):
    def __init__(self, evaluator = WRACCEvaluator(), beamWidth = 5, alpha = 0.05, mult=0.7, **kwds):
        CN2UnorderedLearnerClass.__init__(self, evaluator = evaluator,
                                          beamWidth = beamWidth, alpha = alpha, **kwds)
        self.coverAndRemove = CovererAndRemover_multWeights(mult=mult)

    def __call__(self, examples, weight=0):        
        supervisedClassCheck(examples)
        
        oldExamples = orange.ExampleTable(examples)
        classifier = CN2UnorderedLearnerClass.__call__(self,examples,weight)
        for r in classifier.rules:
            r.filterAndStore(oldExamples,weight,r.classifier.defaultVal)
        return classifier

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
    newData = orange.ExampleTable(data)
    # shuffle data
    cl_num = newData.toNumeric("C")
    random.shuffle(cl_num[0][:,0])
    clData = orange.ExampleTable(orange.Domain([newData.domain.classVar]),cl_num[0])
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
    learner.ruleFinder.evaluator = orange.RuleEvaluator_LRS()
    learner.ruleFinder.evaluator.storeRules = True
    learner.ruleFinder.ruleStoppingValidator = orange.RuleValidator_LRS(alpha=1.0)
    learner.ruleFinder.ruleStoppingValidator.max_rule_complexity = 0  

    # loop through N (sampling repetitions)
    maxVals = []
    for d_i in range(N):
        # create data set (remove and randomize)
        tempData = createRandomDataSet(data)
        learner.ruleFinder.evaluator.rules = orange.RuleList()
        # Next, learn a rule
        bestRule = learner.ruleFinder(tempData,weight,targetClass,orange.RuleList())
        maxVals.append(bestRule.quality)
    extremeDists=[compParameters(maxVals,1.0,1.0)]

    #####################
    ## Restore learner ##
    #####################
    learner.ruleFinder.evaluator = evaluator
    learner.ruleFinder.ruleStoppingValidator = oldStopper
    return extremeDists

def createEVDistList(evdList):
    l = orange.EVDistList()
    for el in evdList:
        l.append(orange.EVDist(mu=el[0],beta=el[1],percentiles=el[2]))
    return l

class CovererAndRemover_Prob(orange.RuleCovererAndRemover):
    """ This class impements probabilistic covering. """
    def __init__(self, probAttribute=None, sigAttribute=None):
        self.indices = None
        self.probAttribute = probAttribute
        self.bestRule = []

    def initialize(self, examples, weightID, targetClass, apriori):
        self.bestRule = [None]*len(examples)
        self.probAttribute = orange.newmetaid()
        examples.addMetaAttribute(self.probAttribute,-1.e-6)
        examples.domain.addmeta(self.probAttribute, orange.FloatVariable("Probs"))
        for example in examples:
##            if targetClass<0 or (example.getclass() == targetClass):
            example[self.probAttribute] = apriori[targetClass]/apriori.abs
        return examples

    def getBestRules(self, currentRules, examples, weightID):
        bestRules = orange.RuleList()
        for r in currentRules:
            if hasattr(r.learner, "argumentRule") and not orngCN2.rule_in_set(r,bestRules):
                bestRules.append(r)
        for r_i,r in enumerate(self.bestRule):
            if r and not rule_in_set(r,bestRules) and examples[r_i].getclass()==r.classifier.defaultValue:
                bestRules.append(r)
        return bestRules

    def remainingExamplesP(self, examples, targetClass):
        pSum, pAll = 0.0, 0.0
        for ex in examples:
            if ex.getclass() == targetClass:
                pSum += ex[self.probAttribute]
                pAll += 1.0
        return pSum/pAll

    def __call__(self, rule, examples, weights, targetClass):
        if targetClass<0:
            for example_i, example in enumerate(examples):
                if rule(example) and rule.quality>example[self.probAttribute]-0.01:
                    example[self.probAttribute] = rule.quality+0.01
                    self.bestRule[example_i]=rule
        else:
            for example_i, example in enumerate(examples): #rule.classifier.defaultVal == example.getclass() and
                if rule(example) and rule.quality>example[self.probAttribute]:
                    example[self.probAttribute] = rule.quality+0.001
                    self.bestRule[example_i]=rule
##                if rule.classifier.defaultVal == example.getclass():
##                    print example[self.probAttribute]
        # compute factor
        return (examples,weights)

def add_sub_rules(rules, examples, weight, learner, dists):
    apriori = orange.Distribution(examples.domain.classVar,examples,weight)
    newRules = orange.RuleList()
    for r in rules:
        newRules.append(r)

    # loop through rules
    for r in rules:
        tmpList = orange.RuleList()
        tmpRle = r.clone()
        tmpRle.filter.conditions = []
        tmpRle.parentRule = None
        tmpRle.filterAndStore(examples,weight,r.classifier.defaultVal)
        tmpList.append(tmpRle)
        while tmpList and len(tmpList[0].filter.conditions) <= len(r.filter.conditions):
            tmpList2 = orange.RuleList()
            for tmpRule in tmpList:
                # evaluate tmpRule
                oldREP = learner.ruleFinder.evaluator.returnExpectedProb
                learner.ruleFinder.evaluator.returnExpectedProb = False
                learner.ruleFinder.evaluator.evDistGetter.dists = createEVDistList(dists[int(r.classifier.defaultVal)])
                tmpRule.quality = learner.ruleFinder.evaluator(tmpRule,examples,weight,r.classifier.defaultVal,apriori)
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
                        tmpRule2.filterAndStore(examples,weight,r.classifier.defaultVal)
                        if tmpRule2.classDistribution.abs < tmpRule.classDistribution.abs:
                            tmpList2.append(tmpRule2)
            tmpList = tmpList2
    for cl in examples.domain.classVar:
        tmpRle = orange.Rule()
        tmpRle.filter = orange.Filter_values(domain = examples.domain)
        tmpRle.parentRule = None
        tmpRle.filterAndStore(examples,weight,int(cl))
        tmpRle.quality = tmpRle.classDistribution[int(cl)]/tmpRle.classDistribution.abs
        newRules.append(tmpRle)
    return newRules

def CN2EVCUnorderedLearner(examples = None, weightID=0, **kwds):
    cn2 = CN2EVCUnorderedLearnerClass(**kwds)
    if examples:
        return cn2(examples, weightID)
    else:
        return cn2
    
class CN2EVCUnorderedLearnerClass(orange.RuleLearner):
    """This is implementation of CN2 + EVC as evaluation + LRC classification.
        Main parameters:
          -- ...
    """
    
    def __init__(self, m=2, opt_reduction=2, min_improved=1, min_improved_perc=0.0, nsampling=100, width=5,
                 rule_sig=0.05, att_sig=0.5, max_rule_complexity=5, min_coverage=5, add_sub_rules = True,
                 min_cl_sig = 0.5, min_beta = 0.0, set_prefix_rules = False, alternative_learner = None, **kwds):
        """
        Parameters:
            EVC related:
                m                   ... m-estimate to be corrected with EVC (default 2)
                opt_reduction       ... types of EVC correction; 0=no correction, 1=pessimistic, 2=normal (default 2)
                nsampling           ... number of samples in estimating extreme value distribution (for EVC) (default 100)

            Probabilistic covering:
                min_improved        ... minimal number of examples improved in probabilistic covering (default 1)
                min_improved_perc   ... minimal percentage of covered examples that need to be improved (default 0.0)

            Beam search:
                width               ... beam width (default 5)

            Rule Validation:
                rule_sig            ... minimal rule significance (default 1.0)
                att_sig             ... minimal attribute significance in rule (default 1.0)
                max_rule_complexity ... maximum number of conditions in rule (default 5)
                min_coverage        ... minimal number of covered examples (default 5)

            Classifier (LCR) related:
                add_sub_rules       ... add sub rules ? (default True)
                min_cl_sig          ... minimal significance of beta in classifier (default 0.5)
                min_beta            ... minimal beta value (default 0.0)
                set_prefix_rules    ... should ordered prefix rules be added? (default False)
                alternative_learner ... use rule-learner as a correction method for other machine learning methods. (default None)
        """
        
        # argument ID which is passed to abcn2 learner
        # two learners, learner with arguments and learners without
        self.ruleFinder = orange.RuleBeamFinder()
        self.ruleFilter = orange.RuleBeamFilter_Width(width=width)
        evdGet = orange.EVDistGetter_Standard()
        self.ruleFinder.evaluator = orange.RuleEvaluator_mEVC(m=m, evDistGetter = evdGet, min_improved = min_improved, min_improved_perc = min_improved_perc)
        self.ruleFinder.evaluator.returnExpectedProb = True
        self.ruleFinder.evaluator.optimismReduction = opt_reduction
        self.ruleFinder.evaluator.ruleAlpha = rule_sig
        self.ruleFinder.evaluator.attributeAlpha = att_sig
        self.ruleFinder.ruleStoppingValidator = orange.RuleValidator_LRS(alpha = 1.0, min_quality = 0., max_rule_complexity = max_rule_complexity - 1)        
        self.ruleFinder.evaluator.validator = orange.RuleValidator_LRS(alpha = 1.0, min_quality = 0., min_coverage=min_coverage)        
        self.coverAndRemove = CovererAndRemover_Prob()
        self.ruleStopping = None
        self.dataStopping = orange.RuleDataStoppingCriteria_NoPositives()        
        self.N = nsampling
        self.add_sub_rules = add_sub_rules
        self.min_cl_sig = min_cl_sig
        self.min_beta = min_beta
        self.set_prefix_rules = set_prefix_rules
        self.alternativeLearner = alternative_learner
        self.__dict__.update(kwds)
    
    def __call__(self, examples, weight=0):        
        supervisedClassCheck(examples)
        apriori = orange.Distribution(examples.domain.classVar, examples, weight)
        ruleSet = orange.RuleList()  # resulting set of rules

        # Progress bar in widgets 
        progress=getattr(self,"progressCallback",None)
        if progress:
            self.progressCallback = progress
            progress.start = 0.0
            progress.end = 0.0
            distrib = orange.Distribution(examples.domain.classVar, examples, weightID)
            distrib.normalize()

        # Main Loop
        temp_dists = []
        for cl_i,cl in enumerate(examples.domain.classVar):
            # rulesForClass ... rules for this class only
            rulesForClass = orange.RuleList()
            if progress:
                progress.start = progress.end
                progress.end += distrib[cl]

            # Compute EVD distribution if not set
            if getattr(self, "dists", None):
                self.ruleFinder.evaluator.evDistGetter.dists = createEVDistList(self.dists[cl_i])
                temp_dists.append(self.dists[cl_i])
            else:
                ds = computeDists(examples, weight=weight, targetClass=cl_i, N=self.N, learner=self)
                self.ruleFinder.evaluator.evDistGetter.dists = createEVDistList(ds)
                temp_dists.append(ds)
            examples = self.coverAndRemove.initialize(examples, weight, cl, apriori)
            self.ruleFinder.evaluator.probVar = examples.domain.getmeta(self.coverAndRemove.probAttribute)
            self.targetClass = cl
            # Learn rules
            while not self.dataStopping(examples,weight,cl):
                # Learn rule
                rule = self.learnRule(examples,weight,cl)
                if not rule or len(rule.filter.conditions)==0: # stop learning if no rule has been learned
                    break
                (examples,weight) = self.coverAndRemove(rule,examples,weight,cl)
                # add rule to rule set
                if not rule_in_set(rule,rulesForClass):
                    rulesForClass.append(rule)
                if progress:
                    progress(self.coverAndRemove.remainingExamplesP(examples,cl),None)
                else:
                    print "%4.2f,"%self.coverAndRemove.remainingExamplesP(examples,cl),
            if not progress:
                print
            ruleSet.extend(self.coverAndRemove.getBestRules(rulesForClass,examples,weight))
            if progress:
                progress(1.0,None)
            self.ruleFinder.evaluator.probVar = None
        if self.add_sub_rules:
            ruleSet = add_sub_rules(ruleSet, examples, weight, self, temp_dists)
        return self.LCR(ruleSet, examples, weight)            

    def learnRule(self,examples,weightID,targetClass):
        self.ruleFinder.evaluator.bestRule = None
        rule = self.ruleFinder(examples,weightID,targetClass,orange.RuleList())
        return self.ruleFinder.evaluator.bestRule

    def LCR(self, rules, examples, weight):
        if self.alternativeLearner:
            classifier = self.alternativeLearner(examples,weight)
            probDist = orange.DistributionList()
            for e in examples:
                probDist.append(classifier(e,orange.GetProbabilities))
            cl = orange.RuleClassifier_logit(rules, self.min_cl_sig, self.min_beta, examples, weight, self.set_prefix_rules, classifier, probDist)
        else:
            cl = orange.RuleClassifier_logit(rules, self.min_cl_sig, self.min_beta, examples, weight, self.set_prefix_rules)

        for ri,r in enumerate(cl.rules):
            cl.rules[ri].setattr("beta",cl.ruleBetas[ri])
        cl.rules = self.sortRules(cl.rules)
        cl.ruleBetas = [r.beta for r in cl.rules]
        cl.data = examples
        return cl

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
        
