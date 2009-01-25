import orange

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
        self.__dict__ = kwds
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
        self.__dict__ = kwds
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