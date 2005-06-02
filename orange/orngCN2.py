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
        sumDist = rule.classDistribution.cases
        if not sumDist or (targetClass>-1 and not rule.classDistribution[targetClass]):
            return 0.
        # get distribution
        if targetClass>-1:
            p = rule.classDistribution[targetClass]+self.m*apriori[targetClass]/apriori.cases
            p = p / (rule.classDistribution.cases + self.m)
        else:
            p = max(rule.classDistribution)+2*self.m*apriori[rule.classDistribution.modus()]/apriori.cases
            p = p / (rule.classDistribution.cases + self.m)      
        return p
    
def CN2Learner(examples = None, weightID=0, **kwds):
    cn2 = CN2LearnerClass(**kwds)
    if examples:
        return cn2(examples, weightID)
    else:
        return cn2

class CN2LearnerClass(orange.RuleLearner):
    def __init__(self, evaluator = orange.RuleEvaluator_Entropy(), beamWidth = 5, alpha = 0.05, **kwds):
        self.__dict__ = kwds
        self.ruleFinder = orange.RuleBeamFinder()
        self.ruleFinder.ruleFilter = orange.RuleBeamFilter_Width(width = beamWidth)
        self.ruleFinder.evaluator = evaluator
        self.ruleFinder.validator = orange.RuleValidator_LRS(alpha = alpha)
        
    def __call__(self, examples, weight=0):
        if examples.domain.classVar.varType == orange.VarTypes.Continuous:
            print "CN2 can learn only on discrete class!"
            return
        cl = orange.RuleLearner.__call__(self,examples,weight)
        rules = cl.rules
        return CN2Classifier(rules, examples, weight)

class CN2Classifier(orange.RuleClassifier):
    def __init__(self, rules, examples, weightID = 0):
        self.rules = rules
        self.examples = examples
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
    def __init__(self, evaluator = LaplaceEvaluator(), beamWidth = 5, alpha = 0.05, **kwds):
        self.__dict__ = kwds
        self.ruleFinder = orange.RuleBeamFinder()
        self.ruleFinder.ruleFilter = orange.RuleBeamFilter_Width(width = beamWidth)
        self.ruleFinder.evaluator = evaluator
        self.ruleFinder.validator = orange.RuleValidator_LRS(alpha = alpha)
        self.ruleStopping = orange.RuleStoppingCriteria_NegativeDistribution()
        self.dataStopping = orange.RuleDataStoppingCriteria_NoPositives()
        
    def __call__(self, examples, weight=0):
        if examples.domain.classVar.varType == orange.VarTypes.Continuous:
            print "CN2 can learn only on discrete class!"
            return
        rules = orange.RuleList()
        for targetClass in examples.domain.classVar:
            self.targetClass = targetClass
            cl = orange.RuleLearner.__call__(self,examples,weight)
            for r in cl.rules:
                rules.append(r)
        return CN2UnorderedClassifier(rules, examples, weight)

class CN2UnorderedClassifier(orange.RuleClassifier):
    def __init__(self, rules, examples, weightID = 0):
        self.rules = rules
        self.examples = examples
        self.prior = orange.Distribution(examples.domain.classVar, examples)

    def __call__(self, example, result_type=orange.GetValue):
        def add(disc1, disc2):
            disc = orange.DiscDistribution(disc1)
            for i,d in enumerate(disc):
                disc[i]+=disc2[i]
            return disc
        
        # create empty distribution
        retDist = orange.DiscDistribution(example.domain.classVar)

        # iterate through examples - add distributions
        for r in self.rules:
            if r(example) and r.classDistribution:
                retDist = add(retDist, r.classDistribution)


        if retDist.cases == 0:
            classifier = orange.DefaultClassifier(example.domain.classVar, self.prior.modus(),
                                                  defaultDistribution = self.prior)
        else:
            classifier = orange.DefaultClassifier(example.domain.classVar, retDist.modus(),
                                                  defaultDistribution = retDist)
        classifier.defaultDistribution.normalize()
        # return classifier(example, result_type=result_type)
        if result_type == orange.GetValue:
          return classifier(example)
        if result_type == orange.GetProbabilities:
          return classifier.defaultDistribution
        return (classifier(example),classifier.defaultDistribution)

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
            for example in examples:
                example[weights] = 1.
        newWeightsID = orange.newmetaid()
        for example in examples:
            if rule(example) and example.getclass() == rule.classifier(example,orange.GetValue):
                example[newWeightsID]=example[weights]*self.mult
            else:
                example[newWeightsID]=example[weights]
        return (examples,newWeightsID)
                
    
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
        
