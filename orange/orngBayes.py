import orange, time

class BayesLearner(orange.Learner):
    def __new__(cls, examples = None, weightID = 0, **argkw):
        self = orange.Learner.__new__(cls, **argkw)
        if examples:
            self.__init__(**argkw)
            return self.__call__(examples, weightID)
        else:
            return self
        
    def __init__(self, **argkw):
        self.learner = None
        self.__dict__.update(argkw)

    def __setattr__(self, name, value):
        if name in ["m", "estimatorConstructor", "conditionalEstimatorConstructor", "conditionalEstimatorConstructorContinuous"]:
            self.learner = None
        self.__dict__[name] = value

    def __call__(self, examples, weight=0):
        if not self.learner:
            self.learner = self.createInstance()
        return self.learner(examples, weight)

    def createInstance(self):
        bayes = orange.BayesLearner()
        if hasattr(self, "estimatorConstructor"):
            bayes.estimatorConstructor = self.estimatorConstructor
            if hasattr(self, "m"):
                if hasattr(bayes.estimatorConstructor, "m"):
                    raise AttributeError, "invalid combination of attributes: 'estimatorConstructor' does not expect 'm'"
                else:
                    self.estimatorConstructor.m = self.m
        elif hasattr(self, "m"):
            bayes.estimatorConstructor = orange.ProbabilityEstimatorConstructor_m(m = self.m)

        if hasattr(self, "conditionalEstimatorConstructor"):
            bayes.conditionalEstimatorConstructor = self.conditionalEstimatorConstructor
        elif bayes.estimatorConstructor:
            bayes.conditionalEstimatorConstructor = orange.ConditionalProbabilityEstimatorConstructor_ByRows()
            bayes.conditionalEstimatorConstructor.estimatorConstructor=bayes.estimatorConstructor
            
        if hasattr(self, "conditionalEstimatorConstructorContinuous"):
            bayes.conditionalEstimatorConstructorContinuous = self.conditionalEstimatorConstructorContinuous
            
        return bayes
            

def printModel(bayesclassifier):
    nValues=len(bayesclassifier.classVar.values)
    frmtStr=' %10.3f'*nValues
    classes=" "*20+ ((' %10s'*nValues) % tuple([i[:10] for i in bayesclassifier.classVar.values]))
    print classes
    print "class probabilities "+(frmtStr % tuple(bayesclassifier.distribution))
    print

    for i in bayesclassifier.conditionalDistributions:
        print "Attribute", i.variable.name
        print classes
        for v in range(len(i.variable.values)):
            print ("%20s" % i.variable.values[v][:20]) + (frmtStr % tuple(i[v]))
        print
