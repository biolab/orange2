import Orange.core as orange

class StepwiseLearner_Class:
  def __init__(self, **kwds):
    import orngStat
    self.removeThreshold = 0.3
    self.addThreshold = 0.2
    self.stat, self.statsign = orngStat.CA, 1
    self.__dict__.update(kwds)

  def __call__(self, examples, weightID = 0, **kwds):
    import orngTest, orngStat, statc
    
    self.__dict__.update(kwds)

    if self.removeThreshold < self.addThreshold:
        raise ValueError("'removeThreshold' should be larger or equal to 'addThreshold'")

    classVar = examples.domain.classVar
    
    indices = orange.MakeRandomIndicesCV(examples, folds = getattr(self, "folds", 10))
    domain = orange.Domain([], classVar)

    res = orngTest.testWithIndices([self.learner], orange.ExampleTable(domain, examples), indices)
    
    oldStat = self.stat(res)[0]
    oldStats = [self.stat(x)[0] for x in orngStat.splitByIterations(res)]
    print ".", oldStat, domain
    stop = False
    while not stop:
        stop = True
        if len(domain.attributes)>=2:
            bestStat = None
            for attr in domain.attributes:
                newdomain = orange.Domain(filter(lambda x: x!=attr, domain.attributes), classVar)
                res = orngTest.testWithIndices([self.learner], (orange.ExampleTable(newdomain, examples), weightID), indices)
                
                newStat = self.stat(res)[0]
                newStats = [self.stat(x)[0] for x in orngStat.splitByIterations(res)] 
                print "-", newStat, newdomain
                ## If stat has increased (ie newStat is better than bestStat)
                if not bestStat or cmp(newStat, bestStat) == self.statsign:
                    if cmp(newStat, oldStat) == self.statsign:
                        bestStat, bestStats, bestAttr = newStat, newStats, attr
                    elif statc.wilcoxont(oldStats, newStats)[1] > self.removeThreshold:
                            bestStat, bestAttr, bestStats = newStat, newStats, attr
            if bestStat:
                domain = orange.Domain(filter(lambda x: x!=bestAttr, domain.attributes), classVar)
                oldStat, oldStats = bestStat, bestStats
                stop = False
                print "removed", bestAttr.name

        bestStat, bestAttr = oldStat, None
        for attr in examples.domain.attributes:
            if not attr in domain.attributes:
                newdomain = orange.Domain(domain.attributes + [attr], classVar)
                res = orngTest.testWithIndices([self.learner], (orange.ExampleTable(newdomain, examples), weightID), indices)
                
                newStat = self.stat(res)[0]
                newStats = [self.stat(x)[0] for x in orngStat.splitByIterations(res)] 
                print "+", newStat, newdomain

                ## If stat has increased (ie newStat is better than bestStat)
                if cmp(newStat, bestStat) == self.statsign and statc.wilcoxont(oldStats, newStats)[1] < self.addThreshold:
                    bestStat, bestStats, bestAttr = newStat, newStats, attr
        if bestAttr:
            domain = orange.Domain(domain.attributes + [bestAttr], classVar)
            oldStat, oldStats = bestStat, bestStats
            stop = False
            print "added", bestAttr.name

    return self.learner(orange.ExampleTable(domain, examples), weightID)

def StepwiseLearner(examples = None, weightID = None, **argkw):
    sl = apply(StepwiseLearner_Class, (), argkw)
    if examples:
        return sl(examples, weightID)
    else:
        return sl
