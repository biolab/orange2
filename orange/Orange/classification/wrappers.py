import Orange.core
import Orange.evaluation.scoring as scoring
import Orange.data
import Orange.evaluation.testing
import Orange.evaluation.scoring

class StepwiseLearner(Orange.core.Learner):
  def __new__(cls, data=None, weightId=None, **kwargs):
      self = Orange.core.Learner.__new__(cls, **kwargs)
      if data is not None:
          self.__init__(**kwargs)
          return self(data, weightId)
      else:
          return self
      
  def __init__(self, **kwds):
    self.removeThreshold = 0.3
    self.addThreshold = 0.2
    self.stat, self.statsign = scoring.CA, 1
    self.__dict__.update(kwds)

  def __call__(self, examples, weightID = 0, **kwds):
    import Orange.evaluation.testing, Orange.evaluation.scoring, statc
    
    self.__dict__.update(kwds)

    if self.removeThreshold < self.addThreshold:
        raise ValueError("'removeThreshold' should be larger or equal to 'addThreshold'")

    classVar = examples.domain.classVar
    
    indices = Orange.core.MakeRandomIndicesCV(examples, folds = getattr(self, "folds", 10))
    domain = Orange.data.Domain([], classVar)

    res = Orange.evaluation.testing.testWithIndices([self.learner], Orange.data.Table(domain, examples), indices)
    
    oldStat = self.stat(res)[0]
    oldStats = [self.stat(x)[0] for x in Orange.evaluation.scoring.splitByIterations(res)]
    print ".", oldStat, domain
    stop = False
    while not stop:
        stop = True
        if len(domain.attributes)>=2:
            bestStat = None
            for attr in domain.attributes:
                newdomain = Orange.data.Domain(filter(lambda x: x!=attr, domain.attributes), classVar)
                res = Orange.evaluation.testing.testWithIndices([self.learner], (Orange.data.Table(newdomain, examples), weightID), indices)
                
                newStat = self.stat(res)[0]
                newStats = [self.stat(x)[0] for x in Orange.evaluation.scoring.splitByIterations(res)] 
                print "-", newStat, newdomain
                ## If stat has increased (ie newStat is better than bestStat)
                if not bestStat or cmp(newStat, bestStat) == self.statsign:
                    if cmp(newStat, oldStat) == self.statsign:
                        bestStat, bestStats, bestAttr = newStat, newStats, attr
                    elif statc.wilcoxont(oldStats, newStats)[1] > self.removeThreshold:
                            bestStat, bestAttr, bestStats = newStat, newStats, attr
            if bestStat:
                domain = Orange.data.Domain(filter(lambda x: x!=bestAttr, domain.attributes), classVar)
                oldStat, oldStats = bestStat, bestStats
                stop = False
                print "removed", bestAttr.name

        bestStat, bestAttr = oldStat, None
        for attr in examples.domain.attributes:
            if not attr in domain.attributes:
                newdomain = Orange.data.Domain(domain.attributes + [attr], classVar)
                res = Orange.evaluation.testing.testWithIndices([self.learner], (Orange.data.Table(newdomain, examples), weightID), indices)
                
                newStat = self.stat(res)[0]
                newStats = [self.stat(x)[0] for x in Orange.evaluation.scoring.splitByIterations(res)] 
                print "+", newStat, newdomain

                ## If stat has increased (ie newStat is better than bestStat)
                if cmp(newStat, bestStat) == self.statsign and statc.wilcoxont(oldStats, newStats)[1] < self.addThreshold:
                    bestStat, bestStats, bestAttr = newStat, newStats, attr
        if bestAttr:
            domain = Orange.data.Domain(domain.attributes + [bestAttr], classVar)
            oldStat, oldStats = bestStat, bestStats
            stop = False
            print "added", bestAttr.name

    return self.learner(Orange.data.Table(domain, examples), weightID)

