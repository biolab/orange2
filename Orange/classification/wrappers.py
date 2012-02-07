import Orange.core
import Orange.evaluation.scoring as scoring
import Orange.data
import Orange.evaluation.testing
import Orange.evaluation.scoring

from Orange.misc import deprecated_members

class StepwiseLearner(Orange.core.Learner):
  def __new__(cls, data=None, weight_id=None, **kwargs):
      self = Orange.core.Learner.__new__(cls, **kwargs)
      if data is not None:
          self.__init__(**kwargs)
          return self(data, weight_id)
      else:
          return self
      
  def __init__(self, **kwds):
    self.remove_threshold = 0.3
    self.add_threshold = 0.2
    self.stat, self.statsign = scoring.CA, 1
    for name, val in kwds.items():
        setattr(self, name, val)

  def __call__(self, data, weight_id = 0, **kwds):
    import Orange.evaluation.testing, Orange.evaluation.scoring, statc
    
    self.__dict__.update(kwds)

    if self.remove_threshold < self.add_threshold:
        raise ValueError("'remove_threshold' should be larger or equal to 'add_threshold'")

    classVar = data.domain.classVar
    
    indices = Orange.core.MakeRandomIndicesCV(data, folds = getattr(self, "folds", 10))
    domain = Orange.data.Domain([], classVar)

    res = Orange.evaluation.testing.test_with_indices([self.learner], Orange.data.Table(domain, data), indices)
    
    oldStat = self.stat(res)[0]
    oldStats = [self.stat(x)[0] for x in Orange.evaluation.scoring.split_by_iterations(res)]
    print ".", oldStat, domain
    stop = False
    while not stop:
        stop = True
        if len(domain.attributes)>=2:
            bestStat = None
            for attr in domain.attributes:
                newdomain = Orange.data.Domain(filter(lambda x: x!=attr, domain.attributes), classVar)
                res = Orange.evaluation.testing.test_with_indices([self.learner], (Orange.data.Table(newdomain, data), weight_id), indices)
                
                newStat = self.stat(res)[0]
                newStats = [self.stat(x)[0] for x in Orange.evaluation.scoring.split_by_iterations(res)] 
                print "-", newStat, newdomain
                ## If stat has increased (ie newStat is better than bestStat)
                if not bestStat or cmp(newStat, bestStat) == self.statsign:
                    if cmp(newStat, oldStat) == self.statsign:
                        bestStat, bestStats, bestAttr = newStat, newStats, attr
                    elif statc.wilcoxont(oldStats, newStats)[1] > self.remove_threshold:
                            bestStat, bestAttr, bestStats = newStat, newStats, attr
            if bestStat:
                domain = Orange.data.Domain(filter(lambda x: x!=bestAttr, domain.attributes), classVar)
                oldStat, oldStats = bestStat, bestStats
                stop = False
                print "removed", bestAttr.name

        bestStat, bestAttr = oldStat, None
        for attr in data.domain.attributes:
            if not attr in domain.attributes:
                newdomain = Orange.data.Domain(domain.attributes + [attr], classVar)
                res = Orange.evaluation.testing.test_with_indices([self.learner], (Orange.data.Table(newdomain, data), weight_id), indices)
                
                newStat = self.stat(res)[0]
                newStats = [self.stat(x)[0] for x in Orange.evaluation.scoring.split_by_iterations(res)] 
                print "+", newStat, newdomain

                ## If stat has increased (ie newStat is better than bestStat)
                if cmp(newStat, bestStat) == self.statsign and statc.wilcoxont(oldStats, newStats)[1] < self.add_threshold:
                    bestStat, bestStats, bestAttr = newStat, newStats, attr
        if bestAttr:
            domain = Orange.data.Domain(domain.attributes + [bestAttr], classVar)
            oldStat, oldStats = bestStat, bestStats
            stop = False
            print "added", bestAttr.name

    return self.learner(Orange.data.Table(domain, data), weight_id)

StepwiseLearner = deprecated_members(
                    {"removeThreshold": "remove_threshold",
                     "addThreshold": "add_threshold"},
                    )(StepwiseLearner)
