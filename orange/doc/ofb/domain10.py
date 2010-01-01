# Description: A simple implementation of wrapper feature subset selection
# Category:    modelling
# Uses:        voting
# Classes:     Domain, orngTest.crossValidation
# Referenced:  domain.htm

import orange, orngTest, orngStat, orngTree

def WrapperFSS(data, learner, verbose=0, folds=10):
  classVar = data.domain.classVar
  currentAtt = []
  freeAttributes = list(data.domain.attributes)

  newDomain = orange.Domain(currentAtt + [classVar])
  d = data.select(newDomain)
  results = orngTest.crossValidation([learner], d, folds=folds)
  maxStat = orngStat.CA(results)[0]
  if verbose>=2:
    print "start (%5.3f)" % maxStat

  while 1:
    stat = []
    for a in freeAttributes:
      newDomain = orange.Domain([a] + currentAtt + [classVar])
      d = data.select(newDomain)
      results = orngTest.crossValidation([learner], d, folds=folds)
      stat.append(orngStat.CA(results)[0])
      if verbose>=2:
        print "  %s gained %5.3f" % (a.name, orngStat.CA(results)[0])

    if (max(stat) > maxStat):
      oldMaxStat = maxStat
      maxStat = max(stat)
      bestVarIndx = stat.index(max(stat))
      if verbose:
        print "gain: %5.3f, attribute: %s" % (maxStat-oldMaxStat, freeAttributes[bestVarIndx].name)
      currentAtt = currentAtt + [freeAttributes[bestVarIndx]]
      del freeAttributes[bestVarIndx]
    else:
      if verbose:
        print "stopped (%5.3f)" % (max(stat) - maxStat)
      return orange.Domain(currentAtt + [classVar])
      break

def WrappedFSSLearner(learner, examples=None, verbose=0, folds=10, **kwds):
  kwds['verbose'] = verbose
  kwds['folds'] = folds
  learner = apply(WrappedFSSLearner_Class, (learner,), kwds)
  if examples:
    return learner(examples)
  else:
    return learner

class WrappedFSSLearner_Class:
  def __init__(self, learner, verbose=0, folds=10, name='learner w wrapper fss'):
    self.name = name
    self.learner = learner
    self.verbose = verbose
    self.folds = folds

  def __call__(self, data, weight=None):
    domain = WrapperFSS(data, self.learner, self.verbose, self.folds)
    selectedData = data.select(domain)
    if self.verbose:
      print 'features:', selectedData.domain
    model = self.learner(selectedData, weight)
    return Classifier(classifier = model)

class Classifier:
  def __init__(self, **kwds):
    self.__dict__.update(kwds)

  def __call__(self, example, resultType = orange.GetValue):
    return self.classifier(example, resultType)


#base = orngTree.TreeLearner(mForPruning=0.5)
#base.name = 'tree'
base = orange.BayesLearner()
base.name = 'bayes'
import warnings
warnings.filterwarnings("ignore", ".*'BayesLearner': .*", orange.KernelWarning)

fssed = WrappedFSSLearner(base, verbose=1, folds=5)
fssed.name = 'w fss'

# evaluation

learners = [base, fssed]
data = orange.ExampleTable("voting")
results = orngTest.crossValidation(learners, data, folds=10)

print "Learner      CA     IS     Brier    AUC"
for i in range(len(learners)):
  print "%-12s %5.3f  %5.3f  %5.3f  %5.3f" % (learners[i].name, \
    orngStat.CA(results)[i], orngStat.IS(results)[i],
    orngStat.BrierScore(results)[i], orngStat.AUC(results)[i])
