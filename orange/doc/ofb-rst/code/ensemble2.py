# Description: Example of how to build ensamble learners in Orange. Takes a list of learners, and for prediction uses the highest predicted class probability.
# Category:    modelling
# Uses:        promoters.tab
# Classes:     orngTest.crossValidation
# Referenced:  o_ensemble.htm

import orange, orngTree
import orngTest, orngStat

# define the learner and classifier such that
# it can be used as similar standard orange classes

def WinnerLearner(examples=None, **kwds):
  learner = apply(WinnerLearner_Class, (), kwds)
  if examples:
    return learner(examples)
  else:
    return learner

class WinnerLearner_Class:
  def __init__(self, name='winner classifier', learners=None):
    self.name = name
    self.learners = learners

  def __call__(self, data, learners=None, weight=None):
    if learners:
      self.learners = learners
    classifiers = []
    for l in self.learners:
      classifiers.append(l(data))
    return WinnerClassifier(classifiers = classifiers)

class WinnerClassifier:
  def __init__(self, **kwds):
    self.__dict__.update(kwds)

  def __call__(self, example, resultType = orange.GetValue):
    pmatrix = []
    for c in self.classifiers:
      pmatrix.append(c(example, orange.GetProbabilities))

    maxp = []  # stores max class probabilities for each classifiers
    for pv in pmatrix:
      maxp.append(max(pv))

    p = max(maxp)  # max class probability
    classifier_index = maxp.index(p)
    c = pmatrix[classifier_index].modus()
    
    if resultType == orange.GetValue:
      return c
    elif resultType == orange.getClassDistribution:
      return pmatrix[classifier_index]
    else:
      return (c, pmatrix[classifier_index])


tree = orngTree.TreeLearner(mForPruning=5.0)
tree.name = 'class. tree'
bayes = orange.BayesLearner()
bayes.name = 'naive bayes'
winner = WinnerLearner(learners=[tree, bayes])
winner.name = 'winner'

majority = orange.MajorityLearner()
majority.name = 'default'
learners = [majority, tree, bayes, winner]

data = orange.ExampleTable("promoters")

results = orngTest.crossValidation(learners, data)
print "Classification Accuracy:"
for i in range(len(learners)):
    print ("%15s: %5.3f") % (learners[i].name, orngStat.CA(results)[i])
