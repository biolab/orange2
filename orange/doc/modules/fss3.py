# Author:      B Zupan
# Version:     1.0
# Description: Compares naive Bayes with and withouth feature subset selection
# Category:    preprocessing
# Uses:        voting.tab
# Referenced:  orngFSS.htm

import orange, orngFSS
data = orange.ExampleTable("voting")

# first, define a new classifier which will use FSS

def BayesFSS(examples=None, **kwds):
  learner = apply(BayesFSS_Class, (), kwds)
  if examples: return learner(examples)
  else: return learner

class BayesFSS_Class:
  def __init__(self, name='Naive Bayes with FSS', N=5):
    self.name = name
    self.N = N

  def __call__(self, data, weight=None):
    ma = orngFSS.attMeasure(data)
    # filtered = orngFSS.selectAttsAboveTresh(data, ma)
    filtered = orngFSS.selectBestNAtts(data, ma, self.N)
    model = orange.BayesLearner(filtered)
    return BayesFSS_Classifier(classifier = model, nAtts = len(filtered.domain.attributes))

class BayesFSS_Classifier:
  def __init__(self, **kwds):
    self.__dict__ = kwds

  def __call__(self, example, resultType = orange.GetValue):
    return self.classifier(example, resultType)

# test above code on an example
# do 10-fold cross-validation

import orngStat, orngTest
learners = (orange.BayesLearner(name='Naive Bayes'), BayesFSS(name="with FSS"))
results = orngTest.crossValidation(learners, data)

# output the results
print "Learner      CA"
for i in range(len(learners)):
  print "%-12s %5.3f" % (learners[i].name, orngStat.CA(results)[i])
