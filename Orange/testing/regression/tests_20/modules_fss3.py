# Description: Compares naive Bayes with and withouth feature subset selection
# Category:    preprocessing
# Uses:        voting.tab
# Referenced:  orngFSS.htm
# Classes:     orngFSS.attMeasure, orngFSS.selectBestNAtts

import orange, orngFSS

class BayesFSS(object):
  def __new__(cls, examples=None, **kwds):
    learner = object.__new__(cls)
    if examples:
      return learner(examples)
    else:
      return learner
    
  def __init__(self, name='Naive Bayes with FSS', N=5):
    self.name = name
    self.N = 5
      
  def __call__(self, data, weight=None):
    ma = orngFSS.attMeasure(data)
    filtered = orngFSS.selectBestNAtts(data, ma, self.N)
    model = orange.BayesLearner(filtered)
    return BayesFSS_Classifier(classifier=model, N=self.N, name=self.name)

class BayesFSS_Classifier:
  def __init__(self, **kwds):
    self.__dict__.update(kwds)
    
  def __call__(self, example, resultType = orange.GetValue):
    return self.classifier(example, resultType)

# test above wraper on a data set
import orngStat, orngTest
data = orange.ExampleTable("voting")
learners = (orange.BayesLearner(name='Naive Bayes'), BayesFSS(name="with FSS"))
results = orngTest.crossValidation(learners, data)

# output the results
print "Learner      CA"
for i in range(len(learners)):
  print "%-12s %5.3f" % (learners[i].name, orngStat.CA(results)[i])
