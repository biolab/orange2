# Description: Uses cross-validation to compare regression tree and k-nearest neighbors
# Category:    modelling, evaluation
# Uses:        housing
# Classes:     orngStat.MSE, orngTest.crossValidation, MajorityLearner, orngTree.TreeLearner, orange.kNNLearner
# Referenced:  regression.htm

import orange, orngTree, orngTest, orngStat

data = orange.ExampleTable("housing.tab")

maj = orange.MajorityLearner()
maj.name = "default"
rt = orngTree.TreeLearner(measure="retis", mForPruning=2, minExamples=20)
rt.name = "regression tree"
k = 5
knn = orange.kNNLearner(k=k)
knn.name = "k-NN (k=%i)" % k
learners = [maj, rt, knn]

data = orange.ExampleTable("housing.tab")
results = orngTest.crossValidation(learners, data, folds=10)
mse = orngStat.MSE(results)

print "Learner        MSE"
for i in range(len(learners)):
  print "%-15s %5.3f" % (learners[i].name, mse[i])
