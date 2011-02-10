# Description: Builds regression models from data and outputs predictions for first five instances
# Category:    modelling
# Uses:        housing
# Classes:     MakeRandomIndices2, MajorityLearner, orngTree.TreeLearner, orange.kNNLearner
# Referenced:  regression.htm

import orange, orngTree, orngTest, orngStat

data = orange.ExampleTable("housing.tab")
selection = orange.MakeRandomIndices2(data, 0.5)
train_data = data.select(selection, 0)
test_data = data.select(selection, 1)

maj = orange.MajorityLearner(train_data)
maj.name = "default"

rt = orngTree.TreeLearner(train_data, measure="retis", mForPruning=2, minExamples=20)
rt.name = "reg. tree"

k = 5
knn = orange.kNNLearner(train_data, k=k)
knn.name = "k-NN (k=%i)" % k

regressors = [maj, rt, knn]

print "\n%10s " % "original",
for r in regressors:
  print "%10s " % r.name,
print

for i in range(10):
  print "%10.1f " % test_data[i].getclass(),
  for r in regressors:
    print "%10.1f " % r(test_data[i]),
  print
