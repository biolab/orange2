# Description: Shows how to use the nearest-neighbour learning
# Category:    learning
# Classes:     kNNLearner, kNNClassifier, ExamplesDistance, ExamplesDistanceConstructor
# Uses:        iris
# Referenced:  kNNLearner.htm

import Orange
table = Orange.data.Table("iris")

print "Testing using euclidean distance"
rndind = Orange.core.MakeRandomIndices2(table, p0=0.8)
train = table.select(rndind, 0)
test = table.select(rndind, 1)

knn = Orange.classification.knn.kNNLearner(train, k=10)
for i in range(5):
    instance = test.randomexample()
    print instance.getclass(), knn(instance)

print "\n"
print "Testing using hamming distance"
table = Orange.data.Table("iris")
knn = Orange.classification.knn.kNNLearner()
knn.k = 10
knn.distanceConstructor = Orange.core.ExamplesDistanceConstructor_Hamming()
knn = knn(train)
for i in range(5):
    instance = test.randomexample()
    print instance.getclass(), knn(instance)