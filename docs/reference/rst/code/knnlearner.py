# Description: Shows how to use the nearest-neighbour learning
# Category:    learning
# Classes:     kNNLearner, kNNClassifier, ExamplesDistance, ExamplesDistanceConstructor
# Uses:        iris
# Referenced:  kNNLearner.htm

import Orange
iris = Orange.data.Table("iris")

print "Testing using euclidean distance"
rndind = Orange.core.MakeRandomIndices2(iris, p0=0.8)
train = iris.select(rndind, 0)
test = iris.select(rndind, 1)

knn = Orange.classification.knn.kNNLearner(train, k=10)
for i in range(5):
    instance = test.random_instance()
    print instance.getclass(), knn(instance)

print "\n"
print "Testing using hamming distance"
iris = Orange.data.Table("iris")
knn = Orange.classification.knn.kNNLearner()
knn.k = 10
knn.distance_constructor = Orange.distance.Hamming()
knn = knn(train)
for i in range(5):
    instance = test.random_instance()
    print instance.getclass(), knn(instance)
