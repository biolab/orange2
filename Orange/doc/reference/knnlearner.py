# Description: Shows how to use the nearest-neighbour learning
# Category:    learning
# Classes:     kNNLearner, kNNClassifier, ExamplesDistance, ExamplesDistanceConstructor
# Uses:        iris
# Referenced:  kNNLearner.htm

import orange, orngTest, orngStat
data = orange.ExampleTable("iris")

rndind = orange.MakeRandomIndices2(data, p0=0.8)
train = data.select(rndind, 0)
test = data.select(rndind, 1)

knn = orange.kNNLearner(train, k=10)
for i in range(5):
    example = test.randomexample()
    print example.getclass(), knn(example)

print "\n\n"
data = orange.ExampleTable("iris")
knn = orange.kNNLearner()
knn.k = 10
knn.distanceConstructor = orange.ExamplesDistanceConstructor_Hamming()
knn = knn(train)
for i in range(5):
    example = test.randomexample()
    print example.getclass(), knn(example)