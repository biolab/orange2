# Description: Shows how to use the nearest-neighbour learning
# Category:    learning
# Classes:     kNNLearner, kNNClassifier, ExamplesDistance, ExamplesDistanceConstructor
# Uses:        iris
# Referenced:  kNNLearner.htm

import orange, orngTest, orngStat
data = orange.ExampleTable("iris")

knn = orange.kNNLearner(data, k=10)
for i in range(5):
    example = data.randomexample()
    print example.getclass(), knn(example)

print "\n\n"
data = orange.ExampleTable("iris")
knn = orange.kNNLearner()
knn.k = 10
knn.distanceConstructor = orange.ExamplesDistanceConstructor_Hamiltonian()
knn = knn(data)
for i in range(5):
    example = data.randomexample()
    print example.getclass(), knn(example)