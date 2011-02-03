knn = Orange.classification.kNNLearner()
knn.k = 10
knn.distanceConstructor = Orange.core.ExamplesDistanceConstructor_Hamming()
knn = knn(train)
for i in range(5):
    instance = test.randomexample()
    print instance.getclass(), knn(instance)