import Orange
iris = Orange.data.Table("iris")

knn = Orange.classification.knn.kNNLearner()
knn.k = 10
knn.distance_constructor = Orange.core.ExamplesDistanceConstructor_Hamming()
knn = knn(iris)
for i in range(5):
    instance = iris.randomexample()
    print instance.getclass(), knn(instance)
