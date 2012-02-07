import Orange
iris = Orange.data.Table("iris")

knn = Orange.classification.knn.kNNLearner()
knn.k = 10
knn.distance_constructor = Orange.distance.Hamming()
knn = knn(iris)
for i in range(5):
    instance = iris.random_example()
    print instance.getclass(), knn(instance)
