import Orange
table = Orange.data.Table("iris")

knn = Orange.classification.knn.kNNLearner()
knn.k = 10
knn.distance_constructor = Orange.core.ExamplesDistanceConstructor_Hamming()
knn = knn(table)
for i in range(5):
    instance = table.randomexample()
    print instance.getclass(), knn(instance)
