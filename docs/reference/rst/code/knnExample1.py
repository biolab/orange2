import Orange
iris = Orange.data.Table("iris")

rndind = Orange.core.MakeRandomIndices2(iris, p0=0.8)
train = iris.select(rndind, 0)
test = iris.select(rndind, 1)

knn = Orange.classification.knn.kNNLearner(train, k=10)
for i in range(5):
    instance = test.randomexample()
    print instance.getclass(), knn(instance)