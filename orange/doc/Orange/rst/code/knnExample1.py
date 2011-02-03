import Orange
table = Orange.data.Table("iris")

rndind = Orange.core.MakeRandomIndices2(table, p0=0.8)
train = table.select(rndind, 0)
test = table.select(rndind, 1)

knn = Orange.classification.knn.kNNLearner(train, k=10)
for i in range(5):
    instance = test.randomexample()
    print instance.getclass(), knn(instance)