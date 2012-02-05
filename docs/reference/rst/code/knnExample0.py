import Orange
iris = Orange.data.Table("iris")

knnLearner = Orange.classification.knn.kNNLearner()
knnLearner.k = 10
knnClassifier = knnLearner(iris)
